#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from functools import lru_cache
import gzip
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from ocrap.algorithms.lcv import normalize_weights, weighted_lcvar
from ocrap.config.defaults import deep_update
from ocrap.config.yaml_io import load_config
from ocrap.data.build.history import construct_history
from ocrap.data.schema import CandidatePrefix, RecoveryOption
from ocrap.data.serialization import load_npz_selected
from ocrap.data.waymax_loader import iter_waymax_womd_scenarios_selected
from ocrap.simulation.futures import generate_counterfactual_futures
from ocrap.simulation.teacher import compute_future_option_margins
from ocrap.v48_89_root_correspondence import nested_tail_influence
from ocrap.v48_90_partition_transport import future_class_keys
from ocrap.v48_91_common_exogenous_physical_margin import future_physical_matrix


ENGINEERING_VERSION='v48.91.3-OC-CEPMI-REPLAYFIX'

KEYS=frozenset({
    'scene_id','original_scenario_id','official_scenario_id','legacy_scenario_id','source_scenario_index',
    'womd_source_role','womd_source_pattern','waymax_max_num_objects','time_index','candidate_index','is_nominal',
    'agent_history','agent_valid','map_polylines','map_valid','dynamic_map','route','bev_occ','ego_state',
    'prefix_states','prefix_controls','prefix_macro_id','prefix_macro_type_id','prefix_macro_name','prefix_param',
    'utility','hard_violation','harm_proxy','feasible','prefix_diagnostics',
    'future_probs','future_sources','future_metadata','future_valid','root_assignments','root_probs','root_valid','c_star',
    'm_star','option_valid','recovery_modes','recovery_params','r_dep_star'
})


def _sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()


def _scalar(d:dict[str,Any],key:str,default:Any)->Any:
    try: return np.asarray(d.get(key,default)).reshape(-1)[0].item()
    except Exception: return default


_LEGACY_SOURCE_INDEX_RE = re.compile(r"__wx(?P<index>[0-9]+)(?:$|[^0-9])")


def _legacy_source_index_from_path(path: str | Path) -> int:
    m = _LEGACY_SOURCE_INDEX_RE.search(Path(str(path)).name)
    return int(m.group("index")) if m else -1


def _legacy_source_index(sample: dict[str, Any]) -> int:
    """Recover the historical Waymax global source index without mutating data.

    Pre-provenance-schema OC-RAP samples encoded the deterministic Waymax source
    index in the legacy scene id / sample filename as ``__wxNNNNNNNN``.  Newer
    datasets also serialize ``source_scenario_index`` explicitly.  The legacy
    suffix is already used elsewhere in OC-RAP as a migration key; it is not a
    semantic label and is never exposed to the planner.
    """
    explicit = int(_scalar(sample, 'source_scenario_index', -1))
    if explicit >= 0:
        return explicit
    values = (
        _scalar(sample, 'legacy_scenario_id', ''),
        _scalar(sample, 'scene_id', ''),
        _scalar(sample, 'original_scenario_id', ''),
        Path(str(sample.get('__path__', ''))).name,
    )
    for value in values:
        m = _LEGACY_SOURCE_INDEX_RE.search(str(value or ''))
        if m:
            return int(m.group('index'))
    return -1


def _resolve_replay_provenance(
    sample: dict[str, Any],
    *,
    source_pattern_override: str | None,
    replay_config_pattern: str | None,
) -> dict[str, Any]:
    explicit_pattern = str(_scalar(sample, 'womd_source_pattern', '') or '').strip()
    override = str(source_pattern_override or '').strip()
    cfg_pattern = str(replay_config_pattern or '').strip()
    if explicit_pattern:
        pattern = explicit_pattern
        pattern_source = 'npz'
    elif override:
        pattern = override
        pattern_source = 'cli_or_env_override'
    elif cfg_pattern:
        pattern = cfg_pattern
        pattern_source = 'replay_config'
    else:
        pattern = ''
        pattern_source = 'missing'

    explicit_idx = int(_scalar(sample, 'source_scenario_index', -1))
    idx = _legacy_source_index(sample)
    index_source = 'npz' if explicit_idx >= 0 else ('legacy_wx_migration_key' if idx >= 0 else 'missing')
    return {
        'pattern': pattern,
        'index': int(idx),
        'pattern_source': pattern_source,
        'index_source': index_source,
    }


@lru_cache(maxsize=2048)
def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    try:
        if path.is_file():
            value=json.loads(path.read_text(encoding='utf-8'))
            return value if isinstance(value,dict) else None
    except Exception:
        return None
    return None


def _index_belongs_to_shard(summary: dict[str, Any], index: int) -> bool:
    try:
        start=max(0,int(summary.get('scenario_start_index',0)))
        stride=max(1,int(summary.get('scenario_stride',1)))
        worker=int(summary.get('scenario_worker_index',0))%stride
        if index < start:
            return False
        return ((index-start)%stride)==worker
    except Exception:
        return False


def _origin_replay_metadata(sample_path: Path, source_index: int) -> dict[str, Any]:
    """Best-effort read-only traversal back to the original calibration shard.

    Protocol roots are scene-disjoint hardlink partitions.  Their
    ``split_provenance.json`` points to ``calibration_{near,contact}``; that root
    may point through ``scene_filter_provenance.json`` to a merged raw root, whose
    ``merged_dataset_summary.json`` lists the original worker shards.  When those
    artifacts still exist, recover the exact shard summary / resume semantic
    config without changing the canonical protocol dataset.
    """
    result:dict[str,Any]={}
    role_root=None
    q=sample_path.resolve().parent
    for _ in range(6):
        if (q/'split_provenance.json').is_file():
            role_root=q;break
        if q.parent==q:break
        q=q.parent
    if role_root is None:
        return result
    split=_read_json_if_exists(role_root/'split_provenance.json') or {}
    result['split_provenance_path']=str((role_root/'split_provenance.json').resolve())
    source_text=str(split.get('source') or '').strip()
    if not source_text:
        return result
    source_root=Path(source_text).expanduser()
    result['protocol_source_root']=str(source_root)
    roots=[source_root]
    sf=_read_json_if_exists(source_root/'scene_filter_provenance.json')
    if sf and sf.get('source'):
        raw=Path(str(sf['source'])).expanduser(); roots.append(raw)
        result['scene_filter_provenance_path']=str((source_root/'scene_filter_provenance.json').resolve())
        result['merged_raw_root']=str(raw)
    input_roots:list[Path]=[]
    for root in roots:
        merged=_read_json_if_exists(root/'merged_dataset_summary.json')
        if merged:
            for x in merged.get('input_roots',[]) or []:
                px=Path(str(x)).expanduser()
                if px not in input_roots: input_roots.append(px)
    # A source root can itself be a materialized builder output.
    candidates=input_roots or roots
    matches=[]
    for root in candidates:
        ds=_read_json_if_exists(root/'dataset_summary.json')
        if ds and _index_belongs_to_shard(ds,source_index):
            matches.append((root,ds))
    if len(matches)!=1:
        return result
    root,ds=matches[0]
    result['origin_shard_root']=str(root)
    result['dataset_summary_path']=str((root/'dataset_summary.json').resolve())
    result['dataset_summary']=ds
    rc=_read_json_if_exists(root/'resume_contract.json')
    if rc and isinstance(rc.get('semantic_config'),dict):
        result['resume_contract_path']=str((root/'resume_contract.json').resolve())
        result['semantic_config']=rc['semantic_config']
    return result


def _json_scalar(v:Any,default:Any)->Any:
    try:
        a=np.asarray(v)
        if a.ndim==0: v=a.item()
    except Exception: pass
    if isinstance(v,bytes): v=v.decode('utf-8','ignore')
    if isinstance(v,str):
        try:return json.loads(v)
        except Exception:return default
    return v


def _strvec(v:Any)->list[str]:
    a=np.asarray(v).reshape(-1); out=[]
    for x in a:
        if isinstance(x,bytes): x=x.decode('utf-8','ignore')
        out.append(str(x))
    return out


def _load(path:Path)->dict[str,Any]:
    d=load_npz_selected(path,KEYS); d['__path__']=str(path.resolve())
    return d


@lru_cache(maxsize=2048)
def _find_summary(path:Path)->Path|None:
    p=path.resolve().parent
    for _ in range(6):
        cand=p/'dataset_summary.json'
        if cand.is_file(): return cand
        if p.parent==p: break
        p=p.parent
    return None




def _stored_future_metadata(sample: dict[str, Any]) -> list[dict[str, Any]]:
    metas = _json_scalar(sample.get('future_metadata'), [])
    if not isinstance(metas, list):
        return []
    return [m for m in metas if isinstance(m, dict)]


def _canonical_v4814_protocol_role(sample_path: str | Path) -> str | None:
    parts = set(Path(str(sample_path)).parts)
    if {'evidence_adapt_dev_near_contact','certificate_pool_near_contact'} & parts:
        return 'near'
    if {'evidence_adapt_dev_contact','certificate_pool_contact'} & parts:
        return 'contact'
    return None


def _canonical_v4814_sample_profile(sample: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recover the exact *sample-local* V48.14 calibration replay profile.

    The canonical Near/Contact calibration was built with a balanced two-pass
    materializer.  Non-artifact samples were generated with hidden-pair mining
    forced OFF, while artifact samples were generated with mining forced ON and
    a local structural override path.  Replaying every sample with the global
    defaults (or even one role-wide config) therefore changes exogenous future
    classes.  This migration uses only immutable protocol role + stored offline
    future metadata to choose the historical pass; it never changes planner
    inputs or canonical NPZ content.
    """
    role = _canonical_v4814_protocol_role(sample.get('__path__',''))
    if role is None:
        return {}, {'profile_id': None, 'role': None, 'artifact_pass': None}
    metas = _stored_future_metadata(sample)
    artifact_pass = any(bool(m.get('artifact_mined', False)) or bool(m.get('artifact_branch')) for m in metas)
    profile: dict[str, Any] = {
        'data_source': 'womd',
        'simulation_backend': 'waymax_closed_loop',
        'num_reactive_futures': 2,
        'num_roots': 8,
        'num_recovery_options': 12,
        'waymax': {
            'dataloader_include_sdc_paths': True,
            'metrics_to_run': ['log_divergence','overlap','offroad','sdc_wrongway','sdc_off_route','sdc_progression','kinematic_infeasibility'],
            'teacher_backend': 'hybrid',
            'teacher_metrics_stride': 0,
            'use_jit_scan_rollouts': True,
            'cache_env_objects': True,
            'cache_postprefix_rollouts': True,
            'cache_teacher_metric_rollouts': True,
            'cache_identical_teacher_rollouts': True,
            'augmented_hidden_from_unknown_only': True,
            'enable_augmented_hidden_roots': True,
            'enable_visible_perturbation_roots': True,
        },
        'artifact': {
            'enable_branch_intent_margin': True,
            'branch_intent_compatible_margin': 1.0,
            'branch_intent_incompatible_margin': -2.5,
            'use_margin_override': False,
        },
        'dataset_quality': {
            'require_nominal_per_scene_time': True,
            'keep_nominal_even_if_quality_fails': True,
            'min_accepted_prefixes_per_scene_time': 2,
            'balanced_two_pass': True,
            'balanced_rotate_prefix_order': True,
            'artifact_pair_mode': 'balanced',
            'artifact_quota_uses_label': True,
            'require_artifact_pairs': True,
            'artifact_pass_use_margin_override': True,
            'artifact_pass_skip_augmented_waymax': True,
            'artifact_pass_apply_override_to_screened': True,
            'artifact_pass_compute_future_metrics': False,
        },
        'regime_thresholds': {
            'include_prefix_collision_in_near': False,
            'include_prefix_contact_in_post': False,
            'use_paper_regime_definitions': True,
        },
    }
    if role == 'near':
        profile.update({
            'num_targeted_futures': 8,
            'targeted_future_kinds': ['hidden_vehicle_yields','hidden_vehicle_accelerates','low_friction_braking','control_delay_noise'],
        })
        profile['artifact'].update({'force_mine': True, 'mine_probability': 0.30})
        profile['dataset_quality'].update({
            'max_accepted_prefixes_per_scene_time': 8,
            'min_artifact_prefixes_per_scene_time': 1,
            'max_artifact_prefixes_per_scene_time': 2,
            'min_nonartifact_prefixes_per_scene_time': 4,
            'max_nonartifact_prefixes_per_scene_time': 6,
            'max_artifact_attempts_per_scene_time': 24,
            'max_nonartifact_attempts_per_scene_time': 12,
        })
    else:
        profile.update({
            'num_targeted_futures': 10,
            'targeted_future_kinds': ['hidden_vehicle_yields','hidden_vehicle_accelerates','contact_impulse_surrogate','secondary_collision_approach','low_friction_braking','control_delay_noise'],
        })
        profile['artifact'].update({'force_mine': True, 'mine_probability': 0.25})
        profile['dataset_quality'].update({
            'max_accepted_prefixes_per_scene_time': 9,
            'min_artifact_prefixes_per_scene_time': 1,
            'max_artifact_prefixes_per_scene_time': 2,
            'min_nonartifact_prefixes_per_scene_time': 5,
            'max_nonartifact_prefixes_per_scene_time': 7,
            'max_artifact_attempts_per_scene_time': 24,
            'max_nonartifact_attempts_per_scene_time': 12,
        })
    # Reconstruct the builder's sample-local balanced pass exactly.
    if artifact_pass:
        profile['artifact'].update({'force_mine': True, 'mine_probability': 1.0, 'use_margin_override': True})
        profile['dataset_quality']['require_artifact_pairs'] = True
        profile['waymax'].update({
            'skip_waymax_rollout_for_augmented_override': True,
            'apply_artifact_override_to_screened_options': True,
        })
    else:
        profile['artifact'].update({'force_mine': False, 'mine_probability': 0.0, 'use_margin_override': False})
        profile['dataset_quality']['require_artifact_pairs'] = False
    return profile, {
        'profile_id': 'calibration_v48_14_prism_4814',
        'role': role,
        'artifact_pass': bool(artifact_pass),
    }



def _apply_v4814_sample_local_balanced_pass(cfg: dict[str, Any], sample: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replay the builder-local no-mine/mined pass for one frozen sample.

    Even an exact shard ``resume_contract.semantic_config`` records the *base*
    Near/Contact configuration, while ``builder._cfg_with_artifact_mining``
    changes several knobs per candidate materialization.  Recover that local
    pass from immutable stored future metadata.
    """
    role = _canonical_v4814_protocol_role(sample.get('__path__',''))
    if role is None:
        return cfg, {'profile_id':None,'role':None,'artifact_pass':None}
    metas = _stored_future_metadata(sample)
    artifact_pass = any(bool(m.get('artifact_mined', False)) or bool(m.get('artifact_branch')) for m in metas)
    out=json.loads(json.dumps(cfg))
    art=dict(out.get('artifact',{}) or {})
    quality=dict(out.get('dataset_quality',{}) or {})
    wx=dict(out.get('waymax',{}) or {})
    if artifact_pass:
        art.update({'force_mine':True,'mine_probability':1.0,'use_margin_override':True})
        quality['require_artifact_pairs']=True
        if bool(quality.get('artifact_pass_skip_augmented_waymax', True)):
            wx['skip_waymax_rollout_for_augmented_override']=True
        if bool(quality.get('artifact_pass_apply_override_to_screened', True)):
            wx['apply_artifact_override_to_screened_options']=True
        if not bool(quality.get('artifact_pass_compute_future_metrics', False)):
            wx['compute_future_metrics']=False
    else:
        art.update({'force_mine':False,'mine_probability':0.0})
        quality['require_artifact_pairs']=False
    out['artifact']=art;out['dataset_quality']=quality;out['waymax']=wx
    return out, {'profile_id':'calibration_v48_14_prism_4814','role':role,'artifact_pass':bool(artifact_pass)}

def _config_for_sample(sample:dict[str,Any], base_cfg:dict[str,Any], *, resolved_pattern:str|None=None, resolved_index:int|None=None, explicit_replay_config:bool=False)->tuple[dict[str,Any],str|None,dict[str,Any]]:
    origin=_origin_replay_metadata(Path(sample['__path__']),int(resolved_index if resolved_index is not None else _legacy_source_index(sample)))
    profile_meta={'profile_id':None,'role':None,'artifact_pass':None}
    if (not explicit_replay_config) and isinstance(origin.get('semantic_config'),dict):
        cfg=deep_update(load_config(None),json.loads(json.dumps(origin['semantic_config'])))
        origin['replay_config_source']='origin_resume_contract'
    elif not explicit_replay_config:
        profile,profile_meta=_canonical_v4814_sample_profile(sample)
        if profile:
            cfg=deep_update(load_config(None),profile)
            origin['replay_config_source']='canonical_v48_14_sample_local_profile'
        else:
            cfg=json.loads(json.dumps(base_cfg))
            origin['replay_config_source']='default_fallback'
    else:
        cfg=json.loads(json.dumps(base_cfg))
        origin['replay_config_source']='explicit_replay_config'
    # The historical balanced builder applies a *sample-local* mining pass on
    # top of the base shard config; recover it even when an exact resume config
    # or explicit replay YAML is available.
    cfg, local_pass_meta = _apply_v4814_sample_local_balanced_pass(cfg, sample)
    if local_pass_meta.get('profile_id') is not None:
        profile_meta = local_pass_meta
    origin['replay_profile']=profile_meta
    origin_summary=origin.get('dataset_summary') if isinstance(origin.get('dataset_summary'),dict) else None
    sp=Path(origin['dataset_summary_path']) if origin.get('dataset_summary_path') else _find_summary(Path(sample['__path__']))
    if origin_summary is not None:
        s=origin_summary
    elif sp:
        s=json.loads(sp.read_text())
    else:
        s=None
    if isinstance(s,dict):
        for key in ('dataset_quality','artifact','waymax','regime_thresholds'):
            if isinstance(s.get(key),dict): cfg[key]=deep_update(dict(cfg.get(key,{}) or {}),s[key])
        g=s.get('generation',{}) if isinstance(s.get('generation',{}),dict) else {}
        for key in ('num_candidate_prefixes','num_reactive_futures','num_targeted_futures','num_roots','num_recovery_options','targeted_future_kinds'):
            if key in g and g[key] is not None: cfg[key]=g[key]
    # Infer settings that are exactly visible in the stored sample.  This does
    # not alter the canonical dataset; it only reconstructs the teacher audit.
    sources=_strvec(sample.get('future_sources',[]))
    cfg['num_reactive_futures']=sum(s=='reactive' for s in sources)
    cfg['num_targeted_futures']=sum(s=='targeted' for s in sources)
    cfg['num_recovery_options']=int(np.asarray(sample['m_star']).shape[1])
    cfg['num_roots']=int(np.asarray(sample['m_star']).shape[0])
    cfg['max_agents']=int(np.asarray(sample['agent_history']).shape[1])
    cfg['data_source']='womd'; cfg['simulation_backend']='waymax_closed_loop'
    cfg['womd_patterns']=str(_scalar(sample,'womd_source_pattern','') or resolved_pattern or cfg.get('womd_patterns') or '')
    # Source scanning is performed externally by exact stored source index.
    cfg['scenario_start_index']=0; cfg['scenario_stride']=1; cfg['scenario_worker_index']=0
    # V48.91 does not consume generic per-future Waymax metric metadata. The
    # exact teacher margin is recomputed separately below, and exogenous class
    # identity ignores ``waymax_metrics``. Skipping these metadata-only metric
    # summaries is guarded by future-class and root-margin replay identity.
    wx = dict(cfg.get('waymax', {}) or {})
    wx['compute_future_metrics'] = False
    wx['use_jit_scan_rollouts'] = True
    wx['cache_env_objects'] = True
    wx['cache_postprefix_rollouts'] = True
    wx['cache_teacher_metric_rollouts'] = True
    wx['cache_identical_teacher_rollouts'] = True
    cfg['waymax'] = wx
    return cfg,(str(sp) if sp else None),origin


def _prefix(sample:dict[str,Any])->CandidatePrefix:
    return CandidatePrefix(
        macro_id=int(_scalar(sample,'prefix_macro_id',_scalar(sample,'candidate_index',-1))),
        macro_name=str(_scalar(sample,'prefix_macro_name','unknown')),
        params=np.asarray(sample['prefix_param'],dtype=np.float32),
        prefix_states=np.asarray(sample['prefix_states'],dtype=np.float32),
        prefix_controls=np.asarray(sample['prefix_controls'],dtype=np.float32),
        utility=float(_scalar(sample,'utility',0.0)), feasible=bool(int(_scalar(sample,'feasible',1))),
        hard_violation=float(_scalar(sample,'hard_violation',0.0)), harm_proxy=float(_scalar(sample,'harm_proxy',0.0)),
        diagnostics=_json_scalar(sample.get('prefix_diagnostics'),{}),
    )


def _options(sample:dict[str,Any], ids:list[int])->list[RecoveryOption]:
    modes=_strvec(sample['recovery_modes']); params=np.asarray(sample['recovery_params'],dtype=np.float32)
    valid=np.asarray(sample['option_valid'],dtype=bool).reshape(-1)
    out=[]
    for l in ids:
        out.append(RecoveryOption(option_id=int(l),mode=modes[l],params=np.asarray(params[l],dtype=np.float32),valid=bool(valid[l])))
    return out


def _history_check(history,sample:dict[str,Any])->dict[str,float]:
    checks={}
    for key,actual in [('agent_history',history.agent_history),('ego_state',history.ego_state)]:
        stored=np.asarray(sample[key],dtype=np.float32)
        if stored.shape!=actual.shape: raise ValueError(f'{key} shape mismatch {actual.shape} != {stored.shape}')
        err=float(np.max(np.abs(stored-np.asarray(actual,dtype=np.float32)))) if stored.size else 0.0
        checks[key]=err
        if err>1e-5: raise ValueError(f'{key} replay mismatch max_abs={err}')
    return checks


def _replay_one(raw,sample:dict[str,Any],option_ids:list[int],base_cfg:dict[str,Any],alpha_intra:float, *, resolved_pattern:str|None=None, resolved_index:int|None=None, explicit_replay_config:bool=False, history_cache:dict[tuple[Any,...],Any]|None=None, timing_accum:dict[str,float]|None=None)->dict[str,Any]:
    timing_accum = timing_accum if timing_accum is not None else {}
    t_stage=time.perf_counter()
    cfg,summary_path,origin=_config_for_sample(sample,base_cfg,resolved_pattern=resolved_pattern,resolved_index=resolved_index,explicit_replay_config=explicit_replay_config)
    timing_accum['config_seconds']=timing_accum.get('config_seconds',0.0)+(time.perf_counter()-t_stage)
    t=int(_scalar(sample,'time_index',-1))
    hk=(
        t, summary_path, int(cfg.get('max_agents',-1)), float(cfg.get('sample_rate_hz',10.0)),
        float(cfg.get('history_horizon_s',1.0)), float(cfg.get('prefix_horizon_s',1.0)),
        float(cfg.get('recovery_horizon_s',4.0)), int(cfg.get('route_points',80)),
        float(cfg.get('local_radius_m',80.0)), float(cfg.get('bev_resolution_m',1.0)),
        int(cfg.get('bev_channels',7)), float(cfg.get('route_width',3.5)),
    )
    history_cache = history_cache if history_cache is not None else {}
    t_stage=time.perf_counter()
    history = history_cache.get(hk)
    history_cache_hit = history is not None
    if history is None:
        history=construct_history(raw,t,cfg)
        history_cache[hk]=history
    hcheck=_history_check(history,sample)
    timing_accum['history_seconds']=timing_accum.get('history_seconds',0.0)+(time.perf_counter()-t_stage)
    prefix=_prefix(sample)
    t_stage=time.perf_counter()
    futures=generate_counterfactual_futures(history,prefix,cfg)
    timing_accum['future_generation_seconds']=timing_accum.get('future_generation_seconds',0.0)+(time.perf_counter()-t_stage)
    t_stage=time.perf_counter()
    stored_probs=np.asarray(sample['future_probs'],dtype=np.float64).reshape(-1)
    probs=np.asarray([f.prior for f in futures],dtype=np.float64); probs=normalize_weights(probs)
    if probs.size!=stored_probs.size: raise ValueError(f'future count mismatch {probs.size}!={stored_probs.size}')
    perr=float(np.max(np.abs(probs-stored_probs))) if probs.size else 0.0
    if perr>1e-6: raise ValueError(f'future probability mismatch max_abs={perr}')
    replay_view={'future_sources':np.asarray([f.source for f in futures]),'future_metadata':json.dumps([f.metadata for f in futures],sort_keys=True),
                 'future_probs':probs,'future_valid':np.asarray([bool(f.agent_valid.any()) for f in futures],dtype=np.float32),
                 'root_assignments':np.asarray(sample['root_assignments'])}
    stored_keys,stored_unres,_=future_class_keys(sample,exogenous=True)
    replay_keys,replay_unres,_=future_class_keys(replay_view,exogenous=True)
    timing_accum['future_identity_seconds']=timing_accum.get('future_identity_seconds',0.0)+(time.perf_counter()-t_stage)
    if stored_keys!=replay_keys or not np.array_equal(stored_unres,replay_unres):
        first = next((i for i,(a,b) in enumerate(zip(stored_keys,replay_keys)) if a!=b), None)
        if first is None and len(stored_keys)!=len(replay_keys):
            first=min(len(stored_keys),len(replay_keys))
        prof=origin.get('replay_profile') or {}
        raise ValueError(
            'exogenous future-class replay mismatch '
            f'profile={prof} first_index={first} stored_n={len(stored_keys)} replay_n={len(replay_keys)} '
            f'stored_key={(stored_keys[first] if first is not None and first < len(stored_keys) else None)!r} '
            f'replay_key={(replay_keys[first] if first is not None and first < len(replay_keys) else None)!r}'
        )

    opts=_options(sample,option_ids)
    t_stage=time.perf_counter()
    m_sub,diags=compute_future_option_margins(history,prefix,futures,opts,cfg)
    timing_accum['teacher_margin_seconds']=timing_accum.get('teacher_margin_seconds',0.0)+(time.perf_counter()-t_stage)
    t_stage=time.perf_counter()
    phys_sub=future_physical_matrix(diags,np.asarray([o.valid for o in opts],dtype=bool))
    F=int(stored_probs.size); L=int(np.asarray(sample['m_star']).shape[1])
    struct=np.full((F,L),np.nan,dtype=np.float64); phys=np.full((F,L),np.nan,dtype=np.float64)
    assign=np.asarray(sample['root_assignments'],dtype=np.int64).reshape(-1)
    validf=np.asarray(sample.get('future_valid',np.ones(F)),dtype=bool).reshape(-1)
    storedM=np.asarray(sample['m_star'],dtype=np.float64)
    root_err=0.0
    for col,l in enumerate(option_ids):
        struct[:,l]=np.asarray(m_sub[:,col],dtype=np.float64); phys[:,l]=np.asarray(phys_sub[:,col],dtype=np.float64)
        for k in range(storedM.shape[0]):
            idx=np.where(validf & (assign==k))[0]
            if not len(idx): continue
            w=normalize_weights(stored_probs[idx])
            agg=float(weighted_lcvar(struct[idx,l],w,float(alpha_intra)))
            root_err=max(root_err,abs(agg-float(storedM[k,l])))
    timing_accum['physical_projection_validation_seconds']=timing_accum.get('physical_projection_validation_seconds',0.0)+(time.perf_counter()-t_stage)
    if root_err>2e-5: raise ValueError(f'active root-margin replay mismatch max_abs={root_err}')
    return {
        'valid':True,'sample_path':sample['__path__'],'sample_sha256':_sha(Path(sample['__path__'])),
        'scene_id':str(_scalar(sample,'scene_id','')),'time_index':t,'candidate_index':int(_scalar(sample,'candidate_index',-1)),
        'source_scenario_index':int(resolved_index if resolved_index is not None else _scalar(sample,'source_scenario_index',-1)),
        'womd_source_pattern':str(resolved_pattern or _scalar(sample,'womd_source_pattern','')),
        'option_ids':[int(x) for x in option_ids],'future_count':F,'option_count':L,
        'm_future_structural':{str(l):struct[:,l].tolist() for l in option_ids},
        'm_future_physical':{str(l):phys[:,l].tolist() for l in option_ids},
        'future_probability_max_abs_error':perr,'active_root_margin_max_abs_error':float(root_err),
        'history_max_abs_error':hcheck,'dataset_summary_path':summary_path,
        'origin_shard_root':origin.get('origin_shard_root'),'origin_resume_contract_path':origin.get('resume_contract_path'),
        'history_cache_hit':bool(history_cache_hit),
        'replay_config_source':str(origin.get('replay_config_source','')),
        'replay_profile':origin.get('replay_profile') or {},
        'audit_future_metadata_metrics_skipped':True,
    }




def _checkpoint_stat(path_text: str) -> tuple[int, int]:
    st=Path(path_text).stat()
    return int(st.st_size), int(st.st_mtime_ns)


def _load_replay_checkpoint(path: Path | None, requested: dict[str, set[int]]) -> dict[str, dict[str, Any]]:
    if path is None or not path.is_file():
        return {}
    out:dict[str,dict[str,Any]]={}
    try:
        with path.open('r',encoding='utf-8') as f:
            for line in f:
                try: rec=json.loads(line)
                except Exception: continue
                if rec.get('engineering_version') != ENGINEERING_VERSION:
                    continue
                p=str(rec.get('sample_path') or '')
                if p not in requested:
                    continue
                if list(map(int,rec.get('option_ids') or [])) != sorted(map(int,requested[p])):
                    continue
                try:
                    size,mtime=_checkpoint_stat(p)
                except Exception:
                    continue
                if int(rec.get('sample_size',-1)) != size or int(rec.get('sample_mtime_ns',-1)) != mtime:
                    continue
                row=rec.get('row')
                if isinstance(row,dict) and row.get('valid') and str(row.get('sample_path'))==p:
                    out[p]=row
    except Exception:
        return {}
    return out


def _append_replay_checkpoint(handle, row: dict[str, Any], option_ids: list[int]) -> None:
    if handle is None:
        return
    p=str(row['sample_path'])
    size,mtime=_checkpoint_stat(p)
    rec={
        'engineering_version':ENGINEERING_VERSION,
        'sample_path':p,
        'option_ids':[int(x) for x in option_ids],
        'sample_size':size,
        'sample_mtime_ns':mtime,
        'row':row,
    }
    handle.write(json.dumps(rec,sort_keys=True)+'\n')
    handle.flush()

def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--v48-90-audit',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True,help='gzip JSONL sidecar')
    ap.add_argument('--summary',type=Path,required=True)
    ap.add_argument('--replay-config',type=str,default=None,help='optional exact dataset build YAML; defaults + dataset-summary inference otherwise')
    ap.add_argument('--womd-source-pattern',type=str,default=None,help='exact raw WOMD pattern used by legacy samples when NPZ provenance fields are absent; provenance only, never a model input')
    ap.add_argument('--alpha',type=float,default=0.2); ap.add_argument('--beta',type=float,default=0.2); ap.add_argument('--top-m',type=int,default=8)
    ap.add_argument('--intra-root-alpha',type=float,default=0.2)
    ap.add_argument('--num-workers',type=int,default=1,help='independent replay shards; launcher can bind each shard to a separate GPU')
    ap.add_argument('--worker-index',type=int,default=0,help='0-based replay shard index')
    ap.add_argument('--progress-every',type=int,default=max(0,int(os.environ.get('V4891_PROGRESS_EVERY','25'))),help='emit replay progress every N samples (0 disables)')
    ap.add_argument('--fail-fast-replay-errors',type=int,default=max(1,int(os.environ.get('V4891_FAIL_FAST_REPLAY_ERRORS','1'))),help='stop the shard after N replay identity/config errors; default 1 prevents multi-hour invalid replays')
    ap.add_argument('--checkpoint',type=Path,default=None,help='append-only valid-row replay checkpoint for exact resume')
    ap.add_argument('--resume-checkpoint',action=argparse.BooleanOptionalAction,default=True,help='reuse valid rows from a same-version checkpoint when NPZ stat and requested options match')
    args=ap.parse_args()
    if args.num_workers < 1 or not (0 <= args.worker_index < args.num_workers):
        raise SystemExit(f'invalid worker partition index={args.worker_index} count={args.num_workers}')
    t_total=time.perf_counter()
    base=load_config(args.replay_config)

    requested:dict[str,set[int]]=defaultdict(set); path_to_role:dict[str,set[str]]=defaultdict(set); pair_count=0
    loaded_samples:dict[str,dict[str,Any]]={}
    def get_sample(path_text:str)->dict[str,Any]:
        s=loaded_samples.get(path_text)
        if s is None:
            s=_load(Path(path_text)); loaded_samples[path_text]=s
        return s
    t0=time.perf_counter(); audit_rows_seen=0; audit_rows_selected=0
    with args.v48_90_audit.open(encoding='utf-8') as f:
        for line in f:
            r=json.loads(line)
            if not(r.get('valid') and r.get('label_available')): continue
            audit_rows_seen += 1
            cp=str(Path(r['sample_path']).resolve()); npth=str(Path(r['nominal_sample_path']).resolve())
            source_hint=_legacy_source_index_from_path(cp)
            if source_hint < 0 and args.num_workers > 1:
                # Newer samples can omit the legacy suffix because they serialize
                # source_scenario_index explicitly.  Load only the candidate row
                # when sharding needs that fallback.
                source_hint=_legacy_source_index(get_sample(cp))
            if args.num_workers > 1:
                if source_hint < 0:
                    raise SystemExit(f'cannot shard replay without source index: {cp}')
                if (source_hint % args.num_workers) != args.worker_index:
                    continue
            audit_rows_selected += 1
            cs=get_sample(cp); mass,*_=nested_tail_influence(cs,alpha=args.alpha,beta=args.beta,top_m=args.top_m)
            ids=np.where(np.sum(mass,axis=0)>1e-12)[0].tolist()
            if not ids: continue
            requested[cp].update(map(int,ids)); requested[npth].update(map(int,ids)); pair_count+=1
            role=str(r['dataset_role']); path_to_role[cp].add(role); path_to_role[npth].add(role)
    audit_scan_seconds=time.perf_counter()-t0
    if not requested:
        raise SystemExit(f'no labeled V48.90 cohort samples/options to replay for shard {args.worker_index}/{args.num_workers}')

    t0=time.perf_counter(); samples={p:(loaded_samples[p] if p in loaded_samples else _load(Path(p))) for p in requested}; loaded_samples.update(samples); sample_load_seconds=time.perf_counter()-t0
    checkpoint_rows=_load_replay_checkpoint(args.checkpoint,requested) if args.resume_checkpoint else {}
    groups:dict[tuple[str,int],dict[int,list[str]]]=defaultdict(lambda:defaultdict(list))
    provenance_resolution_counts:dict[str,int]=defaultdict(int)
    resolved_indices:list[int]=[]
    replay_cfg_pattern=str(base.get('womd_patterns') or '')
    t0=time.perf_counter()
    for p,s in samples.items():
        prov=_resolve_replay_provenance(
            s, source_pattern_override=args.womd_source_pattern, replay_config_pattern=replay_cfg_pattern
        )
        idx=int(prov['index'])
        if args.num_workers > 1 and (idx % args.num_workers) != args.worker_index:
            raise SystemExit(f'worker partition mismatch for {p}: source_index={idx}')
        # Historical protocol roots may still retain a provenance chain back to
        # the original worker shard. Prefer its immutable resume contract before
        # requiring a manual raw-source override.
        if not prov['pattern'] and idx>=0:
            origin_probe=_origin_replay_metadata(Path(p),idx)
            semantic=origin_probe.get('semantic_config') if isinstance(origin_probe.get('semantic_config'),dict) else {}
            origin_pattern=str((semantic or {}).get('womd_patterns') or '').strip()
            if origin_pattern:
                prov['pattern']=origin_pattern; prov['pattern_source']='origin_resume_contract'
        pattern=str(prov['pattern']); maxobj=int(_scalar(s,'waymax_max_num_objects',-1))
        provenance_resolution_counts[f"pattern:{prov['pattern_source']}"] += 1
        provenance_resolution_counts[f"index:{prov['index_source']}"] += 1
        if not pattern or idx<0:
            raise SystemExit(
                f'missing raw replay provenance for {p}: pattern={pattern!r} index={idx}; '
                'legacy OC-RAP calibration NPZs may omit explicit provenance. '
                'Set V4891_WOMD_SOURCE / --womd-source-pattern to the exact raw WOMD pattern; '
                'the source index will be migrated from the __wxNNNN suffix when available. '
                'This does not require rebuilding or modifying the canonical dataset.'
            )
        resolved_indices.append(idx)
        if maxobj<=0: maxobj=int(np.asarray(s['agent_history']).shape[1])
        if p not in checkpoint_rows:
            groups[(pattern,maxobj)][idx].append(p)
    provenance_seconds=time.perf_counter()-t0

    target_source_indices=len(set(resolved_indices))
    active_option_counts=[len(v) for v in requested.values()]
    print(json.dumps({
        'event':'v48.91_replay_plan','engineering_version':ENGINEERING_VERSION,
        'worker_index':args.worker_index,'num_workers':args.num_workers,
        'audit_rows_seen':audit_rows_seen,'audit_rows_selected':audit_rows_selected,
        'labeled_candidate_pairs':pair_count,'requested_samples':len(requested),
        'target_source_indices':target_source_indices,
        'active_options_mean':float(np.mean(active_option_counts)) if active_option_counts else 0.0,
        'active_options_max':max(active_option_counts) if active_option_counts else 0,
        'sparse_source_iterator':True,'metadata_only_future_metrics_skipped':True,
        'canonical_v48_14_sample_local_replay_profile':True,
        'fail_fast_replay_errors':int(args.fail_fast_replay_errors),
        'checkpoint_rows_reused':len(checkpoint_rows),
        'checkpoint_path':str(args.checkpoint) if args.checkpoint else None,
    }),flush=True)

    rows=list(checkpoint_rows.values()); errors=[]; processed=len(checkpoint_rows); history_hits=sum(int(bool(r.get('history_cache_hit'))) for r in rows); replay_seconds=0.0; raw_scan_seconds=0.0
    stage_timing:dict[str,float]={}
    checkpoint_handle=None
    if args.checkpoint is not None:
        args.checkpoint.parent.mkdir(parents=True,exist_ok=True)
        checkpoint_handle=args.checkpoint.open('a',encoding='utf-8')
    try:
      for (pattern,maxobj),byidx in groups.items():
        parser_cfg=json.loads(json.dumps(base)); parser_cfg['data_source']='womd'; parser_cfg['simulation_backend']='waymax_closed_loop'; parser_cfg['womd_patterns']=pattern; parser_cfg['max_agents']=maxobj
        parser_cfg['scenario_start_index']=0; parser_cfg['scenario_stride']=1; parser_cfg['scenario_worker_index']=0
        parser_cfg['_selected_replay_progress_every']=max(0,int(os.environ.get('V4891_SOURCE_SCAN_PROGRESS_EVERY','1000')))
        target_indices=sorted(byidx)
        print(json.dumps({
            'event':'v48.91_raw_scan_start','worker_index':args.worker_index,
            'target_indices':len(target_indices),'min_index':target_indices[0],'max_index':target_indices[-1],
            'max_agents':maxobj,
        }),flush=True)
        seen=set(); scan_start=time.perf_counter()
        for raw in iter_waymax_womd_scenarios_selected(pattern,target_indices,parser_cfg=parser_cfg):
            idx=int(raw.metadata.get('_waymax_scenario_index',-1))
            if idx not in byidx: continue
            seen.add(idx)
            # Reuse the expensive map/route/BEV history construction across all
            # candidate prefixes from the same raw scene/time.
            history_cache:dict[tuple[Any,...],Any]={}
            for p in sorted(byidx[idx]):
                s=samples[p]
                one_start=time.perf_counter()
                try:
                    row=_replay_one(
                        raw,s,sorted(requested[p]),base,args.intra_root_alpha,
                        resolved_pattern=pattern,resolved_index=idx,
                        explicit_replay_config=bool(args.replay_config),history_cache=history_cache,timing_accum=stage_timing,
                    )
                    row['dataset_roles']=sorted(path_to_role[p]); rows.append(row)
                    _append_replay_checkpoint(checkpoint_handle,row,sorted(requested[p]))
                    history_hits += int(bool(row.get('history_cache_hit')))
                except Exception as exc:
                    errors.append(f'{p}: {exc}')
                    rows.append({'valid':False,'sample_path':p,'error':str(exc),'dataset_roles':sorted(path_to_role[p])})
                    if len(errors) >= int(args.fail_fast_replay_errors):
                        print(json.dumps({
                            'event':'v48.91_replay_fail_fast','worker_index':args.worker_index,
                            'errors':len(errors),'last_error':errors[-1],
                            'message':'stopping early before full-cohort replay because identity/config replay is invalid',
                        }),flush=True)
                        replay_seconds += time.perf_counter()-one_start
                        processed += 1
                        break
                replay_seconds += time.perf_counter()-one_start
                processed += 1
                if args.progress_every > 0 and (processed % args.progress_every)==0:
                    elapsed=time.perf_counter()-t_total
                    print(json.dumps({
                        'event':'v48.91_replay_progress','worker_index':args.worker_index,
                        'processed':processed,'requested_samples':len(requested),
                        'valid':sum(1 for r in rows if r.get('valid')),'errors':len(errors),
                        'history_cache_hits':history_hits,'elapsed_seconds':round(elapsed,3),
                        'samples_per_minute':round((processed/max(elapsed,1e-9))*60.0,3),
                    }),flush=True)
            if len(errors) >= int(args.fail_fast_replay_errors):
                break
        raw_scan_seconds += time.perf_counter()-scan_start
        missing=sorted(set(byidx)-seen)
        if missing and len(errors) < int(args.fail_fast_replay_errors):
            errors.append(f'raw source indices not encountered pattern={pattern}: {missing[:20]} count={len(missing)}')
        if len(errors) >= int(args.fail_fast_replay_errors):
            break

    finally:
        if checkpoint_handle is not None:
            checkpoint_handle.close()

    args.output.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(args.output,'wt',encoding='utf-8') as f:
        for r in sorted(rows,key=lambda x:str(x.get('sample_path',''))): f.write(json.dumps(r,sort_keys=True)+'\n')
    valid=[r for r in rows if r.get('valid')]
    total_seconds=time.perf_counter()-t_total
    summary={
        'schema':'ocrap-v48.91-common-exogenous-future-physical-sidecar-v1','engineering_version':ENGINEERING_VERSION,
        'valid':not errors and len(valid)==len(requested),'attribution_ready':not errors and len(valid)==len(requested),'errors':errors[:100],
        'requested_samples':len(requested),'valid_samples':len(valid),'labeled_candidate_pairs':pair_count,
        'output':str(args.output.resolve()),'output_sha256':_sha(args.output),
        'max_future_probability_error':max([float(r['future_probability_max_abs_error']) for r in valid],default=None),
        'max_active_root_margin_error':max([float(r['active_root_margin_max_abs_error']) for r in valid],default=None),
        'replay_config':args.replay_config,'womd_source_pattern_override':args.womd_source_pattern,
        'replay_provenance_resolution_counts':dict(sorted(provenance_resolution_counts.items())),
        'legacy_provenance_migration_used':bool(provenance_resolution_counts.get('index:legacy_wx_migration_key',0) or provenance_resolution_counts.get('pattern:cli_or_env_override',0) or provenance_resolution_counts.get('pattern:replay_config',0)),
        'resolved_source_index_min':min(resolved_indices) if resolved_indices else None,'resolved_source_index_max':max(resolved_indices) if resolved_indices else None,
        'worker_partition':{'num_workers':args.num_workers,'worker_index':args.worker_index,'rule':'source_scenario_index_mod_num_workers'},
        'performance':{
            'total_seconds':total_seconds,'audit_scan_seconds':audit_scan_seconds,'sample_load_seconds':sample_load_seconds,
            'provenance_seconds':provenance_seconds,'raw_scan_and_replay_seconds':raw_scan_seconds,'sample_replay_accumulated_seconds':replay_seconds,
            'processed_samples':processed,'target_source_indices':target_source_indices,'history_cache_hits':history_hits,
            'history_cache_hit_fraction':float(history_hits/max(processed,1)),
            'sparse_source_iterator':True,'metadata_only_future_metrics_skipped':True,
            'canonical_v48_14_sample_local_replay_profile':True,'fail_fast_replay_errors':int(args.fail_fast_replay_errors),
            'checkpoint_rows_reused':len(checkpoint_rows),'checkpoint_path':str(args.checkpoint) if args.checkpoint else None,
            'stage_timing_seconds':{k:float(v) for k,v in sorted(stage_timing.items())},
            'active_options_mean':float(np.mean(active_option_counts)) if active_option_counts else 0.0,
        },
        'canonical_npz_modified':False,'planner_parameters_trained':0,'teacher_labels_changed':False,
        'teacher_metadata_input_to_model':False,'dataset_reconstruction':False,'dataset_reselection':False,'test_roots_read':False,
        'purpose':'offline same-cohort teacher replay sidecar; canonical NPZ files and sample membership are never modified',
    }
    args.summary.write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'valid':summary['valid'],'worker_index':args.worker_index,'num_workers':args.num_workers,'requested_samples':len(requested),'errors':len(errors),'summary':str(args.summary),'performance':summary['performance']}),flush=True)
    return 0 if summary['valid'] else 30

if __name__=='__main__': raise SystemExit(main())
