#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from ocrap.algorithms.lcv import normalize_weights, weighted_lcvar
from ocrap.config.defaults import deep_update
from ocrap.config.yaml_io import load_config
from ocrap.data.build.history import construct_history
from ocrap.data.schema import CandidatePrefix, RecoveryOption
from ocrap.data.serialization import load_npz_selected
from ocrap.data.waymax_loader import iter_waymax_womd_scenarios
from ocrap.simulation.futures import generate_counterfactual_futures
from ocrap.simulation.teacher import compute_future_option_margins
from ocrap.v48_89_root_correspondence import nested_tail_influence
from ocrap.v48_90_partition_transport import future_class_keys
from ocrap.v48_91_common_exogenous_physical_margin import future_physical_matrix


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


def _find_summary(path:Path)->Path|None:
    p=path.resolve().parent
    for _ in range(6):
        cand=p/'dataset_summary.json'
        if cand.is_file(): return cand
        if p.parent==p: break
        p=p.parent
    return None


def _config_for_sample(sample:dict[str,Any], base_cfg:dict[str,Any])->tuple[dict[str,Any],str|None]:
    cfg=json.loads(json.dumps(base_cfg))
    sp=_find_summary(Path(sample['__path__']))
    if sp:
        s=json.loads(sp.read_text())
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
    cfg['womd_patterns']=str(_scalar(sample,'womd_source_pattern',''))
    # Source scanning is performed externally by exact stored source index.
    cfg['scenario_start_index']=0; cfg['scenario_stride']=1; cfg['scenario_worker_index']=0
    return cfg,(str(sp) if sp else None)


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


def _replay_one(raw,sample:dict[str,Any],option_ids:list[int],base_cfg:dict[str,Any],alpha_intra:float)->dict[str,Any]:
    cfg,summary_path=_config_for_sample(sample,base_cfg)
    t=int(_scalar(sample,'time_index',-1)); history=construct_history(raw,t,cfg); hcheck=_history_check(history,sample)
    prefix=_prefix(sample)
    futures=generate_counterfactual_futures(history,prefix,cfg)
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
    if stored_keys!=replay_keys or not np.array_equal(stored_unres,replay_unres):
        raise ValueError('exogenous future-class replay mismatch')

    opts=_options(sample,option_ids)
    m_sub,diags=compute_future_option_margins(history,prefix,futures,opts,cfg)
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
    if root_err>2e-5: raise ValueError(f'active root-margin replay mismatch max_abs={root_err}')
    return {
        'valid':True,'sample_path':sample['__path__'],'sample_sha256':_sha(Path(sample['__path__'])),
        'scene_id':str(_scalar(sample,'scene_id','')),'time_index':t,'candidate_index':int(_scalar(sample,'candidate_index',-1)),
        'source_scenario_index':int(_scalar(sample,'source_scenario_index',-1)),'womd_source_pattern':str(_scalar(sample,'womd_source_pattern','')),
        'option_ids':[int(x) for x in option_ids],'future_count':F,'option_count':L,
        'm_future_structural':{str(l):struct[:,l].tolist() for l in option_ids},
        'm_future_physical':{str(l):phys[:,l].tolist() for l in option_ids},
        'future_probability_max_abs_error':perr,'active_root_margin_max_abs_error':float(root_err),
        'history_max_abs_error':hcheck,'dataset_summary_path':summary_path,
    }


def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--v48-90-audit',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True,help='gzip JSONL sidecar')
    ap.add_argument('--summary',type=Path,required=True)
    ap.add_argument('--replay-config',type=str,default=None,help='optional exact dataset build YAML; defaults + dataset-summary inference otherwise')
    ap.add_argument('--alpha',type=float,default=0.2); ap.add_argument('--beta',type=float,default=0.2); ap.add_argument('--top-m',type=int,default=8)
    ap.add_argument('--intra-root-alpha',type=float,default=0.2)
    args=ap.parse_args()
    base=load_config(args.replay_config)

    requested:dict[str,set[int]]=defaultdict(set); path_to_role:dict[str,set[str]]=defaultdict(set); pair_count=0
    with args.v48_90_audit.open(encoding='utf-8') as f:
        for line in f:
            r=json.loads(line)
            if not(r.get('valid') and r.get('label_available')): continue
            cp=str(Path(r['sample_path']).resolve()); npth=str(Path(r['nominal_sample_path']).resolve())
            cs=_load(Path(cp)); mass,*_=nested_tail_influence(cs,alpha=args.alpha,beta=args.beta,top_m=args.top_m)
            ids=np.where(np.sum(mass,axis=0)>1e-12)[0].tolist()
            if not ids: continue
            requested[cp].update(map(int,ids)); requested[npth].update(map(int,ids)); pair_count+=1
            role=str(r['dataset_role']); path_to_role[cp].add(role); path_to_role[npth].add(role)
    if not requested: raise SystemExit('no labeled V48.90 cohort samples/options to replay')

    samples={p:_load(Path(p)) for p in requested}
    groups:dict[tuple[str,int],dict[int,list[str]]]=defaultdict(lambda:defaultdict(list))
    for p,s in samples.items():
        pattern=str(_scalar(s,'womd_source_pattern','')); idx=int(_scalar(s,'source_scenario_index',-1)); maxobj=int(_scalar(s,'waymax_max_num_objects',-1))
        if not pattern or idx<0: raise SystemExit(f'missing raw replay provenance for {p}: pattern={pattern!r} index={idx}')
        if maxobj<=0: maxobj=int(np.asarray(s['agent_history']).shape[1])
        groups[(pattern,maxobj)][idx].append(p)

    rows=[]; errors=[]
    for (pattern,maxobj),byidx in groups.items():
        parser_cfg=json.loads(json.dumps(base)); parser_cfg['data_source']='womd'; parser_cfg['simulation_backend']='waymax_closed_loop'; parser_cfg['womd_patterns']=pattern; parser_cfg['max_agents']=maxobj
        maxidx=max(byidx)
        seen=set()
        for raw in iter_waymax_womd_scenarios(pattern,max_scenarios=maxidx+1,parser_cfg=parser_cfg):
            idx=int(raw.metadata.get('_waymax_scenario_index',-1))
            if idx not in byidx: continue
            seen.add(idx)
            for p in byidx[idx]:
                s=samples[p]
                try:
                    row=_replay_one(raw,s,sorted(requested[p]),base,args.intra_root_alpha)
                    row['dataset_roles']=sorted(path_to_role[p]); rows.append(row)
                except Exception as exc:
                    errors.append(f'{p}: {exc}')
                    rows.append({'valid':False,'sample_path':p,'error':str(exc),'dataset_roles':sorted(path_to_role[p])})
        missing=sorted(set(byidx)-seen)
        if missing: errors.append(f'raw source indices not encountered pattern={pattern}: {missing[:20]} count={len(missing)}')

    args.output.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(args.output,'wt',encoding='utf-8') as f:
        for r in rows: f.write(json.dumps(r,sort_keys=True)+'\n')
    valid=[r for r in rows if r.get('valid')]
    summary={
        'schema':'ocrap-v48.91-common-exogenous-future-physical-sidecar-v1','engineering_version':'v48.91.0-OC-CEPMI',
        'valid':not errors and len(valid)==len(requested),'attribution_ready':not errors and len(valid)==len(requested),'errors':errors[:100],
        'requested_samples':len(requested),'valid_samples':len(valid),'labeled_candidate_pairs':pair_count,
        'output':str(args.output.resolve()),'output_sha256':_sha(args.output),
        'max_future_probability_error':max([float(r['future_probability_max_abs_error']) for r in valid],default=None),
        'max_active_root_margin_error':max([float(r['active_root_margin_max_abs_error']) for r in valid],default=None),
        'replay_config':args.replay_config,'planner_parameters_trained':0,'teacher_labels_changed':False,
        'teacher_metadata_input_to_model':False,'dataset_reconstruction':False,'dataset_reselection':False,'test_roots_read':False,
        'purpose':'offline same-cohort teacher replay sidecar; canonical NPZ files and sample membership are never modified',
    }
    args.summary.write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'valid':summary['valid'],'requested_samples':len(requested),'errors':len(errors),'summary':str(args.summary)}))
    return 0 if summary['valid'] else 30

if __name__=='__main__': raise SystemExit(main())
