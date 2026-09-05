from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from ocrap.algorithms.lcv import normalize_weights, weighted_lcvar
from ocrap.v48_91_common_exogenous_physical_margin import (
    physical_margin_from_teacher_diag, future_physical_matrix, future_nested_tail_influence,
    audit_future_physical_response,
)

ROOT = Path(__file__).resolve().parents[1]

def sample(assign, sources, metas, mf, probs=None):
    mf=np.asarray(mf,dtype=np.float64);assign=np.asarray(assign,dtype=np.int64);F,L=mf.shape;K=int(assign.max())+1
    probs=np.asarray(probs if probs is not None else np.ones(F)/F,dtype=np.float64);probs=normalize_weights(probs)
    rp=np.zeros(K); M=np.zeros((K,L))
    for k in range(K):
        idx=np.where(assign==k)[0];rp[k]=probs[idx].sum()
        w=normalize_weights(probs[idx])
        for l in range(L):M[k,l]=weighted_lcvar(mf[idx,l],w,.2)
    s={'m_star':M.astype(np.float32),'root_probs':rp.astype(np.float32),'root_valid':np.ones(K,np.float32),'c_star':np.eye(K,dtype=np.float32),'option_valid':np.ones(L,np.float32),'root_assignments':assign,'future_probs':probs.astype(np.float32),'future_valid':np.ones(F,np.float32),'future_sources':np.asarray(sources),'future_metadata':json.dumps(metas,sort_keys=True),'recovery_modes':np.asarray(['stop']*L)}
    # nested-tail helper needs a stored scalar; set exact recomputation.
    from ocrap.v48_89_root_correspondence import nested_tail_influence
    _,r,_,_=nested_tail_influence({**s,'r_dep_star':np.float32(0.)});s['r_dep_star']=np.float32(r)
    return s

def test_pre_structural_physical_min_ignores_inactive_and_structural_value():
    d=SimpleNamespace(active={'clearance':True,'route':False,'stability':True},component_margins={'clearance':.4,'route':-8.,'stability':-.2})
    assert physical_margin_from_teacher_diag(d)==-.2
    m=future_physical_matrix([[d]],np.array([True]))
    assert m.shape==(1,1) and m[0,0]==-.2

def test_future_tail_influence_reconstructs_root_tail_mass():
    mf=np.array([[-1.0],[1.0],[.5]],dtype=float)
    s=sample([0,0,1],['replay','reactive','targeted'],[{}, {'rollout_variant':'x'}, {'targeted_type':'z'}],mf,probs=[.2,.3,.5])
    fmass,err=future_nested_tail_influence(s,mf)
    assert fmass.shape==mf.shape
    assert err<1e-8
    assert fmass.sum()>0

def test_common_exogenous_future_response_is_root_slot_permutation_invariant_and_signed():
    metas=[{'rollout_variant':'shared-a'},{'targeted_type':'shared-b'}]
    # Candidate swaps root slots relative to nominal, but exogenous futures are the same.
    cs=sample([1,0],['reactive','targeted'],metas,[[-.5],[.2]],probs=[.5,.5])
    ns=sample([0,1],['reactive','targeted'],metas,[[-1.0],[.2]],probs=[.5,.5])
    cp=np.array([[.5],[.2]])   # physical response +1.0 on the weak shared-a future
    np_=np.array([[-.5],[.2]])
    rec=audit_future_physical_response(cs,ns,np.array([[-.5],[.2]]),np.array([[-1.0],[.2]]),cp,np_)
    assert rec.valid
    assert rec.common_exogenous_tail_coverage>0.99
    assert rec.response_sign_identifiable_mass>0.99
    assert rec.signed_response_score>0.99

def test_different_exogenous_realization_is_not_matched():
    ca={'targeted_type':'waymax_hidden_vehicle_yield','scenario_augmented':True,'hidden_spawn_xy':[1.,2.],'hidden_actor_object_index':3}
    na={'targeted_type':'waymax_hidden_vehicle_yield','scenario_augmented':True,'hidden_spawn_xy':[5.,2.],'hidden_actor_object_index':4}
    cs=sample([0],['targeted'],[ca],[[-1.]])
    ns=sample([0],['targeted'],[na],[[-1.]])
    rec=audit_future_physical_response(cs,ns,np.array([[-1.]]),np.array([[-1.]]),np.array([[1.]]),np.array([[-1.]]))
    assert rec.valid
    assert rec.common_exogenous_tail_coverage==0.0
    assert rec.response_sign_identifiable_mass==0.0


def _load_v4891_sidecar_tool_module():
    import importlib.util
    path = ROOT / "tools" / "build_v48_91_common_exogenous_physical_sidecar.py"
    spec = importlib.util.spec_from_file_location("v4891_sidecar_tool", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v48_91_1_legacy_wx_source_index_is_a_provenance_migration_key(tmp_path: Path) -> None:
    mod = _load_v4891_sidecar_tool_module()
    sample = {
        "source_scenario_index": np.asarray(-1),
        "womd_source_pattern": np.asarray(""),
        "scene_id": np.asarray("waymax_deadbeef__wx00012659"),
        "legacy_scenario_id": np.asarray(""),
        "original_scenario_id": np.asarray(""),
        "__path__": str(tmp_path / "waymax_deadbeef__wx00012659_t0029_a05.npz"),
    }
    prov = mod._resolve_replay_provenance(
        sample,
        source_pattern_override="/womd/validation/validation_tfexample.tfrecord@150",
        replay_config_pattern=None,
    )
    assert prov["index"] == 12659
    assert prov["index_source"] == "legacy_wx_migration_key"
    assert prov["pattern_source"] == "cli_or_env_override"


def test_v48_91_1_explicit_npz_provenance_remains_authoritative(tmp_path: Path) -> None:
    mod = _load_v4891_sidecar_tool_module()
    sample = {
        "source_scenario_index": np.asarray(42),
        "womd_source_pattern": np.asarray("/stored/validation.tfrecord@150"),
        "scene_id": np.asarray("waymax_deadbeef__wx00012659"),
        "__path__": str(tmp_path / "sample.npz"),
    }
    prov = mod._resolve_replay_provenance(
        sample,
        source_pattern_override="/override/validation.tfrecord@150",
        replay_config_pattern="/config/validation.tfrecord@150",
    )
    assert prov["index"] == 42
    assert prov["index_source"] == "npz"
    assert prov["pattern"] == "/stored/validation.tfrecord@150"
    assert prov["pattern_source"] == "npz"


def test_v48_91_1_runner_accepts_read_only_womd_source_override() -> None:
    runner = (ROOT / "scripts" / "run_v48_91_dcp_drfc_bcde_rifa_cepmi.sh").read_text()
    assert "V4891_WOMD_SOURCE" in runner
    assert "--womd-source-pattern" in runner


def test_v48_91_1_origin_resume_contract_can_recover_pattern_and_build_config(tmp_path: Path) -> None:
    mod = _load_v4891_sidecar_tool_module()
    role = tmp_path / "protocol" / "evidence_adapt_dev_near_contact"
    source = tmp_path / "calibration_near_contact"
    raw = tmp_path / "raw_calibration_near_contact"
    shard2 = tmp_path / "shards" / "calibration_near_w2"
    shard3 = tmp_path / "shards" / "calibration_near_w3"
    (role / "samples").mkdir(parents=True)
    source.mkdir(); raw.mkdir(); shard2.mkdir(parents=True); shard3.mkdir(parents=True)
    (role / "split_provenance.json").write_text(json.dumps({"source": str(source)}))
    (source / "scene_filter_provenance.json").write_text(json.dumps({"source": str(raw)}))
    (raw / "merged_dataset_summary.json").write_text(json.dumps({"input_roots": [str(shard2), str(shard3)]}))
    for root, worker in ((shard2, 2), (shard3, 3)):
        (root / "dataset_summary.json").write_text(json.dumps({
            "scenario_start_index": 11000, "scenario_stride": 6, "scenario_worker_index": worker,
            "generation": {"num_roots": 8, "num_recovery_options": 12},
        }))
        (root / "resume_contract.json").write_text(json.dumps({
            "semantic_config": {"womd_patterns": "/exact/validation.tfrecord@150", "sample_rate_hz": 10}
        }))
    sample_path = role / "samples" / "waymax_deadbeef__wx00012658_t0029_a05.npz"
    sample_path.write_bytes(b"")
    origin = mod._origin_replay_metadata(sample_path, 12658)
    assert origin["origin_shard_root"] == str(shard2)
    assert origin["semantic_config"]["womd_patterns"] == "/exact/validation.tfrecord@150"

def test_v48_91_3_canonical_near_profile_reconstructs_nonartifact_balanced_pass(tmp_path: Path) -> None:
    mod = _load_v4891_sidecar_tool_module()
    role = tmp_path / 'calibration_v48_14_prism_4814' / 'evidence_adapt_dev_near_contact' / 'samples'
    role.mkdir(parents=True)
    sample = {
        '__path__': str(role / 'waymax_deadbeef__wx00011038_t0010_a00.npz'),
        'future_metadata': json.dumps([
            {'rollout_variant':'natural_log_playback','scenario_augmented':False},
            {'visible_perturbation':True,'visible_branch':'visible_brake','scenario_augmented':True,'artifact_mined':False},
        ]),
    }
    prof, meta = mod._canonical_v4814_sample_profile(sample)
    assert meta == {'profile_id':'calibration_v48_14_prism_4814','role':'near','artifact_pass':False}
    assert prof['num_reactive_futures'] == 2
    assert prof['num_targeted_futures'] == 8
    assert prof['targeted_future_kinds'] == ['hidden_vehicle_yields','hidden_vehicle_accelerates','low_friction_braking','control_delay_noise']
    assert prof['waymax']['enable_visible_perturbation_roots'] is True
    assert prof['waymax']['augmented_hidden_from_unknown_only'] is True
    assert prof['artifact']['force_mine'] is False
    assert prof['artifact']['mine_probability'] == 0.0
    assert prof['dataset_quality']['require_artifact_pairs'] is False


def test_v48_91_3_canonical_contact_profile_reconstructs_mined_balanced_pass(tmp_path: Path) -> None:
    mod = _load_v4891_sidecar_tool_module()
    role = tmp_path / 'calibration_v48_14_prism_4814' / 'certificate_pool_contact' / 'samples'
    role.mkdir(parents=True)
    sample = {
        '__path__': str(role / 'waymax_deadbeef__wx00011053_t0010_a13.npz'),
        'future_metadata': json.dumps([
            {'targeted_type':'waymax_hidden_vehicle_yield','scenario_augmented':True,'artifact_mined':True,'artifact_branch':'yield'},
            {'targeted_type':'waymax_hidden_vehicle_accelerate','scenario_augmented':True,'artifact_mined':True,'artifact_branch':'accelerate'},
        ]),
    }
    prof, meta = mod._canonical_v4814_sample_profile(sample)
    assert meta == {'profile_id':'calibration_v48_14_prism_4814','role':'contact','artifact_pass':True}
    assert prof['num_targeted_futures'] == 10
    assert 'contact_impulse_surrogate' in prof['targeted_future_kinds']
    assert 'secondary_collision_approach' in prof['targeted_future_kinds']
    assert prof['artifact']['force_mine'] is True
    assert prof['artifact']['mine_probability'] == 1.0
    assert prof['artifact']['use_margin_override'] is True
    assert prof['dataset_quality']['require_artifact_pairs'] is True
    assert prof['waymax']['skip_waymax_rollout_for_augmented_override'] is True
    assert prof['waymax']['apply_artifact_override_to_screened_options'] is True


def test_v48_91_3_fail_fast_identity_guard_is_defaulted() -> None:
    tool = (ROOT / 'tools' / 'build_v48_91_common_exogenous_physical_sidecar.py').read_text()
    assert '--fail-fast-replay-errors' in tool
    assert 'V4891_FAIL_FAST_REPLAY_ERRORS' in tool
    assert "v48.91_replay_fail_fast" in tool

def test_v48_91_3_checkpoint_resume_requires_same_version_options_and_npz_stat(tmp_path: Path) -> None:
    mod = _load_v4891_sidecar_tool_module()
    sample = tmp_path / 'sample.npz'
    sample.write_bytes(b'abc')
    requested = {str(sample): {1, 3}}
    ck = tmp_path / 'checkpoint.jsonl'
    row = {'valid': True, 'sample_path': str(sample), 'option_ids': [1,3]}
    size, mtime = mod._checkpoint_stat(str(sample))
    ck.write_text(json.dumps({
        'engineering_version': mod.ENGINEERING_VERSION,
        'sample_path': str(sample), 'option_ids': [1,3],
        'sample_size': size, 'sample_mtime_ns': mtime, 'row': row,
    })+'\n')
    got = mod._load_replay_checkpoint(ck, requested)
    assert str(sample) in got
    sample.write_bytes(b'abcd')
    got2 = mod._load_replay_checkpoint(ck, requested)
    assert str(sample) not in got2
