#!/usr/bin/env python3
from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path

from ocrap.v48_100_joint_root_semantic_decoder import (
    ALGORITHM_NAME, ENGINEERING_VERSION, JointRootSemanticDecoder,
    query_gradient_check, trainable_contract_check, zero_delta_decoder_identity_check,
)
from ocrap.v48_97_executable_recovery_state import root_permutation_invariance_check
from tools.run_v48_97_executable_recovery_state import action_strata_match_v48_96_synthetic_check, candidate_only_label_join_synthetic_check

FILES=[
    'scripts/run_v48_100_dcp_drfc_bcde_rifa_jrsd_two_gpu.sh',
    'src/ocrap/v48_100_joint_root_semantic_decoder.py',
    'tools/run_v48_100_joint_root_semantic_decoder.py',
    'tools/compare_v48_100_jrsd.py',
    'tools/check_v48_100_runtime_code_contract.py',
    'tools/check_v48_100_pipeline_complete.py',
]

def sha(p:Path)->str: return hashlib.sha256(p.read_bytes()).hexdigest()

def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument('--repo',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    repo=a.repo.resolve(); errors=[]; runtime_files={}
    for rel in FILES:
        p=(repo/rel).resolve(); inside=(repo==p or repo in p.parents); exists=p.is_file()
        runtime_files[rel]={'exists':exists,'inside_repo':inside,'path':str(p),'sha256':sha(p) if exists else None}
        if not exists or not inside: errors.append(rel)
    d_model=192; num_roots=8; expected=num_roots*d_model+(4*d_model+2)
    checks={
        'fixed_parameter_count':trainable_contract_check(d_model,num_roots),
        'zero_query_delta_reproduces_frozen_decoder':zero_delta_decoder_identity_check(32,5,4),
        'query_delta_receives_gradient':query_gradient_check(32,5,4),
        'semantic_chart_root_permutation_invariant':root_permutation_invariance_check(32),
        'candidate_only_v93_join_preserves_nominal':candidate_only_label_join_synthetic_check(),
        'action_evaluation_strata_match_v48_96':action_strata_match_v48_96_synthetic_check(),
    }
    for k,v in checks.items():
        if not v: errors.append(k)
    out={
        'schema':'ocrap-v48.100-runtime-code-contract-v1','engineering_version':ENGINEERING_VERSION,'valid':not errors,'attribution_ready':not errors,'errors':errors,
        'runtime_files':runtime_files,
        'scientific_contract':{
            'name':ALGORITHM_NAME,'planner_parameters_trained':0,'source_parameters_trained':0,'stage_i_parameters_trained':0,
            'root_decoder_body_parameters_trained':0,'root_logit_head_parameters_trained':0,
            'root_query_parameters_trained':num_roots*d_model,'recovery_chart_parameters_trained':4*d_model+2,'joint_representation_parameter_count':expected,
            'root_query_delta_zero_initialized':True,'v48_97_chart_initialized_then_jointly_trained':True,
            'structured_encoder_frozen':True,'root_cross_attention_weights_frozen':True,'root_self_attention_weights_frozen':True,'root_ffn_frozen':True,
            'root_logit_head_frozen':True,'source_training_off':True,'teacher_components_supervision_only':True,'teacher_metadata_input_to_model':False,
            'root_slot_bijection_assumed':False,'root_set_semantic_readout_permutation_invariant':True,
            'joint_static_and_candidate_delta_semantic_objective':True,'coordinate_scale_invariant_semantic_metric':True,
            'regime_conditioning':False,'boundary_transport':False,'relative_ranker_modified':False,'dataset_reconstruction':False,'dataset_reselection':False,
            'capacity_sweep':False,'threshold_sweep':False,
        },
        'synthetic_checks':checks,'test_roots_read':False,
    }
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'valid':out['valid'],'errors':errors,'parameters':expected}))
    return 0 if out['valid'] else 30
if __name__=='__main__': raise SystemExit(main())
