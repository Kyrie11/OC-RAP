#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from ocrap.v48_106_preencoder_action_orientation_audit import ENGINEERING_VERSION, preencoder_contract_checks

ACTIVE=[
"scripts/run_v48_106_dcp_drfc_bcde_rifa_peao_two_gpu.sh",
"src/ocrap/v48_106_preencoder_action_orientation_audit.py",
"tools/run_v48_106_preencoder_action_orientation_audit.py",
"tools/compare_v48_106_peao.py",
"tools/check_v48_106_runtime_code_contract.py",
"tools/check_v48_106_pipeline_complete.py",
]
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def main()->int:
 ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();repo=a.repo.resolve();errors=[];files={}
 for rel in ACTIVE:
  p=(repo/rel).resolve();inside=str(p).startswith(str(repo));files[rel]={"exists":p.is_file(),"inside_repo":inside,"path":str(p),"sha256":sha(p) if p.is_file() else None}
  if not(p.is_file() and inside):errors.append(f"runtime_file:{rel}")
 checks=preencoder_contract_checks(192)
 for k,v in checks.items():
  if not v:errors.append(k)
 out={"schema":"ocrap-v48.106-peao-runtime-code-contract-v1","engineering_version":ENGINEERING_VERSION,"valid":not errors,"attribution_ready":not errors,"errors":errors,"runtime_files":files,
 "scientific_contract":{
  "audit_only":True,"preencoder_stage_i_tokens_only":True,"before_any_transformer_layer":True,"historical_encoder_layer_count_required":2,
  "same_v48_102_summary_operator":True,"same_v48_102_linear_probe_recipe":True,"semantic_token_positions_preserved":True,"agent_set_summary_permutation_invariant":True,
  "same_v48_105_action_interaction_subspace":True,"action_interaction_definition":"control_plus_scene_context_plus_agent_set_moments_excluding_cls_and_ego_history",
  "signed_train_to_heldout_orientation_diagnostic":True,"signed_orientation_diagnostic_controls_branching":False,
  "within_group_action_permutation_control":True,"stage_i_parameters_trained":0,"root_decoder_parameters_trained":0,"source_parameters_trained":0,"planner_parameters_trained":0,
  "boundary_transport":False,"broad_encoder_training":False,"regime_conditioning":False,"teacher_metadata_input_to_model":False,"capacity_sweep":False,"threshold_sweep":False},
 "synthetic_checks":checks,"test_roots_read":False}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n");print(json.dumps({"valid":out["valid"],"errors":errors}));return 0 if out["valid"] else 30
if __name__=='__main__':raise SystemExit(main())
