#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import torch
from torch import nn
from ocrap.v48_104_nominal_invariant_control_refinement import ENGINEERING_VERSION, initialization_identity_check, NominalInvariantLastBlockRefinement

ACTIVE=["scripts/run_v48_104_dcp_drfc_bcde_rifa_nicr_two_gpu.sh","src/ocrap/v48_104_nominal_invariant_control_refinement.py","tools/run_v48_104_nominal_invariant_control_refinement.py","tools/compare_v48_104_nicr.py","tools/check_v48_104_runtime_code_contract.py","tools/check_v48_104_pipeline_complete.py"]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--repo",type=Path,required=True); ap.add_argument("--output",type=Path,required=True); a=ap.parse_args(); repo=a.repo.resolve(); errors=[]; files={}
 for rel in ACTIVE:
  p=(repo/rel).resolve(); ok=p.is_file() and str(p).startswith(str(repo)); files[rel]={"exists":p.is_file(),"inside_repo":str(p).startswith(str(repo)),"path":str(p),"sha256":sha(p) if p.is_file() else None}
  if not ok: errors.append(f"runtime_file:{rel}")
 layer=nn.TransformerEncoderLayer(d_model=192,nhead=4,dim_feedforward=768,dropout=.1,batch_first=True,activation="gelu",norm_first=True); m=NominalInvariantLastBlockRefinement(layer,nn.LayerNorm(192)); n=m.parameter_count
 if n!=444864: errors.append(f"last_block_parameter_count:{n}")
 if not initialization_identity_check(): errors.append("initialization_identity")
 out={"schema":"ocrap-v48.104-nicr-runtime-code-contract-v1","engineering_version":ENGINEERING_VERSION,"valid":not errors,"attribution_ready":not errors,"errors":errors,"runtime_files":files,"scientific_contract":{"last_stage_i_transformer_block_only":True,"expected_last_block_parameters_d192":444864,"frozen_v103_factorized_readout":True,"frozen_v103_readout_parameters":1540,"counterfactual_residual_against_frozen_base_block":True,"nominal_memory_exact_identity":True,"response_only_objective":True,"dropout_disabled_during_refinement":True,"root_decoder_parameters_trained":0,"source_parameters_trained":0,"planner_parameters_trained":0,"stage_i_other_parameters_trained":0,"boundary_transport":False,"broad_encoder_training":False,"regime_conditioning":False,"teacher_metadata_input_to_model":False,"capacity_sweep":False,"lr_or_epoch_sweep":False},"synthetic_checks":{"initial_function_identity":initialization_identity_check()},"test_roots_read":False}
 a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n"); print(json.dumps({"valid":out["valid"],"parameters":n,"errors":errors})); return 0 if out["valid"] else 30
if __name__=="__main__": raise SystemExit(main())
