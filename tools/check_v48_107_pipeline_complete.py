#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser()
 for k in ("runtime","balanced","precision","balanced_state","precision_state","comparison","v48_106_pipeline","v48_106_comparison"):ap.add_argument("--"+k.replace("_","-"),dest=k,type=Path,required=True)
 ap.add_argument("--output",type=Path,required=True);a=ap.parse_args();errors=[]
 docs={k:json.loads(getattr(a,k).read_text()) for k in ("runtime","balanced","precision","comparison","v48_106_pipeline","v48_106_comparison")}
 if not(docs["runtime"].get("valid") and docs["runtime"].get("attribution_ready")):errors.append("runtime")
 for v in ("balanced","precision"):
  d=docs[v]
  if not(d.get("valid") and d.get("engineering_version")=="v48.107.0-OC-FNAO" and d.get("variant")==v):errors.append(v)
 if not(docs["comparison"].get("valid") and docs["comparison"].get("attribution_ready")):errors.append("comparison")
 if not(docs["v48_106_pipeline"].get("valid") and docs["v48_106_pipeline"].get("preregistered_status")=="PREENCODER_ACTION_ORIENTATION_STOP"):errors.append("v106_pipeline")
 d106=docs["v48_106_comparison"].get("preregistered_decision") or {}
 if not(d106.get("status")=="PREENCODER_ACTION_ORIENTATION_STOP" and d106.get("next_branch")=="preencoder_action_orientation_insufficient_then_preregister_first_stage_i_block_nominal_invariant_action_orientation_objective_no_source_or_broad_encoder_sweep"):errors.append("v106_branch")
 artifacts={}
 for k in ("balanced","precision","balanced_state","precision_state","comparison","runtime"):
  p=getattr(a,k);artifacts[k]={"path":str(p.resolve()),"sha256":sha(p)}
 status=(docs["comparison"].get("preregistered_decision") or {}).get("status")
 out={"schema":"ocrap-v48.107-fnao-pipeline-complete-v1","engineering_version":"v48.107.0-OC-FNAO","valid":not errors,"attribution_ready":not errors,"errors":errors,"experiment_type":"first_stage_i_block_nominal_invariant_ordinal_action_orientation","artifacts":artifacts,"preregistered_status":status,
      "planner_parameters_trained":0,"stage_i_first_block_parameters_trained":444864,"stage_i_other_parameters_trained":0,"root_decoder_parameters_trained":0,"source_parameters_trained":0,"boundary_transport":False,"dataset_reconstruction":False,"regime_conditioning":False,"teacher_metadata_input_to_model":False,"test_roots_read":False,
      "v48_106_pipeline_sha256":sha(a.v48_106_pipeline),"v48_106_comparison_sha256":sha(a.v48_106_comparison)}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n");print(json.dumps({"valid":out["valid"],"status":status,"errors":errors}));return 0 if out["valid"] else 30
if __name__=="__main__":raise SystemExit(main())
