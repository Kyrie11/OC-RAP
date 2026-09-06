#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();
 for k in ("runtime","balanced","precision","balanced_state","precision_state","comparison","v48_103_pipeline","v48_103_comparison"): ap.add_argument("--"+k.replace("_","-"),dest=k,type=Path,required=True)
 ap.add_argument("--output",type=Path,required=True); a=ap.parse_args(); errors=[]
 docs={k:json.loads(getattr(a,k).read_text()) for k in ("runtime","balanced","precision","comparison","v48_103_pipeline","v48_103_comparison")}
 if not docs["runtime"].get("valid") or not docs["runtime"].get("attribution_ready"): errors.append("runtime")
 for v in ("balanced","precision"):
  d=docs[v]
  if not d.get("valid") or d.get("engineering_version")!="v48.104.0-OC-NICR" or d.get("variant")!=v: errors.append(v)
 if not docs["comparison"].get("valid") or not docs["comparison"].get("attribution_ready"): errors.append("comparison")
 if not docs["v48_103_pipeline"].get("valid") or docs["v48_103_pipeline"].get("preregistered_status")!="FACTORIZED_CONTROL_SUFFICIENT_STATE_STOP": errors.append("v103_pipeline")
 d103=docs["v48_103_comparison"].get("preregistered_decision") or {}
 if not(d103.get("factorized_state_representation_go") is True and d103.get("factorized_support_action_go") is False and d103.get("factorized_reserve_debt_go") is False): errors.append("v103_branch")
 artifacts={}
 for k in ("balanced","precision","balanced_state","precision_state","comparison","runtime"):
  p=getattr(a,k); artifacts[k]={"path":str(p.resolve()),"sha256":sha(p)}
 status=(docs["comparison"].get("preregistered_decision") or {}).get("status")
 out={"schema":"ocrap-v48.104-nicr-pipeline-complete-v1","engineering_version":"v48.104.0-OC-NICR","valid":not errors,"attribution_ready":not errors,"errors":errors,"experiment_type":"nominal_invariant_last_stage_i_block_control_refinement","artifacts":artifacts,"preregistered_status":status,"planner_parameters_trained":0,"stage_i_other_parameters_trained":0,"root_decoder_parameters_trained":0,"source_parameters_trained":0,"boundary_transport":False,"dataset_reconstruction":False,"regime_conditioning":False,"teacher_metadata_input_to_model":False,"test_roots_read":False,"v48_103_pipeline_sha256":sha(a.v48_103_pipeline),"v48_103_comparison_sha256":sha(a.v48_103_comparison)}
 a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n"); print(json.dumps({"valid":out["valid"],"status":status,"errors":errors})); return 0 if out["valid"] else 30
if __name__=="__main__": raise SystemExit(main())
