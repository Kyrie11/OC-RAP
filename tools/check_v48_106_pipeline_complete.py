#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import torch
ENGINEERING_VERSION='v48.106.0-OC-PEAO'
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def main()->int:
 ap=argparse.ArgumentParser()
 for k in ('runtime','balanced','precision','balanced_state','precision_state','comparison','v48_105_pipeline','v48_105_comparison','v48_102_comparison'):
  ap.add_argument('--'+k.replace('_','-'),dest=k,type=Path,required=True)
 ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();errors=[]
 docs={k:json.loads(getattr(a,k).read_text()) for k in ('runtime','balanced','precision','comparison','v48_105_pipeline','v48_105_comparison','v48_102_comparison')}
 if not docs['runtime'].get('valid') or not docs['runtime'].get('attribution_ready'):errors.append('runtime')
 for v in ('balanced','precision'):
  d=docs[v]
  if not d.get('valid') or d.get('engineering_version')!=ENGINEERING_VERSION or d.get('variant')!=v or not d.get('audit_only') or not d.get('preencoder_only'):errors.append(v)
  s=torch.load(getattr(a,f'{v}_state'),map_location='cpu',weights_only=False)
  if s.get('engineering_version')!=ENGINEERING_VERSION or s.get('variant')!=v or s.get('schema')!='ocrap-v48.106-peao-probe-state-v1':errors.append(f'{v}_state')
  if 'adapted_last_block_state' in s or 'state_dict' in s or 'adapted_first_block_state' in s:errors.append(f'{v}_state_contains_model_parameters')
 if not docs['comparison'].get('valid') or not docs['comparison'].get('attribution_ready'):errors.append('comparison')
 p105=docs['v48_105_pipeline'];d105=docs['v48_105_comparison'].get('preregistered_decision') or {}
 if not(p105.get('valid') and p105.get('attribution_ready') and p105.get('preregistered_status')=='PRELAST_ACTION_EQUIVARIANCE_LOCALIZATION_STOP'):errors.append('v105_pipeline')
 if not(d105.get('status')=='PRELAST_ACTION_EQUIVARIANCE_LOCALIZATION_STOP' and d105.get('next_branch')=='prelast_action_equivariance_insufficient_then_preregister_one_block_earlier_action_interaction_audit_no_training_or_source_sweep'):errors.append('v105_branch')
 d102=docs['v48_102_comparison'].get('preregistered_decision') or {}
 if d102.get('status')!='STAGE_I_ACTION_INFORMATION_SUFFICIENCY_STOP':errors.append('v102_reference')
 artifacts={}
 for k in ('balanced','precision','balanced_state','precision_state','comparison','runtime'):
  p=getattr(a,k);artifacts[k]={"path":str(p.resolve()),"sha256":sha(p)}
 status=(docs['comparison'].get('preregistered_decision') or {}).get('status')
 out={"schema":"ocrap-v48.106-peao-pipeline-complete-v1","engineering_version":ENGINEERING_VERSION,"valid":not errors,"attribution_ready":not errors,"errors":errors,
 "experiment_type":"audit_only_preencoder_stage_i_action_orientation_lineage","artifacts":artifacts,"preregistered_status":status,"planner_parameters_trained":0,
 "stage_i_parameters_trained":0,"root_decoder_parameters_trained":0,"source_parameters_trained":0,"boundary_transport":False,"dataset_reconstruction":False,
 "regime_conditioning":False,"teacher_metadata_input_to_model":False,"test_roots_read":False,"v48_105_pipeline_sha256":sha(a.v48_105_pipeline),
 "v48_105_comparison_sha256":sha(a.v48_105_comparison),"v48_102_comparison_sha256":sha(a.v48_102_comparison)}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n");print(json.dumps({"valid":out['valid'],"status":status,"errors":errors}));return 0 if out['valid'] else 30
if __name__=='__main__':raise SystemExit(main())
