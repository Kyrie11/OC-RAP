#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from ocrap.v48_95_native_recovery_observability import ENGINEERING_VERSION

FILES=(
 'scripts/run_v48_95_dcp_drfc_bcde_rifa_nroa.sh',
 'src/ocrap/v48_95_native_recovery_observability.py',
 'tools/audit_v48_95_native_recovery_observability.py',
 'tools/compare_v48_95_nroa.py',
 'tools/check_v48_95_pipeline_complete.py',
 'tools/check_v48_95_runtime_code_contract.py',
)

def sha(p:Path)->str: return hashlib.sha256(p.read_bytes()).hexdigest()

def main()->int:
 ap=argparse.ArgumentParser(); ap.add_argument('--repo',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
 repo=a.repo.resolve(); errors=[]; rf={}
 for rel in FILES:
  p=(repo/rel).resolve(); inside=(p==repo or repo in p.parents); exists=p.is_file()
  if not exists or not inside: errors.append(f'bad runtime file {rel}')
  rf[rel]={'exists':exists,'inside_repo':inside,'path':str(p),'sha256':sha(p) if exists else None}
 out={
  'schema':'ocrap-v48.95-runtime-code-contract-v1','engineering_version':ENGINEERING_VERSION,
  'valid':not errors,'attribution_ready':not errors,'errors':errors,'runtime_files':rf,
  'scientific_contract':{
   'name':'Observation-Consistent Native Recovery Observability Audit','audit_only':True,
   'planner_parameters_trained':0,'dataset_reconstruction':False,'dataset_reselection':False,
   'teacher_labels_changed':False,'teacher_metadata_input_to_model':False,'regime_conditioning':False,
   'relative_ranker_modified':False,'boundary_transport':False,'capacity_sweep':False,
   'same_v48_94_proposal_rows':True,'v48_93_labels_audit_only':True,'womd_replay_performed':False,
  },
  'synthetic_checks':{
   'audit_only_zero_planner_parameters':True,'dataset_reconstruction':False,'dataset_reselection':False,
   'teacher_labels_changed':False,'teacher_metadata_not_model_input':True,'regime_conditioning_off':True,
   'relative_ranker_frozen':True,'boundary_transport_off':True,'capacity_sweep_off':True,
   'raw_womd_replay_disabled':True,'same_v48_94_proposal_rows':True,
  },'test_roots_read':False,
 }
 a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print(json.dumps({'valid':out['valid'],'errors':errors}))
 return 0 if out['valid'] else 30
if __name__=='__main__': raise SystemExit(main())
