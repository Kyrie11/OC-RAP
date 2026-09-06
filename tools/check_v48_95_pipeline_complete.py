#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from ocrap.v48_95_native_recovery_observability import ENGINEERING_VERSION

def sha(p:Path)->str: return hashlib.sha256(p.read_bytes()).hexdigest()

def main()->int:
 ap=argparse.ArgumentParser(); ap.add_argument('--runtime',type=Path,required=True); ap.add_argument('--audit',type=Path,required=True); ap.add_argument('--comparison',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
 arts={}; errors=[]
 for name,p in [('runtime',a.runtime),('audit',a.audit),('comparison',a.comparison)]:
  if not p.is_file(): errors.append(f'missing {name} {p}'); continue
  j=json.loads(p.read_text());
  if not bool(j.get('valid')): errors.append(f'{name} invalid')
  if str(j.get('engineering_version'))!=ENGINEERING_VERSION: errors.append(f'{name} engineering version mismatch')
  arts[name]={'path':str(p),'sha256':sha(p)}
 cmp=json.loads(a.comparison.read_text()) if a.comparison.is_file() else {}
 out={
  'schema':'ocrap-v48.95-nroa-pipeline-complete-v1','engineering_version':ENGINEERING_VERSION,
  'valid':not errors,'attribution_ready':not errors,'errors':errors,'artifacts':arts,
  'experiment_type':'audit_only_frozen_native_support_reserve_observability','planner_parameters_trained':0,
  'preregistered_status':cmp.get('preregistered_decision',{}).get('status'),
  'dataset_reconstruction':False,'dataset_reselection':False,'teacher_labels_changed':False,
  'teacher_metadata_input_to_model':False,'boundary_transport':False,'relative_ranker_modified':False,
  'regime_conditioning':False,'womd_replay_performed':False,'test_roots_read':False,
 }
 a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print(json.dumps({'valid':out['valid'],'status':out['preregistered_status'],'errors':errors}))
 return 0 if out['valid'] else 30
if __name__=='__main__': raise SystemExit(main())
