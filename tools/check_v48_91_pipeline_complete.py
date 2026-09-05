#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from ocrap.v48_91_common_exogenous_physical_margin import ENGINEERING_VERSION
def sha(p):
 h=hashlib.sha256();
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();
 for k in ('runtime','source_contract','sidecar','sidecar_summary','audit','audit_summary','comparison','v48_90_comparison'):ap.add_argument('--'+k.replace('_','-'),dest=k,type=Path,required=True)
 ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();errors=[]
 docs={}
 for k in ('runtime','source_contract','sidecar_summary','audit_summary','comparison','v48_90_comparison'):
  p=getattr(a,k)
  if not p.is_file():errors.append(f'missing {k}: {p}');continue
  docs[k]=json.loads(p.read_text())
 if not a.sidecar.is_file():errors.append(f'missing sidecar {a.sidecar}')
 if not a.audit.is_file():errors.append(f'missing audit {a.audit}')
 for k in ('runtime','source_contract','sidecar_summary','audit_summary','comparison'):
  if k in docs and not docs[k].get('valid'):errors.append(f'{k} invalid')
 for k in ('runtime','sidecar_summary','audit_summary','comparison'):
  if k in docs and str(docs[k].get('engineering_version')) != ENGINEERING_VERSION:
   errors.append(f"{k}.engineering_version={docs[k].get('engineering_version')!r} != {ENGINEERING_VERSION!r}")
 q90=(docs.get('v48_90_comparison',{}).get('preregistered_decision') or {})
 if not(q90.get('exogenous_partition_transport_go') and not q90.get('transport_physical_response_identifiability_go')):errors.append('V48.90 prerequisite mismatch')
 for dname in ('runtime','sidecar_summary','audit_summary','comparison'):
  d=docs.get(dname,{})
  for key,want in [('planner_parameters_trained',0),('dataset_reconstruction',False),('teacher_metadata_input_to_model',False)]:
   if key in d and d.get(key)!=want:errors.append(f'{dname}.{key}={d.get(key)!r} != {want!r}')
 if docs.get('sidecar_summary',{}).get('dataset_reselection') is not False:errors.append('sidecar dataset_reselection not false')
 if docs.get('comparison',{}).get('boundary_transport') is not False:errors.append('boundary transport must remain false')
 artifacts={k:{'path':str(getattr(a,k).resolve()),'sha256':sha(getattr(a,k))} for k in ('runtime','source_contract','sidecar','sidecar_summary','audit','audit_summary','comparison') if getattr(a,k).is_file()}
 out={'schema':'ocrap-v48.91-pipeline-complete-v1','engineering_version':ENGINEERING_VERSION,'valid':not errors,'attribution_ready':not errors,'errors':errors,'experiment_type':'audit_only_common_exogenous_future_physical_margin_identifiability','planner_parameters_trained':0,'dataset_reconstruction':False,'dataset_reselection':False,'teacher_labels_changed':False,'teacher_metadata_input_to_model':False,'boundary_transport':False,'relative_ranker_modified':False,'regime_conditioning':False,'test_roots_read':False,'artifacts':artifacts,'preregistered_status':(docs.get('comparison',{}).get('preregistered_decision') or {}).get('status')}
 a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps({'valid':out['valid'],'errors':errors,'status':out['preregistered_status']}));return 0 if out['valid'] else 30
if __name__=='__main__':raise SystemExit(main())
