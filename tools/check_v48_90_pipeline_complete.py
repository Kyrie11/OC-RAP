#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path

def sha(p:Path)->str:
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def main()->int:
 ap=argparse.ArgumentParser()
 for name in ('runtime','audit_index','audit_summary','comparison','v48_89_comparison','output'):
  ap.add_argument('--'+name.replace('_','-'),dest=name,type=Path,required=True)
 a=ap.parse_args(); errors=[]
 docs={}
 for k in ('runtime','audit_summary','comparison','v48_89_comparison'):
  p=getattr(a,k)
  if not p.is_file(): errors.append(f'missing {k}: {p}'); continue
  docs[k]=json.loads(p.read_text())
 if docs.get('runtime',{}).get('engineering_version')!='v48.90.0-OC-CEPT' or not docs.get('runtime',{}).get('valid'): errors.append('runtime invalid/version mismatch')
 if docs.get('audit_summary',{}).get('engineering_version')!='v48.90.0-OC-CEPT' or not docs.get('audit_summary',{}).get('valid'): errors.append('audit summary invalid/version mismatch')
 if docs.get('comparison',{}).get('engineering_version')!='v48.90.0-OC-CEPT' or not docs.get('comparison',{}).get('valid'): errors.append('comparison invalid/version mismatch')
 pp=docs.get('v48_89_comparison',{}).get('preregistered_decision') or {}
 if pp.get('status')!='COUNTERFACTUAL_ROOT_CORRESPONDENCE_STOP' or 'partition_stability' not in str(pp.get('next_branch','')): errors.append('V48.89 branch prerequisite mismatch')
 if not a.audit_index.is_file(): errors.append('audit index missing')
 else:
  line_count=sum(1 for line in a.audit_index.open() if line.strip())
  if line_count!=int(docs.get('audit_summary',{}).get('rows',-1)): errors.append(f'audit row count mismatch {line_count}')
  if sha(a.audit_index)!=docs.get('audit_summary',{}).get('output_sha256'): errors.append('audit index SHA mismatch')
 for role,ident in (docs.get('audit_summary',{}).get('label_identity') or {}).items():
  if not ident.get('teacher_value_identity_on_overlap') or int(ident.get('mismatches',1))!=0: errors.append(f'label identity invalid {role}')
 doc={'schema':'ocrap-v48.90-oc-cept-pipeline-complete-v1','engineering_version':'v48.90.0-OC-CEPT','valid':not errors,'attribution_ready':not errors,'errors':errors,
      'experiment_type':'audit_only_counterfactual_equivalence_partition_transport_adjudication','planner_parameters_trained':0,'teacher_labels_changed':False,'teacher_metadata_input_to_model':False,'dataset_reconstruction':False,'regime_conditioning':False,'boundary_transport':False,'relative_ranker_modified':False,'test_roots_read':False,
      'artifacts':{str(p):sha(p) for p in (a.runtime,a.audit_summary,a.comparison,a.audit_index) if p.is_file()}}
 a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'valid':doc['valid'],'errors':errors})); return 0 if doc['valid'] else 30
if __name__=='__main__': raise SystemExit(main())
