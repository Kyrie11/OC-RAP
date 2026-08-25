#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np

KINDS={
 'dev_near':'dev_diagnostic_near_v48.proposal_rows.jsonl',
 'dev_contact':'dev_diagnostic_contact_v48.proposal_rows.jsonl',
 'certificate_near':'direct_value_risk_near_v48.proposal_rows.jsonl',
 'certificate_contact':'direct_value_risk_contact_v48.proposal_rows.jsonl',
}
VARIANTS=('balanced','precision')


def row_key(r):
 return (r.get('scene'), r.get('time'), r.get('fold'), r.get('candidate'))


def rows(run:Path,v:str,k:str):
 p=run/'candidates'/v/'calibration'/KINDS[k]
 if not p.is_file():
  raise FileNotFoundError(f'missing calibration proposal rows: {p}')
 out={}
 for lineno,line in enumerate(p.read_text().splitlines(),1):
  if not line.strip(): continue
  r=json.loads(line); key=row_key(r)
  if key in out:
   raise ValueError(f'duplicate proposal-row key at {p}:{lineno}: {key!r}')
  out[key]=r
 if not out:
  raise ValueError(f'empty calibration proposal rows: {p}')
 return out


def f(x):
 try:
  z=float(x); return z if math.isfinite(z) else None
 except Exception:return None


def pos(r):
 z=f(r.get('semantic_best_common_viability')); return z is not None and z>0


def passed(r): return bool(r.get('absolute_feasibility_pass',False))
def feasible(r):
 z=f(r.get('teacher_candidate_r_dep')); return z is not None and z>=0


def safe(r,g=.015):
 z=f(r.get('teacher_adv')); return z is not None and z>=g and not bool(r.get('teacher_harmful',False))


def ratio(a,b,key):
 aa=f(a.get(key)); bb=f(b.get(key))
 if aa is None or bb is None or abs(aa)<1e-12:return None
 return bb/aa


def mean(z):
 z=[x for x in z if x is not None and math.isfinite(x)]
 return float(np.mean(z)) if z else None


def summarize(t,d):
 t_keys=set(t); d_keys=set(d); common=t_keys & d_keys
 keys=sorted(common,key=repr)
 missing_in_d=t_keys-d_keys; missing_in_t=d_keys-t_keys
 row_set_equal=(t_keys==d_keys and bool(keys))
 # Certificate identity is a causal contract, so it is valid only under exact
 # row alignment.  Never allow an intersection-only/vacuous PASS.
 cert_equal=row_set_equal and all(pos(t[k])==pos(d[k]) for k in keys)
 transitioned=[]; lost=[]
 for k in keys:
  if (not passed(t[k])) and passed(d[k]): transitioned.append(d[k])
  if passed(t[k]) and (not passed(d[k])): lost.append(t[k])
 groups={
  'safe_positive':[k for k in keys if safe(d[k])],
  'harmful':[k for k in keys if bool(d[k].get('teacher_harmful',False))],
  'teacher_feasible':[k for k in keys if feasible(d[k])],
  'teacher_infeasible':[k for k in keys if not feasible(d[k])],
 }
 out={
  'T_rows_total':len(t),'D_rows_total':len(d),'aligned_rows':len(keys),
  'row_set_equal':row_set_equal,
  'missing_in_D_count':len(missing_in_d),'missing_in_T_count':len(missing_in_t),
  'missing_in_D_examples':[repr(x) for x in sorted(missing_in_d,key=repr)[:5]],
  'missing_in_T_examples':[repr(x) for x in sorted(missing_in_t,key=repr)[:5]],
  'positive_certificate_set_equal':cert_equal,
  'T_positive_certificate_rows':sum(pos(t[k]) for k in keys),
  'D_positive_certificate_rows':sum(pos(d[k]) for k in keys),
  'newly_passed':{
   'rows':len(transitioned),
   'teacher_feasible_precision':sum(feasible(r) for r in transitioned)/len(transitioned) if transitioned else None,
   'safe_positive_rows':sum(safe(r) for r in transitioned),
   'harmful_rows':sum(bool(r.get('teacher_harmful',False)) for r in transitioned),
  },
  'lost_pass':{'rows':len(lost),'teacher_feasible_precision':sum(feasible(r) for r in lost)/len(lost) if lost else None},
  'support_relaxation':{},
 }
 for name,ks in groups.items():
  out['support_relaxation'][name]={
   'rows':len(ks),
   'T_support_mean':mean([f(t[k].get('semantic_max_common_support')) for k in ks]),
   'D_support_mean':mean([f(d[k].get('semantic_max_common_support')) for k in ks]),
   'D_over_T_support_mean':mean([ratio(t[k],d[k],'semantic_max_common_support') for k in ks]),
  }
 return out


def main():
 ap=argparse.ArgumentParser(description='v48.69 read-only demand-tempered trust audit')
 ap.add_argument('--t68',type=Path,required=True); ap.add_argument('--d69',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
 doc={'schema':'ocrap-v48.69-demand-trust-audit-v2','test_roots_read':False,'dataset_reconstruction':False,'comparisons':{}}
 valid=True
 for v in VARIANTS:
  doc['comparisons'][v]={}
  for k in KINDS:
   s=summarize(rows(a.t68,v,k),rows(a.d69,v,k))
   doc['comparisons'][v][k]=s
   valid = valid and bool(s['row_set_equal'])
 doc['valid']=bool(valid)
 a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'event':'v48_69_demand_trust_audit','valid':doc['valid'],'output':str(a.output)}))
 return 0 if doc['valid'] else 30
if __name__=='__main__': raise SystemExit(main())
