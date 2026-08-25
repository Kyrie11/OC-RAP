#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math
from pathlib import Path
KINDS={'dev_near':'dev_diagnostic_near_v48.proposal_rows.jsonl','dev_contact':'dev_diagnostic_contact_v48.proposal_rows.jsonl','certificate_near':'direct_value_risk_near_v48.proposal_rows.jsonl','certificate_contact':'direct_value_risk_contact_v48.proposal_rows.jsonl'}
VARIANTS=('balanced','precision')
def read(run,v,k):
 p=run/'candidates'/v/'calibration'/KINDS[k]
 if not p.is_file(): raise FileNotFoundError(f'missing calibration proposal rows: {p}')
 out=[]
 for l in p.read_text().splitlines():
  if l.strip():out.append(json.loads(l))
 if not out: raise ValueError(f'empty calibration proposal rows: {p}')
 return out
def f(x):
 try:
  z=float(x);return z if math.isfinite(z) else None
 except Exception:return None
def safe(r):
 z=f(r.get('teacher_adv'));return z is not None and z>=.015 and not bool(r.get('teacher_harmful',False))
def feasible(r):
 z=f(r.get('teacher_candidate_r_dep'));return z is not None and z>=0
def floor(r):
 z=f(r.get('teacher_candidate_r_dep'));return z is not None and abs(z-.5)<=1e-7
def pos(r):
 z=f(r.get('semantic_best_common_viability'));return z is not None and z>0
def summ(rows):
 def g(z):
  return {'rows':len(z),'certificate_fraction':sum(pos(r) for r in z)/len(z) if z else None,'pass_fraction':sum(bool(r.get('absolute_feasibility_pass',False)) for r in z)/len(z) if z else None}
 sp=[r for r in rows if safe(r)]
 # Important: 0.0 is a valid teacher-feasible boundary value.  Avoid boolean
 # fallback such as `(value or -1)`, which would silently turn 0.0 into -1.
 tf=[r for r in rows if feasible(r)]
 return {'safe_positive':{'rows':len(sp),'floor_fraction':sum(floor(r) for r in sp)/len(sp) if sp else None,'floor':g([r for r in sp if floor(r)]),'nonfloor':g([r for r in sp if not floor(r)])},'teacher_feasible':{'rows':len(tf),'floor_fraction':sum(floor(r) for r in tf)/len(tf) if tf else None,'floor':g([r for r in tf if floor(r)]),'nonfloor':g([r for r in tf if not floor(r)])}}
def main():
 ap=argparse.ArgumentParser(description='v48.69 read-only teacher structural-floor debt audit');ap.add_argument('--run',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();doc={'schema':'ocrap-v48.69-truth-debt-audit-v2','test_roots_read':False,'dataset_reconstruction':False,'arms':{'D69_DTRW':{}}}
 for v in VARIANTS:doc['arms']['D69_DTRW'][v]={k:summ(read(a.run,v,k)) for k in KINDS}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');print(json.dumps({'event':'v48_69_truth_debt_audit','output':str(a.output)}))
if __name__=='__main__':main()
