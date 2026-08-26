#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
KINDS={'dev_near':'dev_diagnostic_near_v48.proposal_rows.jsonl','dev_contact':'dev_diagnostic_contact_v48.proposal_rows.jsonl','certificate_near':'direct_value_risk_near_v48.proposal_rows.jsonl','certificate_contact':'direct_value_risk_contact_v48.proposal_rows.jsonl'};VARIANTS=('balanced','precision')
def read(run,v,k):
 p=run/'candidates'/v/'calibration'/KINDS[k];return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.is_file() else []
def auc(y,s):
 y=np.asarray(y,dtype=bool);s=np.asarray(s,dtype=float);p=s[y];n=s[~y]
 if not len(p) or not len(n):return None
 return float(((p[:,None]>n[None,:]).sum()+.5*(p[:,None]==n[None,:]).sum())/(len(p)*len(n)))
def f(r,k):return float(r[k])
def summ(rows,tol=1e-8):
 if not rows:return {'rows':0}
 floor=[r for r in rows if f(r,'teacher_candidate_r_dep')>=0 and abs(f(r,'teacher_candidate_r_dep')-.5)<=tol]
 feasible=[r for r in rows if f(r,'teacher_candidate_r_dep')>=0];above=[r for r in feasible if r not in floor]
 safe=[r for r in rows if float(r.get('teacher_adv',-1e9))>=.015 and not bool(r.get('teacher_harmful',False))]
 safe_floor=[r for r in safe if f(r,'teacher_candidate_r_dep')>=0 and abs(f(r,'teacher_candidate_r_dep')-.5)<=tol]
 def passfrac(z):return float(np.mean([f(r,'absolute_feasibility_probability')>=.5 for r in z])) if z else None
 def pmean(z):return float(np.mean([f(r,'absolute_feasibility_probability') for r in z])) if z else None
 full_y=[f(r,'teacher_candidate_r_dep')>=0 for r in rows];full_s=[f(r,'absolute_feasibility_probability') for r in rows]
 nofloor=[r for r in rows if r not in floor];ny=[f(r,'teacher_candidate_r_dep')>=0 for r in nofloor];ns=[f(r,'absolute_feasibility_probability') for r in nofloor]
 return {'rows':len(rows),'teacher_feasible_rows':len(feasible),'teacher_feasible_exact_0p5_rows':len(floor),'teacher_feasible_exact_0p5_fraction':len(floor)/len(feasible) if feasible else None,'teacher_feasible_exact_0p5_pass_fraction':passfrac(floor),'teacher_feasible_exact_0p5_probability_mean':pmean(floor),'teacher_feasible_above_floor_rows':len(above),'teacher_feasible_above_floor_pass_fraction':passfrac(above),'teacher_feasible_above_floor_probability_mean':pmean(above),'safe_positive_rows':len(safe),'safe_positive_exact_0p5_rows':len(safe_floor),'safe_positive_exact_0p5_fraction':len(safe_floor)/len(safe) if safe else None,'safe_positive_exact_0p5_pass_fraction':passfrac(safe_floor),'absolute_feasibility_auc_full':auc(full_y,full_s),'absolute_feasibility_auc_excluding_exact_0p5_feasible':auc(ny,ns)}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--arm',action='append',required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();arms={}
 for spec in a.arm:
  name,raw=spec.split('=',1);run=Path(raw);arms[name]={v:{k:summ(read(run,v,k)) for k in KINDS} for v in VARIANTS}
 doc={'schema':'ocrap-v48.71-truth-floor-strata-audit-v1','arms':arms,'floor_value':.5,'floor_tolerance':1e-8,'interpretation':'read-only diagnostic; never changes labels/datasets. A large gap between full AUC and AUC excluding exact-0.5 teacher-feasible rows indicates structural-floor truth-contract debt may dominate observed source errors.','test_roots_read':False,'dataset_reconstruction':False}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');print(json.dumps({'event':'v48_71_truth_floor_strata_audit','output':str(a.output)}))
if __name__=='__main__':main()
