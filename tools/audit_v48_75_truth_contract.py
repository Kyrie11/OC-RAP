#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np

KINDS={'dev_near':'dev_diagnostic_near_v48.proposal_rows.jsonl','dev_contact':'dev_diagnostic_contact_v48.proposal_rows.jsonl','certificate_near':'direct_value_risk_near_v48.proposal_rows.jsonl','certificate_contact':'direct_value_risk_contact_v48.proposal_rows.jsonl'}
VARIANTS=('balanced','precision'); FLOOR=0.5; TOL=1e-8

def read(run:Path,v:str,k:str):
 p=run/'candidates'/v/'calibration'/KINDS[k]
 if not p.is_file(): raise FileNotFoundError(p)
 return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def f(r,k,default=float('nan')):
 try:return float(r.get(k,default))
 except Exception:return default
def key(r):return (str(r.get('scene','')),int(r.get('time',-1)),int(r.get('candidate',-1)))
def auc(y,s):
 y=np.asarray(y,dtype=bool);s=np.asarray(s,dtype=float);ok=np.isfinite(s);y=y[ok];s=s[ok];p=s[y];n=s[~y]
 if not len(p) or not len(n):return None
 return float(((p[:,None]>n[None,:]).sum()+.5*(p[:,None]==n[None,:]).sum())/(len(p)*len(n)))
def is_floor_feasible(r):return f(r,'teacher_candidate_r_dep')>=0 and abs(f(r,'teacher_candidate_r_dep')-FLOOR)<=TOL
def is_feasible(r):return f(r,'teacher_candidate_r_dep')>=0
def is_safe(r):return f(r,'teacher_adv',-1e9)>=.015 and not bool(r.get('teacher_harmful',False))
def is_harm(r):return bool(r.get('teacher_harmful',False))
def is_poscert(r):return f(r,'semantic_best_common_viability',-1e9)>0
def prob(r):return f(r,'absolute_feasibility_probability')
def pass_(r):return prob(r)>=.5

def summarize(rows):
 feasible=[r for r in rows if is_feasible(r)]; floor=[r for r in feasible if is_floor_feasible(r)]; nonfloor=[r for r in rows if not is_floor_feasible(r)]
 safe=[r for r in rows if is_safe(r)]; safe_floor=[r for r in safe if is_floor_feasible(r)]; safe_non=[r for r in safe if not is_floor_feasible(r)]
 harmful=[r for r in rows if is_harm(r)]; ti=[r for r in rows if not is_feasible(r)]
 def mean(z,fun):return float(np.mean([fun(r) for r in z])) if z else None
 def count(z,fun):return int(sum(bool(fun(r)) for r in z))
 full_auc=auc([is_feasible(r) for r in rows],[prob(r) for r in rows]); nf_auc=auc([is_feasible(r) for r in nonfloor],[prob(r) for r in nonfloor])
 return {'rows':len(rows),'teacher_feasible_rows':len(feasible),'teacher_feasible_floor_rows':len(floor),'teacher_feasible_floor_fraction':len(floor)/len(feasible) if feasible else None,
  'full_source_auc':full_auc,'nonfloor_source_auc':nf_auc,
  'safe_positive_rows':len(safe),'safe_positive_floor_rows':len(safe_floor),'safe_positive_floor_fraction':len(safe_floor)/len(safe) if safe else None,
  'safe_positive_nonfloor_rows':len(safe_non),'safe_positive_positive_certificate_rows':count(safe,is_poscert),'safe_positive_positive_certificate_fraction':mean(safe,is_poscert),
  'safe_positive_pass_fraction':mean(safe,pass_),'safe_positive_nonfloor_pass_fraction':mean(safe_non,pass_),
  'harmful_rows':len(harmful),'harmful_floor_rows':count(harmful,is_floor_feasible),'harmful_pass_fraction':mean(harmful,pass_),
  'teacher_infeasible_rows':len(ti),'teacher_infeasible_pass_fraction':mean(ti,pass_),
  'teacher_feasible_probability_mean':mean(feasible,prob),'teacher_floor_probability_mean':mean(floor,prob),'teacher_nonfloor_feasible_probability_mean':mean([r for r in feasible if not is_floor_feasible(r)],prob)}

def compare(base,new):
 bm={key(r):r for r in base};nm={key(r):r for r in new};common=sorted(set(bm)&set(nm));missing=len(set(bm)-set(nm));extra=len(set(nm)-set(bm)); pairs=[(bm[k],nm[k]) for k in common]
 labels_ok=all(abs(f(a,'teacher_candidate_r_dep')-f(b,'teacher_candidate_r_dep'))<=1e-7 and bool(a.get('teacher_harmful',False))==bool(b.get('teacher_harmful',False)) for a,b in pairs)
 cert_equal=all(is_poscert(a)==is_poscert(b) for a,b in pairs)
 base_nf=[a for a,b in pairs if not is_floor_feasible(a)]; new_nf=[b for a,b in pairs if not is_floor_feasible(a)]
 b_auc=auc([is_feasible(r) for r in base_nf],[prob(r) for r in base_nf]);n_auc=auc([is_feasible(r) for r in new_nf],[prob(r) for r in new_nf])
 b_full=auc([is_feasible(a) for a,b in pairs],[prob(a) for a,b in pairs]);n_full=auc([is_feasible(a) for a,b in pairs],[prob(b) for a,b in pairs])
 def delta_mean(pred):
  z=[prob(b)-prob(a) for a,b in pairs if pred(a)]
  return float(np.mean(z)) if z else None
 return {'aligned_rows':len(pairs),'missing_rows':missing,'extra_rows':extra,'teacher_labels_equal':labels_ok,'positive_certificate_set_equal':cert_equal,
  'nonfloor_auc_base':b_auc,'nonfloor_auc_new':n_auc,'nonfloor_auc_delta':None if b_auc is None or n_auc is None else n_auc-b_auc,
  'full_auc_base':b_full,'full_auc_new':n_full,'full_auc_delta':None if b_full is None or n_full is None else n_full-b_full,
  'probability_delta_teacher_floor':delta_mean(is_floor_feasible),'probability_delta_teacher_nonfloor_feasible':delta_mean(lambda r:is_feasible(r) and not is_floor_feasible(r)),
  'probability_delta_teacher_infeasible':delta_mean(lambda r:not is_feasible(r)),'probability_delta_safe_positive':delta_mean(is_safe),'probability_delta_harmful':delta_mean(is_harm)}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--q67',type=Path,required=True);ap.add_argument('--t68',type=Path,required=True);ap.add_argument('--c75',type=Path,required=True);ap.add_argument('--d75',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 runs={'Q67_CTRLPROJ':a.q67,'T68_FIDELITY':a.t68,'C75_PROJ_CENSORED':a.c75,'D75_FIDELITY_CENSORED':a.d75}
 arms={name:{v:{k:summarize(read(run,v,k)) for k in KINDS} for v in VARIANTS} for name,run in runs.items()}
 comparisons={}
 specs={'C75_minus_Q67':(a.q67,a.c75),'D75_minus_T68':(a.t68,a.d75),'D75_minus_C75':(a.c75,a.d75),'T68_minus_Q67':(a.q67,a.t68)}
 for name,(base,new) in specs.items(): comparisons[name]={v:{k:compare(read(base,v,k),read(new,v,k)) for k in KINDS} for v in VARIANTS}
 doc={'schema':'ocrap-v48.75-stca-truth-contract-audit-v1','arms':arms,'comparisons':comparisons,'floor_value':FLOOR,'floor_tolerance':TOL,
  'supervision_intervention':'exact R_dep*=0.5 rows are censored only from absolute-feasibility BCE/model selection; no relabel, no dataset rewrite, no teacher file modification',
  'factorial':'historical Q67 projection/full vs T68 projection+fidelity/full; new C75 projection/censored vs D75 projection+fidelity/censored',
  'dataset_reconstruction':False,'teacher_labels_changed':False,'test_roots_read':False}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');print(json.dumps({'event':'v48_75_truth_contract_audit','output':str(a.output)}))
if __name__=='__main__':main()
