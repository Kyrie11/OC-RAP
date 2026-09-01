#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np

KINDS={'dev_near':'dev_diagnostic_near_v48.proposal_rows.jsonl','dev_contact':'dev_diagnostic_contact_v48.proposal_rows.jsonl','certificate_near':'direct_value_risk_near_v48.proposal_rows.jsonl','certificate_contact':'direct_value_risk_contact_v48.proposal_rows.jsonl'}
VARIANTS=('balanced','precision'); FLOOR=.5; TOL=1e-8

def read(run:Path,v:str,k:str):
 p=run/'candidates'/v/'calibration'/KINDS[k]
 if not p.is_file():raise FileNotFoundError(p)
 return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def f(r,k,default=float('nan')):
 try:return float(r.get(k,default))
 except Exception:return default
def key(r):return (str(r.get('scene','')),int(r.get('time',-1)),int(r.get('candidate',-1)))
def feasible(r):return f(r,'teacher_candidate_r_dep')>=0
def floor(r):return abs(f(r,'teacher_candidate_r_dep')-FLOOR)<=TOL and feasible(r)
def safe(r):return f(r,'teacher_adv',-1e9)>=.015 and not bool(r.get('teacher_harmful',False))
def harmful(r):return bool(r.get('teacher_harmful',False))
def poscert(r):return f(r,'semantic_best_common_viability',-1e9)>0
def prob(r):return min(1-1e-12,max(1e-12,f(r,'absolute_feasibility_probability')))
def margin(r):
 p=prob(r);return math.log(p/(1-p))
def huber(a,b):
 d=abs(float(a)-float(b));return .5*d*d if d<1. else d-.5
def auc(y,s):
 y=np.asarray(y,dtype=bool);s=np.asarray(s,dtype=float);ok=np.isfinite(s);y=y[ok];s=s[ok];p=s[y];n=s[~y]
 if not len(p) or not len(n):return None
 return float(((p[:,None]>n[None,:]).sum()+.5*(p[:,None]==n[None,:]).sum())/(len(p)*len(n)))
def mean(z,fun):return float(np.mean([fun(r) for r in z])) if z else None

def summarize(rows):
 nf=[r for r in rows if not floor(r)]; safe_rows=[r for r in rows if safe(r)]; safe_nf=[r for r in safe_rows if not floor(r)]; harm=[r for r in rows if harmful(r)];ti=[r for r in rows if not feasible(r)]
 return {'rows':len(rows),'nonfloor_rows':len(nf),'nonfloor_source_auc':auc([feasible(r) for r in nf],[prob(r) for r in nf]),'nonfloor_signed_margin_huber':mean(nf,lambda r:huber(margin(r),f(r,'teacher_candidate_r_dep'))),'nonfloor_signed_margin_mae':mean(nf,lambda r:abs(margin(r)-f(r,'teacher_candidate_r_dep'))),'full_source_auc':auc([feasible(r) for r in rows],[prob(r) for r in rows]),'safe_positive_rows':len(safe_rows),'safe_positive_nonfloor_rows':len(safe_nf),'safe_positive_pass_fraction':mean(safe_rows,lambda r:prob(r)>=.5),'harmful_pass_fraction':mean(harm,lambda r:prob(r)>=.5),'teacher_infeasible_pass_fraction':mean(ti,lambda r:prob(r)>=.5),'positive_certificate_rows':sum(poscert(r) for r in rows)}

def compare(base,new):
 bm={key(r):r for r in base};nm={key(r):r for r in new};common=sorted(set(bm)&set(nm));pairs=[(bm[k],nm[k]) for k in common];missing=len(set(bm)-set(nm));extra=len(set(nm)-set(bm))
 labels=all(abs(f(a,'teacher_candidate_r_dep')-f(b,'teacher_candidate_r_dep'))<=1e-7 and bool(a.get('teacher_harmful',False))==bool(b.get('teacher_harmful',False)) for a,b in pairs);cert=all(poscert(a)==poscert(b) for a,b in pairs)
 bn=[a for a,b in pairs if not floor(a)];nn=[b for a,b in pairs if not floor(a)]
 ba=auc([feasible(r) for r in bn],[prob(r) for r in bn]);na=auc([feasible(r) for r in nn],[prob(r) for r in nn]);bh=mean(bn,lambda r:huber(margin(r),f(r,'teacher_candidate_r_dep')));nh=mean(nn,lambda r:huber(margin(r),f(r,'teacher_candidate_r_dep')))
 bf=auc([feasible(a) for a,b in pairs],[prob(a) for a,b in pairs]);nf=auc([feasible(a) for a,b in pairs],[prob(b) for a,b in pairs])
 return {'aligned_rows':len(pairs),'missing_rows':missing,'extra_rows':extra,'teacher_labels_equal':labels,'positive_certificate_set_equal':cert,'nonfloor_auc_base':ba,'nonfloor_auc_new':na,'nonfloor_auc_delta':None if ba is None or na is None else na-ba,'nonfloor_huber_base':bh,'nonfloor_huber_new':nh,'nonfloor_huber_delta':None if bh is None or nh is None else nh-bh,'full_auc_base':bf,'full_auc_new':nf,'full_auc_delta':None if bf is None or nf is None else nf-bf}

def state(run:Path,v:str,new:bool):
 p=run/'candidates'/v/('V48_76_STAGE_I_STATE_ISOLATION.json' if new else 'V48_75_STAGE_I_STATE_ISOLATION.json')
 if not p.is_file():return None
 d=json.loads(p.read_text());return {'valid':d.get('valid'),'effective_gain':d.get('effective_clamped_semantic_witness_gain'),'raw_gain':d.get('raw_semantic_witness_gain'),'best_checkpoint':d.get('adapted')}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--c75',type=Path,required=True);ap.add_argument('--d75',type=Path,required=True);ap.add_argument('--e76',type=Path,required=True);ap.add_argument('--f76',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 runs={'C75_SIGN_PROJ':(a.c75,False),'D75_SIGN_FIDELITY':(a.d75,False),'E76_MARGIN_PROJ':(a.e76,True),'F76_MARGIN_FIDELITY':(a.f76,True)}
 arms={name:{v:{'state':state(run,v,isnew),'splits':{k:summarize(read(run,v,k)) for k in KINDS}} for v in VARIANTS} for name,(run,isnew) in runs.items()}
 specs={'E76_minus_C75':(a.c75,a.e76),'F76_minus_D75':(a.d75,a.f76),'F76_minus_E76':(a.e76,a.f76),'D75_minus_C75':(a.c75,a.d75)}
 comps={name:{v:{k:compare(read(base,v,k),read(new,v,k)) for k in KINDS} for v in VARIANTS} for name,(base,new) in specs.items()}
 doc={'schema':'ocrap-v48.76-icsm-signed-margin-audit-v1','arms':arms,'comparisons':comps,'truth_contract':'censor_exact_0p5','historical_objective':'binary_sign','new_objective':'signed_margin_huber','huber_beta':1.0,'factorial':'C75/D75 binary-sign censored baselines vs E76/F76 signed-margin censored, with projection fidelity as the orthogonal historical factor','teacher_labels_changed':False,'dataset_reconstruction':False,'test_roots_read':False}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');print(json.dumps({'event':'v48_76_signed_margin_audit','output':str(a.output)}))
if __name__=='__main__':main()
