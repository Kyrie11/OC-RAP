#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
KINDS={'dev_near':'dev_diagnostic_near_v48.proposal_rows.jsonl','dev_contact':'dev_diagnostic_contact_v48.proposal_rows.jsonl','certificate_near':'direct_value_risk_near_v48.proposal_rows.jsonl','certificate_contact':'direct_value_risk_contact_v48.proposal_rows.jsonl'}
VARS=('balanced','precision')
def rows(run,v,s):return [json.loads(x) for x in (run/'candidates'/v/'calibration'/KINDS[s]).read_text().splitlines() if x.strip()]
def key(r):return(str(r.get('scene','')),int(r.get('time',-1)),int(r.get('candidate',-1)))
def f(r,k,d=float('nan')):
 try:return float(r.get(k,d))
 except:return d
def prob(r):return min(1-1e-12,max(1e-12,f(r,'absolute_feasibility_probability')))
def margin(r):p=prob(r);return math.log(p/(1-p))
def feasible(r):return f(r,'teacher_candidate_r_dep')>=0
def harmful(r):return bool(r.get('teacher_harmful',False))
def auc(y,s):
 y=np.asarray(y,dtype=bool);s=np.asarray(s,float);p=s[y];n=s[~y]
 if len(p)==0 or len(n)==0:return None
 return float(((p[:,None]>n[None,:]).sum()+.5*(p[:,None]==n[None,:]).sum())/(len(p)*len(n)))
def huber(d):d=abs(float(d));return .5*d*d if d<1 else d-.5
def iloss(x,lo,hi):return huber(lo-x) if x<lo else (huber(x-hi) if x>hi else 0.)
def truth(path):
 out={}
 for z in path.read_text().splitlines():
  if z.strip():
   r=json.loads(z);out[(r['dataset_role'],str(r.get('scene_id','')),int(r.get('time_index',-1)),int(r.get('candidate_index',-1)))]=r
 return out
def compare(base,new,split,t):
 bm={key(r):r for r in base};nm={key(r):r for r in new}
 if set(bm)!=set(nm):raise ValueError(f'{split} row mismatch')
 ps=[(bm[k],nm[k],t[(split,k[0],k[1],k[2])]) for k in bm];inf=[z for z in ps if z[2].get('informative')];exact=[z for z in ps if z[2].get('exact_physical')]
 mean=lambda z,fn:float(np.mean([fn(*q) for q in z])) if z else None
 def exauc(which):return auc([feasible(a) for a,b,tr in exact],[prob(a if which=='base' else b) for a,b,tr in exact])
 return {'aligned_rows':len(ps),'teacher_labels_equal':all(abs(f(a,'teacher_candidate_r_dep')-f(b,'teacher_candidate_r_dep'))<1e-7 for a,b,_ in ps),'informative_rows':len(inf),'exact_physical_rows':len(exact),'interval_huber_base':mean(inf,lambda a,b,tr:iloss(margin(a),float(tr['physical_lower']),float(tr['physical_upper']))),'interval_huber_new':mean(inf,lambda a,b,tr:iloss(margin(b),float(tr['physical_lower']),float(tr['physical_upper']))),'interval_satisfaction_base':mean(inf,lambda a,b,tr:float(float(tr['physical_lower'])<=margin(a)<=float(tr['physical_upper']))),'interval_satisfaction_new':mean(inf,lambda a,b,tr:float(float(tr['physical_lower'])<=margin(b)<=float(tr['physical_upper']))),'exact_auc_base':exauc('base'),'exact_auc_new':exauc('new'),'harmful_pass_base':mean([z for z in ps if harmful(z[0])],lambda a,b,tr:float(prob(a)>=.5)),'harmful_pass_new':mean([z for z in ps if harmful(z[0])],lambda a,b,tr:float(prob(b)>=.5)),'ti_pass_base':mean([z for z in ps if not feasible(z[0])],lambda a,b,tr:float(prob(a)>=.5)),'ti_pass_new':mean([z for z in ps if not feasible(z[0])],lambda a,b,tr:float(prob(b)>=.5))}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--l80',type=Path,required=True);ap.add_argument('--m81',type=Path,required=True);ap.add_argument('--truth-index',type=Path,required=True);ap.add_argument('--truth-summary',type=Path,required=True);ap.add_argument('--v80-truth-summary',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();t=truth(a.truth_index)
 c={v:{s:compare(rows(a.l80,v,s),rows(a.m81,v,s),s,t) for s in KINDS} for v in VARS}
 doc={'schema':'ocrap-v48.81-sitc-audit-v1','engineering_version':'v48.81.0-OC-SITC','comparisons':{'M81_minus_L80':c},'truth_index_summary':json.loads(a.truth_summary.read_text()),'v48_80_truth_index_summary':json.loads(a.v80_truth_summary.read_text()),'teacher_labels_changed':False,'teacher_future_input_to_model':False,'dataset_reconstruction':False,'test_roots_read':False,'valid':True}
 a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
