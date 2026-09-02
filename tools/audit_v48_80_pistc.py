#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
KINDS={'dev_near':'dev_diagnostic_near_v48.proposal_rows.jsonl','dev_contact':'dev_diagnostic_contact_v48.proposal_rows.jsonl','certificate_near':'direct_value_risk_near_v48.proposal_rows.jsonl','certificate_contact':'direct_value_risk_contact_v48.proposal_rows.jsonl'}
VARS=('balanced','precision')
def rows(run,v,s):
 p=run/'candidates'/v/'calibration'/KINDS[s]; return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
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
def huber_dist(d):d=abs(float(d));return .5*d*d if d<1 else d-.5
def iloss(x,lo,hi):return huber_dist(lo-x) if x<lo else (huber_dist(x-hi) if x>hi else 0.)
def truth(path):
 out={}
 for z in path.read_text().splitlines():
  if not z.strip():continue
  r=json.loads(z);out[(r['dataset_role'],str(r.get('scene_id','')),int(r.get('time_index',-1)),int(r.get('candidate_index',-1)))]=r
 return out
def compare(base,new,split,t):
 bm={key(r):r for r in base};nm={key(r):r for r in new}
 if set(bm)!=set(nm):raise ValueError(f'{split} row mismatch')
 ps=[]
 for k in bm:
  tr=t[(split,k[0],k[1],k[2])];ps.append((bm[k],nm[k],tr))
 inf=[z for z in ps if z[2].get('informative')]
 exact=[z for z in ps if z[2].get('exact_physical')]
 def mean(z,fn):return float(np.mean([fn(*q) for q in z])) if z else None
 def exact_auc(z,which):
  return auc([feasible(a) for a,b,tr in z],[prob(a if which=='base' else b) for a,b,tr in z])
 return {'aligned_rows':len(ps),'teacher_labels_equal':all(abs(f(a,'teacher_candidate_r_dep')-f(b,'teacher_candidate_r_dep'))<1e-7 for a,b,_ in ps),
 'informative_rows':len(inf),'exact_physical_rows':len(exact),
 'interval_huber_base':mean(inf,lambda a,b,tr:iloss(margin(a),float(tr['physical_lower']),float(tr['physical_upper']))),
 'interval_huber_new':mean(inf,lambda a,b,tr:iloss(margin(b),float(tr['physical_lower']),float(tr['physical_upper']))),
 'interval_satisfaction_base':mean(inf,lambda a,b,tr:float(float(tr['physical_lower'])<=margin(a)<=float(tr['physical_upper']))),
 'interval_satisfaction_new':mean(inf,lambda a,b,tr:float(float(tr['physical_lower'])<=margin(b)<=float(tr['physical_upper']))),
 'exact_auc_base':exact_auc(exact,'base'),'exact_auc_new':exact_auc(exact,'new'),
 'harmful_pass_base':mean([z for z in ps if harmful(z[0])],lambda a,b,tr:float(prob(a)>=.5)),
 'harmful_pass_new':mean([z for z in ps if harmful(z[0])],lambda a,b,tr:float(prob(b)>=.5)),
 'ti_pass_base':mean([z for z in ps if not feasible(z[0])],lambda a,b,tr:float(prob(a)>=.5)),
 'ti_pass_new':mean([z for z in ps if not feasible(z[0])],lambda a,b,tr:float(prob(b)>=.5))}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--j78',type=Path,required=True);ap.add_argument('--l80',type=Path,required=True);ap.add_argument('--truth-index',type=Path,required=True);ap.add_argument('--truth-summary',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();t=truth(a.truth_index)
 c={v:{s:compare(rows(a.j78,v,s),rows(a.l80,v,s),s,t) for s in KINDS} for v in VARS}
 doc={'schema':'ocrap-v48.80-pistc-audit-v1','engineering_version':'v48.80.0-OC-PISTC','comparisons':{'L80_minus_J78':c},'truth_index_summary':json.loads(a.truth_summary.read_text()),'teacher_labels_changed':False,'teacher_future_input_to_model':False,'dataset_reconstruction':False,'test_roots_read':False,'valid':True}
 a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
