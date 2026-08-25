#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
KINDS={'dev_near':'dev_diagnostic_near_v48.proposal_rows.jsonl','dev_contact':'dev_diagnostic_contact_v48.proposal_rows.jsonl','certificate_near':'direct_value_risk_near_v48.proposal_rows.jsonl','certificate_contact':'direct_value_risk_contact_v48.proposal_rows.jsonl'}
VARIANTS=('balanced','precision')
def rows(run,v,k):
 p=run/'candidates'/v/'calibration'/KINDS[k];out={}
 if not p.is_file():return out
 for line in p.read_text().splitlines():
  if not line.strip():continue
  r=json.loads(line);key=(r.get('scene'),r.get('time'),r.get('candidate'))
  if key in out:raise RuntimeError(f'duplicate proposal key {key} in {p}')
  out[key]=r
 return out
def f(r,k,default=None):
 try:
  x=float(r.get(k));return x if math.isfinite(x) else default
 except Exception:return default
def pos(r):return (f(r,'semantic_best_common_viability',-1e9) or -1e9)>0
def feas(r):return (f(r,'teacher_candidate_r_dep',-1e9) or -1e9)>=0
def safe(r,g=.015):return (f(r,'teacher_adv',-1e9) or -1e9)>=g and not bool(r.get('teacher_harmful',False))
def passed(r):return (f(r,'absolute_feasibility_probability',0.0) or 0.0)>=.5
def mean(z):return float(np.mean(z)) if z else None
def median(z):return float(np.median(z)) if z else None
def summarize(base,new):
 kb,kn=set(base),set(new);keys=sorted(kb&kn,key=str);missing=sorted(kb-kn,key=str);extra=sorted(kn-kb,key=str)
 cert_equal=all(pos(base[k])==pos(new[k]) for k in keys) and not missing and not extra
 cats={'safe_positive':lambda r:safe(r),'harmful':lambda r:bool(r.get('teacher_harmful',False)),'teacher_feasible':feas,'teacher_infeasible':lambda r:not feas(r)}
 out={'aligned_rows':len(keys),'missing_rows':len(missing),'extra_rows':len(extra),'positive_certificate_set_equal':cert_equal,'positive_certificate_base':sum(pos(base[k]) for k in keys),'positive_certificate_new':sum(pos(new[k]) for k in keys),'newly_passed':0,'lost_pass':0,'categories':{}}
 for k in keys:
  out['newly_passed']+=int((not passed(base[k])) and passed(new[k]));out['lost_pass']+=int(passed(base[k]) and (not passed(new[k])))
 for name,pred in cats.items():
  ratios=[];dprob=[];bs=[];ns=[]
  for k in keys:
   if not pred(base[k]):continue
   b=f(base[k],'semantic_max_common_support');n=f(new[k],'semantic_max_common_support')
   if b is not None:bs.append(b)
   if n is not None:ns.append(n)
   if b is not None and n is not None and b>1e-12:ratios.append(n/b)
   pb=f(base[k],'absolute_feasibility_probability');pn=f(new[k],'absolute_feasibility_probability')
   if pb is not None and pn is not None:dprob.append(pn-pb)
  out['categories'][name]={'rows':sum(pred(base[k]) for k in keys),'support_ratio_mean':mean(ratios),'support_ratio_median':median(ratios),'base_support_mean':mean(bs),'new_support_mean':mean(ns),'probability_delta_mean':mean(dprob)}
 tf=out['categories']['teacher_feasible']['support_ratio_mean'];ti=out['categories']['teacher_infeasible']['support_ratio_mean'];sp=out['categories']['safe_positive']['support_ratio_mean'];hm=out['categories']['harmful']['support_ratio_mean']
 out['selective_contraction']={'teacher_feasible_ratio_gt_teacher_infeasible':(tf is not None and ti is not None and tf>ti),'safe_positive_ratio_gt_harmful':(sp is not None and hm is not None and sp>hm),'teacher_feasible_minus_infeasible':(None if tf is None or ti is None else tf-ti),'safe_positive_minus_harmful':(None if sp is None or hm is None else sp-hm)}
 return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--t68',type=Path,required=True);ap.add_argument('--d69',type=Path,required=True);ap.add_argument('--e70',type=Path,required=True);ap.add_argument('--g70',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();doc={'schema':'ocrap-v48.70-dotw-soft-occupancy-trust-audit-v1','test_roots_read':False,'dataset_reconstruction':False,'comparisons':{}}
 for v in VARIANTS:
  doc['comparisons'][v]={}
  for k in KINDS:
   t=rows(a.t68,v,k);d=rows(a.d69,v,k);e=rows(a.e70,v,k);g=rows(a.g70,v,k)
   doc['comparisons'][v][k]={'E70_minus_T68':summarize(t,e),'G70_minus_D69':summarize(d,g),'D69_minus_T68':summarize(t,d),'G70_minus_E70':summarize(e,g)}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');print(json.dumps({'event':'v48_70_soft_occupancy_trust_audit','output':str(a.output)}))
if __name__=='__main__':main()
