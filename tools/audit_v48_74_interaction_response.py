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
def feas(r):
 v=f(r,'teacher_candidate_r_dep',None);return v is not None and v>=0
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
   bv=f(base[k],'semantic_max_common_support');nv=f(new[k],'semantic_max_common_support')
   if bv is not None:bs.append(bv)
   if nv is not None:ns.append(nv)
   if bv is not None and nv is not None and bv>1e-12:ratios.append(nv/bv)
   pb=f(base[k],'absolute_feasibility_probability');pn=f(new[k],'absolute_feasibility_probability')
   if pb is not None and pn is not None:dprob.append(pn-pb)
  out['categories'][name]={'rows':sum(pred(base[k]) for k in keys),'support_ratio_mean':mean(ratios),'support_ratio_median':median(ratios),'base_support_mean':mean(bs),'new_support_mean':mean(ns),'probability_delta_mean':mean(dprob)}
 tf=out['categories']['teacher_feasible']['support_ratio_mean'];ti=out['categories']['teacher_infeasible']['support_ratio_mean'];sp=out['categories']['safe_positive']['support_ratio_mean'];hm=out['categories']['harmful']['support_ratio_mean']
 out['selective_retention']={'teacher_feasible_ratio_gt_teacher_infeasible':(tf is not None and ti is not None and tf>ti),'safe_positive_ratio_gt_harmful':(sp is not None and hm is not None and sp>hm),'teacher_feasible_minus_infeasible':None if tf is None or ti is None else tf-ti,'safe_positive_minus_harmful':None if sp is None or hm is None else sp-hm}
 return out
def main():
 ap=argparse.ArgumentParser(description='v48.74 signed finite-time viability row-level causal audit');ap.add_argument('--reference',type=Path,required=True);ap.add_argument('--p74',type=Path,required=True);ap.add_argument('--q74',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 doc={'schema':'ocrap-v48.74-svbw-signed-viability-audit-v2','reference_semantics':'accepted projection-fidelity reference','test_roots_read':False,'dataset_reconstruction':False,'comparisons':{}}
 for v in VARIANTS:
  doc['comparisons'][v]={}
  for k in KINDS:
   ref=rows(a.reference,v,k);p=rows(a.p74,v,k);q=rows(a.q74,v,k)
   doc['comparisons'][v][k]={'P74_minus_reference':summarize(ref,p),'Q74_minus_P74':summarize(p,q),'Q74_minus_reference':summarize(ref,q)}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');print(json.dumps({'event':'v48_74_svbw_signed_viability_audit','output':str(a.output)}))
if __name__=='__main__':main()
