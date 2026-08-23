#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math
from collections import Counter
from pathlib import Path
import numpy as np

LIMITING=('clearance','stopping','control','stability')
def read_rows(path:Path):
 rows=[]
 if not path.is_file():return rows
 for line in path.read_text(encoding='utf-8').splitlines():
  if line.strip():rows.append(json.loads(line))
 return rows
def auc(labels,scores):
 y=np.asarray(labels,dtype=bool);s=np.asarray(scores,dtype=float);p=s[y];n=s[~y]
 if not len(p) or not len(n):return None
 return float(((p[:,None]>n[None,:]).sum()+.5*(p[:,None]==n[None,:]).sum())/(len(p)*len(n)))
def mean(rows,key):
 vals=[float(r[key]) for r in rows if r.get(key) is not None and math.isfinite(float(r[key]))];return float(np.mean(vals)) if vals else None
def frac(rows,pred):return float(sum(bool(pred(r)) for r in rows)/len(rows)) if rows else None
def semantic_summary(rows):
 if not rows or not any(r.get('semantic_positive_option_count') is not None for r in rows):return None
 c=Counter();bars=[[],[],[],[]]
 for r in rows:
  x=r.get('semantic_limiting_constraint')
  if x is not None:
   i=int(x);c[LIMITING[i] if 0<=i<len(LIMITING) else f'unknown_{i}']+=1
  b=r.get('semantic_best_barriers')
  if isinstance(b,list) and len(b)>=4:
   for j in range(4):
    try:
     v=float(b[j])
     if math.isfinite(v):bars[j].append(v)
    except Exception:pass
 return {'rows':len(rows),'any_positive_common_option_fraction':frac(rows,lambda r:float(r.get('semantic_positive_option_count') or 0)>0),
         'universal_failure_fraction':frac(rows,lambda r:float(r.get('semantic_universal_failure') or 0)>0),
         'best_common_viability_mean':mean(rows,'semantic_best_common_viability'),'max_common_support_mean':mean(rows,'semantic_max_common_support'),
         'limiting_constraint_counts':dict(c),'limiting_constraint_fractions':{k:float(v/len(rows)) for k,v in c.items()},
         'best_option_barrier_means':{LIMITING[j]:(float(np.mean(bars[j])) if bars[j] else None) for j in range(4)}}
def summarize(rows,positive_gain=.015):
 if not rows:return {'rows':0}
 safe=[r for r in rows if float(r.get('teacher_adv',-1e9))>=positive_gain and not bool(r.get('teacher_harmful',False))];harm=[r for r in rows if bool(r.get('teacher_harmful',False))]
 feas=[r for r in rows if float(r['teacher_candidate_r_dep'])>=0];inf=[r for r in rows if float(r['teacher_candidate_r_dep'])<0];labels=[float(r['teacher_candidate_r_dep'])>=0 for r in rows]
 probs=[r.get('absolute_feasibility_probability') for r in rows];have=all(x is not None and math.isfinite(float(x)) for x in probs)
 d={'rows':len(rows),'groups':len({(r.get('scene'),r.get('time'),r.get('fold')) for r in rows}),'safe_positive_rows':len(safe),'harmful_rows':len(harm),
    'teacher_safe_positive_candidate_r_dep_nonnegative_fraction':sum(float(r['teacher_candidate_r_dep'])>=0 for r in safe)/len(safe) if safe else None,
    'teacher_harmful_candidate_r_dep_negative_fraction':sum(float(r['teacher_candidate_r_dep'])<0 for r in harm)/len(harm) if harm else None}
 if have:
  ps=[float(x) for x in probs];d.update({'absolute_feasibility_auc':auc(labels,ps),'absolute_feasibility_accuracy_at_0_5':float(np.mean([(p>=.5)==y for p,y in zip(ps,labels)])),
   'safe_positive_pass_fraction':sum(float(r['absolute_feasibility_probability'])>=.5 for r in safe)/len(safe) if safe else None,
   'harmful_pass_fraction':sum(float(r['absolute_feasibility_probability'])>=.5 for r in harm)/len(harm) if harm else None,
   'teacher_infeasible_count':len(inf),'teacher_feasible_count':len(feas),'teacher_infeasible_pass_fraction':sum(float(r['absolute_feasibility_probability'])>=.5 for r in inf)/len(inf) if inf else None,
   'teacher_feasible_reject_fraction':sum(float(r['absolute_feasibility_probability'])<.5 for r in feas)/len(feas) if feas else None})
 if any(r.get('quantifier_positive_option_count') is not None for r in rows):
  subs={'safe_positive':safe,'harmful':harm,'teacher_feasible':feas,'teacher_infeasible':inf};q={}
  for n,z in subs.items():q[n]={'rows':len(z),'any_positive_common_option_fraction':frac(z,lambda r:float(r.get('quantifier_positive_option_count') or 0)>0),'universal_failure_fraction':frac(z,lambda r:float(r.get('quantifier_universal_failure') or 0)>0),'best_common_viability_mean':mean(z,'quantifier_best_common_viability'),'max_common_support_mean':mean(z,'quantifier_max_common_support')}
  d['quantifier_coverage_diagnostics']=q
 if any(r.get('semantic_positive_option_count') is not None for r in rows):
  subs={'safe_positive':safe,'harmful':harm,'teacher_feasible':feas,'teacher_infeasible':inf};d['semantic_coverage_diagnostics']={n:semantic_summary(z) for n,z in subs.items()}
 return d
def paths(run:Path,v:str):
 b=run/'candidates'/v/'calibration';return {'dev_near':b/'dev_diagnostic_near_v48.proposal_rows.jsonl','dev_contact':b/'dev_diagnostic_contact_v48.proposal_rows.jsonl','certificate_near':b/'direct_value_risk_near_v48.proposal_rows.jsonl','certificate_contact':b/'direct_value_risk_contact_v48.proposal_rows.jsonl'}
def main():
 ap=argparse.ArgumentParser(description='v48.64 OC-SARW feasibility-role, witness-coverage and limiting-constraint audit');ap.add_argument('--arm',action='append',required=True);ap.add_argument('--variant',action='append',default=[]);ap.add_argument('--positive-gain',type=float,default=.015);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();out={}
 for spec in a.arm:
  name,raw=spec.split('=',1);run=Path(raw);out[name]={}
  for v in a.variant or ['balanced','precision']:out[name][v]={k:summarize(read_rows(p),a.positive_gain) for k,p in paths(run,v).items()}
 doc={'schema':'ocrap-v48.64-sarw-feasibility-role-audit-v1','positive_gain':a.positive_gain,'arms':out,'interpretation_contract':{'teacher_boundary':'candidate R_dep >= 0','stage_ii_threshold':0.5,'safe_positive':'teacher_adv >= positive_gain AND not teacher_harmful','no_threshold_search':True,'test_roots_read':False,'semantic_diagnostics':'diagnostic only; no training/selection use','limiting_constraint_order':list(LIMITING)}}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');print(json.dumps({'event':'v48_64_feasibility_role_audit','output':str(a.output)}))
if __name__=='__main__':main()
