#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np

def read_rows(path: Path):
    rows=[]
    if not path.is_file(): return rows
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.strip(): rows.append(json.loads(line))
    return rows

def auc(labels,scores):
    y=np.asarray(labels,dtype=bool); s=np.asarray(scores,dtype=float)
    p=s[y]; n=s[~y]
    if not len(p) or not len(n): return None
    return float(((p[:,None]>n[None,:]).sum()+.5*(p[:,None]==n[None,:]).sum())/(len(p)*len(n)))

def _mean(rows,key):
    vals=[float(r[key]) for r in rows if r.get(key) is not None and math.isfinite(float(r[key]))]
    return (float(np.mean(vals)) if vals else None)

def _frac(rows,pred):
    return (float(sum(bool(pred(r)) for r in rows)/len(rows)) if rows else None)

def summarize(rows, positive_gain=.015):
    if not rows: return {'rows':0}
    safe=[r for r in rows if float(r.get('teacher_adv',-1e9))>=positive_gain and not bool(r.get('teacher_harmful',False))]
    harmful=[r for r in rows if bool(r.get('teacher_harmful',False))]
    feasible_rows=[r for r in rows if float(r['teacher_candidate_r_dep'])>=0]
    infeasible_rows=[r for r in rows if float(r['teacher_candidate_r_dep'])<0]
    feasible=[float(r['teacher_candidate_r_dep'])>=0 for r in rows]
    probs=[r.get('absolute_feasibility_probability') for r in rows]
    have=all(x is not None and math.isfinite(float(x)) for x in probs)
    d={
      'rows':len(rows),'groups':len({(r.get('scene'),r.get('time'),r.get('fold')) for r in rows}),
      'safe_positive_rows':len(safe),'harmful_rows':len(harmful),
      'teacher_safe_positive_candidate_r_dep_nonnegative_fraction':(sum(float(r['teacher_candidate_r_dep'])>=0 for r in safe)/len(safe) if safe else None),
      'teacher_harmful_candidate_r_dep_negative_fraction':(sum(float(r['teacher_candidate_r_dep'])<0 for r in harmful)/len(harmful) if harmful else None),
    }
    if have:
      ps=[float(x) for x in probs]
      d.update({
       'absolute_feasibility_auc':auc(feasible,ps),
       'absolute_feasibility_accuracy_at_0_5':float(np.mean([(p>=.5)==y for p,y in zip(ps,feasible)])),
       'safe_positive_pass_fraction':(sum(float(r['absolute_feasibility_probability'])>=.5 for r in safe)/len(safe) if safe else None),
       'harmful_pass_fraction':(sum(float(r['absolute_feasibility_probability'])>=.5 for r in harmful)/len(harmful) if harmful else None),
       'teacher_infeasible_count':len(infeasible_rows),'teacher_feasible_count':len(feasible_rows),
       'teacher_infeasible_pass_fraction':(sum(float(r['absolute_feasibility_probability'])>=.5 for r in infeasible_rows)/len(infeasible_rows) if infeasible_rows else None),
       'teacher_feasible_reject_fraction':(sum(float(r['absolute_feasibility_probability'])<.5 for r in feasible_rows)/len(feasible_rows) if feasible_rows else None),
      })
    # v48.63 diagnostic-only quantifier coverage.  These do not enter training,
    # selection, or the primary source gate; they distinguish representation
    # coverage from logical-composition failure.
    have_q=any(r.get('quantifier_positive_option_count') is not None for r in rows)
    if have_q:
      subsets={'safe_positive':safe,'harmful':harmful,'teacher_feasible':feasible_rows,'teacher_infeasible':infeasible_rows}
      q={}
      for name,sub in subsets.items():
        q[name]={
          'rows':len(sub),
          'any_positive_common_option_fraction':_frac(sub,lambda r: float(r.get('quantifier_positive_option_count') or 0.0)>0.0),
          'universal_failure_fraction':_frac(sub,lambda r: float(r.get('quantifier_universal_failure') or 0.0)>0.0),
          'best_common_viability_mean':_mean(sub,'quantifier_best_common_viability'),
          'max_common_support_mean':_mean(sub,'quantifier_max_common_support'),
        }
      d['quantifier_coverage_diagnostics']=q
    return d

def paths(run:Path,variant:str):
    base=run/'candidates'/variant/'calibration'
    return {
      'dev_near':base/'dev_diagnostic_near_v48.proposal_rows.jsonl',
      'dev_contact':base/'dev_diagnostic_contact_v48.proposal_rows.jsonl',
      'certificate_near':base/'direct_value_risk_near_v48.proposal_rows.jsonl',
      'certificate_contact':base/'direct_value_risk_contact_v48.proposal_rows.jsonl',
    }

def main():
    ap=argparse.ArgumentParser(description='v48.63 OC-QARW feasibility-role and quantifier-coverage audit')
    ap.add_argument('--arm', action='append', required=True, help='NAME=RUN_DIR')
    ap.add_argument('--variant', action='append', default=[])
    ap.add_argument('--positive-gain',type=float,default=.015); ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args(); variants=a.variant or ['balanced','precision']; out={}
    for spec in a.arm:
      name,raw=spec.split('=',1); run=Path(raw); out[name]={}
      for v in variants:
        out[name][v]={k:summarize(read_rows(p),a.positive_gain) for k,p in paths(run,v).items()}
    doc={'schema':'ocrap-v48.63-ocqarw-feasibility-role-audit-v1','positive_gain':a.positive_gain,'arms':out,
         'interpretation_contract':{'teacher_boundary':'candidate R_dep >= 0','stage_ii_threshold':0.5,
         'safe_positive':'teacher_adv >= positive_gain AND not teacher_harmful','no_threshold_search':True,'test_roots_read':False,
         'quantifier_diagnostics':'diagnostic only; no training/selection use'}}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'event':'v48_63_feasibility_role_audit','output':str(a.output)}))
if __name__=='__main__': main()
