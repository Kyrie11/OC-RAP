#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
KINDS=('dev_near','dev_contact','certificate_near','certificate_contact'); VARIANTS=('balanced','precision')

def read(p):
 out=[]
 with Path(p).open() as f:
  for line in f:
   if line.strip(): out.append(json.loads(line))
 return out

def path(run,v,k):
 b=Path(run)/'candidates'/v/'calibration'
 m={'dev_near':'dev_diagnostic_near_v48.proposal_rows.jsonl','dev_contact':'dev_diagnostic_contact_v48.proposal_rows.jsonl','certificate_near':'direct_value_risk_near_v48.proposal_rows.jsonl','certificate_contact':'direct_value_risk_contact_v48.proposal_rows.jsonl'}
 return b/m[k]

def frac(rows,fn): return float(sum(bool(fn(r)) for r in rows)/len(rows)) if rows else None

def summarize(rows):
 safe=[r for r in rows if float(r.get('teacher_adv',-1e9))>=.015 and not bool(r.get('teacher_harmful',False))]
 cert=[r for r in safe if r.get('semantic_best_common_viability') is not None and float(r['semantic_best_common_viability'])>0]
 def bar(r,i):
  b=r.get('semantic_best_barriers'); return float(b[i]) if isinstance(b,list) and len(b)>i else float('nan')
 probs=[float(r['absolute_feasibility_probability']) for r in safe]
 return {
  'safe_positive_rows':len(safe),
  'teacher_rdep_eq_0p5_fraction':frac(safe,lambda r:abs(float(r['teacher_candidate_r_dep'])-.5)<=1e-7),
  'safe_positive_pass_fraction':frac(safe,lambda r:float(r['absolute_feasibility_probability'])>=.5),
  'safe_positive_probability_mean':float(np.mean(probs)) if probs else None,
  'safe_positive_probability_max':float(np.max(probs)) if probs else None,
  'positive_certificate_fraction':float(len(cert)/len(safe)) if safe else None,
  'pass_given_positive_certificate':frac(cert,lambda r:float(r['absolute_feasibility_probability'])>=.5),
  'control_nonpositive_fraction':frac(safe,lambda r:bar(r,2)<=0),
  'control_eq_minus1_fraction':frac(safe,lambda r:bar(r,2)<=-0.999999),
  'clearance_nonpositive_fraction':frac(safe,lambda r:bar(r,0)<=0),
  'route_nonpositive_fraction':frac(safe,lambda r:bar(r,4)<=0),
  'base4_all_positive_fraction':frac(safe,lambda r:all(bar(r,i)>0 for i in range(4))),
  'all_enabled_barriers_positive_fraction':frac(safe,lambda r:all(float(x)>0 for x in (r.get('semantic_best_barriers') or []))),
 }

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--arm',action='append',required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 arms={}
 for spec in a.arm:
  name,run=spec.split('=',1);arms[name]={}
  for v in VARIANTS:
   arms[name][v]={k:summarize(read(path(run,v,k))) for k in KINDS}
 doc={'schema':'ocrap-v48.67-boundary-debt-audit-v1','arms':arms,'positive_gain':.015,'threshold':.5,'test_roots_read':False,'dataset_reconstruction':False,
      'interpretation':'read-only diagnostic: separates positive-witness realization from positive-witness-to-absolute-boundary transport; no labels are used as model inputs'}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');print(json.dumps({'event':'v48_67_boundary_debt_audit','output':str(a.output)}))
if __name__=='__main__':main()
