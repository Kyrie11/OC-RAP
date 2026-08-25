#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
KINDS={'dev_near':'dev_diagnostic_near_v48.proposal_rows.jsonl','dev_contact':'dev_diagnostic_contact_v48.proposal_rows.jsonl','certificate_near':'direct_value_risk_near_v48.proposal_rows.jsonl','certificate_contact':'direct_value_risk_contact_v48.proposal_rows.jsonl'}
VARIANTS=('balanced','precision')
def rows(run,v,k):
 p=run/'candidates'/v/'calibration'/KINDS[k];out={}
 if not p.is_file():return out
 for line in p.read_text().splitlines():
  if not line.strip():continue
  r=json.loads(line);out[(r.get('scene'),r.get('time'),r.get('candidate'))]=r
 return out
def pos(r):
 try:return float(r.get('semantic_best_common_viability'))>0
 except Exception:return False
def feas(r):return float(r.get('teacher_candidate_r_dep'))>=0
def safe(r,g=.015):return float(r.get('teacher_adv',-1e9))>=g and not bool(r.get('teacher_harmful',False))
def summarize(a,b):
 keys=set(a)&set(b);cats={'retained':[],'lost':[],'gained':[]}
 for k in keys:
  pa,pb=pos(a[k]),pos(b[k])
  if pa and pb:cats['retained'].append(b[k])
  elif pa and not pb:cats['lost'].append(a[k])
  elif (not pa) and pb:cats['gained'].append(b[k])
 out={'aligned_rows':len(keys)}
 for n,z in cats.items():
  out[n]={'rows':len(z),'teacher_feasible_precision':(sum(feas(r) for r in z)/len(z) if z else None),'safe_positive_rows':sum(safe(r) for r in z),'harmful_rows':sum(bool(r.get('teacher_harmful',False)) for r in z)}
 return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--q67',type=Path,required=True);ap.add_argument('--t',type=Path,required=True);ap.add_argument('--u',type=Path,required=True);ap.add_argument('--v',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();doc={'schema':'ocrap-v48.68-rtrw-robust-trust-audit-v1','test_roots_read':False,'dataset_reconstruction':False,'comparisons':{}}
 for var in VARIANTS:
  doc['comparisons'][var]={}
  for k in KINDS:
   q=rows(a.q67,var,k);doc['comparisons'][var][k]={'T_vs_Q67':summarize(q,rows(a.t,var,k)),'U_vs_Q67':summarize(q,rows(a.u,var,k)),'V_vs_Q67':summarize(q,rows(a.v,var,k))}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');print(json.dumps({'event':'v48_68_rtrw_robust_trust_audit','output':str(a.output)}))
if __name__=='__main__':main()
