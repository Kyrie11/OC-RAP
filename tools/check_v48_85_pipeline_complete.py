#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path

def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for z in iter(lambda:f.read(1<<20),b''):h.update(z)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--q-run',type=Path,required=True);ap.add_argument('--r-run',type=Path,required=True);ap.add_argument('--runtime',type=Path,required=True);ap.add_argument('--reference',type=Path,required=True);ap.add_argument('--audit',type=Path,required=True);ap.add_argument('--comparison',type=Path,required=True);ap.add_argument('--train-truth-index',type=Path,required=True);ap.add_argument('--eval-truth-index',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();errors=[]
 def rd(p):
  if not p.is_file():errors.append(f'missing {p}');return {}
  return json.loads(p.read_text())
 runtime,ref,au,co=map(rd,[a.runtime,a.reference,a.audit,a.comparison])
 if not runtime.get('valid'):errors.append('runtime invalid')
 if not ref.get('valid'):errors.append('reference invalid')
 if not au.get('valid'):errors.append('audit invalid')
 if not co.get('valid'):errors.append('comparison invalid')
 for run,state in [(a.q_run,False),(a.r_run,True)]:
  for v in ('balanced','precision'):
   ck=run/'candidates'/v/'model_v48_trac_sr'/'best.pt';iso=run/'candidates'/v/'V48_85_STAGE_I_STATE_ISOLATION.json'
   if not ck.is_file():errors.append(f'missing best.pt {ck}')
   d=rd(iso)
   if d and (not d.get('valid') or bool(d.get('factor_flags',{}).get('state_conditioning'))!=state):errors.append(f'isolation invalid {iso}')
 for p in (a.train_truth_index,a.eval_truth_index):
  if not p.is_file():errors.append(f'missing truth index {p}')
 doc={'schema':'ocrap-v48.85-sarr-pipeline-complete-v1','engineering_version':'v48.85.1-OC-SARR-ENGFIX','valid':not errors,'attribution_ready':not errors,'errors':errors,'arms':{'Q85_ACTION_RESPONSE':str(a.q_run),'R85_STATE_ACTION':str(a.r_run)},'boundary_transport':False,'dataset_reconstruction':False,'test_roots_read':False,'truth_contract':'V48.80 structural_interval_bounds frozen/reused','artifacts':{str(p):sha(p) for p in [a.runtime,a.reference,a.audit,a.comparison,a.train_truth_index,a.eval_truth_index] if p.is_file()}}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');print(json.dumps({'valid':doc['valid'],'errors':errors}));return 0 if doc['valid'] else 30
if __name__=='__main__':raise SystemExit(main())
