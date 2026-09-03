#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path

def js(p): return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--s-run',type=Path,required=True);ap.add_argument('--t-run',type=Path,required=True);ap.add_argument('--runtime',type=Path,required=True);ap.add_argument('--reference',type=Path,required=True);ap.add_argument('--audit',type=Path,required=True);ap.add_argument('--comparison',type=Path,required=True);ap.add_argument('--response-summary',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();errs=[];arts={}
 for p in [a.runtime,a.reference,a.audit,a.comparison,a.response_summary]:
  if not p.is_file():errs.append(f'missing {p}');continue
  arts[str(p)]=sha(p)
 for run in [a.s_run,a.t_run]:
  for v in ['balanced','precision']:
   c=run/'candidates'/v; ck=c/'model_v48_trac_sr'/'best.pt'; tc=c/'TRAINING_COMPLETE.json'; ec=c/'EVIDENCE_CORRECTION_COMPLETE.json'; iso=c/'V48_86_STAGE_I_STATE_ISOLATION.json'; cal=c/'calibration'/'CERTIFICATE_CALIBRATION_COMPLETE.json'
   for p in [ck,tc,ec,iso,cal]:
    if not p.is_file():errs.append(f'missing {p}')
   if iso.is_file() and not js(iso).get('valid'):errs.append(f'invalid isolation {iso}')
   if tc.is_file() and ec.is_file() and ck.is_file():
    s=sha(ck)
    if js(tc).get('checkpoint_sha256')!=s or js(ec).get('checkpoint_sha256')!=s:errs.append(f'checkpoint sha mismatch {ck}')
 if a.runtime.is_file() and not js(a.runtime).get('valid'):errs.append('runtime invalid')
 if a.reference.is_file() and not js(a.reference).get('valid'):errs.append('reference invalid')
 if a.audit.is_file() and not js(a.audit).get('valid'):errs.append('audit invalid')
 if a.comparison.is_file() and not js(a.comparison).get('valid'):errs.append('comparison invalid')
 doc={'schema':'ocrap-v48.86-crsc-pipeline-complete-v1','engineering_version':'v48.86.0-OC-CRSC','valid':not errs,'attribution_ready':not errs,'errors':errs,'arms':{'S86_RESPONSE_INTERVAL':str(a.s_run),'T86_SELECTIVE_RESPONSE':str(a.t_run)},'artifacts':arts,'boundary_transport':False,'dataset_reconstruction':False,'relative_ranker_modified':False,'test_roots_read':False}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');print(json.dumps({'valid':doc['valid'],'errors':errs}));return 0 if doc['valid'] else 30
if __name__=='__main__':raise SystemExit(main())
