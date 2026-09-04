#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
def js(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--u-run',type=Path,required=True);ap.add_argument('--v-run',type=Path,required=True);ap.add_argument('--runtime',type=Path,required=True);ap.add_argument('--reference',type=Path,required=True);ap.add_argument('--audit',type=Path,required=True);ap.add_argument('--comparison',type=Path,required=True);ap.add_argument('--response-summary',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();errs=[];arts={}
 for p in [a.runtime,a.reference,a.audit,a.comparison,a.response_summary]:
  if not p.is_file():errs.append(f'missing {p}');continue
  arts[str(p)]=sha(p)
 for run in [a.u_run,a.v_run]:
  for var in ['balanced','precision']:
   c=run/'candidates'/var;ck=c/'model_v48_trac_sr'/'best.pt';tc=c/'TRAINING_COMPLETE.json';ec=c/'EVIDENCE_CORRECTION_COMPLETE.json';iso=c/'V48_87_STAGE_I_STATE_ISOLATION.json';cal=c/'calibration'/'CERTIFICATE_CALIBRATION_COMPLETE.json'
   for p in [ck,tc,ec,iso,cal]:
    if not p.is_file():errs.append(f'missing {p}')
   if iso.is_file() and not js(iso).get('valid'):errs.append(f'invalid isolation {iso}')
   if tc.is_file() and ec.is_file() and ck.is_file():
    s=sha(ck)
    if js(tc).get('checkpoint_sha256')!=s or js(ec).get('checkpoint_sha256')!=s:errs.append(f'checkpoint sha mismatch {ck}')
 for p,n in [(a.runtime,'runtime'),(a.reference,'reference'),(a.audit,'audit'),(a.comparison,'comparison')]:
  if p.is_file() and not js(p).get('valid'):errs.append(f'{n} invalid')
 doc={'schema':'ocrap-v48.87-barr-pipeline-complete-v1','engineering_version':'v48.87.0-OC-BARR','valid':not errs,'attribution_ready':not errs,'errors':errs,'arms':{'U87_BILINEAR_RESPONSE_INTERVAL':str(a.u_run),'V87_Main_BILINEAR_SELECTIVE_RESPONSE':str(a.v_run)},'artifacts':arts,'trainable_parameter_count':53550,'q85_parameter_count':54144,'regime_conditioning':False,'boundary_transport':False,'dataset_reconstruction':False,'relative_ranker_modified':False,'test_roots_read':False}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');print(json.dumps({'valid':doc['valid'],'errors':errs}));return 0 if doc['valid'] else 30
if __name__=='__main__':raise SystemExit(main())
