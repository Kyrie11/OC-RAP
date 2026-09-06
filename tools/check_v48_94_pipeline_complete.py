#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from ocrap.v48_94_support_reserve_admission import ENGINEERING_VERSION
def sha(p:Path)->str:
 h=hashlib.sha256();h.update(p.read_bytes());return h.hexdigest()
def main()->int:
 ap=argparse.ArgumentParser()
 for n in ('runtime','audit','comparison','v48_93_pipeline','v48_93_comparison'): ap.add_argument('--'+n.replace('_','-'),type=Path,required=True)
 ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();errors=[]
 docs={k:json.loads(getattr(a,k).read_text()) for k in ('runtime','audit','comparison','v48_93_pipeline','v48_93_comparison')}
 for k in ('runtime','audit','comparison'):
  if not(docs[k].get('valid') and docs[k].get('attribution_ready')):errors.append(f'{k} invalid')
  if docs[k].get('engineering_version')!=ENGINEERING_VERSION:errors.append(f'{k} version mismatch')
 q93=docs['v48_93_comparison'].get('preregistered_decision') or {}
 if not(docs['v48_93_pipeline'].get('valid') and docs['v48_93_pipeline'].get('preregistered_status')=='PCD_FACTOR_COMPLEMENTARITY_GO'):errors.append('V48.93 pipeline complementarity GO prerequisite missing')
 if not(docs['v48_93_comparison'].get('valid') and q93.get('status')=='PCD_FACTOR_COMPLEMENTARITY_GO'):errors.append('V48.93 comparison complementarity GO prerequisite missing')
 d=docs['comparison'].get('preregistered_decision') or {}; allowed={'SUPPORT_RESERVE_ABSOLUTE_SOURCE_GO','SUPPORT_RESERVE_COMPLEMENTARITY_MECHANISM_GO_SOURCE_STOP','SUPPORT_RESERVE_COMPLEMENTARITY_STOP'}
 if d.get('status') not in allowed:errors.append('unexpected V48.94 status')
 out={'schema':'ocrap-v48.94-srca-pipeline-complete-v1','engineering_version':ENGINEERING_VERSION,'valid':not errors,'attribution_ready':not errors,'errors':errors,'experiment_type':'fixed_zero_parameter_support_reserve_complementarity_absolute_source','planner_parameters_trained':0,'dataset_reconstruction':False,'dataset_reselection':False,'teacher_labels_changed':False,'teacher_metadata_input_to_model':False,'regime_conditioning':False,'boundary_transport':False,'relative_ranker_modified':False,'test_roots_read':False,'preregistered_status':d.get('status'),'absolute_source_freeze_authorized':bool(d.get('absolute_source_freeze_authorized')),'artifacts':{k:{'path':str(getattr(a,k).resolve()),'sha256':sha(getattr(a,k))} for k in ('runtime','audit','comparison')}}
 a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps({'valid':out['valid'],'status':out['preregistered_status'],'freeze':out['absolute_source_freeze_authorized']}));return 0 if out['valid'] else 30
if __name__=='__main__':raise SystemExit(main())
