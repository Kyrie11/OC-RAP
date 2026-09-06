#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from ocrap.v48_96_support_reserve_root_observability import ENGINEERING_VERSION

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main()->int:
 ap=argparse.ArgumentParser();
 for x in ['runtime','balanced','precision','comparison','v48-95-pipeline']:ap.add_argument('--'+x,type=Path,required=True)
 ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();errors=[]
 docs={k:json.loads(getattr(a,k.replace('-','_')).read_text()) for k in ['runtime','balanced','precision','comparison']}
 for k,d in docs.items():
  if not d.get('valid'):errors.append(f'{k} invalid')
  if d.get('engineering_version')!=ENGINEERING_VERSION:errors.append(f'{k} engineering version mismatch')
 p95=json.loads(a.v48_95_pipeline.read_text())
 if not(p95.get('valid') and p95.get('attribution_ready') and p95.get('preregistered_status')=='FROZEN_NATIVE_RECOVERY_OBSERVABILITY_STOP'):errors.append('V48.95 STOP prerequisite missing')
 cmp=docs['comparison'];status=(cmp.get('preregistered_decision') or {}).get('status')
 out={'schema':'ocrap-v48.96-srroa-pipeline-complete-v1','engineering_version':ENGINEERING_VERSION,'valid':not errors,'attribution_ready':not errors,'errors':errors,
      'experiment_type':'audit_only_target_specific_frozen_root_support_reserve_observability','planner_parameters_trained':0,'dataset_reconstruction':False,'dataset_reselection':False,'teacher_labels_changed':False,'teacher_metadata_input_to_model':False,'boundary_transport':False,'relative_ranker_modified':False,'regime_conditioning':False,'test_roots_read':False,'womd_replay_performed':False,'preregistered_status':status,
      'artifacts':{k:{'path':str(getattr(a,k).resolve()),'sha256':sha(getattr(a,k))} for k in ['runtime','balanced','precision','comparison']}}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps({'valid':out['valid'],'status':status,'errors':errors}));return 0 if out['valid'] else 30
if __name__=='__main__':raise SystemExit(main())
