#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from ocrap.v48_92_factorized_recovery_advantage import ENGINEERING_VERSION

def sha(p:Path)->str:
 h=hashlib.sha256();h.update(p.read_bytes());return h.hexdigest()
def main()->int:
 ap=argparse.ArgumentParser()
 for n in ('runtime','audit','audit_summary','comparison','v48_91_pipeline','v48_91_comparison'):ap.add_argument('--'+n.replace('_','-'),type=Path,required=True)
 ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();errors=[]
 docs={k:json.loads(getattr(args,k).read_text()) for k in ('runtime','audit_summary','comparison','v48_91_pipeline','v48_91_comparison')}
 for k in ('runtime','audit_summary','comparison'):
  if not(docs[k].get('valid') and docs[k].get('attribution_ready')):errors.append(f'{k} invalid')
  if str(docs[k].get('engineering_version'))!=ENGINEERING_VERSION:errors.append(f'{k} version mismatch')
 if not(docs['v48_91_pipeline'].get('valid') and docs['v48_91_pipeline'].get('attribution_ready')):errors.append('V48.91 pipeline invalid')
 q91=docs['v48_91_comparison'].get('preregistered_decision') or {}
 if q91.get('status')!='COMMON_EXOGENOUS_PHYSICAL_RESPONSE_STOP':errors.append('V48.91 scientific STOP prerequisite missing')
 if str(docs['audit_summary'].get('output_sha256'))!=sha(args.audit):errors.append('audit SHA mismatch')
 d=docs['comparison'].get('preregistered_decision') or {}
 allowed={'SHARED_RECOVERY_ADVANTAGE_MEDIATOR_GO','FACTORIZED_RECOVERY_ADVANTAGE_STOP'}
 if d.get('status') not in allowed:errors.append('unexpected V48.92 scientific status')
 out={'schema':'ocrap-v48.92-pipeline-complete-v1','engineering_version':ENGINEERING_VERSION,'valid':not errors,'attribution_ready':not errors,'errors':errors,
      'experiment_type':'audit_only_factorized_recovery_advantage_mediation','planner_parameters_trained':0,'dataset_reconstruction':False,'dataset_reselection':False,
      'regime_conditioning':False,'boundary_transport':False,'relative_ranker_modified':False,'teacher_labels_changed':False,'teacher_metadata_input_to_model':False,
      'womd_replay_performed':False,'test_roots_read':False,'preregistered_status':d.get('status'),
      'artifacts':{k:{'path':str(getattr(args,k).resolve()),'sha256':sha(getattr(args,k))} for k in ('runtime','audit','audit_summary','comparison')}}
 args.output.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps({'valid':out['valid'],'attribution_ready':out['attribution_ready'],'status':out['preregistered_status']}));return 0 if out['valid'] else 30
if __name__=='__main__':raise SystemExit(main())
