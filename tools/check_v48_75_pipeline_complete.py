#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path

def load(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for z in iter(lambda:f.read(1<<20),b''):h.update(z)
 return h.hexdigest()

def main():
 ap=argparse.ArgumentParser();
 for name in ('reference_contract','runtime_contract','v74_complete','v74_comparison','censored_projection_run','censored_fidelity_run','truth_audit','comparison'):
  ap.add_argument('--'+name.replace('_','-'),required=True,type=Path)
 ap.add_argument('--output',required=True,type=Path);a=ap.parse_args();errors=[];art={}
 for name in ('reference_contract','runtime_contract','v74_complete','v74_comparison','truth_audit','comparison'):
  p=getattr(a,name)
  if not p.is_file():errors.append(f'missing {name}: {p}');continue
  art[str(p)]=sha(p)
 rc=load(a.reference_contract) if a.reference_contract.is_file() else {}; rt=load(a.runtime_contract) if a.runtime_contract.is_file() else {}; v74c=load(a.v74_complete) if a.v74_complete.is_file() else {}; v74=load(a.v74_comparison) if a.v74_comparison.is_file() else {}; cmp=load(a.comparison) if a.comparison.is_file() else {}
 if not (rc.get('valid') and not rc.get('test_roots_read')):errors.append('reference reuse invalid')
 if not (rt.get('valid') and rt.get('attribution_ready') and rt.get('engineering_version')=='v48.75.0-OC-STCA' and not rt.get('test_roots_read')):errors.append('runtime contract invalid')
 pre=(v74.get('preregistered_decision') or {})
 if not (v74c.get('valid') and v74c.get('attribution_ready') and v74c.get('engineering_version')=='v48.74.2-OC-SVBW-ENGFIX' and pre.get('status')=='STOP' and pre.get('next_branch')=='signed_viability_stop_then_supervision_truth_contract_no_parameter_sweep'):errors.append('V48.74 branch prerequisite invalid')
 arms={'C75_PROJ_CENSORED':a.censored_projection_run,'D75_FIDELITY_CENSORED':a.censored_fidelity_run}
 for arm,run in arms.items():
  for p in (run/'V48_75_VARIANT_ISOLATION.json',run/'V48_75_FACTOR_CONTRACT.json',run/'dedicated_recalibration_status.json'):
   if not p.is_file():errors.append(f'{arm}: missing {p}');continue
   art[str(p)]=sha(p)
  vi=load(run/'V48_75_VARIANT_ISOLATION.json') if (run/'V48_75_VARIANT_ISOLATION.json').is_file() else {}
  fc=load(run/'V48_75_FACTOR_CONTRACT.json') if (run/'V48_75_FACTOR_CONTRACT.json').is_file() else {}
  if not vi.get('valid'):errors.append(f'{arm}: variant isolation invalid')
  if not (fc.get('engineering_version')=='v48.75.0-OC-STCA' and fc.get('absolute_feasibility_truth_contract')=='censor_exact_0p5' and fc.get('dataset_reconstruction') is False and fc.get('test_roots_read') is False):errors.append(f'{arm}: factor contract invalid')
  for v in ('balanced','precision'):
   p=run/'candidates'/v/'V48_75_STAGE_I_STATE_ISOLATION.json'
   if not p.is_file():errors.append(f'{arm}/{v}: missing state isolation');continue
   st=load(p);art[str(p)]=sha(p)
   if not st.get('valid'):errors.append(f'{arm}/{v}: state isolation invalid')
 if not (cmp.get('valid') and cmp.get('attribution_ready')):errors.append('V48.75 comparison attribution invalid')
 valid=not errors
 doc={'schema':'ocrap-v48.75-stca-pipeline-complete-v1','algorithm_version':'v48.75-DCP-DRFC-BCDE-RIFA-OC-STCA','engineering_version':'v48.75.0-OC-STCA','valid':valid,'attribution_ready':valid,'errors':errors,'arms':{k:str(v) for k,v in arms.items()},'historical_arms':{'Q67_CTRLPROJ':'historical','T68_FIDELITY':'historical'},'artifact_sha256':art,'dataset_reconstruction':False,'teacher_labels_rewritten':False,'test_roots_read':False}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');print(json.dumps({'event':'v48_75_pipeline_complete','valid':valid,'output':str(a.output)}));return 0 if valid else 30
if __name__=='__main__':raise SystemExit(main())
