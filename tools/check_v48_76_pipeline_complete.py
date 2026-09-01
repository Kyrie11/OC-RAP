#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path

def load(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256();h.update(Path(p).read_bytes());return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();
 for x in ['reference_contract','runtime_contract','v75_complete','v75_comparison','e76_run','f76_run','audit','comparison','output']:ap.add_argument('--'+x.replace('_','-'),dest=x,type=Path,required=True)
 a=ap.parse_args();errors=[]
 rc=load(a.reference_contract);rt=load(a.runtime_contract);v75c=load(a.v75_complete);v75p=load(a.v75_comparison);au=load(a.audit);co=load(a.comparison)
 pre=v75p.get('preregistered_decision') or {}
 if not rc.get('valid'):errors.append('reference contract invalid')
 if not (rt.get('valid') and rt.get('attribution_ready') and rt.get('engineering_version')=='v48.76.0-OC-ICSM'):errors.append('runtime contract invalid')
 if not (v75c.get('valid') and v75c.get('attribution_ready') and v75c.get('engineering_version')=='v48.75.0-OC-STCA' and pre.get('status')=='STOP' and pre.get('next_branch')=='truth_floor_debt_not_dominant_training_cause_audit_absolute_supervision_representation_no_geometry_sweep'):errors.append('V48.75 branch prerequisite invalid')
 arts={}
 for label,run in [('E76_MARGIN_PROJ',a.e76_run),('F76_MARGIN_FIDELITY',a.f76_run)]:
  for fn in ['V48_76_FACTOR_CONTRACT.json','V48_76_VARIANT_ISOLATION.json','dedicated_recalibration_status.json']:
   p=run/fn
   if not p.is_file():errors.append(f'{label}: missing {fn}');continue
   arts[str(p)]=sha(p)
  fc=load(run/'V48_76_FACTOR_CONTRACT.json') if (run/'V48_76_FACTOR_CONTRACT.json').is_file() else {}
  vi=load(run/'V48_76_VARIANT_ISOLATION.json') if (run/'V48_76_VARIANT_ISOLATION.json').is_file() else {}
  if not (fc.get('engineering_version')=='v48.76.0-OC-ICSM' and fc.get('absolute_feasibility_truth_contract')=='censor_exact_0p5' and fc.get('absolute_feasibility_supervision_objective')=='signed_margin_huber' and fc.get('dataset_reconstruction') is False and fc.get('test_roots_read') is False):errors.append(f'{label}: factor contract invalid')
  if not vi.get('valid'):errors.append(f'{label}: variant isolation invalid')
  for v in ('balanced','precision'):
   p=run/'candidates'/v/'V48_76_STAGE_I_STATE_ISOLATION.json'
   if not p.is_file():errors.append(f'{label}/{v}: state isolation missing')
   else:
    d=load(p);arts[str(p)]=sha(p)
    if not (d.get('valid') and d.get('stage_i_bitwise_identity') and d.get('supervision_objective_valid') and d.get('truth_contract_valid')):errors.append(f'{label}/{v}: state isolation invalid')
 if au.get('schema')!='ocrap-v48.76-icsm-signed-margin-audit-v1':errors.append('audit schema invalid')
 if not (co.get('valid') and co.get('attribution_ready')):errors.append('comparison invalid')
 for p in (a.reference_contract,a.runtime_contract,a.v75_complete,a.v75_comparison,a.audit,a.comparison):arts[str(p)]=sha(p)
 valid=not errors
 doc={'schema':'ocrap-v48.76-icsm-pipeline-complete-v1','algorithm_version':'v48.76-DCP-DRFC-BCDE-RIFA-OC-ICSM','engineering_version':'v48.76.0-OC-ICSM','valid':valid,'attribution_ready':valid,'errors':errors,'arms':{'E76_MARGIN_PROJ':str(a.e76_run),'F76_MARGIN_FIDELITY':str(a.f76_run),'historical_C75':'historical','historical_D75':'historical'},'artifact_sha256':arts,'dataset_reconstruction':False,'test_roots_read':False}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');print(json.dumps({'event':'v48_76_pipeline_complete','valid':valid,'output':str(a.output)}));return 0 if valid else 30
if __name__=='__main__':raise SystemExit(main())
