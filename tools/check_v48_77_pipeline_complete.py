#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path

def load(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256();h.update(Path(p).read_bytes());return h.hexdigest()
def main():
 ap=argparse.ArgumentParser()
 for x in ['reference_contract','runtime_contract','v76_complete','v76_comparison','g77_run','h77_run','audit','comparison','output']:ap.add_argument('--'+x.replace('_','-'),dest=x,type=Path,required=True)
 a=ap.parse_args();errors=[]
 rc=load(a.reference_contract);rt=load(a.runtime_contract);v76c=load(a.v76_complete);v76p=load(a.v76_comparison);au=load(a.audit);co=load(a.comparison);pre=v76p.get('preregistered_decision') or {}
 if not rc.get('valid'):errors.append('reference contract invalid')
 if not (rt.get('valid') and rt.get('attribution_ready') and rt.get('engineering_version')=='v48.77.0-OC-ACTSI'):errors.append('runtime contract invalid')
 if not (v76c.get('valid') and v76c.get('attribution_ready') and v76c.get('engineering_version')=='v48.76.0-OC-ICSM' and pre.get('status')=='STOP' and pre.get('signed_margin_supervision_go') is False and pre.get('next_branch')=='signed_margin_supervision_stop_two_gain_transport_representation_bottleneck_then_structured_absolute_source_interface'):errors.append('V48.76 branch prerequisite invalid')
 arts={}
 for label,run,fid in [('G77_TYPED_PROJ',a.g77_run,False),('H77_MAIN_ACTSI',a.h77_run,True)]:
  for fn in ['V48_77_FACTOR_CONTRACT.json','V48_77_VARIANT_ISOLATION.json','dedicated_recalibration_status.json']:
   p=run/fn
   if not p.is_file():errors.append(f'{label}: missing {fn}');continue
   arts[str(p)]=sha(p)
  fc=load(run/'V48_77_FACTOR_CONTRACT.json') if (run/'V48_77_FACTOR_CONTRACT.json').is_file() else {};vi=load(run/'V48_77_VARIANT_ISOLATION.json') if (run/'V48_77_VARIANT_ISOLATION.json').is_file() else {}
  if not (fc.get('engineering_version')=='v48.77.0-OC-ACTSI' and fc.get('absolute_feasibility_truth_contract')=='censor_exact_0p5' and fc.get('absolute_feasibility_supervision_objective')=='signed_margin_huber' and fc.get('active_constraint_typed_source') is True and fc.get('projection_fidelity') is fid and fc.get('trainable_parameters')==12 and fc.get('dataset_reconstruction') is False and fc.get('test_roots_read') is False):errors.append(f'{label}: factor contract invalid')
  if not vi.get('valid'):errors.append(f'{label}: variant isolation invalid')
  for v in ('balanced','precision'):
   p=run/'candidates'/v/'V48_77_STAGE_I_STATE_ISOLATION.json'
   if not p.is_file():errors.append(f'{label}/{v}: state isolation missing')
   else:
    d=load(p);arts[str(p)]=sha(p)
    if not (d.get('valid') and d.get('stage_i_bitwise_identity') and d.get('supervision_objective_valid') and d.get('truth_contract_valid') and d.get('new_tensor_shape')==[6,2] and bool((d.get('factor_flags') or {}).get('active_constraint_typed_source'))):errors.append(f'{label}/{v}: state isolation invalid')
    ckpt=run/'candidates'/v/'model_v48_trac_sr'/'best.pt'
    if not ckpt.is_file():errors.append(f'{label}/{v}: best.pt missing from run; packaging must retain checkpoints')
 if au.get('schema')!='ocrap-v48.77-actsi-audit-v1':errors.append('audit schema invalid')
 if not (co.get('valid') and co.get('attribution_ready')):errors.append('comparison invalid')
 for p in (a.reference_contract,a.runtime_contract,a.v76_complete,a.v76_comparison,a.audit,a.comparison):arts[str(p)]=sha(p)
 valid=not errors
 doc={'schema':'ocrap-v48.77-actsi-pipeline-complete-v1','algorithm_version':'v48.77-DCP-DRFC-BCDE-RIFA-OC-ACTSI','engineering_version':'v48.77.0-OC-ACTSI','valid':valid,'attribution_ready':valid,'errors':errors,'arms':{'G77_TYPED_PROJ':str(a.g77_run),'H77_MAIN_ACTSI':str(a.h77_run),'historical_E76':'historical','historical_F76':'historical','historical_C75_native':'historical','historical_D75_native':'historical'},'artifact_sha256':arts,'checkpoint_packaging_required':True,'dataset_reconstruction':False,'test_roots_read':False}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');print(json.dumps({'event':'v48_77_pipeline_complete','valid':valid,'output':str(a.output)}));return 0 if valid else 30
if __name__=='__main__':raise SystemExit(main())
