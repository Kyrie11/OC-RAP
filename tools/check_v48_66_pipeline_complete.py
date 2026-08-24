#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
RIFA='rank_topk_then_absolute_feasibility_then_relative_filter_then_evidence_rerank'

def load(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for z in iter(lambda:f.read(1<<20),b''):h.update(z)
 return h.hexdigest()

def check_run(run:Path,route:bool,reentry:bool,errors:list[str],hashes:dict):
 vi=run/'V48_66_VARIANT_ISOLATION.json';factor=run/'V48_66_FACTOR_CONTRACT.json';terminal=run/'dedicated_recalibration_status.json';top=[vi,factor,terminal]
 for p in top:
  if not p.is_file():errors.append(f'missing {p}')
 if vi.is_file():
  d=load(vi)
  if not d.get('valid'):errors.append(f'{run.name}: variant isolation invalid')
  if not (d.get('active_set_alignment') is True and d.get('path_stop_alignment') is False and d.get('classlocal_transport') is False and bool(d.get('route_alignment'))==route and bool(d.get('reentry_alignment'))==reentry):errors.append(f'{run.name}: variant factor flags mismatch')
 if factor.is_file():
  d=load(factor);req={'trainable_parameters':2,'threshold':.5,'threshold_search':False,'regime_id_input':False,'proposal_top_k':5,'proposal_expansion':False,'test_roots_read':False,
      'semantic_witness_feature_schema':2,'semantic_witness_feature_source':'active_constraint_coverage_common_executable_recovery_witness','teacher_future_input':False,'relative_score_intervention':False,
      'active_set_alignment':True,'path_stop_alignment':False,'classlocal_transport':False,'route_alignment':route,'reentry_alignment':reentry,
      'correction_locus':'candidate-global same-option common support before native OC-MERO aggregation'}
  for k,v in req.items():
   if d.get(k)!=v:errors.append(f'{run.name}: factor {k}={d.get(k)!r} expected {v!r}')
 for v in ('balanced','precision'):
  base=run/'candidates'/v;cal=base/'calibration';req=[base/'model_v48_trac_sr'/'best.pt',base/'model_v48_trac_sr'/'train_summary.json',base/'TRAINING_COMPLETE.json',base/'EVIDENCE_CORRECTION_COMPLETE.json',base/'V48_66_STAGE_I_STATE_ISOLATION.json',base/'POLICY_CONTRACT.env',cal/'METRIC_CALIBRATION_CONTRACT.json',cal/'dev_diagnostic_near_v48.json',cal/'dev_diagnostic_contact_v48.json',cal/'dev_diagnostic_near_v48.proposal_rows.jsonl',cal/'dev_diagnostic_contact_v48.proposal_rows.jsonl',cal/'direct_value_risk_near_v48.json',cal/'direct_value_risk_contact_v48.json',cal/'direct_value_risk_near_v48.proposal_rows.jsonl',cal/'direct_value_risk_contact_v48.proposal_rows.jsonl']
  miss=[str(p) for p in req if not p.is_file() or p.stat().st_size==0]
  if miss:errors.append(f'{run.name}/{v}: missing/empty {miss}')
  metric=cal/'METRIC_CALIBRATION_CONTRACT.json'
  if metric.is_file():
   md=load(metric);sc=md.get('selection_contract') or {}
   if not (md.get('valid') and sc.get('mode')=='learned' and sc.get('mode_valid') and sc.get('threshold_valid') and sc.get('selection_semantics_valid') and sc.get('expected_selection_semantics')==RIFA and not md.get('test_roots_read')):errors.append(f'{run.name}/{v}: metric contract invalid')
  st=base/'V48_66_STAGE_I_STATE_ISOLATION.json'
  if st.is_file():
   sd=load(st)
   if not (sd.get('valid') and sd.get('semantic_witness_feature_contract_valid') and sd.get('factor_flags_valid') and sd.get('stage_i_bitwise_identity')):errors.append(f'{run.name}/{v}: state/feature isolation invalid')
 for p in top:
  if p.is_file():hashes[str(p)]=sha(p)

def main()->int:
 ap=argparse.ArgumentParser(description='v48.66 OC-ACRW attribution-ready engineering sentinel')
 ap.add_argument('--reference-contract',type=Path,required=True);ap.add_argument('--v65-complete',type=Path,required=True)
 ap.add_argument('--route-run',type=Path,required=True);ap.add_argument('--reentry-run',type=Path,required=True);ap.add_argument('--main-run',type=Path,required=True)
 ap.add_argument('--feasibility-audit',type=Path,required=True);ap.add_argument('--comparison',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();errors=[];hashes={}
 for p in (a.reference_contract,a.v65_complete,a.feasibility_audit,a.comparison):
  if not p.is_file():errors.append(f'missing {p}')
 if a.reference_contract.is_file() and not load(a.reference_contract).get('valid'):errors.append('reference contract invalid')
 if a.v65_complete.is_file():
  d=load(a.v65_complete)
  if not (d.get('valid') and d.get('attribution_ready') and d.get('engineering_version')=='v48.65.0-OC-CLRW' and not d.get('test_roots_read')):errors.append('V48.65 prerequisite invalid')
 check_run(a.route_run,True,False,errors,hashes);check_run(a.reentry_run,False,True,errors,hashes);check_run(a.main_run,True,True,errors,hashes)
 for p in (a.reference_contract,a.v65_complete,a.feasibility_audit,a.comparison):
  if p.is_file():hashes[str(p)]=sha(p)
 valid=not errors;doc={'schema':'ocrap-v48.66-acrw-pipeline-complete-v1','valid':valid,'attribution_ready':valid,'algorithm_version':'v48.66-DCP-DRFC-BCDE-RIFA-OC-ACRW','engineering_version':'v48.66.0-OC-ACRW','errors':errors,'artifact_sha256':hashes,
   'factorial_arms':{'N_ROUTE':str(a.route_run),'O_REENTRY':str(a.reentry_run),'P_Main_OCACRW':str(a.main_run),'historical_I':'v48.64 I_ACTIVESET','historical_M65':'v48.65 OC-CLRW'},
   'test_roots_read':False,'dataset_reconstruction':False}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');print(json.dumps({'event':'v48_66_acrw_pipeline_complete','valid':valid,'output':str(a.output)}));return 0 if valid else 30
if __name__=='__main__':raise SystemExit(main())
