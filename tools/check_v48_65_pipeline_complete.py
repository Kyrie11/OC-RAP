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
def check_run(run:Path,active:bool,errors:list[str],hashes:dict):
 vi=run/'V48_65_VARIANT_ISOLATION.json';factor=run/'V48_65_FACTOR_CONTRACT.json';terminal=run/'dedicated_recalibration_status.json';top=[vi,factor,terminal]
 for p in top:
  if not p.is_file():errors.append(f'missing {p}')
 if vi.is_file():
  d=load(vi)
  if not d.get('valid'):errors.append(f'{run.name}: variant isolation invalid')
  if bool(d.get('active_set_alignment'))!=active or bool(d.get('path_stop_alignment')) or not bool(d.get('classlocal_transport')):errors.append(f'{run.name}: variant factor flags mismatch')
 if factor.is_file():
  d=load(factor);req={'trainable_parameters':2,'threshold':.5,'threshold_search':False,'regime_id_input':False,'proposal_top_k':5,'proposal_expansion':False,'test_roots_read':False,'semantic_witness_feature_schema':1,'semantic_witness_feature_source':'semantics_aligned_common_executable_recovery_witness','teacher_future_input':False,'relative_score_intervention':False,'active_set_alignment':active,'path_stop_alignment':False,'classlocal_transport':True,'correction_locus':'OC-MERO q[i,l] after compatible-root aggregation and before per-observation-class max'}
  for k,v in req.items():
   if d.get(k)!=v:errors.append(f'{run.name}: factor {k}={d.get(k)!r} expected {v!r}')
 for v in ('balanced','precision'):
  base=run/'candidates'/v;cal=base/'calibration';req=[base/'model_v48_trac_sr'/'best.pt',base/'model_v48_trac_sr'/'train_summary.json',base/'TRAINING_COMPLETE.json',base/'EVIDENCE_CORRECTION_COMPLETE.json',base/'V48_65_STAGE_I_STATE_ISOLATION.json',base/'POLICY_CONTRACT.env',cal/'METRIC_CALIBRATION_CONTRACT.json',cal/'dev_diagnostic_near_v48.json',cal/'dev_diagnostic_contact_v48.json',cal/'dev_diagnostic_near_v48.proposal_rows.jsonl',cal/'dev_diagnostic_contact_v48.proposal_rows.jsonl',cal/'direct_value_risk_near_v48.json',cal/'direct_value_risk_contact_v48.json',cal/'direct_value_risk_near_v48.proposal_rows.jsonl',cal/'direct_value_risk_contact_v48.proposal_rows.jsonl']
  miss=[str(p) for p in req if not p.is_file() or p.stat().st_size==0]
  if miss:errors.append(f'{run.name}/{v}: missing/empty {miss}')
  metric=cal/'METRIC_CALIBRATION_CONTRACT.json'
  if metric.is_file():
   md=load(metric);sc=md.get('selection_contract') or {}
   if not (md.get('valid') and sc.get('mode')=='learned' and sc.get('mode_valid') and sc.get('threshold_valid') and sc.get('selection_semantics_valid') and sc.get('expected_selection_semantics')==RIFA and not md.get('test_roots_read')):errors.append(f'{run.name}/{v}: metric contract invalid')
  st=base/'V48_65_STAGE_I_STATE_ISOLATION.json'
  if st.is_file():
   sd=load(st)
   if not (sd.get('valid') and sd.get('semantic_witness_feature_contract_valid') and sd.get('factor_flags_valid') and sd.get('stage_i_bitwise_identity')):errors.append(f'{run.name}/{v}: state/feature isolation invalid')
 for p in top:
  if p.is_file():hashes[str(p)]=sha(p)
def main():
 ap=argparse.ArgumentParser(description='v48.65 OC-CLRW attribution-ready engineering sentinel')
 ap.add_argument('--reference-contract',type=Path,required=True);ap.add_argument('--v64-complete',type=Path,required=True);ap.add_argument('--teacher-semantics-audit',type=Path,required=True);ap.add_argument('--classlocal-run',type=Path,required=True);ap.add_argument('--main-run',type=Path,required=True);ap.add_argument('--feasibility-audit',type=Path,required=True);ap.add_argument('--comparison',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();errors=[];hashes={}
 for p in (a.reference_contract,a.v64_complete,a.teacher_semantics_audit,a.feasibility_audit,a.comparison):
  if not p.is_file():errors.append(f'missing {p}')
 if a.reference_contract.is_file() and not load(a.reference_contract).get('valid'):errors.append('reference contract invalid')
 if a.v64_complete.is_file():
  d=load(a.v64_complete)
  if not (d.get('valid') and d.get('attribution_ready') and d.get('engineering_version')=='v48.64.1-OC-SARW-ENGFIX' and not d.get('test_roots_read')):errors.append('V48.64.1 prerequisite invalid')
 if a.teacher_semantics_audit.is_file():
  d=load(a.teacher_semantics_audit)
  if not (d.get('valid') and d.get('read_only_existing_dataset') is True and d.get('dataset_reconstruction') is False and d.get('test_roots_read') is False):errors.append('teacher certificate semantics audit invalid')
 check_run(a.classlocal_run,False,errors,hashes);check_run(a.main_run,True,errors,hashes)
 for p in (a.reference_contract,a.v64_complete,a.teacher_semantics_audit,a.feasibility_audit,a.comparison):
  if p.is_file():hashes[str(p)]=sha(p)
 valid=not errors;doc={'schema':'ocrap-v48.65-clrw-pipeline-complete-v1','valid':valid,'attribution_ready':valid,'algorithm_version':'v48.65-DCP-DRFC-BCDE-RIFA-OC-CLRW','engineering_version':'v48.65.0-OC-CLRW','errors':errors,'artifact_sha256':hashes,'factorial_arms':{'L_CLASSLOCAL':str(a.classlocal_run),'M_Main_OCCLRW':str(a.main_run),'historical_H':'v48.63 OC-QARW','historical_I':'v48.64 I_ACTIVESET'},'test_roots_read':False,'dataset_reconstruction':False}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');print(json.dumps({'event':'v48_65_clrw_pipeline_complete','valid':valid,'output':str(a.output)}));return 0 if valid else 30
if __name__=='__main__':raise SystemExit(main())
