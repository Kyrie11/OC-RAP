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
def check_run(run,response,errors,hashes):
 vi=run/'V48_73_VARIANT_ISOLATION.json';factor=run/'V48_73_FACTOR_CONTRACT.json';terminal=run/'dedicated_recalibration_status.json';top=[vi,factor,terminal]
 for p in top:
  if not p.is_file():errors.append(f'missing {p}')
 if vi.is_file():
  d=load(vi)
  if not d.get('valid'):errors.append(f'{run.name}: variant isolation invalid')
  if not (d.get('control_projection') is True and d.get('boundary_transport') is False and d.get('projection_fidelity') is True and d.get('demand_normalized_fidelity') is False and d.get('robust_occupancy') is False and d.get('soft_occupancy_disagreement') is False and d.get('boundary_localized_occupancy_trust') is False and d.get('history_occupancy_reachability') is False and d.get('interaction_box_support') is True and d.get('interaction_hull_support') is True and d.get('interaction_anchor_support') is True and d.get('interaction_response_support') is response):errors.append(f'{run.name}: flags mismatch')
 if factor.is_file():
  d=load(factor);req={'trainable_parameters':2,'threshold':.5,'threshold_search':False,'regime_id_input':False,'proposal_top_k':5,'proposal_expansion':False,'test_roots_read':False,'semantic_witness_feature_schema':9,'semantic_witness_feature_source':'interaction_response_history_reachability_projected_recovery_witness','active_set_alignment':True,'path_stop_alignment':False,'classlocal_transport':False,'route_alignment':True,'reentry_alignment':True,'control_projection':True,'boundary_transport':False,'projection_fidelity':True,'demand_normalized_fidelity':False,'robust_occupancy':False,'soft_occupancy_disagreement':False,'boundary_localized_occupancy_trust':False,'history_occupancy_reachability':False,'interaction_box_support':True,'interaction_hull_support':True,'interaction_anchor_support':True,'interaction_response_support':response,'relative_score_intervention':False,'teacher_future_input':False,'dataset_reconstruction':False}
  for k,v in req.items():
   if d.get(k)!=v:errors.append(f'{run.name}: factor {k} mismatch {d.get(k)!r}!={v!r}')
 for v in ('balanced','precision'):
  base=run/'candidates'/v;cal=base/'calibration';req=[base/'model_v48_trac_sr'/'best.pt',base/'model_v48_trac_sr'/'train_summary.json',base/'TRAINING_COMPLETE.json',base/'EVIDENCE_CORRECTION_COMPLETE.json',base/'V48_73_STAGE_I_STATE_ISOLATION.json',base/'POLICY_CONTRACT.env',cal/'METRIC_CALIBRATION_CONTRACT.json',cal/'dev_diagnostic_near_v48.proposal_rows.jsonl',cal/'dev_diagnostic_contact_v48.proposal_rows.jsonl',cal/'direct_value_risk_near_v48.proposal_rows.jsonl',cal/'direct_value_risk_contact_v48.proposal_rows.jsonl']
  miss=[str(p) for p in req if not p.is_file() or p.stat().st_size==0]
  if miss:errors.append(f'{run.name}/{v}: missing/empty {miss}')
  if (cal/'METRIC_CALIBRATION_CONTRACT.json').is_file():
   md=load(cal/'METRIC_CALIBRATION_CONTRACT.json');sc=md.get('selection_contract') or {}
   if not (md.get('valid') and sc.get('mode')=='learned' and sc.get('threshold_valid') and sc.get('selection_semantics_valid') and sc.get('expected_selection_semantics')==RIFA and not md.get('test_roots_read')):errors.append(f'{run.name}/{v}: metric contract invalid')
  if (base/'V48_73_STAGE_I_STATE_ISOLATION.json').is_file():
   st=load(base/'V48_73_STAGE_I_STATE_ISOLATION.json')
   if not (st.get('valid') and st.get('stage_i_bitwise_identity') and st.get('semantic_witness_feature_contract_valid') and st.get('factor_flags_valid')):errors.append(f'{run.name}/{v}: state isolation invalid')
 for p in top:
  if p.is_file():hashes[str(p)]=sha(p)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--reference-contract',type=Path,required=True);ap.add_argument('--v72-complete',type=Path,required=True);ap.add_argument('--v72-comparison',type=Path,required=True);ap.add_argument('--runtime-contract',type=Path,required=True);ap.add_argument('--anchor-run',type=Path,required=True);ap.add_argument('--main-run',type=Path,required=True);ap.add_argument('--feasibility-audit',type=Path,required=True);ap.add_argument('--response-audit',type=Path,required=True);ap.add_argument('--truth-strata',type=Path,required=True);ap.add_argument('--comparison',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();errors=[];hashes={}
 for p in (a.reference_contract,a.v72_complete,a.v72_comparison,a.runtime_contract,a.feasibility_audit,a.response_audit,a.truth_strata,a.comparison):
  if not p.is_file():errors.append(f'missing {p}')
 if a.reference_contract.is_file() and not load(a.reference_contract).get('valid'):errors.append('reference contract invalid')
 if a.runtime_contract.is_file() and not load(a.runtime_contract).get('valid'):errors.append('runtime code contract invalid')
 if a.v72_complete.is_file():
  d=load(a.v72_complete)
  if not (d.get('valid') and d.get('attribution_ready') and d.get('engineering_version')=='v48.72.0-OC-IORW' and not d.get('test_roots_read')):errors.append('V48.72 prerequisite invalid')
 if a.v72_comparison.is_file():
  pr=load(a.v72_comparison).get('preregistered_decision') or {}
  if not (pr.get('status')=='STOP' and pr.get('interaction_oriented_reachability_go') is False and pr.get('next_branch')=='retain_supported_directional_set_then_interaction_response_dynamics'):errors.append('V48.72 scientific branch mismatch')
 check_run(a.anchor_run,False,errors,hashes);check_run(a.main_run,True,errors,hashes)
 for p in (a.reference_contract,a.v72_complete,a.v72_comparison,a.runtime_contract,a.feasibility_audit,a.response_audit,a.truth_strata,a.comparison):
  if p.is_file():hashes[str(p)]=sha(p)
 valid=not errors;doc={'schema':'ocrap-v48.73-irrw-pipeline-complete-v1','valid':valid,'attribution_ready':valid,'algorithm_version':'v48.73-DCP-DRFC-BCDE-RIFA-OC-IRRW','engineering_version':'v48.73.0-OC-IRRW','errors':errors,'artifact_sha256':hashes,'arms':{'historical_M72':'static empirical acceleration hull + directional support','N73_ANCHORED_HULL':str(a.anchor_run),'O73_Main_OCIRRW':str(a.main_run)},'test_roots_read':False,'dataset_reconstruction':False};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');print(json.dumps({'event':'v48_73_irrw_pipeline_complete','valid':valid,'output':str(a.output)}));return 0 if valid else 30
if __name__=='__main__':raise SystemExit(main())
