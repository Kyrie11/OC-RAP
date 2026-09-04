#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math
from pathlib import Path

ROLES=('dev_near','certificate_near','dev_contact','certificate_contact')

def med(d,k):
 try:return float(d[k]['median'])
 except Exception:return float('nan')

def main()->int:
 ap=argparse.ArgumentParser();ap.add_argument('--summary',type=Path,required=True);ap.add_argument('--v48-90-summary',type=Path,required=True);ap.add_argument('--v48-90-comparison',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 s=json.loads(a.summary.read_text());s90=json.loads(a.v48_90_summary.read_text());c90=json.loads(a.v48_90_comparison.read_text());errors=[]
 if not(s.get('valid') and s.get('attribution_ready')):errors.append('V48.91 audit invalid')
 q90=c90.get('preregistered_decision') or {}
 if not(q90.get('exogenous_partition_transport_go') and q90.get('partition_stability_directional_relevance_go') and not q90.get('transport_physical_response_identifiability_go')):errors.append('V48.90 prerequisite branch mismatch')
 rr=s.get('roles',{});rr90=s90.get('roles',{})
 for role in ROLES:
  if role not in rr or role not in rr90:errors.append(f'missing role {role}')
 power={r:int(rr.get(r,{}).get('safe_positive_rows',0))>=10 for r in ROLES}
 coverage={r:med(rr.get(r,{}),'common_exogenous_tail_coverage')>=.80 for r in ROLES}
 sign={r:med(rr.get(r,{}),'response_sign_identifiable_mass')>=.50 for r in ROLES}
 info={r:med(rr.get(r,{}),'response_informative_mass')>=.50 for r in ROLES}
 auc={r:(rr.get(r,{}).get('response_safe_vs_harmful_auc') is not None and float(rr[r]['response_safe_vs_harmful_auc'])>=.60) for r in ROLES}
 macro={r:(rr.get(r,{}).get('response_macro_stratified_auc') is not None and float(rr[r]['response_macro_stratified_auc'])>=.58) for r in ROLES}
 top={r:(rr.get(r,{}).get('response_top1_lift') is not None and float(rr[r]['response_top1_lift'])>=.10) for r in ROLES}
 uplift={}
 for r in ROLES:
  old=med(rr90.get(r,{}),'exogenous_transport_sign_identifiable_mass');new=med(rr.get(r,{}),'response_sign_identifiable_mass');uplift[r]=bool(math.isfinite(old) and math.isfinite(new) and new-old>=.40)
 nsign=sum(sign.values());ninfo=sum(info.values());nauc=sum(auc.values());ntop=sum(top.values());nmacro=sum(macro.values());nuplift=sum(uplift.values())
 near_auc=auc['dev_near'] or auc['certificate_near']; contact_auc=auc['dev_contact'] or auc['certificate_contact']; near_top=top['dev_near'] or top['certificate_near'];contact_top=top['dev_contact'] or top['certificate_contact']
 response_go=bool(all(power.values()) and all(coverage.values()) and nsign>=3 and ninfo>=3 and nauc>=3 and near_auc and contact_auc and ntop>=2 and near_top and contact_top and nuplift>=3)
 # Macro AUC is reported as a robustness diagnostic but deliberately not a
 # second authorization path: physical response must satisfy the direct gate.
 status='COMMON_EXOGENOUS_PHYSICAL_RESPONSE_GO' if response_go else 'COMMON_EXOGENOUS_PHYSICAL_RESPONSE_STOP'
 next_branch=('authorize_v48_92_one_fixed_capacity_transport_coupled_signed_response_operator_no_capacity_sweep_no_boundary_transport' if response_go else 'close_root_local_response_learning_under_current_future_sidecars_retain_partition_stability_only_as_structural_scaffold_no_capacity_or_dataset_sweep')
 out={'schema':'ocrap-v48.91-cepmi-comparison-v1','engineering_version':'v48.91.0-OC-CEPMI','valid':not errors,'attribution_ready':not errors,'errors':errors,'experiment_type':'audit_only_common_exogenous_future_level_physical_margin_identifiability','preregistered_decision':{'v48_90_transport_prerequisite':not bool(errors),'safe_positive_power_gate':power,'common_exogenous_tail_coverage_gate':coverage,'response_sign_identifiable_mass_gate':sign,'response_informative_mass_gate':info,'response_auc_gate':auc,'response_macro_auc_diagnostic':macro,'response_top1_lift_gate':top,'sign_identifiability_uplift_vs_v48_90_root_interval_gate':uplift,'future_level_physical_response_go':response_go,'source_training_authorized':response_go,'status':status,'next_branch':next_branch},'planner_parameters_trained':0,'teacher_labels_changed':False,'teacher_metadata_input_to_model':False,'dataset_reconstruction':False,'dataset_reselection':False,'boundary_transport':False,'regime_conditioning':False,'relative_ranker_modified':False,'test_roots_read':False}
 a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps(out['preregistered_decision'],sort_keys=True));return 0 if not errors else 30
if __name__=='__main__':raise SystemExit(main())
