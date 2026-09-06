#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
from typing import Any
from ocrap.v48_94_support_reserve_admission import ENGINEERING_VERSION
ROLES=("dev_near","dev_contact","certificate_near","certificate_contact")
VARIANTS=("balanced","precision")
def has_nc(roles): return any('near' in r for r in roles) and any('contact' in r for r in roles)
def main()->int:
 ap=argparse.ArgumentParser(); ap.add_argument('--audit',type=Path,required=True); ap.add_argument('--v93-comparison',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
 au=json.loads(a.audit.read_text()); c93=json.loads(a.v93_comparison.read_text()); d93=c93.get('preregistered_decision') or {}; errors=[]
 if not(au.get('valid') and au.get('attribution_ready')): errors.append('invalid V48.94 audit')
 if au.get('engineering_version')!=ENGINEERING_VERSION: errors.append('V48.94 engineering version mismatch')
 if not(c93.get('valid') and d93.get('status')=='PCD_FACTOR_COMPLEMENTARITY_GO'): errors.append('V48.93 complementarity GO prerequisite missing')
 if not all(au.get('cells',{}).get(v,{}).get(r,{}).get('proposal_set_identity') for v in VARIANTS for r in ROLES): errors.append('frozen top-K proposal identity failed')
 mode_roles=[r for r in ROLES if (au['mode_observability_by_role'][r].get('accuracy') is not None and float(au['mode_observability_by_role'][r]['accuracy'])>=.75 and int(au['mode_observability_by_role'][r].get('rows') or 0)>=5)]
 mode_go=len(mode_roles)>=3 and has_nc(mode_roles)
 cells=[(v,r,au['cells'][v][r]) for v in VARIANTS for r in ROLES]
 auc_pos=[(v,r) for v,r,x in cells if x.get('safe_vs_harm_auc_delta') is not None and float(x['safe_vs_harm_auc_delta'])>0]
 auc_mat=[(v,r) for v,r,x in cells if x.get('safe_vs_harm_auc_delta') is not None and float(x['safe_vs_harm_auc_delta'])>=.01]
 safe_nondec=[(v,r) for v,r,x in cells if x.get('safe_positive_pass_delta') is not None and float(x['safe_positive_pass_delta'])>=-1e-12]
 safe_mat=[(v,r) for v,r,x in cells if x.get('safe_positive_pass_delta') is not None and float(x['safe_positive_pass_delta'])>=.05]
 harm_ok=[(v,r) for v,r,x in cells if x.get('harmful_pass_srca') is not None and float(x['harmful_pass_srca'])<=.25 and float(x['harmful_pass_srca'])<=float(x['harmful_pass_native'])+.02+1e-12]
 ti_ok=[(v,r) for v,r,x in cells if x.get('teacher_infeasible_pass_srca') is not None and float(x['teacher_infeasible_pass_srca'])<=.25 and float(x['teacher_infeasible_pass_srca'])<=float(x['teacher_infeasible_pass_native'])+.02+1e-12]
 tf_nonreg=[(v,r) for v,r,x in cells if x.get('teacher_feasible_auc_delta') is not None and float(x['teacher_feasible_auc_delta'])>=-.01]
 # Use a strict source-freeze bar.  If this passes, there is no reason to keep iterating the absolute source internally.
 full_go=bool(mode_go and len(auc_pos)==8 and len(auc_mat)>=6 and len(safe_nondec)==8 and len(safe_mat)>=4 and has_nc([r for _,r in safe_mat]) and len(harm_ok)==8 and len(ti_ok)==8 and len(tf_nonreg)==8)
 mechanism_go=bool(mode_go and len(auc_pos)>=6 and len(auc_mat)>=4 and has_nc([r for _,r in auc_mat]) and len(harm_ok)==8 and len(ti_ok)==8 and len(tf_nonreg)==8)
 if full_go:
  status='SUPPORT_RESERVE_ABSOLUTE_SOURCE_GO'; nxt='freeze_absolute_source_then_verify_frozen_RIFA_and_run_paired_Safe_Near_Contact_closed_loop_then_external_baselines'
 elif mechanism_go:
  status='SUPPORT_RESERVE_COMPLEMENTARITY_MECHANISM_GO_SOURCE_STOP'; nxt='retain_parameter_free_support_state_switch_but_do_not_add_capacity_first_audit_why_safe_positive_admission_did_not_reach_freeze_bar'
 else:
  status='SUPPORT_RESERVE_COMPLEMENTARITY_STOP'; nxt='do_not_sweep_thresholds_or_capacity_audit_frozen_native_support_reserve_observability_before_any_new_absolute_source'
 dec={'mode_observability_roles':mode_roles,'mode_observability_go':mode_go,'safe_vs_harm_auc_positive_cells':auc_pos,'safe_vs_harm_auc_material_cells':auc_mat,'safe_positive_nondecrease_cells':safe_nondec,'safe_positive_material_cells':safe_mat,'harmful_selectivity_ok_cells':harm_ok,'teacher_infeasible_selectivity_ok_cells':ti_ok,'teacher_feasible_auc_nonregression_cells':tf_nonreg,'support_reserve_mechanism_go':mechanism_go,'full_source_go':full_go,'absolute_source_freeze_authorized':full_go,'boundary_transport_authorized':False,'regime_conditioned_policy_authorized':False,'dataset_reconstruction_authorized':False,'status':status,'next_branch':nxt}
 out={'schema':'ocrap-v48.94-srca-comparison-v1','engineering_version':ENGINEERING_VERSION,'valid':not errors,'attribution_ready':not errors,'errors':errors,'experiment_type':'fixed_zero_parameter_support_reserve_complementarity_absolute_source','planner_parameters_trained':0,'dataset_reconstruction':False,'dataset_reselection':False,'teacher_labels_changed':False,'teacher_metadata_input_to_model':False,'boundary_transport':False,'relative_ranker_modified':False,'regime_conditioning':False,'preregistered_decision':dec,'test_roots_read':False}
 a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print(json.dumps({'valid':out['valid'],'status':status,'mode_go':mode_go,'mechanism_go':mechanism_go,'full_source_go':full_go})); return 0 if out['valid'] else 30
if __name__=='__main__': raise SystemExit(main())
