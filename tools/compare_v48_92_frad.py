#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
from typing import Any
from ocrap.v48_92_factorized_recovery_advantage import ENGINEERING_VERSION

ROLES=("dev_near","dev_contact","certificate_near","certificate_contact")
MEDIATORS=("structural_response_score","shapley_drs","shapley_deployability_gate","shapley_gap_discount")


def _regime_coverage(roles:list[str])->bool:
    return any("near" in r for r in roles) and any("contact" in r for r in roles)


def _mediator_gate(summary:dict[str,Any],field:str)->dict[str,Any]:
    auc_roles=[]; macro_roles=[]; top_roles=[]
    per={}
    for role in ROLES:
        s=summary["roles"][role]["scores"][field]
        auc=s.get("safe_vs_harmful_auc"); macro=s.get("macro_stratified_auc"); lift=s.get("top1_lift")
        per[role]={"auc":auc,"macro_auc":macro,"top1_lift":lift}
        if auc is not None and float(auc)>=.65: auc_roles.append(role)
        if macro is not None and float(macro)>=.62: macro_roles.append(role)
        if lift is not None and float(lift)>=.10: top_roles.append(role)
    auc_go=len(auc_roles)>=3 and _regime_coverage(auc_roles)
    macro_go=len(macro_roles)>=3 and _regime_coverage(macro_roles)
    top_go=len(top_roles)>=2 and _regime_coverage(top_roles)
    return {"per_role":per,"auc_roles":auc_roles,"macro_roles":macro_roles,"top1_roles":top_roles,
            "auc_gate":auc_go,"macro_gate":macro_go,"top1_gate":top_go,"go":bool(auc_go and macro_go and top_go)}


def main()->int:
    ap=argparse.ArgumentParser();ap.add_argument('--summary',type=Path,required=True);ap.add_argument('--v48-91-comparison',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    s=json.loads(args.summary.read_text());v91=json.loads(args.v48_91_comparison.read_text())
    errors=[]
    if not(s.get('valid') and s.get('attribution_ready')):errors.append('invalid V48.92 summary')
    if str(s.get('engineering_version'))!=ENGINEERING_VERSION:errors.append('V48.92 engineering version mismatch')
    q91=v91.get('preregistered_decision') or {}
    if not(v91.get('valid') and q91.get('status')=='COMMON_EXOGENOUS_PHYSICAL_RESPONSE_STOP' and not q91.get('source_training_authorized')):
        errors.append('V48.91 STOP prerequisite missing')
    if float(s.get('max_pcd_reconstruction_error',1.0))>2e-6:errors.append('PCD reconstruction identity failed')
    if float(s.get('max_shapley_sum_error',1.0))>1e-10:errors.append('Shapley additivity identity failed')
    if float(s.get('max_v48_91_physical_response_identity_error',1.0))>1e-12:errors.append('V48.91 physical response identity failed')
    power={role:int(s['roles'][role]['safe_positive_rows'])>=10 for role in ROLES}
    mediator={m:_mediator_gate(s,m) for m in MEDIATORS}
    physical=_mediator_gate(s,'physical_response_score')
    partition=_mediator_gate(s,'partition_stability')
    winners=[m for m,g in mediator.items() if g['go']]
    shared_go=bool(winners)
    # The V48.91 physical-response family is closed by preregistration regardless of a later
    # descriptive score.  V48.92 only adjudicates which existing decision semantic deserves
    # the next *separate* experiment; it never authorizes a root-local physical-response head.
    if shared_go:
        next_branch='shared_decision_mediator_identified_then_design_one_fixed_capacity_mediator_specific_experiment_no_physical_response_reopen'
        status='SHARED_RECOVERY_ADVANTAGE_MEDIATOR_GO'
    else:
        next_branch='factorized_recovery_advantage_not_reducible_to_one_shared_mediator_then_audit_stage_i_teacher_action_benefit_semantics_no_capacity_or_dataset_sweep'
        status='FACTORIZED_RECOVERY_ADVANTAGE_STOP'
    decision={
      'safe_positive_power_gate':power,
      'pcd_identity_gate':not errors,
      'physical_response_family_closed':True,
      'partition_stability_scaffold_retain':bool(partition['auc_gate'] and partition['macro_gate']),
      'mediator_gates':mediator,
      'physical_response_diagnostic':physical,
      'shared_mediator_winners':winners,
      'shared_decision_mediator_go':shared_go,
      'source_training_authorized':False,
      'boundary_transport_authorized':False,
      'regime_conditioned_policy_authorized':False,
      'dataset_reconstruction_authorized':False,
      'next_branch':next_branch,
      'status':status,
    }
    out={'schema':'ocrap-v48.92-frad-comparison-v1','engineering_version':ENGINEERING_VERSION,'valid':not errors,'attribution_ready':not errors,'errors':errors,
         'experiment_type':'audit_only_factorized_recovery_advantage_mediation','planner_parameters_trained':0,'dataset_reconstruction':False,'dataset_reselection':False,
         'regime_conditioning':False,'boundary_transport':False,'relative_ranker_modified':False,'teacher_labels_changed':False,'teacher_metadata_input_to_model':False,
         'womd_replay_performed':False,'preregistered_decision':decision,'test_roots_read':False}
    args.output.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps({'valid':out['valid'],'status':status,'winners':winners}));return 0 if out['valid'] else 30
if __name__=='__main__':raise SystemExit(main())
