#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROLES=("dev_near","certificate_near","dev_contact","certificate_contact")
NEAR=("dev_near","certificate_near")
CONTACT=("dev_contact","certificate_contact")


def _v(d: dict[str,Any], path: str, default=None):
    cur: Any=d
    for p in path.split('.'):
        if not isinstance(cur,dict) or p not in cur: return default
        cur=cur[p]
    return cur


def _ge(role: dict[str,Any], path: str, thr: float) -> bool:
    x=_v(role,path)
    return x is not None and float(x)>=thr


def _le(role: dict[str,Any], path: str, thr: float) -> bool:
    x=_v(role,path)
    return x is not None and float(x)<=thr


def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--audit-summary',type=Path,required=True)
    ap.add_argument('--v48-89-comparison',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    a=json.loads(args.audit_summary.read_text())
    prev=json.loads(args.v48_89_comparison.read_text())
    errors=[]
    if not a.get('valid') or not a.get('attribution_ready'): errors.append('V48.90 audit invalid')
    pp=prev.get('preregistered_decision') or {}
    if not(prev.get('valid') and pp.get('status')=='COUNTERFACTUAL_ROOT_CORRESPONDENCE_STOP' and not pp.get('root_correspondence_go') and 'partition_stability' in str(pp.get('next_branch',''))):
        errors.append('V48.89 STOP/partition-stability prerequisite missing')
    roles=a.get('roles') or {}
    if [r for r in ROLES if r not in roles]: errors.append('missing V48.90 roles')

    power={r:int(_v(roles.get(r,{}),'labeled_rows',0) or 0)>=100 for r in ROLES}
    safe_power={r:int(_v(roles.get(r,{}),'safe_positive_rows',0) or 0)>=10 for r in ROLES}

    # Q90: exchangeability quotient.  This specifically tests whether V48.89's
    # weak occurrence identities were an over-granular branch-instance contract.
    q_unresolved={r:_le(roles.get(r,{}),'recipe_unresolved_semantic_mass_candidate.q90',0.05) and _le(roles.get(r,{}),'recipe_unresolved_semantic_mass_nominal.q90',0.05) for r in ROLES}
    q_homogeneous={r:_ge(roles.get(r,{}),'duplicate_root_homogeneity_mass_candidate.q10',0.99) and _ge(roles.get(r,{}),'duplicate_root_homogeneity_mass_nominal.q10',0.99) for r in ROLES}
    q_shared={r:_ge(roles.get(r,{}),'recipe_shared_mass_candidate.median',0.95) and _ge(roles.get(r,{}),'recipe_shared_mass_nominal.median',0.95) for r in ROLES}
    q_tail_cov={r:_ge(roles.get(r,{}),'recipe_tail_transport_coverage.median',0.90) for r in ROLES}
    q_tail_stability={r:_ge(roles.get(r,{}),'recipe_tail_partition_stability.median',0.85) for r in ROLES}
    recipe_quotient_go=bool(not errors and all(power.values()) and all(q_unresolved.values()) and all(q_homogeneous.values()) and all(q_shared.values()) and all(q_tail_cov.values()) and sum(q_tail_stability.values())>=3)

    # T90/Main: require candidate-independent external realization where the
    # generator actually instantiates stochastic/augmented futures.
    t_unresolved={r:_le(roles.get(r,{}),'exogenous_unresolved_mass_candidate.q90',0.10) and _le(roles.get(r,{}),'exogenous_unresolved_mass_nominal.q90',0.10) for r in ROLES}
    t_shared={r:_ge(roles.get(r,{}),'exogenous_shared_mass_candidate.median',0.80) and _ge(roles.get(r,{}),'exogenous_shared_mass_nominal.median',0.80) for r in ROLES}
    t_cov={r:_ge(roles.get(r,{}),'exogenous_tail_transport_coverage.median',0.80) for r in ROLES}
    t_purity={r:_ge(roles.get(r,{}),'exogenous_tail_transport_purity.median',0.90) for r in ROLES}
    t_stability={r:_ge(roles.get(r,{}),'exogenous_tail_partition_stability.median',0.75) for r in ROLES}
    exogenous_transport_go=bool(recipe_quotient_go and all(t_unresolved.values()) and all(t_shared.values()) and all(t_cov.values()) and all(t_purity.values()) and all(t_stability.values()))

    # Pre-registered directional relevance is label-side only and cannot by
    # itself authorize a source.  Macro-stratified AUC prevents macro-type base
    # rates from explaining the signal.
    part_auc={r:_ge(roles.get(r,{}),'partition_stability_safe_vs_harmful_auc',0.65) for r in ROLES}
    part_macro_auc={r:_ge(roles.get(r,{}),'partition_stability_macro_stratified_auc',0.62) for r in ROLES}
    part_mean={r:(_v(roles.get(r,{}),'partition_stability_safe_positive_mean') is not None and _v(roles.get(r,{}),'partition_stability_harmful_mean') is not None and float(_v(roles.get(r,{}),'partition_stability_safe_positive_mean'))>float(_v(roles.get(r,{}),'partition_stability_harmful_mean'))) for r in ROLES}
    partition_directional_go=bool(
        all(safe_power.values()) and sum(part_auc.values())>=3 and any(part_auc[r] for r in NEAR) and any(part_auc[r] for r in CONTACT)
        and sum(part_macro_auc.values())>=3 and any(part_macro_auc[r] for r in NEAR) and any(part_macro_auc[r] for r in CONTACT)
        and sum(part_mean.values())>=3
    )

    sign_mass={r:_ge(roles.get(r,{}),'exogenous_transport_sign_identifiable_mass.median',0.50) for r in ROLES}
    informative_mass={r:_ge(roles.get(r,{}),'exogenous_transport_informative_response_mass.median',0.50) for r in ROLES}
    resp_auc={r:_ge(roles.get(r,{}),'response_safe_vs_harmful_auc',0.60) for r in ROLES}
    resp_top1={r:_ge(roles.get(r,{}),'response_top1_lift',0.10) for r in ROLES}
    response_go=bool(
        exogenous_transport_go and all(safe_power.values()) and sum(sign_mass.values())>=3 and sum(informative_mass.values())>=3
        and sum(resp_auc.values())>=3 and any(resp_auc[r] for r in NEAR) and any(resp_auc[r] for r in CONTACT)
        and sum(resp_top1.values())>=2 and any(resp_top1[r] for r in NEAR) and any(resp_top1[r] for r in CONTACT)
    )
    training_authorized=bool(exogenous_transport_go and response_go)

    if training_authorized:
        status='COUNTERFACTUAL_PARTITION_TRANSPORT_RESPONSE_GO'
        next_branch='train_one_fixed_capacity_transport_coupled_signed_response_operator_no_boundary_transport_no_capacity_sweep'
    elif exogenous_transport_go:
        status='PARTITION_TRANSPORT_GO_PHYSICAL_RESPONSE_UNDERIDENTIFIED'
        if partition_directional_go:
            next_branch='retain_partition_stability_as_structural_rejector_scaffold_only_then_build_non_input_common_exogenous_future_physical_margin_sidecar_same_cohort_no_dataset_reselection_no_source_training'
        else:
            next_branch='do_not_train_new_source_then_build_non_input_common_exogenous_future_physical_margin_sidecar_same_cohort_no_dataset_reselection_no_capacity_sweep'
    elif recipe_quotient_go:
        status='RECIPE_QUOTIENT_GO_EXOGENOUS_TRANSPORT_STOP'
        next_branch='do_not_train_new_source_then_audit_common_randomness_exogenous_realization_provenance_same_dataset_no_encoder_or_adapter_sweep'
    else:
        status='COUNTERFACTUAL_EQUIVALENCE_QUOTIENT_STOP'
        next_branch='close_root_local_causal_response_branch_under_current_counterfactual_sidecars_no_capacity_or_dataset_sweep'

    decision={
        'status':status,'next_branch':next_branch,
        'recipe_equivalence_quotient_go':recipe_quotient_go,
        'exogenous_partition_transport_go':exogenous_transport_go,
        'partition_stability_directional_relevance_go':partition_directional_go,
        'transport_physical_response_identifiability_go':response_go,
        'matched_transport_response_training_authorized':training_authorized,
        'powered_roles':power,'safe_positive_power_gate':safe_power,
        'recipe_unresolved_gate':q_unresolved,'duplicate_root_homogeneity_gate':q_homogeneous,'recipe_shared_mass_gate':q_shared,
        'recipe_tail_coverage_gate':q_tail_cov,'recipe_tail_stability_gate':q_tail_stability,
        'exogenous_unresolved_gate':t_unresolved,'exogenous_shared_mass_gate':t_shared,'exogenous_tail_coverage_gate':t_cov,
        'exogenous_tail_purity_gate':t_purity,'exogenous_tail_stability_gate':t_stability,
        'partition_auc_gate':part_auc,'partition_macro_stratified_auc_gate':part_macro_auc,'partition_mean_order_gate':part_mean,
        'response_sign_identifiable_mass_gate':sign_mass,'response_informative_mass_gate':informative_mass,
        'response_auc_gate':resp_auc,'response_top1_lift_gate':resp_top1,
    }
    doc={
        'schema':'ocrap-v48.90-oc-cept-comparison-v1','engineering_version':'v48.90.0-OC-CEPT',
        'valid':not errors,'attribution_ready':not errors,'errors':errors,'preregistered_decision':decision,
        'scientific_contract':{
            'experiment_type':'audit_only_counterfactual_equivalence_partition_transport_adjudication',
            'planner_parameters_trained':0,'teacher_labels_changed':False,'teacher_metadata_input_to_model':False,
            'dataset_reconstruction':False,'regime_conditioning':False,'boundary_transport':'OFF','relative_ranker_modified':False,
            'root_slot_identity_assumed':False,'individual_duplicate_branch_identity_assumed':False,
            'recipe_quotient':'candidate-independent source + branch-defining metadata; no occurrence suffix',
            'exogenous_realization_contract':'augmented/stochastic branches match only when stored realization fingerprint agrees',
            'root_correspondence':'probability-mass coupling between candidate/nominal root partitions induced by shared counterfactual classes',
            'physical_response':'conservative interval union over transport-supported nominal roots on exact nested candidate tail',
            'capacity_sweep':False,
        },
        'audit_summary':a,
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps({'valid':doc['valid'],'status':status,'training_authorized':training_authorized}))
    return 0 if doc['valid'] else 30

if __name__=='__main__': raise SystemExit(main())
