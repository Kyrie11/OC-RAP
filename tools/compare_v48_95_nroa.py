#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path

from ocrap.v48_95_native_recovery_observability import ENGINEERING_VERSION

ROLES=("dev_near","dev_contact","certificate_near","certificate_contact")
VARIANTS=("balanced","precision")


def pass_auc(a: object, threshold: float) -> bool:
    return a is not None and float(a) >= threshold


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--audit',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    x=json.loads(a.audit.read_text())
    errors=[]
    if not bool(x.get('valid')): errors.append('V48.95 audit invalid')
    if str(x.get('engineering_version')) != ENGINEERING_VERSION: errors.append('engineering version mismatch')

    # The channels are preregistered by semantics, not selected per split after seeing results.
    state_channel='nominal_smooth_support'
    support_channel='delta_smooth_support'
    reserve_channel='delta_deployability'

    state_cells=[]; support_cells=[]; reserve_cells=[]
    state_roles=set(); support_roles=set(); reserve_roles=set()
    for v in VARIANTS:
        for role in ROLES:
            c=x.get('cells',{}).get(v,{}).get(role,{})
            sa=c.get('state_auc',{}).get(state_channel)
            ua=c.get('support_action_auc',{}).get(support_channel)
            ra=c.get('reserve_action_auc',{}).get(reserve_channel)
            if pass_auc(sa,0.70): state_cells.append([v,role]); state_roles.add(role)
            if pass_auc(ua,0.65): support_cells.append([v,role]); support_roles.add(role)
            if pass_auc(ra,0.65): reserve_cells.append([v,role]); reserve_roles.add(role)

    def cross_regime(role_set:set[str], min_roles:int)->bool:
        return len(role_set)>=min_roles and any('near' in r for r in role_set) and any('contact' in r for r in role_set)

    state_go = cross_regime(state_roles,3) and len(state_cells)>=6
    support_go = cross_regime(support_roles,3) and len(support_cells)>=6
    reserve_go = cross_regime(reserve_roles,3) and len(reserve_cells)>=6
    full_go = state_go and support_go and reserve_go

    decision={
      'state_observability_go':state_go,
      'support_action_observability_go':support_go,
      'reserve_action_observability_go':reserve_go,
      'native_recovery_observability_go':full_go,
      'state_channel':state_channel,
      'support_channel':support_channel,
      'reserve_channel':reserve_channel,
      'state_positive_cells':state_cells,
      'support_positive_cells':support_cells,
      'reserve_positive_cells':reserve_cells,
      'state_roles':sorted(state_roles),'support_roles':sorted(support_roles),'reserve_roles':sorted(reserve_roles),
      'absolute_source_training_authorized':False,
      'boundary_transport_authorized':False,
      'dataset_reconstruction_authorized':False,
      'regime_conditioned_policy_authorized':False,
    }
    if full_go:
        decision['status']='FROZEN_NATIVE_RECOVERY_OBSERVABILITY_GO'
        decision['next_branch']='native_certificate_contains_cross_regime_support_reserve_information_then_design_one_fixed_capacity_observation_aligned_source_no_threshold_or_capacity_sweep'
    else:
        decision['status']='FROZEN_NATIVE_RECOVERY_OBSERVABILITY_STOP'
        decision['next_branch']='close_native_certificate_support_reserve_realization_then_audit_stage_i_root_action_observability_no_source_capacity_sweep'
    out={
      'schema':'ocrap-v48.95-nroa-comparison-v1','engineering_version':ENGINEERING_VERSION,
      'valid':not errors,'attribution_ready':not errors,'errors':errors,
      'experiment_type':'audit_only_frozen_native_support_reserve_observability',
      'planner_parameters_trained':0,'dataset_reconstruction':False,'dataset_reselection':False,
      'teacher_labels_changed':False,'teacher_metadata_input_to_model':False,'boundary_transport':False,
      'relative_ranker_modified':False,'regime_conditioning':False,'womd_replay_performed':False,
      'preregistered_decision':decision,'test_roots_read':False,
    }
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'valid':out['valid'],'status':decision['status'],'state':state_go,'support':support_go,'reserve':reserve_go}))
    return 0 if out['valid'] else 30
if __name__=='__main__': raise SystemExit(main())
