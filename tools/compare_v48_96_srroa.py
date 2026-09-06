#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
from ocrap.v48_96_support_reserve_root_observability import ENGINEERING_VERSION

ROLES=("dev_near","dev_contact","certificate_near","certificate_contact")
VARIANTS=("balanced","precision")

def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument('--balanced',type=Path,required=True); ap.add_argument('--precision',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    docs={'balanced':json.loads(a.balanced.read_text()),'precision':json.loads(a.precision.read_text())}; errors=[]
    for v,d in docs.items():
        if not d.get('valid'): errors.append(f'{v} probe invalid')
        if d.get('engineering_version')!=ENGINEERING_VERSION: errors.append(f'{v} version mismatch')
    state_cells=[];support_cells=[];reserve_cells=[];support_top=[];reserve_top=[]
    state_roles=set();support_roles=set();reserve_roles=set();support_top_roles=set();reserve_top_roles=set()
    cells={}
    for v,d in docs.items():
        cells[v]={}
        for role in ROLES:
            c=d.get('cells',{}).get(role,{})
            s=c.get('state',{}); u=c.get('support_true',{}); r=c.get('reserve_true',{})
            cells[v][role]=c
            sa=s.get('auc'); ua=u.get('auc'); ud=u.get('auc_vs_shuffled'); ut=u.get('top1_vs_shuffled'); ra=r.get('auc'); rd=r.get('auc_vs_shuffled'); rt=r.get('top1_vs_shuffled')
            if sa is not None and float(sa)>=0.70:
                state_cells.append([v,role]);state_roles.add(role)
            if ua is not None and ud is not None and float(ua)>=0.65 and float(ud)>=0.05:
                support_cells.append([v,role]);support_roles.add(role)
            if ra is not None and rd is not None and float(ra)>=0.65 and float(rd)>=0.05:
                reserve_cells.append([v,role]);reserve_roles.add(role)
            if ut is not None and float(ut)>=0.10:
                support_top.append([v,role]);support_top_roles.add(role)
            if rt is not None and float(rt)>=0.10:
                reserve_top.append([v,role]);reserve_top_roles.add(role)
    def cross(rs:set[str],n:int)->bool:
        return len(rs)>=n and any('near' in x for x in rs) and any('contact' in x for x in rs)
    state_go=len(state_cells)>=6 and cross(state_roles,3)
    support_go=len(support_cells)>=6 and cross(support_roles,3) and len(support_top)>=4 and cross(support_top_roles,2)
    reserve_go=len(reserve_cells)>=6 and cross(reserve_roles,3) and len(reserve_top)>=4 and cross(reserve_top_roles,2)
    full_go=state_go and support_go and reserve_go
    decision={
      'state_root_observability_go':state_go,'support_root_action_observability_go':support_go,'reserve_root_action_observability_go':reserve_go,
      'frozen_root_support_reserve_observability_go':full_go,
      'state_positive_cells':state_cells,'support_positive_cells':support_cells,'reserve_positive_cells':reserve_cells,
      'support_top1_material_cells':support_top,'reserve_top1_material_cells':reserve_top,
      'state_roles':sorted(state_roles),'support_roles':sorted(support_roles),'reserve_roles':sorted(reserve_roles),
      'support_top1_roles':sorted(support_top_roles),'reserve_top1_roles':sorted(reserve_top_roles),
      'absolute_source_training_authorized':bool(full_go),'boundary_transport_authorized':False,'dataset_reconstruction_authorized':False,'regime_conditioned_policy_authorized':False,
    }
    if full_go:
        decision['status']='FROZEN_ROOT_SUPPORT_RESERVE_OBSERVABILITY_GO'
        decision['next_branch']='one_final_fixed_capacity_observation_aligned_support_reserve_source_then_freeze_or_close_no_capacity_or_threshold_sweep'
    else:
        decision['status']='FROZEN_ROOT_SUPPORT_RESERVE_OBSERVABILITY_STOP'
        decision['next_branch']='close_frozen_stage_i_root_and_native_certificate_source_realizations_for_support_reserve_semantics_no_more_source_adapter_sweeps'
    out={'schema':'ocrap-v48.96-srroa-comparison-v1','engineering_version':ENGINEERING_VERSION,'valid':not errors,'attribution_ready':not errors,'errors':errors,
         'experiment_type':'audit_only_target_specific_frozen_root_support_reserve_observability','planner_parameters_trained':0,'dataset_reconstruction':False,'dataset_reselection':False,
         'teacher_labels_changed':False,'teacher_metadata_input_to_model':False,'boundary_transport':False,'relative_ranker_modified':False,'regime_conditioning':False,'womd_replay_performed':False,
         'cells':cells,'preregistered_decision':decision,'test_roots_read':False}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'valid':out['valid'],'status':decision['status'],'state':state_go,'support':support_go,'reserve':reserve_go}))
    return 0 if out['valid'] else 30
if __name__=='__main__': raise SystemExit(main())
