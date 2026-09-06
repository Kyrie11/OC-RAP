#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ocrap.v48_100_joint_root_semantic_decoder import ENGINEERING_VERSION

ROLES=("dev_near","dev_contact","certificate_near","certificate_contact")

def _ok(v:Any,t:float)->bool:
    try: return v is not None and float(v)>=float(t)
    except Exception: return False

def _sha(p:Path)->str:
    h=hashlib.sha256();
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def _result_errors(obj:dict[str,Any],variant:str)->list[str]:
    e=[]
    if not obj.get('valid') or obj.get('engineering_version')!=ENGINEERING_VERSION: e.append(f'{variant}:contract')
    if int(obj.get('joint_representation_parameters_trained',0))<=0: e.append(f'{variant}:no_joint_representation_training')
    if int(obj.get('root_query_parameters_trained',-1))!=1536: e.append(f'{variant}:root_query_parameter_count')
    if int(obj.get('recovery_chart_parameters_trained',-1))!=770: e.append(f'{variant}:chart_parameter_count')
    if int(obj.get('joint_representation_parameters_trained',-1))!=2306: e.append(f'{variant}:joint_parameter_count')
    for k in ('planner_parameters_trained','source_parameters_trained','stage_i_parameters_trained','root_decoder_body_parameters_trained','root_logit_head_parameters_trained'):
        if int(obj.get(k,-1))!=0: e.append(f'{variant}:{k}')
    for role in ROLES:
        c=(obj.get('cells') or {}).get(role) or {}
        for name in ('state','support_true','reserve_true'):
            m=c.get(name) or {}
            if int(m.get('rows',0))<=0 or m.get('auc') is None: e.append(f'{variant}:{role}:{name}:empty_or_null')
        rc=(obj.get('evaluation_contracts') or {}).get(role) or {}
        if not rc.get('valid'): e.append(f'{variant}:{role}:evaluation_contract')
    sc=obj.get('semantic_metric_scales') or {}
    for k in ('support','reserve','delta_support','delta_reserve'):
        if float(sc.get(k,0.0))<=0.0: e.append(f'{variant}:scale:{k}')
    return e

def _population_errors(obj:dict[str,Any],ref:dict[str,Any],variant:str)->list[str]:
    e=[]
    for role in ROLES:
        a=(obj.get('cells') or {}).get(role) or {}; b=(ref.get('cells') or {}).get(role) or {}
        for name,fields in (
            ('state',('rows','drs_state_rows','dep_state_rows')),
            ('support_true',('rows','positive_rows','negative_rows','powered_groups')),
            ('support_shuffled',('rows','positive_rows','negative_rows','powered_groups')),
            ('reserve_true',('rows','positive_rows','negative_rows','powered_groups')),
            ('reserve_shuffled',('rows','positive_rows','negative_rows','powered_groups')),
        ):
            for f in fields:
                if int((a.get(name) or {}).get(f,-1))!=int((b.get(name) or {}).get(f,-2)):
                    e.append(f'{variant}:{role}:{name}:{f}:population_drift')
    return e

def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--balanced',type=Path,required=True); ap.add_argument('--precision',type=Path,required=True)
    ap.add_argument('--v48-99-balanced',type=Path,required=True); ap.add_argument('--v48-99-precision',type=Path,required=True)
    ap.add_argument('--v48-99-comparison',type=Path,required=True); ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args(); b=json.loads(a.balanced.read_text()); p=json.loads(a.precision.read_text()); rb=json.loads(a.v48_99_balanced.read_text()); rp=json.loads(a.v48_99_precision.read_text()); rc=json.loads(a.v48_99_comparison.read_text())
    errors=[]; rd=rc.get('preregistered_decision') or {}
    if not(rc.get('valid') and rc.get('attribution_ready') and rd.get('status')=='RECOVERY_JACOBIAN_ALIGNMENT_STOP'):
        errors.append('v48_99_stop_prerequisite_missing')
    if not(rd.get('state_chart_preserved') is True and rd.get('support_jacobian_go') is False and rd.get('reserve_debt_jacobian_go') is False):
        errors.append('v48_99_branch_shape_mismatch')
    errors += _result_errors(b,'balanced')+_result_errors(p,'precision')
    errors += _population_errors(b,rb,'balanced')+_population_errors(p,rp,'precision')

    state_cells=[]; sup_cells=[]; res_cells=[]; sup_top=[]; res_top=[]
    state_roles=set(); sup_roles=set(); res_roles=set(); sup_top_roles=set(); res_top_roles=set()
    if not errors:
        for variant,obj in (('balanced',b),('precision',p)):
            for role in ROLES:
                c=obj['cells'][role]; st=c['state']; s=c['support_true']; r=c['reserve_true']
                if _ok(st.get('auc'),0.70): state_cells.append([variant,role]); state_roles.add(role)
                if _ok(s.get('auc'),0.65) and _ok(s.get('auc_vs_shuffled'),0.05): sup_cells.append([variant,role]); sup_roles.add(role)
                if _ok(r.get('auc'),0.65) and _ok(r.get('auc_vs_shuffled'),0.05): res_cells.append([variant,role]); res_roles.add(role)
                if _ok(s.get('top1_vs_shuffled'),0.10): sup_top.append([variant,role]); sup_top_roles.add(role)
                if _ok(r.get('top1_vs_shuffled'),0.10): res_top.append([variant,role]); res_top_roles.add(role)
    def cross(x:set[str])->bool: return any('near' in r for r in x) and any('contact' in r for r in x)
    state_go=bool(not errors and len(state_cells)>=6 and len(state_roles)>=3 and cross(state_roles))
    support_go=bool(not errors and len(sup_cells)>=6 and len(sup_roles)>=3 and cross(sup_roles) and len(sup_top)>=4 and cross(sup_top_roles))
    reserve_go=bool(not errors and len(res_cells)>=6 and len(res_roles)>=3 and cross(res_roles) and len(res_top)>=4 and cross(res_top_roles))
    full_go=bool(state_go and support_go and reserve_go)
    if errors:
        status='V48_100_ENGINEERING_STOP'; next_branch='fix_v48_100_engineering_and_rerun_same_joint_root_semantic_decoder'
    elif full_go:
        status='JOINT_ROOT_SEMANTIC_DECODER_GO'; next_branch='one_final_fixed_capacity_observation_aligned_source_then_freeze_if_source_gate_passes'
    elif state_go and reserve_go and not support_go:
        status='JOINT_ROOT_SEMANTIC_SUPPORT_GUARD_STOP'; next_branch='retain_dynamic_recovery_chart_then_adjudicate_hybrid_support_guard_no_source_or_threshold_sweep'
    elif state_go and support_go and not reserve_go:
        status='JOINT_ROOT_SEMANTIC_RESERVE_FLOW_STOP'; next_branch='retain_support_chart_then_adjudicate_supported_reserve_flow_no_source_sweep'
    else:
        status='JOINT_ROOT_SEMANTIC_DECODER_STOP'; next_branch='close_root_query_plus_chart_family_then_preregister_root_cross_attention_semantic_objective_no_source_sweep'
    decision={
        'state_representation_go':state_go,'support_action_representation_go':support_go,'reserve_debt_representation_go':reserve_go,
        'joint_root_semantic_decoder_go':full_go,'state_positive_cells':state_cells,'state_roles':sorted(state_roles),
        'support_positive_cells':sup_cells,'support_roles':sorted(sup_roles),'support_top1_material_cells':sup_top,
        'reserve_positive_cells':res_cells,'reserve_roles':sorted(res_roles),'reserve_top1_material_cells':res_top,
        'final_source_experiment_authorized':bool(full_go and not errors),'boundary_transport_authorized':False,
        'regime_conditioned_policy_authorized':False,'dataset_reconstruction_authorized':False,'status':status,'next_branch':next_branch,
    }
    out={
        'schema':'ocrap-v48.100-jrsd-comparison-v1','engineering_version':ENGINEERING_VERSION,'valid':not errors,'attribution_ready':not errors,'errors':errors,
        'experiment_type':'joint_root_query_and_recovery_chart_semantic_learning','joint_representation_parameters_trained':int(b.get('joint_representation_parameters_trained',0)),
        'root_query_parameters_trained':int(b.get('root_query_parameters_trained',0)),'recovery_chart_parameters_trained':int(b.get('recovery_chart_parameters_trained',0)),
        'planner_parameters_trained':0,'source_parameters_trained':0,'stage_i_parameters_trained':0,'root_decoder_body_parameters_trained':0,'root_logit_head_parameters_trained':0,
        'boundary_transport':False,'regime_conditioning':False,'relative_ranker_modified':False,'teacher_metadata_input_to_model':False,'dataset_reconstruction':False,'dataset_reselection':False,
        'evaluation_population_contract':'exact_v48_99_equals_v48_98_equals_v48_97_2_equals_v48_96_strata',
        'v48_99_comparison_sha256':_sha(a.v48_99_comparison),'v48_99_balanced_sha256':_sha(a.v48_99_balanced),'v48_99_precision_sha256':_sha(a.v48_99_precision),
        'preregistered_decision':decision,
    }
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'valid':out['valid'],'status':status,'errors':errors}))
    return 0 if out['valid'] else 30
if __name__=='__main__': raise SystemExit(main())
