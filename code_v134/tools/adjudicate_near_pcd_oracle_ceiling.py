#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
from typing import Any
from ocrap.audits.fixed_main_stability import NEAR_BENEFIT,NEAR_HARD_NO_HARM,_gate_benefit,_gate_no_harm,target_keys

SECONDARY={
    'closed_loop_bounded_NUP':'higher',
    'route_progression_m':'higher',
    'near_contact_exposure_duration_s':'lower',
}
REFERENCE_STATUS='OBSERVATION_LEGAL_NEAR_SYSTEM_AXIS_STOP'
REFERENCE_BRANCH='relative_evidence_alignment_bottleneck_better_admitted_teacher_pcd_candidates_exist_but_relative_head_does_not_support_them'

def load(p:str|Path)->dict[str,Any]: return json.loads(Path(p).read_text(encoding='utf-8'))

def main()->int:
    ap=argparse.ArgumentParser(description='Adjudicate the privileged PCD-oracle Near ceiling diagnostic.')
    ap.add_argument('--reference-contract',required=True)
    ap.add_argument('--runtime-contract',required=True)
    ap.add_argument('--candidate-adjudication',required=True)
    ap.add_argument('--nominal-full',required=True)
    for v in ('balanced','precision'):
        ap.add_argument(f'--{v}-subset',required=True)
        ap.add_argument(f'--{v}-full',required=True)
        ap.add_argument(f'--{v}-vs-nominal',required=True)
        ap.add_argument(f'--{v}-vs-base',required=True)
    ap.add_argument('--route-audit',required=True)
    ap.add_argument('--output',required=True)
    a=ap.parse_args(); errors=[]
    ref=load(a.reference_contract); runtime=load(a.runtime_contract); cand=load(a.candidate_adjudication); route=load(a.route_audit)
    if not ref.get('valid'): errors.append('reference_contract_invalid')
    if not(runtime.get('valid') and runtime.get('attribution_ready')): errors.append('runtime_contract_invalid')
    if not(cand.get('valid') and cand.get('attribution_ready') and cand.get('diagnostic_branch')==REFERENCE_BRANCH): errors.append('candidate_quality_branch_not_established')
    if cand.get('publication_evidence') is not False: errors.append('candidate_quality_publication_flag_unexpected')
    if not route.get('valid'): errors.append('route_audit_invalid')
    nominal=load(a.nominal_full); nkeys=target_keys(nominal)
    if len(nkeys)!=250: errors.append(f'nominal_population:{len(nkeys)}')
    rows={}; common=None
    total_trigger=0; total_oracle_intervention=0; total_positive=0
    for v in ('balanced','precision'):
        subset=load(getattr(a,f'{v}_subset')); full=load(getattr(a,f'{v}_full'))
        comp=load(getattr(a,f'{v}_vs_nominal')); compbase=load(getattr(a,f'{v}_vs_base'))
        if target_keys(full)!=nkeys: errors.append(f'target_mismatch:{v}')
        if int(comp.get('num_paired_scenes') or -1)!=250: errors.append(f'nominal_pair_coverage:{v}')
        if int(compbase.get('num_paired_scenes') or -1)!=250: errors.append(f'base_pair_coverage:{v}')
        recon=full.get('exact_trigger_gated_subset_reconstruction') or {}
        if not recon.get('internal_diagnostic_only'): errors.append(f'missing_trigger_reconstruction_contract:{v}')
        subset_scenes=subset.get('scenes') or []
        if len(subset_scenes)!=11: errors.append(f'unexpected_subset_scene_count:{v}:{len(subset_scenes)}')
        if not all(bool(s.get('privileged_pcd_oracle_ceiling')) for s in subset_scenes): errors.append(f'oracle_marker_missing:{v}')
        trigger=sum(int(s.get('privileged_pcd_oracle_trigger_count',0) or 0) for s in subset_scenes)
        positive=0; oracle_int=0
        for s in subset_scenes:
            for r in s.get('privileged_pcd_oracle_records') or []:
                if float(r.get('best_admitted_teacher_pcd_advantage') or 0.0)>1e-6: positive+=1
                if str(r.get('oracle_selection_reason'))=='privileged_pcd_oracle_best_admitted': oracle_int+=1
        # The *trigger rule* is frozen, not the historical trigger schedule. Once the
        # oracle changes an action, later states can differ and the Base trigger may
        # fire at different times/counts. Only the 11-scene historical intervention
        # cohort is fixed; requiring exactly 51 fresh triggers would be invalid.
        if trigger <= 0: errors.append(f'no_base_trigger_on_fresh_oracle_replay:{v}')
        for s in subset_scenes:
            for r in s.get('privileged_pcd_oracle_records') or []:
                if int(r.get('num_candidates_labeled', -1)) != 24:
                    errors.append(f'oracle_candidate_label_count:{v}:{r.get("num_candidates_labeled")}')
                if int(r.get('num_absolute_admitted_non_nominal', -1)) < 0:
                    errors.append(f'oracle_absolute_admission_missing:{v}')
        if positive!=oracle_int: errors.append(f'positive_oracle_action_count_mismatch:{v}:{positive}:{oracle_int}')
        total_trigger+=trigger; total_positive+=positive; total_oracle_intervention+=oracle_int
        hard=_gate_no_harm(comp,NEAR_HARD_NO_HARM); benefit=_gate_benefit(comp,NEAR_BENEFIT); secondary=_gate_no_harm(comp,SECONDARY)
        bs=set(benefit['beneficial_metrics']); common=bs if common is None else common & bs
        rows[v]={
            'num_subset_scenes':len(subset_scenes),
            'base_trigger_decisions':trigger,
            'oracle_positive_admitted_decisions':positive,
            'oracle_executed_interventions':oracle_int,
            'hard_no_harm':hard,'benefit':benefit,'secondary_system_no_harm':secondary,
            'primary_near_go':bool(hard['go'] and benefit['go']),
            'comparison_vs_nominal':comp,
            'comparison_vs_base':compbase,
        }
    common=common or set()
    go=bool(not errors and common and all(rows[v]['primary_near_go'] and rows[v]['secondary_system_no_harm']['go'] for v in rows))
    if errors:
        status='PCD_ORACLE_CEILING_ATTRIBUTION_NOT_ENTERED'; next_branch='fix_engineering_only_do_not_interpret_ceiling'
    elif go:
        status='PCD_ORACLE_CEILING_GO'
        next_branch=(
            'authorize_v48_125_deployable_relative_evidence_realignment_only_keep_candidate_library_absolute_admission_'
            'base_trigger_recovery_mechanism_route_dynamics_and_horizon_frozen_train_calibrate_on_train_val_calib_only'
        )
    else:
        status='PCD_ORACLE_CEILING_STOP'
        next_branch=(
            'close_relative_only_repair_as_insufficient_keep_relative_abstention_nonharm_guard_and_localize_candidate_action_'
            'realization_or_base_trigger_support_no_threshold_or_relative_head_sweep'
        )
    out={
        'schema':'ocrap-v48.124.10.4-pcd-oracle-ceiling-v1',
        'engineering_version':'v48.124.10.4-PCD-ORACLE-CEILING-DIAGNOSTIC',
        'scientific_version':'v48.124-OC-FMSA',
        'valid':not errors,'attribution_ready':not errors,'errors':errors,
        'publication_evidence':False,
        'status':status,'go':go,'common_beneficial_metrics':sorted(common),'next_branch':next_branch,
        'causal_question':'With the candidate library, absolute-admission rule, and Base intervention-trigger rule held fixed, is privileged execution-consistent teacher-PCD preference sufficient to close the Near benefit + no-harm gate?',
        'interpretation_rule':(
            'GO is empirical closed-loop sufficiency of the existing trigger/action support under a privileged relative oracle on this diagnostic cohort; '
            'it is not a deployable algorithm and not a theorem. STOP rules out relative-evidence realignment alone as sufficient.'
        ),
        'aggregate':{
            'base_trigger_decisions_across_variants':total_trigger,
            'oracle_positive_admitted_decisions_across_variants':total_positive,
            'oracle_executed_interventions_across_variants':total_oracle_intervention,
        },
        'variants':rows,
    }
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'valid':out['valid'],'status':status,'go':go,'next_branch':next_branch,'output':a.output},indent=2))
    return 0 if not errors else 30
if __name__=='__main__': raise SystemExit(main())
