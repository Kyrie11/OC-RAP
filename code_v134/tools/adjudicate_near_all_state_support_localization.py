#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math
from pathlib import Path
from typing import Any

REFERENCE_STATUS='PCD_ORACLE_CEILING_STOP'
REFERENCE_NEXT='close_relative_only_repair_as_insufficient_keep_relative_abstention_nonharm_guard_and_localize_candidate_action_realization_or_base_trigger_support_no_threshold_or_relative_head_sweep'

def load(p:str|Path)->dict[str,Any]: return json.loads(Path(p).read_text(encoding='utf-8'))
def finite_equal(a:Any,b:Any,atol:float=1e-9)->bool:
    if a is None or b is None: return a is b
    try: af,bf=float(a),float(b)
    except Exception: return a==b
    if math.isnan(af) or math.isnan(bf): return math.isnan(af) and math.isnan(bf)
    if not(math.isfinite(af) and math.isfinite(bf)): return af==bf
    return abs(af-bf)<=atol

def scene_map(d): return {str(s.get('target_key')):s for s in (d.get('scenes') or []) if s.get('target_key')}
def compare_behavior(base,audit):
    errs=[]; bm=scene_map(base); am=scene_map(audit)
    if not am: return ['audit result contains no embedded scenes']
    for k,s in am.items():
        r=bm.get(k)
        if r is None: errs.append(f'audit target absent from Base: {k}'); continue
        for field in ('num_decisions','macro_counts','selection_reason_counts'):
            if s.get(field)!=r.get(field): errs.append(f'policy behavior mismatch {k}:{field}')
        for field in ('intervention_rate','closed_loop_bounded_NUP','closed_loop_direct_recovery_advantage'):
            if not finite_equal(s.get(field),r.get(field)): errs.append(f'policy scalar mismatch {k}:{field}')
        sm=s.get('metric_summary') or {}; rm=r.get('metric_summary') or {}
        for metric in sorted(set(sm).intersection(rm)):
            if not finite_equal(sm.get(metric),rm.get(metric)):
                errs.append(f'metric mismatch {k}:{metric}'); break
    return errs

def summarize(result:dict[str,Any],eps:float=1e-6):
    errors=[]; records=[]; scenes=result.get('scenes') or []
    total_decisions=sum(int(s.get('num_decisions',0) or 0) for s in scenes)
    for s in scenes:
        key=str(s.get('target_key'))
        if bool(s.get('audit_intervention_only')): errors.append(f'{key}: audit_intervention_only must be false')
        if str(s.get('audit_candidate_scope'))!='all': errors.append(f'{key}: audit_candidate_scope must be all')
        if not bool(s.get('audit_store_candidate_records')): errors.append(f'{key}: candidate records missing')
        rr=s.get('candidate_quality_audit_records') or []
        if len(rr)!=int(s.get('num_decisions',0) or 0): errors.append(f'{key}: record count {len(rr)} != decisions {s.get("num_decisions")}')
        for r in rr:
            x=dict(r); x['target_key']=key; records.append(x)
    if len(records)!=total_decisions: errors.append(f'total record count {len(records)} != total decisions {total_decisions}')
    bins={
      'triggered':{'decisions':0,'pcd_positive_anywhere':0,'pcd_positive_admitted':0,'pcd_positive_anywhere_but_none_admitted':0,'sum_best_all_positive_advantage':0.0,'sum_best_admitted_positive_advantage':0.0,'max_best_all_advantage':0.0,'max_best_admitted_advantage':0.0},
      'nominal':{'decisions':0,'pcd_positive_anywhere':0,'pcd_positive_admitted':0,'pcd_positive_anywhere_but_none_admitted':0,'sum_best_all_positive_advantage':0.0,'sum_best_admitted_positive_advantage':0.0,'max_best_all_advantage':0.0,'max_best_admitted_advantage':0.0},
    }
    compact=[]
    for r in records:
        rows=r.get('candidates') or []
        if int(r.get('num_candidates_labeled',0) or 0)!=24 or len(rows)!=24:
            errors.append(f'{r.get("target_key")} step {r.get("step_index")}: expected 24 candidate labels got field={r.get("num_candidates_labeled")} rows={len(rows)}'); continue
        nom=next((x for x in rows if int(x.get('candidate_index',-1))==0),None)
        sel=next((x for x in rows if bool(x.get('selected'))),None)
        if nom is None or sel is None:
            errors.append(f'{r.get("target_key")} step {r.get("step_index")}: nominal/selected missing'); continue
        npcd=float(nom['teacher_pcd'])
        nonnom=[x for x in rows if int(x.get('candidate_index',-1))!=0]
        better=[x for x in nonnom if float(x.get('teacher_pcd',-1e30))>npcd+eps]
        better_adm=[x for x in better if bool(x.get('absolute_admitted'))]
        best_all=max([float(x.get('teacher_pcd',-1e30))-npcd for x in nonnom],default=-float('inf'))
        adm=[x for x in nonnom if bool(x.get('absolute_admitted'))]
        best_adm=max([float(x.get('teacher_pcd',-1e30))-npcd for x in adm],default=-float('inf'))
        triggered=int(sel.get('candidate_index',0))!=0
        b=bins['triggered' if triggered else 'nominal']; b['decisions']+=1
        has_any=bool(better); has_adm=bool(better_adm)
        b['pcd_positive_anywhere']+=int(has_any); b['pcd_positive_admitted']+=int(has_adm); b['pcd_positive_anywhere_but_none_admitted']+=int(has_any and not has_adm)
        if math.isfinite(best_all) and best_all>eps:
            b['sum_best_all_positive_advantage']+=best_all; b['max_best_all_advantage']=max(b['max_best_all_advantage'],best_all)
        if math.isfinite(best_adm) and best_adm>eps:
            b['sum_best_admitted_positive_advantage']+=best_adm; b['max_best_admitted_advantage']=max(b['max_best_admitted_advantage'],best_adm)
        if has_any or has_adm:
            compact.append({'target_key':r.get('target_key'),'step_index':r.get('step_index'),'base_triggered':triggered,'selected_candidate_index':r.get('selected_candidate_index'),'best_all_teacher_pcd_advantage':None if not math.isfinite(best_all) else best_all,'best_admitted_teacher_pcd_advantage':None if not math.isfinite(best_adm) else best_adm,'pcd_positive_anywhere':has_any,'pcd_positive_admitted':has_adm})
    total_pos=bins['triggered']['pcd_positive_admitted']+bins['nominal']['pcd_positive_admitted']
    recall=(bins['triggered']['pcd_positive_admitted']/total_pos) if total_pos else None
    any_total=bins['triggered']['pcd_positive_anywhere']+bins['nominal']['pcd_positive_anywhere']
    adm_recall=(total_pos/any_total) if any_total else None
    return {
      'num_scenes':len(scenes),'num_decisions':total_decisions,'num_candidate_audit_records':len(records),'all_candidates_labeled_every_decision':not errors,
      'by_base_decision':bins,'positive_admitted_trigger_recall':recall,'absolute_admission_recall_of_positive_anywhere':adm_recall,'opportunity_records':compact,
    },errors

def branch(variants):
    vals=list(variants.values())
    missed=[v['by_base_decision']['nominal']['pcd_positive_admitted'] for v in vals]
    missed_any=[v['by_base_decision']['nominal']['pcd_positive_anywhere'] for v in vals]
    if vals and all(x>0 for x in missed):
        return 'BASE_TRIGGER_MISSES_POSITIVE_ADMITTED_PCD_OPPORTUNITIES_ON_FROZEN_INTERVENTION_COHORT'
    if vals and all(x==0 for x in missed) and all(x>0 for x in missed_any):
        return 'ABSOLUTE_ADMISSION_BLOCKS_ALL_NONTRIGGER_POSITIVE_PCD_OPPORTUNITIES_ON_FROZEN_INTERVENTION_COHORT'
    if vals and all(x==0 for x in missed_any):
        return 'NO_NONTRIGGER_POSITIVE_PCD_SUPPORT_ON_FROZEN_INTERVENTION_COHORT_CANDIDATE_ACTION_OR_PCD_TARGET_SUPPORT_REMAINS_LIMITING'
    return 'MIXED_ROBUSTNESS_SUPPORT_LOCALIZATION_REQUIRES_RECORD_LEVEL_INSPECTION'

def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument('--reference-adjudication',required=True); ap.add_argument('--runtime-contract',required=True)
    for v in ('balanced','precision'):
        ap.add_argument(f'--base-{v}',required=True); ap.add_argument(f'--audit-{v}',required=True)
    ap.add_argument('--route-audit',required=True); ap.add_argument('--output',required=True); a=ap.parse_args(); errors=[]
    ref=load(a.reference_adjudication); rt=load(a.runtime_contract); route=load(a.route_audit)
    if not(ref.get('valid') and ref.get('attribution_ready') and ref.get('status')==REFERENCE_STATUS and ref.get('next_branch')==REFERENCE_NEXT): errors.append('10.4 STOP reference does not authorize support localization')
    if not(rt.get('valid') and rt.get('attribution_ready')): errors.append('runtime contract invalid')
    if not route.get('valid'): errors.append('route audit invalid')
    variants={}
    for v in ('balanced','precision'):
        b=load(getattr(a,f'base_{v}')); x=load(getattr(a,f'audit_{v}'))
        errors.extend([f'{v}:{e}' for e in compare_behavior(b,x)])
        s,e=summarize(x); variants[v]=s; errors.extend([f'{v}:{q}' for q in e])
    br=branch(variants) if not errors else 'ENGINEERING_FIX_REQUIRED_BEFORE_SCIENTIFIC_LOCALIZATION'
    if br.startswith('BASE_TRIGGER_MISSES'):
        nxt='run_small_trigger_free_admitted_teacher_pcd_oracle_on_same_11_scene_cohort_before_any_v48_125_design'
    elif br.startswith('ABSOLUTE_ADMISSION_BLOCKS'):
        nxt='audit_absolute_admission_false_negative_semantics_on_missed_nontrigger_opportunities_no_threshold_sweep'
    elif br.startswith('NO_NONTRIGGER'):
        nxt='close_trigger_repair_on_this_cohort_and_localize_candidate_action_realization_vs_teacher_pcd_target_semantics_no_relative_head_sweep'
    elif errors:
        nxt='fix_engineering_only_do_not_interpret_support_localization'
    else:
        nxt='inspect_record_level_support_disagreement_before_any_algorithm_change'
    out={
      'schema':'ocrap-v48.124.10.5-all-state-support-localization-v1','engineering_version':'v48.124.10.5-ALL-STATE-SUPPORT-LOCALIZATION-DIAGNOSTIC','scientific_version':'v48.124-OC-FMSA',
      'valid':not errors,'attribution_ready':not errors,'errors':errors,'publication_evidence':False,'algorithm_modified':False,
      'reference_status':ref.get('status'),'diagnostic_branch':br,'next_branch':nxt,'variants':variants,
      'causal_question':'Along the exact frozen Base trajectories of the 11 historical Near intervention scenes, do teacher-PCD-positive frozen candidates exist at Base-nominal decisions, and if so are they already absolute-admitted?',
      'interpretation_rule':'This is a support/trigger localization audit, not a policy arm. Positive admitted opportunities at Base-nominal states prove trigger misses on this cohort; their absence does not prove global absence outside the cohort. No result authorizes threshold or relative-head sweeping.',
    }
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'valid':out['valid'],'diagnostic_branch':br,'next_branch':nxt,'output':a.output},indent=2)); return 0 if not errors else 30
if __name__=='__main__': raise SystemExit(main())
