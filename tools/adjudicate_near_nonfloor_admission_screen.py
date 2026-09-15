#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math
from pathlib import Path
from typing import Any

PRIMARY={
 'critical_ttc_exposure_duration_s':'lower','clearance_deficit_auc_m_s':'lower','ttc_deficit_auc_s2':'lower',
 'min_clearance_m_min':'higher','ttc_s_min':'higher',
}
HARD={'overlap_any':'lower','offroad_any':'lower'}
EPS=1e-9

def load(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def key(s): return str(s.get('target_key') or '')
def val(s,m):
    x=s.get(m,(s.get('metric_summary') or {}).get(m))
    try:
        y=float(x); return y if math.isfinite(y) else None
    except Exception:return None

def local_effects(nominal:dict[str,Any], method:dict[str,Any], seed_keys:set[str]):
    nm={key(s):s for s in nominal.get('scenes') or []}; mm={key(s):s for s in method.get('scenes') or []}; errors=[]
    if set(mm)!=seed_keys: errors.append(f'method seed key mismatch: got={sorted(mm)} expected={sorted(seed_keys)}')
    missing=seed_keys-set(nm)
    if missing: errors.append(f'nominal missing seed keys: {sorted(missing)}')
    hard_ok=True; scene_rows={}; metric_delta={m:[] for m in PRIMARY}
    for k in sorted(seed_keys & set(nm) & set(mm)):
        sr={'hard':{},'primary':{}}
        for m,d in HARD.items():
            a,b=val(nm[k],m),val(mm[k],m)
            ok=(a is not None and b is not None and (b<=a+EPS if d=='lower' else b>=a-EPS))
            sr['hard'][m]={'nominal':a,'method':b,'pass':ok}; hard_ok &= ok
        for m,d in PRIMARY.items():
            a,b=val(nm[k],m),val(mm[k],m)
            if a is None or b is None: continue
            delta=b-a; metric_delta[m].append(delta)
            favorable=(delta < -EPS if d=='lower' else delta > EPS)
            unfavorable=(delta > EPS if d=='lower' else delta < -EPS)
            sr['primary'][m]={'nominal':a,'method':b,'delta':delta,'direction':d,'improved':favorable,'worsened':unfavorable}
        scene_rows[k]=sr
    summary={}; improved=set(); worsened=set()
    for m,d in PRIMARY.items():
        ds=metric_delta[m]
        if not ds: continue
        mean=sum(ds)/len(ds); imp=(mean < -EPS if d=='lower' else mean > EPS); bad=(mean > EPS if d=='lower' else mean < -EPS)
        if imp: improved.add(m)
        if bad: worsened.add(m)
        summary[m]={'n':len(ds),'mean_paired_delta':mean,'direction':d,'improved':imp,'worsened':bad}
    return {'errors':errors,'hard_no_harm':hard_ok,'scene_effects':scene_rows,'primary_summary':summary,'improved_metrics':sorted(improved),'worsened_metrics':sorted(worsened)}

def main()->int:
    ap=argparse.ArgumentParser(description='Adjudicate the tiny non-floor privileged admission screen. This is a falsification screen, not a population GO test.')
    ap.add_argument('--reference-contract',required=True); ap.add_argument('--runtime-contract',required=True); ap.add_argument('--seed-cohort',required=True)
    ap.add_argument('--nominal-full',required=True); ap.add_argument('--route-audit',required=True)
    for v in ('balanced','precision'):
        ap.add_argument(f'--{v}-result',required=True); ap.add_argument(f'--{v}-vs-nominal',required=True); ap.add_argument(f'--{v}-vs-base',required=True)
    ap.add_argument('--output',required=True); a=ap.parse_args(); errors=[]
    ref=load(a.reference_contract); runtime=load(a.runtime_contract); seeds=load(a.seed_cohort); route=load(a.route_audit); nominal=load(a.nominal_full)
    if not(ref.get('valid') and ref.get('attribution_ready')): errors.append('reference_contract_invalid')
    if not(runtime.get('valid') and runtime.get('attribution_ready')): errors.append('runtime_contract_invalid')
    if not(seeds.get('valid') and seeds.get('attribution_ready')): errors.append('seed_cohort_invalid')
    if not route.get('valid'): errors.append('route_audit_invalid')
    seed_keys=set((seeds.get('seeds') or {}).keys())
    if not seed_keys: errors.append('empty_seed_cohort')
    rows={}; common_improved=None; any_worsened=False; all_hard=True; all_executed=True
    for v in ('balanced','precision'):
        result=load(getattr(a,f'{v}_result')); comp=load(getattr(a,f'{v}_vs_nominal')); compbase=load(getattr(a,f'{v}_vs_base'))
        scenes=result.get('scenes') or []
        if len(scenes)!=len(seed_keys): errors.append(f'seed_scene_count:{v}:{len(scenes)}:{len(seed_keys)}')
        if set(key(s) for s in scenes)!=seed_keys: errors.append(f'seed_keys:{v}')
        if int(comp.get('num_paired_scenes',-1))!=len(seed_keys): errors.append(f'nominal_pair_coverage:{v}')
        if int(compbase.get('num_paired_scenes',-1))!=len(seed_keys): errors.append(f'base_pair_coverage:{v}')
        interventions=0; labels=0; prestart_ok=True
        for s in scenes:
            if not bool(s.get('privileged_nonfloor_pcd_oracle_ceiling')): errors.append(f'nonfloor_oracle_marker_missing:{v}:{key(s)}')
            start=int(s.get('privileged_nonfloor_seed_start_step',-1)); expected=int((seeds.get('seeds') or {}).get(key(s),{}).get('start_step',-2))
            if start!=expected: errors.append(f'seed_start_mismatch:{v}:{key(s)}:{start}:{expected}')
            labels+=int(s.get('privileged_nonfloor_pcd_oracle_label_count',0) or 0)
            recs=s.get('privileged_nonfloor_pcd_oracle_records') or []
            interventions+=sum(1 for r in recs if int(r.get('oracle_selected_candidate_index',0) or 0)!=0)
            if any(int(r.get('step_index',-1))<start for r in recs): prestart_ok=False
            if any(int(r.get('num_candidates_labeled',-1))!=24 for r in recs): errors.append(f'candidate_label_count:{v}:{key(s)}')
            if any(abs(float(r.get('oracle_selected_teacher_r_dep',0.0))-0.5)<=1e-8 and int(r.get('oracle_selected_candidate_index',0) or 0)!=0 for r in recs):
                errors.append(f'structural_floor_executed:{v}:{key(s)}')
        eff=local_effects(nominal,result,seed_keys); errors.extend(f'{v}:{e}' for e in eff['errors'])
        all_hard &= bool(eff['hard_no_harm']); all_executed &= interventions>0 and prestart_ok
        ims=set(eff['improved_metrics']); common_improved=ims if common_improved is None else common_improved & ims
        any_worsened |= bool(eff['worsened_metrics'])
        rows[v]={'num_seed_scenes':len(scenes),'oracle_executed_interventions':interventions,'teacher_candidate_labels':labels,'no_teacher_records_before_seed_start':prestart_ok,
                 'hard_no_harm':eff['hard_no_harm'],'improved_primary_metrics':eff['improved_metrics'],'worsened_primary_metrics':eff['worsened_metrics'],
                 'local_effects':eff,'comparison_vs_nominal_descriptive_only':comp,'comparison_vs_base_descriptive_only':compbase}
    common_improved=common_improved or set()
    # This is deliberately a one-sided falsification screen. It never authorizes
    # V48.125. PROMISING only means the non-floor admission hypothesis has
    # survived a tiny privileged upper-bound test and merits a broader prevalence audit.
    promising=bool(not errors and all_hard and all_executed and common_improved and not any_worsened)
    if errors:
        status='NONFLOOR_ADMISSION_SEED_SCREEN_ATTRIBUTION_NOT_ENTERED'; next_branch='fix_engineering_only_do_not_interpret_screen'
    elif promising:
        status='NONFLOOR_ADMISSION_SEED_SCREEN_PROMISING'
        next_branch='broaden_nonfloor_support_prevalence_on_small_scene_disjoint_near_sample_before_any_v48_125_no_threshold_or_model_change'
    else:
        status='NONFLOOR_ADMISSION_SEED_SCREEN_NOT_PROMISING'
        next_branch='do_not_authorize_v48_125_absolute_admission_repair_from_current_evidence_audit_pcd_to_system_metric_alignment_or_action_realization_no_threshold_sweep'
    out={
      'schema':'ocrap-v48.124.10.6-nonfloor-admission-seed-screen-v1','engineering_version':'v48.124.10.6-NONFLOOR-ADMISSION-SEED-SCREEN','scientific_version':'v48.124-OC-FMSA',
      'valid':not errors,'attribution_ready':not errors,'algorithm_modified':False,'publication_evidence':False,'errors':errors,
      'status':status,'promising':promising,'common_improved_primary_metrics':sorted(common_improved),'next_branch':next_branch,'variants':rows,
      'causal_question':'Do the non-floor, pre-first-trigger candidate opportunities identified by V48.124.10.5 survive a privileged admission+selection upper-bound screen as actual closed-loop Near recovery signals?',
      'interpretation_rule':'This tiny seed screen may falsify the admission-repair hypothesis early. Non-floor only removes the known exact-0.5 plateau and is not asserted to be fully point-identified physical truth. A PROMISING result is not population evidence, not an arm GO, and does not authorize V48.125 by itself; it only licenses a broader scene-disjoint support-prevalence audit.',
      'statistical_note':'With only a few changed scenes, a 250-scene bootstrap significance gate is structurally underpowered because many resamples contain no changed scene. Therefore the seed screen uses deterministic local direction/no-hard-harm criteria and keeps the paired bootstrap reports descriptive only.',
    }
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({'valid':out['valid'],'status':status,'promising':promising,'next_branch':next_branch,'output':a.output},indent=2)); return 0 if not errors else 30
if __name__=='__main__': raise SystemExit(main())
