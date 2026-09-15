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
def skey(s): return str(s.get('target_key') or '')
def val(s,m):
    x=s.get(m,(s.get('metric_summary') or {}).get(m))
    try:
        y=float(x); return y if math.isfinite(y) else None
    except Exception:return None

def effects(nominal:dict[str,Any], method:dict[str,Any], seed_keys:set[str]):
    nm={skey(s):s for s in nominal.get('scenes') or []}; mm={skey(s):s for s in method.get('scenes') or []}; errors=[]
    if set(mm)!=seed_keys: errors.append(f'method seed key mismatch: got={sorted(mm)} expected={sorted(seed_keys)}')
    miss=seed_keys-set(nm)
    if miss: errors.append(f'nominal missing seed keys: {sorted(miss)}')
    scene_rows={}; aggregate={m:[] for m in PRIMARY}; hard_ok=True
    for k in sorted(seed_keys & set(nm) & set(mm)):
        row={'hard':{},'primary':{},'classification':'neutral'}; any_imp=False; any_bad=False
        for m,d in HARD.items():
            a,b=val(nm[k],m),val(mm[k],m); ok=(a is not None and b is not None and (b<=a+EPS if d=='lower' else b>=a-EPS))
            row['hard'][m]={'nominal':a,'method':b,'pass':ok}; hard_ok &= ok
        for m,d in PRIMARY.items():
            a,b=val(nm[k],m),val(mm[k],m)
            if a is None or b is None: continue
            delta=b-a; aggregate[m].append(delta)
            imp=(delta < -EPS if d=='lower' else delta > EPS); bad=(delta > EPS if d=='lower' else delta < -EPS)
            any_imp |= imp; any_bad |= bad
            row['primary'][m]={'nominal':a,'method':b,'delta':delta,'direction':d,'improved':imp,'worsened':bad}
        if any_bad: row['classification']='worsened'
        elif any_imp: row['classification']='locally_positive'
        scene_rows[k]=row
    summary={}
    for m,d in PRIMARY.items():
        ds=aggregate[m]
        if ds:
            mean=sum(ds)/len(ds); summary[m]={'n':len(ds),'mean_paired_delta':mean,'direction':d,
                'improved':(mean < -EPS if d=='lower' else mean > EPS),'worsened':(mean > EPS if d=='lower' else mean < -EPS)}
    return {'errors':errors,'hard_no_harm':hard_ok,'scene_effects':scene_rows,'primary_summary':summary}

def main()->int:
    ap=argparse.ArgumentParser(description='Adjudicate V48.124.10.7 one-shot seed action-realization ceiling (diagnostic only).')
    ap.add_argument('--predecessor-adjudication',required=True); ap.add_argument('--seed-cohort',required=True); ap.add_argument('--nominal-full',required=True); ap.add_argument('--route-audit',required=True)
    for v in ('balanced','precision'):
        ap.add_argument(f'--{v}-result',required=True); ap.add_argument(f'--{v}-vs-nominal',required=True)
    ap.add_argument('--output',required=True); a=ap.parse_args(); errors=[]
    pred=load(a.predecessor_adjudication); seeds=load(a.seed_cohort); nominal=load(a.nominal_full); route=load(a.route_audit)
    if not(pred.get('valid') and pred.get('attribution_ready') and pred.get('status')=='NONFLOOR_ADMISSION_SEED_SCREEN_NOT_PROMISING'): errors.append('invalid_predecessor_10_6')
    if not(seeds.get('valid') and seeds.get('attribution_ready')): errors.append('invalid_seed_cohort')
    if not route.get('valid'): errors.append('invalid_route_audit')
    seed_rows=seeds.get('seeds') or {}; seed_keys=set(seed_rows)
    variants={}; all_hard=True; all_exact=True; classifications=[]
    for v in ('balanced','precision'):
        result=load(getattr(a,f'{v}_result')); comp=load(getattr(a,f'{v}_vs_nominal'))
        scenes=result.get('scenes') or []
        if set(skey(s) for s in scenes)!=seed_keys: errors.append(f'{v}:seed_key_mismatch')
        if int(comp.get('num_paired_scenes',-1))!=len(seed_keys): errors.append(f'{v}:pair_coverage')
        per_scene={}; exact=True
        for s in scenes:
            k=skey(s); plan=seed_rows.get(k) or {}; recs=s.get('privileged_nonfloor_one_shot_records') or []
            if not bool(s.get('privileged_nonfloor_one_shot_seed_realization')): errors.append(f'{v}:{k}:marker_missing')
            if int(s.get('privileged_nonfloor_one_shot_trigger_count',0) or 0)!=1 or len(recs)!=1: errors.append(f'{v}:{k}:expected_exactly_one_trigger'); exact=False
            if int(s.get('privileged_nonfloor_one_shot_label_count',0) or 0)!=2: errors.append(f'{v}:{k}:expected_two_labels'); exact=False
            if recs:
                r=recs[0]; want_step=int(plan.get('best_pretrigger_step',-99)); want_cid=int(plan.get('best_pretrigger_candidate_index',-99))
                ok=(int(r.get('step_index',-1))==want_step and int(r.get('selected_candidate_index',-1))==want_cid and bool(r.get('nonfloor_contract_valid')))
                if not ok: errors.append(f'{v}:{k}:target_mismatch'); exact=False
                per_scene[k]={'step':int(r.get('step_index',-1)),'candidate_index':int(r.get('selected_candidate_index',-1)),
                              'teacher_pcd_advantage':float(r.get('teacher_pcd_advantage',float('nan'))),'teacher_r_dep':float(r.get('teacher_r_dep',float('nan')))}
        eff=effects(nominal,result,seed_keys); errors.extend(f'{v}:{e}' for e in eff['errors']); all_hard &= eff['hard_no_harm']; all_exact &= exact
        cls={k:r['classification'] for k,r in eff['scene_effects'].items()}; classifications.extend(cls.values())
        variants[v]={'exact_one_shot_contract':exact,'hard_no_harm':eff['hard_no_harm'],'seed_execution':per_scene,'effects':eff,'comparison_vs_nominal_descriptive_only':comp}
    if errors:
        status='ONE_SHOT_ACTION_REALIZATION_ATTRIBUTION_NOT_ENTERED'; next_branch='fix_engineering_only'
    else:
        any_bad=any(c=='worsened' for c in classifications); any_pos=any(c=='locally_positive' for c in classifications)
        if all_hard and all_exact and any_pos and not any_bad:
            status='ONE_SHOT_ACTION_REALIZATION_LOCALLY_POSITIVE'
            next_branch='repeated_pcd_policy_compounding_or_system_metric_alignment_is_the_remaining_diagnostic_axis_keep_mechanism_frozen'
        elif all_hard and all_exact and any_pos and any_bad:
            status='ONE_SHOT_ACTION_REALIZATION_MIXED'
            next_branch='candidate_physical_effect_is_not_uniform_under_near_endpoints_close_absolute_admission_repair_keep_mechanism_frozen'
        else:
            status='ONE_SHOT_ACTION_REALIZATION_UNSUPPORTED'
            next_branch='close_current_nonfloor_admission_repair_hypothesis_keep_mechanism_frozen'
    out={'schema':'ocrap-v48.124.10.7-one-shot-action-realization-v1','engineering_version':'v48.124.10.7-ONE-SHOT-ACTION-REALIZATION-CEILING',
         'scientific_version':'v48.124-OC-FMSA','valid':not errors,'attribution_ready':not errors,'algorithm_modified':False,'publication_evidence':False,
         'errors':errors,'status':status,'next_branch':next_branch,'variants':variants,
         'causal_question':'From the untouched exact-nominal trajectory, does executing exactly the strongest preregistered V48.124.10.5 non-floor seed candidate once improve Near physical endpoints when all subsequent decisions are forced back to exact nominal?',
         'interpretation_rule':'This is a terminal diagnostic ceiling, not a deployable arm, population GO test, mechanism promotion, or authorization to retrain admission. It isolates single-action realization from the repeated privileged PCD policy used in V48.124.10.6.'}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({'valid':out['valid'],'status':status,'next_branch':next_branch,'output':a.output},indent=2)); return 0 if not errors else 30
if __name__=='__main__': raise SystemExit(main())
