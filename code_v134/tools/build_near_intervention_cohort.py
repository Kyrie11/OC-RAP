#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from pathlib import Path
from collections import Counter, defaultdict


def _load(path: str):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _finite(x):
    try: return math.isfinite(float(x))
    except Exception: return False


def main():
    ap=argparse.ArgumentParser(description='Build the exact V48.124.9 Near intervention cohort and diagnosis summary.')
    ap.add_argument('--balanced', required=True)
    ap.add_argument('--precision', required=True)
    ap.add_argument('--target-keys-output', required=True)
    ap.add_argument('--audit-output', required=True)
    args=ap.parse_args()
    b=_load(args.balanced); p=_load(args.precision)
    if int(b.get('num_scenes',0)) != int(p.get('num_scenes',0)) or not b.get('scenes') or not p.get('scenes'):
        raise SystemExit('balanced/precision full Near results with embedded scenes are required')
    def scene_map(d): return {str(s['target_key']):s for s in d['scenes']}
    bm,pm=scene_map(b),scene_map(p)
    if set(bm)!=set(pm): raise SystemExit('balanced/precision target sets differ')
    def intervention_keys(m):
        return {k for k,s in m.items() if float(s.get('intervention_rate',0.0) or 0.0) > 0.0}
    bi,pi=intervention_keys(bm),intervention_keys(pm)
    if bi!=pi: raise SystemExit(f'intervention cohort differs across robustness variants: balanced_only={len(bi-pi)} precision_only={len(pi-bi)}')
    if not bi: raise SystemExit('no Near interventions found; this diagnostic is not applicable')
    keys=sorted(bi)
    Path(args.target_keys_output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.target_keys_output).write_text(json.dumps({'target_keys':keys},indent=2,sort_keys=True)+'\n',encoding='utf-8')

    variants={"balanced":bm,"precision":pm}
    out={
        'valid': True,
        'scientific_scope':'V48.124.9 Near deployed-system axis only',
        'full_scene_count':len(bm),
        'intervention_scene_count':len(keys),
        'untouched_nominal_scene_count':len(bm)-len(keys),
        'shared_intervention_target_keys':keys,
        'exact_reduced_replay_argument': (
            'The diagnostic selectors are downstream restrictions/re-rankers of the historical absolute-admitted set. '
            'They preserve every historical nominal short-circuit and cannot create a non-nominal action at a state where '
            'V48.124.9 chose nominal. Therefore a baseline-zero-intervention scene remains state/action-identical by induction; '
            'only the shared intervention cohort must be rerun for internal system-axis adjudication.'
        ),
        'variants':{},
    }
    for name,m in variants.items():
        n_int=0; advs=[]; opps=[]; weighted_adv_sum=0.0; weighted_opp_sum=0.0; macro=Counter(); reasons=Counter(); metric_harm=defaultdict(int); metric_benefit=defaultdict(int)
        for k in keys:
            s=m[k]; nd=int(s.get('num_decisions',0)); ni=int(round(float(s.get('intervention_rate',0.0) or 0.0)*nd)); n_int += ni
            if ni>0 and _finite(s.get('closed_loop_direct_recovery_advantage')):
                # Nominal actions have zero candidate-vs-nominal advantage, so scene mean * decisions / intervention count
                # is the exact mean advantage over intervention decisions for that scene.
                scene_adv_sum=float(s['closed_loop_direct_recovery_advantage'])*nd
                advs.append(scene_adv_sum/ni)
                weighted_adv_sum += scene_adv_sum
            if ni>0 and _finite(s.get('closed_loop_direct_recovery_opportunity')):
                # Nominal relative opportunity is 0.5 by construction after nominal centering.
                scene_opp_sum=float(s['closed_loop_direct_recovery_opportunity'])*nd - 0.5*(nd-ni)
                val=scene_opp_sum/ni
                opps.append(val)
                weighted_opp_sum += scene_opp_sum
            macro.update(s.get('macro_counts') or {}); reasons.update(s.get('selection_reason_counts') or {})
        # Remove nominal counts from macro diagnostic.
        macro.pop('nominal',None)
        out['variants'][name]={
            'intervention_decisions':n_int,
            'intervention_macro_counts':dict(macro),
            'intervention_decision_weighted_direct_advantage_mean': (weighted_adv_sum/n_int if n_int else None),
            'intervention_decision_weighted_relative_opportunity_mean': (weighted_opp_sum/n_int if n_int else None),
            'selection_reason_counts_on_intervention_scenes':dict(reasons),
            'per_intervention_scene_mean_direct_advantage':{
                'count':len(advs),'negative_count':sum(x<0 for x in advs),'positive_count':sum(x>0 for x in advs),
                'mean':sum(advs)/len(advs) if advs else None,
                'median':sorted(advs)[len(advs)//2] if advs else None,
                'min':min(advs) if advs else None,'max':max(advs) if advs else None,
            },
            'per_intervention_scene_mean_relative_opportunity':{
                'count':len(opps),'below_0_5_count':sum(x<0.5 for x in opps),'mean':sum(opps)/len(opps) if opps else None,
            },
        }
    Path(args.audit_output).write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'valid':True,'intervention_scene_count':len(keys),'target_keys_output':args.target_keys_output,'audit_output':args.audit_output},indent=2))

if __name__=='__main__': main()
