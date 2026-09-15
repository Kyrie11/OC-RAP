#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math
from pathlib import Path
from typing import Any

FLOOR=0.5; TOL=1e-8; EPS=1e-6

def load(p): return json.loads(Path(p).read_text(encoding='utf-8'))

def extract(result:dict[str,Any])->dict[str,Any]:
    witnesses=[]; first_trigger={}
    for s in result.get('scenes') or []:
        key=str(s.get('target_key') or '')
        rr=s.get('candidate_quality_audit_records') or []
        triggers=[int(r.get('step_index',-1)) for r in rr if int(r.get('selected_candidate_index',0) or 0)!=0]
        first_trigger[key]=min(triggers) if triggers else None
        for r in rr:
            step=int(r.get('step_index',-1)); selected=int(r.get('selected_candidate_index',0) or 0)
            if selected!=0: continue
            for c in r.get('candidates') or []:
                cid=int(c.get('candidate_index',-1));
                if cid==0: continue
                rd=float(c.get('teacher_r_dep_star',float('nan'))); adv=float(c.get('teacher_pcd_advantage_vs_nominal',float('nan')))
                if not(math.isfinite(rd) and math.isfinite(adv)): continue
                if rd<=0 or abs(rd-FLOOR)<=TOL or adv<=EPS: continue
                witnesses.append({
                    'target_key':key,'step_index':step,'candidate_index':cid,'macro':str(c.get('macro') or ''),
                    'teacher_pcd_advantage':adv,'teacher_r_dep_star':rd,'teacher_drs':float(c.get('teacher_drs',float('nan'))),
                    'teacher_oracle_gap_star':float(c.get('teacher_oracle_gap_star',float('nan'))),
                    'pred_r_dep':float(c.get('pred_r_dep',float('nan'))),'pred_gap':float(c.get('pred_gap',float('nan'))),
                    'absolute_admitted':bool(c.get('absolute_admitted')),'first_base_trigger_step':first_trigger[key],
                })
    return {'witnesses':witnesses,'first_trigger':first_trigger}

def signature(rows):
    return [(r['target_key'],r['step_index'],r['candidate_index'],r['macro'],round(r['teacher_pcd_advantage'],12),round(r['teacher_r_dep_star'],12),r['absolute_admitted'],r['first_base_trigger_step']) for r in rows]

def main()->int:
    ap=argparse.ArgumentParser(description='Derive the non-floor, pre-first-trigger falsification seed cohort from V48.124.10.5.')
    ap.add_argument('--balanced-audit',required=True); ap.add_argument('--precision-audit',required=True)
    ap.add_argument('--adjudication',required=True); ap.add_argument('--output',required=True); ap.add_argument('--target-keys-output',required=True)
    a=ap.parse_args(); errors=[]
    adj=load(a.adjudication)
    if not(adj.get('valid') and adj.get('attribution_ready') and adj.get('algorithm_modified') is False): errors.append('10.5 adjudication invalid')
    b=extract(load(a.balanced_audit)); p=extract(load(a.precision_audit))
    if signature(b['witnesses'])!=signature(p['witnesses']): errors.append('balanced_precision_nonfloor_witness_mismatch')
    rows=b['witnesses']
    if not rows: errors.append('no_nonfloor_witnesses')
    if any(r['absolute_admitted'] for r in rows): errors.append('nonfloor_witness_already_admitted')
    for r in rows:
        ft=r['first_base_trigger_step']
        if ft is None or int(r['step_index'])>=int(ft): errors.append(f'nonfloor_witness_not_pre_first_trigger:{r["target_key"]}:{r["step_index"]}:{ft}')
    by_scene={}
    for r in rows: by_scene.setdefault(r['target_key'],[]).append(r)
    # For the causal screen, begin teacher evaluation at the earliest clean first-deviation
    # opportunity in each seed scene.  Before that step Base must remain exact nominal.
    seeds={}
    for key,rr in sorted(by_scene.items()):
        earliest=min(int(x['step_index']) for x in rr)
        best=max(rr,key=lambda x:x['teacher_pcd_advantage'])
        seeds[key]={
            'start_step':earliest,'first_base_trigger_step':int(best['first_base_trigger_step']),
            'num_nonfloor_witness_rows':len(rr),'num_nonfloor_witness_decisions':len({int(x['step_index']) for x in rr}),
            'best_pretrigger_step':int(best['step_index']),'best_pretrigger_candidate_index':int(best['candidate_index']),
            'best_pretrigger_macro':best['macro'],'best_pretrigger_teacher_pcd_advantage':float(best['teacher_pcd_advantage']),
            'best_pretrigger_teacher_r_dep_star':float(best['teacher_r_dep_star']),
        }
    # Current evidence should be tiny; fail closed if it unexpectedly balloons because
    # that would invalidate the intended fast falsification screen design.
    if len(seeds)>8: errors.append(f'too_many_seed_scenes_for_screen:{len(seeds)}')
    out={
        'schema':'ocrap-v48.124.10.6-nonfloor-seed-cohort-v1','valid':not errors,'attribution_ready':not errors,'errors':errors,
        'scientific_version':'v48.124-OC-FMSA','publication_evidence':False,'teacher_r_dep_structural_floor':FLOOR,'floor_tolerance':TOL,'pcd_epsilon':EPS,
        'num_nonfloor_positive_rows':len(rows),'num_nonfloor_positive_decisions':len({(r['target_key'],r['step_index']) for r in rows}),
        'num_seed_scenes':len(seeds),'seeds':seeds,'witness_rows':rows,
        'interpretation':'These are non-floor candidate-level opportunities on Base-nominal states before each scene first Base intervention. Non-floor excludes the known exact 0.5 plateau but does not establish full physical point-identifiability under the V48.80 structural-interval truth contract; this is a falsification seed cohort, not a population-prevalence estimate.',
    }
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+'\n')
    Path(a.target_keys_output).write_text(json.dumps({'target_keys':sorted(seeds)},indent=2,sort_keys=True)+'\n')
    print(json.dumps({'valid':out['valid'],'num_seed_scenes':len(seeds),'num_rows':len(rows),'output':a.output},indent=2))
    return 0 if not errors else 30
if __name__=='__main__': raise SystemExit(main())
