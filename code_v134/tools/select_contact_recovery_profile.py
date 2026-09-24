#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(description='Freeze a Contact selector profile using validation-only audits.')
    ap.add_argument('--audit',action='append',default=[],help='NAME=path')
    ap.add_argument('--min-intervention-rate',type=float,default=0.01)
    ap.add_argument('--max-offroad-regression',type=float,default=0.02)
    ap.add_argument('--max-recontact-regression',type=float,default=0.02)
    ap.add_argument('--min-controlled-gain',type=float,default=0.05)
    ap.add_argument('--min-overlap-improvement-s',type=float,default=0.10)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    docs={}
    for spec in a.audit:
        if '=' not in spec: raise SystemExit(f'bad --audit {spec}')
        name,p=spec.split('=',1); docs[name]=json.load(open(p,encoding='utf-8'))
    if 'strict_abs' not in docs: raise SystemExit('strict_abs validation audit is required')
    base=docs['strict_abs']
    bo=float(base.get('offroad_scene_rate') or 0); br=float(base.get('recontact_scene_rate') or 0)
    bc=float(base.get('controlled_recovery_success_rate') or 0); bov=float(base.get('mean_post_contact_overlap_duration_s') or 1e9)
    candidates=[]
    for name,d in docs.items():
        if name=='strict_abs': continue
        interv=float(d.get('intervention_rate') or 0)
        off=float(d.get('offroad_scene_rate') or 0); rec=float(d.get('recontact_scene_rate') or 0)
        ctrl=float(d.get('controlled_recovery_success_rate') or 0); ov=float(d.get('mean_post_contact_overlap_duration_s') or 1e9)
        safety_ok=off <= bo + a.max_offroad_regression + 1e-12 and rec <= br + a.max_recontact_regression + 1e-12
        behavior_ok=interv >= a.min_intervention_rate - 1e-12
        efficacy_ok=(ctrl >= bc + a.min_controlled_gain - 1e-12) or (ov <= bov - a.min_overlap_improvement_s + 1e-12)
        candidates.append({'name':name,'audit':d,'safety_ok':safety_ok,'behavior_ok':behavior_ok,'efficacy_ok':efficacy_ok})
    valid=[x for x in candidates if x['safety_ok'] and x['behavior_ok'] and x['efficacy_ok']]
    # Validation-only lexicographic choice: controlled success first, then offroad/recontact,
    # then overlap duration, then capped terminal clearance, then intervention rate.
    valid.sort(key=lambda x:(
        -float(x['audit'].get('controlled_recovery_success_rate') or 0),
        float(x['audit'].get('offroad_scene_rate') or 0),
        float(x['audit'].get('recontact_scene_rate') or 0),
        float(x['audit'].get('mean_post_contact_overlap_duration_s') or 1e9),
        -float(x['audit'].get('mean_terminal_clearance_capped_m') or 0),
        -float(x['audit'].get('intervention_rate') or 0),
        x['name'],
    ))
    chosen=valid[0]['name'] if valid else None
    profiles={
      'strict_abs': {'ocrap_selector':'lcb_constrained','require_absolute_admission_for_intervention':True},
      'guarded_fallback': {'ocrap_selector':'lcb_constrained','require_absolute_admission_for_intervention':False},
      'calibrated_guarded': {'ocrap_selector':'calibrated_constrained','require_absolute_admission_for_intervention':False},
    }
    out={
      'event':'contact_recovery_profile_selection_v1','valid':bool(chosen),'selected_profile':chosen,
      'selected_runtime':profiles.get(chosen), 'baseline_profile':'strict_abs',
      'validation_only':True,
      'selection_rule':{
        'min_intervention_rate':a.min_intervention_rate,
        'max_offroad_regression':a.max_offroad_regression,
        'max_recontact_regression':a.max_recontact_regression,
        'min_controlled_gain':a.min_controlled_gain,
        'min_overlap_improvement_s':a.min_overlap_improvement_s,
      },
      'strict_abs_audit':base,
      'candidate_assessments':candidates,
      'note':('A non-strict Contact profile is frozen only if it causes real interventions and improves controlled recovery/overlap on held-out validation without materially worsening off-road or re-contact. No test result is used.' if chosen else 'No alternative profile met the validation-only safety/efficacy rule. Do not retune on test; consider an observed-contact recovery model/dataset instead.'),
    }
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(out,indent=2))
    return 0 if chosen else 30
if __name__=='__main__': raise SystemExit(main())
