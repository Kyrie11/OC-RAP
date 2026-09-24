#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path


def main() -> int:
    ap=argparse.ArgumentParser(description="Summarize a small held-out Contact selector diagnostic; does not freeze a publication profile.")
    ap.add_argument('--strict',type=Path,required=True)
    ap.add_argument('--guarded',type=Path,required=True)
    ap.add_argument('--calibrated',type=Path,default=None)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    docs={'strict_abs':json.loads(a.strict.read_text()),'guarded_fallback':json.loads(a.guarded.read_text())}
    if a.calibrated and a.calibrated.is_file(): docs['calibrated_guarded']=json.loads(a.calibrated.read_text())
    b=docs['strict_abs']; bn=int(b.get('num_scenes') or 0)
    def cnt(d,k,ratek):
        if d.get(k) is not None:return int(d[k])
        return round(float(d.get(ratek) or 0.0)*int(d.get('num_scenes') or 0))
    rows=[]
    for name,d in docs.items():
        row={
          'name':name,'num_scenes':int(d.get('num_scenes') or 0),'intervention_rate':float(d.get('intervention_rate') or 0),
          'offroad_count':cnt(d,'offroad_scene_count','offroad_scene_rate'),'offroad_rate':float(d.get('offroad_scene_rate') or 0),
          'recontact_count':cnt(d,'recontact_scene_count','recontact_scene_rate'),'recontact_rate':float(d.get('recontact_scene_rate') or 0),
          'controlled_count':cnt(d,'controlled_recovery_success_count','controlled_recovery_success_rate'),'controlled_rate':float(d.get('controlled_recovery_success_rate') or 0),
          'mean_overlap_s':d.get('mean_post_contact_overlap_duration_s'),'runaway_count':cnt(d,'runaway_offroad_terminal_gt10m_count','runaway_offroad_terminal_gt10m_rate'),
          'selection_reason_counts':d.get('selection_reason_counts') or {},
        }
        rows.append(row)
    base=rows[0]
    assessments=[]
    for row in rows[1:]:
        safety_ok=row['offroad_count']<=base['offroad_count'] and row['recontact_count']<=base['recontact_count'] and row['runaway_count']<=base['runaway_count']
        behavior_ok=row['intervention_rate']>=0.01
        ctrl_gain=row['controlled_count']-base['controlled_count']
        ov0=base['mean_overlap_s']; ov1=row['mean_overlap_s']
        overlap_gain=(ov0 is not None and ov1 is not None and float(ov1)<=float(ov0)-0.10)
        efficacy_ok=ctrl_gain>=1 or overlap_gain
        assessments.append({'name':row['name'],'safety_ok':safety_ok,'behavior_ok':behavior_ok,'efficacy_ok':efficacy_ok,'controlled_scene_gain':ctrl_gain,'overlap_improved_by_at_least_0.10s':overlap_gain,'promising':bool(safety_ok and behavior_ok and efficacy_ok)})
    promising=[x['name'] for x in assessments if x['promising']]
    out={
      'event':'contact_recovery_quick_diagnostic_v135','diagnostic_only':True,'publication_profile_frozen':False,
      'num_validation_anchors':bn,'rows':rows,'assessments':assessments,'promising_profiles':promising,
      'recommended_next_step':('Rerun OC-RAP only on the existing frozen Contact cohort with the best promising validation profile; existing baseline closed-loop results remain reusable because target lock/horizon/baseline configs are unchanged.' if promising else 'Do not touch test_contact yet. Guarded selector did not show a clean validation signal on this diagnostic subset; inspect audits or increase diagnostic target sample before considering observed-contact retraining.'),
      'scientific_note':'This 100-target-style diagnostic is a compute-saving screen, not a publication profile-selection study. It is intended to decide whether a larger validation or frozen-cohort OC-RAP-only confirmation is worth the cost.'
    }
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
    return 0
if __name__=='__main__': raise SystemExit(main())
