#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path


def _keys(path: Path) -> list[str]:
    d=json.loads(path.read_text(encoding='utf-8'))
    rows=d if isinstance(d,list) else (d.get('target_keys') or []) if isinstance(d,dict) else []
    out=[str(x) for x in rows if str(x)]
    if not out or len(out)!=len(set(out)): raise SystemExit(f'invalid target keys: {path}')
    return out


def main() -> int:
    ap=argparse.ArgumentParser(description='Summarize a small held-out Contact selector diagnostic; does not freeze a publication profile.')
    ap.add_argument('--strict',type=Path,required=True); ap.add_argument('--guarded',type=Path,required=True)
    ap.add_argument('--calibrated',type=Path,default=None); ap.add_argument('--target-keys-file',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    docs={'strict_abs':json.loads(a.strict.read_text()),'guarded_fallback':json.loads(a.guarded.read_text())}
    if a.calibrated and a.calibrated.is_file(): docs['calibrated_guarded']=json.loads(a.calibrated.read_text())
    expected_n=len(_keys(a.target_keys_file))
    for name,d in docs.items():
        if int(d.get('num_scenes') or -1)!=expected_n:
            raise SystemExit(f'{name}: audit num_scenes={d.get("num_scenes")} != target lock={expected_n}')
    def cnt(d,k): return int(d.get(k) or 0)
    rows=[]
    for name,d in docs.items():
        rows.append({
          'name':name,'num_scenes':expected_n,'intervention_rate':float(d.get('intervention_rate') or 0),
          'offroad_count':cnt(d,'offroad_scene_count'),'offroad_rate':float(d.get('offroad_scene_rate') or 0),
          'recontact_count':cnt(d,'recontact_scene_count'),'recontact_rate':float(d.get('recontact_scene_rate') or 0),
          'controlled_count':cnt(d,'controlled_recovery_success_count'),'controlled_rate':float(d.get('controlled_recovery_success_rate') or 0),
          'mean_overlap_s':d.get('mean_post_contact_overlap_duration_s'),'runaway_count':cnt(d,'runaway_offroad_terminal_gt10m_count'),
          'selection_reason_counts':d.get('selection_reason_counts') or {},
          'duplicate_equivalent_journal_rows_ignored':int(d.get('duplicate_equivalent_journal_rows_ignored') or 0),
        })
    base=rows[0]; assessments=[]
    for row in rows[1:]:
        safety_ok=row['offroad_count']<=base['offroad_count'] and row['recontact_count']<=base['recontact_count'] and row['runaway_count']<=base['runaway_count']
        behavior_ok=row['intervention_rate']>=0.01
        ctrl_gain=row['controlled_count']-base['controlled_count']
        ov0=base['mean_overlap_s']; ov1=row['mean_overlap_s']
        overlap_gain=(ov0 is not None and ov1 is not None and float(ov1)<=float(ov0)-0.10)
        efficacy_ok=ctrl_gain>=1 or overlap_gain
        assessments.append({'name':row['name'],'safety_ok':safety_ok,'behavior_ok':behavior_ok,'efficacy_ok':efficacy_ok,'controlled_scene_gain':ctrl_gain,'overlap_improved_by_at_least_0.10s':overlap_gain,'promising':bool(safety_ok and behavior_ok and efficacy_ok)})
    promising=[x['name'] for x in assessments if x['promising']]
    out={'event':'contact_recovery_quick_diagnostic_v137','diagnostic_only':True,'publication_profile_frozen':False,'num_validation_anchors':expected_n,'rows':rows,'assessments':assessments,'promising_profiles':promising,
      'recommended_next_step':('Confirm the best promising profile on the existing frozen Contact cohort by rerunning OC-RAP only; baseline results remain paired/reusable.' if promising else 'Do not promote guarded/calibrated execution to the reported Contact configuration from this screen. If neither existing selector profile improves clean recovery on the fixed validation anchors, retain the stable Contact configuration rather than tuning on test for prettier videos.'),
      'scientific_note':'Target count comes from the fixed exact-a0 target lock, not journal row count. All profile audits must match the same target set; equivalent resume duplicates are ignored and conflicting duplicates fail closed.'}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
