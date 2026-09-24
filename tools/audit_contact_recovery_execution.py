#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from pathlib import Path
from typing import Any


def finite(x: Any):
    try:
        v=float(x)
    except Exception:
        return None
    return v if math.isfinite(v) else None


def load_scenes(path: Path):
    rows=[]
    with path.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            e=json.loads(line); s=e.get('scene',e)
            if isinstance(s,dict): rows.append(s)
    return rows


def m(scene,key):
    return finite((scene.get('metric_summary') or {}).get(key))


def main():
    ap=argparse.ArgumentParser(description='Audit Contact closed-loop execution without rewarding runaway clearance.')
    ap.add_argument('--result',type=Path,required=True)
    ap.add_argument('--scenes',type=Path,default=None)
    ap.add_argument('--terminal-clearance-cap-m',type=float,default=5.0)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    d=json.loads(a.result.read_text(encoding='utf-8'))
    sp=a.scenes or Path(str(a.result)+'.scenes.jsonl')
    rows=load_scenes(sp)
    if not rows: raise SystemExit(f'no scenes in {sp}')
    controlled=[]; clean=[]; overlap=[]; capped=[]; runaway=[]
    for s in rows:
        off=(m(s,'offroad_any') or 0)>0.5
        rec=(m(s,'recontact_event') or 0)>0.5
        esc=(m(s,'post_contact_escape_event') or 0)>0.5
        term=m(s,'post_contact_terminal_clearance_m')
        ov=m(s,'post_contact_overlap_duration_s')
        ctrl=(not off) and (not rec) and esc and term is not None and term>=0.5
        controlled.append(ctrl); clean.append((not off) and (not rec))
        if ov is not None: overlap.append(ov)
        if term is not None: capped.append(min(term,float(a.terminal_clearance_cap_m)))
        runaway.append(bool(off and term is not None and term>10.0))
    n=len(rows)
    sel=d.get('selection_reason_counts') or {}
    intervention_rate=finite(d.get('intervention_rate')) or 0.0
    doc={
      'event':'contact_recovery_execution_audit_v1',
      'result':str(a.result), 'scene_journal':str(sp), 'num_scenes':n,
      'method':d.get('method'), 'run_fingerprint':d.get('run_fingerprint'),
      'intervention_rate':intervention_rate,
      'intervention_scene_rate':finite(d.get('intervention_scene_rate')),
      'selection_reason_counts':sel,
      'offroad_scene_rate':finite(d.get('offroad_scene_rate')),
      'recontact_scene_rate':finite(d.get('recontact_scene_rate')),
      'post_contact_escape_scene_rate':finite(d.get('post_contact_escape_scene_rate')),
      'controlled_recovery_success_count':sum(controlled),
      'controlled_recovery_success_rate':sum(controlled)/n,
      'clean_no_offroad_no_recontact_count':sum(clean),
      'clean_no_offroad_no_recontact_rate':sum(clean)/n,
      'mean_post_contact_overlap_duration_s':sum(overlap)/len(overlap) if overlap else None,
      'mean_terminal_clearance_capped_m':sum(capped)/len(capped) if capped else None,
      'offroad_scene_count':round((finite(d.get('offroad_scene_rate')) or 0.0)*n),
      'recontact_scene_count':round((finite(d.get('recontact_scene_rate')) or 0.0)*n),
      'post_contact_escape_scene_count':round((finite(d.get('post_contact_escape_scene_rate')) or 0.0)*n),
      'runaway_offroad_terminal_gt10m_count':sum(runaway),
      'runaway_offroad_terminal_gt10m_rate':sum(runaway)/n,
      'terminal_clearance_cap_m':float(a.terminal_clearance_cap_m),
      'scientific_note':'controlled success requires on-road, no re-contact, sustained post-contact escape, and terminal clearance >=0.5 m; capped terminal clearance prevents runaway/off-road distance from inflating the audit score.',
    }
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(doc,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(doc,indent=2))
    return 0
if __name__=='__main__': raise SystemExit(main())
