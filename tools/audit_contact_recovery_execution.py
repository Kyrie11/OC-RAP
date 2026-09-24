#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
from typing import Any


def finite(x: Any):
    try:
        v=float(x)
    except Exception:
        return None
    return v if math.isfinite(v) else None


def _scene_key(scene: dict[str, Any], env: dict[str, Any]) -> str:
    key=str(scene.get('target_key') or env.get('resume_key') or '')
    if key.startswith('target:'):
        key=key[len('target:'):]
    if key:
        return key
    sid=str(scene.get('scene_id') or '')
    ti=scene.get('target_time_index')
    return f'{sid}:t{ti}' if sid and ti is not None else sid


def _load_keys(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    d=json.loads(path.read_text(encoding='utf-8'))
    if isinstance(d,list): rows=d
    elif isinstance(d,dict): rows=d.get('target_keys') or []
    else: rows=[]
    keys=[str(x) for x in rows if str(x)]
    if not keys or len(keys)!=len(set(keys)):
        raise SystemExit(f'target-key file is empty/invalid/duplicated: {path}')
    return keys


def _scientific_signature(scene: dict[str, Any]) -> str:
    # Resume/profile timing metadata may differ across equivalent duplicate rows;
    # compare only target-level scientific outputs used by this audit.
    payload={
        'target_key': scene.get('target_key'),
        'scene_id': scene.get('scene_id'),
        'target_time_index': scene.get('target_time_index'),
        'metric_summary': scene.get('metric_summary') or {},
        'selection_reason_counts': scene.get('selection_reason_counts') or {},
        'intervention_count': scene.get('intervention_count'),
        'num_decisions': scene.get('num_decisions'),
    }
    raw=json.dumps(payload,sort_keys=True,separators=(',',':'),ensure_ascii=False,default=str).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def load_scenes(path: Path, expected_keys: list[str] | None):
    allowed=set(expected_keys) if expected_keys is not None else None
    by_key: dict[str, dict[str, Any]]={}
    sigs: dict[str,str]={}
    duplicates=0
    ignored=0
    with path.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            env=json.loads(line); s=env.get('scene',env)
            if not isinstance(s,dict): continue
            key=_scene_key(s,env)
            if not key: continue
            if allowed is not None and key not in allowed:
                ignored+=1; continue
            sig=_scientific_signature(s)
            if key in by_key:
                duplicates+=1
                if sigs[key] != sig:
                    raise SystemExit(f'conflicting duplicate scientific rows for target {key}: {path}')
                continue
            by_key[key]=s; sigs[key]=sig
    if expected_keys is not None:
        missing=[k for k in expected_keys if k not in by_key]
        extra=sorted(set(by_key)-set(expected_keys))
        if missing or extra:
            raise SystemExit('scene journal/target lock mismatch: '+json.dumps({'missing':missing[:20],'extra':extra[:20]},ensure_ascii=False))
        rows=[by_key[k] for k in expected_keys]
    else:
        rows=[by_key[k] for k in sorted(by_key)]
    if not rows: raise SystemExit(f'no scenes in {path}')
    return rows, duplicates, ignored


def m(scene,key):
    return finite((scene.get('metric_summary') or {}).get(key))


def main():
    ap=argparse.ArgumentParser(description='Audit Contact closed-loop execution without rewarding runaway clearance.')
    ap.add_argument('--result',type=Path,required=True)
    ap.add_argument('--scenes',type=Path,default=None)
    ap.add_argument('--target-keys-file',type=Path,default=None)
    ap.add_argument('--terminal-clearance-cap-m',type=float,default=5.0)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    d=json.loads(a.result.read_text(encoding='utf-8'))
    sp=a.scenes or Path(str(a.result)+'.scenes.jsonl')
    expected=_load_keys(a.target_keys_file)
    rows, duplicate_rows, ignored_rows=load_scenes(sp,expected)
    n=len(rows)
    if expected is not None and int(d.get('num_scenes') or -1) != len(expected):
        raise SystemExit(f'result num_scenes={d.get("num_scenes")} does not match target lock={len(expected)}: {a.result}')
    controlled=[]; clean=[]; overlap=[]; capped=[]; runaway=[]; off_flags=[]; rec_flags=[]; esc_flags=[]
    for s in rows:
        off=(m(s,'offroad_any') or 0)>0.5
        rec=(m(s,'recontact_event') or 0)>0.5
        esc=(m(s,'post_contact_escape_event') or 0)>0.5
        term=m(s,'post_contact_terminal_clearance_m')
        ov=m(s,'post_contact_overlap_duration_s')
        ctrl=(not off) and (not rec) and esc and term is not None and term>=0.5
        off_flags.append(off); rec_flags.append(rec); esc_flags.append(esc)
        controlled.append(ctrl); clean.append((not off) and (not rec))
        if ov is not None: overlap.append(ov)
        if term is not None: capped.append(min(term,float(a.terminal_clearance_cap_m)))
        runaway.append(bool(off and term is not None and term>10.0))
    sel=d.get('selection_reason_counts') or {}
    intervention_rate=finite(d.get('intervention_rate')) or 0.0
    key_sha=hashlib.sha256(('\n'.join(expected or sorted(_scene_key(s,{}) for s in rows))+'\n').encode()).hexdigest()
    doc={
      'event':'contact_recovery_execution_audit_v2',
      'result':str(a.result), 'scene_journal':str(sp), 'num_scenes':n,
      'target_keys_file':str(a.target_keys_file) if a.target_keys_file else None,
      'target_keys_sha256':key_sha,
      'duplicate_equivalent_journal_rows_ignored':duplicate_rows,
      'out_of_cohort_journal_rows_ignored':ignored_rows,
      'method':d.get('method'), 'run_fingerprint':d.get('run_fingerprint'),
      'intervention_rate':intervention_rate,
      'intervention_scene_rate':finite(d.get('intervention_scene_rate')),
      'selection_reason_counts':sel,
      'offroad_scene_count':sum(off_flags), 'offroad_scene_rate':sum(off_flags)/n,
      'recontact_scene_count':sum(rec_flags), 'recontact_scene_rate':sum(rec_flags)/n,
      'post_contact_escape_scene_count':sum(esc_flags), 'post_contact_escape_scene_rate':sum(esc_flags)/n,
      'controlled_recovery_success_count':sum(controlled),
      'controlled_recovery_success_rate':sum(controlled)/n,
      'clean_no_offroad_no_recontact_count':sum(clean),
      'clean_no_offroad_no_recontact_rate':sum(clean)/n,
      'mean_post_contact_overlap_duration_s':sum(overlap)/len(overlap) if overlap else None,
      'mean_terminal_clearance_capped_m':sum(capped)/len(capped) if capped else None,
      'runaway_offroad_terminal_gt10m_count':sum(runaway),
      'runaway_offroad_terminal_gt10m_rate':sum(runaway)/n,
      'terminal_clearance_cap_m':float(a.terminal_clearance_cap_m),
      'scientific_note':'Counts/rates are recomputed from one unique target-locked scene per anchor. Equivalent resume duplicates are ignored; conflicting duplicates fail closed. Controlled success requires on-road, no re-contact, sustained post-contact escape, and terminal clearance >=0.5 m.',
    }
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(doc,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(doc,indent=2))
    return 0
if __name__=='__main__': raise SystemExit(main())
