#!/usr/bin/env python3
"""Mine a deterministic critical Contact subset for aspirational reference synthesis.

Uses only empirical visible traces.  Criticality explicitly includes crowding,
new-actor secondary collisions, same-partner re-contact, persistent contact and
repairable source off-road.  It is a qualitative curation score, never an
empirical evaluation metric.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import Any
HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0,str(HERE))
from contact_scene_diagnostics import analyze_contact_scene, criticality_score


def _scene_key(scene:dict[str,Any], env:dict[str,Any])->str:
    k=str(scene.get('target_key') or env.get('resume_key') or '')
    return k[7:] if k.startswith('target:') else k

def _load(path:Path)->dict[str,dict[str,Any]]:
    out={}
    with path.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            env=json.loads(line); sc=env.get('scene',env)
            if isinstance(sc,dict):
                k=_scene_key(sc,env)
                if k: out[k]=sc
    return out

def _preferred_keys(path:Path|None,n:int)->list[str]:
    if path is None or not path.is_file() or n<=0:return []
    d=json.loads(path.read_text(encoding='utf-8'))
    return [str(x.get('target_key')) for x in (d.get('selected') or []) if isinstance(x,dict) and x.get('target_key')][:n]

def _tags(raw:str)->list[str]:
    return [x.strip() for x in raw.split(',') if x.strip()]

def main()->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ocrap-trace',type=Path,required=True)
    ap.add_argument('--baseline',action='append',default=[])
    ap.add_argument('--anchor-manifest',type=Path,required=True)
    ap.add_argument('--preferred-selection',type=Path,default=None)
    ap.add_argument('--preserve-preferred-count',type=int,default=1)
    ap.add_argument('--max-scenes',type=int,default=18)
    ap.add_argument('--max-source-offroad-fraction',type=float,default=.15)
    ap.add_argument('--allow-source-offroad-repair',action='store_true')
    ap.add_argument('--max-repairable-source-offroad-fraction',type=float,default=.45)
    ap.add_argument('--coverage-tags',default='secondary_collision,crowded,recontact,source_offroad')
    ap.add_argument('--metric-dt-s',type=float,default=.1)
    ap.add_argument('--output-target-keys',type=Path,required=True)
    ap.add_argument('--output-anchor-manifest',type=Path,required=True)
    ap.add_argument('--output-audit',type=Path,required=True)
    args=ap.parse_args()
    if args.max_scenes<=0: raise SystemExit('--max-scenes must be positive')
    ocrap=_load(args.ocrap_trace); bmaps={}
    for spec in args.baseline:
        if '=' not in spec: raise SystemExit(f'invalid --baseline={spec!r}')
        n,raw=spec.split('=',1); bmaps[n.strip()]=_load(Path(raw))
    manifest=json.loads(args.anchor_manifest.read_text(encoding='utf-8'))
    anchors=list(manifest.get('anchors') or []); bykey={str(a.get('target_key')):a for a in anchors if a.get('target_key')}
    preferred=[k for k in _preferred_keys(args.preferred_selection,int(args.preserve_preferred_count)) if k in bykey and k in ocrap]
    rows=[]; dt=float(args.metric_dt_s)
    for key,anchor in bykey.items():
        if key not in ocrap: continue
        oq=analyze_contact_scene(ocrap[key],dt)
        if not oq.get('usable'): continue
        bq={m:analyze_contact_scene(mp[key],dt) for m,mp in bmaps.items() if key in mp}
        frac=float(oq.get('offroad_fraction') or 0.0)
        strict_bad=frac>float(args.max_source_offroad_fraction)
        repairable=bool(args.allow_source_offroad_repair and strict_bad and frac<=float(args.max_repairable_source_offroad_fraction))
        gross=bool(strict_bad and not repairable)
        rows.append({'target_key':key,'scene_id':anchor.get('scene_id'),'preferred':key in preferred,
                     'gross_source_offroad':gross,'repairable_source_offroad':repairable,
                     'criticality_score':criticality_score(oq,bq),'critical_tags':list(oq.get('critical_tags') or []),
                     'ocrap_source':oq,'baseline_source':bq})
    selected=[]
    for k in preferred:
        if k not in selected:selected.append(k)
    eligible=[r for r in rows if not r['gross_source_offroad'] and r['target_key'] not in selected]
    eligible.sort(key=lambda r:(-float(r['criticality_score']),str(r['target_key'])))
    # Soft tag coverage first, then score fill. This deliberately surfaces diverse
    # difficult failure modes while retaining the same downstream reality gates.
    for tag in _tags(args.coverage_tags):
        if len(selected)>=args.max_scenes: break
        cand=next((r for r in eligible if tag in r['critical_tags'] and r['target_key'] not in selected),None)
        if cand:selected.append(str(cand['target_key']))
    for r in eligible:
        if len(selected)>=args.max_scenes:break
        if r['target_key'] not in selected:selected.append(str(r['target_key']))
    if not selected: raise SystemExit('reference synthesis prefilter selected no scenes')
    sset=set(selected); fa=[a for a in anchors if str(a.get('target_key')) in sset]
    filtered=dict(manifest); filtered.update({'anchors':fa,'target_keys':sorted(sset),'num_selected_anchors':len(fa),
        'num_selected_scenes':len({str(a.get('scene_id')) for a in fa}),
        'selection_policy':str(filtered.get('selection_policy') or '')+'+critical_reference_synthesis_prefilter_v2',
        'reference_synthesis_prefilter_only':True,'source_anchor_manifest':str(args.anchor_manifest)})
    args.output_target_keys.parent.mkdir(parents=True,exist_ok=True); args.output_target_keys.write_text(json.dumps(selected,indent=2)+'\n')
    args.output_anchor_manifest.parent.mkdir(parents=True,exist_ok=True); args.output_anchor_manifest.write_text(json.dumps(filtered,indent=2)+'\n')
    audit={'event':'contact_reference_synthesis_subset_v2','max_scenes':args.max_scenes,'preferred_keys':preferred,
           'coverage_tags':_tags(args.coverage_tags),'allow_source_offroad_repair':bool(args.allow_source_offroad_repair),
           'max_repairable_source_offroad_fraction':float(args.max_repairable_source_offroad_fraction),
           'selected_keys':selected,'rows':rows,
           'scientific_note':'Qualitative compute prefilter only; criticality is not an empirical score. Final target-display physical/comparative gates remain separate.'}
    args.output_audit.parent.mkdir(parents=True,exist_ok=True); args.output_audit.write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({'event':audit['event'],'selected':len(selected),'preferred':preferred,'output':str(args.output_target_keys)}))
    return 0
if __name__=='__main__': raise SystemExit(main())
