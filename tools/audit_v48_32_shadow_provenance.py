#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,re
from pathlib import Path
from collections import Counter
from ocrap.models.data import iter_sample_paths_many, scalar_metadata_for_path


def role(text:str)->str:
    x=text.lower()
    if 'validation_interactive' in x:return 'validation_interactive'
    if re.search(r'(^|[/_])validation([/_]|$)',x):return 'validation'
    if 'training' in x:return 'training'
    if 'testing' in x or '/test' in x:return 'test'
    return 'unknown'


def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--targets',required=True)
    ap.add_argument('--womd-source',required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--expected-source-role',default='validation')
    args=ap.parse_args()
    paths=iter_sample_paths_many(args.targets)
    roles=Counter(); id_sources=Counter(); official=legacy=indexed=0; examples=[]
    for p in paths:
        r=str(scalar_metadata_for_path(p,'womd_source_role','') or 'unknown'); roles[r]+=1
        ids=str(scalar_metadata_for_path(p,'scenario_id_source','') or 'unknown'); id_sources[ids]+=1
        if str(scalar_metadata_for_path(p,'official_scenario_id','') or ''): official+=1
        scene=str(scalar_metadata_for_path(p,'scene_id','') or '')
        if str(scalar_metadata_for_path(p,'legacy_scenario_id','') or '') or scene.startswith('waymax_'): legacy+=1
        idx=scalar_metadata_for_path(p,'source_scenario_index',-1)
        try: idx=int(float(idx))
        except Exception: idx=-1
        if idx>=0 or re.search(r'__wx\d{8}$',scene): indexed+=1
        if len(examples)<5: examples.append(scene)
    raw_role=role(args.womd_source)
    expected=args.expected_source_role
    known={r for r in roles if r not in {'','unknown'}}
    valid=(expected == 'auto' or raw_role==expected) and (not known or raw_role in known)
    result={'event':'v48_32_shadow_provenance_audit','targets':args.targets,'womd_source':args.womd_source,
            'raw_source_role':raw_role,'expected_source_role':expected,'target_metadata_roles':dict(roles),
            'scenario_id_sources':dict(id_sources),'num_sample_paths':len(paths),
            'official_id_samples':official,'legacy_id_samples':legacy,'source_index_migratable_samples':indexed,
            'scene_examples':examples,'valid':valid,
            'migration_mode_required': official==0 and indexed>0,
            'interpretation':'Official WOMD scenario/id is primary. Source-index matching is allowed only for legacy targets on the identical split/shard order.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))
    return 0 if valid else 4
if __name__=='__main__': raise SystemExit(main())
