#!/usr/bin/env python3
"""Validate dedicated adaptation/dev/certificate role labels and scene disjointness."""
from __future__ import annotations
import argparse, csv, json
from collections import Counter
from pathlib import Path

ROLES = {
    "evidence_adapt_train_near_contact": "evidence_adapt_train",
    "evidence_adapt_train_contact": "evidence_adapt_train",
    "evidence_adapt_dev_near_contact": "evidence_adapt_dev",
    "evidence_adapt_dev_contact": "evidence_adapt_dev",
    "certificate_pool_near_contact": "certificate_pool",
    "certificate_pool_contact": "certificate_pool",
}

def read_root(root: Path, expected: str) -> dict:
    manifest=root/'manifest.csv'
    if not manifest.is_file(): raise SystemExit(f'missing manifest: {manifest}')
    splits=Counter(); scenes=set(); rows=0
    with manifest.open(newline='',encoding='utf-8') as f:
        for row in csv.DictReader(f):
            rows+=1; splits[str(row.get('split_id','')).strip()]+=1
            scene=str(row.get('scene_id','')).strip()
            if scene: scenes.add(scene)
    bad={k:v for k,v in splits.items() if k != expected}
    if rows <= 0 or bad: raise SystemExit(f'bad protocol role in {root}: expected={expected} rows={rows} splits={dict(splits)}')
    return {'root':str(root),'rows':rows,'scenes':scenes,'split_counts':dict(splits)}

def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument('--protocol-root',required=True); ap.add_argument('--output',required=True)
    args=ap.parse_args(); root=Path(args.protocol_root); docs={}
    for name,role in ROLES.items(): docs[name]=read_root(root/name,role)
    pairs=[
      ('evidence_adapt_train_near_contact','evidence_adapt_dev_near_contact'),
      ('evidence_adapt_train_near_contact','certificate_pool_near_contact'),
      ('evidence_adapt_dev_near_contact','certificate_pool_near_contact'),
      ('evidence_adapt_train_contact','evidence_adapt_dev_contact'),
      ('evidence_adapt_train_contact','certificate_pool_contact'),
      ('evidence_adapt_dev_contact','certificate_pool_contact'),
    ]
    overlaps={f'{a}__{b}':len(docs[a]['scenes'] & docs[b]['scenes']) for a,b in pairs}
    if any(overlaps.values()): raise SystemExit(f'dedicated protocol scene leakage: {overlaps}')
    out={'valid':True,'protocol_root':str(root),'roles':{},'scene_overlaps':overlaps}
    for k,v in docs.items(): out['roles'][k]={x:y for x,y in v.items() if x!='scenes'}|{'num_scenes':len(v['scenes'])}
    op=Path(args.output); op.parent.mkdir(parents=True,exist_ok=True); op.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out))
    return 0
if __name__=='__main__': raise SystemExit(main())
