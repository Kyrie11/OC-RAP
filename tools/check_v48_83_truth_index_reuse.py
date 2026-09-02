#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

def sha(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for z in iter(lambda:f.read(1<<20), b''): h.update(z)
    return h.hexdigest()

def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument('--index',type=Path,required=True); ap.add_argument('--summary',type=Path,required=True); ap.add_argument('--roles',required=True); a=ap.parse_args()
    errs=[]
    if not a.index.is_file(): errs.append('index missing')
    if not a.summary.is_file(): errs.append('summary missing')
    d={}
    if not errs:
        try:d=json.loads(a.summary.read_text())
        except Exception as e: errs.append(f'summary parse: {e}')
    roles=[x for x in a.roles.split(',') if x]
    if d:
        if not d.get('valid'): errs.append('summary invalid')
        if d.get('dataset_reconstruction'): errs.append('dataset reconstruction true')
        if d.get('test_roots_read'): errs.append('test roots read')
        if d.get('schema')!='ocrap-v48.80-interval-truth-index-summary-v1': errs.append('wrong schema')
        if d.get('output_sha256')!=sha(a.index): errs.append('sha mismatch')
        rr=d.get('roles') or {}
        for r in roles:
            if r not in rr: errs.append(f'missing role {r}')
            elif float(rr[r].get('informative_fraction',0))!=1.0: errs.append(f'role {r} not fully informative')
    print(json.dumps({'valid':not errs,'errors':errs,'reused':not errs,'index':str(a.index),'summary':str(a.summary)}))
    return 0 if not errs else 30
if __name__=='__main__': raise SystemExit(main())
