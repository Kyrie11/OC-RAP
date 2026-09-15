#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path

def sha(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--historical-adjudication',type=Path,required=True)
    ap.add_argument('--balanced-checkpoint',type=Path,required=True)
    ap.add_argument('--precision-checkpoint',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args(); hist=json.loads(a.historical_adjudication.read_text()); prov=hist.get('provenance') or {}; errors=[]; rows={}
    for variant,path,key in [('balanced',a.balanced_checkpoint,'balanced_checkpoint'),('precision',a.precision_checkpoint,'precision_checkpoint')]:
        exp=prov.get(key) or {}
        if not path.is_file():
            errors.append(f'{variant}:checkpoint_missing'); continue
        got={'path':str(path.resolve()),'sha256':sha(path),'size':path.stat().st_size}
        rows[variant]={'expected':{'sha256':exp.get('sha256'),'size':exp.get('size')},'actual':got}
        if not exp.get('sha256') or got['sha256']!=exp.get('sha256'): errors.append(f'{variant}:checkpoint_sha_mismatch')
        if exp.get('size') is not None and int(got['size'])!=int(exp.get('size')): errors.append(f'{variant}:checkpoint_size_mismatch')
    out={'schema':'ocrap-frozen-checkpoint-contract-v1','valid':not errors,'attribution_ready':not errors,'historical_engineering_version':hist.get('engineering_version'),'historical_scientific_version':hist.get('scientific_version'),'checkpoints':rows,'errors':errors}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'valid':out['valid'],'errors':errors,'output':str(a.output)},indent=2)); return 0 if out['valid'] else 30
if __name__=='__main__': raise SystemExit(main())
