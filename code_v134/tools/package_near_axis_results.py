#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,zipfile
from pathlib import Path

def sha(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True); ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--manifest',type=Path,required=True); ap.add_argument('--exit-code',type=int,default=0)
    a=ap.parse_args(); root=a.root.resolve(); files={}
    excluded_parts={'jax_cache','.pytest_cache','__pycache__'}
    for p in sorted(root.rglob('*')):
        if not p.is_file() or any(x in excluded_parts for x in p.parts): continue
        rel=p.relative_to(root).as_posix()
        if p.resolve()==a.manifest.resolve(): continue
        files[rel]={'sha256':sha(p),'size':p.stat().st_size}
    manifest={'schema':'ocrap-near-axis-result-bundle-v2','root':str(root),'pipeline_exit_code':int(a.exit_code),'complete_exit_zero':int(a.exit_code)==0,'num_files':len(files),'files':files}
    a.manifest.parent.mkdir(parents=True,exist_ok=True); a.manifest.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    files[a.manifest.relative_to(root).as_posix()]={'sha256':sha(a.manifest),'size':a.manifest.stat().st_size}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists(): a.output.unlink()
    with zipfile.ZipFile(a.output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for rel in sorted(files): z.write(root/rel,rel)
    print(json.dumps({'output':str(a.output),'manifest':str(a.manifest),'num_files':len(files),'pipeline_exit_code':a.exit_code},indent=2))
    return 0
if __name__=='__main__': raise SystemExit(main())
