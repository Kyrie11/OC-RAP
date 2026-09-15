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
    ap=argparse.ArgumentParser(); ap.add_argument('--repo',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--manifest',type=Path,required=True); a=ap.parse_args()
    repo=a.repo.resolve(); skip={'.git','.pytest_cache','__pycache__','.mypy_cache','.ruff_cache'}; files={}
    for p in sorted(repo.rglob('*')):
        if not p.is_file() or any(part in skip for part in p.parts) or p.suffix=='.pyc': continue
        rel=p.relative_to(repo).as_posix(); files[rel]={'sha256':sha(p),'size':p.stat().st_size}
    tree=hashlib.sha256('\n'.join(f"{rel}\t{files[rel]['sha256']}\t{files[rel]['size']}" for rel in sorted(files)).encode()).hexdigest()
    manifest={'schema':'ocrap-runtime-source-snapshot-v1','repo':str(repo),'num_files':len(files),'tree_sha256':tree,'files':files}
    a.manifest.parent.mkdir(parents=True,exist_ok=True); a.manifest.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists(): a.output.unlink()
    with zipfile.ZipFile(a.output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for rel in sorted(files): z.write(repo/rel,rel)
    print(json.dumps({'output':str(a.output),'manifest':str(a.manifest),'num_files':len(files),'tree_sha256':tree},indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
