#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path
import zipfile

EXCLUDE_PARTS={'.jax_compilation_cache','__pycache__','.pytest_cache'}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); ap.add_argument('--output',required=True); ap.add_argument('--manifest',required=True); ap.add_argument('--exit-code',type=int,required=True); args=ap.parse_args()
    root=Path(args.root).resolve(); out=Path(args.output).resolve(); manifest_path=Path(args.manifest).resolve(); out.parent.mkdir(parents=True,exist_ok=True)
    files=[]
    for p in sorted(root.rglob('*')):
        if not p.is_file(): continue
        rel=p.relative_to(root).as_posix()
        if any(part in EXCLUDE_PARTS for part in p.relative_to(root).parts): continue
        if p.resolve() in {out,manifest_path}: continue
        files.append((rel,p))
    entries={}
    for rel,p in files:
        raw=p.read_bytes(); entries[rel]={'sha256':hashlib.sha256(raw).hexdigest(),'size':len(raw)}
    manifest={
        'schema':'ocrap-v48.124.10.3-candidate-quality-result-bundle-v1',
        'root':str(root),'pipeline_exit_code':int(args.exit_code),'complete_exit_zero':int(args.exit_code)==0,
        'num_files':len(entries),'files':entries,
    }
    manifest_path.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    # include the manifest itself last and rebuild archive atomically
    tmp=out.with_suffix(out.suffix+'.tmp')
    if tmp.exists(): tmp.unlink()
    with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for rel,p in files: z.write(p,rel)
        z.write(manifest_path,manifest_path.relative_to(root).as_posix())
    os.replace(tmp,out)
    print(json.dumps({'output':str(out),'manifest':str(manifest_path),'num_files':len(entries)+1,'pipeline_exit_code':int(args.exit_code)},indent=2))
if __name__=='__main__': main()
