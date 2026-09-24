#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,os,zipfile
from pathlib import Path
EXCLUDE={'.jax_compilation_cache','__pycache__','.pytest_cache'}
def main()->None:
    ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); ap.add_argument('--output',required=True); ap.add_argument('--manifest',required=True); ap.add_argument('--exit-code',type=int,required=True); a=ap.parse_args()
    root=Path(a.root).resolve(); out=Path(a.output).resolve(); man=Path(a.manifest).resolve(); out.parent.mkdir(parents=True,exist_ok=True)
    files=[]
    for p in sorted(root.rglob('*')):
        if not p.is_file(): continue
        if any(q in EXCLUDE for q in p.relative_to(root).parts): continue
        if p.resolve() in {out,man}: continue
        files.append((p.relative_to(root).as_posix(),p))
    entries={}
    for rel,p in files:
        raw=p.read_bytes(); entries[rel]={'sha256':hashlib.sha256(raw).hexdigest(),'size':len(raw)}
    doc={'schema':'ocrap-v48.124.10.7.1-one-shot-action-realization-result-bundle-v1','root':str(root),'pipeline_exit_code':int(a.exit_code),'complete_exit_zero':int(a.exit_code)==0,'num_files':len(entries),'files':entries}
    man.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n')
    tmp=out.with_suffix(out.suffix+'.tmp'); tmp.unlink(missing_ok=True)
    with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for rel,p in files: z.write(p,rel)
        z.write(man,man.relative_to(root).as_posix())
    os.replace(tmp,out); print(json.dumps({'output':str(out),'manifest':str(man),'num_files':len(entries)+1,'pipeline_exit_code':int(a.exit_code)},indent=2))
if __name__=='__main__': main()
