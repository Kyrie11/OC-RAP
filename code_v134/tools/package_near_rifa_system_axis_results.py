#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, zipfile
from pathlib import Path

ENGINEERING_VERSION="v48.124.10-OC-FMSA-NEAR-RIFA-SYSTEM-AXIS-AUDIT"

def sha(p: Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def main()->int:
    ap=argparse.ArgumentParser(description='Package V48.124.10 Near system-axis diagnostic artifacts with SHA manifest.')
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--manifest',type=Path,required=True)
    a=ap.parse_args(); root=a.root.resolve(); out=a.output.resolve(); manifest_path=a.manifest.resolve()
    include=[]
    for p in sorted(root.rglob('*')):
        if not p.is_file(): continue
        rel=p.relative_to(root).as_posix()
        if rel.startswith('jax_cache/') or '/.jax_compilation_cache/' in rel or rel.endswith('.lock'):
            continue
        if '/scene_journal/' in rel or rel.startswith('scene_journal/'):
            continue
        # Keep the canonical subset/fused/comparison/provenance artifacts; omit bulky transient partials.
        if p.name.endswith('.partial.json') or p.name.endswith('.resume.json'):
            continue
        include.append((p,rel))
    files={rel:{'sha256':sha(p),'size':p.stat().st_size} for p,rel in include}
    doc={
      'schema':'ocrap-v48.124.10-near-rifa-system-axis-result-manifest-v1',
      'engineering_version':ENGINEERING_VERSION,'valid':True,'num_files':len(files),'files':files,
    }
    manifest_path.parent.mkdir(parents=True,exist_ok=True)
    manifest_path.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n')
    # Manifest is included under a stable top-level name even when stored outside root.
    out.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p,rel in include: z.write(p,arcname=rel)
        z.write(manifest_path,arcname=manifest_path.name)
    print(json.dumps({'valid':True,'output':str(out),'manifest':str(manifest_path),'num_files':len(files),'zip_sha256':sha(out)},indent=2))
    return 0
if __name__=='__main__': raise SystemExit(main())
