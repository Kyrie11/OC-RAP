#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, importlib, json, os, sys
from pathlib import Path
os.environ.setdefault("OCRAP_V48_74_SIGNED_VIABILITY", "1")

def sha(p: Path) -> str:
    h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()

def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("--repo",default="."); ap.add_argument("--output",required=True); a=ap.parse_args()
    repo=Path(a.repo).resolve(); sys.path.insert(0,str(repo/'src')); sys.path.insert(0,str(repo))
    mod=importlib.import_module("ocrap.v48_74_signed_viability")
    frag=mod.runtime_contract_fragment()
    paths={}
    for name in ("ocrap","ocrap.data","ocrap.ocrap","ocrap.inference","ocrap.v48_74_signed_viability"):
        try:
            m=importlib.import_module(name); p=Path(m.__file__).resolve(); paths[name]={"path":str(p),"sha256":sha(p),"inside_repo":repo in p.parents}
        except Exception as e: paths[name]={"error":repr(e),"inside_repo":False}
    errors=[]
    if not frag["enabled"]: errors.append("V48.74 overlay not enabled")
    if frag["schema"]!=10 or frag["feature_dim"]!=22: errors.append("schema/dimension mismatch")
    if not all(v.get("inside_repo",False) for v in paths.values()): errors.append("runtime import outside repository")
    out={"valid":not errors,"attribution_ready":not errors,"errors":errors,"contract":frag,"runtime_modules":paths,"test_roots_read":False,"dataset_reconstruction":False}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps(out,indent=2,sort_keys=True)); return 0 if not errors else 30
if __name__=="__main__": raise SystemExit(main())
