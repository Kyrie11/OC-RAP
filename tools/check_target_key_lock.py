#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any


def load_keys(path: Path) -> set[str]:
    d=json.loads(path.read_text(encoding='utf-8'))
    def collect(x: Any, out:set[str]):
        if isinstance(x,str):
            if x.strip(): out.add(x.strip())
        elif isinstance(x,list):
            for v in x: collect(v,out)
        elif isinstance(x,dict):
            if x.get('target_key'):
                out.add(str(x['target_key']).strip()); return
            for k in ('target_keys','selected','items','scenes'):
                if k in x: collect(x[k],out)
    out:set[str]=set(); collect(d,out); return out


def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--expected',type=Path,required=True)
    ap.add_argument('--observed',type=Path,required=True)
    a=ap.parse_args()
    e,o=load_keys(a.expected),load_keys(a.observed)
    missing=sorted(e-o); extra=sorted(o-e)
    doc={'event':'target_key_lock_check','expected':len(e),'observed':len(o),'missing':missing[:20],'extra':extra[:20],'equal':not missing and not extra}
    print(json.dumps(doc))
    return 0 if doc['equal'] else 30
if __name__=='__main__': raise SystemExit(main())
