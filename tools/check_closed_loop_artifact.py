#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any


def read(path: Path) -> dict[str, Any] | None:
    try:
        x=json.loads(path.read_text(encoding='utf-8')); return x if isinstance(x,dict) else None
    except Exception:
        return None


def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--quiet',action='store_true'); args=ap.parse_args()
    p=args.output; prog=read(p.with_suffix(p.suffix+'.progress.json')); result=read(p); journal=p.with_suffix(p.suffix+'.scenes.jsonl')
    complete=bool(result and prog and prog.get('status')=='complete' and journal.is_file())
    if complete and result.get('bucket_target_count') not in (None,0):
        complete=int(result.get('num_scenes') or 0)==int(result.get('bucket_target_count') or 0)
    doc={'event':'closed_loop_artifact_check','output':str(p),'complete':complete,'result_exists':p.is_file(),'journal_exists':journal.is_file(),'progress_status':prog.get('status') if prog else None,'num_scenes':result.get('num_scenes') if result else None,'bucket_target_count':result.get('bucket_target_count') if result else None}
    if not args.quiet: print(json.dumps(doc))
    return 0 if complete else 1
if __name__=='__main__': raise SystemExit(main())
