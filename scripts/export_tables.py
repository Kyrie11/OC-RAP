#!/usr/bin/env python
from __future__ import annotations

# Allow running scripts directly from a source checkout without pip install -e .
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import argparse, json, csv
from pathlib import Path

if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--eval-dirs", nargs="+", required=True); ap.add_argument("--output", required=True)
    args=ap.parse_args(); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for d in args.eval_dirs:
        p=Path(d)/"metrics.json"
        if p.exists(): rows.append(json.loads(p.read_text()))
    with (out/"main_closed_loop.csv").open("w",newline="") as f:
        keys=sorted(set().union(*[r.keys() for r in rows])) if rows else ["method"]
        wr=csv.DictWriter(f,fieldnames=keys); wr.writeheader(); wr.writerows(rows)
    print(f"wrote {out/'main_closed_loop.csv'}")
