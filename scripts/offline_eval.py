#!/usr/bin/env python
from __future__ import annotations

# Allow running scripts directly from a source checkout without pip install -e .
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import argparse, json
from pathlib import Path
from recap.teacher.dataset_writer import read_dataset
from recap.evaluation.offline_eval import evaluate_offline

if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--config", default=None); ap.add_argument("--dataset", required=True); ap.add_argument("--method", default="oracle"); ap.add_argument("--output", default=None)
    args=ap.parse_args(); arrays,meta=read_dataset(args.dataset); res=evaluate_offline(arrays,args.method)
    if args.output:
        p=Path(args.output); p.mkdir(parents=True,exist_ok=True); (p/"metrics.json").write_text(json.dumps(res,indent=2))
    print(json.dumps(res,indent=2))
