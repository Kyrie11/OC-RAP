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

if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--base-config", required=True); ap.add_argument("--ablation", required=True); ap.add_argument("--dataset", required=True); ap.add_argument("--checkpoint", default=None); ap.add_argument("--output", required=True)
    args=ap.parse_args(); out=Path(args.output); out.mkdir(parents=True,exist_ok=True); (out/"ablation_flags.json").write_text(json.dumps({"enabled":True,"name":args.ablation,args.ablation:True,"combined_ablation":False},indent=2)); print(f"wrote {out/'ablation_flags.json'}")
