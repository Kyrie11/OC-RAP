#!/usr/bin/env python
from __future__ import annotations

# Allow running scripts directly from a source checkout without pip install -e .
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import argparse
from ocrap.training.train_action_proposal import train_action_proposal

if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--config", default=None); ap.add_argument("--dataset", required=True); ap.add_argument("--output", required=True)
    args=ap.parse_args(); print(train_action_proposal(args.dataset,args.output))
