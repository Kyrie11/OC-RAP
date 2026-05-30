#!/usr/bin/env python
from __future__ import annotations

# Allow running scripts directly from a source checkout without pip install -e .
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import argparse
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
from ocrap.training.train_care import train_care

if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--config", default=None); ap.add_argument("--dataset", required=True); ap.add_argument("--proposal-checkpoint", default=None); ap.add_argument("--output", required=True); ap.add_argument("--epochs", type=int, default=None); ap.add_argument("--batch-size", type=int, default=None); ap.add_argument("--lr", type=float, default=None)
    args=ap.parse_args(); print(train_care(args.dataset, args.output, args.epochs, args.batch_size, args.lr, config=args.config, proposal_checkpoint=args.proposal_checkpoint), flush=True)
