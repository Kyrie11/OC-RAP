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
import numpy as np
from recap.teacher.dataset_writer import read_dataset
from recap.training.calibrate_selector import calibrate_q

if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--config", default=None); ap.add_argument("--dataset", required=True); ap.add_argument("--checkpoint", default=None); ap.add_argument("--split", default="calib"); ap.add_argument("--output", required=True)
    args=ap.parse_args(); arrays,meta=read_dataset(args.dataset)
    R=arrays["R_star"] # For MVP oracle calibration; learned CARE path should write R_pred.
    H=arrays.get("H_action_star", arrays["H_star"].max(axis=-1)); dH=H-np.min(np.where(arrays["action_mask"],H,np.inf),axis=1,keepdims=True)
    q=calibrate_q(R,arrays["R_star"],dH,dH,arrays["action_mask"]); q["dataset_version"]=meta.get("dataset_version",""); q["split"]=args.split
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True); (out/"q_values.json").write_text(json.dumps(q,indent=2)); print(json.dumps(q,indent=2))
