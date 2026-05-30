#!/usr/bin/env python
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import argparse, json
from pathlib import Path
from ocrap.teacher.dataset_writer import read_dataset
from ocrap.evaluation.closed_loop_eval import evaluate_closed_loop_or_offline

if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset", default="data/ocrap/test.zarr")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--method", default="ours")
    ap.add_argument("--ablation", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--output", required=True)
    ap.add_argument("--require-simulator", action="store_true", help="Fail instead of using the diagnostic offline same-candidate fallback.")
    args=ap.parse_args()
    arrays,meta=read_dataset(args.dataset); arrays=dict(arrays)
    if args.checkpoint and args.method.lower() in ("ours","ocrap","crisp"):
        from ocrap.evaluation.inference import predict_profiles
        ocmero_params = {"use_observation_consistency": False} if args.ablation == "no_observation_consistency" else None
        arrays.update(predict_profiles(args.dataset, args.checkpoint, batch_size=args.batch_size, ocmero_params=ocmero_params))
    calib=None
    if args.calibration:
        cp=Path(args.calibration)
        if cp.is_dir(): cp=cp/"q_values.json"
        calib=json.loads(cp.read_text())
    res=evaluate_closed_loop_or_offline(arrays,args.method,allow_offline_fallback=not args.require_simulator,calibration=calib,ablation=args.ablation)
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    (out/"metrics.json").write_text(json.dumps(res,indent=2))
    alignment={"uses_oracle_selector_for_ours": False,"uses_calibrated_crisp_for_ours": args.method.lower() in ("ours","ocrap","crisp"),"closed_loop_backend": res.get("closed_loop_backend"),"notes":"If MetaDrive/CARLA backend is not connected, this is an offline same-candidate fallback, not a paper-final closed-loop run."}
    (out/"alignment_report.json").write_text(json.dumps(alignment,indent=2))
    print(json.dumps(res,indent=2))
