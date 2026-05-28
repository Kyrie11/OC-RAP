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
from ocrap.evaluation.offline_eval import evaluate_offline

if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--method", default="ours")
    ap.add_argument("--ablation", default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--eta-R", type=float, default=0.70)
    ap.add_argument("--eta-H", type=float, default=0.50)
    ap.add_argument("--epsilon-H", type=float, default=0.05)
    ap.add_argument("--output", default=None)
    args=ap.parse_args()
    arrays,meta=read_dataset(args.dataset)
    arrays=dict(arrays)
    pred_meta={}
    if args.checkpoint and args.method.lower() in ("ours","ocrap","crisp"):
        from ocrap.evaluation.inference import predict_profiles
        pred = predict_profiles(args.dataset, args.checkpoint, batch_size=args.batch_size)
        arrays.update(pred)
        pred_meta={"checkpoint": args.checkpoint, "prediction_arrays": sorted(pred.keys())}
    calib = None
    if args.calibration:
        calib_path = Path(args.calibration)
        if calib_path.is_dir():
            calib_path = calib_path / "q_values.json"
        calib = json.loads(calib_path.read_text())
    res=evaluate_offline(arrays,args.method,eta_R=args.eta_R,eta_H=args.eta_H,epsilon_H=args.epsilon_H,calibration=calib,ablation=args.ablation)
    res["dataset_version"]=meta.get("dataset_version","")
    res.update(pred_meta)
    if args.output:
        p=Path(args.output); p.mkdir(parents=True,exist_ok=True); (p/"metrics.json").write_text(json.dumps(res,indent=2))
    print(json.dumps(res,indent=2))
