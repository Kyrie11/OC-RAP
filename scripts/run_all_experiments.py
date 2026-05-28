#!/usr/bin/env python
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import argparse, json
from pathlib import Path
from recap.teacher.dataset_writer import read_dataset
from recap.evaluation.offline_eval import evaluate_offline

DEFAULT_METHODS = ["ours", "nominal", "risk_aware", "backup_filter", "oracle"]
DEFAULT_ABLATIONS = ["no_harm_constraint", "no_rule_constraint", "no_controlled_relaxation", "no_recovery_constraint", "penalize_uncertainty"]

if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--experiment-list", default=None, help="Optional JSON list or file containing methods/ablations. If omitted, runs the paper OC-RAP non-baseline suite.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--skip-baselines", action="store_true", help="Run only OC-RAP and ablations; external baselines are out of scope.")
    args=ap.parse_args()
    arrays,meta=read_dataset(args.dataset); arrays=dict(arrays)
    if args.checkpoint:
        from recap.evaluation.inference import predict_profiles
        arrays.update(predict_profiles(args.dataset, args.checkpoint, batch_size=args.batch_size))
    calib=None
    if args.calibration:
        cp=Path(args.calibration)
        if cp.is_dir(): cp=cp/"q_values.json"
        calib=json.loads(cp.read_text())
    methods = ["ours"] if args.skip_baselines else DEFAULT_METHODS
    ablations = DEFAULT_ABLATIONS
    if args.experiment_list:
        ep = Path(args.experiment_list)
        obj = json.loads(ep.read_text()) if ep.exists() else json.loads(args.experiment_list)
        methods = obj.get("methods", methods)
        ablations = obj.get("ablations", ablations)
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    results={"dataset_version":meta.get("dataset_version",""),"methods":{},"ablations":{}}
    for m in methods:
        results["methods"][m]=evaluate_offline(arrays,m,calibration=calib)
    for ab in ablations:
        results["ablations"][ab]=evaluate_offline(arrays,"ours",calibration=calib,ablation=ab)
    (out/"all_metrics.json").write_text(json.dumps(results,indent=2))
    # Flat CSV for paper table export without requiring pandas.
    rows=["group,name,OCS,FAR,SLR,OLG,SRR,HNIV,MIR,utility_mean,uses_learned_profiles"]
    for group in ["methods","ablations"]:
        for name,r in results[group].items():
            rows.append(",".join(str(x) for x in [group,name,r.get("OCS"),r.get("FAR"),r.get("SLR"),r.get("OLG"),r.get("SRR"),r.get("HNIV"),r.get("MIR"),r.get("utility_mean"),r.get("uses_learned_profiles")]))
    (out/"summary.csv").write_text("\n".join(rows)+"\n")
    print(json.dumps(results,indent=2))
