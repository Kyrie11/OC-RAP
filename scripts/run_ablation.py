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

ABLATIONS = {
    "full": None,
    "no_observation_consistency": "no_observation_consistency",  # diagnostic: still uses stored labels unless pred mu is supplied
    "oracle_witness": "oracle_witness",
    "no_harm_constraint": "no_harm_constraint",
    "no_rule_constraint": "no_rule_constraint",
    "no_controlled_relaxation": "no_controlled_relaxation",
    "no_recovery_constraint": "no_recovery_constraint",
    "penalize_uncertainty": "penalize_uncertainty",
}

if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--base-config", default=None)
    ap.add_argument("--ablation", required=True, choices=sorted(ABLATIONS))
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch-size", type=int, default=4)
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
    # Oracle-witness ablation uses non-deployable option max as the selected-label diagnostic.
    ab=ABLATIONS[args.ablation]
    if args.ablation == "oracle_witness" and "Y_option" in arrays:
        # Keep CRISP selection but metrics will expose ORS and OLG; this flag records the ablation.
        ab = "oracle_witness"
    res=evaluate_offline(arrays,"ours",calibration=calib,ablation=ab)
    res["ablation_name"]=args.ablation
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    (out/"metrics.json").write_text(json.dumps(res,indent=2))
    (out/"ablation_flags.json").write_text(json.dumps({"name":args.ablation,"flag":ab,"uses_checkpoint":bool(args.checkpoint)},indent=2))
    print(json.dumps(res,indent=2))
