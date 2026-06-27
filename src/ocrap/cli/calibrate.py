from __future__ import annotations

from pathlib import Path

import numpy as np

from ocrap.algorithms.lcv import finite_sample_upper_quantile
from ocrap.data.serialization import load_npz, write_json
from ocrap.models.data import iter_sample_paths_many
from ocrap.models.inference import load_model_bundle, predict_sample


def calibrate(dataset: str, checkpoint: str | None = None, output: str | None = None, cfg: dict | None = None) -> dict:
    cfg = cfg or {}
    deltas = cfg.get("calibration", {}).get("deltas", [0.01, 0.05, 0.10])
    strict = bool(cfg.get("calibration", {}).get("strict", True))
    numerical_margin = float(cfg.get("calibration", {}).get("numerical_margin", 0.0))
    required = int(cfg.get("calibration", {}).get("required_min_for_delta", 100))
    bundle = load_model_bundle(checkpoint, cfg)
    scores = []
    teacher = []
    used_splits = []
    for p in iter_sample_paths_many(dataset):
        d = load_npz(p)
        split = str(np.asarray(d.get("split_id", "")).item())
        if split != "calibration":
            continue
        pred = predict_sample(d, bundle, cfg)
        scores.append(float(pred.r_dep))
        teacher.append(float(np.asarray(d["r_dep_star"]).item()))
        used_splits.append(split)
    warnings = []
    if not scores:
        warnings.append("no calibration split found; falling back to validation split")
        for p in iter_sample_paths_many(dataset):
            d = load_npz(p)
            split = str(np.asarray(d.get("split_id", "")).item())
            if split != "val":
                continue
            pred = predict_sample(d, bundle, cfg)
            scores.append(float(pred.r_dep))
            teacher.append(float(np.asarray(d["r_dep_star"]).item()))
            used_splits.append(split)
    scores = np.asarray(scores, dtype=float)
    teacher = np.asarray(teacher, dtype=float)
    neg = scores[teacher < 0]
    if len(neg) < required:
        warnings.append(f"num_negative < required_min_for_delta ({len(neg)} < {required})")
    thresholds = {}
    for delta in deltas:
        try:
            thresholds[str(delta)] = finite_sample_upper_quantile(neg, float(delta), numerical_margin, strict)
        except ValueError:
            thresholds[str(delta)] = float("inf")
    default_delta = str(deltas[0]) if deltas else "0.05"
    result = {
        "num_samples": int(len(scores)),
        "num_negative": int(len(neg)),
        "source": "model" if bundle is not None else "teacher_fallback",
        "splits": sorted(set(used_splits)),
        "thresholds": thresholds,
        "gamma_rec": thresholds.get(default_delta, 0.0),
        "warnings": warnings,
    }
    if output:
        write_json(result, output)
    return result
