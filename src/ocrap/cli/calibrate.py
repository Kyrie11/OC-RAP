from __future__ import annotations

from pathlib import Path

import numpy as np

from ocrap.algorithms.lcv import finite_sample_upper_quantile
from ocrap.data.build.diagnose import iter_sample_paths
from ocrap.data.serialization import load_npz, write_json


def calibrate(dataset: str, checkpoint: str | None = None, output: str | None = None, cfg: dict | None = None) -> dict:
    cfg = cfg or {}
    deltas = cfg.get("calibration", {}).get("deltas", [0.01, 0.05, 0.10])
    strict = bool(cfg.get("calibration", {}).get("strict", True))
    numerical_margin = float(cfg.get("calibration", {}).get("numerical_margin", 0.0))
    required = int(cfg.get("calibration", {}).get("required_min_for_delta", 100))
    scores = []
    teacher = []
    for p in iter_sample_paths(dataset):
        d = load_npz(p)
        split = str(np.asarray(d.get("split_id", "")).item())
        if split not in {"calibration", "val", "train"}:  # fallback keeps small fixtures usable
            continue
        r = float(np.asarray(d["r_dep_star"]).item())
        scores.append(r)
        teacher.append(r)
    scores = np.asarray(scores, dtype=float)
    teacher = np.asarray(teacher, dtype=float)
    neg = scores[teacher < 0]
    warnings = []
    if len(neg) < required:
        warnings.append(f"num_negative < required_min_for_delta ({len(neg)} < {required})")
    thresholds = {}
    for delta in deltas:
        try:
            thresholds[str(delta)] = finite_sample_upper_quantile(neg, float(delta), numerical_margin, strict)
        except ValueError:
            thresholds[str(delta)] = float("inf")
    result = {"num_samples": int(len(scores)), "num_negative": int(len(neg)), "thresholds": thresholds, "gamma_rec": thresholds.get(str(deltas[0]), 0.0), "warnings": warnings}
    if output:
        write_json(result, output)
    return result
