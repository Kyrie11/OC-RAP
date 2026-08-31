#!/usr/bin/env python3
from __future__ import annotations

"""Calibrate external near-contact safety-filter admission on a held-out split.

The deployable policy never sees teacher labels.  This script is the only place
where the calibration split may compare the observation-only risk score against
OC-RAP's hard-violation label, then export a scalar admission threshold used by
closed-loop evaluation.  Test labels are never used for threshold selection.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from ocrap.config import load_config
from ocrap.data.serialization import write_json
from ocrap.external_baselines.data import group_sample_paths, load_external_sample
from ocrap.external_baselines.observed_risk import observed_risk_profiles


def split_conformal_qhat(scores: np.ndarray, alpha: float) -> float:
    scores = np.sort(np.asarray(scores, dtype=float).reshape(-1))
    if scores.size == 0:
        raise ValueError("No calibration scores")
    alpha = float(np.clip(alpha, 1e-4, 0.9999))
    k = int(math.ceil((scores.size + 1) * (1.0 - alpha)))
    k = min(max(k, 1), scores.size)
    return float(scores[k - 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="calibration")
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    groups = group_sample_paths(args.dataset, split=args.split)
    if not groups:
        raise SystemExit(f"No grouped samples in {args.dataset!r} for split={args.split!r}")

    probabilities: list[float] = []
    labels: list[int] = []
    for paths in groups:
        samples = [load_external_sample(p) for p in paths]
        profiles = observed_risk_profiles(samples, cfg)
        for sample, profile in zip(samples, profiles):
            probabilities.append(float(np.clip(profile.collision_probability, 0.0, 1.0)))
            labels.append(int(float(np.asarray(sample.get("hard_violation", 0.0)).item()) > 0.0))

    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels, dtype=int)
    # Binary label nonconformity: 1 - model probability assigned to the true label.
    scores = np.where(y > 0, 1.0 - p, p)
    qhat = split_conformal_qhat(scores, args.alpha)
    # A candidate is admitted only if the unsafe label is excluded from the
    # conformal prediction set: 1-p > qhat => p < 1-qhat.
    threshold = float(np.clip(1.0 - qhat, 0.0, 1.0))

    config_fingerprint = hashlib.sha256(
        json.dumps(cfg, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()
    result = {
        "protocol": "split_conformal_binary_admission_on_observation_only_collision_probability",
        "requested_config_fingerprint": config_fingerprint,
        "dataset": str(Path(args.dataset)),
        "split": args.split,
        "alpha": float(args.alpha),
        "num_groups": len(groups),
        "num_candidates": int(p.size),
        "positive_hard_violation_rate": float(y.mean()) if y.size else float("nan"),
        "conformal_qhat": qhat,
        "conformal_collision_probability_threshold": threshold,
        "test_labels_used": False,
        "note": "Teacher hard_violation is used only on the calibration split. Runtime selection consumes observation-only risk probabilities and this frozen scalar threshold.",
    }
    write_json(result, args.output)
    print(result)


if __name__ == "__main__":
    main()
