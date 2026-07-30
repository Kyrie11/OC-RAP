from __future__ import annotations

from pathlib import Path

import numpy as np

from ocrap.algorithms.lcv import finite_sample_upper_quantile
from ocrap.data.serialization import load_npz, write_json
from ocrap.models.data import expand_split_roles, iter_sample_paths_many, scalar_metadata_for_path
from ocrap.models.inference import load_model_bundle, predict_sample
from ocrap.evaluation.metrics import predicted_shared_option_success


def _calibration_score(d: dict, pred, cfg: dict) -> float:
    mode = str((cfg.get("calibration", {}) or {}).get("score", "r_dep")).strip().lower()
    if mode in {"r_dep", "rec", "recoverability"}:
        return float(pred.r_dep)
    if mode in {"rec_lcb", "lcb", "r_dep_lcb"}:
        beta = float((cfg.get("selection", {}) or {}).get("lcb_beta", 0.10))
        return float(pred.r_dep - beta * max(0.0, pred.gap))
    if mode in {"pred_drs", "drs", "shared_option_success", "option_drs"}:
        gamma = float((cfg.get("selection", {}) or {}).get("drs_success_gamma", 0.0))
        return float(predicted_shared_option_success(
            pred.q, pred.root_probs, gamma=gamma,
            root_valid=d.get("root_valid", None), option_valid=d.get("option_valid", None)
        ))
    raise ValueError(f"Unknown calibration.score={mode!r}; expected r_dep, rec_lcb, or pred_drs")


def calibrate(dataset: str, checkpoint: str | None = None, output: str | None = None, cfg: dict | None = None) -> dict:
    cfg = cfg or {}
    deltas = cfg.get("calibration", {}).get("deltas", [0.01, 0.05, 0.10])
    strict = bool(cfg.get("calibration", {}).get("strict", True))
    numerical_margin = float(cfg.get("calibration", {}).get("numerical_margin", 0.0))
    cal_cfg = cfg.get("calibration", {}) or {}
    required = int(cal_cfg.get("required_min_for_delta", 100))
    allowed_raw = cal_cfg.get("allowed_split_ids", "calibration")
    if isinstance(allowed_raw, str):
        requested_splits = {x.strip() for x in allowed_raw.split(",") if x.strip()}
    else:
        requested_splits = {str(x).strip() for x in allowed_raw if str(x).strip()}
    allowed_splits = expand_split_roles(requested_splits or {"calibration"})
    allow_val_fallback = bool(cal_cfg.get("allow_validation_fallback", True))
    bundle = load_model_bundle(checkpoint, cfg)
    paths = iter_sample_paths_many(dataset)
    print({"event": "calibrate_start", "num_npz_paths": len(paths), "dataset": str(dataset)}, flush=True)
    scores = []
    teacher = []
    used_splits = []
    for idx, p in enumerate(paths, 1):
        if idx == 1 or idx % 1000 == 0:
            print({"event": "calibrate_progress", "seen": idx, "kept": len(scores)}, flush=True)
        split = str(scalar_metadata_for_path(p, "split_id", ""))
        if split not in allowed_splits:
            continue
        d = load_npz(p)
        pred = predict_sample(d, bundle, cfg)
        scores.append(_calibration_score(d, pred, cfg))
        teacher.append(float(np.asarray(d["r_dep_star"]).item()))
        used_splits.append(split)
    warnings = []
    if not scores and allow_val_fallback:
        warnings.append("no requested calibration split found; falling back to validation role")
        val_splits = expand_split_roles({"val"})
        for idx, p in enumerate(paths, 1):
            if idx == 1 or idx % 1000 == 0:
                print({"event": "calibrate_val_fallback_progress", "seen": idx, "kept": len(scores)}, flush=True)
            split = str(scalar_metadata_for_path(p, "split_id", ""))
            if split not in val_splits:
                continue
            d = load_npz(p)
            pred = predict_sample(d, bundle, cfg)
            scores.append(_calibration_score(d, pred, cfg))
            teacher.append(float(np.asarray(d["r_dep_star"]).item()))
            used_splits.append(split)
    scores = np.asarray(scores, dtype=float)
    teacher = np.asarray(teacher, dtype=float)
    neg = scores[teacher < 0]
    if len(scores) == 0:
        warnings.append("no samples matched the requested calibration split role(s)")
    if len(neg) < required:
        warnings.append(f"num_negative < required_min_for_delta ({len(neg)} < {required})")
    def _delta_key(x) -> str:
        try:
            return f"{float(x):g}"
        except Exception:
            return str(x)

    thresholds = {}
    for delta in deltas:
        try:
            thresholds[_delta_key(delta)] = finite_sample_upper_quantile(neg, float(delta), numerical_margin, strict)
        except ValueError:
            thresholds[_delta_key(delta)] = float("inf")
    eval_delta = (cfg.get("evaluation", {}) or {}).get("delta", None) if isinstance(cfg.get("evaluation", {}), dict) else None
    default_delta = _delta_key(eval_delta if eval_delta is not None else (deltas[0] if deltas else 0.05))
    if default_delta not in thresholds and thresholds:
        default_delta = next(iter(thresholds.keys()))
    result = {
        "num_samples": int(len(scores)),
        "num_negative": int(len(neg)),
        "source": "model" if bundle is not None else "teacher_fallback",
        "score_mode": str((cfg.get("calibration", {}) or {}).get("score", "r_dep")),
        "splits": sorted(set(used_splits)),
        "requested_split_roles": sorted(requested_splits),
        "allowed_split_ids": sorted(allowed_splits),
        "allow_validation_fallback": allow_val_fallback,
        "thresholds": thresholds,
        "default_delta": default_delta,
        "gamma_rec": thresholds.get(default_delta, 0.0),
        "warnings": warnings,
    }
    if output:
        write_json(result, output)
    return result
