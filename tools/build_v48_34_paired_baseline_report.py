#!/usr/bin/env python3
"""Build fail-closed paired closed-loop comparisons on an identical scene set.

v48.34.1 reports both absolute method/reference values and paired deltas.  This
prevents a positive oriented delta from being shown without the underlying
physical scale, and emits regime-specific metric completeness metadata for
progress presentations.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _rows(path: Path) -> dict[str, dict[str, Any]]:
    if path.suffix == ".json" and not path.name.endswith(".scenes.jsonl"):
        alt = Path(str(path) + ".scenes.jsonl")
        if alt.is_file():
            path = alt
    if not path.is_file():
        raise FileNotFoundError(path)
    out: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            e = json.loads(line)
            s = e.get("scene", e)
            key = str(s.get("target_key") or e.get("resume_key") or "")
            if not key:
                scene_id = str(s.get("scene_id") or "")
                time_index = s.get("target_time_index")
                if scene_id and time_index is not None:
                    key = f"{scene_id}:t{int(time_index)}"
                elif scene_id:
                    key = scene_id
            if not key:
                raise ValueError(f"scene row without a pairing key in {path}")
            if key in out:
                raise ValueError(f"duplicate key {key} in {path}")
            out[key] = s
    if not out:
        raise ValueError(f"empty scene journal: {path}")
    return out


def _f(x: Any) -> float | None:
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


# location, orientation (+1 means larger is better), display unit
COMMON = {
    "closed_loop_bounded_NUP": ("top", 1, "score"),
    "intervention_rate": ("top", -1, "rate"),
    "overlap_any": ("metric_summary", -1, "rate"),
    "offroad_any": ("metric_summary", -1, "rate"),
}
REGIME_METRICS = {
    "safe": COMMON | {
        "route_progression_m": ("metric_summary", 1, "m"),
        "acceleration_abs_p95_mps2": ("metric_summary", -1, "m/s^2"),
        "jerk_p95": ("metric_summary", -1, "m/s^3"),
        "yaw_rate_p95": ("metric_summary", -1, "rad/s"),
        "min_clearance_m_p05": ("metric_summary", 1, "m"),
        "ttc_s_p05": ("metric_summary", 1, "s"),
    },
    "near": COMMON | {
        "ttc_s_p05": ("metric_summary", 1, "s"),
        "terminal_ttc_s": ("metric_summary", 1, "s"),
        "min_clearance_m_p05": ("metric_summary", 1, "m"),
        "terminal_clearance_m": ("metric_summary", 1, "m"),
        "critical_ttc_exposure_duration_s": ("metric_summary", -1, "s"),
        "near_zero_clearance_exposure_rate": ("metric_summary", -1, "rate"),
        "clearance_deficit_auc_m_s": ("metric_summary", -1, "m*s"),
        "ttc_deficit_auc_s2": ("metric_summary", -1, "s^2"),
    },
    "contact": COMMON | {
        "post_contact_terminal_clearance_m": ("metric_summary", 1, "m"),
        "post_contact_free_space_auc_normalized_m": ("metric_summary", 1, "m"),
        "post_contact_clearance_gain_m": ("metric_summary", 1, "m"),
        "post_contact_clearance_deficit_auc_m_s": ("metric_summary", -1, "m*s"),
        "post_contact_escape_event": ("metric_summary", 1, "rate"),
        "recontact_event": ("metric_summary", -1, "rate"),
        "stable_stop_quality_event": ("metric_summary", 1, "rate"),
        "overlap_duration_s": ("metric_summary", -1, "s"),
    },
}
REQUIRED_CORE = {
    "safe": {"closed_loop_bounded_NUP", "intervention_rate", "overlap_any", "offroad_any"},
    "near": {"closed_loop_bounded_NUP", "intervention_rate", "ttc_s_p05", "min_clearance_m_p05", "critical_ttc_exposure_duration_s", "overlap_any", "offroad_any"},
    "contact": {"closed_loop_bounded_NUP", "intervention_rate", "post_contact_terminal_clearance_m", "post_contact_free_space_auc_normalized_m", "recontact_event", "stable_stop_quality_event", "offroad_any"},
}


def _value(scene: dict[str, Any], metric: str, loc: str) -> float | None:
    return _f(scene.get(metric) if loc == "top" else (scene.get("metric_summary", {}) or {}).get(metric))


def _bootstrap_ci(arr: np.ndarray, *, rng: np.random.Generator, n_bootstrap: int) -> list[float | None]:
    if arr.size == 0 or n_bootstrap <= 0:
        return [None, None]
    # Vectorized paired bootstrap.  Chunking bounds memory for larger exploratory runs.
    chunk = 1024
    means: list[np.ndarray] = []
    remaining = int(n_bootstrap)
    while remaining > 0:
        n = min(chunk, remaining)
        idx = rng.integers(0, arr.size, size=(n, arr.size), endpoint=False)
        means.append(arr[idx].mean(axis=1))
        remaining -= n
    boot = np.concatenate(means)
    return [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True, help="NAME=SCENES_JSONL")
    ap.add_argument("--method", action="append", default=[], help="NAME=SCENES_JSONL")
    ap.add_argument("--regime", choices=("safe", "near", "contact"), required=True)
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--output-csv", type=Path, required=True)
    ap.add_argument("--output-wide-csv", type=Path)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=48341)
    ap.add_argument("--require-core-metrics", action="store_true")
    args = ap.parse_args()

    def spec(text: str) -> tuple[str, Path]:
        if "=" not in text:
            raise ValueError(f"invalid method spec {text}")
        n, p = text.split("=", 1)
        return n, Path(p)

    ref_name, ref_path = spec(args.reference)
    ref = _rows(ref_path)
    methods: list[tuple[str, dict[str, dict[str, Any]], str]] = []
    for text in args.method:
        name, path = spec(text)
        methods.append((name, _rows(path), str(path)))
    if not methods:
        raise SystemExit("at least one --method is required")

    metrics = REGIME_METRICS[args.regime]
    rng = np.random.default_rng(args.seed)
    reports: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    missing_core_by_method: dict[str, list[str]] = {}

    for name, rows, path in methods:
        if set(rows) != set(ref):
            raise SystemExit(
                f"scene set mismatch for {name}: method_only={len(set(rows)-set(ref))}, "
                f"reference_only={len(set(ref)-set(rows))}"
            )
        metric_reports: dict[str, Any] = {}
        for metric, (loc, direction, unit) in metrics.items():
            method_vals: list[float] = []
            reference_vals: list[float] = []
            for key in sorted(ref):
                a = _value(rows[key], metric, loc)
                b = _value(ref[key], metric, loc)
                if a is not None and b is not None:
                    method_vals.append(a)
                    reference_vals.append(b)
            if not method_vals:
                continue
            method_arr = np.asarray(method_vals, dtype=float)
            reference_arr = np.asarray(reference_vals, dtype=float)
            raw = method_arr - reference_arr
            oriented = float(direction) * raw
            rep = {
                "n": int(oriented.size),
                "missing_pair_count": int(len(ref) - oriented.size),
                "unit": unit,
                "raw_direction": int(direction),
                "reference_mean": float(reference_arr.mean()),
                "reference_median": float(np.median(reference_arr)),
                "method_mean": float(method_arr.mean()),
                "method_median": float(np.median(method_arr)),
                "raw_delta_mean": float(raw.mean()),
                "raw_delta_median": float(np.median(raw)),
                "oriented_delta_mean": float(oriented.mean()),
                "oriented_delta_median": float(np.median(oriented)),
                "oriented_delta_ci95": _bootstrap_ci(oriented, rng=rng, n_bootstrap=args.bootstrap),
                "higher_is_better_after_orientation": True,
            }
            metric_reports[metric] = rep
            csv_rows.append({"regime": args.regime, "reference": ref_name, "method": name, "metric": metric, **rep})
        missing_core = sorted(REQUIRED_CORE[args.regime] - set(metric_reports))
        missing_core_by_method[name] = missing_core
        reports.append({
            "method": name,
            "path": path,
            "num_scenes": len(rows),
            "core_metrics_complete": not missing_core,
            "missing_core_metrics": missing_core,
            "metrics": metric_reports,
        })

    valid = not any(missing_core_by_method.values())
    doc = {
        "event": "v48_34_1_paired_baseline_report",
        "version": "v48.34.1-RC30-MODEL-CONTRACT-HOTFIX",
        "regime": args.regime,
        "reference": ref_name,
        "reference_path": str(ref_path),
        "num_scenes": len(ref),
        "scene_set_exact_match": True,
        "core_metrics_required": sorted(REQUIRED_CORE[args.regime]),
        "core_metrics_complete": valid,
        "missing_core_metrics_by_method": missing_core_by_method,
        "exploratory_only": True,
        "paper_claim_allowed": False,
        "methods": reports,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fields = [
        "regime", "reference", "method", "metric", "n", "missing_pair_count", "unit",
        "reference_mean", "reference_median", "method_mean", "method_median",
        "raw_delta_mean", "raw_delta_median", "oriented_delta_mean", "oriented_delta_median",
        "oriented_delta_ci95", "higher_is_better_after_orientation", "raw_direction",
    ]
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in csv_rows:
            row = dict(row)
            row["oriented_delta_ci95"] = json.dumps(row["oriented_delta_ci95"])
            w.writerow(row)

    wide_path = args.output_wide_csv or args.output_csv.with_name(args.output_csv.stem + ".wide.csv")
    wide_fields = ["regime", "method", "num_scenes", "core_metrics_complete", "missing_core_metrics"]
    for metric in metrics:
        wide_fields.extend([f"{metric}__method_mean", f"{metric}__raw_delta_mean", f"{metric}__oriented_delta_mean"])
    with wide_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=wide_fields)
        w.writeheader()
        for report in reports:
            row: dict[str, Any] = {
                "regime": args.regime,
                "method": report["method"],
                "num_scenes": report["num_scenes"],
                "core_metrics_complete": report["core_metrics_complete"],
                "missing_core_metrics": ",".join(report["missing_core_metrics"]),
            }
            for metric, rep in report["metrics"].items():
                row[f"{metric}__method_mean"] = rep["method_mean"]
                row[f"{metric}__raw_delta_mean"] = rep["raw_delta_mean"]
                row[f"{metric}__oriented_delta_mean"] = rep["oriented_delta_mean"]
            w.writerow(row)

    print(json.dumps({
        "event": doc["event"], "regime": args.regime, "num_methods": len(methods),
        "num_scenes": len(ref), "core_metrics_complete": valid,
        "output_wide_csv": str(wide_path),
    }))
    return 0 if valid or not args.require_core_metrics else 4


if __name__ == "__main__":
    raise SystemExit(main())
