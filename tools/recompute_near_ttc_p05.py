#!/usr/bin/env python3
"""Recompute/audit Near-Contact scene-level TTC p05 from closed-loop journals.

This does *not* rerun planners. Publication TTC p05 is an aggregate of the
per-scene ``metric_summary.ttc_s_min`` values already persisted in the
scene-granular closed-loop journal. Re-aggregating those records is therefore
exact, faster, and avoids introducing a second simulation realization.

The tool also reports how many scene minima are exactly zero. With NumPy's
linear quantile definition and N=250, q=0.05 lies at zero-based rank 12.45:
13 zero-valued scenes can still yield a positive interpolated p05, whereas
14 zero-valued scenes force p05 to 0 when the next value is non-negative.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

BASELINE_METHODS = (
    "marc_lite",
    "racp_lite",
    "robust_scenario_mpc",
    "predictive_safety_filter",
    "dr_cvar_safety_filter",
    "conformal_predictive_safety_filter",
    "flow_planner",
    "plan_r1",
    "betopnet",
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - error reporting path
        raise SystemExit(f"failed to read JSON {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"expected object JSON in {path}")
    return data


def _journal_path(result_path: Path) -> Path:
    return result_path.with_suffix(result_path.suffix + ".scenes.jsonl")


def _load_journal_scenes(journal: Path) -> list[dict[str, Any]]:
    if not journal.is_file():
        raise SystemExit(f"missing scene journal: {journal}")
    # Last record wins for a duplicate resume key. Normal journals contain one
    # record per target, but this makes the audit robust to manually concatenated
    # or resumed journals without double-counting a scene.
    by_key: dict[str, dict[str, Any]] = {}
    anonymous: list[dict[str, Any]] = []
    with journal.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # Match the runner's resume semantics: tolerate a torn final
                # append record, but do not invent a scene from it.
                continue
            if not isinstance(rec, dict) or not isinstance(rec.get("scene"), dict):
                continue
            scene = rec["scene"]
            key = str(rec.get("resume_key") or scene.get("target_key") or "").strip()
            if key:
                by_key[key] = scene
            else:
                anonymous.append(scene)
    return list(by_key.values()) + anonymous


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _stored_ttc_p05(result: dict[str, Any]) -> float | None:
    wm = result.get("waymax_metrics")
    if isinstance(wm, dict):
        v = _finite_float(wm.get("scene_ttc_s_p05"))
        if v is not None:
            return v
    return _finite_float(result.get("scene_ttc_s_p05"))


def _audit_one(label: str, result_path: Path, expected_scenes: int, allow_partial: bool) -> dict[str, Any]:
    result = _read_json(result_path) if result_path.is_file() else {}
    journal = _journal_path(result_path)
    scenes = _load_journal_scenes(journal)
    minima: list[float] = []
    overlap_flags: list[float] = []
    missing_ttc = 0
    for scene in scenes:
        ms = scene.get("metric_summary") or {}
        if not isinstance(ms, dict):
            missing_ttc += 1
            continue
        ttc = _finite_float(ms.get("ttc_s_min"))
        if ttc is None:
            missing_ttc += 1
            continue
        minima.append(ttc)
        ov = _finite_float(ms.get("overlap_any"))
        if ov is not None:
            overlap_flags.append(ov)

    n = len(minima)
    if n == 0:
        raise SystemExit(f"{label}: no finite metric_summary.ttc_s_min values in {journal}")
    if expected_scenes > 0 and n != expected_scenes and not allow_partial:
        raise SystemExit(
            f"{label}: expected {expected_scenes} TTC scenes but found {n}; "
            "finish/resume the closed-loop run first or pass --allow-partial for diagnostics"
        )

    arr = np.asarray(minima, dtype=np.float64)
    zero_count = int(np.count_nonzero(arr <= 1.0e-12))
    # Explicitly pin the method used by the current publication implementation.
    p05 = float(np.quantile(arr, 0.05, method="linear"))
    positive = arr[arr > 1.0e-12]
    nonzero_p05 = float(np.quantile(positive, 0.05, method="linear")) if positive.size else None
    stored = _stored_ttc_p05(result)
    overlap_rate = (
        float(np.mean(np.asarray(overlap_flags, dtype=np.float64) > 0.5))
        if overlap_flags else None
    )
    q_index = 0.05 * (n - 1)
    zeros_needed_to_force_linear_p05_zero = int(math.floor(q_index) + 2)
    # If q lands exactly on an integer rank, that rank itself is sufficient.
    if math.isclose(q_index, round(q_index), rel_tol=0.0, abs_tol=1e-12):
        zeros_needed_to_force_linear_p05_zero = int(round(q_index)) + 1

    return {
        "label": label,
        "result": str(result_path),
        "journal": str(journal),
        "num_journal_scenes": len(scenes),
        "num_finite_ttc_scenes": n,
        "missing_ttc_scene_count": int(missing_ttc),
        "zero_ttc_scene_count": zero_count,
        "zero_ttc_scene_rate": float(zero_count / n),
        "overlap_scene_rate_from_journal": overlap_rate,
        "ttc_s_min_min": float(np.min(arr)),
        "ttc_s_min_median": float(np.median(arr)),
        "scene_ttc_s_p05_recomputed_linear": p05,
        "scene_ttc_s_p05_stored": stored,
        "stored_minus_recomputed": (float(stored - p05) if stored is not None else None),
        "positive_scene_ttc_s_p05": nonzero_p05,
        "linear_quantile_rank_0_based": q_index,
        "zero_scenes_needed_to_force_linear_p05_zero": zeros_needed_to_force_linear_p05_zero,
        "zero_rate_needed_to_force_linear_p05_zero": float(zeros_needed_to_force_linear_p05_zero / n),
        "metric_semantics_version": result.get("metric_semantics_version"),
        "publication_geometry_metric": result.get("publication_geometry_metric")
        or (result.get("runtime_contract") or {}).get("publication_geometry_metric"),
    }


def _iter_inputs(baseline_run: Path | None, ocrap_run: Path | None, methods: Iterable[str]) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    if baseline_run is not None:
        for method in methods:
            out.append((method, baseline_run / f"closed_loop_{method}.json"))
    if ocrap_run is not None:
        for variant in ("balanced", "precision"):
            out.append((f"ocrap_{variant}", ocrap_run / variant / "near" / "closed_loop_ocrap.json"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Recompute Near-Contact scene TTC p05 from persisted closed-loop scene journals.")
    ap.add_argument("--baseline-run", type=Path, default=None, help="Near baseline RUN directory, e.g. runs/.../near")
    ap.add_argument("--ocrap-run", type=Path, default=None, help="OC-RAP characterization root containing balanced/near and precision/near")
    ap.add_argument("--methods", default=",".join(BASELINE_METHODS), help="Comma-separated baseline method aliases")
    ap.add_argument("--expected-scenes", type=int, default=250)
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    if args.baseline_run is None and args.ocrap_run is None:
        raise SystemExit("provide --baseline-run and/or --ocrap-run")
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    for label, path in _iter_inputs(args.baseline_run, args.ocrap_run, methods):
        journal = _journal_path(path)
        if not journal.is_file():
            missing.append({"label": label, "result": str(path), "journal": str(journal)})
            continue
        rows.append(_audit_one(label, path, args.expected_scenes, args.allow_partial))

    doc = {
        "schema": "ocrap-near-ttc-p05-reaggregation-audit-v1",
        "quantile_definition": "numpy.quantile(q=0.05, method=linear) over scene-level metric_summary.ttc_s_min",
        "expected_scenes": int(args.expected_scenes),
        "rows": rows,
        "missing": missing,
    }
    text = json.dumps(doc, indent=2, ensure_ascii=False, sort_keys=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(json.dumps({"event": "near_ttc_p05_recomputed", "output": str(args.output), "num_rows": len(rows), "missing": len(missing)}))
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
