#!/usr/bin/env python3
"""Precompute exact teacher PCD and factorized evidence metadata.

Engineering fast paths added in v48.56:
- read only the NPZ members actually consumed by teacher construction;
- optionally parallelize independent per-sample decoding/OC-MERO work;
- optionally reuse raw teacher coordinates from a contract-matched prior index
  and recompute only arm-specific component labels.

All fast paths preserve row order and the numerical teacher/component semantics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ocrap.algorithms.evidence_targets import ComponentVetoTolerances, component_veto_margin_numpy
from ocrap.algorithms.ocmero import oc_mero
from ocrap.data.serialization import load_npz_selected
from ocrap.evaluation.metrics import (
    best_option_indices,
    deployable_recovery_success,
    post_contact_deployability_score,
)
from ocrap.models.data import TEACHER_PCD_NPZ_KEYS, bucket_id_for_path, iter_sample_paths_many


RAW_ROW_REQUIRED_KEYS = frozenset({
    "path", "bucket", "scene", "time", "candidate", "macro", "nominal",
    "teacher_pcd", "teacher_drs", "teacher_r_dep", "teacher_gap",
    "teacher_hard_violation", "teacher_harm_proxy",
})


def _scalar(d: dict[str, Any], key: str, default: Any) -> Any:
    a = np.asarray(d.get(key, default))
    return a.item() if a.shape == () else a


def teacher_components(
    d: dict[str, Any], *, alpha: float, beta: float, top_m: int,
    option_execution_semantics: str = "global",
) -> dict[str, float]:
    m = np.asarray(d["m_star"], dtype=np.float64)
    p = np.asarray(d["root_probs"], dtype=np.float64)
    c = np.asarray(d.get("c_star", np.eye(m.shape[0])), dtype=np.float64)
    rv = np.asarray(d.get("root_valid", np.ones(m.shape[0])), dtype=bool)
    ov = np.asarray(d.get("option_valid", np.ones(m.shape[1])), dtype=bool)
    result = oc_mero(
        m, p, c, alpha=alpha, beta=beta, option_valid=ov, root_valid=rv,
        use_lcvar=True, use_obs_kernel=True, top_m=top_m,
    )
    option = best_option_indices(
        result.q, p, gamma=0.0, root_valid=rv, option_valid=ov,
        semantics=option_execution_semantics,
    )
    drs = float(deployable_recovery_success(m, p, option, root_valid=rv))
    r_dep = float(_scalar(d, "r_dep_star", result.r_dep))
    r_orc = float(_scalar(d, "r_orc_star", result.r_orc))
    gap = max(0.0, r_orc - r_dep)
    return {
        "teacher_pcd": float(post_contact_deployability_score(drs, r_dep, gap)),
        "teacher_drs": drs,
        "teacher_r_dep": r_dep,
        "teacher_gap": gap,
        "teacher_hard_violation": float(_scalar(d, "hard_violation", 0.0)),
        "teacher_harm_proxy": float(_scalar(d, "harm_proxy", 0.0)),
    }


def _build_raw_row(task: tuple[str, float, float, int, str]) -> dict[str, Any]:
    path_raw, alpha, beta, top_m, option_execution_semantics = task
    path = Path(path_raw)
    d = load_npz_selected(path, TEACHER_PCD_NPZ_KEYS)
    scene = str(_scalar(d, "scene_id", path.stem))
    time_index = int(_scalar(d, "time_index", 0))
    bucket = int(bucket_id_for_path(path))
    return {
        "path": str(path.resolve()),
        "bucket": bucket,
        "scene": scene,
        "time": time_index,
        "candidate": int(_scalar(d, "candidate_index", 0)),
        "macro": int(_scalar(d, "prefix_macro_type_id", _scalar(d, "prefix_macro_id", -1))),
        "nominal": bool(float(_scalar(d, "is_nominal", 0.0)) > 0.5),
        **teacher_components(
            d, alpha=alpha, beta=beta, top_m=top_m,
            option_execution_semantics=option_execution_semantics,
        ),
    }


def _dataset_roots(dataset: str) -> list[str]:
    return [str(Path(x.strip()).resolve()) for x in str(dataset).split(",") if x.strip()]


def _dataset_manifest_records(dataset: str) -> list[dict[str, Any]]:
    out = []
    for dataset_root in _dataset_roots(dataset):
        manifest = Path(dataset_root) / "manifest.csv"
        out.append({
            "root": dataset_root,
            "manifest": str(manifest),
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.is_file() else None,
        })
    return out


def _float_same(a: Any, b: Any) -> bool:
    try:
        return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1.0e-12)
    except (TypeError, ValueError):
        return False


def _raw_reuse_validation(
    *, source_summary: dict[str, Any], dataset: str, alpha: float, beta: float,
    top_m: int, option_execution_semantics: str,
) -> list[str]:
    failures: list[str] = []
    contract = source_summary.get("index_contract") or {}
    if contract.get("dataset_roots") != _dataset_roots(dataset):
        failures.append("dataset_roots")
    if contract.get("dataset_manifests") != _dataset_manifest_records(dataset):
        failures.append("dataset_manifests")
    for key, expected in (("alpha", alpha), ("beta", beta)):
        if not _float_same(source_summary.get(key, contract.get(key)), expected):
            failures.append(key)
    try:
        if int(source_summary.get("top_m", contract.get("top_m", -1))) != int(top_m):
            failures.append("top_m")
    except Exception:
        failures.append("top_m")
    if str(source_summary.get("option_execution_semantics", "")) != str(option_execution_semantics):
        failures.append("option_execution_semantics")
    return failures


def _read_source_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = RAW_ROW_REQUIRED_KEYS.difference(row)
            if missing:
                raise ValueError(f"source raw teacher index line {lineno} missing keys: {sorted(missing)}")
            rows.append(row)
    return rows


def _progress(*, seen: int, total: int, groups: int, t0: float, workers: int, mode: str) -> None:
    elapsed = max(1.0e-9, time.perf_counter() - t0)
    rate = float(seen) / elapsed
    eta = max(0.0, float(total - seen) / rate) if rate > 0 else None
    print({
        "event": "teacher_pcd_index_progress", "seen": seen, "total": total, "groups": groups,
        "elapsed_seconds": elapsed, "samples_per_second": rate, "eta_seconds": eta,
        "workers": int(workers), "source_mode": mode, "npz_key_count": len(TEACHER_PCD_NPZ_KEYS),
    }, flush=True)


def _build_rows_from_dataset(args: argparse.Namespace) -> tuple[list[dict[str, Any]], str]:
    paths = iter_sample_paths_many(args.dataset)
    total = len(paths)
    workers = max(1, int(args.workers))
    t0 = time.perf_counter()
    rows: list[dict[str, Any]] = []
    groups: set[tuple[int, str, int]] = set()
    tasks: Iterable[tuple[str, float, float, int, str]] = (
        (str(path), float(args.alpha), float(args.beta), int(args.top_m), str(args.option_execution_semantics))
        for path in paths
    )
    if workers == 1:
        iterator = map(_build_raw_row, tasks)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        iterator = executor.map(_build_raw_row, tasks, chunksize=max(1, int(args.worker_chunksize)))
    try:
        for i, row in enumerate(iterator, 1):
            rows.append(row)
            groups.add((int(row["bucket"]), str(row["scene"]), int(row["time"])))
            if i == 1 or i % max(1, args.progress_every) == 0 or i == total:
                _progress(seen=i, total=total, groups=len(groups), t0=t0, workers=workers, mode="npz")
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
    return rows, "npz"


def _load_or_build_raw_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], str]:
    if args.reuse_raw_index and args.reuse_raw_summary:
        try:
            source_summary = json.loads(args.reuse_raw_summary.read_text(encoding="utf-8"))
            failures = _raw_reuse_validation(
                source_summary=source_summary, dataset=args.dataset, alpha=args.alpha,
                beta=args.beta, top_m=args.top_m,
                option_execution_semantics=args.option_execution_semantics,
            )
            if failures:
                raise ValueError("raw teacher cache contract mismatch: " + ",".join(failures))
            t0 = time.perf_counter()
            rows = _read_source_rows(args.reuse_raw_index)
            if int(source_summary.get("num_samples", -1)) != len(rows):
                raise ValueError("raw teacher cache row count mismatch")
            groups = {(int(r["bucket"]), str(r["scene"]), int(r["time"])) for r in rows}
            print({
                "event": "teacher_pcd_raw_reuse", "source_index": str(args.reuse_raw_index),
                "source_summary": str(args.reuse_raw_summary), "rows": len(rows), "groups": len(groups),
                "seconds": time.perf_counter() - t0, "dataset_manifests": _dataset_manifest_records(args.dataset),
            }, flush=True)
            return rows, "raw_reuse"
        except Exception as exc:
            if not args.reuse_raw_fallback_to_build:
                raise
            print({
                "event": "teacher_pcd_raw_reuse_rejected", "reason": repr(exc),
                "fallback": "rebuild_from_npz",
            }, flush=True)
    return _build_rows_from_dataset(args)


def _group_state(rows: list[dict[str, Any]]) -> tuple[
    set[tuple[int, str, int]], dict[tuple[int, str, int], dict[str, Any]],
    dict[tuple[int, str, int], dict[str, Any]],
]:
    groups: set[tuple[int, str, int]] = set()
    group_targets: dict[tuple[int, str, int], dict[str, Any]] = {}
    nominal_rows: dict[tuple[int, str, int], dict[str, Any]] = {}
    for row in rows:
        bucket, scene, time_index = int(row["bucket"]), str(row["scene"]), int(row["time"])
        key = (bucket, scene, time_index)
        groups.add(key)
        stats = group_targets.setdefault(key, {
            "nominal": float("-inf"), "best_recovery": float("-inf"),
            "best_macro": -1, "best_deployable_recovery": float("-inf"),
            "best_deployable_macro": -1, "scene": scene, "bucket": bucket,
        })
        if bool(row["nominal"]):
            if float(row["teacher_pcd"]) > float(stats["nominal"]):
                stats["nominal"] = float(row["teacher_pcd"])
                nominal_rows[key] = row
        else:
            if float(row["teacher_pcd"]) > float(stats["best_recovery"]):
                stats["best_recovery"] = float(row["teacher_pcd"])
                stats["best_macro"] = int(row["macro"])
    return groups, group_targets, nominal_rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Comma-separated OC-RAP roots")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--beta", type=float, default=0.2)
    ap.add_argument("--top-m", type=int, default=8)
    ap.add_argument("--option-execution-semantics", choices=["global", "observation_class"], default="global")
    ap.add_argument("--progress-every", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=max(1, int(os.environ.get("OCRAP_TEACHER_INDEX_WORKERS", "1"))))
    ap.add_argument("--worker-chunksize", type=int, default=max(1, int(os.environ.get("OCRAP_TEACHER_INDEX_CHUNKSIZE", "16"))))
    ap.add_argument("--reuse-raw-index", type=Path)
    ap.add_argument("--reuse-raw-summary", type=Path)
    ap.add_argument("--reuse-raw-fallback-to-build", action="store_true")
    ap.add_argument("--positive-gain", type=float, default=0.015)
    ap.add_argument("--deployable-macro-ids", default="2,3,5,6,7")
    ap.add_argument("--component-harm-drs-tolerance", type=float, default=0.05)
    ap.add_argument("--component-harm-dep-tolerance", type=float, default=0.05)
    ap.add_argument("--component-harm-gap-tolerance", type=float, default=0.05)
    ap.add_argument("--component-harm-hard-tolerance", type=float, default=0.05)
    ap.add_argument("--component-harm-proxy-tolerance", type=float, default=0.05)
    ap.add_argument("--dep-boundary-aligned", action="store_true")
    ap.add_argument("--gap-ordinal-only", action="store_true")
    ap.add_argument("--summary-output", type=Path)
    ap.add_argument("--min-positive-groups-near", type=int, default=0)
    ap.add_argument("--min-positive-groups-contact", type=int, default=0)
    ap.add_argument("--min-positive-scenes-near", type=int, default=0)
    ap.add_argument("--min-positive-scenes-contact", type=int, default=0)
    ap.add_argument("--quality-mode", choices=["strict", "warn", "off"], default="strict")
    args = ap.parse_args()
    if bool(args.reuse_raw_index) != bool(args.reuse_raw_summary):
        ap.error("--reuse-raw-index and --reuse-raw-summary must be supplied together")

    total_t0 = time.perf_counter()
    deployable_macro_ids = {int(x.strip()) for x in str(args.deployable_macro_ids).split(",") if x.strip()}
    tolerances = ComponentVetoTolerances(
        drs=args.component_harm_drs_tolerance,
        deployability_gate=args.component_harm_dep_tolerance,
        gap_discount=args.component_harm_gap_tolerance,
        hard_violation=args.component_harm_hard_tolerance,
        harm_proxy=args.component_harm_proxy_tolerance,
        deployability_boundary_aligned=bool(args.dep_boundary_aligned),
        gap_ordinal_only=bool(args.gap_ordinal_only),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_t0 = time.perf_counter()
    rows, source_mode = _load_or_build_raw_rows(args)
    raw_seconds = time.perf_counter() - raw_t0

    groups, group_targets, nominal_rows = _group_state(rows)
    # Second pass exactly matches the historical logic, including the strict
    # positive-gain and component-veto comparisons.
    harmful_candidates = beneficial_candidates = overlap_candidates = safe_beneficial_candidates = 0
    harmful_groups: set[tuple[int, str, int]] = set()
    overlap_groups: set[tuple[int, str, int]] = set()
    safe_beneficial_groups: set[tuple[int, str, int]] = set()
    factorized_candidate_counts: dict[int, Counter[str]] = {1: Counter(), 2: Counter()}
    factorized_harmful_groups: dict[int, set[tuple[int, str, int]]] = {1: set(), 2: set()}
    factorized_overlap_groups: dict[int, set[tuple[int, str, int]]] = {1: set(), 2: set()}
    factorized_safe_groups: dict[int, set[tuple[int, str, int]]] = {1: set(), 2: set()}
    for row in rows:
        key = (int(row["bucket"]), str(row["scene"]), int(row["time"]))
        nominal = nominal_rows.get(key)
        if row["nominal"] or nominal is None:
            row["component_veto_margin"] = 0.0
            row["component_harmful"] = False
            row["beneficial"] = False
            continue
        margin = component_veto_margin_numpy(
            candidate_drs=float(row["teacher_drs"]), nominal_drs=float(nominal["teacher_drs"]),
            candidate_r_dep=float(row["teacher_r_dep"]), nominal_r_dep=float(nominal["teacher_r_dep"]),
            candidate_gap=float(row["teacher_gap"]), nominal_gap=float(nominal["teacher_gap"]),
            candidate_hard=float(row["teacher_hard_violation"]), nominal_hard=float(nominal["teacher_hard_violation"]),
            candidate_harm_proxy=float(row["teacher_harm_proxy"]), nominal_harm_proxy=float(nominal["teacher_harm_proxy"]),
            tolerances=tolerances,
        )
        beneficial = float(row["teacher_pcd"]) - float(nominal["teacher_pcd"]) >= float(args.positive_gain)
        harmful = margin > 0.0
        row["component_veto_margin"] = float(margin)
        row["component_harmful"] = bool(harmful)
        row["beneficial"] = bool(beneficial)
        if int(row["macro"]) in deployable_macro_ids:
            beneficial_candidates += int(beneficial)
            harmful_candidates += int(harmful)
            overlap_candidates += int(beneficial and harmful)
            safe_beneficial = bool(beneficial and not harmful)
            safe_beneficial_candidates += int(safe_beneficial)
            bucket_counts = factorized_candidate_counts.setdefault(int(row["bucket"]), Counter())
            bucket_counts["deployable_candidates"] += 1
            bucket_counts["beneficial_candidates"] += int(beneficial)
            bucket_counts["component_harmful_candidates"] += int(harmful)
            bucket_counts["overlap_candidates"] += int(beneficial and harmful)
            bucket_counts["safe_beneficial_candidates"] += int(safe_beneficial)
            if harmful:
                harmful_groups.add(key)
                factorized_harmful_groups.setdefault(int(row["bucket"]), set()).add(key)
            if beneficial and harmful:
                overlap_groups.add(key)
                factorized_overlap_groups.setdefault(int(row["bucket"]), set()).add(key)
            if safe_beneficial:
                safe_beneficial_groups.add(key)
                factorized_safe_groups.setdefault(int(row["bucket"]), set()).add(key)

    # Recompute deployable bests here so raw-source reuse is independent of the
    # deployable macro set used by the source index.
    for stats in group_targets.values():
        stats["best_deployable_recovery"] = float("-inf")
        stats["best_deployable_macro"] = -1
    for row in rows:
        if row["nominal"] or int(row["macro"]) not in deployable_macro_ids:
            continue
        key = (int(row["bucket"]), str(row["scene"]), int(row["time"]))
        stats = group_targets[key]
        if float(row["teacher_pcd"]) > float(stats["best_deployable_recovery"]):
            stats["best_deployable_recovery"] = float(row["teacher_pcd"])
            stats["best_deployable_macro"] = int(row["macro"])

    write_t0 = time.perf_counter()
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(args.output)
    write_seconds = time.perf_counter() - write_t0

    positive_all = [
        stats for stats in group_targets.values()
        if np.isfinite(stats["nominal"]) and np.isfinite(stats["best_recovery"])
        and stats["best_recovery"] - stats["nominal"] >= args.positive_gain
    ]
    positive = [
        stats for stats in group_targets.values()
        if np.isfinite(stats["nominal"]) and np.isfinite(stats["best_deployable_recovery"])
        and stats["best_deployable_recovery"] - stats["nominal"] >= args.positive_gain
    ]

    def bucket_summary(bucket: int, rows_source: list[dict[str, Any]], macro_key: str) -> dict[str, Any]:
        selected = [x for x in rows_source if int(x["bucket"]) == bucket]
        scenes = Counter(str(x["scene"]) for x in selected)
        macros = Counter(int(x[macro_key]) for x in selected)
        n = len(selected)
        return {
            "positive_groups": n,
            "positive_scenes": len(scenes),
            "positive_macro_counts": {str(k): int(v) for k, v in sorted(macros.items())},
            "max_positive_macro_share": (max(macros.values()) / n) if n and macros else None,
            "top10_positive_scene_share": (sum(v for _, v in scenes.most_common(10)) / n) if n else None,
        }

    by_bucket = {"near": bucket_summary(1, positive, "best_deployable_macro"),
                 "contact": bucket_summary(2, positive, "best_deployable_macro")}
    all_macro_by_bucket = {"near": bucket_summary(1, positive_all, "best_macro"),
                           "contact": bucket_summary(2, positive_all, "best_macro")}
    screening_thresholds = {
        "near": {"adequate_groups": 80, "adequate_scenes": 40, "marginal_groups": 20, "marginal_scenes": 10},
        "contact": {"adequate_groups": 60, "adequate_scenes": 30, "marginal_groups": 15, "marginal_scenes": 8},
    }
    for name, lim in screening_thresholds.items():
        got = by_bucket[name]
        groups_n, scenes_n = int(got["positive_groups"]), int(got["positive_scenes"])
        concentrated = bool(
            (got.get("max_positive_macro_share") is not None and float(got["max_positive_macro_share"]) > 0.80)
            or (got.get("top10_positive_scene_share") is not None and float(got["top10_positive_scene_share"]) > 0.60)
        )
        if groups_n >= lim["adequate_groups"] and scenes_n >= lim["adequate_scenes"] and not concentrated:
            status, action = "adequate_for_direction_screening", "reuse_existing_dataset"
        elif groups_n >= lim["marginal_groups"] and scenes_n >= lim["marginal_scenes"]:
            status, action = "marginal_debug_only", "run_screening_but_do_not_make_strong_claims"
        else:
            status, action = "data_limited", "consider_targeted_increment_or_rebuild_after_audit"
        got.update(screening_status=status, screening_concentration_warning=concentrated, screening_recommended_action=action)

    dataset_roots = _dataset_roots(args.dataset)
    dataset_manifests = _dataset_manifest_records(args.dataset)
    index_contract = {
        "dataset_roots": dataset_roots,
        "dataset_manifests": dataset_manifests,
        "alpha": float(args.alpha), "beta": float(args.beta), "top_m": int(args.top_m),
        "positive_gain": float(args.positive_gain),
        "deployable_macro_ids": sorted(deployable_macro_ids),
        "component_harm_tolerances": tolerances.__dict__,
    }
    summary = {
        "event": "teacher_pcd_index_complete", "output": str(args.output),
        "num_samples": len(rows), "num_groups": len(groups), "alpha": args.alpha,
        "beta": args.beta, "top_m": args.top_m, "positive_gain": args.positive_gain,
        "option_execution_semantics": args.option_execution_semantics,
        "deployable_macro_ids": sorted(deployable_macro_ids),
        "positive_advantage_groups": len(positive),
        "positive_advantage_groups_all_macros": len(positive_all),
        "component_harm_tolerances": tolerances.__dict__,
        "component_harmful_candidates": harmful_candidates,
        "beneficial_candidates": beneficial_candidates,
        "beneficial_and_component_harmful_candidates": overlap_candidates,
        "safe_beneficial_candidates": safe_beneficial_candidates,
        "component_harmful_groups": len(harmful_groups),
        "beneficial_and_component_harmful_groups": len(overlap_groups),
        "safe_beneficial_groups": len(safe_beneficial_groups),
        "safe_beneficial_scenes": len({(bucket, scene) for bucket, scene, _ in safe_beneficial_groups}),
        "index_contract": index_contract,
        "factorized_harm_support_by_bucket": {
            name: {
                **{k: int(v) for k, v in factorized_candidate_counts.get(bucket, Counter()).items()},
                "component_harmful_groups": len(factorized_harmful_groups.get(bucket, set())),
                "beneficial_and_component_harmful_groups": len(factorized_overlap_groups.get(bucket, set())),
                "safe_beneficial_groups": len(factorized_safe_groups.get(bucket, set())),
                "safe_beneficial_scenes": len({scene for _, scene, _ in factorized_safe_groups.get(bucket, set())}),
            }
            for name, bucket in (("near", 1), ("contact", 2))
        },
        "by_bucket": by_bucket, "all_macro_by_bucket": all_macro_by_bucket,
        "quality_mode": args.quality_mode,
        "runtime": {
            "source_mode": source_mode, "workers": int(args.workers),
            "worker_chunksize": int(args.worker_chunksize), "npz_key_count": len(TEACHER_PCD_NPZ_KEYS),
            "raw_seconds": raw_seconds, "write_seconds": write_seconds,
            "total_seconds": time.perf_counter() - total_t0,
        },
    }
    failures: list[str] = []
    limits = {
        "near": (args.min_positive_groups_near, args.min_positive_scenes_near),
        "contact": (args.min_positive_groups_contact, args.min_positive_scenes_contact),
    }
    for name, (min_groups, min_scenes) in limits.items():
        got = by_bucket[name]
        if int(got["positive_groups"]) < min_groups:
            failures.append(f"{name}: positive groups {got['positive_groups']} < {min_groups}")
        if int(got["positive_scenes"]) < min_scenes:
            failures.append(f"{name}: positive scenes {got['positive_scenes']} < {min_scenes}")
    if args.quality_mode == "off":
        failures = []
    summary["quality_failures"] = failures
    summary["quality_passed"] = not failures
    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(summary, flush=True)
    if failures:
        event = "teacher_pcd_index_quality_failure" if args.quality_mode == "strict" else "teacher_pcd_index_quality_warning"
        print({"event": event, "failures": failures}, flush=True)
        if args.quality_mode == "strict":
            return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
