#!/usr/bin/env python3
"""Precompute exact teacher PCD and factorized evidence metadata.

The index is separate from the dataset and can be regenerated without changing
any OC-RAP sample.  v48.19 adds a non-compensatory component-harm label so the
training sampler and held-out certificate share the same evidence semantics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ocrap.algorithms.evidence_targets import ComponentVetoTolerances, component_veto_margin_numpy
from ocrap.algorithms.ocmero import oc_mero
from ocrap.data.serialization import load_npz_selected
from ocrap.evaluation.metrics import (
    best_option_indices,
    deployable_recovery_success,
    post_contact_deployability_score,
)
from ocrap.models.data import MODEL_SAMPLE_NPZ_KEYS, bucket_id_for_path, iter_sample_paths_many


def _scalar(d: dict[str, Any], key: str, default: Any) -> Any:
    a = np.asarray(d.get(key, default))
    return a.item() if a.shape == () else a


def teacher_components(d: dict[str, Any], *, alpha: float, beta: float, top_m: int, option_execution_semantics: str = "global") -> dict[str, Any]:
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
    fresh_r_dep = float(result.r_dep)
    fresh_r_orc = float(result.r_orc)
    cached_r_dep_present = "r_dep_star" in d
    cached_r_orc_present = "r_orc_star" in d
    r_dep = float(_scalar(d, "r_dep_star", fresh_r_dep))
    r_orc = float(_scalar(d, "r_orc_star", fresh_r_orc))
    gap = max(0.0, r_orc - r_dep)
    fresh_gap = max(0.0, fresh_r_orc - fresh_r_dep)
    return {
        "teacher_pcd": float(post_contact_deployability_score(drs, r_dep, gap)),
        "teacher_drs": drs,
        "teacher_r_dep": r_dep,
        "teacher_gap": gap,
        # The index already recomputes OC-MERO from stored m_star/root_probs/C.
        # Persist those fresh values so v48.56 source-label correctness can be
        # audited without reopening every NPZ a second time after index build.
        "fresh_ocmero_r_dep": fresh_r_dep,
        "fresh_ocmero_r_orc": fresh_r_orc,
        "fresh_ocmero_gap": fresh_gap,
        "fresh_ocmero_r_dep_abs_error": abs(r_dep - fresh_r_dep),
        "fresh_ocmero_r_orc_abs_error": abs(r_orc - fresh_r_orc),
        "fresh_ocmero_gap_abs_error": abs(gap - fresh_gap),
        "cached_r_dep_present": bool(cached_r_dep_present),
        "cached_r_orc_present": bool(cached_r_orc_present),
        "teacher_hard_violation": float(_scalar(d, "hard_violation", 0.0)),
        "teacher_harm_proxy": float(_scalar(d, "harm_proxy", 0.0)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Comma-separated OC-RAP roots")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--beta", type=float, default=0.2)
    ap.add_argument("--top-m", type=int, default=8)
    ap.add_argument("--option-execution-semantics", choices=["global", "observation_class"], default="global")
    ap.add_argument("--progress-every", type=int, default=1000)
    ap.add_argument("--positive-gain", type=float, default=0.015)
    ap.add_argument("--deployable-macro-ids", default="2,3,5,6,7")
    ap.add_argument("--component-harm-drs-tolerance", type=float, default=0.05)
    ap.add_argument("--component-harm-dep-tolerance", type=float, default=0.05)
    ap.add_argument("--component-harm-gap-tolerance", type=float, default=0.05)
    ap.add_argument("--component-harm-hard-tolerance", type=float, default=0.05)
    ap.add_argument("--component-harm-proxy-tolerance", type=float, default=0.05)
    ap.add_argument("--summary-output", type=Path)
    ap.add_argument("--min-positive-groups-near", type=int, default=0)
    ap.add_argument("--min-positive-groups-contact", type=int, default=0)
    ap.add_argument("--min-positive-scenes-near", type=int, default=0)
    ap.add_argument("--min-positive-scenes-contact", type=int, default=0)
    ap.add_argument("--quality-mode", choices=["strict", "warn", "off"], default="strict")
    args = ap.parse_args()

    deployable_macro_ids = {int(x.strip()) for x in str(args.deployable_macro_ids).split(",") if x.strip()}
    tolerances = ComponentVetoTolerances(
        drs=args.component_harm_drs_tolerance,
        deployability_gate=args.component_harm_dep_tolerance,
        gap_discount=args.component_harm_gap_tolerance,
        hard_violation=args.component_harm_hard_tolerance,
        harm_proxy=args.component_harm_proxy_tolerance,
    )
    paths = iter_sample_paths_many(args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    groups: set[tuple[int, str, int]] = set()
    group_targets: dict[tuple[int, str, int], dict[str, Any]] = {}
    nominal_rows: dict[tuple[int, str, int], dict[str, Any]] = {}

    for i, path in enumerate(paths, 1):
        d = load_npz_selected(path, MODEL_SAMPLE_NPZ_KEYS)
        scene = str(_scalar(d, "scene_id", path.stem))
        time_index = int(_scalar(d, "time_index", 0))
        bucket = int(bucket_id_for_path(path))
        row: dict[str, Any] = {
            "path": str(path.resolve()),
            "bucket": bucket,
            "scene": scene,
            "time": time_index,
            "candidate": int(_scalar(d, "candidate_index", 0)),
            "macro": int(_scalar(d, "prefix_macro_type_id", _scalar(d, "prefix_macro_id", -1))),
            "nominal": bool(float(_scalar(d, "is_nominal", 0.0)) > 0.5),
            **teacher_components(
                d, alpha=args.alpha, beta=args.beta, top_m=args.top_m,
                option_execution_semantics=args.option_execution_semantics,
            ),
        }
        rows.append(row)
        key = (bucket, scene, time_index)
        groups.add(key)
        stats = group_targets.setdefault(key, {
            "nominal": float("-inf"), "best_recovery": float("-inf"),
            "best_macro": -1, "best_deployable_recovery": float("-inf"),
            "best_deployable_macro": -1, "scene": scene, "bucket": bucket,
        })
        if row["nominal"]:
            if float(row["teacher_pcd"]) > float(stats["nominal"]):
                stats["nominal"] = float(row["teacher_pcd"])
                nominal_rows[key] = row
        else:
            if float(row["teacher_pcd"]) > float(stats["best_recovery"]):
                stats["best_recovery"] = float(row["teacher_pcd"])
                stats["best_macro"] = int(row["macro"])
            if int(row["macro"]) in deployable_macro_ids and float(row["teacher_pcd"]) > float(stats["best_deployable_recovery"]):
                stats["best_deployable_recovery"] = float(row["teacher_pcd"])
                stats["best_deployable_macro"] = int(row["macro"])
        if i == 1 or i % max(1, args.progress_every) == 0 or i == len(paths):
            print({"event": "teacher_pcd_index_progress", "seen": i, "total": len(paths), "groups": len(groups)}, flush=True)

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

    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(args.output)

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

    dataset_roots = [str(Path(x.strip()).resolve()) for x in str(args.dataset).split(",") if x.strip()]
    dataset_manifests = []
    for dataset_root in dataset_roots:
        manifest = Path(dataset_root) / "manifest.csv"
        dataset_manifests.append({
            "root": dataset_root,
            "manifest": str(manifest),
            "manifest_sha256": (
                hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.is_file() else None
            ),
        })
    index_contract = {
        "dataset_roots": dataset_roots,
        "dataset_manifests": dataset_manifests,
        "alpha": float(args.alpha),
        "beta": float(args.beta),
        "top_m": int(args.top_m),
        "positive_gain": float(args.positive_gain),
        "deployable_macro_ids": sorted(deployable_macro_ids),
        "component_harm_tolerances": tolerances.__dict__,
    }
    summary = {
        "event": "teacher_pcd_index_complete", "output": str(args.output),
        "num_samples": len(paths), "num_groups": len(groups), "alpha": args.alpha,
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
        "source_recomputation": {
            "checked_samples": len(rows),
            "atol": 1e-6,
            "max_abs_error": {
                "r_dep": max((float(r.get("fresh_ocmero_r_dep_abs_error", 0.0)) for r in rows), default=0.0),
                "r_orc": max((float(r.get("fresh_ocmero_r_orc_abs_error", 0.0)) for r in rows), default=0.0),
                "gap": max((float(r.get("fresh_ocmero_gap_abs_error", 0.0)) for r in rows), default=0.0),
            },
            "mismatch_counts": {
                "r_dep": sum(float(r.get("fresh_ocmero_r_dep_abs_error", 0.0)) > 1e-6 for r in rows),
                "r_orc": sum(float(r.get("fresh_ocmero_r_orc_abs_error", 0.0)) > 1e-6 for r in rows),
                "gap": sum(float(r.get("fresh_ocmero_gap_abs_error", 0.0)) > 1e-6 for r in rows),
            },
            "missing_cached_label_counts": {
                "r_dep": sum(not bool(r.get("cached_r_dep_present", False)) for r in rows),
                "r_orc": sum(not bool(r.get("cached_r_orc_present", False)) for r in rows),
            },
        },
    }
    summary["source_recomputation"]["source_labels_match_fresh_ocmero"] = (
        all(int(v) == 0 for v in summary["source_recomputation"]["mismatch_counts"].values())
        and all(int(v) == 0 for v in summary["source_recomputation"]["missing_cached_label_counts"].values())
    )
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
