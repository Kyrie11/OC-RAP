#!/usr/bin/env python3
"""Preflight the OC-RAP offline-root -> WOMD/Waymax closed-loop contract.

The v48.34.1 checker also validates split selection, target source role and any
explicit @N scan limit against stored source_scenario_index values.  It still
does not claim raw TFRecord membership without actually running the evaluator.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from ocrap.models.data import iter_sample_paths_many, scalar_metadata_for_path


def _split_limit(spec: str) -> tuple[str, int | None]:
    text = str(spec or "").strip()
    if "@" in text:
        base, suffix = text.rsplit("@", 1)
        if suffix.isdigit():
            return base, int(suffix)
    return text, None


def _resolve_raw_specs(spec: str) -> list[str]:
    found: list[str] = []
    for part in (x.strip() for x in str(spec or "").split(",")):
        if not part:
            continue
        part, _ = _split_limit(part)
        matches = sorted(glob.glob(part)) if any(c in part for c in "*?[") else ([part] if Path(part).exists() else [])
        found.extend(matches)
    return sorted(dict.fromkeys(found))


def _source_role(text: str) -> str:
    x = text.lower()
    if "validation_interactive" in x:
        return "validation_interactive"
    if re.search(r"(^|[/_])validation([/_]|$)", x):
        return "validation"
    if "training" in x:
        return "training"
    if "testing" in x or "/test" in x:
        return "test"
    return "unknown"


def _root_label(path: Path) -> str:
    for parent in [path.parent, *path.parents]:
        low = parent.name.lower()
        if "near" in low:
            return "near_contact"
        if "contact" in low:
            return "contact"
        if "safe" in low:
            return "safe"
    return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description="Check whether an OC-RAP bucket can drive Waymax closed-loop replay.")
    ap.add_argument("--dataset", required=True, help="OC-RAP offline dataset root/spec containing .npz samples")
    ap.add_argument("--womd-pattern", required=True, help="Raw WOMD TFRecord path/glob used by closed-loop")
    ap.add_argument("--split", default="", help="Optional exact split_id filter")
    ap.add_argument("--expected-source-role", default="auto", help="auto or exact target/raw WOMD source role")
    ap.add_argument("--max-scan", type=int, default=0, help="Limit metadata scan; 0 means all samples")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    paths = list(iter_sample_paths_many(args.dataset))
    if args.max_scan > 0:
        paths = paths[: args.max_scan]
    split_counts: Counter[str] = Counter()
    bucket_counts: Counter[str] = Counter()
    target_roles: Counter[str] = Counter()
    missing_scene = 0
    missing_time = 0
    valid_rows = 0
    unique_targets: set[tuple[str, int]] = set()
    source_indices: list[int] = []
    official_id_rows = 0
    examples: list[dict[str, Any]] = []

    for p in paths:
        split = str(scalar_metadata_for_path(p, "split_id", ""))
        split_counts[split or "<missing>"] += 1
        if args.split and split != args.split:
            continue
        target_roles[str(scalar_metadata_for_path(p, "womd_source_role", "") or "unknown")] += 1
        scene = str(scalar_metadata_for_path(p, "scene_id", "") or "").strip()
        official = str(scalar_metadata_for_path(p, "official_scenario_id", "") or "").strip()
        if official:
            official_id_rows += 1
        if not scene and not official:
            missing_scene += 1
            continue
        raw_time = scalar_metadata_for_path(p, "time_index", scalar_metadata_for_path(p, "target_time_index", None))
        try:
            if raw_time is None:
                raise ValueError
            time_index = int(float(raw_time))
        except Exception:
            missing_time += 1
            continue
        raw_idx = scalar_metadata_for_path(p, "source_scenario_index", -1)
        try:
            idx = int(float(raw_idx))
        except Exception:
            idx = -1
        if idx >= 0:
            source_indices.append(idx)
        valid_rows += 1
        bucket = _root_label(Path(p)); bucket_counts[bucket] += 1
        unique_targets.add((official or scene, time_index))
        if len(examples) < 5:
            examples.append({"scene_id": scene, "official_scenario_id": official, "time_index": time_index, "source_scenario_index": idx, "bucket": bucket, "sample": str(p)})

    raw_files = _resolve_raw_specs(args.womd_pattern)
    limits = [_split_limit(x.strip())[1] for x in str(args.womd_pattern).split(",") if x.strip()]
    finite_limits = [x for x in limits if x is not None]
    scan_limit = finite_limits[0] if len(finite_limits) == 1 else None
    max_source_index = max(source_indices) if source_indices else None
    scan_limit_covers_indices = scan_limit is None or max_source_index is None or max_source_index < scan_limit
    raw_role = _source_role(args.womd_pattern)
    known_roles = {x for x in target_roles if x not in {"", "unknown"}}
    source_role_valid = (args.expected_source_role == "auto" or raw_role == args.expected_source_role) and (not known_roles or raw_role in known_roles)
    split_valid = bool(valid_rows > 0)
    report = {
        "event": "v48_34_1_closed_loop_dataset_support",
        "dataset": args.dataset,
        "womd_pattern": args.womd_pattern,
        "split_filter": args.split,
        "num_samples_scanned": len(paths),
        "num_valid_sample_rows": valid_rows,
        "num_unique_scene_time_targets": len(unique_targets),
        "missing_scene_id_rows": missing_scene,
        "missing_time_index_rows": missing_time,
        "split_counts": dict(sorted(split_counts.items())),
        "bucket_counts_after_split": dict(sorted(bucket_counts.items())),
        "target_source_roles_after_split": dict(sorted(target_roles.items())),
        "raw_source_role": raw_role,
        "expected_source_role": args.expected_source_role,
        "source_role_valid": source_role_valid,
        "official_id_rows": official_id_rows,
        "source_index_rows": len(source_indices),
        "max_source_scenario_index": max_source_index,
        "raw_scenario_scan_limit": scan_limit,
        "scan_limit_covers_target_indices": scan_limit_covers_indices,
        "resolved_womd_files": raw_files[:20],
        "num_resolved_womd_files": len(raw_files),
        "examples": examples,
        "schema_supports_closed_loop": bool(split_valid and unique_targets and raw_files and source_role_valid and scan_limit_covers_indices),
        "limitations": [
            "This preflight verifies metadata, split, source-role and explicit scan-limit contracts.",
            "The closed-loop runner must still match scenario IDs while scanning the specified WOMD files.",
            "Route metrics require raw v1.3.1 path_samples/sdc_paths fields and closed_loop.use_sdc_paths=true; route-free dynamics metrics do not.",
        ],
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["schema_supports_closed_loop"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
