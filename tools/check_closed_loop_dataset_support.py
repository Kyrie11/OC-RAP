#!/usr/bin/env python3
"""Preflight the OC-RAP offline-root -> WOMD/Waymax closed-loop contract.

Important: a suffix such as ``validation_tfexample.tfrecord@150`` is a
TensorFlow/Waymax *shard-count declaration*.  It resolves files named
``...-00000-of-00150`` ... ``...-00149-of-00150`` and is never interpreted as
an upper bound on scenario indices.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from ocrap.data.womd.sharded_path import resolve_womd_spec
from ocrap.models.data import iter_sample_paths_many, scalar_metadata_for_path

_LEGACY_SOURCE_INDEX_RE = re.compile(r"__wx(?P<index>[0-9]+)(?:$|[^0-9])")


def _legacy_source_index(*values: str) -> int:
    for value in values:
        match = _LEGACY_SOURCE_INDEX_RE.search(str(value or ""))
        if match:
            return int(match.group("index"))
    return -1




def _canonical_scene_id(value: Any) -> str:
    return re.sub(r"__wx\d{8}$", "", str(value or "").strip())


def _load_target_keys(path_value: str | Path | None) -> set[str]:
    raw = str(path_value or "").strip()
    if not raw:
        return set()
    path = Path(raw)
    if not path.is_file():
        raise FileNotFoundError(f"target keys file does not exist: {path}")

    def collect(value: Any, out: set[str]) -> None:
        if isinstance(value, str):
            value = value.strip()
            if value.startswith("target:"):
                value = value[len("target:"):]
            if value:
                out.add(value)
        elif isinstance(value, list):
            for item in value:
                collect(item, out)
        elif isinstance(value, dict):
            for key in ("target_key", "resume_key"):
                if value.get(key):
                    collect(str(value[key]), out)
                    return
            for key in ("target_keys", "selected", "items", "scenes"):
                if key in value:
                    collect(value[key], out)

    keys: set[str] = set()
    text = path.read_text(encoding="utf-8")
    try:
        collect(json.loads(text), keys)
    except json.JSONDecodeError:
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                collect(json.loads(line), keys)
            except json.JSONDecodeError:
                collect(line, keys)
    return keys


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


def _dataset_label(path: Path) -> str:
    """Return the exact bucket label used by the closed-loop runner."""
    try:
        return path.parent.parent.name if path.parent.name == "samples" else path.parent.name
    except Exception:
        return "dataset"


def _regime_label(path: Path) -> str:
    """Return a human-readable regime label without changing target-key identity."""
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
    ap.add_argument("--womd-pattern", "--womd", dest="womd_pattern", required=True, help="Raw WOMD TFRecord file/glob or sharded spec such as prefix@150")
    ap.add_argument("--target-keys-file", default="", help="Optional selected target-key JSON/JSONL/text file")
    ap.add_argument("--require-target-keys", action="store_true", help="Fail when requested target keys are absent from the bucket dataset")
    ap.add_argument("--split", default="", help="Optional exact split_id filter")
    ap.add_argument("--expected-source-role", default="auto", help="auto or exact target/raw WOMD source role")
    ap.add_argument("--max-scan", type=int, default=0, help="Limit offline metadata scan; 0 means all samples")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    paths = list(iter_sample_paths_many(args.dataset))
    if args.max_scan > 0:
        paths = paths[: args.max_scan]
    split_counts: Counter[str] = Counter()
    bucket_counts: Counter[str] = Counter()
    regime_counts: Counter[str] = Counter()
    target_roles: Counter[str] = Counter()
    missing_scene = 0
    missing_time = 0
    valid_rows = 0
    requested_target_keys = _load_target_keys(args.target_keys_file)
    available_target_keys: set[str] = set()
    unique_targets: set[tuple[str, int]] = set()
    unique_scenes: set[str] = set()
    source_indices: list[int] = []
    legacy_source_index_rows = 0
    official_id_rows = 0
    examples: list[dict[str, Any]] = []

    for p in paths:
        split = str(scalar_metadata_for_path(p, "split_id", ""))
        split_counts[split or "<missing>"] += 1
        if args.split and split != args.split:
            continue
        target_roles[str(scalar_metadata_for_path(p, "womd_source_role", "") or "unknown")] += 1
        scene = str(scalar_metadata_for_path(p, "scene_id", "") or "").strip()
        original = str(scalar_metadata_for_path(p, "original_scenario_id", "") or "").strip()
        official = str(scalar_metadata_for_path(p, "official_scenario_id", "") or "").strip()
        if official:
            official_id_rows += 1
        canonical = _canonical_scene_id(official or original or scene)
        if not canonical:
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
        if idx < 0:
            idx = _legacy_source_index(original, scene)
            if idx >= 0:
                legacy_source_index_rows += 1
        if idx >= 0:
            source_indices.append(idx)
        valid_rows += 1
        bucket = _dataset_label(Path(p))
        bucket_counts[bucket] += 1
        regime_counts[_regime_label(Path(p))] += 1
        unique_targets.add((canonical, time_index))
        unique_scenes.add(canonical)
        available_target_keys.add(f"{bucket}:{canonical}:t{time_index}")
        if len(examples) < 5:
            examples.append({
                "scene_id": scene,
                "official_scenario_id": official,
                "time_index": time_index,
                "source_scenario_index": idx,
                "bucket": bucket,
                "sample": str(p),
            })

    missing_target_keys = sorted(requested_target_keys - available_target_keys)
    target_keys_valid = not missing_target_keys
    womd = resolve_womd_spec(args.womd_pattern)
    raw_role = _source_role(args.womd_pattern)
    known_roles = {x for x in target_roles if x not in {"", "unknown"}}
    expected_role_valid = args.expected_source_role == "auto" or raw_role == args.expected_source_role
    metadata_role_valid = not known_roles or raw_role in known_roles
    source_role_valid = expected_role_valid and metadata_role_valid
    split_valid = bool(valid_rows > 0)
    limitations = [
        "This preflight verifies offline target metadata, source role and the complete WOMD shard/file set.",
        "The @N suffix declares N TFRecord shards; it does not cap scenarios or source_scenario_index.",
        "The closed-loop runner must still match scenario IDs while decoding the specified WOMD records.",
        "Route metrics require raw v1.3.1 path_samples/sdc_paths fields and closed_loop.use_sdc_paths=true; route-free dynamics metrics do not.",
    ]
    if source_indices:
        limitations.append(
            "source_scenario_index is used only for legacy target matching and an optional trailing-scan bound; it is independent of the shard count."
        )
    report = {
        "event": "v50_closed_loop_dataset_support",
        "dataset": args.dataset,
        "womd_pattern": args.womd_pattern,
        "split_filter": args.split,
        "target_keys_file": args.target_keys_file or None,
        "num_requested_target_keys": len(requested_target_keys),
        "num_matching_requested_target_keys": len(requested_target_keys) - len(missing_target_keys),
        "missing_requested_target_keys": missing_target_keys[:50],
        "target_keys_valid": target_keys_valid,
        "num_samples_scanned": len(paths),
        "num_valid_sample_rows": valid_rows,
        "num_unique_scene_time_targets": len(unique_targets),
        "num_unique_scenes": len(unique_scenes),
        "default_closed_loop_targets_at_max_per_scene_1": len(unique_scenes),
        "missing_scene_id_rows": missing_scene,
        "missing_time_index_rows": missing_time,
        "split_counts": dict(sorted(split_counts.items())),
        "bucket_counts_after_split": dict(sorted(bucket_counts.items())),
        "regime_counts_after_split": dict(sorted(regime_counts.items())),
        "target_source_roles_after_split": dict(sorted(target_roles.items())),
        "raw_source_role": raw_role,
        "expected_source_role": args.expected_source_role,
        "source_role_valid": source_role_valid,
        "official_id_rows": official_id_rows,
        "source_index_rows": len(source_indices),
        "legacy_source_index_rows": legacy_source_index_rows,
        "max_source_scenario_index": max(source_indices) if source_indices else None,
        # Kept as an explicit null for downstream readers of the previous schema.
        "raw_scenario_scan_limit": None,
        "scan_limit_covers_target_indices": None,
        "womd_shard_spec": womd.as_dict(),
        "resolved_womd_files": list(womd.files[:20]),
        "num_resolved_womd_files": len(womd.files),
        "examples": examples,
        "schema_supports_closed_loop": bool(split_valid and unique_targets and womd.valid and source_role_valid and (target_keys_valid or not args.require_target_keys)),
        "limitations": limitations,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["schema_supports_closed_loop"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
