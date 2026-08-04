#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ocrap.data.serialization import write_json
from ocrap.simulation.closed_loop_runner import (
    _aggregate_with_buckets,
    _scene_resume_key,
    _scene_storage_view,
)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def _journal_path(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".scenes.jsonl")


def _progress_path(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".progress.json")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Finalize a closed-loop summary from a complete append-only scene journal without rerunning simulation."
    )
    ap.add_argument("--output", type=Path, required=True, help="closed_loop_*.json path")
    ap.add_argument("--expected-count", type=int, default=0)
    ap.add_argument("--source", default="")
    ap.add_argument("--allow-incomplete", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    output = args.output
    journal = _journal_path(output)
    progress_path = _progress_path(output)
    if not journal.is_file():
        print(json.dumps({"event": "closed_loop_journal_finalize", "status": "no_journal", "journal": str(journal)}))
        return 3

    progress = _read_json(progress_path) or {}
    expected = int(args.expected_count or progress.get("requested_rollouts") or 0)
    scenes: list[dict[str, Any]] = []
    seen: set[str] = set()
    fingerprints: set[str] = set()
    torn_lines = 0
    with journal.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                torn_lines += 1
                continue
            if not isinstance(record, dict):
                continue
            scene = record.get("scene", record)
            if not isinstance(scene, dict):
                continue
            fp = str(record.get("run_fingerprint", "") or "")
            if fp:
                fingerprints.add(fp)
            key = _scene_resume_key(scene)
            if not key or key in seen:
                continue
            seen.add(key)
            # Old v50 journals may contain every decision. Keep only the fields
            # required by aggregation, pairing and critical-scene selection.
            scenes.append(_scene_storage_view(scene, "metrics"))

    if len(fingerprints) > 1:
        raise SystemExit(f"journal contains multiple run fingerprints: {sorted(fingerprints)}")
    if expected <= 0:
        print(json.dumps({
            "event": "closed_loop_journal_finalize",
            "status": "unknown_expected_count",
            "journal": str(journal),
            "completed": len(scenes),
            "hint": "Pass --expected-count, or retain the original progress JSON with requested_rollouts.",
        }))
        return 5
    complete = len(scenes) == expected
    if not complete and not args.allow_incomplete:
        print(json.dumps({
            "event": "closed_loop_journal_finalize",
            "status": "incomplete",
            "journal": str(journal),
            "completed": len(scenes),
            "expected": expected,
            "torn_lines": torn_lines,
        }))
        return 4
    if not scenes:
        return 4

    method = str(scenes[0].get("method") or "unknown")
    source = str(args.source or ("model" if method == "ocrap" else "observation_only_external_policy"))
    result = _aggregate_with_buckets(scenes, method, source)
    result.update({
        "bucket_target_count": expected,
        "bucket_matched_rollouts": len(scenes),
        "run_fingerprint": next(iter(fingerprints), str(progress.get("run_fingerprint") or "")),
        "resume_supported": True,
        "resume": {
            "enabled": True,
            "resumed_rollouts": len(scenes),
            "sources": ["journal_finalize"],
            "legacy_sources": [],
            "journal_path": str(journal),
            "progress_path": str(progress_path),
            "granularity": "completed_scene_or_bucket_target",
        },
        "scene_storage_detail": "metrics",
        "scene_journal_detail": "metrics_or_legacy_full",
        "scenes_embedded": False,
        "gamma_rec": float(scenes[0].get("gamma_rec", 0.0) or 0.0),
        "closed_loop_speed_config": {
            "journal_finalize": True,
            "result_scene_detail": "metrics",
            "scene_journal_detail": "metrics_or_legacy_full",
            "memory_scene_detail": "metrics",
            "include_scenes_in_result": False,
            "include_scenes_in_partial": False,
        },
        "warnings": [
            "Final summary reconstructed from the append-only scene journal after an interrupted finalization stage."
        ],
    })
    buckets = sorted({str(s.get("bucket_name")) for s in scenes if s.get("bucket_name")})
    result["bucket_dataset"] = None
    result["reconstructed_bucket_names"] = buckets
    if args.dry_run:
        print(json.dumps({"event": "closed_loop_journal_finalize", "status": "would_finalize", "completed": len(scenes), "expected": expected, "output": str(output)}))
        return 0

    write_json(result, output)
    write_json({
        "version": 1,
        "run_fingerprint": result["run_fingerprint"],
        "status": "complete" if complete else "incomplete_finalized",
        "completed_rollouts": len(scenes),
        "requested_rollouts": expected,
        "resumed_rollouts": int(progress.get("resumed_rollouts") or 0),
        "current": None,
        "finalized_from_journal": True,
    }, progress_path)
    print(json.dumps({
        "event": "closed_loop_journal_finalize",
        "status": "complete" if complete else "incomplete_finalized",
        "completed": len(scenes),
        "expected": expected,
        "torn_lines": torn_lines,
        "output": str(output),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
