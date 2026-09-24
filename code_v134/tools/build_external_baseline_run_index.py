#!/usr/bin/env python3
"""Build an auditable run index from artifacts, even after launcher failure."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ocrap.external_baselines.provenance import MAIN_TABLE_BY_REGIME

EXPECTED = {regime: list(methods) for regime, methods in MAIN_TABLE_BY_REGIME.items()}


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def method_result(regime_dir: Path, method: str) -> dict[str, Any]:
    result_path = regime_dir / f"closed_loop_{method}.json"
    progress_path = result_path.with_suffix(result_path.suffix + ".progress.json")
    journal_path = result_path.with_suffix(result_path.suffix + ".scenes.jsonl")
    result = read_json(result_path)
    progress = read_json(progress_path)
    num_scenes = result.get("num_scenes") if result else None
    target_count = result.get("bucket_target_count") if result else None
    progress_complete = bool(progress and progress.get("status") == "complete")
    journal_exists = journal_path.is_file()
    complete = bool(result and progress_complete and journal_exists)
    if complete and target_count not in (None, 0):
        complete = int(num_scenes or 0) == int(target_count)
    return {
        "method": method,
        "result": str(result_path),
        "result_exists": result_path.is_file(),
        "progress": str(progress_path),
        "progress_status": progress.get("status") if progress else None,
        "scene_journal": str(journal_path),
        "scene_journal_exists": journal_exists,
        "num_scenes": num_scenes,
        "bucket_target_count": target_count,
        "run_fingerprint": result.get("run_fingerprint") if result else None,
        "complete": complete,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--closed-loop-enabled", default="true")
    ap.add_argument("--oracle-enabled", default="false")
    ap.add_argument("--launcher-exit-code", type=int, default=0)
    args = ap.parse_args()
    closed_loop = str(args.closed_loop_enabled).lower() in {"1", "true", "yes", "on"}
    oracle = str(args.oracle_enabled).lower() in {"1", "true", "yes", "on"}
    regimes: dict[str, Any] = {}
    failed: list[str] = []
    for regime, base_expected in EXPECTED.items():
        phase = read_json(args.root / f"{regime}.phase.json") or {"status": "not_started", "exit_code": None}
        expected = list(base_expected)
        if regime == "near" and oracle:
            expected.insert(0, "oracle_recovery_filter")
        methods = [method_result(args.root / regime, method) for method in expected] if closed_loop else []
        artifacts_complete = bool(methods) and all(row["complete"] for row in methods) if closed_loop else False
        if closed_loop:
            # Completed result/progress/journal triples are authoritative. This
            # allows a stale phase="failed" file to be recovered after the
            # launcher died while writing its final index.
            regime_complete = artifacts_complete
        else:
            regime_complete = phase.get("status") in {"complete", "skipped"}
        if not regime_complete:
            failed.append(regime)
        phase_status = str(phase.get("status") or "not_started")
        effective = "complete_from_artifacts" if regime_complete and phase_status != "complete" else phase_status
        regimes[regime] = {
            "phase": phase,
            "phase_effective_status": effective,
            "expected_closed_loop_methods": expected if closed_loop else [],
            "closed_loop_methods": methods,
            "complete": regime_complete,
            "preflight": str(args.root / regime / "closed_loop_dataset_support.json"),
            "launcher_log": str(args.root / f"{regime}.launcher.log"),
        }
    complete = not failed and int(args.launcher_exit_code) == 0
    doc = {
        "event": "all_regime_external_baselines_v50_4",
        "schema_version": 4,
        "root": str(args.root),
        "launcher_exit_code": int(args.launcher_exit_code),
        "closed_loop_enabled": closed_loop,
        "status": "complete" if complete else "failed_or_incomplete",
        "complete": complete,
        "failed_or_incomplete_regimes": failed,
        "regimes": regimes,
    }
    args.root.mkdir(parents=True, exist_ok=True)
    out = args.root / "EXTERNAL_BASELINE_RUN_INDEX.json"
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": doc["event"], "output": str(out), "complete": complete, "failed": failed}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
