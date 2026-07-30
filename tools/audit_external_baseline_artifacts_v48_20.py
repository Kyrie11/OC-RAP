#!/usr/bin/env python3
"""Reject stale or partially-written closed-loop baseline summaries.

A JSON summary is paper-usable only when its progress journal is complete and
its declared scene count agrees with both the progress count and scene journal.
Offline eval JSONs do not have progress journals and are reported separately.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _jsonl_count(path: Path) -> int | None:
    if not path.is_file():
        return None
    count = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def audit_root(root: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    for progress_path in sorted(root.rglob("*.json.progress.json")):
        summary_path = Path(str(progress_path)[: -len(".progress.json")])
        scenes_path = Path(str(summary_path) + ".scenes.jsonl")
        progress = _read_json(progress_path)
        completed = int(progress.get("completed_rollouts", 0) or 0)
        requested = int(progress.get("requested_rollouts", 0) or 0)
        status = str(progress.get("status", ""))
        summary_exists = summary_path.is_file()
        summary_num = None
        if summary_exists:
            summary_num = int(_read_json(summary_path).get("num_scenes", 0) or 0)
        journal_num = _jsonl_count(scenes_path)
        usable = (
            status == "complete"
            and requested > 0
            and completed == requested
            and summary_exists
            and summary_num == completed
            and (journal_num is None or journal_num == completed)
        )
        record = {
            "summary": str(summary_path),
            "progress": str(progress_path),
            "scene_journal": str(scenes_path) if scenes_path.exists() else None,
            "status": status,
            "completed_rollouts": completed,
            "requested_rollouts": requested,
            "summary_num_scenes": summary_num,
            "scene_journal_rows": journal_num,
            "paper_usable": usable,
        }
        records.append(record)
        if not usable:
            failures.append(
                f"stale/incomplete closed-loop artifact: {summary_path}; "
                f"status={status} progress={completed}/{requested} summary={summary_num} journal={journal_num}"
            )
    offline = sorted(str(p) for p in root.rglob("eval_*.json"))
    return {
        "root": str(root.resolve()),
        "closed_loop_records": records,
        "offline_eval_jsons": offline,
        "valid": not failures,
        "failures": failures,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", required=True, help="Baseline result root; repeatable")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--allow-incomplete", action="store_true", help="Write audit but return zero")
    args = ap.parse_args()
    roots = [Path(value) for value in args.root]
    missing = [str(p) for p in roots if not p.is_dir()]
    if missing:
        raise SystemExit("missing baseline roots: " + ",".join(missing))
    audits = [audit_root(p) for p in roots]
    failures = [item for audit in audits for item in audit["failures"]]
    doc = {
        "event": "v48_20_external_baseline_artifact_audit",
        "valid": not failures,
        "audits": audits,
        "failures": failures,
        "rule": "closed-loop summary is usable only if progress=complete and progress/summary/journal counts agree",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(doc, ensure_ascii=False, indent=2))
    return 0 if (not failures or args.allow_incomplete) else 30


if __name__ == "__main__":
    raise SystemExit(main())
