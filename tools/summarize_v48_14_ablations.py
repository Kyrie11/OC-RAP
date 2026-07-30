#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}, got {type(value).__name__}")
    return value


def _risk_path(task: Path, variant: str, bucket: str) -> Path:
    dedicated = (
        task
        / "dedicated_candidates"
        / variant
        / "calibration"
        / f"direct_value_risk_{bucket}_v48.json"
    )
    if dedicated.is_file():
        return dedicated
    return (
        task
        / "candidates"
        / variant
        / "calibration"
        / f"direct_value_risk_{bucket}_v48.json"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", default="v48.16-ANCHOR")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    tasks_root = args.root / "tasks"
    for task in sorted(tasks_root.glob("*")):
        if not task.is_dir():
            continue
        group, separator, variant = task.name.rpartition("_")
        if not separator:
            group, variant = task.name, "unknown"
        row: dict[str, Any] = {
            "task": task.name,
            "group": group,
            "variant": variant,
            "complete": (task / "TASK_COMPLETE.json").is_file(),
        }
        for bucket in ("near", "contact"):
            risk_path = _risk_path(task, variant, bucket)
            try:
                document = _load_json(risk_path)
            except Exception as exc:  # report incomplete/corrupt ablations instead of hiding them
                row[bucket] = {"missing": str(exc), "path": str(risk_path)}
                continue
            row[bucket] = {
                "path": str(risk_path),
                "valid": document.get("valid_for_deployment"),
                "candidate_auc": document.get("candidate_positive_auc"),
                "harm_auc": document.get("candidate_harm_auc"),
                "top1_corr": document.get("unconstrained_group_top1_correlation"),
                "proposal_oracle_best_hit": document.get(
                    "proposal_oracle_best_hit_rate_positive_groups",
                    document.get("proposal_oracle_best_hit_rate_positive"),
                ),
                "proposal_any_positive_hit": document.get(
                    "proposal_any_positive_hit_rate_positive_groups",
                    document.get("proposal_any_positive_hit_rate_positive"),
                ),
                "verify": document.get("verify"),
                "warnings": document.get("warnings"),
            }
        rows.append(row)

    output = {
        "version": args.version,
        "tasks": rows,
        "num_tasks": len(rows),
        "complete_tasks": sum(bool(row["complete"]) for row in rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
