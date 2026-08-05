#!/usr/bin/env python3
"""Resolve and validate the authoritative terminal state of a v48.35 run.

Legacy runs used several top-level marker files.  A stale marker copied from an
older attempt (or retained by updating an existing ZIP) must never override a
newer V48_35_COMPLETE.json.  This tool resolves state from content, attempt IDs
and timestamps, validates the RC/NEXT_COMMANDS contract, and can archive stale
markers without modifying model or certificate artifacts.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Mapping


TERMINAL_EVENT = "v48_35_continuous_frontier_controller_complete"
FAILURE_EVENT = "v48_35_pipeline_failed"


def _read_json(path: Path) -> tuple[dict[str, Any], str | None]:
    if not path.is_file():
        return {}, None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - exact parser text is platform-dependent
        return {}, f"{path}: {exc!r}"
    if not isinstance(value, dict):
        return {}, f"{path}: top-level JSON is not an object"
    return value, None


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _integer(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _marker_relation(marker: Mapping[str, Any], complete: Mapping[str, Any]) -> str:
    """Return same_attempt, stale, newer_or_ambiguous, or absent metadata."""
    marker_attempt = marker.get("attempt_id")
    complete_attempt = complete.get("attempt_id")
    if marker_attempt and complete_attempt:
        return "same_attempt" if marker_attempt == complete_attempt else "stale"
    marker_time = _number(marker.get("created_unix"))
    complete_time = _number(complete.get("created_unix"))
    if marker_time is not None and complete_time is not None:
        return "stale" if marker_time < complete_time else "newer_or_ambiguous"
    return "newer_or_ambiguous"


def resolve(run: Path) -> dict[str, Any]:
    paths = {
        "complete": run / "V48_35_COMPLETE.json",
        "pipeline_failed": run / "PIPELINE_FAILED.json",
        "gate_failed": run / "GATE_FAILED.json",
        "calibration_failed": run / "CALIBRATION_FAILED.json",
        "next_status": run / "NEXT_COMMANDS_STATUS.json",
        "next_blocked": run / "NEXT_COMMANDS_BLOCKED.json",
    }
    docs: dict[str, dict[str, Any]] = {}
    read_errors: list[str] = []
    for name, path in paths.items():
        doc, error = _read_json(path)
        docs[name] = doc
        if error:
            read_errors.append(error)

    complete = docs["complete"]
    failed = docs["pipeline_failed"]
    next_status = docs["next_status"]
    rc = _integer(complete.get("pipeline_exit_code")) if complete else None
    complete_is_terminal = complete.get("event") == TERMINAL_EVENT and rc in (0, 20, 30)
    pipeline_valid = complete.get("pipeline_valid") is True
    next_commands_present = (run / "NEXT_COMMANDS.txt").is_file()
    blocked_present = paths["next_blocked"].is_file()
    gate_failed_present = paths["gate_failed"].is_file()
    calibration_failed_present = paths["calibration_failed"].is_file()
    pipeline_failed_present = paths["pipeline_failed"].is_file()

    stale_markers: list[dict[str, Any]] = []
    active_contradictions: list[str] = []
    relations: dict[str, str] = {}
    if complete_is_terminal:
        for name in ("pipeline_failed", "gate_failed", "calibration_failed"):
            if not paths[name].is_file():
                continue
            relation = _marker_relation(docs[name], complete)
            expected_active = ((name == "pipeline_failed" and rc == 30) or (name == "gate_failed" and rc == 20) or (name == "calibration_failed" and rc == 30))
            if expected_active:
                marker_attempt = docs[name].get("attempt_id")
                complete_attempt = complete.get("attempt_id")
                if marker_attempt and complete_attempt and marker_attempt != complete_attempt:
                    relations[name] = "stale"
                    active_contradictions.append(f"{name} belongs to an older attempt but is required for RC={rc}")
                else:
                    # Legacy markers are normally written immediately before the terminal
                    # completion record, so an earlier timestamp is expected here.
                    relations[name] = "same_attempt" if marker_attempt and complete_attempt else "same_attempt_legacy"
            elif relation == "stale":
                relations[name] = relation
                stale_markers.append({"name": paths[name].name, "path": str(paths[name]), "relation": relation})
            else:
                relations[name] = relation
                active_contradictions.append(f"unexpected active {paths[name].name} for terminal RC={rc}")

    checks: dict[str, bool] = {
        "json_readable": not read_errors,
        "terminal_completion_present": complete_is_terminal,
        "no_active_status_contradictions": not active_contradictions,
    }

    if complete_is_terminal and rc == 0:
        checks.update(
            {
                "pipeline_valid": pipeline_valid,
                "certificate_executed": complete.get("certificate_executed") is True,
                "gate_evaluated": complete.get("gate_evaluated") is True,
                "gate_passed": complete.get("gate_passed") is True,
                "next_commands_generated": complete.get("next_commands_generated") is True,
                "next_commands_present": next_commands_present,
                "next_commands_not_blocked": not blocked_present,
                "next_status_generated": next_status.get("generated") is True,
                "no_gate_failure_marker": not gate_failed_present or relations.get("gate_failed") == "stale",
                "no_calibration_failure_marker": not calibration_failed_present or relations.get("calibration_failed") == "stale",
                "no_pipeline_failure_marker": not pipeline_failed_present or relations.get("pipeline_failed") == "stale",
            }
        )
    elif complete_is_terminal and rc == 20:
        checks.update(
            {
                "pipeline_valid": pipeline_valid,
                "certificate_executed": complete.get("certificate_executed") is True,
                "gate_evaluated": complete.get("gate_evaluated") is True,
                "gate_passed_false": complete.get("gate_passed") is False,
                "next_commands_not_generated": complete.get("next_commands_generated") is False,
                "next_commands_absent": not next_commands_present,
                "next_commands_blocked": blocked_present,
                "next_status_is_natural_gate_failure": (
                    next_status.get("reason") == "natural_gate_failed"
                    and _integer(next_status.get("exit_code")) == 20
                    and next_status.get("gate_evaluated") is True
                ),
                "gate_failure_marker_present": gate_failed_present and relations.get("gate_failed") != "stale",
                "no_calibration_failure_marker": not calibration_failed_present or relations.get("calibration_failed") == "stale",
                "no_pipeline_failure_marker": not pipeline_failed_present or relations.get("pipeline_failed") == "stale",
            }
        )
    elif complete_is_terminal and rc == 30:
        checks.update(
            {
                "pipeline_invalid": complete.get("pipeline_valid") is False,
                "pipeline_failure_marker_present": pipeline_failed_present and relations.get("pipeline_failed") != "stale",
                "next_commands_absent": not next_commands_present,
                "next_commands_blocked": blocked_present,
            }
        )
    else:
        checks["terminal_rc_registered"] = False

    valid = all(checks.values())
    return {
        "event": "v48_35_authoritative_run_state",
        "version": "v48.35.2-ENGINEERING-INTEGRITY",
        "created_unix": time.time(),
        "run": str(run),
        "valid": valid,
        "authoritative_source": str(paths["complete"]) if complete_is_terminal else None,
        "authoritative_exit_code": rc,
        "pipeline_valid": pipeline_valid if complete_is_terminal else False,
        "attempt_id": complete.get("attempt_id") if complete else None,
        "checks": checks,
        "read_errors": read_errors,
        "active_contradictions": active_contradictions,
        "stale_markers": stale_markers,
        "marker_relations": relations,
        "artifacts": {
            "next_commands_present": next_commands_present,
            "next_commands_blocked_present": blocked_present,
            "pipeline_failed_present": pipeline_failed_present,
            "gate_failed_present": gate_failed_present,
            "calibration_failed_present": calibration_failed_present,
        },
        "test_roots_read": False,
    }


def _archive_stale(run: Path, report: Mapping[str, Any]) -> list[str]:
    archived: list[str] = []
    attempt = str(report.get("attempt_id") or "legacy")
    stamp = int(_number(report.get("created_unix")) or time.time())
    target_dir = run / "status_history" / f"stale-before-{attempt}-{stamp}"
    for item in report.get("stale_markers") or []:
        source = Path(str(item.get("path", "")))
        if not source.is_file() or source.parent.resolve() != run.resolve():
            continue
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / source.name
        if target.exists():
            target = target_dir / f"{source.stem}.{time.time_ns()}{source.suffix}"
        shutil.move(str(source), str(target))
        archived.append(str(target))
    return archived


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expect-exit-code", type=int, choices=(0, 20, 30))
    parser.add_argument("--expect-attempt-id")
    parser.add_argument("--archive-stale-markers", action="store_true")
    args = parser.parse_args()

    run = args.run
    output = args.output or run / "AUTHORITATIVE_RUN_STATUS.json"
    report = resolve(run)
    if args.expect_exit_code is not None:
        ok = report.get("authoritative_exit_code") == args.expect_exit_code
        report["checks"]["expected_exit_code"] = ok
        report["expected_exit_code"] = args.expect_exit_code
    if args.expect_attempt_id is not None:
        ok = report.get("attempt_id") == args.expect_attempt_id
        report["checks"]["expected_attempt_id"] = ok
        report["expected_attempt_id"] = args.expect_attempt_id
    report["valid"] = all(report["checks"].values())

    if args.archive_stale_markers and report["valid"]:
        archived = _archive_stale(run, report)
        report["archived_stale_markers"] = archived
        if archived:
            refreshed = resolve(run)
            refreshed.update(
                {
                    "archived_stale_markers": archived,
                    "expected_exit_code": report.get("expected_exit_code"),
                    "expected_attempt_id": report.get("expected_attempt_id"),
                }
            )
            if args.expect_exit_code is not None:
                refreshed["checks"]["expected_exit_code"] = refreshed.get("authoritative_exit_code") == args.expect_exit_code
            if args.expect_attempt_id is not None:
                refreshed["checks"]["expected_attempt_id"] = refreshed.get("attempt_id") == args.expect_attempt_id
            refreshed["valid"] = all(refreshed["checks"].values())
            report = refreshed

    _atomic_write_json(output, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["valid"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
