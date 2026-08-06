#!/usr/bin/env python3
"""Fail-closed re-entry contract for the v48.36 controller.

The controller historically converted a rejected resume into a new active RC=30
and moved a previously valid terminal state into status_history.  This contract
runs before ATTEMPT_STARTED.json is written.  It either:

* returns an already valid active RC=0/20 result without mutating it;
* identifies an exact RC=20 resume-refusal clobber that can be restored safely;
* allows a genuinely incomplete/failed run to proceed; or
* refuses contradictory/corrupt terminal metadata.

It never reads test roots and never inspects or changes model scores, thresholds,
checkpoints, certificate statistics, or gate decisions.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

EVENT = "v48_36_reentry_contract"
VERSION = "v48.36-OCAF"
IMPLEMENTATION_VERSION = "v48.36.4-IDEMPOTENT-TERMINAL-STATE-HOTFIX"
TERMINAL_FILES = (
    "V48_36_COMPLETE.json",
    "AUTHORITATIVE_RUN_STATUS.json",
    "NEXT_COMMANDS_STATUS.json",
    "NEXT_COMMANDS_BLOCKED.json",
    "GATE_FAILED.json",
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _read_optional(path: Path) -> tuple[dict[str, Any], str | None]:
    if not path.is_file():
        return {}, None
    try:
        return _read(path), None
    except Exception as exc:  # report remains machine-readable
        return {}, f"{type(exc).__name__}: {exc}"


def _atomic(path: Path, doc: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(doc, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _integer(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _terminal_bundle(directory: Path) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    complete, err = _read_optional(directory / "V48_36_COMPLETE.json")
    if err:
        return None, [f"complete_read_error:{err}"]
    state, err = _read_optional(directory / "AUTHORITATIVE_RUN_STATUS.json")
    if err:
        return None, [f"state_read_error:{err}"]
    if not complete and not state:
        return None, []
    if not complete or not state:
        return None, ["terminal_bundle_incomplete"]

    rc = _integer(complete.get("pipeline_exit_code"))
    attempt = str(complete.get("attempt_id") or "")
    # RC=30 is a failed/incomplete run, not an already-valid terminal bundle.
    # Leave it to the resume-refusal recognizer or the normal controller path.
    if rc == 30:
        return None, []
    if complete.get("event") != "v48_36_ocaf_controller_complete":
        errors.append("completion_event")
    if rc not in (0, 20):
        errors.append("completion_exit_code")
    if complete.get("pipeline_valid") is not True:
        errors.append("completion_pipeline_valid")
    if complete.get("certificate_executed") is not True or complete.get("gate_evaluated") is not True:
        errors.append("completion_certificate_gate")
    if not attempt or attempt == "legacy-untracked":
        errors.append("completion_attempt")
    if state.get("valid") is not True:
        errors.append("authoritative_valid")
    if _integer(state.get("authoritative_exit_code")) != rc:
        errors.append("authoritative_exit_code")
    if state.get("pipeline_valid") is not True:
        errors.append("authoritative_pipeline_valid")
    if state.get("attempt_id") != attempt:
        errors.append("authoritative_attempt")
    if (directory / "PIPELINE_FAILED.json").exists():
        errors.append("pipeline_failed_present")
    if (directory / "CALIBRATION_FAILED.json").exists():
        errors.append("calibration_failed_present")

    next_status, err = _read_optional(directory / "NEXT_COMMANDS_STATUS.json")
    if err:
        errors.append(f"next_status_read_error:{err}")
    if rc == 20:
        gate, gate_err = _read_optional(directory / "GATE_FAILED.json")
        blocked, blocked_err = _read_optional(directory / "NEXT_COMMANDS_BLOCKED.json")
        if gate_err:
            errors.append(f"gate_failed_read_error:{gate_err}")
        if blocked_err:
            errors.append(f"blocked_read_error:{blocked_err}")
        if not gate or gate.get("attempt_id") != attempt:
            errors.append("gate_failed_attempt")
        if not blocked or blocked.get("attempt_id") != attempt:
            errors.append("blocked_attempt")
        if next_status.get("attempt_id") != attempt:
            errors.append("next_status_attempt")
        if next_status.get("reason") != "natural_gate_failed" or _integer(next_status.get("exit_code")) != 20:
            errors.append("next_status_semantics")
        if (directory / "NEXT_COMMANDS.txt").exists():
            errors.append("rc20_next_commands_present")
    elif rc == 0:
        if not (directory / "NEXT_COMMANDS.txt").is_file():
            errors.append("rc0_next_commands_missing")
        if (directory / "GATE_FAILED.json").exists() or (directory / "NEXT_COMMANDS_BLOCKED.json").exists():
            errors.append("rc0_failure_marker_present")
        if next_status.get("attempt_id") != attempt or next_status.get("generated") is not True:
            errors.append("rc0_next_status")

    if errors:
        return None, errors
    return {
        "directory": str(directory),
        "attempt_id": attempt,
        "exit_code": rc,
        "complete_created_unix": complete.get("created_unix"),
        "state_created_unix": state.get("created_unix"),
    }, []


def _active_resume_refusal(root: Path) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    required = {}
    for name in ("PIPELINE_FAILED.json", "V48_36_COMPLETE.json", "AUTHORITATIVE_RUN_STATUS.json", "ATTEMPT_STARTED.json"):
        doc, err = _read_optional(root / name)
        if err:
            errors.append(f"{name}:{err}")
        required[name] = doc
    if errors:
        return None, errors
    failed = required["PIPELINE_FAILED.json"]
    complete = required["V48_36_COMPLETE.json"]
    state = required["AUTHORITATIVE_RUN_STATUS.json"]
    attempt = required["ATTEMPT_STARTED.json"]
    if not all(required.values()):
        return None, []
    active_attempt = str(complete.get("attempt_id") or "")
    exact = (
        failed.get("event") == "v48_36_pipeline_failed"
        and failed.get("stage") == "resume_authorization"
        and _integer(failed.get("raw_exit_code")) == 4
        and _integer(failed.get("pipeline_exit_code")) == 30
        and failed.get("certificate_executed") is False
        and failed.get("gate_evaluated") is False
        and failed.get("test_roots_read") is False
        and complete.get("event") == "v48_36_ocaf_controller_complete"
        and complete.get("failure_stage") == "resume_authorization"
        and _integer(complete.get("pipeline_exit_code")) == 30
        and complete.get("pipeline_valid") is False
        and complete.get("certificate_executed") is False
        and complete.get("gate_evaluated") is False
        and state.get("valid") is True
        and _integer(state.get("authoritative_exit_code")) == 30
        and state.get("pipeline_valid") is False
        and attempt.get("resume_after_adaptation") is True
        and failed.get("attempt_id") == active_attempt
        and state.get("attempt_id") == active_attempt
        and attempt.get("attempt_id") == active_attempt
        and bool(active_attempt)
        and active_attempt != "legacy-untracked"
    )
    if not exact:
        return None, []
    return {"attempt_id": active_attempt}, []


def _archive_matches_current_certificate(root: Path, bundle: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    attempt = str(bundle["attempt_id"])
    archive = Path(str(bundle["directory"]))
    details: dict[str, Any] = {"attempt_id": attempt, "archive": str(archive)}
    # Historical controller versions did not archive NEXT_COMMANDS.txt.  Therefore
    # an archived RC=0 bundle cannot be reconstructed byte-completely and must not
    # be advertised as recoverable.  Active RC=0 remains idempotently returnable.
    if int(bundle["exit_code"]) != 20:
        details["reason"] = "archived_rc0_not_recoverable_without_archived_next_commands"
        return False, details
    try:
        archived_gate = _read(archive / "GATE_FAILED.json") if int(bundle["exit_code"]) == 20 else {}
        selection = _read(root / "dedicated_recalibration_status.json")
        gate_spec = _read(root / "GATE_SPEC.json")
        status_contract = _read(root / "V48_36_CERTIFICATE_STATUS_CONTRACT.json")
    except Exception as exc:
        details["read_error"] = f"{type(exc).__name__}: {exc}"
        return False, details
    checks = {
        "selection_attempt": selection.get("attempt_id") == attempt,
        "gate_spec_attempt": gate_spec.get("attempt_id") == attempt,
        "status_contract_valid": status_contract.get("valid") is True,
        "status_contract_attempt": status_contract.get("expected_attempt_id") == attempt,
        "status_contract_exit_code": _integer(status_contract.get("expected_exit_code")) == int(bundle["exit_code"]),
        "selection_matches_archived_gate": int(bundle["exit_code"]) != 20 or selection == archived_gate,
        "selection_certificate_executed": selection.get("certificate_executed") is True,
        "selection_gate_evaluated": selection.get("gate_evaluated") is True,
        "selection_test_roots_sealed": selection.get("test_roots_read") is False,
    }
    requested = selection.get("requested_variants")
    if not isinstance(requested, list) or not requested:
        checks["requested_variants"] = False
        requested = []
    else:
        checks["requested_variants"] = True
    candidate_attempts: dict[str, Any] = {}
    for variant in requested:
        candidate_attempts[variant] = {}
        for rel, label in (
            ("calibration/CERTIFICATE_CALIBRATION_COMPLETE.json", "certificate"),
            ("calibration/SAFE_REGIME_STATUS.json", "safe"),
        ):
            try:
                doc = _read(root / "candidates" / variant / rel)
            except Exception as exc:
                checks[f"{variant}_{label}_readable"] = False
                candidate_attempts[variant][label] = f"{type(exc).__name__}: {exc}"
                continue
            checks[f"{variant}_{label}_readable"] = True
            checks[f"{variant}_{label}_attempt"] = doc.get("attempt_id") == attempt
            checks[f"{variant}_{label}_test_roots_sealed"] = doc.get("test_roots_read") is False
            candidate_attempts[variant][label] = doc.get("attempt_id")
    details["checks"] = checks
    details["candidate_attempts"] = candidate_attempts
    return all(checks.values()), details


def audit(root: Path, mode: str, allow_completed_overwrite: bool) -> dict[str, Any]:
    active_bundle, active_errors = _terminal_bundle(root)
    if active_bundle is not None:
        action = "proceed" if allow_completed_overwrite else "return_existing_terminal"
        return {
            "event": EVENT,
            "version": VERSION,
            "implementation_version": IMPLEMENTATION_VERSION,
            "created_unix": time.time(),
            "run": str(root),
            "mode": mode,
            "valid": True,
            "action": action,
            "existing_terminal": active_bundle,
            "existing_exit_code": active_bundle["exit_code"],
            "algorithm_changed": False,
            "test_roots_read": False,
        }
    if active_errors:
        return {
            "event": EVENT,
            "version": VERSION,
            "implementation_version": IMPLEMENTATION_VERSION,
            "created_unix": time.time(),
            "run": str(root),
            "mode": mode,
            "valid": False,
            "action": "refuse",
            "errors": active_errors,
            "algorithm_changed": False,
            "test_roots_read": False,
        }

    refusal, refusal_errors = _active_resume_refusal(root)
    if refusal_errors:
        return {
            "event": EVENT,
            "version": VERSION,
            "implementation_version": IMPLEMENTATION_VERSION,
            "created_unix": time.time(),
            "run": str(root),
            "mode": mode,
            "valid": False,
            "action": "refuse",
            "errors": refusal_errors,
            "algorithm_changed": False,
            "test_roots_read": False,
        }
    if refusal is not None:
        candidates: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        for state_path in sorted((root / "status_history").glob("**/AUTHORITATIVE_RUN_STATUS.json")):
            bundle, errors = _terminal_bundle(state_path.parent)
            if bundle is None:
                if errors:
                    rejected.append({"directory": str(state_path.parent), "errors": errors})
                continue
            matches, detail = _archive_matches_current_certificate(root, bundle)
            if matches:
                candidates.append(bundle)
            else:
                rejected.append(detail)
        if candidates:
            candidates.sort(key=lambda item: float(item.get("state_created_unix") or 0.0), reverse=True)
            chosen = candidates[0]
            return {
                "event": EVENT,
                "version": VERSION,
                "implementation_version": IMPLEMENTATION_VERSION,
                "created_unix": time.time(),
                "run": str(root),
                "mode": mode,
                "valid": True,
                "action": "restore_archived_terminal",
                "active_resume_refusal": refusal,
                "archive": chosen,
                "matching_archives": candidates,
                "rejected_archives": rejected,
                "existing_exit_code": chosen["exit_code"],
                "algorithm_changed": False,
                "test_roots_read": False,
            }
        return {
            "event": EVENT,
            "version": VERSION,
            "implementation_version": IMPLEMENTATION_VERSION,
            "created_unix": time.time(),
            "run": str(root),
            "mode": mode,
            "valid": True,
            "action": "refuse_preserve_current",
            "active_resume_refusal": refusal,
            "reason": "no archive passed the exact terminal/certificate consistency contract",
            "rejected_archives": rejected,
            "algorithm_changed": False,
            "test_roots_read": False,
        }

    return {
        "event": EVENT,
        "version": VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "created_unix": time.time(),
        "run": str(root),
        "mode": mode,
        "valid": True,
        "action": "proceed",
        "algorithm_changed": False,
        "test_roots_read": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--mode", choices=("resume", "fresh"), required=True)
    parser.add_argument("--allow-completed-overwrite", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.run / "V48_36_REENTRY_CONTRACT.json"
    report = audit(args.run, args.mode, args.allow_completed_overwrite)
    _atomic(output, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report.get("valid") is True else 4


if __name__ == "__main__":
    raise SystemExit(main())
