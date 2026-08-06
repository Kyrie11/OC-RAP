#!/usr/bin/env python3
"""Audit v48.36 certificate/gate status files against one controller attempt.

This is an engineering contract only.  It does not inspect or alter scores,
thresholds, model outputs, or gate decisions.  It prevents a natural gate result
from being reclassified as RC=30 because a copied launcher wrote status markers
under a stale/legacy attempt identifier.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

EVENT = "v48_36_certificate_status_contract"
VERSION = "v48.36-OCAF"
IMPLEMENTATION_VERSION = "v48.36.3-TERMINAL-STATE-HOTFIX"
VARIANTS = ("balanced", "precision")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, doc: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(doc, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _attempt_matches(doc: Mapping[str, Any], expected: str) -> bool:
    return doc.get("attempt_id") == expected and expected not in {"", "legacy-untracked"}


def audit(run: Path, expected_attempt_id: str) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    errors: list[str] = []

    def record(name: str, condition: bool, detail: Any = None) -> None:
        checks[name] = bool(condition)
        if detail is not None:
            details[name] = detail
        if not condition:
            errors.append(name)

    docs: dict[str, dict[str, Any]] = {}
    paths = {
        "gate_spec": run / "GATE_SPEC.json",
        "selection": run / "dedicated_recalibration_status.json",
        "next_status": run / "NEXT_COMMANDS_STATUS.json",
    }
    read_errors: dict[str, str] = {}
    for name, path in paths.items():
        try:
            docs[name] = _read_json(path)
        except Exception as exc:  # contract report must remain readable
            docs[name] = {}
            read_errors[name] = f"{type(exc).__name__}: {exc}"
    record("required_status_json_readable", not read_errors, read_errors)
    record("expected_attempt_id_registered", bool(expected_attempt_id) and expected_attempt_id != "legacy-untracked")

    for name, doc in docs.items():
        record(
            f"{name}_attempt_matches",
            _attempt_matches(doc, expected_attempt_id),
            {"path": str(paths[name]), "attempt_id": doc.get("attempt_id")},
        )

    selection = docs["selection"]
    next_status = docs["next_status"]
    controller_codes = selection.get("controller_exit_codes")
    controller_codes = controller_codes if isinstance(controller_codes, Mapping) else {}
    requested_field = selection.get("requested_variants")
    if isinstance(requested_field, list):
        requested = [name for name in VARIANTS if name in requested_field]
    else:
        requested = [name for name in VARIANTS if name in controller_codes]
    record("certificate_executed", selection.get("certificate_executed") is True)
    record("gate_evaluated", selection.get("gate_evaluated") is True)
    record("requested_variants_present", bool(requested), requested)
    record(
        "controller_codes_registered",
        bool(requested) and all(controller_codes.get(name) in (0, 20) for name in requested),
        dict(controller_codes),
    )

    generated = next_status.get("generated") is True
    natural_failure = (
        next_status.get("reason") == "natural_gate_failed"
        and int(next_status.get("exit_code", -1)) == 20
        and next_status.get("gate_evaluated") is True
    )
    record("terminal_next_status_recognized", generated or natural_failure, dict(next_status))
    expected_rc = 0 if generated else 20 if natural_failure else None

    next_commands = run / "NEXT_COMMANDS.txt"
    blocked = run / "NEXT_COMMANDS_BLOCKED.json"
    gate_failed = run / "GATE_FAILED.json"
    calibration_failed = run / "CALIBRATION_FAILED.json"
    if expected_rc == 0:
        record("rc0_next_commands_present", next_commands.is_file())
        record("rc0_blocked_absent", not blocked.exists())
        record("rc0_gate_failed_absent", not gate_failed.exists())
        record("rc0_calibration_failed_absent", not calibration_failed.exists())
    elif expected_rc == 20:
        record("rc20_next_commands_absent", not next_commands.exists())
        record("rc20_blocked_present", blocked.is_file())
        record("rc20_gate_failed_present", gate_failed.is_file())
        record("rc20_calibration_failed_absent", not calibration_failed.exists())
        for name, path in (("blocked", blocked), ("gate_failed", gate_failed)):
            try:
                doc = _read_json(path)
            except Exception as exc:
                record(f"{name}_readable", False, f"{type(exc).__name__}: {exc}")
            else:
                record(f"{name}_readable", True)
                record(
                    f"{name}_attempt_matches",
                    _attempt_matches(doc, expected_attempt_id),
                    {"path": str(path), "attempt_id": doc.get("attempt_id")},
                )

    candidate_attempts: dict[str, dict[str, Any]] = {}
    for variant in requested:
        candidate_attempts[variant] = {}
        for rel, label in (
            ("calibration/CERTIFICATE_CALIBRATION_COMPLETE.json", "certificate_complete"),
            ("calibration/SAFE_REGIME_STATUS.json", "safe_status"),
        ):
            path = run / "candidates" / variant / rel
            try:
                doc = _read_json(path)
            except Exception as exc:
                record(f"{variant}_{label}_readable", False, f"{type(exc).__name__}: {exc}")
                continue
            record(f"{variant}_{label}_readable", True)
            candidate_attempts[variant][label] = doc.get("attempt_id")
            record(
                f"{variant}_{label}_attempt_matches",
                _attempt_matches(doc, expected_attempt_id),
                {"path": str(path), "attempt_id": doc.get("attempt_id")},
            )
            record(f"{variant}_{label}_test_roots_sealed", doc.get("test_roots_read") is False)
        cert = run / "candidates" / variant / "calibration" / "CERTIFICATE_CALIBRATION_COMPLETE.json"
        if cert.is_file():
            doc = _read_json(cert)
            record(f"{variant}_certificate_executed", doc.get("certificate_executed") is True)
            record(f"{variant}_gate_evaluated", doc.get("gate_evaluated") is True)

    record("root_status_test_roots_sealed", selection.get("test_roots_read") is False and next_status.get("test_roots_read") is False)
    valid = all(checks.values())
    return {
        "event": EVENT,
        "version": VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "created_unix": time.time(),
        "run": str(run),
        "expected_attempt_id": expected_attempt_id,
        "expected_exit_code": expected_rc,
        "valid": valid,
        "checks": checks,
        "details": details,
        "candidate_attempt_ids": candidate_attempts,
        "errors": errors,
        "algorithm_changed": False,
        "gate_changed": False,
        "test_roots_read": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--expected-attempt-id", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.run / "V48_36_CERTIFICATE_STATUS_CONTRACT.json"
    report = audit(args.run, args.expected_attempt_id)
    _atomic_json(output, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["valid"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
