#!/usr/bin/env python3
"""Repair the exact v48.36.2 attempt-ID terminal-state false RC=30.

The v48.36.2 certificate launcher accidentally used V4835_ATTEMPT_ID while the
v48.36 controller exported V4836_ATTEMPT_ID.  Calibration and Natural-gate
artifacts were valid, but their status markers were written as
``legacy-untracked``.  The authoritative resolver correctly rejected that marker
as belonging to another attempt and the controller normalized its audit RC=4 to
pipeline RC=30.

This tool changes status/provenance metadata only.  It accepts only that exact
signature, verifies checkpoint bytes and certificate artifacts, preserves every
pre-repair file, reconstructs one attempt-consistent RC=20 state, and then reruns
both the certificate-status and authoritative-state contracts.  Unknown states
fail closed and are rolled back byte-for-byte.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

BASE_VERSION = "v48.36-OCAF"
IMPLEMENTATION_VERSION = "v48.36.4-IDEMPOTENT-TERMINAL-STATE-HOTFIX"
EVENT = "v48_36_3_terminal_state_repair"
VARIANTS = ("balanced", "precision")
LEGACY_ATTEMPT = "legacy-untracked"


def _json(path: Path) -> dict[str, Any]:
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _same_path(left: Any, right: Any) -> bool:
    try:
        return Path(str(left)).resolve() == Path(str(right)).resolve()
    except Exception:
        return False


def _parse_json_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        raise ValueError(f"empty JSON log: {path}")
    return json.loads(text.splitlines()[-1])


def _copy_backup(root: Path, backup: Path, paths: list[Path]) -> dict[str, bool]:
    existed: dict[str, bool] = {}
    for path in paths:
        rel = path.relative_to(root)
        existed[str(rel)] = path.exists()
        if path.is_file() or path.is_symlink():
            target = backup / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target, follow_symlinks=False)
    _atomic_json(backup / "ORIGINAL_FILE_PRESENCE.json", existed)
    return existed


def _rollback(root: Path, backup: Path, paths: list[Path], existed: Mapping[str, bool]) -> None:
    for path in paths:
        rel = path.relative_to(root)
        original = backup / rel
        if existed.get(str(rel), False):
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, path, follow_symlinks=False)
        else:
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def _run(command: list[str], cwd: Path) -> None:
    completed = subprocess.run(command, cwd=cwd, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"command failed rc={completed.returncode}: {' '.join(command)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = args.run
    repo = args.repo.resolve()
    output = args.output or root / "V48_36_3_TERMINAL_STATE_REPAIR.json"
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    errors: list[str] = []

    def record(name: str, condition: bool, detail: Any = None) -> None:
        checks[name] = bool(condition)
        if detail is not None:
            details[name] = detail
        if not condition:
            errors.append(name)

    try:
        failed = _json(root / "PIPELINE_FAILED.json")
        complete = _json(root / "V48_36_COMPLETE.json")
        current_state = _json(root / "AUTHORITATIVE_RUN_STATUS.json")
        selection = _json(root / "dedicated_recalibration_status.json")
        gate_spec = _json(root / "GATE_SPEC.json")
        stage_repair = _json(root / "V48_36_2_STAGE_TRANSFER_REPAIR.json")
        resume = _json(root / "V48_36_RESUME_CONTRACT.json")
        failed_rc20_audit = _parse_json_log(root / "logs" / "authoritative_run_state.log")
    except Exception as exc:
        doc = {
            "event": EVENT,
            "version": BASE_VERSION,
            "implementation_version": IMPLEMENTATION_VERSION,
            "created_unix": time.time(),
            "run": str(root),
            "valid": False,
            "errors": [f"status_read_error: {type(exc).__name__}: {exc}"],
            "algorithm_changed": False,
            "retraining_performed": False,
            "recalibration_performed": False,
            "test_roots_read": False,
        }
        _atomic_json(output, doc)
        print(json.dumps(doc, ensure_ascii=False))
        return 4

    attempt_id = str(complete.get("attempt_id") or "")
    record("terminal_failure_event", failed.get("event") == "v48_36_pipeline_failed")
    record("terminal_failure_stage_exact", failed.get("stage") == "terminal_state_contract")
    record("terminal_failure_raw_rc_exact", failed.get("raw_exit_code") == 4)
    record("terminal_failure_normalized_rc", failed.get("pipeline_exit_code") == 30)
    record("failure_and_completion_attempt_match", failed.get("attempt_id") == attempt_id and bool(attempt_id), attempt_id)
    record("attempt_is_v48_36", attempt_id.startswith("v4836-") and attempt_id != LEGACY_ATTEMPT, attempt_id)
    record(
        "certificate_and_gate_were_executed",
        failed.get("certificate_executed") is True
        and failed.get("gate_evaluated") is True
        and complete.get("certificate_executed") is True
        and complete.get("gate_evaluated") is True,
    )
    record(
        "completion_is_terminal_contract_rc30",
        complete.get("event") == "v48_36_ocaf_controller_complete"
        and complete.get("pipeline_exit_code") == 30
        and complete.get("failure_stage") == "terminal_state_contract"
        and complete.get("raw_certificate_exit_code") == 4
        and complete.get("pipeline_valid") is False,
    )
    adaptation = failed.get("adaptation_exit_codes")
    adaptation = adaptation if isinstance(adaptation, Mapping) else {}
    record("adaptation_succeeded", adaptation.get("balanced") == 0 and adaptation.get("precision") == 0, dict(adaptation))
    record("current_rc30_state_valid", current_state.get("valid") is True and current_state.get("authoritative_exit_code") == 30)
    record("stage_transfer_repair_valid", stage_repair.get("valid") is True and stage_repair.get("retraining_performed") is False)
    record("resume_contract_valid", resume.get("valid") is True and resume.get("failure_mode") == "repaired_stage_transfer")
    record(
        "selection_is_exact_legacy_attempt",
        selection.get("event") == "v48_36_certificate_candidate_selection"
        and selection.get("attempt_id") == LEGACY_ATTEMPT
        and selection.get("certificate_executed") is True
        and selection.get("gate_evaluated") is True
        and selection.get("valid_candidates") == [],
        {"attempt_id": selection.get("attempt_id"), "valid_candidates": selection.get("valid_candidates")},
    )
    codes = selection.get("controller_exit_codes")
    codes = codes if isinstance(codes, Mapping) else {}
    record("both_variants_natural_gate_failed", codes.get("balanced") == 20 and codes.get("precision") == 20, dict(codes))
    record("gate_spec_has_legacy_attempt_only", gate_spec.get("attempt_id") == LEGACY_ATTEMPT and bool(gate_spec.get("protocol_sha256")))
    record("no_next_commands", not (root / "NEXT_COMMANDS.txt").exists())
    record("no_calibration_failed_marker", not (root / "CALIBRATION_FAILED.json").exists())

    audit_checks = failed_rc20_audit.get("checks")
    audit_checks = audit_checks if isinstance(audit_checks, Mapping) else {}
    contradictions = failed_rc20_audit.get("active_contradictions")
    record(
        "failed_rc20_audit_signature_exact",
        failed_rc20_audit.get("authoritative_exit_code") == 20
        and failed_rc20_audit.get("pipeline_valid") is True
        and failed_rc20_audit.get("attempt_id") == attempt_id
        and contradictions == ["gate_failed belongs to an older attempt but is required for RC=20"]
        and all(
            bool(value)
            for key, value in audit_checks.items()
            if key not in {"no_active_status_contradictions", "gate_failure_marker_present"}
        )
        and audit_checks.get("no_active_status_contradictions") is False
        and audit_checks.get("gate_failure_marker_present") is False,
        {"active_contradictions": contradictions, "checks": dict(audit_checks)},
    )

    archived_gate_candidates: list[Path] = []
    for path in sorted((root / "status_history").glob("**/GATE_FAILED.json")):
        try:
            doc = _json(path)
        except Exception:
            continue
        if (
            doc.get("event") == "v48_36_certificate_candidate_selection"
            and doc.get("attempt_id") == LEGACY_ATTEMPT
            and doc.get("controller_exit_codes") == {"balanced": 20, "precision": 20}
            and doc.get("valid_candidates") == []
        ):
            archived_gate_candidates.append(path)
    record("single_archived_legacy_gate_marker", len(archived_gate_candidates) == 1, [str(p) for p in archived_gate_candidates])
    archived_gate = archived_gate_candidates[0] if len(archived_gate_candidates) == 1 else None
    if archived_gate is not None:
        try:
            archived_gate_doc = _json(archived_gate)
        except Exception as exc:
            record("archived_gate_readable", False, f"{type(exc).__name__}: {exc}")
        else:
            record("archived_gate_readable", True)
            record("archived_gate_matches_selection_exactly", archived_gate_doc == selection)

    controller_variants = complete.get("variants")
    controller_variants = controller_variants if isinstance(controller_variants, Mapping) else {}
    repair_variants = stage_repair.get("variants")
    repair_variants = repair_variants if isinstance(repair_variants, Mapping) else {}
    variant_details: dict[str, Any] = {}
    for variant in VARIANTS:
        variant_checks: dict[str, bool] = {}
        controller = controller_variants.get(variant)
        controller = controller if isinstance(controller, Mapping) else {}
        repaired = repair_variants.get(variant)
        repaired = repaired if isinstance(repaired, Mapping) else {}
        ckpt = Path(str(controller.get("checkpoint") or ""))
        expected_sha = str(repaired.get("final_sha256") or "")
        variant_checks["checkpoint_exists"] = ckpt.is_file()
        variant_checks["controller_hash_matches_stage_repair"] = controller.get("sha256") == expected_sha and bool(expected_sha)
        variant_checks["checkpoint_bytes_match"] = ckpt.is_file() and _sha256(ckpt) == expected_sha
        calibration = root / "candidates" / variant / "calibration"
        cert_complete = calibration / "CERTIFICATE_CALIBRATION_COMPLETE.json"
        safe_status = calibration / "SAFE_REGIME_STATUS.json"
        try:
            cert_doc = _json(cert_complete)
            safe_doc = _json(safe_status)
        except Exception as exc:
            variant_checks["certificate_status_readable"] = False
            variant_details[variant] = {"checks": variant_checks, "error": f"{type(exc).__name__}: {exc}"}
            for key, value in variant_checks.items():
                record(f"{variant}_{key}", value)
            continue
        variant_checks["certificate_status_readable"] = True
        variant_checks["legacy_attempt_exact"] = cert_doc.get("attempt_id") == LEGACY_ATTEMPT and safe_doc.get("attempt_id") == LEGACY_ATTEMPT
        variant_checks["certificate_natural_failure_codes"] = cert_doc.get("near_exit_code") == 3 and cert_doc.get("contact_exit_code") == 3
        variant_checks["certificate_data_valid"] = cert_doc.get("certificate_data_valid") is True and cert_doc.get("gate_evaluated") is True
        variant_checks["test_roots_sealed"] = cert_doc.get("test_roots_read") is False and safe_doc.get("test_roots_read") is False
        risk_validity: dict[str, Any] = {}
        for bucket in ("near", "contact"):
            risk = _json(calibration / f"direct_value_risk_{bucket}_v48.json")
            risk_validity[bucket] = risk.get("valid_for_deployment")
            variant_checks[f"{bucket}_certificate_is_natural_rejection"] = (
                risk.get("certificate_data_valid") is True
                and risk.get("gate_evaluated") is True
                and risk.get("valid_for_deployment") is False
            )
        variant_details[variant] = {
            "checkpoint": str(ckpt),
            "expected_sha256": expected_sha,
            "risk_valid_for_deployment": risk_validity,
            "checks": variant_checks,
        }
        for key, value in variant_checks.items():
            record(f"{variant}_{key}", value)

    record(
        "test_roots_sealed_globally",
        all(
            doc.get("test_roots_read") is False
            for doc in (failed, complete, current_state, selection, stage_repair, resume, failed_rc20_audit)
        ),
    )
    record("source_run_consistent", _same_path(complete.get("source_run"), stage_repair.get("source_run")))
    record("protocol_root_consistent", _same_path(complete.get("protocol_root"), stage_repair.get("protocol_root")))

    if errors:
        doc = {
            "event": EVENT,
            "version": BASE_VERSION,
            "implementation_version": IMPLEMENTATION_VERSION,
            "created_unix": time.time(),
            "run": str(root),
            "valid": False,
            "checks": checks,
            "details": details,
            "variants": variant_details,
            "errors": errors,
            "algorithm_changed": False,
            "retraining_performed": False,
            "recalibration_performed": False,
            "test_roots_read": False,
        }
        _atomic_json(output, doc)
        print(json.dumps(doc, ensure_ascii=False))
        return 4

    assert archived_gate is not None
    stamp = time.time_ns()
    backup = root / "repair_history" / f"v48.36.3-terminal-state-{stamp}"
    touched = [
        root / "ATTEMPT_STARTED.json",
        root / "GATE_SPEC.json",
        root / "dedicated_recalibration_status.json",
        root / "NEXT_COMMANDS_STATUS.json",
        root / "NEXT_COMMANDS_BLOCKED.json",
        root / "GATE_FAILED.json",
        root / "CALIBRATION_FAILED.json",
        root / "PIPELINE_FAILED.json",
        root / "V48_36_COMPLETE.json",
        root / "AUTHORITATIVE_RUN_STATUS.json",
        root / "V48_36_CERTIFICATE_STATUS_CONTRACT.json",
    ]
    for variant in VARIANTS:
        touched.extend(
            [
                root / "candidates" / variant / "calibration" / "CERTIFICATE_CALIBRATION_COMPLETE.json",
                root / "candidates" / variant / "calibration" / "SAFE_REGIME_STATUS.json",
            ]
        )
    existed = _copy_backup(root, backup, touched)
    shutil.copy2(archived_gate, backup / "SOURCE_ARCHIVED_GATE_FAILED.json")
    _atomic_json(
        backup / "REPAIR_INPUT_AUDIT.json",
        {
            "event": EVENT,
            "created_unix": time.time(),
            "attempt_id": attempt_id,
            "checks": checks,
            "details": details,
            "variants": variant_details,
            "source_archived_gate": str(archived_gate),
        },
    )

    try:
        attempt_doc = {
            "event": "v48_36_attempt_started",
            "version": BASE_VERSION,
            "implementation_version": IMPLEMENTATION_VERSION,
            "created_unix": time.time(),
            "attempt_id": attempt_id,
            "source_run": complete.get("source_run"),
            "protocol_root": complete.get("protocol_root"),
            "resume_after_adaptation": True,
            "reconstructed_from_exact_terminal_state_failure": True,
            "repair_history": str(backup),
            "test_roots_read": False,
        }
        _atomic_json(root / "ATTEMPT_STARTED.json", attempt_doc)

        def corrected(doc: Mapping[str, Any]) -> dict[str, Any]:
            result = dict(doc)
            result["attempt_id"] = attempt_id
            result["implementation_version"] = IMPLEMENTATION_VERSION
            result["status_metadata_repaired"] = True
            result["status_metadata_repair_event"] = EVENT
            result["status_metadata_repair_history"] = str(backup)
            result["algorithm_changed"] = False
            result["test_roots_read"] = False
            return result

        _atomic_json(root / "GATE_SPEC.json", corrected(gate_spec))
        _atomic_json(root / "dedicated_recalibration_status.json", corrected(selection))
        gate_failed_doc = corrected(_json(archived_gate))
        _atomic_json(root / "GATE_FAILED.json", gate_failed_doc)
        blocked_doc = {
            "event": "v48_36_next_commands_blocked",
            "version": BASE_VERSION,
            "implementation_version": IMPLEMENTATION_VERSION,
            "created_unix": time.time(),
            "attempt_id": attempt_id,
            "reason": "natural_gate_failed",
            "exit_code": 20,
            "certificate_executed": True,
            "gate_evaluated": True,
            "test_roots_read": False,
            "controller_exit_codes": {"balanced": 20, "precision": 20},
            "status_metadata_repaired": True,
            "status_metadata_repair_history": str(backup),
        }
        _atomic_json(root / "NEXT_COMMANDS_BLOCKED.json", blocked_doc)
        _atomic_json(root / "NEXT_COMMANDS_STATUS.json", {**blocked_doc, "generated": False})
        for variant in VARIANTS:
            for name in ("CERTIFICATE_CALIBRATION_COMPLETE.json", "SAFE_REGIME_STATUS.json"):
                path = root / "candidates" / variant / "calibration" / name
                _atomic_json(path, corrected(_json(path)))

        for path in (root / "PIPELINE_FAILED.json", root / "CALIBRATION_FAILED.json", root / "NEXT_COMMANDS.txt"):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

        repaired_complete = {
            "event": "v48_36_ocaf_controller_complete",
            "version": BASE_VERSION,
            "implementation_version": IMPLEMENTATION_VERSION,
            "created_unix": time.time(),
            "attempt_id": attempt_id,
            "source_run": complete.get("source_run"),
            "protocol_root": complete.get("protocol_root"),
            "variants": complete.get("variants"),
            "raw_certificate_exit_code": 20,
            "certificate_exit_code": 20,
            "pipeline_exit_code": 20,
            "certificate_executed": True,
            "gate_evaluated": True,
            "gate_passed": False,
            "next_commands_generated": False,
            "pipeline_valid": True,
            "adaptation_reused_without_retraining": True,
            "resume_contract": str(root / "V48_36_RESUME_CONTRACT.json"),
            "terminal_state_repaired_without_recalibration": True,
            "terminal_state_repair_history": str(backup),
            "test_roots_read": False,
        }
        _atomic_json(root / "V48_36_COMPLETE.json", repaired_complete)

        _run(
            [
                sys.executable,
                str(repo / "tools" / "check_v48_36_certificate_status_contract.py"),
                "--run",
                str(root),
                "--expected-attempt-id",
                attempt_id,
                "--output",
                str(root / "V48_36_CERTIFICATE_STATUS_CONTRACT.json"),
            ],
            repo,
        )
        _run(
            [
                sys.executable,
                str(repo / "tools" / "resolve_v48_36_authoritative_result.py"),
                "--run",
                str(root),
                "--output",
                str(root / "AUTHORITATIVE_RUN_STATUS.json"),
                "--expect-exit-code",
                "20",
                "--expect-attempt-id",
                attempt_id,
            ],
            repo,
        )
        authoritative = _json(root / "AUTHORITATIVE_RUN_STATUS.json")
        if not (authoritative.get("valid") is True and authoritative.get("authoritative_exit_code") == 20):
            raise RuntimeError("authoritative RC=20 validation did not pass")
    except Exception as exc:
        _rollback(root, backup, touched, existed)
        doc = {
            "event": EVENT,
            "version": BASE_VERSION,
            "implementation_version": IMPLEMENTATION_VERSION,
            "created_unix": time.time(),
            "run": str(root),
            "valid": False,
            "rolled_back": True,
            "repair_history": str(backup),
            "checks": checks,
            "details": details,
            "variants": variant_details,
            "errors": [f"repair_execution_error: {type(exc).__name__}: {exc}"],
            "algorithm_changed": False,
            "retraining_performed": False,
            "recalibration_performed": False,
            "test_roots_read": False,
        }
        _atomic_json(output, doc)
        print(json.dumps(doc, ensure_ascii=False))
        return 4

    doc = {
        "event": EVENT,
        "version": BASE_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "created_unix": time.time(),
        "run": str(root),
        "attempt_id": attempt_id,
        "valid": True,
        "authoritative_exit_code": 20,
        "pipeline_valid": True,
        "natural_gate_failed": True,
        "repair_history": str(backup),
        "checks": checks,
        "details": details,
        "variants": variant_details,
        "algorithm_changed": False,
        "retraining_performed": False,
        "recalibration_performed": False,
        "certificate_scores_changed": False,
        "gate_decision_changed": False,
        "status_metadata_changed": True,
        "authorized_next_action": "analyze RC=20 gate outputs; do not execute RC=0-only Safe/stress commands",
        "test_roots_read": False,
    }
    _atomic_json(output, doc)
    print(json.dumps(doc, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
