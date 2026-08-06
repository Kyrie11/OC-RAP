#!/usr/bin/env python3
"""Restore an already-authoritative v48.36 terminal state after resume clobber.

This is deliberately narrower than the v48.36.3 metadata repair.  It accepts
only an exact RC=30 resume_authorization attempt that did not execute adaptation,
calibration, certificate, gate, or test access, and restores the newest archived
RC=20 bundle that still matches all active certificate/gate attempt metadata.
No checkpoint bytes, scores, thresholds, certificate statistics, or gate decisions
are changed.  Every touched file is backed up and post-restore contracts are rerun;
any failure triggers byte-for-byte rollback.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

EVENT = "v48_36_4_resume_clobber_restore"
VERSION = "v48.36-OCAF"
IMPLEMENTATION_VERSION = "v48.36.4-IDEMPOTENT-TERMINAL-STATE-HOTFIX"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _atomic(path: Path, doc: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(doc, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _copy_backup(root: Path, backup: Path, paths: list[Path]) -> dict[str, bool]:
    presence: dict[str, bool] = {}
    for path in paths:
        rel = path.relative_to(root)
        presence[str(rel)] = path.exists()
        if path.is_file() or path.is_symlink():
            target = backup / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target, follow_symlinks=False)
    _atomic(backup / "ORIGINAL_FILE_PRESENCE.json", presence)
    return presence


def _rollback(root: Path, backup: Path, paths: list[Path], presence: Mapping[str, bool]) -> None:
    for path in paths:
        rel = path.relative_to(root)
        original = backup / rel
        if presence.get(str(rel), False):
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
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.run
    repo = args.repo.resolve()
    output = args.output or root / "V48_36_4_REENTRY_RESTORE.json"

    contract_path = root / "V48_36_REENTRY_CONTRACT.json"
    try:
        if args.archive is None:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(repo / "tools" / "check_v48_36_reentry_contract.py"),
                    "--run",
                    str(root),
                    "--mode",
                    "resume",
                    "--output",
                    str(contract_path),
                ],
                cwd=repo,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(f"reentry contract failed rc={completed.returncode}")
            contract = _read(contract_path)
            if contract.get("action") == "return_existing_terminal":
                doc = {
                    "event": EVENT,
                    "version": VERSION,
                    "implementation_version": IMPLEMENTATION_VERSION,
                    "created_unix": time.time(),
                    "run": str(root),
                    "valid": True,
                    "idempotent_noop": True,
                    "restored": False,
                    "authoritative_exit_code": contract.get("existing_exit_code"),
                    "pipeline_valid": True,
                    "algorithm_changed": False,
                    "retraining_performed": False,
                    "recalibration_performed": False,
                    "gate_decision_changed": False,
                    "test_roots_read": False,
                }
                _atomic(output, doc)
                print(json.dumps(doc, ensure_ascii=False))
                return 0
            if contract.get("action") != "restore_archived_terminal":
                raise RuntimeError(f"reentry action is {contract.get('action')!r}, not restore_archived_terminal")
            archive = Path(str(contract["archive"]["directory"]))
        else:
            archive = args.archive
            contract = {}

        archived_complete = _read(archive / "V48_36_COMPLETE.json")
        archived_state = _read(archive / "AUTHORITATIVE_RUN_STATUS.json")
        archived_next = _read(archive / "NEXT_COMMANDS_STATUS.json")
        archived_blocked = _read(archive / "NEXT_COMMANDS_BLOCKED.json")
        restored_rc = int(archived_complete["pipeline_exit_code"])
        attempt_id = str(archived_complete["attempt_id"])
        if restored_rc != 20:
            raise RuntimeError(
                f"only archived RC=20 is recoverable; historical RC=0 archives omit NEXT_COMMANDS.txt: {restored_rc}"
            )
        if not (
            archived_complete.get("pipeline_valid") is True
            and archived_state.get("valid") is True
            and int(archived_state.get("authoritative_exit_code", -1)) == restored_rc
            and archived_state.get("attempt_id") == attempt_id
            and archived_next.get("attempt_id") == attempt_id
            and archived_blocked.get("attempt_id") == attempt_id
        ):
            raise RuntimeError("archive terminal bundle is internally inconsistent")
        archived_gate = _read(archive / "GATE_FAILED.json")
        selection = _read(root / "dedicated_recalibration_status.json")
        if archived_gate != selection:
            raise RuntimeError("archived GATE_FAILED does not exactly match active candidate selection")
        if selection.get("attempt_id") != attempt_id:
            raise RuntimeError("active candidate selection belongs to a different attempt")
    except Exception as exc:
        doc = {
            "event": EVENT,
            "version": VERSION,
            "implementation_version": IMPLEMENTATION_VERSION,
            "created_unix": time.time(),
            "run": str(root),
            "valid": False,
            "errors": [f"preflight_error: {type(exc).__name__}: {exc}"],
            "algorithm_changed": False,
            "retraining_performed": False,
            "recalibration_performed": False,
            "gate_decision_changed": False,
            "test_roots_read": False,
        }
        _atomic(output, doc)
        print(json.dumps(doc, ensure_ascii=False))
        return 4

    touched = [
        root / "ATTEMPT_STARTED.json",
        root / "PIPELINE_FAILED.json",
        root / "CALIBRATION_FAILED.json",
        root / "V48_36_COMPLETE.json",
        root / "AUTHORITATIVE_RUN_STATUS.json",
        root / "NEXT_COMMANDS_STATUS.json",
        root / "NEXT_COMMANDS_BLOCKED.json",
        root / "GATE_FAILED.json",
        root / "NEXT_COMMANDS.txt",
        root / "V48_36_RESUME_CONTRACT.json",
        root / "V48_36_CERTIFICATE_STATUS_CONTRACT.json",
    ]
    backup = root / "repair_history" / f"v48.36.4-resume-clobber-{time.time_ns()}"
    presence = _copy_backup(root, backup, touched)
    for name in ("V48_36_COMPLETE.json", "AUTHORITATIVE_RUN_STATUS.json", "NEXT_COMMANDS_STATUS.json", "NEXT_COMMANDS_BLOCKED.json"):
        shutil.copy2(archive / name, backup / f"SOURCE_ARCHIVE_{name}")
    if archived_gate is not None:
        shutil.copy2(archive / "GATE_FAILED.json", backup / "SOURCE_ARCHIVE_GATE_FAILED.json")
    _atomic(
        backup / "RESTORE_INPUT_AUDIT.json",
        {
            "event": EVENT,
            "created_unix": time.time(),
            "archive": str(archive),
            "attempt_id": attempt_id,
            "restored_exit_code": restored_rc,
            "reentry_contract": contract,
            "algorithm_changed": False,
            "test_roots_read": False,
        },
    )

    try:
        for name in ("V48_36_COMPLETE.json", "NEXT_COMMANDS_STATUS.json", "NEXT_COMMANDS_BLOCKED.json"):
            shutil.copy2(archive / name, root / name)
        assert archived_gate is not None
        shutil.copy2(archive / "GATE_FAILED.json", root / "GATE_FAILED.json")
        try:
            (root / "NEXT_COMMANDS.txt").unlink()
        except FileNotFoundError:
            pass
        for path in (root / "PIPELINE_FAILED.json", root / "CALIBRATION_FAILED.json", root / "V48_36_RESUME_CONTRACT.json"):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        attempt_doc = {
            "event": "v48_36_attempt_started",
            "version": VERSION,
            "implementation_version": archived_complete.get("implementation_version"),
            "terminal_state_management_version": IMPLEMENTATION_VERSION,
            "created_unix": time.time(),
            "attempt_id": attempt_id,
            "source_run": archived_complete.get("source_run"),
            "protocol_root": archived_complete.get("protocol_root"),
            "resume_after_adaptation": bool(archived_complete.get("adaptation_reused_without_retraining")),
            "restored_after_non_mutating_resume_refusal": True,
            "restore_history": str(backup),
            "test_roots_read": False,
        }
        _atomic(root / "ATTEMPT_STARTED.json", attempt_doc)

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
                str(restored_rc),
                "--expect-attempt-id",
                attempt_id,
            ],
            repo,
        )
        final_state = _read(root / "AUTHORITATIVE_RUN_STATUS.json")
        final_complete = _read(root / "V48_36_COMPLETE.json")
        if not (
            final_state.get("valid") is True
            and int(final_state.get("authoritative_exit_code", -1)) == restored_rc
            and final_complete.get("pipeline_valid") is True
            and int(final_complete.get("pipeline_exit_code", -1)) == restored_rc
        ):
            raise RuntimeError("post-restore authoritative state is invalid")
    except Exception as exc:
        _rollback(root, backup, touched, presence)
        doc = {
            "event": EVENT,
            "version": VERSION,
            "implementation_version": IMPLEMENTATION_VERSION,
            "created_unix": time.time(),
            "run": str(root),
            "valid": False,
            "rolled_back": True,
            "restore_history": str(backup),
            "errors": [f"restore_error: {type(exc).__name__}: {exc}"],
            "algorithm_changed": False,
            "retraining_performed": False,
            "recalibration_performed": False,
            "gate_decision_changed": False,
            "test_roots_read": False,
        }
        _atomic(output, doc)
        print(json.dumps(doc, ensure_ascii=False))
        return 4

    doc = {
        "event": EVENT,
        "version": VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "created_unix": time.time(),
        "run": str(root),
        "attempt_id": attempt_id,
        "valid": True,
        "restored": True,
        "idempotent_noop": False,
        "source_archive": str(archive),
        "restore_history": str(backup),
        "authoritative_exit_code": restored_rc,
        "pipeline_valid": True,
        "natural_gate_failed": restored_rc == 20,
        "algorithm_changed": False,
        "retraining_performed": False,
        "recalibration_performed": False,
        "certificate_scores_changed": False,
        "gate_decision_changed": False,
        "test_roots_read": False,
    }
    _atomic(output, doc)
    print(json.dumps(doc, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
