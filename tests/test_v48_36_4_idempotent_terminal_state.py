from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ATTEMPT = "v4836-2000000000000000000-validrc20"
FAILED_ATTEMPT = "v4836-2000000000000000001-refused"
VERSION = "v48.36.4-IDEMPOTENT-TERMINAL-STATE-HOTFIX"


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _build_valid_rc20(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    run.mkdir()
    selection = {
        "event": "v48_36_certificate_candidate_selection",
        "version": "v48.36-OCAF",
        "implementation_version": VERSION,
        "created_unix": 100.0,
        "attempt_id": ATTEMPT,
        "requested_variants": ["balanced", "precision"],
        "controller_exit_codes": {"balanced": 20, "precision": 20},
        "certificate_executed": True,
        "gate_evaluated": True,
        "valid_candidates": [],
        "candidates": {},
        "test_roots_read": False,
    }
    _write(run / "dedicated_recalibration_status.json", selection)
    _write(run / "GATE_FAILED.json", selection)
    _write(
        run / "GATE_SPEC.json",
        {
            "event": "v48_36_gate_protocol_preregistered",
            "attempt_id": ATTEMPT,
            "protocol_sha256": "abc",
            "test_roots_read": False,
        },
    )
    next_status = {
        "event": "v48_36_next_commands_blocked",
        "version": "v48.36-OCAF",
        "implementation_version": VERSION,
        "created_unix": 101.0,
        "attempt_id": ATTEMPT,
        "reason": "natural_gate_failed",
        "exit_code": 20,
        "certificate_executed": True,
        "gate_evaluated": True,
        "test_roots_read": False,
        "controller_exit_codes": {"balanced": 20, "precision": 20},
    }
    _write(run / "NEXT_COMMANDS_BLOCKED.json", next_status)
    _write(run / "NEXT_COMMANDS_STATUS.json", {**next_status, "generated": False})
    for variant in ("balanced", "precision"):
        cal = run / "candidates" / variant / "calibration"
        _write(
            cal / "CERTIFICATE_CALIBRATION_COMPLETE.json",
            {
                "event": "v48_36_certificate_pool_calibration_complete",
                "attempt_id": ATTEMPT,
                "near_exit_code": 3,
                "contact_exit_code": 3,
                "certificate_executed": True,
                "gate_evaluated": True,
                "certificate_data_valid": True,
                "test_roots_read": False,
            },
        )
        _write(
            cal / "SAFE_REGIME_STATUS.json",
            {"event": "v48_36_safe_regime_status", "attempt_id": ATTEMPT, "test_roots_read": False},
        )
    complete = {
        "event": "v48_36_ocaf_controller_complete",
        "version": "v48.36-OCAF",
        "implementation_version": VERSION,
        "created_unix": 102.0,
        "attempt_id": ATTEMPT,
        "source_run": str(tmp_path / "source"),
        "protocol_root": str(tmp_path / "protocol"),
        "variants": {},
        "raw_certificate_exit_code": 20,
        "certificate_exit_code": 20,
        "pipeline_exit_code": 20,
        "certificate_executed": True,
        "gate_evaluated": True,
        "gate_passed": False,
        "next_commands_generated": False,
        "pipeline_valid": True,
        "adaptation_reused_without_retraining": False,
        "resume_contract": None,
        "test_roots_read": False,
    }
    _write(run / "V48_36_COMPLETE.json", complete)
    _write(
        run / "ATTEMPT_STARTED.json",
        {
            "event": "v48_36_attempt_started",
            "attempt_id": ATTEMPT,
            "source_run": complete["source_run"],
            "protocol_root": complete["protocol_root"],
            "resume_after_adaptation": False,
            "test_roots_read": False,
        },
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "check_v48_36_certificate_status_contract.py"),
            "--run",
            str(run),
            "--expected-attempt-id",
            ATTEMPT,
            "--output",
            str(run / "V48_36_CERTIFICATE_STATUS_CONTRACT.json"),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "resolve_v48_36_authoritative_result.py"),
            "--run",
            str(run),
            "--output",
            str(run / "AUTHORITATIVE_RUN_STATUS.json"),
            "--expect-exit-code",
            "20",
            "--expect-attempt-id",
            ATTEMPT,
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return run


def _clobber_with_refused_resume(run: Path) -> Path:
    archive = run / "status_history" / "resume-refused-valid-archive"
    archive.mkdir(parents=True)
    for name in (
        "V48_36_COMPLETE.json",
        "AUTHORITATIVE_RUN_STATUS.json",
        "NEXT_COMMANDS_STATUS.json",
        "NEXT_COMMANDS_BLOCKED.json",
        "GATE_FAILED.json",
    ):
        shutil.move(str(run / name), str(archive / name))
    _write(
        run / "ATTEMPT_STARTED.json",
        {
            "event": "v48_36_attempt_started",
            "attempt_id": FAILED_ATTEMPT,
            "resume_after_adaptation": True,
            "test_roots_read": False,
        },
    )
    _write(
        run / "PIPELINE_FAILED.json",
        {
            "event": "v48_36_pipeline_failed",
            "attempt_id": FAILED_ATTEMPT,
            "stage": "resume_authorization",
            "raw_exit_code": 4,
            "normalized_exit_code": 30,
            "pipeline_exit_code": 30,
            "certificate_executed": False,
            "gate_evaluated": False,
            "pipeline_valid": False,
            "test_roots_read": False,
        },
    )
    _write(
        run / "V48_36_COMPLETE.json",
        {
            "event": "v48_36_ocaf_controller_complete",
            "attempt_id": FAILED_ATTEMPT,
            "pipeline_exit_code": 30,
            "certificate_executed": False,
            "gate_evaluated": False,
            "pipeline_valid": False,
            "failure_stage": "resume_authorization",
            "test_roots_read": False,
        },
    )
    blocked = {
        "event": "v48_36_next_commands_blocked",
        "attempt_id": FAILED_ATTEMPT,
        "reason": "pipeline_failure",
        "failure_stage": "resume_authorization",
        "pipeline_exit_code": 30,
        "certificate_executed": False,
        "gate_evaluated": False,
        "test_roots_read": False,
    }
    _write(run / "NEXT_COMMANDS_STATUS.json", blocked)
    _write(run / "NEXT_COMMANDS_BLOCKED.json", blocked)
    _write(
        run / "AUTHORITATIVE_RUN_STATUS.json",
        {
            "event": "v48_36_authoritative_run_state",
            "valid": True,
            "authoritative_exit_code": 30,
            "pipeline_valid": False,
            "attempt_id": FAILED_ATTEMPT,
            "test_roots_read": False,
        },
    )
    _write(run / "V48_36_RESUME_CONTRACT.json", {"valid": False, "test_roots_read": False})
    return archive


def _terminal_bytes(run: Path) -> dict[str, bytes]:
    names = (
        "V48_36_COMPLETE.json",
        "AUTHORITATIVE_RUN_STATUS.json",
        "NEXT_COMMANDS_STATUS.json",
        "NEXT_COMMANDS_BLOCKED.json",
        "GATE_FAILED.json",
    )
    return {name: (run / name).read_bytes() for name in names if (run / name).is_file()}


def test_reentry_returns_existing_rc20_without_terminal_mutation(tmp_path: Path) -> None:
    run = _build_valid_rc20(tmp_path)
    before = _terminal_bytes(run)
    completed = subprocess.run(
        ["bash", str(ROOT / "scripts" / "run_v48_36_ocaf_dedicated.sh")],
        cwd=ROOT,
        env={
            **os.environ,
            "OCRAP_REPO": str(ROOT),
            "OUTPUTDIR": str(run),
            "RESUME_AFTER_ADAPTATION": "1",
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 20, completed.stdout + completed.stderr
    assert "already has a valid authoritative terminal state" in completed.stdout
    assert _terminal_bytes(run) == before
    assert not (run / "V48_36_RESUME_REFUSED.json").exists()


def test_exact_resume_clobber_restores_latest_authoritative_rc20(tmp_path: Path) -> None:
    run = _build_valid_rc20(tmp_path)
    archive = _clobber_with_refused_resume(run)
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "restore_v48_36_terminal_state_after_refused_resume.py"),
            "--run",
            str(run),
            "--repo",
            str(ROOT),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    restore = json.loads((run / "V48_36_4_REENTRY_RESTORE.json").read_text())
    state = json.loads((run / "AUTHORITATIVE_RUN_STATUS.json").read_text())
    complete = json.loads((run / "V48_36_COMPLETE.json").read_text())
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert restore["valid"] is True and restore["restored"] is True
    assert restore["source_archive"] == str(archive)
    assert state["valid"] is True and state["authoritative_exit_code"] == 20
    assert complete["pipeline_valid"] is True and complete["pipeline_exit_code"] == 20
    assert not (run / "PIPELINE_FAILED.json").exists()
    assert not (run / "V48_36_RESUME_CONTRACT.json").exists()
    assert json.loads((run / "GATE_FAILED.json").read_text())["attempt_id"] == ATTEMPT


def test_controller_auto_restores_resume_clobber_and_returns_rc20(tmp_path: Path) -> None:
    run = _build_valid_rc20(tmp_path)
    _clobber_with_refused_resume(run)
    completed = subprocess.run(
        ["bash", str(ROOT / "scripts" / "run_v48_36_ocaf_dedicated.sh")],
        cwd=ROOT,
        env={
            **os.environ,
            "OCRAP_REPO": str(ROOT),
            "OUTPUTDIR": str(run),
            "RESUME_AFTER_ADAPTATION": "1",
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 20, completed.stdout + completed.stderr
    assert "restored the previous authoritative terminal state" in completed.stdout
    assert json.loads((run / "AUTHORITATIVE_RUN_STATUS.json").read_text())["authoritative_exit_code"] == 20


def test_unknown_resume_refusal_does_not_overwrite_active_rc30(tmp_path: Path) -> None:
    run = tmp_path / "unknown"
    run.mkdir()
    active = {
        "event": "v48_36_ocaf_controller_complete",
        "attempt_id": "v4836-unknown",
        "pipeline_exit_code": 30,
        "pipeline_valid": False,
        "failure_stage": "adaptation",
        "test_roots_read": False,
    }
    failed = {
        "event": "v48_36_pipeline_failed",
        "attempt_id": "v4836-unknown",
        "stage": "adaptation",
        "raw_exit_code": 30,
        "pipeline_exit_code": 30,
        "pipeline_valid": False,
        "test_roots_read": False,
    }
    state = {
        "event": "v48_36_authoritative_run_state",
        "valid": True,
        "authoritative_exit_code": 30,
        "pipeline_valid": False,
        "attempt_id": "v4836-unknown",
        "test_roots_read": False,
    }
    blocked = {"attempt_id": "v4836-unknown", "pipeline_exit_code": 30, "test_roots_read": False}
    _write(run / "V48_36_COMPLETE.json", active)
    _write(run / "PIPELINE_FAILED.json", failed)
    _write(run / "AUTHORITATIVE_RUN_STATUS.json", state)
    _write(run / "NEXT_COMMANDS_STATUS.json", blocked)
    _write(run / "NEXT_COMMANDS_BLOCKED.json", blocked)
    before = {p.name: p.read_bytes() for p in run.glob("*.json")}
    completed = subprocess.run(
        ["bash", str(ROOT / "scripts" / "run_v48_36_ocaf_dedicated.sh")],
        cwd=ROOT,
        env={
            **os.environ,
            "OCRAP_REPO": str(ROOT),
            "OUTPUTDIR": str(run),
            "RESUME_AFTER_ADAPTATION": "1",
            "SOURCE_RUN": str(tmp_path / "source"),
            "PROTOCOL_ROOT": str(tmp_path / "protocol"),
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 30
    for name, payload in before.items():
        assert (run / name).read_bytes() == payload
    refusal = json.loads((run / "V48_36_RESUME_REFUSED.json").read_text())
    assert refusal["active_terminal_state_preserved"] is True
    assert refusal["attempt_created"] is False


def test_restore_rolls_back_on_post_contract_failure(tmp_path: Path) -> None:
    run = _build_valid_rc20(tmp_path)
    _clobber_with_refused_resume(run)
    tracked = [
        run / "ATTEMPT_STARTED.json",
        run / "PIPELINE_FAILED.json",
        run / "V48_36_COMPLETE.json",
        run / "AUTHORITATIVE_RUN_STATUS.json",
        run / "NEXT_COMMANDS_STATUS.json",
        run / "NEXT_COMMANDS_BLOCKED.json",
        run / "V48_36_RESUME_CONTRACT.json",
        run / "V48_36_CERTIFICATE_STATUS_CONTRACT.json",
    ]
    before = {str(p.relative_to(run)): p.read_bytes() for p in tracked if p.exists()}
    fake = tmp_path / "fake-repo"
    tools = fake / "tools"
    tools.mkdir(parents=True)
    for name in ("check_v48_36_reentry_contract.py", "check_v48_36_certificate_status_contract.py"):
        shutil.copy2(ROOT / "tools" / name, tools / name)
    (tools / "resolve_v48_36_authoritative_result.py").write_text("raise SystemExit(4)\n", encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "restore_v48_36_terminal_state_after_refused_resume.py"),
            "--run",
            str(run),
            "--repo",
            str(fake),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    restore = json.loads((run / "V48_36_4_REENTRY_RESTORE.json").read_text())
    assert completed.returncode == 4
    assert restore["valid"] is False and restore["rolled_back"] is True
    for rel, payload in before.items():
        assert (run / rel).read_bytes() == payload


def test_controller_orders_reentry_before_attempt_creation_and_resume_failure_is_non_destructive() -> None:
    text = (ROOT / "scripts" / "run_v48_36_ocaf_dedicated.sh").read_text()
    assert text.index("check_v48_36_reentry_contract.py") < text.index("ATTEMPT_ID=")
    assert text.index("check_v48_36_resume_contract.py") < text.index("ATTEMPT_ID=")
    refusal_block = text[text.index("if [[ \"$resume_contract_rc\" != 0 ]]"): text.index("ATTEMPT_ID=")]
    assert "write_pipeline_failure" not in refusal_block
    assert "shutil.move" not in refusal_block
    assert "active terminal state was preserved" in refusal_block
