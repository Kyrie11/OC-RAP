from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _run(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    merged = os.environ.copy()
    merged["PYTHONPATH"] = f"{ROOT / 'tools'}:{ROOT / 'src'}" + (f":{merged['PYTHONPATH']}" if merged.get("PYTHONPATH") else "")
    if env:
        merged.update(env)
    return subprocess.run(args, cwd=ROOT, env=merged, text=True, capture_output=True, check=False)


def _make_rc20_run(root: Path, *, attempt_id: str | None = None, stale_pipeline: bool = True) -> None:
    complete = {
        "event": "v48_35_continuous_frontier_controller_complete",
        "created_unix": 200.0,
        "pipeline_exit_code": 20,
        "certificate_exit_code": 20,
        "pipeline_valid": True,
        "certificate_executed": True,
        "gate_evaluated": True,
        "gate_passed": False,
        "next_commands_generated": False,
        "test_roots_read": False,
    }
    gate = {
        "event": "v48_35_certificate_candidate_selection",
        "created_unix": 190.0,
        "certificate_executed": True,
        "gate_evaluated": True,
        "valid_candidates": [],
        "test_roots_read": False,
    }
    blocked = {
        "event": "v48_35_next_commands_blocked",
        "created_unix": 191.0,
        "reason": "natural_gate_failed",
        "exit_code": 20,
        "generated": False,
        "certificate_executed": True,
        "gate_evaluated": True,
        "test_roots_read": False,
    }
    if attempt_id:
        complete["attempt_id"] = attempt_id
        gate["attempt_id"] = attempt_id
        blocked["attempt_id"] = attempt_id
    _write_json(root / "V48_35_COMPLETE.json", complete)
    _write_json(root / "GATE_FAILED.json", gate)
    _write_json(root / "NEXT_COMMANDS_BLOCKED.json", blocked)
    _write_json(root / "NEXT_COMMANDS_STATUS.json", blocked)
    (root / "artifact.txt").write_text("current\n", encoding="utf-8")
    if stale_pipeline:
        failed = {
            "event": "v48_35_pipeline_failed",
            "created_unix": 100.0,
            "pipeline_exit_code": 30,
            "normalized_exit_code": 30,
            "pipeline_valid": False,
            "test_roots_read": False,
        }
        _write_json(root / "PIPELINE_FAILED.json", failed)


def test_authoritative_state_resolves_legacy_stale_rc30_marker(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    _make_rc20_run(run)
    output = run / "AUTHORITATIVE_RUN_STATUS.json"
    proc = _run(sys.executable, "tools/audit_v48_35_run_state.py", "--run", str(run), "--output", str(output))
    assert proc.returncode == 0, proc.stderr + proc.stdout
    doc = json.loads(output.read_text(encoding="utf-8"))
    assert doc["valid"] is True
    assert doc["authoritative_exit_code"] == 20
    assert doc["pipeline_valid"] is True
    assert [item["name"] for item in doc["stale_markers"]] == ["PIPELINE_FAILED.json"]


def test_same_attempt_pipeline_failure_cannot_be_hidden(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    _make_rc20_run(run, attempt_id="attempt-a", stale_pipeline=False)
    failed = json.loads((run / "PIPELINE_FAILED.json").read_text()) if (run / "PIPELINE_FAILED.json").exists() else {
        "event": "v48_35_pipeline_failed",
        "created_unix": 199.0,
        "pipeline_exit_code": 30,
    }
    failed["attempt_id"] = "attempt-a"
    _write_json(run / "PIPELINE_FAILED.json", failed)
    output = run / "AUTHORITATIVE_RUN_STATUS.json"
    proc = _run(sys.executable, "tools/audit_v48_35_run_state.py", "--run", str(run), "--output", str(output))
    assert proc.returncode == 4
    doc = json.loads(output.read_text(encoding="utf-8"))
    assert doc["valid"] is False
    assert any("PIPELINE_FAILED" in item for item in doc["active_contradictions"])


def test_packager_recreates_zip_and_excludes_stale_marker(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    _make_rc20_run(run)
    _write_json(run / "AUTHORITATIVE_RUN_STATUS.json", {"obsolete": True})
    _write_json(run / "PACKAGING_MANIFEST.json", {"obsolete": True})
    output = tmp_path / "result.zip"
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr(f"{run.name}/PIPELINE_FAILED.json", "obsolete")
        archive.writestr("obsolete.txt", "obsolete")
    proc = _run(sys.executable, "tools/package_v48_35_results.py", "--run", str(run), "--output", str(output))
    assert proc.returncode == 0, proc.stderr + proc.stdout
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names))
        assert f"{run.name}/PIPELINE_FAILED.json" not in names
        assert "obsolete.txt" not in names
        assert f"{run.name}/GATE_FAILED.json" in names
        status = json.loads(archive.read(f"{run.name}/AUTHORITATIVE_RUN_STATUS.json"))
        assert status["authoritative_exit_code"] == 20
        assert status["stale_markers_excluded_from_archive"] == ["PIPELINE_FAILED.json"]


def test_shared_rule_gate_decomposition_is_supported(tmp_path: Path) -> None:
    run = tmp_path / "run"
    for variant in ("balanced", "precision"):
        cal = run / "candidates" / variant / "calibration"
        shared = {
            "valid": False,
            "valid_for_deployment": False,
            "diagnostic_fit_rule": {"opportunity_threshold": 0.5},
            "fit": {
                "constraint_failures": 2,
                "constraint_deficit": 3.0,
                "by_stratum": {"near": {"num_selected": 1}, "contact": {"num_selected": 2}},
                "pooled": {"num_selected": 3},
            },
        }
        _write_json(cal / "dev_frozen_shared_rule_v48.json", shared)
        for regime in ("near", "contact"):
            cert = {
                "valid_for_deployment": False,
                "rejection_kind": "development_rule_fit_rejection",
                "proposal_constrained_oracle_gate": {"verify": {"feasible": True, "proposal_safe_positive_groups": 4}},
                "verify": {"num_selected": 0},
            }
            _write_json(cal / f"direct_value_risk_{regime}_v48.json", cert)
    output = run / "GATE_FAILURE_DECOMPOSITION.json"
    proc = _run(sys.executable, "tools/summarize_v48_34_gate_failure.py", "--run", str(run), "--output", str(output))
    assert proc.returncode == 0, proc.stderr + proc.stdout
    doc = json.loads(output.read_text(encoding="utf-8"))
    assert doc["artifact_valid"] is True
    assert doc["development_rule_modes"] == ["shared"]
    assert doc["errors"] == []


def test_v4835_shell_python_heredocs_compile() -> None:
    scripts = [
        ROOT / "scripts" / "run_v48_35_continuous_frontier_dedicated.sh",
        ROOT / "scripts" / "calibrate_v48_35_shared_certificate_pool.sh",
    ]
    pattern = re.compile(r"<<'(?P<tag>PY[A-Z0-9_]*)'\n(?P<body>.*?)\n(?P=tag)(?:\n|$)", re.DOTALL)
    for script in scripts:
        text = script.read_text(encoding="utf-8")
        blocks = list(pattern.finditer(text))
        assert blocks, script
        for index, match in enumerate(blocks):
            compile(match.group("body"), f"{script.name}:heredoc:{index}", "exec")


def test_controller_has_attempt_scoped_state_and_required_diagnostics() -> None:
    controller = (ROOT / "scripts" / "run_v48_35_continuous_frontier_dedicated.sh").read_text(encoding="utf-8")
    assert "ATTEMPT_ID=" in controller
    assert "status_history" in controller
    assert "audit_v48_35_run_state.py" in controller
    assert "post_certificate_diagnostics" in controller
    assert "summarize_v48_34_gate_failure.py" in controller
    assert "summarize_v48_34_gate_failure.py --run" in controller
    assert "summarize_v48_34_gate_failure.py --run \"$OUTPUTDIR\" --output \"$OUTPUTDIR/GATE_FAILURE_DECOMPOSITION.json\" || true" not in controller


def test_generated_followup_commands_are_portable_and_attempt_scoped() -> None:
    calibration = (ROOT / "scripts" / "calibrate_v48_35_shared_certificate_pool.sh").read_text(encoding="utf-8")
    assert "V4835_ATTEMPT_ID" in calibration
    assert "SAFE_WOMD_SOURCE={shlex.quote(safe_source)}" in calibration
    assert "SAFE_WOMD_SOURCE=/data0/senzeyu2/dataset/WOMD" not in calibration.split("safe_source=os.environ['SAFE_WOMD_SOURCE']", 1)[1]


def test_packager_excludes_its_own_outputs_when_target_is_inside_run(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    _make_rc20_run(run)
    output = run / "clean-result.zip"
    output.write_bytes(b"obsolete archive")
    output.with_suffix(".zip.sha256").write_text("obsolete hash\n", encoding="utf-8")
    proc = _run(sys.executable, "tools/package_v48_35_results.py", "--run", str(run), "--output", str(output))
    assert proc.returncode == 0, proc.stderr + proc.stdout
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        assert f"{run.name}/clean-result.zip" not in names
        assert f"{run.name}/clean-result.zip.sha256" not in names


def test_resume_rejection_publishes_attempt_scoped_terminal_rc30() -> None:
    controller = (ROOT / "scripts" / "run_v48_35_continuous_frontier_dedicated.sh").read_text(encoding="utf-8")
    resume_start = controller.index('if [[ "$RESUME_AFTER_ADAPTATION" == 1 ]]')
    resume_end = controller.index("# Preserve previous active status", resume_start)
    block = controller[resume_start:resume_end]
    assert "resume-refused-{attempt}" in block
    assert 'write_pipeline_failure "resume_authorization" "$resume_contract_rc"' in block
    assert "exit 30" in block


def test_authoritative_state_accepts_consistent_rc0(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    attempt = "attempt-pass"
    complete = {
        "event": "v48_35_continuous_frontier_controller_complete",
        "created_unix": 20.0,
        "attempt_id": attempt,
        "pipeline_exit_code": 0,
        "pipeline_valid": True,
        "certificate_executed": True,
        "gate_evaluated": True,
        "gate_passed": True,
        "next_commands_generated": True,
    }
    status = {"created_unix": 19.0, "attempt_id": attempt, "generated": True}
    _write_json(run / "V48_35_COMPLETE.json", complete)
    _write_json(run / "NEXT_COMMANDS_STATUS.json", status)
    (run / "NEXT_COMMANDS.txt").write_text("echo authorized\n", encoding="utf-8")
    output = run / "AUTHORITATIVE_RUN_STATUS.json"
    proc = _run(sys.executable, "tools/audit_v48_35_run_state.py", "--run", str(run), "--output", str(output), "--expect-exit-code", "0")
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert json.loads(output.read_text())["valid"] is True


def test_authoritative_state_accepts_consistent_rc30(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    attempt = "attempt-fail"
    complete = {
        "event": "v48_35_continuous_frontier_controller_complete",
        "created_unix": 20.0,
        "attempt_id": attempt,
        "pipeline_exit_code": 30,
        "pipeline_valid": False,
        "certificate_executed": False,
        "gate_evaluated": False,
        "gate_passed": False,
        "next_commands_generated": False,
    }
    failed = {
        "event": "v48_35_pipeline_failed",
        "created_unix": 19.0,
        "attempt_id": attempt,
        "pipeline_exit_code": 30,
        "pipeline_valid": False,
    }
    blocked = {
        "created_unix": 19.5,
        "attempt_id": attempt,
        "reason": "pipeline_failure",
        "generated": False,
        "pipeline_exit_code": 30,
    }
    _write_json(run / "V48_35_COMPLETE.json", complete)
    _write_json(run / "PIPELINE_FAILED.json", failed)
    _write_json(run / "NEXT_COMMANDS_BLOCKED.json", blocked)
    _write_json(run / "NEXT_COMMANDS_STATUS.json", blocked)
    output = run / "AUTHORITATIVE_RUN_STATUS.json"
    proc = _run(sys.executable, "tools/audit_v48_35_run_state.py", "--run", str(run), "--output", str(output), "--expect-exit-code", "30")
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert json.loads(output.read_text())["valid"] is True
