from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_source_checkpoint_contract_valid_and_missing(tmp_path: Path) -> None:
    source = tmp_path / "runs" / "ocrap_v48_13_terra_proxy_4801"
    for variant in ("balanced", "precision"):
        p = source / "candidates" / variant / "model_v48_trac_sr" / "best.pt"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((variant + "-checkpoint").encode())
    out = tmp_path / "contract.json"
    ok = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "check_v48_36_source_checkpoint_contract.py"),
            "--source-run",
            str(source),
            "--output",
            str(out),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert ok.returncode == 0, ok.stdout + ok.stderr
    doc = json.loads(out.read_text())
    assert doc["valid"] is True
    assert doc["checks"]["balanced"]["sha256"]
    assert doc["checks"]["precision"]["sha256"]
    assert doc["test_roots_read"] is False

    (source / "candidates" / "precision" / "model_v48_trac_sr" / "best.pt").unlink()
    bad = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "check_v48_36_source_checkpoint_contract.py"),
            "--source-run",
            str(source),
            "--output",
            str(out),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert bad.returncode != 0
    doc = json.loads(out.read_text())
    assert doc["valid"] is False
    assert doc["checks"]["balanced"]["exists"] is True
    assert doc["checks"]["precision"]["exists"] is False


def test_dedicated_missing_source_fails_before_dataset_and_adaptation(tmp_path: Path) -> None:
    run = tmp_path / "run"
    missing_source = tmp_path / "persistent-runs" / "ocrap_v48_13_terra_proxy_4801"
    completed = subprocess.run(
        ["bash", str(ROOT / "scripts" / "run_v48_36_ocaf_dedicated.sh")],
        cwd=ROOT,
        env={
            **os.environ,
            "OCRAP_REPO": str(ROOT),
            "OUTPUTDIR": str(run),
            "SOURCE_RUN": str(missing_source),
            "PROTOCOL_ROOT": str(tmp_path / "missing-protocol"),
            "CAL_SAFE": str(tmp_path / "missing-safe"),
            "RESUME_AFTER_ADAPTATION": "0",
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 30, completed.stdout + completed.stderr
    failed = json.loads((run / "PIPELINE_FAILED.json").read_text())
    contract = json.loads((run / "SOURCE_CHECKPOINT_CONTRACT.json").read_text())
    assert failed["stage"] == "source_checkpoint_contract"
    assert failed["certificate_executed"] is False
    assert failed["gate_evaluated"] is False
    assert contract["valid"] is False
    assert not (run / "DATASET_ROOT_CONTRACT.json").exists()
    assert not (run / "candidates").exists()


def _make_fake_parallel_repo(tmp_path: Path, rc_map: dict[str, int]) -> Path:
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy2(ROOT / "scripts" / "run_v48_45_sowr_2x2_parallel.sh", scripts / "run_v48_45_sowr_2x2_parallel.sh")
    cases = "\n".join(f"  {arm}) exit {rc} ;;" for arm, rc in rc_map.items())
    (scripts / "run_v48_45_sowr_ablation_arm.sh").write_text(
        "#!/usr/bin/env bash\nset -Eeuo pipefail\ncase \"$1\" in\n" + cases + "\n  *) exit 2 ;;\nesac\n",
        encoding="utf-8",
    )
    os.chmod(scripts / "run_v48_45_sowr_ablation_arm.sh", 0o755)
    return repo


def test_parallel_launcher_accepts_rc20_as_valid_ablation(tmp_path: Path) -> None:
    repo = _make_fake_parallel_repo(tmp_path, {"A": 20, "B": 20, "C": 20, "D": 20})
    out = tmp_path / "runs"
    completed = subprocess.run(
        ["bash", str(repo / "scripts" / "run_v48_45_sowr_2x2_parallel.sh")],
        cwd=repo,
        env={**os.environ, "OCRAP_REPO": str(repo), "BASE_OUT": str(out), "MAX_PARALLEL_ARMS": "2"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout.count("valid Natural-gate failure") == 4
    status = json.loads((out / "ocrap_v48_45_sowr_parallel_status.json").read_text())
    assert status["engineering_failed"] is False
    assert status["any_natural_gate_failure"] is True
    assert all(row["classification"] == "valid_natural_gate_failure" for row in status["arms"].values())


def test_parallel_launcher_rejects_engineering_failure(tmp_path: Path) -> None:
    repo = _make_fake_parallel_repo(tmp_path, {"A": 20, "B": 30, "C": 0, "D": 20})
    out = tmp_path / "runs"
    completed = subprocess.run(
        ["bash", str(repo / "scripts" / "run_v48_45_sowr_2x2_parallel.sh")],
        cwd=repo,
        env={**os.environ, "OCRAP_REPO": str(repo), "BASE_OUT": str(out), "MAX_PARALLEL_ARMS": "2"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 1
    assert "arm B ENGINEERING FAILED: RC=30" in completed.stderr
    status = json.loads((out / "ocrap_v48_45_sowr_parallel_status.json").read_text())
    assert status["engineering_failed"] is True
    assert status["arms"]["B"]["classification"] == "engineering_failure"


def test_v4845_arm_resolves_reference_source_from_base_out() -> None:
    text = (ROOT / "scripts" / "run_v48_45_sowr_ablation_arm.sh").read_text()
    assert 'SOURCE_RUN_BASENAME="${V4845_SOURCE_RUN_BASENAME:-ocrap_v48_13_terra_proxy_4801}"' in text
    assert '$BASE_OUT/$SOURCE_RUN_BASENAME/candidates/balanced/model_v48_trac_sr/best.pt' in text
    assert '$BASE_OUT/$SOURCE_RUN_BASENAME/candidates/precision/model_v48_trac_sr/best.pt' in text
    assert 'export SOURCE_RUN' in text


def test_v4845_sowr_controls_explicitly_cross_variant_process_boundary() -> None:
    text = (ROOT / "scripts" / "run_v48_36_ocaf_dedicated.sh").read_text()
    assert 'V4845_SOWR_MARGIN_WITNESS="${V4845_SOWR_MARGIN_WITNESS:-0}"' in text
    assert 'V4845_SOWR_OBS_KERNEL="${V4845_SOWR_OBS_KERNEL:-0}"' in text
    assert 'SOWR_LR="${SOWR_LR:-0.00005}"' in text
    assert text.index("check_v48_36_source_checkpoint_contract.py") < text.index("check_v48_36_dataset_root_contract.py")


def test_sowr_env_command_has_no_backslash_comment_break():
    text = (ROOT / "scripts" / "adapt_ocrap_v48_45_sowr_stage.sh").read_text()
    # A comment immediately after a backslash-continued assignment silently
    # terminates the env-assignment command after line-continuation removal.
    assert not re.search(r"\\\n\s*#", text)
    assert "SKIP_POST_TRAIN_CALIBRATION=1 \\" in text
