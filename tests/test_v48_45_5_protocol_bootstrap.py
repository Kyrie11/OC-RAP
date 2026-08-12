from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _dataset(root: Path, name: str, split: str, scenes: int) -> Path:
    d = root / name
    (d / "samples").mkdir(parents=True)
    rows = []
    for i in range(scenes):
        scene = f"{name}_scene_{i:03d}"
        for a in range(2):
            fn = f"{scene}_t0001_a{a:02d}.npz"
            (d / "samples" / fn).write_bytes(b"fixture")
            rows.append({
                "path": f"samples/{fn}",
                "scene_id": scene,
                "original_scenario_id": scene,
                "split_id": split,
                "candidate_id": str(a),
            })
    with (d / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return d


def test_protocol_bootstrap_builds_exact_scene_disjoint_roles_and_reuses(tmp_path: Path) -> None:
    _dataset(tmp_path, "calibration_near_contact", "calibration", 40)
    _dataset(tmp_path, "calibration_contact", "calibration", 50)
    _dataset(tmp_path, "calibration_safe", "calibration", 12)
    env = os.environ.copy()
    env.update({"OCRAP_REPO": str(ROOT), "OCRAP_ROOT": str(tmp_path), "V4845_PROTOCOL_LINK_MODE": "hardlink"})
    cmd = ["bash", str(ROOT / "scripts" / "prepare_v48_45_protocol.sh")]
    first = subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True, check=False)
    assert first.returncode == 0, first.stdout + first.stderr
    protocol = tmp_path / "calibration_v48_14_prism_4814"
    seal = json.loads((protocol / "V48_45_PROTOCOL_SEAL.json").read_text())
    assert seal["valid"] is True
    assert seal["test_roots_read"] is False
    contract_out = tmp_path / "dataset_root_contract.json"
    contract = subprocess.run([
        sys.executable, str(ROOT / "tools" / "check_v48_36_dataset_root_contract.py"),
        "--protocol-root", str(protocol), "--safe-root", str(tmp_path / "calibration_safe"),
        "--train-near", str(protocol / "evidence_adapt_train_near_contact"),
        "--train-contact", str(protocol / "evidence_adapt_train_contact"),
        "--dev-near", str(protocol / "evidence_adapt_dev_near_contact"),
        "--dev-contact", str(protocol / "evidence_adapt_dev_contact"),
        "--cert-near", str(protocol / "certificate_pool_near_contact"),
        "--cert-contact", str(protocol / "certificate_pool_contact"),
        "--output", str(contract_out),
    ], cwd=ROOT, text=True, capture_output=True, check=False)
    assert contract.returncode == 0, contract.stdout + contract.stderr
    assert json.loads(contract_out.read_text())["valid"] is True
    for regime in ("near_contact", "contact"):
        assert seal["checks"][f"{regime}_scene_disjoint"] is True
        assert seal["checks"][f"{regime}_scene_union_exact"] is True
        assert seal["checks"][f"{regime}_deterministic_assignment_exact"] is True
        counts = seal["details"][f"{regime}_actual_scene_counts"]
        assert all(v > 0 for v in counts.values())
    complete = protocol / "CALIBRATION_PROTOCOL_COMPLETE.json"
    before = complete.stat().st_mtime_ns
    second = subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True, check=False)
    assert second.returncode == 0, second.stdout + second.stderr
    assert "reusing" in second.stdout
    assert complete.stat().st_mtime_ns == before


def test_protocol_seal_rejects_noncalibration_source_split(tmp_path: Path) -> None:
    near = _dataset(tmp_path, "calibration_near_contact", "val", 20)
    contact = _dataset(tmp_path, "calibration_contact", "calibration", 20)
    safe = _dataset(tmp_path, "calibration_safe", "calibration", 8)
    protocol = tmp_path / "calibration_v48_14_prism_4814"
    # Build is allowed to run on the fixture so the independent seal is the failing layer.
    part = subprocess.run([
        sys.executable, str(ROOT / "tools" / "partition_dedicated_calibration_v48_14.py"),
        "--near", str(near), "--contact", str(contact), "--output-root", str(protocol),
        "--seed", "4814", "--link-mode", "hardlink",
    ], cwd=ROOT, text=True, capture_output=True, check=False)
    assert part.returncode == 0, part.stdout + part.stderr
    out = tmp_path / "seal.json"
    checked = subprocess.run([
        sys.executable, str(ROOT / "tools" / "check_v48_45_protocol_seal.py"),
        "--protocol-root", str(protocol), "--near-source", str(near),
        "--contact-source", str(contact), "--safe-root", str(safe), "--output", str(out),
    ], cwd=ROOT, text=True, capture_output=True, check=False)
    assert checked.returncode != 0
    doc = json.loads(out.read_text())
    assert doc["valid"] is False
    assert doc["checks"]["source_manifests_valid"] is False
    assert doc["test_roots_read"] is False


def test_v48455_launcher_prepares_one_shared_protocol_before_arms() -> None:
    parallel = (ROOT / "scripts" / "run_v48_45_sowr_2x2_parallel.sh").read_text()
    arm = (ROOT / "scripts" / "run_v48_45_sowr_ablation_arm.sh").read_text()
    assert "bash scripts/prepare_v48_45_protocol.sh" in parallel
    assert parallel.index("prepare_v48_45_protocol.sh") < parallel.index('arms=(A B C D)')
    assert "V48_45_PROTOCOL_SEAL.json" in parallel
    assert "export V4845_SKIP_PROTOCOL_PREPARE=1" in parallel
    assert "bash scripts/prepare_v48_45_protocol.sh" in arm
    assert 'PROTOCOL_ROOT="${PROTOCOL_ROOT:-$OCRAP_ROOT/calibration_v48_14_prism_4814}"' in arm
    assert "--skip-sample-file-check" in arm
    assert "V4845_PROTOCOL_SEAL_SHA256" in arm
    dedicated = (ROOT / "scripts" / "run_v48_36_ocaf_dedicated.sh").read_text()
    assert "protocol_seal_sha256" in dedicated
    assert "V4845_PROTOCOL_SEAL_SHA256" in dedicated
    assert "v48.45.5-A-" in arm and "v48.45.5-D-" in arm


def test_v48455_operator_commands_prepare_protocol_before_deleting_or_launching_arms() -> None:
    text = (ROOT / "OC-RAP-v48.45.5-protocol-bootstrap-and-SOWR-run-commands-ZH.txt").read_text()
    prep = text.index("bash scripts/prepare_v48_45_protocol.sh")
    cleanup = text.index('rm -rf \\\n  "$BASE_OUT/ocrap_v48_45_sowr_ablation_A"')
    launch = text.index("bash scripts/run_v48_45_sowr_2x2_parallel.sh")
    assert prep < cleanup < launch
    assert 'CAL_NEAR="$OCRAP_ROOT/calibration_near_contact"' in text
    assert 'CAL_CONTACT="$OCRAP_ROOT/calibration_contact"' in text
    assert 'CAL_SAFE="$OCRAP_ROOT/calibration_safe"' in text
    assert "V4845_PROTOCOL_SEED=4814" in text
    assert "V4845_ADAPT_TRAIN_FRACTION=0.45" in text
    assert "V4845_ADAPT_DEV_FRACTION=0.15" in text
    assert "test_*" in text
