from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from ocrap.algorithms.lcv import _exclusive_cumulative_weights
from ocrap.models.losses import direct_uncertainty_recovery_value_loss

ROOT = Path(__file__).resolve().parents[1]


def test_multigroup_adaptive_margin_preflight_passes(tmp_path: Path) -> None:
    output = tmp_path / "loss_contract.json"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "check_v48_32_1_multigroup_loss_contract.py"), "--output", str(output)],
        check=False,
    )
    doc = json.loads(output.read_text())
    assert proc.returncode == 0
    assert doc["valid"] is True
    assert doc["group_count"] == 2
    assert doc["outer_loop_collisions"] == []


def test_strict_shape_contract_rejects_silent_truncation() -> None:
    common = dict(
        pred_logit=torch.zeros(3), pred_logvar=torch.zeros(2),
        teacher_r_dep=torch.zeros(3), teacher_r_orc=torch.zeros(3),
        teacher_q=torch.ones((3, 1, 1)), root_probs=torch.ones((3, 1)),
        root_valid=torch.ones((3, 1), dtype=torch.bool), option_valid=torch.ones((3, 1), dtype=torch.bool),
        scene_hash=torch.ones(3, dtype=torch.long), time_index=torch.zeros(3, dtype=torch.long),
        macro_type_id=torch.tensor([0, 2, 2]), is_nominal=torch.tensor([1.0, 0.0, 0.0]),
        bucket_id=torch.ones(3, dtype=torch.long), strict_shape_contract=True,
    )
    try:
        direct_uncertainty_recovery_value_loss(**common)
    except ValueError as exc:
        assert "shape contract" in str(exc)
    else:
        raise AssertionError("strict shape mismatch must fail closed")


def test_deterministic_exclusive_prefix_matches_cumsum_cpu() -> None:
    weights = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]])
    expected = torch.cumsum(weights, dim=-1) - weights
    assert torch.allclose(_exclusive_cumulative_weights(weights), expected)


def test_factor_cache_materialization_verifies_sha_and_rewrites_paths(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    model = source / "model_v48_trac_sr"
    model.mkdir(parents=True)
    checkpoint = model / "best.pt"
    checkpoint.write_bytes(b"factor-checkpoint")
    import hashlib
    sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    (source / "STAGE_ARCHITECTURE.json").write_text("{}\n")
    (source / "POLICY_CONTRACT.env").write_text("RISK_SOURCE=ordinal_evidence\n")
    (source / "FACTOR_CACHE_CONTRACT.json").write_text("{}\n")
    for name in ("TRAINING_COMPLETE.json", "EVIDENCE_CORRECTION_COMPLETE.json"):
        (source / name).write_text(json.dumps({"checkpoint": str(checkpoint), "checkpoint_sha256": sha}) + "\n")
    output = tmp_path / "materialize.json"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "materialize_v48_32_1_factor_cache.py"),
         "--source-stage", str(source), "--destination-stage", str(destination), "--output", str(output)],
        check=False,
    )
    assert proc.returncode == 0
    report = json.loads(output.read_text())
    assert report["valid"] is True
    metadata = json.loads((destination / "TRAINING_COMPLETE.json").read_text())
    assert metadata["checkpoint"] == str(destination.resolve() / "model_v48_trac_sr" / "best.pt")
    assert metadata["checkpoint_sha256"] == sha
    assert metadata["factor_cache_reused"] is True


def test_factor_cache_materialization_rejects_corrupt_metadata(tmp_path: Path) -> None:
    source = tmp_path / "source"
    model = source / "model_v48_trac_sr"
    model.mkdir(parents=True)
    (model / "best.pt").write_bytes(b"factor-checkpoint")
    (source / "STAGE_ARCHITECTURE.json").write_text("{}\n")
    (source / "POLICY_CONTRACT.env").write_text("x=1\n")
    (source / "FACTOR_CACHE_CONTRACT.json").write_text("{}\n")
    for name in ("TRAINING_COMPLETE.json", "EVIDENCE_CORRECTION_COMPLETE.json"):
        (source / name).write_text(json.dumps({"checkpoint_sha256": "bad"}) + "\n")
    output = tmp_path / "materialize.json"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "materialize_v48_32_1_factor_cache.py"),
         "--source-stage", str(source), "--destination-stage", str(tmp_path / "destination"), "--output", str(output)],
        check=False,
    )
    assert proc.returncode == 30
    assert json.loads(output.read_text())["valid"] is False


def test_dedicated_controller_runs_loss_preflight_before_training_and_supports_exact_factor_reuse() -> None:
    script = (ROOT / "scripts" / "run_v48_32_1_rc30_integrity_hotfix_dedicated.sh").read_text()
    preflight = script.index("check_v48_32_1_multigroup_loss_contract.py")
    training = script.index("run_variant()")
    assert preflight < training
    assert "V48321_FACTOR_CACHE_BALANCED" in script
    assert "V48321_FACTOR_CACHE_PRECISION" in script
    assert "adapt_ocrap_v48_32_1_identity_utility_variant.sh" in script


def test_certificate_rc30_does_not_claim_natural_gate_evaluated() -> None:
    script = (ROOT / "scripts" / "calibrate_v48_32_1_certificate_pool.sh").read_text()
    assert "certificate_data_valid=(near_rc in (0,3) and contact_rc in (0,3))" in script
    assert "'gate_evaluated':certificate_data_valid" in script
    assert "if reason=='natural_gate_failed' else False" in script


def test_controller_completion_separates_pipeline_and_certificate_exit_codes() -> None:
    script = (ROOT / "scripts" / "run_v48_32_1_rc30_integrity_hotfix_dedicated.sh").read_text()
    assert "'certificate_executed':certificate_executed" in script
    assert "'raw_certificate_exit_code':raw_rc if certificate_executed else None" in script
    assert "'pipeline_exit_code':30" in script


def test_certificate_metric_contract_tool_is_packaged() -> None:
    tool = ROOT / "tools" / "check_v48_32_1_metric_calibration_contract.py"
    assert tool.is_file()
    script = (ROOT / "scripts" / "calibrate_v48_32_1_certificate_pool.sh").read_text()
    assert "tools/check_v48_32_1_metric_calibration_contract.py" in script
