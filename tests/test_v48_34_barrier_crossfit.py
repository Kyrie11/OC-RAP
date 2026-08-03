from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _run(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    merged = dict(os.environ)
    merged["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + merged.get("PYTHONPATH", "")
    if env:
        merged.update(env)
    return subprocess.run(args, cwd=ROOT, env=merged, text=True, capture_output=True, check=False)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v48_34_multigroup_barrier_boundary_gradients(tmp_path: Path) -> None:
    out = tmp_path / "contract.json"
    proc = _run(sys.executable, "tools/check_v48_34_multigroup_loss_contract.py", "--output", str(out))
    assert proc.returncode == 0, proc.stderr
    doc = json.loads(out.read_text())
    assert doc["valid"] is True
    assert doc["group_count"] == 2
    assert doc["hard_boundary_continuation"] is True
    assert doc["admission_gradient_l1"] > 0
    assert doc["opportunity_gradient_l1"] > 0
    assert doc["harm_gradient_l1"] > 0
    assert doc["outer_loop_collisions"] == []


def test_barrier_prior_gates_benefit_and_residual() -> None:
    text = (ROOT / "src/ocrap/models/ocrap.py").read_text()
    assert '"barrier_gated_slack"' in text
    assert "residual_safety_gate * prior_benefit" in text
    assert "admission_residual = residual_safety_gate * admission_residual" in text
    assert "softplus" in text


def test_lexicographic_checkpoint_precedes_soft_risk() -> None:
    text = (ROOT / "src/ocrap/cli/train.py").read_text()
    start = text.index('if metric_name != "direct_contract_lexicographic"')
    block = text[start:text.index("def _checkpoint_display_metric", start)]
    assert block.index("direct_contract_zero_valid_safe_admission_regimes") < block.index("direct_contract_safe_rank_risk")
    assert block.index("direct_integrity_invalid_admission_max") < block.index("direct_contract_safe_rank_risk")
    assert block.index("direct_contract_safe_top1_recall_fold_min") < block.index("direct_contract_safe_rank_risk")


def test_factor_cache_reuse_ignores_stage2_settings_but_checks_semantics(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"; source.write_bytes(b"checkpoint")
    group = tmp_path / "group.jsonl"; group.write_text('{"x":1}\n')
    val = tmp_path / "val.jsonl"; val.write_text('{"x":2}\n')
    support = tmp_path / "support.json"
    support.write_text(json.dumps({
        "version": "x", "semantic": "global", "num_rows": 2, "num_groups": 2,
        "num_eligible_candidates": 1, "component_tolerances": {}, "eligibility": {},
        "components": {}, "component_order": ["drs"], "reliability": [1.0],
        "independent_measured_hard_veto_preserved": True, "skipped": False,
    }))
    contract = tmp_path / "contract.json"
    base = [
        sys.executable, "tools/manage_v48_32_factor_cache.py",
        "--source-checkpoint", str(source), "--group-index", str(group),
        "--validation-group-index", str(val), "--support-contract", str(support),
        "--train-mix", "train", "--validation-mix", "val", "--variant", "balanced",
        "--contract", str(contract),
    ]
    create = _run(*base[:2], "--mode", "create", *base[2:], "--setting", "batch_size=96")
    assert create.returncode == 0, create.stderr
    verify = _run(*base[:2], "--mode", "verify-reuse", *base[2:], "--setting", "batch_size=72")
    assert verify.returncode == 0, verify.stderr
    group.write_text('{"x":changed}\n')
    reject = _run(*base[:2], "--mode", "verify-reuse", *base[2:], "--setting", "batch_size=72")
    assert reject.returncode == 30


def test_factor_cache_materialization_preserves_checkpoint_and_rewrites_metadata(tmp_path: Path) -> None:
    src = tmp_path / "source_stage"; dst = tmp_path / "destination_stage"
    (src / "model_v48_trac_sr").mkdir(parents=True)
    best = src / "model_v48_trac_sr/best.pt"; best.write_bytes(b"tensor-bytes")
    sha = _sha(best)
    for name in ("TRAINING_COMPLETE.json", "EVIDENCE_CORRECTION_COMPLETE.json"):
        (src / name).write_text(json.dumps({"checkpoint": str(best), "checkpoint_sha256": sha}))
    (src / "FACTOR_CACHE_CONTRACT.json").write_text(json.dumps({"contract_sha256": "diagnostic"}))
    out = tmp_path / "materialization.json"
    proc = _run(
        sys.executable, "tools/materialize_v48_34_factor_cache.py",
        "--source-stage", str(src), "--destination-stage", str(dst), "--output", str(out),
    )
    assert proc.returncode == 0, proc.stderr
    assert _sha(dst / "model_v48_trac_sr/best.pt") == sha
    metadata = json.loads((dst / "TRAINING_COMPLETE.json").read_text())
    assert str(dst) in metadata["checkpoint"]
    assert str(src) not in metadata["checkpoint"]
    assert json.loads(out.read_text())["checkpoint_identity_preserved"] is True


def test_calibration_emits_legacy_and_exact_eligible_diagnostics() -> None:
    text = (ROOT / "tools/calibrate_policy_risk_v48.py").read_text()
    assert "legacy_evidence_only" in text
    assert "proposal_exact_eligible_top1" in text
    assert "proposal_rows_output" in text


def test_exploratory_script_requires_explicit_rc20_and_test_authorization() -> None:
    text = (ROOT / "scripts/run_v48_34_exploratory_closed_loop_baselines_and_videos.sh").read_text()
    assert 'ALLOW_DIAGNOSTIC_RC20' in text
    assert 'ALLOW_HELDOUT_TEST_DIAGNOSTIC' in text
    assert 'paper_claim_allowed' in text
    assert 'build_v48_34_paired_baseline_report.py' in text
    assert 'select_critical_scenes_v48_34.py' in text
    assert 'render_critical_scenes_v48_34.py' in text


def test_target_subset_requires_exact_unambiguous_target(tmp_path: Path) -> None:
    root = tmp_path / "dataset"; (root / "samples").mkdir(parents=True)
    sample = root / "samples/a.npz"; sample.write_bytes(b"x")
    (root / "manifest.csv").write_text(
        "path,original_scenario_id,target_time_index,target_key\n"
        "samples/a.npz,scene-a,10,bucket:scene-a:t10\n"
    )
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"selected": [{"target_key": "bucket:scene-a:t10", "scene_id": "scene-a", "target_time_index": 10}]}))
    out = tmp_path / "subset"
    proc = _run(
        sys.executable, "tools/subset_dataset_targets_v48_34.py",
        "--input", str(root), "--selection", str(selection), "--output", str(out),
    )
    assert proc.returncode == 0, proc.stderr
    doc = json.loads((out / "TARGET_SUBSET_PROVENANCE.json").read_text())
    assert doc["output_targets"] == 1
    assert (out / "samples/a.npz").exists()
