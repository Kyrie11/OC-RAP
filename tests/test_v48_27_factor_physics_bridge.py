from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import torch

from ocrap.models.ocrap import OCRAPModel
from ocrap.simulation.closed_loop_runner import (
    _aggregate_scene_results,
    _canonical_womd_scene_id,
)

ROOT = Path(__file__).resolve().parents[1]


def _model() -> OCRAPModel:
    return OCRAPModel(
        input_dim=12,
        num_roots=2,
        num_options=3,
        d_model=8,
        d_obs=4,
        encoder_type="mlp",
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        direct_recovery_value_head=True,
        direct_recovery_value_output="score",
        direct_recovery_relative_features_include_absolute=False,
        direct_recovery_set_tournament=True,
        direct_recovery_set_tournament_hidden=16,
        direct_recovery_set_tournament_heads=2,
        direct_recovery_set_tournament_dropout=0.0,
        direct_recovery_set_tournament_replace_base=True,
        direct_recovery_delta_head=True,
        direct_recovery_delta_regime_experts=True,
        direct_recovery_delta_policy_features=True,
        direct_recovery_delta_hidden=16,
        direct_recovery_delta_dropout=0.0,
        direct_recovery_delta_mode="ordinal_evidence",
        direct_recovery_evidence_calibrator=True,
        direct_recovery_evidence_calibrator_hidden=12,
        direct_recovery_evidence_calibrator_scale=0.75,
        direct_recovery_evidence_calibrator_mode="dual_tail_context",
        direct_recovery_evidence_calibrator_context=True,
        direct_recovery_evidence_calibrator_context_detach=True,
        direct_recovery_evidence_calibrator_context_source="tournament",
        direct_recovery_evidence_unified_experts=True,
        direct_recovery_evidence_component_heads=True,
        direct_recovery_evidence_component_count=5,
        direct_recovery_evidence_component_scale=4.0,
        direct_recovery_evidence_concord=True,
        direct_recovery_evidence_consensus_disagreement_penalty=0.15,
        direct_recovery_evidence_admission_head=True,
        direct_recovery_evidence_admission_scale=1.0,
        direct_recovery_evidence_admission_bounded=True,
        direct_recovery_evidence_frontier=True,
        direct_recovery_evidence_component_prior_logit=-2.0,
    ).eval()


def test_five_noncompensatory_harm_heads_are_executable_and_prior_centered() -> None:
    model = _model()
    assert model.direct_recovery_evidence_component_count == 5
    assert model.direct_evidence_concord_harm_calibrator[-1].out_features == 5
    torch.manual_seed(4827)
    x = torch.randn(6, 12)
    groups = torch.tensor([[0], [0], [0], [1], [1], [1]])
    nominal = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    with torch.no_grad():
        out = model(
            x,
            bucket_id=torch.ones(6, dtype=torch.long),
            group_index=groups,
            is_nominal=nominal,
            direct_only=True,
        )
    component = out["direct_recovery_evidence_component_harm_logits"]
    assert component.shape == (6, 5)
    recovery = nominal < 0.5
    assert torch.allclose(component[recovery], torch.full_like(component[recovery], -2.0))
    assert torch.allclose(
        out["direct_recovery_admission_logit"][recovery],
        out["direct_recovery_evidence_benefit_logit"][recovery],
        atol=1e-7,
    )


def _constructor_keywords(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "OCRAPModel"
    ]
    assert calls
    return {kw.arg for kw in calls[-1].keywords if kw.arg}


def test_train_inference_and_checkpoint_persist_component_count() -> None:
    required = {"direct_recovery_evidence_component_count"}
    assert required <= _constructor_keywords(ROOT / "src" / "ocrap" / "cli" / "train.py")
    assert required <= _constructor_keywords(ROOT / "src" / "ocrap" / "models" / "inference.py")
    train = (ROOT / "src" / "ocrap" / "cli" / "train.py").read_text()
    assert '"direct_recovery_evidence_component_count"' in train


def test_safe_utility_listwise_and_frontier_use_deployed_score_scale() -> None:
    source = (ROOT / "src" / "ocrap" / "models" / "losses.py").read_text()
    assert "deployed_safe_utility = (" in source
    assert "torch.sigmoid(admission_delta_logits[deployment_idx]) - 0.5" in source
    assert "safe_student = torch.cat([" in source
    assert "deployed_safe_utility" in source[source.index("safe_utility_listwise_weight"):]
    assert "safe_logits = deployed_safe_utility[frontier_safe]" in source
    assert "bad_logits = deployed_safe_utility[frontier_bad]" in source
    assert "factorized_component_margins[:, :3]" not in source
    assert "component_harm_logits[recs, :3]" not in source


def test_raw_benefit_learning_and_safe_gate_are_explicitly_separated() -> None:
    adapt = (ROOT / "scripts" / "adapt_ocrap_v48_27_factor_physics_variant.sh").read_text()
    certificate = (ROOT / "scripts" / "calibrate_v48_27_certificate_pool.sh").read_text()
    assert "ORDINAL_EVIDENCE_SAFE_BENEFIT_TARGET=false" in adapt
    assert 'export OPPORTUNITY_LABEL_MODE="${OPPORTUNITY_LABEL_MODE:-raw_benefit}"' in certificate
    assert "GATE_POSITIVE_MODE" in certificate
    assert "--gate-positive-mode=\"$GATE_POSITIVE_MODE\"" in certificate


def test_closed_loop_target_matching_ignores_loader_suffix() -> None:
    assert _canonical_womd_scene_id("waymax_abc__wx00011519") == "waymax_abc"
    assert _canonical_womd_scene_id("waymax_abc") == "waymax_abc"


def test_empty_closed_loop_metrics_are_invalid_not_zero_evidence() -> None:
    out = _aggregate_scene_results([], "v48", "model")
    assert out["num_scenes"] == 0
    assert out["metrics_valid"] is False
    assert out["empty_reason"] == "no_closed_loop_scenes"
    assert out["minimum_clearance_m"] is None
    assert out["minimum_ttc_s"] is None
    assert out["secondary_overlap_scene_rate"] is None


def test_shadow_scan_is_complete_strict_and_repairable() -> None:
    runner = (ROOT / "scripts" / "run_ocrap_v48_trac_sr.sh").read_text()
    shadow = (ROOT / "scripts" / "run_v48_27_dev_shadow_closed_loop.sh").read_text()
    assert "closed_loop.require_bucket_targets=true" in runner
    assert "closed_loop.raw_max_scenarios=${DEV_SHADOW_RAW_MAX_SCENARIOS:-0}" in runner
    assert "DEV_SHADOW_WOMD_SOURCE" in runner
    assert "compare_paired_closed_loop.py" in shadow
    assert (ROOT / "scripts" / "repair_v48_26_dev_shadow_with_v48_27.sh").is_file()


def test_factor_then_admission_training_prevents_sparse_gradient_corruption() -> None:
    staged = (ROOT / "scripts" / "adapt_ocrap_v48_27_factor_physics_variant.sh").read_text()
    assert "Stage 1: dense raw-benefit ordering + five non-compensatory harm factors" in staged
    assert "EVIDENCE_COMPONENT_COUNT=5" in staged
    assert "EVIDENCE_ADMISSION_HEAD=false" in staged
    assert "SETWISE_W=0" in staged
    assert "SELECTIVE_RISK_WEIGHT=0 SELECTIVE_COVERAGE_WEIGHT=0" in staged
    assert "EVIDENCE_TRAINABLE_PREFIXES_OVERRIDE=direct_evidence_concord_admission_calibrator" in staged
    assert 'SETWISE_W="${ADMISSION_SETWISE_WEIGHT:-0.50}"' in staged
    assert "EVIDENCE_ADMISSION_BOUNDED=true" in staged


def test_ablation_scheduler_balances_four_waves_across_two_gpus() -> None:
    text = (ROOT / "scripts" / "run_v48_27_parallel_ablations.sh").read_text()
    for name in (
        "A_three_factor_joint",
        "B_five_factor_joint",
        "C_five_factor_two_stage_regression",
        "D_full_factor_physics_bridge",
    ):
        assert name in text
    assert 'run_task "$group" balanced "$GPU0"' in text
    assert 'run_task "$group" precision "$GPU1"' in text
