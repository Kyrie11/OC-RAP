from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from ocrap.models.ocrap import OCRAPModel
from test_v48_23_frontier_bridge import _loss_common

ROOT = Path(__file__).resolve().parents[1]


def _joint_model() -> OCRAPModel:
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
        direct_recovery_evidence_component_scale=6.0,
        direct_recovery_evidence_component_reliability="1,1,1,0,0",
        direct_recovery_evidence_concord=True,
        direct_recovery_evidence_admission_head=False,
        direct_recovery_evidence_admission_bounded=False,
        direct_recovery_evidence_admission_prior_mode="joint_reserve",
        direct_recovery_evidence_benefit_margin_temperature=0.05,
        direct_recovery_evidence_slack_temperature=0.025,
        direct_recovery_evidence_joint_reserve_temperature=0.05,
        direct_recovery_evidence_frontier=True,
        direct_recovery_evidence_component_prior_logit=-2.0,
    ).eval()


def test_joint_reserve_is_exact_noncompensatory_and_regime_free() -> None:
    torch.manual_seed(4838)
    model = _joint_model()
    x = torch.randn(6, 12)
    groups = torch.tensor([[0], [0], [0], [1], [1], [1]])
    nominal = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    with torch.no_grad():
        near = model(
            x, bucket_id=torch.ones(6, dtype=torch.long), group_index=groups,
            is_nominal=nominal, direct_only=True,
        )
        contact = model(
            x, bucket_id=torch.full((6,), 2, dtype=torch.long), group_index=groups,
            is_nominal=nominal, direct_only=True,
        )
    for out in (near, contact):
        benefit = out["direct_recovery_evidence_predicted_benefit_margin"]
        safety = out["direct_recovery_evidence_predicted_safety_headroom"]
        component_margins = out["direct_recovery_evidence_predicted_component_margins"]
        reserve = out["direct_recovery_evidence_joint_reserve_margin"]
        # The support contract is [1,1,1,0,0]; reliability-zero coordinates are
        # not allowed to clamp the learned safety reserve to zero.
        assert torch.allclose(safety, -component_margins[:, :3].amax(dim=-1), atol=1e-7)
        assert torch.allclose(reserve, torch.minimum(benefit, safety), atol=1e-7)
        assert torch.all(reserve <= benefit + 1e-7)
        assert torch.all(reserve <= safety + 1e-7)
        assert torch.allclose(
            out["direct_recovery_admission_probability"],
            torch.sigmoid(reserve / 0.05), atol=1e-7,
        )
        assert torch.allclose(reserve[nominal > 0.5], torch.zeros(2), atol=1e-7)
    # The joint reserve is built only from concord benefit/component factors. The
    # legacy bucket ID may still exist elsewhere for backward compatibility, but
    # changing it cannot switch this shared physical reserve into another policy.
    assert torch.allclose(
        near["direct_recovery_evidence_joint_reserve_margin"],
        contact["direct_recovery_evidence_joint_reserve_margin"], atol=1e-7,
    )


def test_false_safe_component_underestimation_gets_one_sided_gradient() -> None:
    component = torch.zeros((3, 5), requires_grad=True)
    loss = _loss_common(
        pred_opportunity_logit=torch.zeros(3),
        pred_harm_logit=torch.zeros(3),
        pred_component_harm_logits=component,
        teacher_hard_violation=torch.tensor([0.0, 1.0, 0.0]),
        ordinal_evidence_ordered_nll_all_weight=1e-12,
        ordinal_evidence_component_underestimation_weight=1.0,
        ordinal_evidence_component_reliability="1,1,1,1,1",
    )
    loss.backward()
    assert component.grad is not None
    # Candidate 1 has a true hard-veto violation. Negative gradient means SGD
    # increases its predicted component margin, correcting the false-safe tail.
    assert component.grad[1, 3].item() < 0.0
    assert abs(component.grad[2, 3].item()) < 1e-8


def test_safe_positive_component_overestimate_is_pushed_down() -> None:
    component = torch.tensor(
        [[0.0] * 5, [0.0] * 5, [5.0] * 5], requires_grad=True
    )
    loss = _loss_common(
        pred_opportunity_logit=torch.zeros(3),
        pred_harm_logit=torch.zeros(3),
        pred_component_harm_logits=component,
        ordinal_evidence_ordered_nll_all_weight=1e-12,
        ordinal_evidence_safe_positive_component_overestimation_weight=1.0,
        ordinal_evidence_component_reliability="1,1,1,1,1",
    )
    loss.backward()
    assert component.grad is not None
    # Candidate 2 is the safe-beneficial recovery in this fixture. Positive
    # gradient lowers its inflated harm logits under gradient descent.
    assert bool((component.grad[2] > 0.0).all())
    # Harmful/non-safe candidate 1 must not receive this safe-positive correction.
    assert torch.allclose(component.grad[1], torch.zeros_like(component.grad[1]), atol=1e-8)


def test_joint_reserve_regression_pushes_safe_gain_across_shared_boundary() -> None:
    component = torch.zeros((3, 5), requires_grad=True)
    opportunity = torch.zeros(3, requires_grad=True)
    loss = _loss_common(
        pred_opportunity_logit=opportunity,
        pred_harm_logit=torch.zeros(3),
        pred_component_harm_logits=component,
        ordinal_evidence_ordered_nll_all_weight=1e-12,
        ordinal_evidence_joint_reserve_regression_weight=1.0,
        ordinal_evidence_joint_reserve_boundary_weight=2.0,
        ordinal_evidence_joint_reserve_boundary_width=0.05,
        ordinal_evidence_benefit_margin_temperature=0.05,
        ordinal_evidence_component_reliability="1,1,1,1,1",
    )
    loss.backward()
    assert opportunity.grad is not None
    # Candidate 2 is safe-beneficial; the reserve target raises its benefit
    # headroom rather than learning a regime-specific admission residual.
    assert opportunity.grad[2].item() < 0.0


def _save_checkpoint(path: Path, state: dict[str, torch.Tensor]) -> None:
    torch.save({"model_state": state}, path)


def test_stage_transfer_accepts_byte_identical_skipped_identity(tmp_path: Path) -> None:
    state = {
        "direct_evidence_concord_benefit_calibrator.0.weight": torch.ones(2),
        "direct_evidence_concord_harm_calibrator.0.weight": torch.ones(2),
        "direct_evidence_interaction_bridge.action_raw.weight": torch.ones(2),
        "encoder.weight": torch.ones(2),
    }
    factor, identity, final = [tmp_path / n for n in ("factor.pt", "identity.pt", "final.pt")]
    for p in (factor, identity, final):
        _save_checkpoint(p, {k: v.clone() for k, v in state.items()})
    arch = {
        "version": "v48.38-RFR",
        "algorithm_variant": "v48.38-RFR-test",
        "trainable": [],
        "context_source": "physical_interaction",
        "observation_conditioned_action_frontier": True,
        "interaction_bridge_trainable_this_stage": False,
        "identity_stage_skipped": True,
        "regime_id_exposed_to_evidence_model": False,
        "shared_deployment_rule_required": True,
    }
    identity_arch = tmp_path / "identity.json"
    final_arch = tmp_path / "final.json"
    identity_arch.write_text(json.dumps(arch), encoding="utf-8")
    final_arch.write_text(json.dumps(arch), encoding="utf-8")
    output = tmp_path / "transfer.json"
    cp = subprocess.run(
        [
            sys.executable, str(ROOT / "tools" / "check_v48_36_stage_transfer.py"),
            "--factor", str(factor), "--identity", str(identity), "--final", str(final),
            "--identity-architecture", str(identity_arch),
            "--final-architecture", str(final_arch),
            "--identity-allowed-prefixes", "", "--final-allowed-prefixes", "",
            "--identity-stage-skipped", "--final-stage-disabled",
            "--implementation-version", "v48.38-RFR-test", "--output", str(output),
        ], cwd=ROOT, check=False,
    )
    doc = json.loads(output.read_text(encoding="utf-8"))
    assert cp.returncode == 0
    assert doc["valid"] is True
    assert doc["identity_stage_skipped"] is True
    assert doc["identity_allowed_changed_parameter_count"] == 0
    assert doc["identity_disallowed_changed_parameter_count"] == 0



def test_materializer_records_zero_training_instead_of_copying_factor_history(tmp_path: Path) -> None:
    factor = tmp_path / "factor"
    model = factor / "model_v48_trac_sr"
    model.mkdir(parents=True)
    torch.save({"model_state": {"w": torch.ones(1)}}, model / "best.pt")
    (model / "train_summary.json").write_text(json.dumps({
        "best_metric": "direct_factor_supervised_risk", "best_metric_value": 1.25,
        "best_epoch": 7, "epochs_completed": 9,
        "history": [{"epoch": 1, "val": {"direct_factor_supervised_risk": 2.0}}],
    }), encoding="utf-8")
    (factor / "STAGE_ARCHITECTURE.json").write_text(json.dumps({
        "admission_prior_mode": "joint_reserve", "admission_head": False,
        "deterministic_joint_reserve": True, "regime_id_exposed_to_evidence_model": False,
        "shared_deployment_rule_required": True, "context_source": "physical_interaction",
        "observation_conditioned_action_frontier": True, "trainable": ["factor"],
    }), encoding="utf-8")
    (factor / "TRAINING_COMPLETE.json").write_text(json.dumps({
        "best_metric": "direct_factor_supervised_risk", "best_epoch": 7,
    }), encoding="utf-8")
    (factor / "EVIDENCE_CORRECTION_COMPLETE.json").write_text(json.dumps({
        "event": "factor", "trainable_prefixes": ["factor"],
        "regime_id_exposed_to_evidence_model": False,
    }), encoding="utf-8")
    (factor / "POLICY_CONTRACT.env").write_text("ADMISSION_LABEL_MODE=deterministic_joint_physical_reserve\n", encoding="utf-8")
    dest = tmp_path / "identity"
    cp = subprocess.run([
        sys.executable, str(ROOT / "tools" / "materialize_v48_38_reserve_stage.py"),
        "--factor-stage", str(factor), "--destination", str(dest),
        "--role", "identity", "--implementation-version", "test",
    ], cwd=ROOT, check=False)
    assert cp.returncode == 0
    complete = json.loads((dest / "TRAINING_COMPLETE.json").read_text())
    summary = json.loads((dest / "model_v48_trac_sr" / "train_summary.json").read_text())
    correction = json.loads((dest / "EVIDENCE_CORRECTION_COMPLETE.json").read_text())
    assert complete["best_epoch"] == 0 and complete["epochs_completed"] == 0
    arch = json.loads((dest / "STAGE_ARCHITECTURE.json").read_text())
    assert arch["trainable"] == []
    assert complete["materialized_without_training"] is True
    assert summary["history"] == [] and summary["total_train_steps"] == 0
    assert summary["checkpoint_materialization"] in {"hardlink", "copy"}
    assert {p.name for p in (dest / "model_v48_trac_sr").iterdir()} == {"best.pt", "train_summary.json"}
    assert summary["initial_checkpoint"]["direct_factor_supervised_risk"] == 1.25
    assert summary["metric_source_train_summary"] == str(model / "train_summary.json")
    assert len(summary["metric_source_train_summary_sha256"]) == 64
    assert summary["metric_source_checkpoint_sha256"] == summary["source_factor_checkpoint_sha256"]
    assert correction["trainable_prefixes"] == []
    assert correction["parameter_update_performed"] is False

def test_rfr_wrappers_are_fail_closed_regime_free_and_parallel() -> None:
    main = (ROOT / "scripts" / "run_v48_38_rfr_dedicated.sh").read_text()
    arm = (ROOT / "scripts" / "run_v48_38_rfr_ablation_arm.sh").read_text()
    parallel = (ROOT / "scripts" / "run_v48_38_rfr_ablations_parallel.sh").read_text()
    variant = (ROOT / "scripts" / "adapt_ocrap_v48_36_ocaf_variant.sh").read_text()
    losses = (ROOT / "src" / "ocrap" / "models" / "losses.py").read_text()
    assert 'EVIDENCE_ADMISSION_PRIOR_MODE="joint_reserve"' in main
    assert 'V4838_RFR_RESERVE_ONLY="1"' in main
    assert 'PROPOSAL_TOP_K="5"' in main
    assert 'RESUME_AFTER_ADAPTATION="0"' in main
    assert 'RESUME_AFTER_ADAPTATION="0"' in arm
    assert "FACTOR_COMPONENT_UNDERESTIMATION_WEIGHT" in main
    assert "FACTOR_SAFE_POSITIVE_COMPONENT_OVERESTIMATION_WEIGHT" in main
    assert "FACTOR_JOINT_RESERVE_REGRESSION_WEIGHT" in main
    assert "materialize_v48_38_reserve_stage.py" in variant
    rfr_block = losses[losses.index("# v48.38 RFR:") : losses.index("if admission_delta_logits is not None", losses.index("# v48.38 RFR:"))]
    assert "bucket_id" not in rfr_block
    assert "regime_id" not in rfr_block
    for letter in "ABCD":
        assert f'{letter})' in arm
    assert "arms=(A B C)" in parallel
    assert "wait \"${pids[$arm]}\"" in parallel
    assert "OMP_NUM_THREADS" in parallel and "NUM_WORKERS" in parallel


def test_joint_reserve_ignores_zero_reliability_coordinates_at_inference() -> None:
    """A neutral unsupported component must not force all reserve values <= 0."""
    import torch
    from ocrap.models.ocrap import OCRAPModel

    # Exercise the exact masking invariant from source as a targeted contract;
    # the full model geometry is covered by the existing forward test above.
    source = (ROOT / "src" / "ocrap" / "models" / "ocrap.py").read_text()
    assert "reserve_supported" in source
    assert "predicted_component_margins.new_tensor(-1.0e6)" in source
    reliability = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0])
    margins = torch.tensor([[-0.20, -0.10, -0.05, 0.0, 0.0]])
    supported = reliability > 0
    masked = torch.where(supported.unsqueeze(0), margins, margins.new_tensor(-1.0e6))
    safety_headroom = -masked.amax(dim=-1)
    assert float(safety_headroom.item()) > 0.0


def test_safe_positive_tail_mask_uses_full_teacher_veto() -> None:
    """Unsupported learned coordinates cannot relabel a teacher-harmful row safe-positive."""
    source = (ROOT / "src" / "ocrap" / "models" / "losses.py").read_text()
    assert "target_full_worst_margin_for_tail = target_component_margins.amax(dim=-1)" in source
    assert "target_full_worst_margin_for_tail <= 0.0" in source
