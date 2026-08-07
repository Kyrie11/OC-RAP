from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from ocrap.models.ocrap import OCRAPModel

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "v4836_metric_contract", ROOT / "tools" / "check_v48_36_metric_calibration_contract.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_METRIC = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_METRIC)
_load_metric_row = _METRIC._load_metric_row


def _model(*, aligned: bool, unbounded_benefit: bool = False, unbounded_harm: bool = False) -> OCRAPModel:
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
        direct_recovery_evidence_benefit_residual_scale=6.0,
        direct_recovery_evidence_unbounded_benefit_factor=unbounded_benefit,
        direct_recovery_evidence_unbounded_harm_factors=unbounded_harm,
        direct_recovery_evidence_component_reliability="1,1,1,0,0",
        direct_recovery_evidence_concord=True,
        direct_recovery_evidence_consensus_prior_scale=0.0,
        direct_recovery_evidence_admission_head=False,
        direct_recovery_evidence_admission_bounded=False,
        direct_recovery_evidence_admission_prior_mode="joint_reserve",
        direct_recovery_evidence_benefit_margin_temperature=0.05,
        direct_recovery_evidence_slack_temperature=0.025,
        direct_recovery_evidence_joint_reserve_temperature=0.05,
        direct_recovery_evidence_reserve_factor_alignment=aligned,
        direct_recovery_evidence_frontier=True,
        direct_recovery_evidence_component_prior_logit=-2.0,
    ).eval()


def _forward(model: OCRAPModel) -> dict[str, torch.Tensor]:
    torch.manual_seed(4839)
    x = torch.randn(6, 12)
    groups = torch.tensor([[0], [0], [0], [1], [1], [1]])
    nominal = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    with torch.no_grad():
        return model(
            x,
            bucket_id=torch.ones(6, dtype=torch.long),
            group_index=groups,
            is_nominal=nominal,
            direct_only=True,
        )


def test_aligned_reserve_preserves_component_semantic_prior() -> None:
    """v48.38 double subtraction cancelled the -2 component safety prior."""
    aligned = _model(aligned=True)
    legacy = _model(aligned=False)
    # Identical zero-initialised network parameters.
    legacy.load_state_dict(aligned.state_dict())
    out_aligned = _forward(aligned)
    out_legacy = _forward(legacy)
    nominal = torch.tensor([True, False, False, True, False, False])
    # At zero harm residual, candidate component logits are the semantic prior -2.
    # Factor supervision therefore sees -0.05 physical margin. Aligned deployment
    # must see exactly the same coordinate; legacy candidate-minus-nominal sees 0.
    aligned_margin = out_aligned["direct_recovery_evidence_predicted_component_margins"][~nominal, 0]
    legacy_margin = out_legacy["direct_recovery_evidence_predicted_component_margins"][~nominal, 0]
    assert torch.allclose(aligned_margin, torch.full_like(aligned_margin, -0.05), atol=1e-6)
    assert torch.allclose(legacy_margin, torch.zeros_like(legacy_margin), atol=1e-6)


def test_unbounded_harm_factor_removes_legacy_physical_margin_ceiling() -> None:
    bounded = _model(aligned=True, unbounded_harm=False)
    unbounded = _model(aligned=True, unbounded_harm=True)
    unbounded.load_state_dict(bounded.state_dict())
    for model in (bounded, unbounded):
        assert model.direct_evidence_concord_harm_calibrator is not None
        projection = model.direct_evidence_concord_harm_calibrator[-1]
        assert isinstance(projection, torch.nn.Linear)
        with torch.no_grad():
            projection.weight.zero_()
            projection.bias.fill_(10.0)
    b = _forward(bounded)["direct_recovery_evidence_predicted_component_margins"][:, 0]
    u = _forward(unbounded)["direct_recovery_evidence_predicted_component_margins"][:, 0]
    # Legacy tanh*6 + prior(-2), times tau_h=.025, asymptotes to 0.10.
    assert float(b.max()) <= 0.10001
    # Unbounded signed regression can represent the observed O(1) teacher margins.
    assert float(u.max()) > 1.0


def test_unbounded_benefit_factor_removes_legacy_headroom_ceiling() -> None:
    bounded = _model(aligned=True, unbounded_benefit=False)
    unbounded = _model(aligned=True, unbounded_benefit=True)
    unbounded.load_state_dict(bounded.state_dict())
    for model in (bounded, unbounded):
        assert model.direct_evidence_concord_benefit_calibrator is not None
        projection = model.direct_evidence_concord_benefit_calibrator[-1]
        assert isinstance(projection, torch.nn.Linear)
        with torch.no_grad():
            projection.weight.zero_()
            projection.bias.fill_(10.0)
    b = _forward(bounded)["direct_recovery_evidence_predicted_benefit_margin"]
    u = _forward(unbounded)["direct_recovery_evidence_predicted_benefit_margin"]
    # With consensus scale zero the bounded benefit correction is <= .75*.05.
    assert float(b.max()) <= 0.03751
    assert float(u.max()) > 2.0


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_zero_update_metric_fixture(tmp_path: Path) -> tuple[Path, Path]:
    factor_model = tmp_path / "factor_stage" / "model_v48_trac_sr"
    final_model = tmp_path / "model_v48_trac_sr"
    factor_model.mkdir(parents=True)
    final_model.mkdir(parents=True)
    factor_ckpt = factor_model / "best.pt"
    factor_ckpt.write_bytes(b"factor-checkpoint-bytes")
    source_summary = factor_model / "train_summary.json"
    source_summary.write_text(json.dumps({
        "best_epoch": 4,
        "best_metric": "direct_factor_supervised_risk",
        "history": [{"epoch": 4, "val": {
            "direct_group_count_near": 110,
            "direct_group_count_contact": 279,
            "direct_safe_opportunity_group_count_near": 8,
            "direct_safe_opportunity_group_count_contact": 17,
        }}],
    }), encoding="utf-8")
    final_ckpt = final_model / "best.pt"
    final_ckpt.write_bytes(factor_ckpt.read_bytes())
    stage_summary = final_model / "train_summary.json"
    stage_summary.write_text(json.dumps({
        "best_epoch": 0,
        "best_metric": "direct_factor_supervised_risk",
        "epochs_completed": 0,
        "total_train_steps": 0,
        "history": [],
        "materialized_without_training": True,
        "parameter_update_performed": False,
        "factor_checkpoint_reused_without_parameter_update": True,
        "source_factor_checkpoint": str(factor_ckpt),
        "source_factor_checkpoint_sha256": _sha(factor_ckpt),
        "metric_source_checkpoint_sha256": _sha(factor_ckpt),
        "metric_source_train_summary": str(source_summary),
        "metric_source_train_summary_sha256": _sha(source_summary),
        "source_factor_best_epoch": 4,
    }), encoding="utf-8")
    return stage_summary, source_summary


def test_metric_contract_accepts_audited_zero_update_stage_without_fake_epoch(tmp_path: Path) -> None:
    stage_summary, _ = _make_zero_update_metric_fixture(tmp_path)
    summary, best, epoch, provenance = _load_metric_row(stage_summary)
    assert epoch == 4
    assert best["epoch"] == 4
    assert summary["best_metric"] == "direct_factor_supervised_risk"
    assert provenance["materialized_without_training"] is True
    assert provenance["kind"] == "factor_training_history_for_zero_update_stage"


def test_metric_contract_fails_closed_on_tampered_factor_summary(tmp_path: Path) -> None:
    stage_summary, source_summary = _make_zero_update_metric_fixture(tmp_path)
    source_summary.write_text(source_summary.read_text() + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="SHA256"):
        _load_metric_row(stage_summary)


def test_v4839_wrappers_are_one_rule_and_clean_2x2_dynamic_range_ablation() -> None:
    main = (ROOT / "scripts" / "run_v48_39_drfr_dedicated.sh").read_text()
    arm = (ROOT / "scripts" / "run_v48_39_drfr_ablation_arm.sh").read_text()
    parallel = (ROOT / "scripts" / "run_v48_39_drfr_ablations_parallel.sh").read_text()
    variant = (ROOT / "scripts" / "adapt_ocrap_v48_36_ocaf_variant.sh").read_text()
    assert 'EVIDENCE_RESERVE_FACTOR_ALIGNMENT="true"' in main
    assert 'EVIDENCE_UNBOUNDED_BENEFIT_FACTOR="true"' in main
    assert 'EVIDENCE_UNBOUNDED_HARM_FACTORS="true"' in main
    assert 'V4838_RFR_RESERVE_ONLY="1"' in main
    assert 'PROPOSAL_TOP_K="5"' in main
    assert 'FACTOR_COMPONENT_UNDERESTIMATION_WEIGHT="0"' in main
    assert 'FACTOR_SAFE_POSITIVE_COMPONENT_OVERESTIMATION_WEIGHT="0"' in main
    assert 'FACTOR_JOINT_RESERVE_REGRESSION_WEIGHT="0"' in main
    assert 'EVIDENCE_ADMISSION_PRIOR_MODE="joint_reserve"' in arm
    for letter in "ABCD":
        assert f'{letter})' in arm
    assert 'EVIDENCE_UNBOUNDED_HARM_FACTORS="true"' in arm
    assert 'EVIDENCE_UNBOUNDED_BENEFIT_FACTOR="true"' in arm
    assert "arms=(A B C)" in parallel
    assert 'wait "${pids[$arm]}"' in parallel
    # Exact physical semantics are part of factor-cache identity.
    assert 'reserve_factor_alignment=${EVIDENCE_RESERVE_FACTOR_ALIGNMENT:-false}' in variant
    assert 'unbounded_benefit_factor=${EVIDENCE_UNBOUNDED_BENEFIT_FACTOR:-false}' in variant
    assert 'unbounded_harm_factors=${EVIDENCE_UNBOUNDED_HARM_FACTORS:-false}' in variant
    # No regime-conditioned mechanism is introduced by the new wrappers.
    for text in (main, arm):
        assert "REGIME_CONDITION" not in text
        assert "NEAR_THRESHOLD" not in text
        assert "CONTACT_THRESHOLD" not in text
