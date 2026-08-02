from __future__ import annotations

from pathlib import Path

from ocrap.cli.train import _finalize_direct_policy_stats
from ocrap.models.ocrap import OCRAPModel

ROOT = Path(__file__).resolve().parents[1]


def _model(mode: str = "safety_slack") -> OCRAPModel:
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
        direct_recovery_evidence_concord=True,
        direct_recovery_evidence_admission_head=True,
        direct_recovery_evidence_admission_scale=2.0,
        direct_recovery_evidence_admission_bounded=True,
        direct_recovery_evidence_admission_prior_mode=mode,
        direct_recovery_evidence_slack_temperature=0.025,
        direct_recovery_evidence_slack_penalty=1.0,
        direct_recovery_evidence_frontier=True,
        direct_recovery_evidence_component_prior_logit=-2.0,
    ).eval()


def test_safety_slack_mode_is_fail_closed_and_persistable() -> None:
    model = _model()
    assert model.direct_recovery_evidence_admission_prior_mode == "safety_slack"
    assert model.direct_recovery_evidence_slack_temperature == 0.025
    assert model.direct_recovery_evidence_slack_penalty == 1.0
    try:
        _model("regime_router")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown prior mode must fail closed")


def test_safety_slack_uses_signed_margin_hinge_not_regime_routing() -> None:
    source = (ROOT / "src" / "ocrap" / "models" / "ocrap.py").read_text()
    block = source[source.index('admission_prior_mode == "safety_slack"'):]
    assert "direct_recovery_evidence_slack_temperature" in block[:2500]
    assert "torch.relu(max_predicted_veto_margin)" in block[:2500]
    assert "prior_benefit" in block[:2500]
    assert "direct_recovery_evidence_admission_prior_detach" in source
    assert "bucket_id" not in block[:1800]
    assert "regime_id" not in block[:1800]


def test_component_margin_regression_restores_distance_to_veto_boundary() -> None:
    source = (ROOT / "src" / "ocrap" / "models" / "losses.py").read_text()
    block = source[source.index("v48.30: BCE identifies the side") :]
    assert "ordinal_evidence_factorized_harm_temperature" in block[:1800]
    assert "F.smooth_l1_loss" in block[:1800]
    assert "factorized_component_margins" in block[:1800]


def _stats(harm: float, false: float, regret: float, recall: float, safe_mass: float) -> dict[str, float]:
    out: dict[str, float] = {}
    for regime in ("near", "contact"):
        count = 10.0
        safe = 2.0
        out[f"group_count_{regime}"] = count
        out[f"positive_count_{regime}"] = safe
        out[f"safe_opportunity_count_{regime}"] = safe
        out[f"soft_safe_group_{regime}"] = safe
        out[f"soft_safe_nll_sum_{regime}"] = 0.2 * count
        out[f"soft_harmful_mass_sum_{regime}"] = harm * count
        out[f"soft_frontier_harmful_mass_sum_{regime}"] = harm * count
        out[f"soft_safe_recall_sum_{regime}"] = recall * safe
        out[f"soft_false_admission_sum_{regime}"] = false * (count - safe)
        out[f"soft_safe_mass_sum_{regime}"] = safe_mass * safe
        out[f"soft_safe_regret_sum_{regime}"] = regret * safe
        out[f"admission_count_{regime}"] = 1.0
        out[f"valid_safe_admission_count_{regime}"] = 1.0
        out[f"invalid_admission_count_{regime}"] = 0.0
        out[f"safe_positive_admission_hit_{regime}"] = 1.0
        out[f"evidence_safe_top1_hit_{regime}"] = 1.0
        out[f"evidence_safe_top1_regret_sum_{regime}"] = regret * safe
    return out


def test_population_checkpoint_metric_penalizes_over_admission() -> None:
    good = _finalize_direct_policy_stats(_stats(0.05, 0.05, 0.02, 0.8, 0.55), {})
    bad = _finalize_direct_policy_stats(_stats(0.45, 0.55, 0.20, 0.4, 0.20), {})
    assert "direct_population_safe_rank_risk" in good
    assert bad["direct_population_safe_rank_risk"] > good["direct_population_safe_rank_risk"]


def test_stage2_uses_natural_population_without_replacement() -> None:
    text = (ROOT / "scripts" / "adapt_ocrap_v48_30_slack_rank_variant.sh").read_text()
    stage2 = text[text.index("# Stage 2") :]
    assert "GROUP_BATCH_STRATIFIED=false" in stage2
    assert "GROUP_BATCHING_REPLACEMENT=false" in stage2
    assert "BEST_METRIC=direct_population_safe_rank_risk" in stage2
    assert "ADMISSION_SETWISE_WEIGHT:-0.10" in stage2


def test_v48_30_ablations_launch_all_eight_concurrently() -> None:
    text = (ROOT / "scripts" / "run_v48_30_parallel_ablations.sh").read_text()
    for name in (
        "A_natural_population_reference",
        "B_add_signed_component_margin",
        "C_add_safety_slack_projection",
        "D_full_slack_rank",
    ):
        assert name in text
    assert "max_concurrent_tasks':8" in text
    assert 'run_task "$group" balanced "$GPU0" &' in text
    assert 'run_task "$group" precision "$GPU1" &' in text
