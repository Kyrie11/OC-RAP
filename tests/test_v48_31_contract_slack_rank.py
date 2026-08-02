from __future__ import annotations

from pathlib import Path

import numpy as np

from ocrap.cli.train import _finalize_direct_policy_stats
from ocrap.models.data import _nominal_deviation_by_path
from ocrap.models.ocrap import OCRAPModel

ROOT = Path(__file__).resolve().parents[1]


def _model(reliability: str = "1,0.5,1,0,0") -> OCRAPModel:
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
        direct_recovery_evidence_component_reliability=reliability,
        direct_recovery_evidence_concord=True,
        direct_recovery_evidence_admission_head=True,
        direct_recovery_evidence_admission_scale=2.0,
        direct_recovery_evidence_admission_bounded=True,
        direct_recovery_evidence_admission_prior_mode="safety_slack",
        direct_recovery_evidence_slack_temperature=0.025,
        direct_recovery_evidence_slack_penalty=1.0,
        direct_recovery_evidence_frontier=True,
        direct_recovery_evidence_component_prior_logit=-2.0,
    ).eval()


def _stats(*, safe_hits: float, valid_safe: float, invalid: float) -> dict[str, float]:
    stats: dict[str, float] = {}
    for regime in ("near", "contact"):
        count = 10.0
        safe = 2.0
        stats[f"group_count_{regime}"] = count
        stats[f"positive_count_{regime}"] = safe
        stats[f"safe_opportunity_count_{regime}"] = safe
        stats[f"soft_safe_group_{regime}"] = safe
        stats[f"soft_safe_nll_sum_{regime}"] = 1.0
        stats[f"soft_harmful_mass_sum_{regime}"] = 0.2
        stats[f"soft_frontier_harmful_mass_sum_{regime}"] = 0.1
        stats[f"soft_safe_recall_sum_{regime}"] = 1.0
        stats[f"soft_false_admission_sum_{regime}"] = 0.1
        stats[f"soft_safe_mass_sum_{regime}"] = 1.0
        stats[f"soft_safe_regret_sum_{regime}"] = 0.1
        stats[f"admission_count_{regime}"] = valid_safe + invalid
        stats[f"valid_safe_admission_count_{regime}"] = valid_safe
        stats[f"invalid_admission_count_{regime}"] = invalid
        stats[f"safe_positive_admission_hit_{regime}"] = valid_safe
        stats[f"evidence_safe_top1_hit_{regime}"] = safe_hits
        stats[f"evidence_safe_top1_regret_sum_{regime}"] = 0.1
    return stats


def test_component_support_reliability_is_global_and_persisted() -> None:
    model = _model()
    assert model.direct_recovery_evidence_component_reliability == (1.0, 0.5, 1.0, 0.0, 0.0)
    source = (ROOT / "src" / "ocrap" / "models" / "ocrap.py").read_text()
    block = source[source.index("v48.31 CONTRACT-SLACK-RANK") :]
    assert "effective_component_harm_logits" in block[:1800]
    assert "bucket_id" not in block[:1800]
    assert "regime_id" not in block[:1800]


def test_invalid_switches_do_not_defeat_exact_all_abstain_barrier() -> None:
    invalid_only = _finalize_direct_policy_stats(_stats(safe_hits=0.0, valid_safe=0.0, invalid=2.0), {})
    safe_active = _finalize_direct_policy_stats(_stats(safe_hits=1.0, valid_safe=1.0, invalid=0.0), {})
    assert invalid_only["direct_raw_admission_rate_near"] > 0.0
    assert invalid_only["direct_integrity_all_abstain"] == 1.0
    assert safe_active["direct_integrity_all_abstain"] == 0.0


def test_contract_checkpoint_metric_requires_safe_top1_support() -> None:
    zero_top1 = _finalize_direct_policy_stats(_stats(safe_hits=0.0, valid_safe=1.0, invalid=0.0), {})
    supported = _finalize_direct_policy_stats(_stats(safe_hits=1.0, valid_safe=1.0, invalid=0.0), {})
    assert zero_top1["direct_contract_zero_safe_top1_regimes"] == 2.0
    assert supported["direct_contract_zero_safe_top1_regimes"] == 0.0
    assert zero_top1["direct_contract_safe_rank_risk"] > supported["direct_contract_safe_rank_risk"]


def test_nominal_deviation_matches_certificate_geometry(tmp_path: Path) -> None:
    nominal = tmp_path / "n.npz"
    candidate = tmp_path / "c.npz"
    prefix0 = np.zeros((4, 3), dtype=np.float32)
    prefix1 = prefix0.copy()
    prefix1[:, 0] = 5.0
    np.savez(nominal, scene_id=np.asarray("s"), time_index=np.asarray(1), is_nominal=np.asarray(1.0), prefix_states=prefix0)
    np.savez(candidate, scene_id=np.asarray("s"), time_index=np.asarray(1), is_nominal=np.asarray(0.0), prefix_states=prefix1)
    values = _nominal_deviation_by_path([nominal, candidate])
    assert values[0] == 0.0
    assert abs(values[1] - 1.0) < 1.0e-7


def test_all_training_stages_use_natural_population_and_exact_contract() -> None:
    staged = (ROOT / "scripts" / "adapt_ocrap_v48_31_contract_slack_rank_variant.sh").read_text()
    assert staged.count("GROUP_BATCH_STRATIFIED=false GROUP_BATCHING_REPLACEMENT=false") >= 3
    assert "BEST_METRIC=direct_contract_safe_rank_risk" in staged
    assert "direct_evidence_concord_benefit_calibrator,direct_evidence_concord_harm_calibrator,direct_evidence_concord_admission_calibrator" in staged
    single = (ROOT / "scripts" / "adapt_ocrap_v48_31_contract_slack_rank_single_stage.sh").read_text()
    assert "POLICY_METRIC_EXACT_ELIGIBILITY=true" in single


def test_metric_calibration_contract_uses_safe_not_raw_proposal_opportunities() -> None:
    checker = (ROOT / "tools" / "check_v48_31_metric_calibration_contract.py").read_text()
    assert "proposal_safe_positive_groups" in checker
    assert 'rule.get("proposal_positive_group_count"' not in checker


def test_ablations_isolate_reliability_and_joint_refinement_in_four_waves() -> None:
    text = (ROOT / "scripts" / "run_v48_31_parallel_ablations.sh").read_text()
    for name in (
        "A_contract_natural_no_reliability_no_joint",
        "B_add_support_reliability_no_joint",
        "C_add_joint_refinement_no_reliability",
        "D_full_contract_slack_rank",
    ):
        assert name in text
    assert "max_concurrent_tasks':2" in text
    assert 'run_task "$group" balanced "$GPU0"' in text
    assert 'run_task "$group" precision "$GPU1"' in text


def test_dev_shadow_references_existing_v48_31_fail_closed_tools() -> None:
    script = (ROOT / "scripts" / "run_v48_31_dev_shadow_closed_loop.sh").read_text()
    tools = (
        "audit_v48_31_shadow_provenance.py",
        "check_v48_31_shadow_runtime_contract.py",
        "check_v48_31_physical_target_support.py",
        "check_v48_31_regime_targets.py",
    )
    for name in tools:
        assert (ROOT / "tools" / name).is_file()
        assert f"tools/{name}" in script
    assert "audit_v48_30_shadow_provenance.py" not in script
    assert "check_v48_30_shadow_runtime_contract.py" not in script
