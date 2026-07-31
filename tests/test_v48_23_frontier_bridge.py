from __future__ import annotations

from pathlib import Path

import torch

from ocrap.cli.train import _finalize_direct_policy_stats
from ocrap.models.losses import direct_uncertainty_recovery_value_loss
from ocrap.models.ocrap import OCRAPModel


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
        direct_recovery_evidence_component_scale=4.0,
        direct_recovery_evidence_concord=True,
        direct_recovery_evidence_consensus_disagreement_penalty=0.15,
        direct_recovery_evidence_admission_head=True,
        direct_recovery_evidence_admission_scale=2.0,
        direct_recovery_evidence_frontier=True,
        direct_recovery_evidence_component_prior_logit=-2.0,
    )


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(4823)
    x = torch.randn(6, 12)
    groups = torch.tensor([[0], [0], [0], [1], [1], [1]])
    nominal = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    return x, groups, nominal


def test_frontier_semantic_prior_is_nonharmful_and_identity_preserving() -> None:
    model = _model().eval()
    x, groups, nominal = _inputs()
    with torch.no_grad():
        out = model(
            x,
            bucket_id=torch.ones(6, dtype=torch.long),
            group_index=groups,
            is_nominal=nominal,
            direct_only=True,
        )
    recovery = nominal < 0.5
    component = out["direct_recovery_evidence_component_harm_logits"]
    assert torch.allclose(component[recovery], torch.full_like(component[recovery], -2.0))
    assert torch.allclose(
        out["direct_recovery_evidence_harm_logit"][recovery],
        torch.full_like(out["direct_recovery_evidence_harm_logit"][recovery], -2.0),
    )
    assert torch.allclose(
        out["direct_recovery_admission_logit"][recovery],
        out["direct_recovery_evidence_benefit_logit"][recovery],
        atol=1.0e-7,
    )
    assert torch.count_nonzero(out["direct_recovery_evidence_calibrator_residual"]) == 0


def _loss_common(**extra):
    kwargs = dict(
        pred_logit=torch.zeros(3),
        pred_logvar=torch.zeros(3),
        pred_rank_logit=torch.tensor([0.0, 3.0, 2.0]),
        teacher_r_dep=torch.tensor([0.0, 1.4, 1.4]),
        teacher_r_orc=torch.tensor([0.0, 1.4, 1.4]),
        teacher_q=torch.ones((3, 5, 1)),
        teacher_m_star=torch.tensor(
            [
                [[1.0], [1.0], [1.0], [1.0], [1.0]],
                [[-1.0], [1.0], [1.0], [1.0], [1.0]],
                [[1.0], [1.0], [1.0], [1.0], [1.0]],
            ]
        ),
        teacher_hard_violation=torch.zeros(3),
        teacher_harm_proxy=torch.zeros(3),
        root_probs=torch.full((3, 5), 0.2),
        root_valid=torch.ones((3, 5), dtype=torch.bool),
        option_valid=torch.ones((3, 1), dtype=torch.bool),
        scene_hash=torch.tensor([23, 23, 23]),
        time_index=torch.zeros(3, dtype=torch.long),
        macro_type_id=torch.tensor([0, 2, 3]),
        is_nominal=torch.tensor([1.0, 0.0, 0.0]),
        bucket_id=torch.ones(3, dtype=torch.long),
        macro_ids=(2, 3),
        bucket_ids=(1,),
        output_mode="score",
        exact_teacher_pcd=True,
        positive_gain=0.01,
        negative_gain=0.01,
        point_weight=0.0,
        centered_weight=0.0,
        listwise_weight=0.0,
        advantage_weight=0.0,
        pairwise_weight=0.0,
        top_rank_weight=0.0,
        opportunity_weight=0.0,
        harm_weight=0.0,
        setwise_admission_weight=0.0,
        selective_risk_weight=0.0,
        selective_coverage_weight=0.0,
        policy_distill_weight=0.0,
        policy_regret_weight=0.0,
        preference_weight=0.0,
        preference_regret_weight=0.0,
        preference_listwise_weight=0.0,
        preference_gap_weight=0.0,
        preference_set_weight=0.0,
        preference_all_group_set_weight=0.0,
        delta_nll_weight=0.0,
        ordinal_evidence_independent_tails=True,
        ordinal_evidence_factorized_harm=True,
        ordinal_evidence_component_tail_weight=0.0,
        ordinal_evidence_safe_benefit_target=False,
        ordinal_evidence_group_opportunity_weight=0.0,
        ordinal_evidence_admission_weight=0.0,
        ordinal_evidence_batch_balanced=False,
        ordinal_evidence_ordered_nll_top1_weight=0.0,
        ordinal_evidence_ordered_nll_all_weight=0.0,
        ordinal_evidence_proposal_topk_weight=0.0,
        ordinal_evidence_proposal_topk=2,
        ordinal_evidence_intragroup_benefit_weight=0.0,
        ordinal_evidence_intragroup_harm_weight=0.0,
        ordinal_evidence_factorized_harm_drs_tolerance=0.05,
    )
    kwargs.update(extra)
    return direct_uncertainty_recovery_value_loss(**kwargs)


def test_continuous_benefit_listwise_moves_topk_toward_larger_gain() -> None:
    # Candidate 2 is the larger safe gain; candidate 1 is also a raw-benefit
    # positive but component-harmful. The continuous teacher should raise the
    # larger-gain logit and lower the inferior candidate logit.
    opportunity = torch.tensor([0.0, 1.0, -1.0], requires_grad=True)
    loss = _loss_common(
        pred_opportunity_logit=opportunity,
        pred_harm_logit=torch.tensor([0.0, 4.0, -4.0]),
        pred_component_harm_logits=torch.tensor(
            [[0.0, 0.0, 0.0], [4.0, -4.0, -4.0], [-4.0, -4.0, -4.0]]
        ),
        pred_admission_logit=torch.tensor([0.0, -1.0, 1.0]),
        ordinal_evidence_benefit_listwise_weight=1.0,
        ordinal_evidence_benefit_listwise_temperature=0.08,
    )
    loss.backward()
    assert opportunity.grad is not None
    assert opportunity.grad[2].item() < 0.0
    assert opportunity.grad[1].item() > 0.0


def test_frontier_contrast_separates_safe_gain_from_harmful_gain() -> None:
    admission = torch.zeros(3, requires_grad=True)
    loss = _loss_common(
        pred_opportunity_logit=torch.tensor([0.0, 4.0, 4.0]),
        pred_harm_logit=torch.tensor([0.0, 4.0, -4.0]),
        pred_component_harm_logits=torch.tensor(
            [[0.0, 0.0, 0.0], [4.0, -4.0, -4.0], [-4.0, -4.0, -4.0]]
        ),
        pred_admission_logit=admission,
        ordinal_evidence_frontier_pairwise_weight=1.0,
        ordinal_evidence_frontier_pairwise_margin=0.25,
    )
    loss.backward()
    assert admission.grad is not None
    # Candidate 1 is beneficial-but-harmful; candidate 2 is safe beneficial.
    assert admission.grad[1].item() > 0.0
    assert admission.grad[2].item() < 0.0


def test_categorical_group_policy_is_less_overconfident_than_noisy_or() -> None:
    common = dict(
        pred_opportunity_logit=torch.tensor([0.0, 0.0, -5.0]),
        pred_harm_logit=torch.tensor([0.0, 4.0, -4.0]),
        pred_component_harm_logits=torch.tensor(
            [[0.0, 0.0, 0.0], [4.0, -4.0, -4.0], [-4.0, -4.0, -4.0]]
        ),
        pred_admission_logit=torch.tensor([0.0, 0.0, -5.0]),
        ordinal_evidence_group_opportunity_weight=1.0,
    )
    noisy = _loss_common(**common, ordinal_evidence_categorical_group_policy=False)
    categorical = _loss_common(**common, ordinal_evidence_categorical_group_policy=True)
    assert categorical.item() < noisy.item()



def test_frontier_checkpoint_metric_penalizes_harmful_raw_gain_mass() -> None:
    def make(frontier_mass: float) -> dict[str, float]:
        stats: dict[str, float] = {}
        for regime in ("near", "contact"):
            stats[f"group_count_{regime}"] = 10.0
            stats[f"soft_safe_nll_sum_{regime}"] = 1.0
            stats[f"soft_safe_group_{regime}"] = 2.0
            stats[f"soft_safe_recall_sum_{regime}"] = 1.0
            stats[f"soft_false_admission_sum_{regime}"] = 1.0
            stats[f"soft_harmful_mass_sum_{regime}"] = 2.0
            stats[f"soft_frontier_harmful_mass_sum_{regime}"] = frontier_mass * 10.0
            stats[f"soft_safe_mass_sum_{regime}"] = 1.0
            stats[f"soft_safe_regret_sum_{regime}"] = 0.2
        return stats

    low = _finalize_direct_policy_stats(make(0.02), {})
    high = _finalize_direct_policy_stats(make(0.20), {})
    assert high["direct_frontier_selection_risk"] > low["direct_frontier_selection_risk"]
    assert high["direct_covenant_selection_risk"] == low["direct_covenant_selection_risk"]

def test_frontier_pipeline_adds_oracle_audit_dev_shadow_and_balanced_parallelism() -> None:
    root = Path(__file__).resolve().parents[1]
    calibration = (root / "tools" / "calibrate_policy_risk_v48.py").read_text()
    parallel = (root / "scripts" / "run_v48_23_parallel_ablations.sh").read_text()
    shadow = (root / "scripts" / "run_v48_23_dev_shadow_closed_loop.sh").read_text()
    assert "proposal_constrained_oracle_gate" in calibration
    assert "all eight tasks launched together; four tasks per A30" in parallel
    assert "max_concurrent_tasks':8" in parallel
    assert "TASK_GPU_ASSIGNMENT.txt" in parallel
    assert "uses_test_or_stress':False" in shadow
    assert "DIAGNOSTIC_ONLY_NO_PAPER.json" in shadow
    assert "BUCKET_SPLIT=evidence_adapt_dev" in shadow


def test_contact_event_metrics_aggregate_as_scene_rates_not_any_scene() -> None:
    from ocrap.simulation.closed_loop_runner import _aggregate_scene_results

    scenes = [
        {
            "num_decisions": 1,
            "num_metric_steps": 4,
            "metric_summary": {
                "secondary_overlap_event": 1.0,
                "new_stable_stop_event": 0.0,
                "post_contact_escape_event": 1.0,
                "overlap_any": 1.0,
                "overlap_mean": 0.25,
                "offroad_any": 0.0,
                "offroad_mean": 0.0,
            },
        },
        {
            "num_decisions": 1,
            "num_metric_steps": 4,
            "metric_summary": {
                "secondary_overlap_event": 0.0,
                "new_stable_stop_event": 1.0,
                "post_contact_escape_event": 0.0,
                "overlap_any": 0.0,
                "overlap_mean": 0.0,
                "offroad_any": 0.0,
                "offroad_mean": 0.0,
            },
        },
    ]
    agg = _aggregate_scene_results(scenes, method="v48", source="synthetic")
    assert agg["waymax_metrics"]["secondary_overlap_event"] == 0.5
    assert agg["secondary_overlap_scene_rate"] == 0.5
    assert agg["new_stable_stop_scene_rate"] == 0.5
    assert agg["post_contact_escape_scene_rate"] == 0.5
