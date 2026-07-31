from __future__ import annotations

from pathlib import Path

import torch

from ocrap.models.losses import direct_uncertainty_recovery_value_loss
from ocrap.models.ocrap import OCRAPModel


def _model(*, admission: bool = True, component_heads: bool = True) -> OCRAPModel:
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
        direct_recovery_evidence_calibrator_scale=0.5,
        direct_recovery_evidence_calibrator_mode="dual_tail_context",
        direct_recovery_evidence_calibrator_context=True,
        direct_recovery_evidence_calibrator_context_detach=True,
        direct_recovery_evidence_calibrator_context_source="tournament",
        direct_recovery_evidence_unified_experts=True,
        direct_recovery_evidence_component_heads=component_heads,
        direct_recovery_evidence_component_scale=2.0,
        direct_recovery_evidence_concord=True,
        direct_recovery_evidence_consensus_disagreement_penalty=0.15,
        direct_recovery_evidence_admission_head=admission,
        direct_recovery_evidence_admission_scale=2.0,
    )


def _inputs():
    torch.manual_seed(4822)
    x = torch.randn(6, 12)
    groups = torch.tensor([[0], [0], [0], [1], [1], [1]])
    nominal = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    return x, groups, nominal


def test_covenant_has_three_bucket_invariant_hypotheses_and_is_small() -> None:
    model = _model().eval()
    assert model.direct_evidence_concord_benefit_calibrator is not None
    assert model.direct_evidence_concord_harm_calibrator is not None
    assert model.direct_evidence_concord_admission_calibrator is not None
    trainable = sum(
        p.numel()
        for name, p in model.named_parameters()
        if name.startswith("direct_evidence_concord_")
    )
    assert 0 < trainable < 8000
    x, groups, nominal = _inputs()
    with torch.no_grad():
        near = model(
            x,
            bucket_id=torch.ones(6, dtype=torch.long),
            group_index=groups,
            is_nominal=nominal,
            direct_only=True,
        )
        contact = model(
            x,
            bucket_id=torch.full((6,), 2, dtype=torch.long),
            group_index=groups,
            is_nominal=nominal,
            direct_only=True,
        )
    for key in (
        "direct_recovery_evidence_benefit_logit",
        "direct_recovery_evidence_harm_logit",
        "direct_recovery_admission_logit",
        "direct_recovery_evidence_score",
    ):
        assert torch.allclose(near[key], contact[key], atol=1e-7), key
    nominal_rows = nominal > 0.5
    assert torch.count_nonzero(near["direct_recovery_admission_logit"][nominal_rows]) == 0
    assert torch.count_nonzero(near["direct_recovery_evidence_score"][nominal_rows]) == 0


def test_sparse_admission_gradient_does_not_distort_benefit_or_harm_heads() -> None:
    model = _model().train()
    x, groups, nominal = _inputs()
    out = model(
        x,
        bucket_id=torch.ones(6, dtype=torch.long),
        group_index=groups,
        is_nominal=nominal,
        direct_only=True,
    )
    loss = out["direct_recovery_admission_logit"][nominal < 0.5].sum()
    loss.backward()
    admission_grads = [
        p.grad for p in model.direct_evidence_concord_admission_calibrator.parameters()
    ]
    benefit_grads = [
        p.grad for p in model.direct_evidence_concord_benefit_calibrator.parameters()
    ]
    harm_grads = [p.grad for p in model.direct_evidence_concord_harm_calibrator.parameters()]
    assert any(g is not None and bool(torch.count_nonzero(g)) for g in admission_grads)
    assert all(g is None or not bool(torch.count_nonzero(g)) for g in benefit_grads)
    assert all(g is None or not bool(torch.count_nonzero(g)) for g in harm_grads)


def _harmful_positive_group_loss(admission_candidate_logit: float) -> torch.Tensor:
    # Candidate 1 has a total PCD gain but a 0.20 DRS degradation, so it is a
    # raw-benefit positive and simultaneously component-harmful. Candidate 2 is
    # a dead-zone candidate. The safe-opportunity group target must therefore be 0.
    return direct_uncertainty_recovery_value_loss(
        pred_logit=torch.zeros(3),
        pred_logvar=torch.zeros(3),
        pred_rank_logit=torch.tensor([0.0, 3.0, 2.0]),
        pred_opportunity_logit=torch.tensor([0.0, 5.0, -5.0]),
        pred_harm_logit=torch.tensor([0.0, 5.0, -5.0]),
        pred_component_harm_logits=torch.tensor(
            [[0.0, 0.0, 0.0], [5.0, -5.0, -5.0], [-5.0, -5.0, -5.0]]
        ),
        pred_admission_logit=torch.tensor([0.0, admission_candidate_logit, -5.0]),
        teacher_r_dep=torch.tensor([0.0, 1.4, 0.0]),
        teacher_r_orc=torch.tensor([0.0, 1.4, 0.0]),
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
        scene_hash=torch.tensor([9, 9, 9]),
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
        ordinal_evidence_group_opportunity_weight=1.0,
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


def test_group_mil_penalizes_high_admission_on_harmful_benefit_candidate() -> None:
    low = _harmful_positive_group_loss(-5.0)
    high = _harmful_positive_group_loss(5.0)
    assert high.item() > low.item() + 1.0


def test_safe_sampler_is_decoupled_from_raw_benefit_head() -> None:
    root = Path(__file__).resolve().parents[1]
    train_text = (root / "src" / "ocrap" / "cli" / "train.py").read_text()
    script_text = (root / "scripts" / "adapt_ocrap_v48_22_covenant_variant.sh").read_text()
    assert '"group_batch_safe_positive_target"' in train_text
    assert 'GROUP_BATCH_SAFE_POSITIVE_TARGET="${GROUP_BATCH_SAFE_POSITIVE_TARGET:-true}"' in script_text
    assert 'ORDINAL_EVIDENCE_SAFE_BENEFIT_TARGET="$SAFE_BENEFIT_TARGET"' in script_text
    assert 'SAFE_BENEFIT_TARGET="${ORDINAL_EVIDENCE_SAFE_BENEFIT_TARGET:-false}"' in script_text


def test_v4822_runs_all_eight_ablations_together_four_per_gpu() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts" / "run_v48_22_parallel_ablations.sh").read_text()
    for group in (
        "A_two_head_safe_probability",
        "B_triad_candidate_only",
        "C_triad_group_mil_aggregate",
        "D_full_covenant",
    ):
        assert group in text
    assert "max_concurrent_tasks':8" in text
    assert "all eight tasks launched together; four tasks per A30" in text
    assert "run_wave" not in text
    assert 'BATCH_SIZE="${BATCH_SIZE:-48}"' in text
    assert "TASK_GPU_ASSIGNMENT.txt" in text


def test_v4822_checkpoint_and_certificate_use_explicit_admission_score() -> None:
    root = Path(__file__).resolve().parents[1]
    train_text = (root / "src" / "ocrap" / "cli" / "train.py").read_text()
    evaluator_text = (root / "src" / "ocrap" / "evaluation" / "evaluator.py").read_text()
    calibration_text = (root / "tools" / "calibrate_policy_risk_v48.py").read_text()
    assert "direct_covenant_selection_risk" in train_text
    assert "evaluate_initial_checkpoint" in train_text
    assert "pred_direct_delta" in evaluator_text
    assert 'pred_adv = float(r["delta"])' in calibration_text
