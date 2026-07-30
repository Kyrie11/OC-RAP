from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ocrap.algorithms.evidence_targets import (
    ComponentVetoTolerances,
    component_veto_margin_numpy,
    component_veto_terms_numpy,
    component_veto_terms_torch,
)
from ocrap.cli.train import _finalize_direct_policy_stats
from ocrap.models.losses import direct_uncertainty_recovery_value_loss
from ocrap.models.ocrap import OCRAPModel


def _model(*, component_heads: bool = True) -> OCRAPModel:
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
    ).eval()


def _inputs():
    torch.manual_seed(4820)
    x = torch.randn(6, 12)
    groups = torch.tensor([[0], [0], [0], [1], [1], [1]])
    nominal = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    return x, groups, nominal


def test_unified_evidence_is_bucket_invariant_and_uses_both_source_experts() -> None:
    model = _model()
    x, groups, nominal = _inputs()
    bucket_near = torch.ones(6, dtype=torch.long)
    bucket_contact = torch.full((6,), 2, dtype=torch.long)
    with torch.no_grad():
        near = model(x, bucket_id=bucket_near, group_index=groups, is_nominal=nominal, direct_only=True)
        contact = model(x, bucket_id=bucket_contact, group_index=groups, is_nominal=nominal, direct_only=True)
    assert near["direct_recovery_delta_expert_outputs"].shape == (6, 2, 2)
    assert near["direct_recovery_evidence_calibrator_input"].shape[-1] == 26
    for key in (
        "direct_recovery_rank_logit",
        "direct_recovery_evidence_benefit_logit",
        "direct_recovery_evidence_harm_logit",
        "direct_recovery_evidence_component_harm_logits",
        "direct_recovery_evidence_calibrator_input",
    ):
        assert torch.allclose(near[key], contact[key], atol=1e-7), key


def test_unified_calibrator_is_small_bounded_and_nominal_pinned() -> None:
    model = _model()
    assert model.direct_evidence_unified_calibrator is not None
    assert model.direct_evidence_calibrators is None
    assert sum(p.numel() for p in model.direct_evidence_unified_calibrator.parameters()) < 3000
    with torch.no_grad():
        model.direct_evidence_unified_calibrator[-1].bias.fill_(10.0)
    x, groups, nominal = _inputs()
    buckets = torch.tensor([1, 1, 1, 2, 2, 2])
    with torch.no_grad():
        out = model(x, bucket_id=buckets, group_index=groups, is_nominal=nominal, direct_only=True)
    residual = out["direct_recovery_evidence_calibrator_residual"]
    assert float(residual[:, 0].abs().max()) <= 0.500001
    assert float(residual[:, 1:].abs().max()) <= 2.000001
    nominal_rows = nominal > 0.5
    assert torch.count_nonzero(out["direct_recovery_evidence_benefit_logit"][nominal_rows]) == 0
    assert torch.count_nonzero(out["direct_recovery_evidence_harm_logit"][nominal_rows]) == 0
    assert torch.count_nonzero(out["direct_recovery_evidence_component_harm_logits"][nominal_rows]) == 0



def test_unified_envelopes_are_non_compensatory_and_identity_preserving() -> None:
    model = _model()
    x, groups, nominal = _inputs()
    buckets = torch.tensor([1, 1, 1, 2, 2, 2])
    with torch.no_grad():
        out = model(x, bucket_id=buckets, group_index=groups, is_nominal=nominal, direct_only=True)
    benefit_experts = out["direct_recovery_evidence_expert_benefit_logits"]
    harm_experts = out["direct_recovery_evidence_expert_harm_logits"]
    base = out["direct_recovery_evidence_expert_base"]
    assert torch.allclose(base[:, 0], benefit_experts.amin(dim=1), atol=1e-7)
    assert torch.allclose(base[:, 1], harm_experts.amax(dim=1), atol=1e-7)
    components = out["direct_recovery_evidence_component_harm_logits"]
    aggregate = out["direct_recovery_evidence_harm_logit"]
    non_nominal = nominal < 0.5
    assert torch.allclose(aggregate[non_nominal], components[non_nominal].amax(dim=-1), atol=1e-7)


def test_component_harm_is_semantically_reset_not_anchored_to_old_source_harm() -> None:
    model = _model()
    x, groups, nominal = _inputs()
    buckets = torch.tensor([1, 1, 1, 2, 2, 2])
    with torch.no_grad():
        out = model(x, bucket_id=buckets, group_index=groups, is_nominal=nominal, direct_only=True)
    non_nominal = nominal < 0.5
    # The old signed-PCD source harm is retained as an input/diagnostic, but the
    # new DRS/DEP/GAP heads start at a neutral candidate-vs-nominal logit of 0.
    assert bool((out["direct_recovery_evidence_expert_base"][non_nominal, 1].abs() > 1.0e-6).any())
    assert torch.count_nonzero(out["direct_recovery_evidence_component_harm_logits"]) == 0
    assert torch.count_nonzero(out["direct_recovery_evidence_harm_logit"]) == 0


def _factorized_intragroup_harm_loss(harm_logits: torch.Tensor) -> torch.Tensor:
    return direct_uncertainty_recovery_value_loss(
        pred_logit=torch.zeros(3),
        pred_logvar=torch.zeros(3),
        pred_rank_logit=torch.tensor([0.0, 2.0, 1.0]),
        pred_opportunity_logit=torch.zeros(3),
        pred_harm_logit=harm_logits,
        # nominal/candidate-2: DRS=1, r_dep=0 -> PCD=0.5;
        # candidate-1: DRS=0.8, sigmoid(r_dep)=0.625 -> PCD=0.5.
        teacher_r_dep=torch.tensor([0.0, 0.5108256, 0.0]),
        teacher_r_orc=torch.tensor([0.0, 0.5108256, 0.0]),
        teacher_q=torch.ones((3, 5, 1)),
        teacher_m_star=torch.tensor([
            [[1.0], [1.0], [1.0], [1.0], [1.0]],
            [[-1.0], [1.0], [1.0], [1.0], [1.0]],
            [[1.0], [1.0], [1.0], [1.0], [1.0]],
        ]),
        teacher_hard_violation=torch.zeros(3),
        teacher_harm_proxy=torch.zeros(3),
        root_probs=torch.full((3, 5), 0.2),
        root_valid=torch.ones((3, 5), dtype=torch.bool),
        option_valid=torch.ones((3, 1), dtype=torch.bool),
        scene_hash=torch.tensor([7, 7, 7]),
        time_index=torch.zeros(3, dtype=torch.long),
        macro_type_id=torch.tensor([0, 5, 3]),
        is_nominal=torch.tensor([1.0, 0.0, 0.0]),
        bucket_id=torch.ones(3, dtype=torch.long),
        macro_ids=(3, 5),
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
        ordinal_evidence_batch_balanced=True,
        ordinal_evidence_class_balanced_weight=0.0,
        ordinal_evidence_benefit_margin_weight=0.0,
        ordinal_evidence_harm_margin_weight=0.0,
        ordinal_evidence_ordered_nll_top1_weight=0.0,
        ordinal_evidence_ordered_nll_all_weight=0.0,
        ordinal_evidence_proposal_topk_weight=0.0,
        ordinal_evidence_proposal_topk=2,
        ordinal_evidence_intragroup_harm_weight=1.0,
        ordinal_evidence_intragroup_benefit_weight=0.0,
        ordinal_evidence_factorized_harm_drs_tolerance=0.05,
    )


def test_intragroup_harm_ranking_uses_component_veto_not_signed_total_delta() -> None:
    # Both candidates have identical total PCD. Candidate 1 is nevertheless
    # component-harmful because its DRS drops by 0.20. The factorized pairwise
    # objective must reward ranking candidate 1 as riskier than candidate 2.
    good = _factorized_intragroup_harm_loss(torch.tensor([0.0, 3.0, -3.0]))
    bad = _factorized_intragroup_harm_loss(torch.tensor([0.0, -3.0, 3.0]))
    assert good.item() < bad.item()


def test_component_veto_vector_terms_match_scalar_margin() -> None:
    kwargs = dict(
        candidate_drs=0.72,
        nominal_drs=0.80,
        candidate_r_dep=0.40,
        nominal_r_dep=0.60,
        candidate_gap=0.31,
        nominal_gap=0.20,
        candidate_hard=0.0,
        nominal_hard=0.0,
        candidate_harm_proxy=0.0,
        nominal_harm_proxy=0.0,
        tolerances=ComponentVetoTolerances(drs=0.05, deployability_gate=0.05, gap_discount=0.05),
    )
    terms_np = component_veto_terms_numpy(**kwargs)
    margin = component_veto_margin_numpy(**kwargs)
    torch_kwargs = {
        k: (torch.tensor([v], dtype=torch.float32) if isinstance(v, (int, float)) else v)
        for k, v in kwargs.items()
    }
    terms_t = component_veto_terms_torch(**torch_kwargs)
    assert terms_np.shape == (5,)
    assert terms_t.shape == (1, 5)
    assert np.isclose(float(np.max(terms_np)), margin, atol=1e-7)
    assert np.allclose(terms_t.squeeze(0).numpy(), terms_np, atol=1e-6)


def test_unison_metric_is_robust_evaluation_not_model_routing() -> None:
    stats: dict[str, float] = {}
    for regime, recall_hits, harmful, false in (
        ("near", 3.0, 1.0, 2.0),
        ("contact", 2.0, 1.0, 3.0),
    ):
        stats[f"group_count_{regime}"] = 40.0
        stats[f"positive_count_{regime}"] = 10.0
        stats[f"positive_admission_hit_{regime}"] = recall_hits
        stats[f"certificate_positive_regret_sum_{regime}"] = 1.0
        stats[f"admitted_harmful_{regime}"] = harmful
        stats[f"false_intervention_{regime}"] = false
    out = _finalize_direct_policy_stats(stats, {"direct_policy_metric_facet_min_recall": 0.20})
    assert out["direct_unison_selection_risk"] == out["direct_facet_selection_risk"]


def test_v4820_training_contract_keeps_group_erm_primary() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "adapt_ocrap_v48_20_unison_variant.sh"
    text = script.read_text(encoding="utf-8")
    assert "ORDINAL_EVIDENCE_BALANCED_REPLACES_ERM=false" in text
    assert 'SETWISE_W="$SETWISE_WEIGHT"' in text
    assert "EVIDENCE_UNIFIED_EXPERTS=true" in text
    assert "TRAINABLE_PARAM_PREFIXES='direct_evidence_unified_calibrator'" in text


def test_safe_set_uses_deployed_topk_evidence_score_only() -> None:
    # Candidate 3 is the only safe positive but is outside the frozen top-2
    # proposal.  The deployment-exact set loss must therefore target nominal and
    # must not backpropagate through candidate 3.  v48.19's all-candidate set loss
    # did backpropagate through it, creating a train/deploy mismatch.
    opp = torch.zeros(4, requires_grad=True)
    harm = torch.zeros(4, requires_grad=True)
    loss = direct_uncertainty_recovery_value_loss(
        pred_logit=torch.zeros(4),
        pred_logvar=torch.zeros(4),
        pred_rank_logit=torch.tensor([0.0, 3.0, 2.0, -1.0]),
        pred_opportunity_logit=opp,
        pred_harm_logit=harm,
        teacher_r_dep=torch.tensor([0.0, 0.0, 0.0, 1.0]),
        teacher_r_orc=torch.tensor([0.0, 0.0, 0.0, 1.0]),
        teacher_q=torch.ones((4, 5, 1)),
        teacher_m_star=torch.ones((4, 5, 1)),
        teacher_hard_violation=torch.zeros(4),
        teacher_harm_proxy=torch.zeros(4),
        root_probs=torch.full((4, 5), 0.2),
        root_valid=torch.ones((4, 5), dtype=torch.bool),
        option_valid=torch.ones((4, 1), dtype=torch.bool),
        scene_hash=torch.tensor([11, 11, 11, 11]),
        time_index=torch.zeros(4, dtype=torch.long),
        macro_type_id=torch.tensor([0, 2, 3, 5]),
        is_nominal=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        bucket_id=torch.ones(4, dtype=torch.long),
        macro_ids=(2, 3, 5),
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
        setwise_admission_weight=1.0,
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
        ordinal_evidence_batch_balanced=False,
        ordinal_evidence_ordered_nll_top1_weight=0.0,
        ordinal_evidence_ordered_nll_all_weight=0.0,
        ordinal_evidence_proposal_topk_weight=0.0,
        ordinal_evidence_proposal_topk=2,
    )
    loss.backward()
    assert opp.grad is not None and harm.grad is not None
    assert abs(float(opp.grad[3])) < 1.0e-8
    assert abs(float(harm.grad[3])) < 1.0e-8
    # The two deployed candidates receive gradients that lower their evidence
    # score because the top-k contains no safe positive and nominal is the target.
    assert float(opp.grad[1]) > 0.0 and float(opp.grad[2]) > 0.0
    assert float(harm.grad[1]) < 0.0 and float(harm.grad[2]) < 0.0
