from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

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
        direct_recovery_evidence_concord=True,
        direct_recovery_evidence_consensus_disagreement_penalty=0.15,
    ).eval()


def _inputs():
    torch.manual_seed(4821)
    x = torch.randn(6, 12)
    groups = torch.tensor([[0], [0], [0], [1], [1], [1]])
    nominal = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    return x, groups, nominal


def test_concord_is_bucket_invariant_decoupled_and_small() -> None:
    model = _model()
    assert model.direct_evidence_unified_calibrator is None
    assert model.direct_evidence_concord_benefit_calibrator is not None
    assert model.direct_evidence_concord_harm_calibrator is not None
    n = sum(p.numel() for p in model.direct_evidence_concord_benefit_calibrator.parameters())
    n += sum(p.numel() for p in model.direct_evidence_concord_harm_calibrator.parameters())
    assert n < 5000
    x, groups, nominal = _inputs()
    with torch.no_grad():
        near = model(x, bucket_id=torch.ones(6, dtype=torch.long), group_index=groups, is_nominal=nominal, direct_only=True)
        contact = model(x, bucket_id=torch.full((6,), 2, dtype=torch.long), group_index=groups, is_nominal=nominal, direct_only=True)
    for key in (
        "direct_recovery_evidence_benefit_logit",
        "direct_recovery_evidence_harm_logit",
        "direct_recovery_evidence_component_harm_logits",
        "direct_recovery_evidence_calibrator_input",
    ):
        assert torch.allclose(near[key], contact[key], atol=1e-7), key


def test_concord_consensus_is_mean_minus_disagreement_not_hard_min() -> None:
    model = _model()
    x, groups, nominal = _inputs()
    with torch.no_grad():
        out = model(x, bucket_id=torch.ones(6, dtype=torch.long), group_index=groups, is_nominal=nominal, direct_only=True)
    experts = out["direct_recovery_evidence_expert_benefit_logits"]
    expected = experts.mean(dim=1) - 0.15 * (experts.amax(dim=1) - experts.amin(dim=1))
    base = out["direct_recovery_evidence_expert_base"][:, 0]
    assert torch.allclose(base, expected, atol=1e-7)
    assert torch.all(base >= experts.amin(dim=1) - 1e-7)


def test_group_opportunity_noisy_or_targets_only_frozen_topk() -> None:
    opp = torch.zeros(4, requires_grad=True)
    harm = torch.zeros(4, requires_grad=True)
    loss = direct_uncertainty_recovery_value_loss(
        pred_logit=torch.zeros(4), pred_logvar=torch.zeros(4),
        pred_rank_logit=torch.tensor([0.0, 3.0, 2.0, -1.0]),
        pred_opportunity_logit=opp, pred_harm_logit=harm,
        teacher_r_dep=torch.tensor([0.0, 0.0, 0.0, 1.0]),
        teacher_r_orc=torch.tensor([0.0, 0.0, 0.0, 1.0]),
        teacher_q=torch.ones((4, 5, 1)), teacher_m_star=torch.ones((4, 5, 1)),
        teacher_hard_violation=torch.zeros(4), teacher_harm_proxy=torch.zeros(4),
        root_probs=torch.full((4, 5), 0.2), root_valid=torch.ones((4, 5), dtype=torch.bool),
        option_valid=torch.ones((4, 1), dtype=torch.bool), scene_hash=torch.tensor([11, 11, 11, 11]),
        time_index=torch.zeros(4, dtype=torch.long), macro_type_id=torch.tensor([0, 2, 3, 5]),
        is_nominal=torch.tensor([1.0, 0.0, 0.0, 0.0]), bucket_id=torch.ones(4, dtype=torch.long),
        macro_ids=(2, 3, 5), bucket_ids=(1,), output_mode="score", exact_teacher_pcd=True,
        positive_gain=0.01, negative_gain=0.01, point_weight=0.0, centered_weight=0.0,
        listwise_weight=0.0, advantage_weight=0.0, pairwise_weight=0.0, top_rank_weight=0.0,
        opportunity_weight=0.0, harm_weight=0.0, setwise_admission_weight=0.0,
        selective_risk_weight=0.0, selective_coverage_weight=0.0, policy_distill_weight=0.0,
        policy_regret_weight=0.0, preference_weight=0.0, preference_regret_weight=0.0,
        preference_listwise_weight=0.0, preference_gap_weight=0.0, preference_set_weight=0.0,
        preference_all_group_set_weight=0.0, delta_nll_weight=0.0,
        ordinal_evidence_independent_tails=True, ordinal_evidence_factorized_harm=True,
        ordinal_evidence_safe_benefit_target=True, ordinal_evidence_group_opportunity_weight=1.0,
        ordinal_evidence_batch_balanced=False, ordinal_evidence_ordered_nll_top1_weight=0.0,
        ordinal_evidence_ordered_nll_all_weight=0.0, ordinal_evidence_proposal_topk_weight=0.0,
        ordinal_evidence_proposal_topk=2,
    )
    loss.backward()
    assert opp.grad is not None
    # Candidate 3 is outside the deployed top-k and cannot affect group opportunity.
    assert abs(float(opp.grad[3])) < 1e-8
    assert abs(float(opp.grad[1])) > 0 and abs(float(opp.grad[2])) > 0


def test_concord_checkpoint_risk_is_threshold_free() -> None:
    stats = {}
    for regime in ("near", "contact"):
        stats[f"group_count_{regime}"] = 20.0
        stats[f"soft_safe_group_{regime}"] = 5.0
        stats[f"soft_safe_nll_sum_{regime}"] = 4.0
        stats[f"soft_safe_recall_sum_{regime}"] = 3.0
        stats[f"soft_false_admission_sum_{regime}"] = 1.5
        stats[f"soft_harmful_mass_sum_{regime}"] = 1.0
        stats[f"soft_safe_mass_sum_{regime}"] = 2.5
        stats[f"soft_safe_regret_sum_{regime}"] = 0.25
    out = _finalize_direct_policy_stats(stats, {})
    assert out["direct_concord_selection_risk"] > 0
    assert "direct_positive_admission_recall_near" not in out


def test_certificate_supports_safe_opportunity_semantics_without_changing_legacy_default() -> None:
    tool = Path(__file__).resolve().parents[1] / "tools" / "calibrate_policy_risk_v48.py"
    spec = importlib.util.spec_from_file_location("calibrate_policy_risk_v48", tool)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    groups = [{
        "pairs": [
            {"teacher_adv": 0.10, "teacher_harmful": True, "pred_adv": 1.0, "rank_margin": 1.0, "macro": 2, "candidate": 1},
            {"teacher_adv": 0.08, "teacher_harmful": False, "pred_adv": 0.9, "rank_margin": 1.0, "macro": 3, "candidate": 2},
        ],
        "oracle_best_teacher_adv": 0.10,
    }]
    selected = [groups[0]["pairs"][0]]
    raw = module._metrics(groups, selected, 0.0, 0.0, 0.05, 0.02)
    safe = module._metrics(groups, selected, 0.0, 0.0, 0.05, 0.02, safe_positive_only=True)
    assert raw["num_positive_selected"] == 1
    assert safe["num_positive_selected"] == 0
    assert safe["num_opportunities"] == 1


def test_v4821_script_contract_is_unified_and_non_regime_specific() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts" / "adapt_ocrap_v48_21_concord_variant.sh").read_text()
    assert "EVIDENCE_CONCORD=\"$CONCORD_ENABLED\"" in text
    assert "ORDINAL_EVIDENCE_SAFE_BENEFIT_TARGET=\"$SAFE_BENEFIT_TARGET\"" in text
    assert "ORDINAL_EVIDENCE_GROUP_OPPORTUNITY_WEIGHT=\"$GROUP_OPPORTUNITY_WEIGHT\"" in text
    assert "BEST_METRIC_NAME=\"${BEST_METRIC:-direct_concord_selection_risk}\"" in text
    assert "regime_id_exposed_to_evidence_model\": false" in text


def test_safe_benefit_sampler_and_primary_certificate_contract_are_aligned() -> None:
    root = Path(__file__).resolve().parents[1]
    train_text = (root / "src" / "ocrap" / "cli" / "train.py").read_text()
    adapt_text = (root / "scripts" / "adapt_ocrap_v48_21_concord_variant.sh").read_text()
    assert "safe_positive_sampler" in train_text
    assert '"safe_teacher_pcd" if group_index and safe_positive_sampler' in train_text
    # Training uses safe-benefit opportunities, while the primary held-out gate
    # preserves the preregistered raw-benefit + independent harm-veto semantics.
    assert "ORDINAL_EVIDENCE_SAFE_BENEFIT_TARGET=\"$SAFE_BENEFIT_TARGET\"" in adapt_text
    assert "OPPORTUNITY_LABEL_MODE=raw_benefit" in adapt_text
