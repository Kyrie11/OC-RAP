from __future__ import annotations

import torch

from ocrap.cli.train import _finalize_direct_policy_stats
from ocrap.models.ocrap import OCRAPModel


def _model(*, calibrator: bool = True, context_source: str = "tournament") -> OCRAPModel:
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
        direct_recovery_evidence_calibrator=calibrator,
        direct_recovery_evidence_calibrator_hidden=12,
        direct_recovery_evidence_calibrator_scale=0.5,
        direct_recovery_evidence_calibrator_mode="dual_tail_context",
        direct_recovery_evidence_calibrator_context=True,
        direct_recovery_evidence_calibrator_context_detach=True,
        direct_recovery_evidence_calibrator_context_source=context_source,
    ).eval()


def _inputs():
    torch.manual_seed(4818)
    x = torch.randn(6, 12)
    groups = torch.tensor([[0], [0], [0], [1], [1], [1]])
    nominal = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    buckets = torch.tensor([1, 1, 1, 2, 2, 2])
    return x, groups, nominal, buckets


def test_dual_tail_zero_residual_is_identity_and_uses_tournament_context() -> None:
    torch.manual_seed(18)
    source = _model(calibrator=False)
    corrected = _model(calibrator=True)
    current = corrected.state_dict()
    corrected.load_state_dict(
        {k: v for k, v in source.state_dict().items() if k in current and current[k].shape == v.shape},
        strict=False,
    )
    x, groups, nominal, buckets = _inputs()
    with torch.no_grad():
        base = source(x, bucket_id=buckets, group_index=groups, is_nominal=nominal, direct_only=True)
        out = corrected(x, bucket_id=buckets, group_index=groups, is_nominal=nominal, direct_only=True)
    assert out["direct_recovery_tournament_context"].shape == (6, 16)
    assert out["direct_recovery_evidence_calibrator_input"].shape[-1] == 20
    assert torch.count_nonzero(out["direct_recovery_evidence_calibrator_residual"]) == 0
    assert torch.allclose(base["direct_recovery_evidence_benefit_logit"], out["direct_recovery_evidence_benefit_logit"])
    assert torch.allclose(base["direct_recovery_evidence_harm_logit"], out["direct_recovery_evidence_harm_logit"])


def test_dual_tail_calibrator_is_small_and_does_not_force_simplex_competition() -> None:
    model = _model()
    assert model.direct_evidence_calibrators is not None
    assert 0 < sum(p.numel() for p in model.direct_evidence_calibrators.parameters()) < 5000
    with torch.no_grad():
        for adapter in model.direct_evidence_calibrators:
            adapter[-1].weight.zero_()
            adapter[-1].bias.fill_(10.0)
    x, groups, nominal, buckets = _inputs()
    with torch.no_grad():
        out = model(x, bucket_id=buckets, group_index=groups, is_nominal=nominal, direct_only=True)
    benefit = torch.sigmoid(out["direct_recovery_evidence_benefit_logit"])
    harm = torch.sigmoid(out["direct_recovery_evidence_harm_logit"])
    recovery = nominal < 0.5
    nominal_rows = nominal > 0.5
    assert float(out["direct_recovery_evidence_calibrator_residual"].abs().max()) <= 0.500001
    assert bool(((benefit + harm)[recovery] > 1.0).any())
    assert torch.allclose(
        out["direct_recovery_evidence_benefit_logit"][nominal_rows],
        torch.zeros_like(out["direct_recovery_evidence_benefit_logit"][nominal_rows]),
    )
    assert torch.allclose(
        out["direct_recovery_evidence_harm_logit"][nominal_rows],
        torch.zeros_like(out["direct_recovery_evidence_harm_logit"][nominal_rows]),
    )


def test_duet_checkpoint_metric_penalizes_cross_regime_recall_collapse() -> None:
    stats = {}
    for regime, recall_hits in (("near", 3.0), ("contact", 0.0)):
        stats[f"group_count_{regime}"] = 10.0
        stats[f"positive_count_{regime}"] = 4.0
        stats[f"positive_admission_hit_{regime}"] = recall_hits
        stats[f"certificate_positive_regret_sum_{regime}"] = 0.4
        stats[f"admitted_harmful_{regime}"] = 0.0
        stats[f"false_intervention_{regime}"] = 0.0
    out = _finalize_direct_policy_stats(
        stats,
        {
            "direct_policy_metric_min_positive_recall": 0.25,
            "direct_policy_metric_cross_regime_min_recall": 0.25,
            "direct_policy_metric_cross_regime_recall_weight": 4.0,
        },
    )
    assert out["direct_duet_cross_regime_recall_min"] == 0.0
    assert out["direct_duet_cross_regime_recall_shortfall"] == 0.25
    assert out["direct_duet_selection_risk"] > out["direct_certificate_risk_mean_worst"]
