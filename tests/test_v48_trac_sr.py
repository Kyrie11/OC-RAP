from __future__ import annotations

import numpy as np
import torch

from ocrap.models.ocrap import OCRAPModel
from ocrap.planning.selector import calibrated_constrained_select


def _risk_select(predicted_harm: float):
    return calibrated_constrained_select(
        utility=np.array([1.0, 0.05]),
        r_dep=np.array([0.5, -1.0]),
        hard=np.zeros(2),
        harm=np.zeros(2),
        feasible=np.ones(2, dtype=bool),
        gamma_rec=0.0,
        pred_gap=np.zeros(2),
        pred_drs=np.ones(2),
        nominal_deviation=np.array([0.0, 0.02]),
        pred_direct_value=np.array([0.0, 0.8]),
        pred_direct_std=np.zeros(2),
        pred_direct_opportunity=np.array([0.5, 0.9]),
        pred_direct_harm=np.array([0.0, predicted_harm]),
        candidate_macro_names=["nominal", "yield"],
        regime_name="test_near_contact",
        direct_value_certificate=True,
        direct_value_macro_allowlist="yield",
        direct_value_uncertainty_mode="risk_controlled",
        direct_value_min_advantage_lcb=0.5,
        direct_value_score_mode=True,
        direct_value_opportunity_threshold=0.7,
        direct_value_harm_threshold=0.25,
        direct_value_top1_only=True,
        direct_value_risk_controlled_admission=True,
        direct_value_challenge_nominal=True,
        direct_value_bonus=1.0,
        stress_rescue_challenge_nominal=True,
    )


def test_predicted_harm_is_a_real_selector_gate():
    assert _risk_select(0.10).selected_index == 1
    assert _risk_select(0.80).selected_index == 0


def test_robust_expert_aggregation_is_conservative_and_bucket_free():
    model = OCRAPModel(
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
        direct_recovery_value_pooling="scene",
        direct_recovery_value_output="score",
        direct_recovery_opportunity_head=True,
        direct_recovery_harm_head=True,
        direct_recovery_value_experts=True,
        direct_recovery_value_num_experts=2,
        direct_recovery_value_expert_routing="robust_ensemble",
        direct_recovery_expert_disagreement_penalty=0.5,
    ).eval()
    assert model.direct_value_heads is not None
    with torch.no_grad():
        for head in model.direct_value_heads:
            for param in head.parameters():
                param.zero_()
        model.direct_value_heads[0][-1].bias.copy_(torch.tensor([2.0, 0.0, 2.0, -2.0]))
        model.direct_value_heads[1][-1].bias.copy_(torch.tensor([0.0, 0.0, 0.0, 2.0]))
        x = torch.zeros(2, 12)
        out = model(x, bucket_id=torch.tensor([1, 2]), direct_only=True)
    # mean +/- 0.5 std: gain/opportunity lower bound, harm upper bound.
    assert torch.allclose(out["direct_recovery_value_logit"], torch.tensor([0.5, 0.5]))
    assert torch.allclose(out["direct_recovery_opportunity_logit"], torch.tensor([0.5, 0.5]))
    assert torch.allclose(out["direct_recovery_harm_logit"], torch.tensor([1.0, 1.0]))
    assert torch.allclose(out["direct_expert_weights"], torch.full((2, 2), 0.5))
