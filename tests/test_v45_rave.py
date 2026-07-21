from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from ocrap.models.ocrap import OCRAPModel


def test_regime_experts_route_near_and_contact_to_different_heads():
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
        direct_recovery_value_experts=True,
        direct_recovery_value_num_experts=2,
    ).eval()
    assert model.direct_value_heads is not None
    with torch.no_grad():
        for head in model.direct_value_heads:
            for param in head.parameters():
                param.zero_()
        model.direct_value_heads[0][-1].bias.copy_(torch.tensor([1.0, 0.0, -1.0]))
        model.direct_value_heads[1][-1].bias.copy_(torch.tensor([2.0, 0.0, 1.0]))
        x = torch.zeros(2, 12)
        out = model(x, bucket_id=torch.tensor([1, 2]))
    assert torch.allclose(out["direct_recovery_value_logit"], torch.tensor([1.0, 2.0]))
    assert torch.allclose(out["direct_recovery_opportunity_logit"], torch.tensor([-1.0, 1.0]))


def test_v45_calibration_search_does_not_drop_probabilities_below_point_zero_five():
    tool = Path(__file__).resolve().parents[1] / "tools" / "calibrate_direct_value_risk_v45.py"
    spec = importlib.util.spec_from_file_location("v45_cal", tool)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    groups = [{"pairs": [{"opportunity": 0.004}, {"opportunity": 0.021}]}]
    thresholds = module._candidate_opportunity_thresholds(groups, 0.0)
    assert thresholds
    assert min(thresholds) <= 0.004
    assert any(0.0 < value < 0.05 for value in thresholds)
