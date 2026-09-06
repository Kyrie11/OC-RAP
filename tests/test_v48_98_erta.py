from __future__ import annotations

import json
from pathlib import Path

import torch

from ocrap.models.encoders import FlatFeatureLayout, StructuredTokenEncoder
from ocrap.v48_98_executable_recovery_tangent import (
    ENGINEERING_VERSION,
    ExecutableRecoveryTangentAdapter,
    nominal_identity_synthetic_check,
    orthonormal_tangent_basis_synthetic_check,
    scene_tokens_with_recovery_tangent,
)


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = StructuredTokenEncoder(FlatFeatureLayout(feature_max_agents=2), d_model=16, num_layers=1, num_heads=4, dropout=0.0)


def test_v48_98_parameter_count_and_basis():
    m = ExecutableRecoveryTangentAdapter(d_model=192, prefix_param_dim=5, prefix_state_dim=80, control_dim=40)
    assert m.trainable_parameter_count == 2 * 192 + 2 * (5 + 80 + 40)
    assert nominal_identity_synthetic_check(16)
    assert orthonormal_tangent_basis_synthetic_check(16)


def test_v48_98_nominal_scene_tokens_exact_identity():
    torch.manual_seed(1)
    model = _Model().eval()
    layout = model.encoder.layout
    x = torch.randn(3, layout.total_dim)
    adapter = ExecutableRecoveryTangentAdapter(d_model=16, prefix_param_dim=layout.prefix_param_dim, prefix_state_dim=layout.prefix_flat_dim, control_dim=layout.control_flat_dim)
    with torch.no_grad():
        base = model.encoder.forward_tokens(x)
        got = scene_tokens_with_recovery_tangent(model, adapter, x, x)
    assert torch.allclose(base, got, atol=1e-6, rtol=0.0)


def test_v48_98_only_candidate_relative_delta_activates_update():
    torch.manual_seed(2)
    layout = FlatFeatureLayout(feature_max_agents=2)
    adapter = ExecutableRecoveryTangentAdapter(d_model=16, prefix_param_dim=layout.prefix_param_dim, prefix_state_dim=layout.prefix_flat_dim, control_dim=layout.control_flat_dim)
    with torch.no_grad():
        adapter.prefix_param_map[0, 0] = 1.0
    z = torch.zeros(2, layout.prefix_param_dim)
    s = torch.zeros(2, layout.prefix_flat_dim)
    c = torch.zeros(2, layout.control_flat_dim)
    out0 = adapter(prefix_param_delta=z, prefix_state_delta=s, control_delta=c)
    z1 = z.clone(); z1[:, 0] = 1.0
    out1 = adapter(prefix_param_delta=z1, prefix_state_delta=s, control_delta=c)
    assert torch.count_nonzero(out0["prefix_param"]) == 0
    assert torch.linalg.norm(out1["prefix_param"]).item() > 0
    assert torch.count_nonzero(out1["prefix_state"]) == 0
    assert torch.count_nonzero(out1["control"]) == 0


def test_v48_98_engineering_version():
    assert ENGINEERING_VERSION == "v48.98.0-OC-ERTA"
