from __future__ import annotations

import torch

from ocrap.models.encoders import FlatFeatureLayout, StructuredTokenEncoder
from ocrap.v48_105_prelast_action_equivariance_localization import (
    action_interaction_dimension_check,
    action_interaction_slice,
    agent_permutation_invariance_check,
    candidate_zero_delta_check,
    prelast_memory_summary,
    summary_group_slices,
    summary_partition_check,
)
from tools.run_v48_105_prelast_action_equivariance_localization_audit import _prelast_memory


def test_summary_partition_and_action_interaction_dimension():
    assert summary_partition_check(16)
    assert action_interaction_dimension_check(192)
    g = summary_group_slices(16)
    assert g["cls"] == slice(0, 16)
    assert g["control"] == slice(80, 96)
    assert g["agents"] == slice(176, 240)
    assert action_interaction_slice(16) == slice(80, 240)


def test_candidate_nominal_zero_delta():
    assert candidate_zero_delta_check(16)


def test_agent_set_permutation_invariant():
    assert agent_permutation_invariance_check(16)


def test_prelast_summary_dimension():
    x = torch.randn(3, 43, 16)
    z = prelast_memory_summary(x)
    assert z.shape == (3, 240)


def test_prelast_reconstructs_historical_final_tokens():
    torch.manual_seed(48105)
    layout = FlatFeatureLayout(feature_max_agents=4)
    enc = StructuredTokenEncoder(layout, d_model=16, num_layers=2, num_heads=4, dropout=0.1).eval()
    x = torch.randn(5, layout.total_dim)
    with torch.no_grad():
        pre, final = _prelast_memory(enc, x)
        direct = enc.forward_tokens(x)
    assert pre.shape[1] == 15  # CLS + 10 fixed semantic + 4 agent tokens.
    assert torch.equal(final, direct)


def test_action_interaction_excludes_cls_and_ego_history():
    d = 8
    z = torch.zeros(2, 15 * d)
    z[:, :5 * d] = 7.0
    sl = action_interaction_slice(d)
    assert torch.count_nonzero(z[:, sl]).item() == 0
