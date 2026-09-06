from __future__ import annotations

import torch

from ocrap.v48_102_action_information_transport_sufficiency import (
    ALGORITHM_NAME,
    ENGINEERING_VERSION,
    agent_permutation_invariance_check,
    candidate_zero_delta_check,
    semantic_position_sensitivity_check,
    stage_i_action_features,
    stage_i_memory_summary,
)


def test_v48_102_stage_i_summary_dim():
    x = torch.randn(5, 43, 192)
    y = stage_i_memory_summary(x, semantic_token_count=11)
    assert y.shape == (5, 2880)


def test_v48_102_candidate_zero_delta():
    assert candidate_zero_delta_check(16)


def test_v48_102_agent_permutation_invariant():
    assert agent_permutation_invariance_check(16)


def test_v48_102_semantic_positions_preserved():
    assert semantic_position_sensitivity_check(16)


def test_v48_102_action_feature_shapes_and_nominal_purity():
    torch.manual_seed(102)
    x = torch.randn(6, 43, 32)
    state, delta, context = stage_i_action_features(x, semantic_token_count=11)
    assert state.shape == delta.shape == context.shape == (5, 15 * 32)
    assert torch.equal(state[0], state[-1])


def test_v48_102_version_and_name():
    assert ENGINEERING_VERSION == "v48.102.0-OC-AITS"
    assert "Action-Information Transport" in ALGORITHM_NAME
