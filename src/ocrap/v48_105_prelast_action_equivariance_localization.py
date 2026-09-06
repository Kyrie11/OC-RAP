from __future__ import annotations

"""V48.105 OC-PAEL: pre-last action-equivariance/localization audit.

V48.104 proves that a nominal-invariant residual can improve both registered
candidate-minus-nominal losses without changing the nominal recovery state, but
Support and Reserve still fail the absolute cross-cell gates.  The preregistered
next question is therefore *where* the missing response geometry lives before
that trainable last Stage-I block.

This module is audit-only.  It reuses the exact V48.102 architecture-aware
summary and probe family, but applies them one transformer block earlier.  This
isolates the historical last Stage-I block without opening any new model
capacity.  The summary is partitioned into fixed architectural groups so token
localization is reported without post-hoc feature selection.
"""

import torch

from ocrap.v48_102_action_information_transport_sufficiency import (
    agent_permutation_invariance_check as _v102_agent_permutation_check,
    stage_i_action_features,
    stage_i_memory_summary,
)

ENGINEERING_VERSION = "v48.105.0-OC-PAEL"
ALGORITHM_NAME = "Observation-Consistent Pre-Last Action-Equivariance Localization Audit"
SEMANTIC_TOKEN_COUNT = 11

# The V48.102 summary is [11 fixed semantic tokens, agent mean/std/max/min].
# Fixed-token positions are:
# 0 CLS
# 1 ego
# 2 prefix_param
# 3 macro+scalar
# 4 prefix_state
# 5 control
# 6 agent_summary
# 7 BEV
# 8 route
# 9 map
# 10 dynamics
TOKEN_GROUP_ORDER = ("cls", "ego_history", "control", "scene_context", "agents")


def prelast_memory_summary(memory: torch.Tensor) -> torch.Tensor:
    return stage_i_memory_summary(memory, semantic_token_count=SEMANTIC_TOKEN_COUNT)


def prelast_action_features(memory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return stage_i_action_features(memory, semantic_token_count=SEMANTIC_TOKEN_COUNT)


def summary_group_slices(d_model: int) -> dict[str, slice]:
    d = int(d_model)
    if d <= 0:
        raise ValueError("d_model must be positive")
    # 15*d total: 11 fixed positions + 4 agent-set moments.
    return {
        "cls": slice(0 * d, 1 * d),
        "ego_history": slice(1 * d, 5 * d),
        "control": slice(5 * d, 6 * d),
        "scene_context": slice(6 * d, 11 * d),
        "agents": slice(11 * d, 15 * d),
    }


def action_interaction_slice(d_model: int) -> slice:
    """Predeclared response-carrying subspace.

    It contains the candidate control token plus scene/interaction context and
    permutation-invariant agent-set moments, while excluding CLS and nominal
    ego/history coordinates.  This is one fixed hypothesis, not a searched
    subset.
    """
    d = int(d_model)
    if d <= 0:
        raise ValueError("d_model must be positive")
    return slice(5 * d, 15 * d)


def select_summary_group(x: torch.Tensor, group: str, d_model: int) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError("summary features must be [B,D]")
    groups = summary_group_slices(d_model)
    if group == "action_interaction":
        sl = action_interaction_slice(d_model)
    else:
        if group not in groups:
            raise KeyError(group)
        sl = groups[group]
    if x.shape[1] != 15 * int(d_model):
        raise ValueError(f"summary dimension mismatch: {x.shape[1]} != {15 * int(d_model)}")
    return x[:, sl]


def summary_partition_check(d_model: int = 16) -> bool:
    groups = summary_group_slices(d_model)
    spans = []
    for name in TOKEN_GROUP_ORDER:
        sl = groups[name]
        spans.extend(range(int(sl.start), int(sl.stop)))
    return spans == list(range(15 * int(d_model)))


def candidate_zero_delta_check(d_model: int = 16) -> bool:
    torch.manual_seed(48105)
    memory = torch.randn(4, 19, d_model)
    memory[1] = memory[0]
    state, delta, context = prelast_action_features(memory)
    return bool(
        torch.equal(state[0], state[1])
        and torch.count_nonzero(delta[0]).item() == 0
        and torch.count_nonzero(context[0]).item() == 0
    )


def agent_permutation_invariance_check(d_model: int = 16) -> bool:
    # Same operator as V48.102; keep an explicit V48.105 contract entry.
    return bool(_v102_agent_permutation_check(d_model))


def action_interaction_dimension_check(d_model: int = 192) -> bool:
    sl = action_interaction_slice(d_model)
    return int(sl.stop - sl.start) == 10 * int(d_model)
