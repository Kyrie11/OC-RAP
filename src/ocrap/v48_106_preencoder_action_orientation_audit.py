from __future__ import annotations

"""V48.106 OC-PEAO: pre-encoder signed action-orientation audit.

V48.105 shows that the token set immediately before the historical final
Stage-I Transformer block does not expose a transferable State+Support+Reserve
statistic.  At the same time, magnitude-only localization (linear CKA) can be
strong where the held-out signed action ranking is wrong, so association is not
sufficient evidence for transferable orientation.

V48.106 moves exactly one Transformer block earlier, to the Stage-I token set
before *any* Transformer layer, while preserving the exact V48.102 summary,
linear-probe family, target-specific semantics and within-group permutation
null.  Signed train-to-held-out mean-difference cosine is reported only as a
fixed diagnostic; it never substitutes for the preregistered AUC gates.
"""

import numpy as np
import torch

from ocrap.v48_102_action_information_transport_sufficiency import (
    agent_permutation_invariance_check as _v102_agent_permutation_check,
    stage_i_action_features,
    stage_i_memory_summary,
)
from ocrap.v48_105_prelast_action_equivariance_localization import (
    TOKEN_GROUP_ORDER,
    action_interaction_dimension_check,
    action_interaction_slice,
    summary_group_slices,
    summary_partition_check,
)

ENGINEERING_VERSION = "v48.106.0-OC-PEAO"
ALGORITHM_NAME = "Observation-Consistent Pre-Encoder Action-Orientation Audit"
SEMANTIC_TOKEN_COUNT = 11
ORIENTATION_GROUP_ORDER = TOKEN_GROUP_ORDER + ("action_interaction",)


def preencoder_memory_summary(memory: torch.Tensor) -> torch.Tensor:
    return stage_i_memory_summary(memory, semantic_token_count=SEMANTIC_TOKEN_COUNT)


def preencoder_action_features(memory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return stage_i_action_features(memory, semantic_token_count=SEMANTIC_TOKEN_COUNT)


def candidate_zero_delta_check(d_model: int = 16) -> bool:
    torch.manual_seed(48106)
    memory = torch.randn(4, 19, d_model)
    memory[1] = memory[0]
    state, delta, context = preencoder_action_features(memory)
    return bool(
        torch.equal(state[0], state[1])
        and torch.count_nonzero(delta[0]).item() == 0
        and torch.count_nonzero(context[0]).item() == 0
    )


def agent_permutation_invariance_check(d_model: int = 16) -> bool:
    return bool(_v102_agent_permutation_check(d_model))


def mean_difference_direction(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    if X.ndim != 2 or len(X) != len(y):
        raise ValueError("mean_difference_direction shape mismatch")
    pos, neg = X[y == 1], X[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.zeros(X.shape[1], dtype=np.float64)
    return pos.mean(axis=0) - neg.mean(axis=0)


def signed_orientation_cosine(train_direction: np.ndarray, heldout_direction: np.ndarray) -> float | None:
    a = np.asarray(train_direction, dtype=np.float64).reshape(-1)
    b = np.asarray(heldout_direction, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("signed orientation direction mismatch")
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den <= 1.0e-12:
        return None
    return float(np.dot(a, b) / den)


def signed_orientation_sign_flip_check() -> bool:
    # A deterministic contract: same direction -> +1, reversed direction -> -1.
    a = np.asarray([1.0, -2.0, 0.5], dtype=np.float64)
    p = signed_orientation_cosine(a, 3.0 * a)
    n = signed_orientation_cosine(a, -2.0 * a)
    return bool(p is not None and n is not None and abs(p - 1.0) < 1.0e-12 and abs(n + 1.0) < 1.0e-12)


def preencoder_contract_checks(d_model: int = 192) -> dict[str, bool]:
    return {
        "summary_partition_15d": bool(summary_partition_check(16)),
        "candidate_nominal_zero_delta": bool(candidate_zero_delta_check(16)),
        "agent_set_permutation_invariant": bool(agent_permutation_invariance_check(16)),
        "action_interaction_dimension_1920_at_d192": bool(action_interaction_dimension_check(d_model)),
        "signed_orientation_sign_flip_detected": bool(signed_orientation_sign_flip_check()),
    }
