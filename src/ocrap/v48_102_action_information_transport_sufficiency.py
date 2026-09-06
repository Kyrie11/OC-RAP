from __future__ import annotations

"""V48.102 OC-AITS: Stage-I action-information transport sufficiency audit.

Audit-only helpers.  V48.101 improves support-action semantics by opening the
root cross-attention kernel, but fails the joint State+Support+Reserve gate and
regresses the Near static state coordinate.  The preregistered next question is
therefore upstream: does the frozen Stage-I structured memory already contain a
jointly decodable recovery state and candidate-induced support/reserve response,
or is the missing control-sufficient statistic absent before the root decoder?

This module does not modify the planner.  It constructs a deterministic,
architecture-aware summary of the frozen structured memory and keeps the same
V48.93 target-specific state/support/reserve semantics and within-group action
permutation control used by V48.96.
"""

import torch

ENGINEERING_VERSION = "v48.102.0-OC-AITS"
ALGORITHM_NAME = "Observation-Consistent Action-Information Transport Sufficiency Audit"


def _moments(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.ndim != 3 or tokens.shape[1] <= 0:
        raise ValueError("V48.102 token moments require [B,T,D] with T>0")
    x = tokens.float()
    mean = x.mean(dim=1)
    std = x.var(dim=1, unbiased=False).clamp_min(0.0).sqrt()
    vmax = x.amax(dim=1)
    vmin = x.amin(dim=1)
    return torch.cat([mean, std, vmax, vmin], dim=-1)


def stage_i_memory_summary(
    memory: torch.Tensor,
    *,
    semantic_token_count: int = 11,
) -> torch.Tensor:
    """Deterministic summary of frozen Stage-I structured memory.

    StructuredTokenEncoder has a fixed leading block consisting of CLS plus ten
    semantic tokens (ego, prefix, macro/scalars, prefix-state, control, agent
    summary, BEV, route, map, dynamics), followed by a set of agent tokens.

    We keep the fixed semantic positions explicitly and summarize only the agent
    set with permutation-invariant moments.  This preserves action-relevant token
    identity without introducing a learned probe encoder or an agent-slot ID.
    """
    if memory.ndim != 3:
        raise ValueError("V48.102 memory must have shape [B,T,D]")
    B, T, D = memory.shape
    s = int(semantic_token_count)
    if s < 1 or T < s:
        raise ValueError(f"V48.102 semantic token contract invalid: T={T}, semantic={s}")
    fixed = memory[:, :s, :].float().reshape(B, s * D)
    if T == s:
        # MLP/synthetic fallback: keep feature dimension deterministic.
        agent_stats = memory.new_zeros((B, 4 * D), dtype=torch.float32)
    else:
        agent_stats = _moments(memory[:, s:, :])
    return torch.cat([fixed, agent_stats], dim=-1)


def stage_i_action_features(
    memory: torch.Tensor,
    *,
    semantic_token_count: int = 11,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return nominal state, candidate delta and state-conditioned delta.

    Row 0 is nominal and rows 1.. are candidates from the same scene-time group.
    The contract is action-relative but does not assume any latent-root slot
    correspondence.  The context construction deliberately mirrors V48.96 so
    the support/reserve observability gates are comparable across layers.
    """
    if memory.ndim != 3 or memory.shape[0] < 2:
        raise ValueError("V48.102 requires nominal + at least one candidate memory")
    z = stage_i_memory_summary(memory, semantic_token_count=semantic_token_count)
    nominal = z[0:1]
    candidate = z[1:]
    delta = candidate - nominal
    state = nominal.expand(candidate.shape[0], -1)
    context = delta * (1.0 + torch.tanh(state))
    return state, delta, context


def candidate_zero_delta_check(d_model: int = 16) -> bool:
    torch.manual_seed(48102)
    memory = torch.randn(4, 15, d_model)
    memory[1] = memory[0]
    state, delta, context = stage_i_action_features(memory, semantic_token_count=11)
    return bool(
        torch.equal(state[0], state[1])
        and torch.count_nonzero(delta[0]).item() == 0
        and torch.count_nonzero(context[0]).item() == 0
    )


def agent_permutation_invariance_check(d_model: int = 16) -> bool:
    torch.manual_seed(48103)
    memory = torch.randn(3, 19, d_model)
    a = stage_i_memory_summary(memory, semantic_token_count=11)
    perm = torch.tensor([7, 0, 5, 2, 1, 6, 3, 4])
    shuffled = memory.clone()
    shuffled[:, 11:, :] = shuffled[:, 11:, :][:, perm, :]
    b = stage_i_memory_summary(shuffled, semantic_token_count=11)
    return bool(torch.allclose(a, b, atol=1.0e-7, rtol=0.0))


def semantic_position_sensitivity_check(d_model: int = 16) -> bool:
    """Fixed semantic positions are intentionally not permutation invariant."""
    torch.manual_seed(48104)
    memory = torch.randn(3, 19, d_model)
    a = stage_i_memory_summary(memory, semantic_token_count=11)
    shuffled = memory.clone()
    shuffled[:, [1, 2], :] = shuffled[:, [2, 1], :]
    b = stage_i_memory_summary(shuffled, semantic_token_count=11)
    return bool(not torch.equal(a, b))
