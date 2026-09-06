from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

ENGINEERING_VERSION = "v48.97.2-OC-ERSS-STRATAFIX"
ALGORITHM_NAME = "Observation-Consistent Executable-Recovery Sufficient State"


class ExecutableRecoverySufficientState(nn.Module):
    """Minimal two-coordinate representation over an observation-compatible root set.

    The module does *not* output an admission logit.  It learns exactly two
    decision-semantic coordinates from a frozen latent-root set:

    * shared executable-recovery support D in [0, 1];
    * signed deployability reserve/debt R in R.

    Two learned semantic queries attend to the unordered root set.  Root
    probability is used only as the observation-consistent base measure.  No
    root id, option id, regime id, teacher metadata, proposal rank, or external
    future is consumed.

    The representation is intentionally tiny (4*d_model + 2 scalars) and has no
    hidden MLP or tunable rank/width.  This is a representation-learning
    adjudication after V48.96, not a new absolute-source head.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = int(d_model)
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        self.root_norm = nn.LayerNorm(self.d_model, elementwise_affine=False)
        self.support_query = nn.Parameter(torch.zeros(self.d_model, dtype=torch.float32))
        self.reserve_query = nn.Parameter(torch.zeros(self.d_model, dtype=torch.float32))
        self.support_readout = nn.Parameter(torch.zeros(self.d_model, dtype=torch.float32))
        self.reserve_readout = nn.Parameter(torch.zeros(self.d_model, dtype=torch.float32))
        self.support_bias = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.reserve_bias = nn.Parameter(torch.zeros((), dtype=torch.float32))
        # Small isotropic init prevents symmetry from making both attention
        # queries identical while remaining capacity-fixed and deterministic.
        nn.init.normal_(self.support_query, mean=0.0, std=1.0 / math.sqrt(self.d_model))
        nn.init.normal_(self.reserve_query, mean=0.0, std=1.0 / math.sqrt(self.d_model))
        nn.init.normal_(self.support_readout, mean=0.0, std=1.0 / math.sqrt(self.d_model))
        nn.init.normal_(self.reserve_readout, mean=0.0, std=1.0 / math.sqrt(self.d_model))

    @property
    def trainable_parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def _pool(
        self,
        root_tokens: torch.Tensor,
        root_probs: torch.Tensor,
        root_valid: torch.Tensor,
        query: torch.Tensor,
    ) -> torch.Tensor:
        if root_tokens.ndim != 3:
            raise ValueError("root_tokens must be [B,K,D]")
        if root_tokens.shape[-1] != self.d_model:
            raise ValueError("root token dimension mismatch")
        if root_probs.shape != root_tokens.shape[:2] or root_valid.shape != root_tokens.shape[:2]:
            raise ValueError("root_probs/root_valid shape mismatch")
        r = self.root_norm(root_tokens.float())
        valid = root_valid.bool()
        p = root_probs.float().clamp_min(0.0) * valid.float()
        p = p / p.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
        # Probability-weighted semantic attention.  log(p) makes the attention
        # invariant to root-slot permutations and respects the frozen OC-MERO
        # observation-compatible root measure.
        score = torch.einsum("bkd,d->bk", r, query.float()) / math.sqrt(self.d_model)
        score = score + torch.log(p.clamp_min(1.0e-12))
        score = score.masked_fill(~valid, -1.0e9)
        a = torch.softmax(score, dim=-1) * valid.float()
        a = a / a.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
        return torch.einsum("bk,bkd->bd", a, r)

    def forward(
        self,
        root_tokens: torch.Tensor,
        root_probs: torch.Tensor,
        root_valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        hs = self._pool(root_tokens, root_probs, root_valid, self.support_query)
        hr = self._pool(root_tokens, root_probs, root_valid, self.reserve_query)
        support_logit = (
            torch.einsum("bd,d->b", hs, self.support_readout.float()) / math.sqrt(self.d_model)
            + self.support_bias.float()
        )
        reserve = (
            torch.einsum("bd,d->b", hr, self.reserve_readout.float()) / math.sqrt(self.d_model)
            + self.reserve_bias.float()
        )
        return {
            "support_logit": support_logit,
            "support": torch.sigmoid(support_logit),
            "reserve_debt": reserve,
            "support_state": hs,
            "reserve_state": hr,
        }


def semantic_loss(
    out: dict[str, torch.Tensor],
    teacher_drs: torch.Tensor,
    teacher_r_dep: torch.Tensor,
    candidate_index: torch.Tensor,
    nominal_index: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Equal-weight dense state + counterfactual-response representation loss.

    Every term is a semantic representation target, not an admission target:
    absolute DRS, absolute signed R_dep, candidate-minus-nominal DRS change, and
    candidate-minus-nominal R_dep change.  Equal weighting is fixed by contract;
    there is no component-weight sweep.
    """
    support = out["support"].float()
    reserve = out["reserve_debt"].float()
    td = teacher_drs.float().clamp(0.0, 1.0)
    tr = teacher_r_dep.float()
    l_support = torch.nn.functional.smooth_l1_loss(support, td, beta=1.0)
    l_reserve = torch.nn.functional.smooth_l1_loss(reserve, tr, beta=1.0)
    if candidate_index.numel() > 0:
        ci = candidate_index.long()
        ni = nominal_index.long()
        l_d_support = torch.nn.functional.smooth_l1_loss(
            support.index_select(0, ci) - support.index_select(0, ni),
            td.index_select(0, ci) - td.index_select(0, ni),
            beta=1.0,
        )
        l_d_reserve = torch.nn.functional.smooth_l1_loss(
            reserve.index_select(0, ci) - reserve.index_select(0, ni),
            tr.index_select(0, ci) - tr.index_select(0, ni),
            beta=1.0,
        )
    else:
        z = support.sum() * 0.0
        l_d_support = z
        l_d_reserve = z
    total = (l_support + l_reserve + l_d_support + l_d_reserve) / 4.0
    return total, {
        "support": l_support,
        "reserve": l_reserve,
        "delta_support": l_d_support,
        "delta_reserve": l_d_reserve,
    }


def root_permutation_invariance_check(d_model: int = 16) -> bool:
    torch.manual_seed(97)
    m = ExecutableRecoverySufficientState(d_model).eval()
    r = torch.randn(3, 7, d_model)
    p = torch.softmax(torch.randn(3, 7), dim=-1)
    v = torch.ones(3, 7, dtype=torch.bool)
    perm = torch.tensor([4, 0, 6, 2, 1, 5, 3])
    with torch.no_grad():
        a = m(r, p, v)
        b = m(r[:, perm], p[:, perm], v[:, perm])
    return bool(
        torch.allclose(a["support"], b["support"], atol=1e-6, rtol=0.0)
        and torch.allclose(a["reserve_debt"], b["reserve_debt"], atol=1e-6, rtol=0.0)
    )
