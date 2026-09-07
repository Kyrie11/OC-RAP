from __future__ import annotations

"""V48.107 OC-FNAO: first-block nominal-invariant action orientation.

V48.106 closes the frozen Stage-I location audit: before any Transformer layer,
full State/Support/Reserve control sufficiency is absent, yet the fixed
pre-registered action-interaction subspace already carries a transferable
reserve/debt signal while support orientation remains incomplete.  V48.107 is
therefore the preregistered first-Stage-I-block experiment.

The historical first Transformer block is adapted through a counterfactual
residual that is centered on the unique nominal action in each scene-time group:

    H1_theta(a) = H1_0(a) + [E_theta(a) - E_theta(a0)]

where E_theta(a)=B_theta(H0(a))-B_0(H0(a)).  Hence H1_theta(a0)=H1_0(a0)
exactly, and the frozen second block plus the frozen V48.103 readout preserve
the nominal recovery state by construction.

Unlike V48.104, training targets *within-group ordinal action orientation* rather
than calibrated response magnitude.  For each response axis, every pair of
rows with different teacher response is constrained to have the same ordering
in the predicted response.  This directly matches action selection/AUC/top-1
semantics and is invariant to population-specific monotone calibration shifts.
No learned token router, new head, rank/width, regime id, source residual or
loss-weight sweep is introduced.
"""

import copy
from typing import Sequence

import torch
from torch import nn

ENGINEERING_VERSION = "v48.107.0-OC-FNAO"
ALGORITHM_NAME = "Observation-Consistent First-Block Nominal-Invariant Action Orientation"


def trainable_parameter_count(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


class NominalInvariantFirstBlockOrientation(nn.Module):
    """Adapt only historical Stage-I block 1; freeze block 2 and final norm."""

    def __init__(self, base_first_layer: nn.Module, frozen_tail_layers: Sequence[nn.Module], final_norm: nn.Module):
        super().__init__()
        self.base_first = copy.deepcopy(base_first_layer)
        self.adapted_first = copy.deepcopy(base_first_layer)
        self.frozen_tail = nn.ModuleList([copy.deepcopy(x) for x in frozen_tail_layers])
        self.final_norm = copy.deepcopy(final_norm)
        for p in self.base_first.parameters():
            p.requires_grad_(False)
        for layer in self.frozen_tail:
            for p in layer.parameters():
                p.requires_grad_(False)
        for p in self.final_norm.parameters():
            p.requires_grad_(False)
        for p in self.adapted_first.parameters():
            p.requires_grad_(True)
        self.train(False)

    @property
    def parameter_count(self) -> int:
        return trainable_parameter_count(self)

    def train(self, mode: bool = True):
        # Deterministic contract: all dropout remains disabled while gradients
        # flow through adapted_first.  This keeps exact baseline identities.
        super().train(False)
        self.base_first.eval(); self.adapted_first.eval(); self.final_norm.eval()
        for layer in self.frozen_tail:
            layer.eval()
        return self

    def base_after_first(self, input_tokens: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.base_first(input_tokens)

    def tail_memory(self, after_first: torch.Tensor) -> torch.Tensor:
        h = after_first
        for layer in self.frozen_tail:
            h = layer(h)
        return self.final_norm(h)

    def base_memory(self, input_tokens: torch.Tensor) -> torch.Tensor:
        return self.tail_memory(self.base_after_first(input_tokens))

    def refined_after_first(
        self,
        input_tokens: torch.Tensor,
        nominal_index: torch.Tensor,
        base_after_first: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_tokens.ndim != 3:
            raise ValueError("input_tokens must be [B,T,D]")
        ni = nominal_index.long().to(input_tokens.device)
        if ni.ndim != 1 or ni.shape[0] != input_tokens.shape[0]:
            raise ValueError("nominal_index must be [B]")
        if int(ni.min()) < 0 or int(ni.max()) >= input_tokens.shape[0]:
            raise ValueError("nominal_index out of range")
        base1 = self.base_after_first(input_tokens) if base_after_first is None else base_after_first.to(
            device=input_tokens.device, dtype=input_tokens.dtype
        )
        adapted1 = self.adapted_first(input_tokens)
        residual = adapted1 - base1
        anchor = residual.index_select(0, ni)
        centered = residual - anchor
        return base1 + centered

    def refined_memory(
        self,
        input_tokens: torch.Tensor,
        nominal_index: torch.Tensor,
        base_after_first: torch.Tensor | None = None,
    ) -> torch.Tensor:
        base1 = self.base_after_first(input_tokens) if base_after_first is None else base_after_first.to(
            device=input_tokens.device, dtype=input_tokens.dtype
        )
        refined1 = self.refined_after_first(input_tokens, nominal_index, base1)
        out = self.tail_memory(refined1)
        # The frozen tail is mathematically sample-wise, but batched GEMM/attention
        # kernels are not guaranteed bit-identical when other batch rows change.
        # Enforce the preregistered nominal identity at the final-memory interface
        # by copying the historical frozen-tail output for nominal rows exactly.
        # Candidate rows keep the differentiable refined path unchanged.
        with torch.no_grad():
            base_final = self.tail_memory(base1)
        ni = nominal_index.long().to(input_tokens.device)
        is_nom = torch.arange(input_tokens.shape[0], device=input_tokens.device) == ni
        if bool(is_nom.any()):
            out = torch.where(is_nom[:, None, None], base_final, out)
        return out

    def nominal_identity_error(
        self,
        input_tokens: torch.Tensor,
        nominal_index: torch.Tensor,
        base_after_first: torch.Tensor | None = None,
    ) -> float:
        base1 = self.base_after_first(input_tokens) if base_after_first is None else base_after_first.to(input_tokens.device)
        base = self.tail_memory(base1)
        out = self.refined_memory(input_tokens, nominal_index, base1)
        ni = nominal_index.long().to(input_tokens.device)
        is_nom = torch.arange(input_tokens.shape[0], device=input_tokens.device) == ni
        if not bool(is_nom.any()):
            return 0.0
        return float((out[is_nom] - base[is_nom]).abs().max().item())


def _pairwise_axis_orientation_sum(
    prediction: torch.Tensor,
    teacher: torch.Tensor,
    group_ids: torch.Tensor,
    scale: float | torch.Tensor,
    tie_eps: float = 1.0e-6,
) -> tuple[torch.Tensor, int]:
    """Ordinal logistic loss over all non-tied within-group row pairs.

    The target magnitude is deliberately discarded after deciding the sign of
    the ordering.  This makes the objective about signed action orientation,
    not population-specific calibration.
    """
    p = prediction.float().reshape(-1)
    t = teacher.float().reshape(-1)
    g = group_ids.long().reshape(-1)
    if not (len(p) == len(t) == len(g)):
        raise ValueError("orientation loss shape mismatch")
    sc = torch.as_tensor(scale, dtype=p.dtype, device=p.device).clamp_min(1.0e-6)
    pieces: list[torch.Tensor] = []
    count = 0
    for gid in torch.unique(g, sorted=True):
        idx = torch.where(g == gid)[0]
        if idx.numel() < 2:
            continue
        for a in range(int(idx.numel())):
            i = idx[a]
            for b in range(a + 1, int(idx.numel())):
                j = idx[b]
                dt = t[i] - t[j]
                if float(dt.abs().detach().item()) <= tie_eps:
                    continue
                sign = torch.sign(dt).detach()
                margin = sign * (p[i] - p[j]) / sc
                pieces.append(torch.nn.functional.softplus(-margin))
                count += 1
    if count == 0:
        return p.sum() * 0.0, 0
    return torch.stack(pieces).sum(), count


def ordinal_action_orientation_loss_sum(
    support: torch.Tensor,
    reserve: torch.Tensor,
    teacher_support: torch.Tensor,
    teacher_reserve: torch.Tensor,
    group_ids: torch.Tensor,
    scales: dict[str, float | torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor | int]]:
    """Equal-weight support/reserve ordinal orientation objective."""
    ls, ns = _pairwise_axis_orientation_sum(
        support, teacher_support.clamp(0.0, 1.0), group_ids, scales["delta_support"]
    )
    lr, nr = _pairwise_axis_orientation_sum(
        reserve, teacher_reserve, group_ids, scales["delta_reserve"]
    )
    if ns <= 0 or nr <= 0:
        raise ValueError(f"orientation objective requires both axes, got support_pairs={ns} reserve_pairs={nr}")
    ms = ls / float(ns)
    mr = lr / float(nr)
    return (ms + mr) / 2.0, {
        "support_orientation_sum": ls,
        "reserve_orientation_sum": lr,
        "support_pairs": ns,
        "reserve_pairs": nr,
        "support_orientation_mean": ms,
        "reserve_orientation_mean": mr,
    }


def initialization_identity_check(d_model: int = 16, nhead: int = 4) -> bool:
    torch.manual_seed(48107)
    first = nn.TransformerEncoderLayer(
        d_model=d_model, nhead=nhead, dim_feedforward=4*d_model, dropout=.1,
        batch_first=True, activation="gelu", norm_first=True,
    )
    second = nn.TransformerEncoderLayer(
        d_model=d_model, nhead=nhead, dim_feedforward=4*d_model, dropout=.1,
        batch_first=True, activation="gelu", norm_first=True,
    )
    m = NominalInvariantFirstBlockOrientation(first, [second], nn.LayerNorm(d_model)).eval()
    x = torch.randn(6, 9, d_model)
    ni = torch.tensor([0, 0, 0, 3, 3, 3])
    base1 = m.base_after_first(x)
    base = m.tail_memory(base1)
    out = m.refined_memory(x, ni, base1)
    return bool(
        torch.allclose(out, base, atol=1.0e-6, rtol=0.0)
        and torch.equal(out[[0, 3]], base[[0, 3]])
        and m.nominal_identity_error(x, ni, base1) == 0.0
    )


def orientation_loss_sign_check() -> bool:
    # Correct ordering must have lower loss than reversed ordering.
    support = torch.tensor([0.2, 0.8, 0.1, 0.9])
    reserve = torch.tensor([-1.0, 1.0, -0.5, 0.5])
    ts = torch.tensor([0.0, 1.0, 0.0, 1.0])
    tr = torch.tensor([-1.0, 1.0, -0.5, 0.5])
    g = torch.tensor([0, 0, 1, 1])
    scales = {"delta_support": 1.0, "delta_reserve": 1.0}
    good, _ = ordinal_action_orientation_loss_sum(support, reserve, ts, tr, g, scales)
    bad, _ = ordinal_action_orientation_loss_sum(-support, -reserve, ts, tr, g, scales)
    return bool(float(good) < float(bad))
