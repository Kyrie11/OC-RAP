from __future__ import annotations

"""V48.104 OC-NICR: nominal-invariant control refinement.

V48.103 shows that a factorized readout over the full frozen Stage-I memory can
recover the nominal recovery-state axis, but it cannot make support/reserve
response simultaneously stable.  In particular the response branch is still
free to trade counterfactual fidelity against absolute candidate fit.

V48.104 is the preregistered last-Stage-I-block experiment.  It keeps the
successful V48.103 readout frozen and adapts exactly one existing Stage-I
Transformer block through a counterfactual residual:

    M_theta(a) = M_0(a) + [R_theta(a) - R_theta(a0)]

where R_theta is the change produced by an adapted copy of the historical last
encoder block relative to its frozen base copy.  Therefore M_theta(a0)=M_0(a0)
exactly for the unique nominal action.  Static sufficiency is preserved by
construction; the only learned degree of freedom is candidate-induced token
refinement.  Training uses only the registered delta-support and delta-reserve
terms, so absolute-state fit cannot buy a lower loss by corrupting the response
coordinate.
"""

import copy
from typing import Iterable

import torch
from torch import nn

ENGINEERING_VERSION = "v48.104.0-OC-NICR"
ALGORITHM_NAME = "Observation-Consistent Nominal-Invariant Control Refinement"


def trainable_parameter_count(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


class NominalInvariantLastBlockRefinement(nn.Module):
    """Counterfactual residual adaptation of exactly one Transformer block."""

    def __init__(self, base_last_layer: nn.Module, final_norm: nn.Module):
        super().__init__()
        self.base_last = copy.deepcopy(base_last_layer)
        self.adapted_last = copy.deepcopy(base_last_layer)
        self.final_norm = copy.deepcopy(final_norm)
        for p in self.base_last.parameters():
            p.requires_grad_(False)
        for p in self.final_norm.parameters():
            p.requires_grad_(False)
        for p in self.adapted_last.parameters():
            p.requires_grad_(True)
        self.base_last.eval(); self.adapted_last.eval(); self.final_norm.eval()

    @property
    def parameter_count(self) -> int:
        return trainable_parameter_count(self)

    def train(self, mode: bool = True):
        # Deterministic fine-tuning contract: historical Stage-I semantic
        # experiments use eval-mode frozen features.  Gradients are valid in
        # eval mode; keeping dropout disabled makes initialization/function
        # identity and cached pre-last activations exact.
        super().train(False)
        self.base_last.eval(); self.adapted_last.eval(); self.final_norm.eval()
        return self

    def base_raw(self, prelast: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.base_last(prelast)

    def base_memory(self, prelast: torch.Tensor) -> torch.Tensor:
        return self.final_norm(self.base_raw(prelast))

    def refined_memory(
        self,
        prelast: torch.Tensor,
        nominal_index: torch.Tensor,
        base_raw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if prelast.ndim != 3:
            raise ValueError("prelast must be [B,T,D]")
        ni = nominal_index.long().to(prelast.device)
        if ni.ndim != 1 or ni.shape[0] != prelast.shape[0]:
            raise ValueError("nominal_index must be [B]")
        if int(ni.min()) < 0 or int(ni.max()) >= prelast.shape[0]:
            raise ValueError("nominal_index out of range")
        if base_raw is None:
            base_raw = self.base_raw(prelast)
        else:
            base_raw = base_raw.to(device=prelast.device, dtype=prelast.dtype)
        adapted_raw = self.adapted_last(prelast)
        residual = adapted_raw - base_raw
        anchor = residual.index_select(0, ni)
        # Apply the historical final norm once after counterfactual centering.
        # At initialization residual is exactly zero; for nominal rows the
        # centered residual is exactly zero for every parameter value.
        return self.final_norm(base_raw + residual - anchor)

    def nominal_identity_error(
        self,
        prelast: torch.Tensor,
        nominal_index: torch.Tensor,
        base_raw: torch.Tensor | None = None,
    ) -> float:
        raw = self.base_raw(prelast) if base_raw is None else base_raw.to(prelast.device)
        base = self.final_norm(raw)
        out = self.refined_memory(prelast, nominal_index, raw)
        ni = nominal_index.long().to(prelast.device)
        is_nom = torch.arange(prelast.shape[0], device=prelast.device) == ni
        if not bool(is_nom.any()):
            return 0.0
        return float((out[is_nom] - base[is_nom]).abs().max().item())


def response_only_loss(
    support: torch.Tensor,
    reserve: torch.Tensor,
    teacher_support: torch.Tensor,
    teacher_reserve: torch.Tensor,
    candidate_index: torch.Tensor,
    nominal_index: torch.Tensor,
    scales: dict[str, float | torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Registered candidate-minus-nominal objective; no static loss term."""
    ci = candidate_index.long(); ni = nominal_index.long()
    if ci.numel() == 0:
        z = support.sum() * 0.0
        return z, {"delta_support_normalized": z, "delta_reserve_normalized": z}
    s = support.float(); r = reserve.float()
    td = teacher_support.float().clamp(0.0, 1.0); tr = teacher_reserve.float()
    ds = (s.index_select(0, ci) - s.index_select(0, ni)) - (td.index_select(0, ci) - td.index_select(0, ni))
    dr = (r.index_select(0, ci) - r.index_select(0, ni)) - (tr.index_select(0, ci) - tr.index_select(0, ni))

    def huber(err: torch.Tensor, scale: float | torch.Tensor) -> torch.Tensor:
        sc = torch.as_tensor(scale, dtype=err.dtype, device=err.device).clamp_min(1.0e-6)
        z = err / sc
        return torch.nn.functional.smooth_l1_loss(z, torch.zeros_like(z), beta=1.0)

    l_ds = huber(ds, scales["delta_support"])
    l_dr = huber(dr, scales["delta_reserve"])
    return (l_ds + l_dr) / 2.0, {
        "delta_support_normalized": l_ds,
        "delta_reserve_normalized": l_dr,
    }


def response_only_loss_sum(
    support: torch.Tensor,
    reserve: torch.Tensor,
    teacher_support: torch.Tensor,
    teacher_reserve: torch.Tensor,
    candidate_index: torch.Tensor,
    nominal_index: torch.Tensor,
    scales: dict[str, float | torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor], int]:
    """Sum-reduction version for exact full-dataset gradient accumulation."""
    ci = candidate_index.long(); ni = nominal_index.long(); n = int(ci.numel())
    if n == 0:
        z = support.sum() * 0.0
        return z, {"delta_support_sum": z, "delta_reserve_sum": z}, 0
    s = support.float(); r = reserve.float()
    td = teacher_support.float().clamp(0.0, 1.0); tr = teacher_reserve.float()
    ds = (s.index_select(0, ci) - s.index_select(0, ni)) - (td.index_select(0, ci) - td.index_select(0, ni))
    dr = (r.index_select(0, ci) - r.index_select(0, ni)) - (tr.index_select(0, ci) - tr.index_select(0, ni))

    def hs(err: torch.Tensor, scale: float | torch.Tensor) -> torch.Tensor:
        sc = torch.as_tensor(scale, dtype=err.dtype, device=err.device).clamp_min(1.0e-6)
        z = err / sc
        return torch.nn.functional.smooth_l1_loss(z, torch.zeros_like(z), beta=1.0, reduction="sum")

    a = hs(ds, scales["delta_support"]); b = hs(dr, scales["delta_reserve"])
    return (a + b) / (2.0 * float(n)), {"delta_support_sum": a, "delta_reserve_sum": b}, n


def initialization_identity_check(d_model: int = 16, nhead: int = 4) -> bool:
    torch.manual_seed(48104)
    layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4*d_model,
                                       dropout=0.1, batch_first=True, activation="gelu", norm_first=True)
    norm = nn.LayerNorm(d_model)
    m = NominalInvariantLastBlockRefinement(layer, norm).eval()
    pre = torch.randn(6, 9, d_model)
    ni = torch.tensor([0,0,0,3,3,3])
    raw = m.base_raw(pre); base = m.final_norm(raw)
    out = m.refined_memory(pre, ni, raw)
    return bool(torch.allclose(out, base, atol=1.0e-6, rtol=0.0) and m.nominal_identity_error(pre, ni, raw) == 0.0)
