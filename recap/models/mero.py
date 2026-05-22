from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositiveLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.raw_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.xavier_uniform_(self.raw_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = F.softplus(self.raw_weight)
        return F.linear(x, w, self.bias)


class MonoCalibrator(nn.Module):
    """Constrained monotone MLP for signed evidence (-P,G,C,-U,-Kdef)."""

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(PositiveLinear(5, hidden), nn.Softplus(), PositiveLinear(hidden, hidden), nn.Softplus(), PositiveLinear(hidden, 1))

    def forward(self, signed_evidence: torch.Tensor) -> torch.Tensor:
        return self.net(signed_evidence).squeeze(-1)


def _default_mono_score(P, G, C, U, Kdef):
    # Signed monotone linear score: H is intentionally not an input.
    return 1.2 * (-P) + 1.0 * G + 0.8 * C + 0.6 * (-U) + 0.8 * (-Kdef)


def mono_option_score(P: torch.Tensor, G: torch.Tensor, C: torch.Tensor, U: torch.Tensor, Kdef: torch.Tensor, option_mask: torch.Tensor, calibrator: Optional[nn.Module] = None) -> torch.Tensor:
    """Return v [B,K,L,M]. H is forbidden here by function signature."""
    if U.dim() != 3:
        raise ValueError("U must be action-level [B,K,M], not option-level")
    Ue = U.unsqueeze(2).expand_as(P)
    if calibrator is None:
        v = _default_mono_score(P, G, C, Ue, Kdef)
    else:
        x = torch.stack([-P, G, C, -Ue, -Kdef], dim=-1)
        v = calibrator(x)
    mask = option_mask.bool().unsqueeze(-1).expand_as(v)
    return torch.where(mask, v, torch.full_like(v, -torch.inf))


def existential_option_aggregate(v: torch.Tensor, option_mask: torch.Tensor, tau_R: float = 0.25, c_R: float = 0.0) -> torch.Tensor:
    """Masked logsumexp over recovery options with valid-option normalization.

    c_R_eff = c_R + tau_R * log(N_valid_options) prevents duplicated options
    from increasing recoverability.
    """
    if v.dim() != 4:
        raise ValueError("v must have shape [B,K,L,M]")
    mask = option_mask.bool().unsqueeze(-1).expand_as(v)
    v_masked = torch.where(mask, v, torch.full_like(v, -torch.inf))
    lse = tau_R * torch.logsumexp(v_masked / max(tau_R, 1e-8), dim=2)  # [B,K,M]
    n_valid = option_mask.float().sum(dim=2).clamp_min(1.0)  # [B,K]
    c_eff = c_R + tau_R * torch.log(n_valid).unsqueeze(-1)
    logits = lse - c_eff
    pi_am = torch.sigmoid(logits)
    any_valid = option_mask.any(dim=2).unsqueeze(-1)
    return torch.where(any_valid, pi_am, torch.zeros_like(pi_am))


def _normalize_weights(weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return weights.clamp_min(0) / weights.clamp_min(0).sum(dim=-1, keepdim=True).clamp_min(eps)


def weighted_lcvar(values: torch.Tensor, weights: torch.Tensor, alpha: float = 0.2, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Lower-tail weighted CVaR along the final mode dimension."""
    if values.shape[-1] != weights.shape[-1]:
        raise ValueError("values and weights must share final mode dimension")
    orig = values.shape[:-1]
    M = values.shape[-1]
    vals = values.reshape(-1, M)
    if weights.dim() == 2 and values.dim() >= 3:
        # weights [B,M] -> broadcast over K/other dims
        repeat = int(vals.shape[0] / weights.shape[0])
        w = weights[:, None, :].expand(weights.shape[0], repeat, M).reshape(-1, M)
    else:
        w = weights.reshape(-1, M).expand_as(vals)
    if mask is not None:
        m = mask.reshape(-1, M).bool()
        w = torch.where(m, w, torch.zeros_like(w))
    w = _normalize_weights(w)
    order = torch.argsort(vals, dim=-1, descending=False)
    sorted_vals = torch.gather(vals, -1, order)
    sorted_w = torch.gather(w, -1, order)
    remaining = torch.full((vals.shape[0], 1), float(alpha), device=values.device, dtype=values.dtype)
    included = []
    for j in range(M):
        take = torch.minimum(sorted_w[:, j:j+1], remaining).clamp_min(0.0)
        included.append(take * sorted_vals[:, j:j+1])
        remaining = (remaining - take).clamp_min(0.0)
    out = torch.cat(included, dim=-1).sum(dim=-1) / max(alpha, 1e-8)
    return out.reshape(orig)


def upper_tail_cvar(values: torch.Tensor, weights: torch.Tensor, alpha: float = 0.2, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    return -weighted_lcvar(-values, weights, alpha, mask)


@dataclass
class MEROParams:
    tau_R: float = 0.25
    c_R: float = 0.0
    alpha_R: float = 0.20
    alpha_B: float = 0.20
    alpha_U: float = 0.20
    alpha_H: float = 0.20
    alpha_K: float = 0.20
    eta_G: float = 0.5
    eta_C: float = 0.5
    mean_over_options: bool = False


def _gather_witness(x: torch.Tensor, witness: torch.Tensor) -> torch.Tensor:
    # x [B,K,L,M], witness [B,K,M]
    idx = witness.unsqueeze(2)
    return torch.gather(x, 2, idx).squeeze(2)


def compute_profiles(pred: Dict[str, torch.Tensor], masks: Dict[str, torch.Tensor], params: MEROParams | dict | None = None, calibrator: Optional[nn.Module] = None) -> Dict[str, torch.Tensor]:
    if params is None:
        params = MEROParams()
    if isinstance(params, dict):
        params = MEROParams(**{k: v for k, v in params.items() if k in MEROParams.__annotations__})
    P, G, C = pred["P"], pred["G"], pred["C"]
    U, H = pred["U"], pred["H"]
    Kdef = pred.get("Kdef", pred.get("K_post"))
    mode_probs = pred.get("mode_probs")
    if mode_probs is None:
        mode_probs = torch.ones(P.shape[0], P.shape[-1], device=P.device, dtype=P.dtype) / P.shape[-1]
    action_mask = masks["action_mask"].bool()
    option_mask = masks["option_mask"].bool() & action_mask.unsqueeze(-1)

    v = mono_option_score(P, G, C, U, Kdef, option_mask, calibrator=calibrator)
    if params.mean_over_options:
        # Explicit ablation only.
        m = option_mask.float().unsqueeze(-1)
        pi_am = torch.sigmoid(torch.where(m.bool(), v, torch.zeros_like(v))).sum(dim=2) / m.sum(dim=2).clamp_min(1.0)
    else:
        pi_am = existential_option_aggregate(v, option_mask, params.tau_R, params.c_R)
    R = weighted_lcvar(pi_am, mode_probs, params.alpha_R)
    R = torch.where(action_mask, R, torch.zeros_like(R))

    witness = torch.argmax(torch.where(option_mask.unsqueeze(-1), v, torch.full_like(v, -torch.inf)), dim=2)
    P_w = _gather_witness(P, witness)
    G_w = _gather_witness(G, witness)
    C_w = _gather_witness(C, witness)
    K_w = _gather_witness(Kdef, witness)
    bottleneck = torch.maximum(P_w, torch.maximum(params.eta_G - G_w, params.eta_C - C_w))
    B = upper_tail_cvar(bottleneck, mode_probs, params.alpha_B)
    U_prof = upper_tail_cvar(U, mode_probs, params.alpha_U)
    H_prof = upper_tail_cvar(H, mode_probs, params.alpha_H)
    min_H = torch.where(action_mask, H_prof, torch.full_like(H_prof, torch.inf)).min(dim=1, keepdim=True).values
    dH = H_prof - min_H
    K_post = upper_tail_cvar(K_w, mode_probs, params.alpha_K)
    prob_w = torch.softmax(torch.where(option_mask.unsqueeze(-1), v / max(params.tau_R, 1e-8), torch.full_like(v, -torch.inf)), dim=2)
    logp = torch.where(prob_w > 0, torch.log(prob_w.clamp_min(1e-8)), torch.zeros_like(prob_w))
    entropy = -(prob_w * logp).sum(dim=2)
    n_valid = option_mask.float().sum(dim=2).clamp_min(1.0)
    W_m = 1.0 - entropy / torch.log(n_valid).clamp_min(1.0).unsqueeze(-1)
    W = (W_m * _normalize_weights(mode_probs).unsqueeze(1)).sum(dim=-1)

    for name, t in [("B", B), ("U", U_prof), ("H", H_prof), ("dH", dH), ("K_post", K_post), ("W", W)]:
        locals()[name] = torch.where(action_mask, t, torch.zeros_like(t))
    return {"R": R, "B": B, "U": U_prof, "H": H_prof, "dH": dH, "K_post": K_post, "witness": witness, "W": W, "pi_am": pi_am, "v": v}
