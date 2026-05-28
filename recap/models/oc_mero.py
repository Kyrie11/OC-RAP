from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from .mero import PositiveLinear, weighted_lcvar, upper_tail_cvar, _normalize_weights


class MonotoneRecoveryCalibrator(nn.Module):
    """Monotone calibrator over [g_hat, -harm, -K_post, -uncertainty]."""
    def __init__(self, g_dim: int = 9, hidden: int = 32):
        super().__init__()
        self.g_dim = g_dim
        self.net = nn.Sequential(
            PositiveLinear(g_dim + 3, hidden), nn.Softplus(),
            PositiveLinear(hidden, hidden), nn.Softplus(),
            PositiveLinear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int):
    logits = logits.masked_fill(~mask, -1e9)
    probs = torch.softmax(logits, dim=dim)
    return probs.masked_fill(~mask, 0.0)


def existential_mu_aggregate(v: torch.Tensor, mu_logits: torch.Tensor, option_mask: torch.Tensor, tau_R: float = 0.2, c_R: float = 0.0):
    if v.dim() != 4:
        raise ValueError("v must have shape [B,K,L,M]")
    mask = option_mask.bool().unsqueeze(-1).expand_as(v)
    mu = masked_softmax(mu_logits, mask, dim=2)
    log_mu = torch.log(mu.clamp_min(1e-8))
    combined = log_mu + v / max(tau_R, 1e-8)
    lse = tau_R * torch.logsumexp(combined.masked_fill(~mask, -1e9), dim=2)
    pi = torch.sigmoid(lse - c_R)
    any_valid = option_mask.any(dim=2).unsqueeze(-1)
    pi = torch.where(any_valid, pi, torch.zeros_like(pi))
    return pi, mu


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p=p.clamp_min(eps); q=q.clamp_min(eps)
    p=p/p.sum(dim=-1, keepdim=True).clamp_min(eps)
    q=q/q.sum(dim=-1, keepdim=True).clamp_min(eps)
    m=0.5*(p+q)
    return 0.5*((p*(p/m).log()).sum(dim=-1) + (q*(q/m).log()).sum(dim=-1))


@dataclass
class OCMEROParams:
    tau_R: float = 0.20
    c_R: float = 0.0
    alpha_R: float = 0.20
    alpha_B: float = 0.20
    alpha_U: float = 0.20
    alpha_H: float = 0.20
    alpha_K: float = 0.20
    mean_over_options: bool = False


def _gather_witness(x: torch.Tensor, witness: torch.Tensor) -> torch.Tensor:
    return torch.gather(x, 2, witness.unsqueeze(2)).squeeze(2)


def _default_v(g_hat, h_hat, k_hat, u_hat):
    # Signed margin vector is positive-good; harm/deficit/uncertainty are negative-good inputs.
    h = h_hat.unsqueeze(2).expand_as(k_hat)
    u = u_hat.unsqueeze(2).expand_as(k_hat) if u_hat.dim() == 3 else u_hat
    return g_hat.mean(dim=-1) - h - k_hat - u


def compute_ocmero_profiles(pred: Dict[str, torch.Tensor], masks: Dict[str, torch.Tensor], params: OCMEROParams | dict | None = None, calibrator: Optional[nn.Module] = None) -> Dict[str, torch.Tensor]:
    if params is None: params = OCMEROParams()
    if isinstance(params, dict): params = OCMEROParams(**{k:v for k,v in params.items() if k in OCMEROParams.__annotations__})
    g_hat = pred["g_hat"]
    h_hat = pred.get("h_hat", pred.get("H"))
    k_hat = pred.get("k_hat", pred.get("Kdef", pred.get("K_post")))
    u_hat = pred.get("u_hat", pred.get("U", torch.zeros_like(h_hat)))
    mu_logits = pred.get("mu_logits", pred.get("option_logits", torch.zeros_like(k_hat)))
    mode_probs = pred.get("mode_probs")
    if mode_probs is None:
        mode_probs = torch.ones(g_hat.shape[0], g_hat.shape[-2], device=g_hat.device, dtype=g_hat.dtype) / g_hat.shape[-2]
    action_mask = masks["action_mask"].bool()
    option_mask = masks["option_mask"].bool() & action_mask.unsqueeze(-1)
    if calibrator is None and "v_hat" in pred:
        v = pred["v_hat"]
    elif calibrator is None:
        v = _default_v(g_hat, h_hat, k_hat, u_hat)
    else:
        h = h_hat.unsqueeze(2).unsqueeze(-1).expand(*g_hat.shape[:-1], 1)
        u = (u_hat.unsqueeze(2) if u_hat.dim()==3 else u_hat).unsqueeze(-1)
        if u.shape[:-1] != g_hat.shape[:-1]: u = u.expand(*g_hat.shape[:-1], 1)
        x = torch.cat([g_hat, -h, -k_hat.unsqueeze(-1), -u], dim=-1)
        v = calibrator(x)
    mask = option_mask.unsqueeze(-1).expand_as(v)
    v = torch.where(mask, v, torch.full_like(v, -torch.inf))
    if params.mean_over_options:
        pi_am = torch.sigmoid(torch.where(mask, v, torch.zeros_like(v))).sum(dim=2) / option_mask.float().sum(dim=2).clamp_min(1.0).unsqueeze(-1)
        mu = masked_softmax(torch.zeros_like(mu_logits), mask, dim=2)
    else:
        pi_am, mu = existential_mu_aggregate(v, mu_logits, option_mask, params.tau_R, params.c_R)
    R = weighted_lcvar(pi_am, mode_probs, params.alpha_R)
    witness_score = torch.log(mu.clamp_min(1e-8)) + v / max(params.tau_R, 1e-8)
    witness = torch.argmax(witness_score.masked_fill(~mask, -1e9), dim=2)
    K_w = _gather_witness(k_hat, witness)
    U_mode = u_hat if u_hat.dim()==3 else _gather_witness(u_hat, witness)
    B = upper_tail_cvar(1.0 - pi_am, mode_probs, params.alpha_B)
    U_prof = upper_tail_cvar(U_mode, mode_probs, params.alpha_U)
    H = upper_tail_cvar(h_hat, mode_probs, params.alpha_H)
    min_H = torch.where(action_mask, H, torch.full_like(H, torch.inf)).min(dim=1, keepdim=True).values
    dH = H - min_H
    K_post = upper_tail_cvar(K_w, mode_probs, params.alpha_K)
    C = pred.get("c_rule_hat", pred.get("C", torch.zeros_like(H)))
    C = upper_tail_cvar(C, mode_probs, params.alpha_H) if C.dim()==3 else C
    W = (mu.max(dim=2).values * _normalize_weights(mode_probs).unsqueeze(1)).sum(dim=-1)
    for tname in ["R","B","U_prof","H","dH","K_post","C","W"]:
        pass
    return {"R": torch.where(action_mask,R,torch.zeros_like(R)), "B": torch.where(action_mask,B,torch.zeros_like(B)), "U": torch.where(action_mask,U_prof,torch.zeros_like(U_prof)), "H": torch.where(action_mask,H,torch.zeros_like(H)), "dH": torch.where(action_mask,dH,torch.zeros_like(dH)), "K_post": torch.where(action_mask,K_post,torch.zeros_like(K_post)), "C": torch.where(action_mask,C,torch.zeros_like(C)), "witness": witness, "W": torch.where(action_mask,W,torch.zeros_like(W)), "pi_am": pi_am, "v": v, "mu": mu, "beta": torch.softmax(pred.get("beta_logits", torch.zeros(g_hat.shape[0], g_hat.shape[1], g_hat.shape[3], g_hat.shape[3], device=g_hat.device, dtype=g_hat.dtype)), dim=-1)}
