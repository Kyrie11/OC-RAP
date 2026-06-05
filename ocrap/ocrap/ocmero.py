from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .lcv import EPS, normalize_weights, torch_normalize_weights, torch_weighted_lcvar, torch_weighted_mean, weighted_lcvar, weighted_mean


@dataclass
class OCMEROResult:
    r_dep: float
    r_orc: float
    gap: float
    q: np.ndarray
    r_per_root: np.ndarray
    best_option: np.ndarray


def masked_max_np(values: np.ndarray, valid: np.ndarray, axis: int) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    mask = np.asarray(valid).astype(bool)
    if axis == 1:
        masked = np.where(mask[None, :], v, -1e9)
    elif axis == 0:
        masked = np.where(mask[:, None], v, -1e9)
    else:
        raise ValueError("masked_max_np only supports axis 0 or 1")
    return np.max(masked, axis=axis)


def sparsify_compatibility(C: np.ndarray, top_m: int | None) -> np.ndarray:
    C = np.asarray(C, dtype=np.float64).copy()
    if top_m is None or top_m <= 0 or top_m >= C.shape[-1]:
        return C
    K = C.shape[0]
    out = np.zeros_like(C)
    for i in range(K):
        idx = np.argsort(C[i])[::-1][:top_m]
        out[i, idx] = C[i, idx]
        out[i, i] = max(out[i, i], C[i, i])
    return out


def oc_mero(
    M: np.ndarray,
    p: np.ndarray,
    C: np.ndarray,
    alpha: float = 0.2,
    beta: float = 0.2,
    option_valid: np.ndarray | None = None,
    use_lcvar: bool = True,
    use_obs_kernel: bool = True,
    top_m: int | None = None,
    eps: float = EPS,
) -> OCMEROResult:
    M = np.asarray(M, dtype=np.float64)
    K, L = M.shape
    p_norm = normalize_weights(p, eps)
    if option_valid is None:
        option_valid = np.ones(L, dtype=bool)
    option_valid = np.asarray(option_valid).astype(bool)
    C_eff = np.eye(K, dtype=np.float64) if not use_obs_kernel else sparsify_compatibility(C, top_m)

    agg = weighted_lcvar if use_lcvar else weighted_mean
    oracle_per_root = masked_max_np(M, option_valid, axis=1)
    r_orc = agg(oracle_per_root, p_norm, beta) if use_lcvar else agg(oracle_per_root, p_norm)

    q = np.full((K, L), -1e9, dtype=np.float64)
    for i in range(K):
        w = normalize_weights(C_eff[i] * p_norm, eps)
        for l in range(L):
            if option_valid[l]:
                q[i, l] = agg(M[:, l], w, beta) if use_lcvar else agg(M[:, l], w)
    best_option = np.argmax(q, axis=1)
    r_per_root = q[np.arange(K), best_option]
    r_dep = agg(r_per_root, p_norm, alpha) if use_lcvar else agg(r_per_root, p_norm)
    return OCMEROResult(float(r_dep), float(r_orc), float(r_orc - r_dep), q, r_per_root, best_option)


def torch_oc_mero(
    M: torch.Tensor,
    p: torch.Tensor,
    C: torch.Tensor,
    alpha: float = 0.2,
    beta: float = 0.2,
    option_valid: torch.Tensor | None = None,
    use_lcvar: bool = True,
    use_obs_kernel: bool = True,
    eps: float = EPS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # M: [B,K,L], p: [B,K], C: [B,K,K]
    B, K, L = M.shape
    if option_valid is None:
        option_valid = torch.ones((B, L), dtype=torch.bool, device=M.device)
    if option_valid.dim() == 1:
        option_valid = option_valid.unsqueeze(0).expand(B, -1)
    p_norm = torch_normalize_weights(p, eps)
    C_eff = torch.eye(K, dtype=M.dtype, device=M.device).unsqueeze(0).expand(B, -1, -1) if not use_obs_kernel else C
    mask = option_valid.unsqueeze(1).expand(-1, K, -1)
    M_masked = torch.where(mask, M, torch.full_like(M, -1e9))
    oracle_per_root = M_masked.max(dim=-1).values
    if use_lcvar:
        r_orc = torch_weighted_lcvar(oracle_per_root, p_norm, beta, eps)
    else:
        r_orc = torch_weighted_mean(oracle_per_root, p_norm, eps)
    q_list = []
    for i in range(K):
        w = torch_normalize_weights(C_eff[:, i, :] * p_norm, eps)
        scores = M.transpose(1, 2)  # [B,L,K]
        w_expand = w.unsqueeze(1).expand(-1, L, -1)
        if use_lcvar:
            q_i = torch_weighted_lcvar(scores, w_expand, beta, eps)
        else:
            q_i = torch_weighted_mean(scores, w_expand, eps)
        q_i = torch.where(option_valid, q_i, torch.full_like(q_i, -1e9))
        q_list.append(q_i)
    q = torch.stack(q_list, dim=1)  # [B,K,L]
    r_per_root = q.max(dim=-1).values
    if use_lcvar:
        r_dep = torch_weighted_lcvar(r_per_root, p_norm, alpha, eps)
    else:
        r_dep = torch_weighted_mean(r_per_root, p_norm, eps)
    return r_dep, r_orc, r_orc - r_dep, q
