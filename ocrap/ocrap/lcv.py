from __future__ import annotations

import math

import numpy as np
import torch


EPS = 1e-8


def normalize_weights(weights: np.ndarray, eps: float = EPS) -> np.ndarray:
    w = np.clip(np.asarray(weights, dtype=np.float64), 0.0, None)
    s = float(w.sum())
    if s <= eps:
        if w.size == 0:
            return w
        return np.ones_like(w, dtype=np.float64) / max(w.size, 1)
    return w / (s + eps)


def weighted_lcvar(scores: np.ndarray, weights: np.ndarray, alpha: float, eps: float = EPS) -> float:
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if s.size == 0:
        raise ValueError("weighted_lcvar requires at least one score")
    a = float(alpha)
    if not (0.0 < a <= 1.0):
        raise ValueError(f"alpha must be in (0,1], got {alpha}")
    w = normalize_weights(np.asarray(weights, dtype=np.float64).reshape(-1), eps=eps)
    if w.size != s.size:
        raise ValueError(f"scores and weights length mismatch: {s.size} vs {w.size}")
    idx = np.argsort(s, kind="mergesort")
    remaining = a
    total = 0.0
    for ii in idx:
        take = min(float(w[ii]), remaining)
        if take > 0:
            total += take * float(s[ii])
            remaining -= take
        if remaining <= 1e-12:
            break
    if remaining > 1e-8:
        total += remaining * float(s[idx[-1]])
    return float(total / a)


def weighted_mean(scores: np.ndarray, weights: np.ndarray, eps: float = EPS) -> float:
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    w = normalize_weights(weights, eps=eps)
    return float(np.sum(s * w))


def finite_sample_upper_quantile(scores: np.ndarray, delta: float, numerical_margin: float = 0.0, strict: bool = True) -> float:
    vals = np.sort(np.asarray(scores, dtype=np.float64).reshape(-1))
    n = vals.size
    if n == 0:
        raise ValueError("Cannot calibrate: no negative deployability scores in calibration split.")
    if not (0.0 < delta < 1.0):
        raise ValueError(f"delta must be in (0,1), got {delta}")
    k = int(math.ceil((n + 1) * (1.0 - float(delta))))
    if strict and k > n:
        return float("inf")
    return float(vals[min(k, n) - 1] + numerical_margin)


def torch_normalize_weights(weights: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    w = torch.clamp(weights, min=0.0)
    denom = w.sum(dim=-1, keepdim=True)
    fallback = torch.ones_like(w) / max(w.shape[-1], 1)
    return torch.where(denom > eps, w / (denom + eps), fallback)


def torch_weighted_lcvar(scores: torch.Tensor, weights: torch.Tensor, alpha: float, eps: float = EPS) -> torch.Tensor:
    # scores, weights: [..., N]
    if not (0.0 < float(alpha) <= 1.0):
        raise ValueError(f"alpha must be in (0,1], got {alpha}")
    w = torch_normalize_weights(weights, eps)
    sorted_scores, idx = torch.sort(scores, dim=-1, descending=False, stable=True)
    sorted_weights = torch.gather(w, -1, idx)
    cumsum_prev = torch.cumsum(sorted_weights, dim=-1) - sorted_weights
    remaining_before = torch.clamp(float(alpha) - cumsum_prev, min=0.0)
    take = torch.minimum(sorted_weights, remaining_before)
    return (take * sorted_scores).sum(dim=-1) / float(alpha)


def torch_weighted_mean(scores: torch.Tensor, weights: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    w = torch_normalize_weights(weights, eps)
    return (scores * w).sum(dim=-1)
