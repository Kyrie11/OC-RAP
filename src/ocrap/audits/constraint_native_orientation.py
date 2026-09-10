from __future__ import annotations

"""Constraint-native candidate recovery orientation audit primitives.

This is the cleaned implementation used by the V48.111 CNRO audit.  It is
self-contained with respect to historical experiment modules: no previous
versioned source file is imported.  Frozen checkpoint/index artifacts may still
be supplied as scientific inputs, but the code path depends only on current
OC-RAP core modules.
"""

import copy
from dataclasses import dataclass
from math import exp
from typing import Any, Mapping

import numpy as np
import torch

from ocrap.models.encoders import FlatFeatureLayout

ENGINEERING_VERSION = "v48.111.1-OC-CNRO-RESULTFIX"
ALGORITHM_NAME = "Observation-Consistent Constraint-Native Recovery Orientation Audit"

RAW_CANDIDATE_DIM = 156
AGENT_DIM = 10
PREFIX_STATE_START = 36
PREFIX_STATE_WIDTH = 9
PREFIX_COMPLETE_STEPS = 8
PAIR_SIZE = 2
GEOMETRY_CHANNELS = 2
GEOMETRY_DIM = PAIR_SIZE * PREFIX_COMPLETE_STEPS * GEOMETRY_CHANNELS
MATCHED_DIM = RAW_CANDIDATE_DIM + GEOMETRY_DIM

DEPLOYABLE_MACROS = frozenset({2, 3, 5, 6, 7})
VALID_MODES = frozenset({"drs_activation", "deployability_gain"})
POSITIVE_GAIN = 0.015
FEATURE_ONLY_TRUTH_CONTRACT = "legacy_full"
FEATURE_ONLY_SUPERVISION_OBJECTIVE = "binary_sign"
FACTOR_NAMES = ("drs", "deployability_gate", "gap_discount")


def _layout_slices(layout: FlatFeatureLayout) -> dict[str, slice]:
    dims = [
        ("ego", layout.ego_dim),
        ("prefix_param", layout.prefix_param_dim),
        ("macro", layout.num_macros),
        ("scalar", layout.scalar_dim),
        ("prefix_state", layout.prefix_flat_dim),
        ("control", layout.control_flat_dim),
        ("agent_summary", layout.agent_summary_dim),
        ("agents", layout.feature_max_agents * layout.agent_token_dim),
        ("bev", layout.bev_dim),
        ("route", layout.route_stats_dim + layout.route_flat_dim),
        ("map", layout.map_stats_dim + layout.map_flat_dim),
        ("dyn", layout.dyn_stats_dim + layout.dyn_flat_dim),
    ]
    out: dict[str, slice] = {}
    i = 0
    for name, d in dims:
        out[name] = slice(i, i + int(d))
        i += int(d)
    if i != int(layout.total_dim):
        raise ValueError(f"layout slice mismatch {i} != {layout.total_dim}")
    out["macro_scalar"] = slice(out["macro"].start, out["scalar"].stop)
    return out


def raw_candidate_pathway(x: torch.Tensor, layout: FlatFeatureLayout) -> torch.Tensor:
    if x.ndim != 2 or x.shape[1] != int(layout.total_dim):
        raise ValueError("raw_candidate_pathway expects [B,total_dim]")
    s = _layout_slices(layout)
    groups = ("ego", "prefix_param", "macro_scalar", "prefix_state", "control")
    out = torch.cat([x[:, s[g]] for g in groups], dim=-1).float()
    if out.shape[1] != RAW_CANDIDATE_DIM:
        raise ValueError(f"raw candidate dimension mismatch {out.shape[1]} != {RAW_CANDIDATE_DIM}")
    return out


def action_features(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if z.ndim != 2 or z.shape[0] < 2:
        raise ValueError("action_features requires nominal + candidate rows")
    nominal = z[0:1]
    candidate = z[1:]
    delta = candidate - nominal
    state = nominal.expand(candidate.shape[0], -1)
    context = delta * (1.0 + torch.tanh(state))
    return state, delta, context


def raw_agent_set(x: torch.Tensor, layout: FlatFeatureLayout) -> tuple[torch.Tensor, torch.Tensor]:
    if x.ndim != 2 or x.shape[1] != int(layout.total_dim):
        raise ValueError("raw_agent_set expects [B,total_dim]")
    s = _layout_slices(layout)
    a = x[:, s["agents"]].reshape(x.shape[0], layout.feature_max_agents, layout.agent_token_dim).float()
    if a.shape[-1] != AGENT_DIM:
        raise ValueError(f"agent token dimension mismatch {a.shape[-1]} != {AGENT_DIM}")
    mask = torch.linalg.vector_norm(a, dim=-1) > 1.0e-8
    return a, mask


def decode_prefix_xy_and_ego(raw_candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = np.asarray(raw_candidate, dtype=np.float64)
    if r.ndim != 2 or r.shape[1] != RAW_CANDIDATE_DIM:
        raise ValueError("raw candidate shape mismatch")
    flat = r[:, PREFIX_STATE_START:PREFIX_STATE_START + PREFIX_COMPLETE_STEPS * PREFIX_STATE_WIDTH]
    st = flat.reshape(len(r), PREFIX_COMPLETE_STEPS, PREFIX_STATE_WIDTH)
    ego_xy = r[:, 0:2]
    ego_len = np.maximum(np.abs(r[:, 7]), 1.0e-3)
    ego_wid = np.maximum(np.abs(r[:, 8]), 1.0e-3)
    ego_rad = 0.5 * np.sqrt(ego_len * ego_len + ego_wid * ego_wid)
    return st[:, :, 0:2], ego_xy, ego_rad


def candidate_agent_signed_clearance_paths(
    raw_candidate: np.ndarray,
    agents: np.ndarray,
    agent_mask: np.ndarray,
    sample_rate_hz: float,
) -> np.ndarray:
    r = np.asarray(raw_candidate, dtype=np.float64)
    a = np.asarray(agents, dtype=np.float64)
    m = np.asarray(agent_mask, dtype=bool)
    if r.ndim != 2 or r.shape[1] != RAW_CANDIDATE_DIM:
        raise ValueError("raw candidate shape mismatch")
    if a.ndim != 3 or a.shape[0] != len(r) or a.shape[2] != AGENT_DIM or m.shape != a.shape[:2]:
        raise ValueError("agent shape mismatch")
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("invalid sample rate")

    pxy, ego_xy, ego_rad = decode_prefix_xy_and_ego(r)
    prefix_rel = pxy - ego_xy[:, None, :]
    rel0 = a[:, :, 0:2] * 80.0
    vel = a[:, :, 2:4] * 20.0
    alen = np.maximum(np.abs(a[:, :, 7] * 10.0), 1.0e-3)
    awid = np.maximum(np.abs(a[:, :, 8] * 5.0), 1.0e-3)
    arad = 0.5 * np.sqrt(alen * alen + awid * awid)
    times = (np.arange(PREFIX_COMPLETE_STEPS, dtype=np.float64) + 1.0) / float(sample_rate_hz)
    future = rel0[:, :, None, :] + vel[:, :, None, :] * times[None, None, :, None]
    delta = prefix_rel[:, None, :, :] - future
    clear = np.linalg.norm(delta, axis=-1) - ego_rad[:, None, None] - arad[:, :, None]
    return np.where(m[:, :, None], clear, np.inf)


def pair_indices_from_clearance(
    clearance: np.ndarray, agents: np.ndarray, agent_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    h = np.asarray(clearance, dtype=np.float64)
    a = np.asarray(agents, dtype=np.float64)
    m = np.asarray(agent_mask, dtype=bool)
    if h.ndim != 3 or h.shape[:2] != m.shape or h.shape[2] != PREFIX_COMPLETE_STEPS:
        raise ValueError("clearance shape mismatch")
    rel0 = a[:, :, 0:2] * 80.0
    r2 = np.sum(rel0 * rel0, axis=-1)
    r2 = np.where(m, r2, np.inf)
    cmin = np.min(h, axis=2)
    n = len(h)
    active = np.zeros((n, PAIR_SIZE), dtype=np.int64)
    nearest = np.zeros((n, PAIR_SIZE), dtype=np.int64)
    for k in range(n):
        ids = np.flatnonzero(m[k])
        if ids.size == 0:
            continue
        oa = ids[np.argsort(cmin[k, ids], kind="stable")]
        on = ids[np.argsort(r2[k, ids], kind="stable")]
        active[k, 0] = int(oa[0])
        active[k, 1] = int(oa[1] if len(oa) > 1 else oa[0])
        nearest[k, 0] = int(on[0])
        nearest[k, 1] = int(on[1] if len(on) > 1 else on[0])
    return active, nearest


@dataclass
class ConstraintNativeScaler:
    u_scale: np.ndarray
    clearance_scale: float


def fit_constraint_native_scaler(
    u: np.ndarray, hc: np.ndarray, h0: np.ndarray, agent_mask: np.ndarray
) -> ConstraintNativeScaler:
    u = np.asarray(u, dtype=np.float64)
    hc = np.asarray(hc, dtype=np.float64)
    h0 = np.asarray(h0, dtype=np.float64)
    m = np.asarray(agent_mask, dtype=bool)
    if u.ndim != 2 or u.shape[1] != RAW_CANDIDATE_DIM:
        raise ValueError("candidate response dimension mismatch")
    u_scale = np.sqrt(np.mean(u * u, axis=0, keepdims=True))
    u_scale = np.where(u_scale > 1.0e-8, u_scale, 1.0)
    finite = np.broadcast_to(m[:, :, None], hc.shape) & np.isfinite(hc) & np.isfinite(h0)
    vals = np.concatenate([hc[finite], h0[finite]]) if np.any(finite) else np.array([1.0])
    hs = float(np.sqrt(np.mean(vals * vals)))
    if not np.isfinite(hs) or hs <= 1.0e-8:
        hs = 1.0
    return ConstraintNativeScaler(u_scale=u_scale, clearance_scale=hs)


def base_features(u: np.ndarray, scaler: ConstraintNativeScaler) -> np.ndarray:
    return np.asarray(u, dtype=np.float64) / scaler.u_scale


def _pair_geometry(
    hc: np.ndarray, h0: np.ndarray, pair: np.ndarray, scaler: ConstraintNativeScaler
) -> np.ndarray:
    n = len(hc)
    ix = np.arange(n)[:, None]
    cand = hc[ix, pair]
    nominal = h0[ix, pair]
    valid = np.isfinite(cand) & np.isfinite(nominal)
    dh = np.where(valid, cand - nominal, 0.0) / scaler.clearance_scale
    h0z = np.where(valid, nominal, 0.0) / scaler.clearance_scale
    geom = np.stack([dh, dh * h0z], axis=-1).reshape(n, -1)
    if geom.shape[1] != GEOMETRY_DIM:
        raise ValueError(f"geometry dim mismatch {geom.shape[1]} != {GEOMETRY_DIM}")
    return geom


def matched_features(
    u: np.ndarray,
    hc: np.ndarray,
    h0: np.ndarray,
    pair: np.ndarray,
    scaler: ConstraintNativeScaler,
) -> np.ndarray:
    out = np.concatenate([base_features(u, scaler), _pair_geometry(hc, h0, pair, scaler)], axis=1)
    if out.shape[1] != MATCHED_DIM:
        raise ValueError(f"matched dim mismatch {out.shape[1]} != {MATCHED_DIM}")
    return out


@dataclass
class ConvexRidgeOrientation:
    coef: np.ndarray
    ridge_lambda: float
    objective: float
    normal_equation_residual: float


def _balanced_weights(y01: np.ndarray) -> np.ndarray:
    y = np.asarray(y01, dtype=np.int64)
    n = len(y)
    pos = int(y.sum())
    neg = int(n - pos)
    if n < 4 or pos == 0 or neg == 0:
        raise ValueError("convex orientation probe requires both classes")
    w = np.empty(n, dtype=np.float64)
    w[y == 1] = n / (2.0 * pos)
    w[y == 0] = n / (2.0 * neg)
    return w


def fit_closed_form_ridge(x: np.ndarray, y01: np.ndarray) -> ConvexRidgeOrientation:
    x = np.asarray(x, dtype=np.float64)
    y01 = np.asarray(y01, dtype=np.int64)
    if x.ndim != 2 or len(x) != len(y01):
        raise ValueError("ridge input shape mismatch")
    n = len(y01)
    weights = _balanced_weights(y01)
    y = 2.0 * y01.astype(np.float64) - 1.0
    sw = np.sqrt(weights / weights.sum())
    a = x * sw[:, None]
    b = y * sw
    lam = 1.0 / float(n)
    kernel = a @ a.T
    kernel.flat[:: n + 1] += lam
    alpha = np.linalg.solve(kernel, b)
    coef = a.T @ alpha
    pred = x @ coef
    obj = float(np.sum(weights * (pred - y) ** 2) / weights.sum() + lam * np.dot(coef, coef))
    grad = 2.0 * (x.T @ (weights * (pred - y)) / weights.sum() + lam * coef)
    resid = float(np.linalg.norm(grad) / max(1.0, np.linalg.norm(coef)))
    return ConvexRidgeOrientation(coef=coef, ridge_lambda=lam, objective=obj, normal_equation_residual=resid)


def ridge_scores(model: ConvexRidgeOrientation, x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64) @ np.asarray(model.coef, dtype=np.float64)


def sigmoid(x: float) -> float:
    z = float(x)
    if z >= 0:
        e = float(np.exp(-z))
        return 1.0 / (1.0 + e)
    e = float(np.exp(z))
    return e / (1.0 + e)


def _clip01(x: float) -> float:
    return float(np.clip(float(x), 0.0, 1.0))


def _pcd_product(factors: Mapping[str, float]) -> float:
    out = 1.0
    for name in FACTOR_NAMES:
        out *= _clip01(float(factors[name]))
    return float(np.clip(out, 0.0, 1.0))


def _mediation_mode(
    nominal: Mapping[str, float],
    candidate: Mapping[str, float],
    *,
    positive_gain: float = POSITIVE_GAIN,
) -> tuple[str, float]:
    p_nom = _pcd_product(nominal)
    p_cand = _pcd_product(candidate)
    full = float(p_cand - p_nom)
    necessary: list[str] = []
    for name in FACTOR_NAMES:
        ko = dict(candidate)
        ko[name] = nominal[name]
        if full >= float(positive_gain) and (_pcd_product(ko) - p_nom) < float(positive_gain):
            necessary.append(name)
    if len(necessary) == 1:
        mode = {
            "drs": "drs_activation",
            "deployability_gate": "deployability_gain",
            "gap_discount": "gap_gain",
        }[necessary[0]]
    elif len(necessary) > 1:
        mode = "multi_factor_necessary"
    else:
        mode = "redundant_or_interaction"
    return mode, full


def teacher_factor_tuple(row: Mapping[str, Any]) -> dict[str, float]:
    return {
        "drs": float(np.clip(float(row["teacher_drs"]), 0.0, 1.0)),
        "deployability_gate": sigmoid(float(row["teacher_r_dep"])),
        "gap_discount": float(np.clip(exp(-max(0.0, float(row["teacher_gap"]))), 0.0, 1.0)),
    }


def derive_candidate_semantics(
    nominal: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    positive_gain: float = POSITIVE_GAIN,
) -> dict[str, Any]:
    n = teacher_factor_tuple(nominal)
    c = teacher_factor_tuple(candidate)
    mode, full_advantage = _mediation_mode(n, c, positive_gain=positive_gain)
    teacher_adv = float(candidate["teacher_pcd"]) - float(nominal["teacher_pcd"])
    if abs(full_advantage - teacher_adv) > 2.0e-6:
        raise ValueError(f"PCD advantage mismatch: {full_advantage} vs {teacher_adv}")
    harmful = bool(candidate.get("component_harmful", False))
    safe_positive = bool(teacher_adv >= float(positive_gain) and not harmful)
    return {
        "teacher_adv": teacher_adv,
        "teacher_harmful": harmful,
        "safe_positive": safe_positive,
        "mediation_mode": mode if safe_positive else None,
        "nominal_drs": n["drs"],
        "candidate_drs": c["drs"],
        "nominal_deployability_gate": n["deployability_gate"],
        "candidate_deployability_gate": c["deployability_gate"],
    }


def feature_only_dataset_cfg(
    cfg: Mapping[str, Any], *, cache_dir: str, workers: int = 8
) -> tuple[dict[str, Any], dict[str, Any]]:
    out = copy.deepcopy(dict(cfg))
    training = out.setdefault("training", {})
    if not isinstance(training, dict):
        raise ValueError("checkpoint training config must be a mapping")
    before = {
        "truth_contract": str(training.get("direct_value_absolute_feasibility_truth_contract", "legacy_full")),
        "truth_index": str(training.get("direct_value_absolute_feasibility_truth_index", "") or ""),
        "supervision_objective": str(training.get("direct_value_absolute_feasibility_supervision_objective", "binary_sign")),
        "action_response_truth_index": str(training.get("direct_value_action_response_truth_index", "") or ""),
    }
    training["direct_value_absolute_feasibility_truth_contract"] = FEATURE_ONLY_TRUTH_CONTRACT
    training["direct_value_absolute_feasibility_truth_index"] = ""
    training["direct_value_absolute_feasibility_supervision_objective"] = FEATURE_ONLY_SUPERVISION_OBJECTIVE
    training["direct_value_action_response_truth_index"] = ""
    training["persistent_tensor_cache"] = True
    training["persistent_tensor_cache_dir"] = str(cache_dir)
    training["persistent_tensor_cache_build_workers"] = max(1, int(workers))
    return out, {
        "feature_only_dataset": True,
        "checkpoint_supervision": before,
        "effective_truth_contract": FEATURE_ONLY_TRUTH_CONTRACT,
        "effective_supervision_objective": FEATURE_ONLY_SUPERVISION_OBJECTIVE,
        "truth_sidecars_attached": False,
    }


def contract_checks() -> dict[str, bool]:
    rng = np.random.default_rng(48111)
    n, j = 6, 5
    agents = rng.normal(size=(n, j, AGENT_DIM))
    mask = np.ones((n, j), dtype=bool)
    agents[:, :, 0:4] *= 0.05
    agents[:, :, 7] = 0.48
    agents[:, :, 8] = 0.4
    r0 = np.zeros((n, RAW_CANDIDATE_DIM))
    r0[:, 7] = 4.8
    r0[:, 8] = 2.0
    rc = r0.copy()
    for k in range(n):
        st0 = np.zeros((PREFIX_COMPLETE_STEPS, 9))
        stc = np.zeros_like(st0)
        st0[:, 0] = np.linspace(0.2, 2.0, PREFIX_COMPLETE_STEPS)
        stc[:, 0] = st0[:, 0] + 0.1 * (k + 1)
        stc[:, 1] = 0.05 * ((-1) ** k) * np.arange(1, PREFIX_COMPLETE_STEPS + 1)
        st0[:, 7] = stc[:, 7] = 4.8
        st0[:, 8] = stc[:, 8] = 2.0
        r0[k, 36:108] = st0.reshape(-1)
        rc[k, 36:108] = stc.reshape(-1)
    hc = candidate_agent_signed_clearance_paths(rc, agents, mask, 10.0)
    h0 = candidate_agent_signed_clearance_paths(r0, agents, mask, 10.0)
    active, nearest = pair_indices_from_clearance(hc, agents, mask)
    u = rng.normal(size=(n, RAW_CANDIDATE_DIM))
    scaler = fit_constraint_native_scaler(u, hc, h0, mask)
    fa = matched_features(u, hc, h0, active, scaler)
    fn = matched_features(u, hc, h0, nearest, scaler)
    z = matched_features(np.zeros_like(u), h0, h0, active, scaler)
    perm = np.array([2, 4, 1, 0, 3])
    a2, n2 = pair_indices_from_clearance(hc[:, perm], agents[:, perm], mask[:, perm])
    perm_ok = bool(np.array_equal(active, perm[a2]) and np.array_equal(nearest, perm[n2]))
    return {
        "geometry_dim_32": GEOMETRY_DIM == 32,
        "matched_dim_188": MATCHED_DIM == 188,
        "matched_family_same_dim": fa.shape[1] == fn.shape[1] == MATCHED_DIM,
        "nominal_zero_exact": bool(np.count_nonzero(z) == 0),
        "candidate_agent_permutation_invariant": perm_ok,
        "clearance_scale_finite_positive": bool(np.isfinite(scaler.clearance_scale) and scaler.clearance_scale > 0),
    }
