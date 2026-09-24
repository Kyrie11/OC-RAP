from __future__ import annotations

"""Observation-consistent heterogeneous active-constraint normal-cone audit.

The module is the current V48.112 scientific intervention.  It deliberately
keeps the historical V48.111 candidate response coordinate and convex probe,
but replaces the *single constraint family* (fixed-CV circle clearance to two
agents) by a heterogeneous signed constraint field over:

  clearance, stopping, route, persistent re-entry.

For each of the first eight complete candidate-prefix states, the module
computes candidate-minus-nominal signed constraint response.  Two equal-width
families differ only in which active constraint type selects the response:

  nominal_cone   -- active type selected from the nominal prefix;
  candidate_cone -- active type selected from the candidate prefix.

Each selected constraint contributes two channels (signed response and the
same response modulated by the nominal signed boundary state), so both families
are exactly 4 constraints x 8 times x 2 channels = 64 dimensions.  Appended to
the unchanged 156-D candidate response coordinate, both probes are 220-D.

No regime id, teacher future, hidden root, trainable planner/source parameter,
or learned selector is used here.
"""

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from ocrap.audits.constraint_native_orientation import (
    PREFIX_COMPLETE_STEPS,
    PREFIX_STATE_START,
    PREFIX_STATE_WIDTH,
    RAW_CANDIDATE_DIM,
)

ENGINEERING_VERSION = "v48.112.0-OC-HCNC"
SCIENTIFIC_VERSION = "v48.112-OC-HCNC"
ALGORITHM_NAME = "Observation-Consistent Heterogeneous Constraint Normal-Cone Audit"

CONSTRAINT_NAMES = ("clearance", "stopping", "route", "reentry")
NUM_CONSTRAINTS = len(CONSTRAINT_NAMES)
CONE_CHANNELS = 2
CONE_GEOMETRY_DIM = PREFIX_COMPLETE_STEPS * NUM_CONSTRAINTS * CONE_CHANNELS
MATCHED_DIM = RAW_CANDIDATE_DIM + CONE_GEOMETRY_DIM


@dataclass(frozen=True)
class ConstraintConeConfig:
    d_safe0_m: float
    safe_time_headway_s: float
    distance_scale: float
    stop_scale: float
    route_scale: float
    route_dev_max_m: float
    a_min_mps2: float
    default_available_distance_m: float


def cone_config_from_mapping(cfg: Mapping[str, Any]) -> ConstraintConeConfig:
    scales = cfg.get("margin_scales", {}) if isinstance(cfg.get("margin_scales", {}), Mapping) else {}
    limits = cfg.get("control_limits", {}) if isinstance(cfg.get("control_limits", {}), Mapping) else {}

    def pos(v: Any, default: float) -> float:
        x = abs(float(v if v is not None else default))
        return x if np.isfinite(x) and x > 1.0e-8 else float(default)

    return ConstraintConeConfig(
        d_safe0_m=float(cfg.get("d_safe0_m", 1.0)),
        safe_time_headway_s=float(cfg.get("safe_time_headway_s", 0.5)),
        distance_scale=pos(scales.get("distance", 2.0), 2.0),
        stop_scale=pos(scales.get("stop", 5.0), 5.0),
        route_scale=pos(scales.get("route", 1.0), 1.0),
        route_dev_max_m=float(cfg.get("route_dev_max_m", 2.5)),
        a_min_mps2=-pos(limits.get("a_min", -6.0), 6.0),
        default_available_distance_m=pos(cfg.get("default_available_distance_m", 60.0), 60.0),
    )


def decode_prefix_states(raw_candidate: np.ndarray) -> np.ndarray:
    r = np.asarray(raw_candidate, dtype=np.float64)
    if r.ndim != 2 or r.shape[1] != RAW_CANDIDATE_DIM:
        raise ValueError(f"raw candidate shape mismatch {r.shape}")
    flat = r[:, PREFIX_STATE_START:PREFIX_STATE_START + PREFIX_COMPLETE_STEPS * PREFIX_STATE_WIDTH]
    return flat.reshape(len(r), PREFIX_COMPLETE_STEPS, PREFIX_STATE_WIDTH)


def _ego_and_agent_radii(raw_candidate: np.ndarray, agents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = np.asarray(raw_candidate, dtype=np.float64)
    a = np.asarray(agents, dtype=np.float64)
    ego_len = np.maximum(np.abs(r[:, 7]), 1.0e-3)
    ego_wid = np.maximum(np.abs(r[:, 8]), 1.0e-3)
    ego_rad = 0.5 * np.hypot(ego_len, ego_wid)
    agent_len = np.maximum(np.abs(a[:, :, 7] * 10.0), 1.0e-3)
    agent_wid = np.maximum(np.abs(a[:, :, 8] * 5.0), 1.0e-3)
    agent_rad = 0.5 * np.hypot(agent_len, agent_wid)
    return ego_rad, agent_rad


def physical_clearance_paths(
    raw_candidate: np.ndarray,
    agents: np.ndarray,
    agent_mask: np.ndarray,
    sample_rate_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Circle signed-clearance for prefix times and current observation.

    This preserves V48.111's observation-only CV continuation solely as the
    clearance member of a *heterogeneous* constraint set.  The audit no longer
    treats agent identity / clearance ownership as the complete geometry.
    """
    r = np.asarray(raw_candidate, dtype=np.float64)
    a = np.asarray(agents, dtype=np.float64)
    m = np.asarray(agent_mask, dtype=bool)
    if a.ndim != 3 or a.shape[0] != len(r) or a.shape[2] != 10 or m.shape != a.shape[:2]:
        raise ValueError("agent shape mismatch")
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("invalid sample_rate_hz")

    st = decode_prefix_states(r)
    ego_xy = r[:, 0:2]
    prefix_rel = st[:, :, 0:2] - ego_xy[:, None, :]
    rel0 = a[:, :, 0:2] * 80.0
    vel = a[:, :, 2:4] * 20.0
    ego_rad, agent_rad = _ego_and_agent_radii(r, a)
    times = (np.arange(PREFIX_COMPLETE_STEPS, dtype=np.float64) + 1.0) / float(sample_rate_hz)
    future = rel0[:, :, None, :] + vel[:, :, None, :] * times[None, None, :, None]
    delta = prefix_rel[:, None, :, :] - future
    clear = np.linalg.norm(delta, axis=-1) - ego_rad[:, None, None] - agent_rad[:, :, None]
    clear = np.where(m[:, :, None], clear, np.inf)

    current = np.linalg.norm(rel0, axis=-1) - ego_rad[:, None] - agent_rad
    current = np.where(m, current, np.inf)
    return clear, current


def _remaining_path_distance_to_conflict(
    xy: np.ndarray,
    ego_xy: np.ndarray,
    safety_reserve: np.ndarray,
    default_available: float,
) -> np.ndarray:
    n, t, _ = xy.shape
    out = np.full((n, t), float(default_available), dtype=np.float64)
    for i in range(n):
        pts = np.concatenate([ego_xy[i : i + 1], xy[i]], axis=0)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=-1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        for q in range(t):
            hit = np.flatnonzero(safety_reserve[i, q:] <= 0.0)
            if hit.size:
                u = q + int(hit[0])
                out[i, q] = max(0.0, float(cum[u + 1] - cum[q + 1]))
    return out


def heterogeneous_constraint_paths(
    raw_candidate: np.ndarray,
    agents: np.ndarray,
    agent_mask: np.ndarray,
    *,
    sample_rate_hz: float,
    config: ConstraintConeConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return normalized signed constraints [N,T,4] and active masks.

    Positive values denote reserve and negative values denote debt/violation.
    The normalization is physical/config based and *not centered*, so zero stays
    the actual constraint boundary.
    """
    r = np.asarray(raw_candidate, dtype=np.float64)
    a = np.asarray(agents, dtype=np.float64)
    m = np.asarray(agent_mask, dtype=bool)
    st = decode_prefix_states(r)
    n, t, _ = st.shape
    if t != PREFIX_COMPLETE_STEPS:
        raise ValueError("prefix step mismatch")

    clear_agent, current_agent_clear = physical_clearance_paths(r, a, m, sample_rate_hz)
    has_agents = np.any(m, axis=1)
    raw_clear = np.min(clear_agent, axis=1)
    raw_clear = np.where(has_agents[:, None], raw_clear, np.nan)
    current_raw_clear = np.min(current_agent_clear, axis=1)
    current_raw_clear = np.where(has_agents, current_raw_clear, np.nan)

    speed = np.maximum(np.abs(st[:, :, 6]), 0.0)
    d_safe = float(config.d_safe0_m) + float(config.safe_time_headway_s) * speed
    safety_reserve_m = raw_clear - d_safe

    values = np.zeros((n, t, NUM_CONSTRAINTS), dtype=np.float64)
    masks = np.zeros((n, t, NUM_CONSTRAINTS), dtype=bool)

    # 0) clearance: observation-only CV physical separation reserve.
    clr = np.where(has_agents[:, None], safety_reserve_m / config.distance_scale, 0.0)
    values[:, :, 0] = clr
    masks[:, :, 0] = has_agents[:, None]

    # 1) stopping: remaining path distance before the first predicted safety
    # boundary crossing minus physically available braking distance.
    xy = st[:, :, 0:2]
    ego_xy = r[:, 0:2]
    safety_for_conflict = np.where(has_agents[:, None], safety_reserve_m, np.inf)
    available = _remaining_path_distance_to_conflict(
        xy, ego_xy, safety_for_conflict, config.default_available_distance_m
    )
    decel = max(abs(float(config.a_min_mps2)), 1.0e-6)
    stop_required = speed * speed / (2.0 * decel)
    values[:, :, 1] = (available - stop_required) / config.stop_scale
    masks[:, :, 1] = True

    # 2) route: signed route corridor reserve in the same local route coordinate
    # already used by the stable recovery semantics.
    values[:, :, 2] = (float(config.route_dev_max_m) - np.abs(st[:, :, 1])) / config.route_scale
    masks[:, :, 2] = True

    # 3) persistent re-entry: inactive until contact has been observed or caused
    # by the candidate prefix.  Once active, negative reserve is debt; after
    # crossing to nonnegative reserve, the suffix minimum enforces persistence.
    # The underlying value is still computed even when inactive so a
    # candidate-induced activation has a well-defined candidate-minus-nominal
    # response instead of an arbitrary sentinel subtraction.
    reentry = np.zeros((n, t), dtype=np.float64)
    reentry_mask = np.zeros((n, t), dtype=bool)
    for i in range(n):
        if not has_agents[i]:
            continue
        raw = raw_clear[i]
        res = safety_reserve_m[i]
        observed_contact = bool(np.isfinite(current_raw_clear[i]) and current_raw_clear[i] <= 0.0)
        seen_contact = observed_contact
        suffix_min = np.minimum.accumulate(res[::-1])[::-1]
        for q in range(t):
            seen_contact = seen_contact or bool(np.isfinite(raw[q]) and raw[q] <= 0.0)
            if res[q] < 0.0:
                val = res[q]
            else:
                val = suffix_min[q]
            reentry[i, q] = val / config.distance_scale
            reentry_mask[i, q] = seen_contact
    values[:, :, 3] = reentry
    masks[:, :, 3] = reentry_mask

    if not np.isfinite(values).all():
        bad = np.argwhere(~np.isfinite(values))[:5].tolist()
        raise ValueError(f"nonfinite heterogeneous constraint values examples={bad}")
    if np.any(~np.any(masks, axis=2)):
        raise ValueError("every prefix time must have at least one active heterogeneous constraint")

    diag = {
        "constraint_names": list(CONSTRAINT_NAMES),
        "rows": int(n),
        "prefix_steps": int(t),
        "has_agent_fraction": float(np.mean(has_agents)) if n else 0.0,
        "observed_contact_fraction": float(np.mean(np.nan_to_num(current_raw_clear, nan=np.inf) <= 0.0)) if n else 0.0,
        "reentry_active_fraction": float(np.mean(reentry_mask)) if n else 0.0,
    }
    return values, masks, diag


def active_constraint_indices(values: np.ndarray, masks: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    m = np.asarray(masks, dtype=bool)
    if v.shape != m.shape or v.ndim != 3 or v.shape[2] != NUM_CONSTRAINTS:
        raise ValueError("constraint selector shape mismatch")
    if np.any(~np.any(m, axis=2)):
        raise ValueError("no active constraint at some prefix time")
    z = np.where(m, v, np.inf)
    # np.argmin is deterministic and uses constraint order as the tie breaker.
    return np.argmin(z, axis=2).astype(np.int64)


@dataclass(frozen=True)
class ConeScaler:
    u_scale: np.ndarray


def fit_cone_scaler(u: np.ndarray) -> ConeScaler:
    x = np.asarray(u, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != RAW_CANDIDATE_DIM:
        raise ValueError("candidate response dimension mismatch")
    scale = np.sqrt(np.mean(x * x, axis=0, keepdims=True))
    scale = np.where(scale > 1.0e-8, scale, 1.0)
    return ConeScaler(u_scale=scale)


def base_features(u: np.ndarray, scaler: ConeScaler) -> np.ndarray:
    return np.asarray(u, dtype=np.float64) / scaler.u_scale


def cone_geometry(
    candidate_constraints: np.ndarray,
    nominal_constraints: np.ndarray,
    selector: np.ndarray,
) -> np.ndarray:
    hc = np.asarray(candidate_constraints, dtype=np.float64)
    h0 = np.asarray(nominal_constraints, dtype=np.float64)
    sel = np.asarray(selector, dtype=np.int64)
    if hc.shape != h0.shape or hc.ndim != 3 or hc.shape[1:] != (PREFIX_COMPLETE_STEPS, NUM_CONSTRAINTS):
        raise ValueError("constraint geometry shape mismatch")
    if sel.shape != hc.shape[:2]:
        raise ValueError("selector shape mismatch")
    n, t, _ = hc.shape
    delta = hc - h0
    geom = np.zeros((n, t, NUM_CONSTRAINTS, CONE_CHANNELS), dtype=np.float64)
    ii = np.arange(n)[:, None]
    tt = np.arange(t)[None, :]
    chosen_delta = delta[ii, tt, sel]
    chosen_nominal = h0[ii, tt, sel]
    geom[ii, tt, sel, 0] = chosen_delta
    geom[ii, tt, sel, 1] = chosen_delta * chosen_nominal
    out = geom.reshape(n, -1)
    if out.shape[1] != CONE_GEOMETRY_DIM:
        raise ValueError(f"cone geometry dim mismatch {out.shape[1]} != {CONE_GEOMETRY_DIM}")
    return out


def matched_features(
    u: np.ndarray,
    candidate_constraints: np.ndarray,
    nominal_constraints: np.ndarray,
    selector: np.ndarray,
    scaler: ConeScaler,
) -> np.ndarray:
    out = np.concatenate(
        [base_features(u, scaler), cone_geometry(candidate_constraints, nominal_constraints, selector)],
        axis=1,
    )
    if out.shape[1] != MATCHED_DIM:
        raise ValueError(f"matched dim mismatch {out.shape[1]} != {MATCHED_DIM}")
    return out


def selector_diagnostics(nominal_selector: np.ndarray, candidate_selector: np.ndarray) -> dict[str, Any]:
    nsel = np.asarray(nominal_selector, dtype=np.int64)
    csel = np.asarray(candidate_selector, dtype=np.int64)
    if nsel.shape != csel.shape or nsel.ndim != 2:
        raise ValueError("selector diagnostic shape mismatch")
    changed = nsel != csel
    nominal_counts = np.bincount(nsel.reshape(-1), minlength=NUM_CONSTRAINTS)
    candidate_counts = np.bincount(csel.reshape(-1), minlength=NUM_CONSTRAINTS)
    return {
        "selector_switch_fraction": float(np.mean(changed)) if changed.size else 0.0,
        "candidate_any_switch_fraction": float(np.mean(np.any(changed, axis=1))) if len(changed) else 0.0,
        "nominal_active_type_counts": {CONSTRAINT_NAMES[i]: int(nominal_counts[i]) for i in range(NUM_CONSTRAINTS)},
        "candidate_active_type_counts": {CONSTRAINT_NAMES[i]: int(candidate_counts[i]) for i in range(NUM_CONSTRAINTS)},
    }


def contract_checks() -> dict[str, bool]:
    rng = np.random.default_rng(48112)
    n, j = 7, 5
    r0 = np.zeros((n, RAW_CANDIDATE_DIM), dtype=np.float64)
    rc = r0.copy()
    r0[:, 7] = rc[:, 7] = 4.8
    r0[:, 8] = rc[:, 8] = 2.0
    for i in range(n):
        st0 = np.zeros((PREFIX_COMPLETE_STEPS, PREFIX_STATE_WIDTH), dtype=np.float64)
        stc = np.zeros_like(st0)
        st0[:, 0] = np.linspace(0.5, 5.0, PREFIX_COMPLETE_STEPS)
        stc[:, 0] = st0[:, 0] + 0.1 * (i + 1)
        stc[:, 1] = (0.1 + 0.08 * i) * np.linspace(0.0, 1.0, PREFIX_COMPLETE_STEPS)
        st0[:, 6] = stc[:, 6] = 5.0 + 0.2 * i
        st0[:, 7] = stc[:, 7] = 4.8
        st0[:, 8] = stc[:, 8] = 2.0
        r0[i, PREFIX_STATE_START:PREFIX_STATE_START + PREFIX_COMPLETE_STEPS * PREFIX_STATE_WIDTH] = st0.reshape(-1)
        rc[i, PREFIX_STATE_START:PREFIX_STATE_START + PREFIX_COMPLETE_STEPS * PREFIX_STATE_WIDTH] = stc.reshape(-1)
    agents = rng.normal(size=(n, j, 10))
    agents[:, :, 0:4] *= 0.04
    agents[:, :, 7] = 0.48
    agents[:, :, 8] = 0.4
    mask = np.ones((n, j), dtype=bool)
    cfg = ConstraintConeConfig(1.0, 0.5, 2.0, 5.0, 1.0, 2.5, -6.0, 60.0)
    hc, mc, _ = heterogeneous_constraint_paths(rc, agents, mask, sample_rate_hz=10.0, config=cfg)
    h0, m0, _ = heterogeneous_constraint_paths(r0, agents, mask, sample_rate_hz=10.0, config=cfg)
    cs = active_constraint_indices(hc, mc)
    ns = active_constraint_indices(h0, m0)
    u = rng.normal(size=(n, RAW_CANDIDATE_DIM))
    sc = fit_cone_scaler(u)
    fc = matched_features(u, hc, h0, cs, sc)
    fn = matched_features(u, hc, h0, ns, sc)
    zero = matched_features(np.zeros_like(u), h0, h0, ns, sc)

    perm = np.array([2, 4, 1, 0, 3])
    hcp, mcp, _ = heterogeneous_constraint_paths(rc, agents[:, perm], mask[:, perm], sample_rate_hz=10.0, config=cfg)
    h0p, m0p, _ = heterogeneous_constraint_paths(r0, agents[:, perm], mask[:, perm], sample_rate_hz=10.0, config=cfg)
    agent_perm_ok = bool(
        np.allclose(hc, hcp, rtol=0.0, atol=1.0e-12)
        and np.array_equal(mc, mcp)
        and np.allclose(h0, h0p, rtol=0.0, atol=1.0e-12)
        and np.array_equal(m0, m0p)
    )
    return {
        "constraint_count_4": NUM_CONSTRAINTS == 4,
        "prefix_steps_8": PREFIX_COMPLETE_STEPS == 8,
        "cone_geometry_dim_64": CONE_GEOMETRY_DIM == 64,
        "matched_dim_220": MATCHED_DIM == 220,
        "matched_family_same_dim": fc.shape[1] == fn.shape[1] == MATCHED_DIM,
        "nominal_zero_exact": bool(np.count_nonzero(zero) == 0),
        "agent_permutation_invariant": agent_perm_ok,
        "selectors_in_range": bool(np.all((cs >= 0) & (cs < NUM_CONSTRAINTS)) and np.all((ns >= 0) & (ns < NUM_CONSTRAINTS))),
        "constraint_values_finite": bool(np.isfinite(hc).all() and np.isfinite(h0).all()),
        "u_scale_finite_positive": bool(np.isfinite(sc.u_scale).all() and np.all(sc.u_scale > 0.0)),
    }
