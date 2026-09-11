from __future__ import annotations

"""Observation-consistent common-option constraint-work audit.

V48.113 established that actuator-projected executable continuations, heterogeneous
constraint activity, candidate-conditioned option switching, and observed-contact
persistent re-entry are all present in the audit population.  It nevertheless
failed to produce a transferable Support/Reserve orientation from eight sparse
pointwise samples of a selected option's candidate-minus-nominal constraint path.

V48.114 keeps the option selector, recovery horizon, physical constraints, frozen
156-D candidate response, and 220-D linear capacity fixed.  It changes only the
temporal response operator and preregisters a factorized audit:

  integral_response -- eight contiguous full-horizon bins, with the historical
                       ECJ channels E[Delta h] and E[Delta h * h0];
  constraint_work   -- the same bins and same selected option, decomposed into
                       positive-reserve work and negative-debt repayment work.

For a signed constraint h, candidate ha, and nominal h0, define per-state work

    w_res = [ha]_+ - [h0]_+
    w_debt = [-h0]_+ - [-ha]_+ .

Both are positive when the candidate improves the corresponding signed recovery
quantity, and w_res + w_debt = ha - h0 exactly.  Averaging each channel inside
eight fixed, contiguous bins gives 4 constraints x 8 bins x 2 channels = 64-D,
so every readout remains 156 + 64 = 220-D.  The bins cover every state of the
already configured recovery horizon; there is no horizon, threshold, capacity,
or regime sweep.

Candidate and nominal are always evaluated with the same recovery-option identity
inside a response.  Both the nominal-selected and candidate-selected controls are
retained, allowing the temporal-integration effect, signed-work decomposition,
and adaptive selector effect to be judged separately.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from ocrap.audits.constraint_native_orientation import RAW_CANDIDATE_DIM
from ocrap.audits.executable_constraint_jacobian import (
    ExecutableConstraintField,
    best_option_index,
    contract_checks as ecj_contract_checks,
    executable_constraint_field_from_sample,
    prefix_from_sample,
    recovery_options_from_sample,
)
from ocrap.audits.heterogeneous_constraint_normal_cone import NUM_CONSTRAINTS

ENGINEERING_VERSION = "v48.114.0-OC-CCW"
SCIENTIFIC_VERSION = "v48.114-OC-CCW"
ALGORITHM_NAME = "Observation-Consistent Common-Option Constraint Work Audit"

WORK_BINS = 8
WORK_CHANNELS = 2
WORK_GEOMETRY_DIM = WORK_BINS * NUM_CONSTRAINTS * WORK_CHANNELS
MATCHED_DIM = RAW_CANDIDATE_DIM + WORK_GEOMETRY_DIM


@dataclass(frozen=True)
class WorkScaler:
    u_scale: np.ndarray


def _bin_edges(length: int) -> np.ndarray:
    if int(length) < WORK_BINS:
        raise ValueError(f"CCW full horizon has {length} states, fewer than {WORK_BINS} work bins")
    # Integer partition that uses every horizon state exactly once.  The existing
    # production horizons are much longer than eight states; fail closed if a
    # future configuration cannot support the preregistered capacity-matched bins.
    edges = np.floor(np.linspace(0, int(length), WORK_BINS + 1)).astype(np.int64)
    edges[-1] = int(length)
    if np.any(np.diff(edges) <= 0):
        raise ValueError(f"CCW produced empty work bin for horizon length={length}: {edges.tolist()}")
    return edges


def _paired_full_paths(
    candidate: ExecutableConstraintField,
    nominal: ExecutableConstraintField,
    option_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    l = int(option_index)
    if l < 0 or l >= len(candidate.option_valid):
        raise ValueError(f"CCW selected option index out of range: {l}")
    if not bool(candidate.option_valid[l]) or not bool(nominal.option_valid[l]):
        raise ValueError(f"CCW selected invalid common option {l}")
    if candidate.full_values.shape != nominal.full_values.shape:
        raise ValueError(
            f"CCW candidate/nominal full-value mismatch {candidate.full_values.shape} vs {nominal.full_values.shape}"
        )
    if candidate.full_masks.shape != nominal.full_masks.shape:
        raise ValueError(
            f"CCW candidate/nominal full-mask mismatch {candidate.full_masks.shape} vs {nominal.full_masks.shape}"
        )
    if candidate.full_values.ndim != 3 or candidate.full_values.shape[2] != NUM_CONSTRAINTS:
        raise ValueError(f"CCW invalid full constraint field {candidate.full_values.shape}")

    active = candidate.full_masks[l] | nominal.full_masks[l]
    ha = np.where(active, candidate.full_values[l], 0.0).astype(np.float64, copy=False)
    h0 = np.where(active, nominal.full_values[l], 0.0).astype(np.float64, copy=False)
    if not (np.isfinite(ha).all() and np.isfinite(h0).all()):
        raise ValueError("CCW non-finite full-horizon constraint path")
    return ha, h0, active


def _bin_mean(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != NUM_CONSTRAINTS:
        raise ValueError(f"CCW expected [T,{NUM_CONSTRAINTS}] path, got {arr.shape}")
    edges = _bin_edges(arr.shape[0])
    out = np.zeros((WORK_BINS, NUM_CONSTRAINTS), dtype=np.float64)
    for b in range(WORK_BINS):
        lo, hi = int(edges[b]), int(edges[b + 1])
        out[b] = np.mean(arr[lo:hi], axis=0)
    return out


def integral_response_geometry(
    candidate: ExecutableConstraintField,
    nominal: ExecutableConstraintField,
    option_index: int,
) -> np.ndarray:
    """Full-horizon bin-integral control with the historical ECJ two channels."""
    ha, h0, _ = _paired_full_paths(candidate, nominal, option_index)
    delta = ha - h0
    geom = np.zeros((WORK_BINS, NUM_CONSTRAINTS, WORK_CHANNELS), dtype=np.float64)
    geom[:, :, 0] = _bin_mean(delta)
    geom[:, :, 1] = _bin_mean(delta * h0)
    out = geom.reshape(-1)
    if out.size != WORK_GEOMETRY_DIM:
        raise ValueError(f"CCW integral geometry dim mismatch {out.size} != {WORK_GEOMETRY_DIM}")
    return out


def constraint_work_geometry(
    candidate: ExecutableConstraintField,
    nominal: ExecutableConstraintField,
    option_index: int,
) -> np.ndarray:
    """Normalized signed work over the full executable recovery continuation.

    Channel 0 is positive-reserve work; channel 1 is debt-repayment work.  Their
    sum is exactly the bin-average signed response, so the transformation changes
    coordinates but does not invent candidate effect.
    """
    ha, h0, _ = _paired_full_paths(candidate, nominal, option_index)
    reserve_work = np.maximum(ha, 0.0) - np.maximum(h0, 0.0)
    debt_work = np.maximum(-h0, 0.0) - np.maximum(-ha, 0.0)
    response = ha - h0
    if not np.allclose(reserve_work + debt_work, response, rtol=0.0, atol=1.0e-12):
        raise ValueError("CCW reserve/debt work conservation failed")

    geom = np.zeros((WORK_BINS, NUM_CONSTRAINTS, WORK_CHANNELS), dtype=np.float64)
    geom[:, :, 0] = _bin_mean(reserve_work)
    geom[:, :, 1] = _bin_mean(debt_work)
    out = geom.reshape(-1)
    if out.size != WORK_GEOMETRY_DIM:
        raise ValueError(f"CCW work geometry dim mismatch {out.size} != {WORK_GEOMETRY_DIM}")
    return out


def work_conservation_error(
    candidate: ExecutableConstraintField,
    nominal: ExecutableConstraintField,
    option_index: int,
) -> float:
    ha, h0, _ = _paired_full_paths(candidate, nominal, option_index)
    direct = _bin_mean(ha - h0)
    g = constraint_work_geometry(candidate, nominal, option_index).reshape(
        WORK_BINS, NUM_CONSTRAINTS, WORK_CHANNELS
    )
    return float(np.max(np.abs((g[:, :, 0] + g[:, :, 1]) - direct)))


def fit_work_scaler(u: np.ndarray) -> WorkScaler:
    x = np.asarray(u, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != RAW_CANDIDATE_DIM:
        raise ValueError(f"CCW candidate response dimension mismatch {x.shape}")
    scale = np.sqrt(np.mean(x * x, axis=0, keepdims=True))
    scale = np.where(scale > 1.0e-8, scale, 1.0)
    return WorkScaler(u_scale=scale)


def base_features(u: np.ndarray, scaler: WorkScaler) -> np.ndarray:
    return np.asarray(u, dtype=np.float64) / scaler.u_scale


def matched_features(u: np.ndarray, geometry: np.ndarray, scaler: WorkScaler) -> np.ndarray:
    x = np.asarray(u, dtype=np.float64)
    g = np.asarray(geometry, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != RAW_CANDIDATE_DIM:
        raise ValueError(f"CCW candidate response batch mismatch {x.shape}")
    if g.ndim != 2 or g.shape != (len(x), WORK_GEOMETRY_DIM):
        raise ValueError(f"CCW work geometry batch mismatch u={x.shape} geometry={g.shape}")
    out = np.concatenate([base_features(x, scaler), g], axis=1)
    if out.shape[1] != MATCHED_DIM:
        raise ValueError(f"CCW matched dim mismatch {out.shape[1]} != {MATCHED_DIM}")
    return out


def pair_work_diagnostics(
    candidate: ExecutableConstraintField,
    nominal: ExecutableConstraintField,
) -> dict[str, Any]:
    lc = best_option_index(candidate)
    ln = best_option_index(nominal)
    gc = constraint_work_geometry(candidate, nominal, lc).reshape(
        WORK_BINS, NUM_CONSTRAINTS, WORK_CHANNELS
    )
    gi = integral_response_geometry(candidate, nominal, lc).reshape(
        WORK_BINS, NUM_CONSTRAINTS, WORK_CHANNELS
    )
    cvals = candidate.full_values[lc]
    cmask = candidate.full_masks[lc]
    z = np.where(cmask, cvals, np.inf)
    active = np.argmin(z, axis=1)
    names = ("clearance", "stopping", "route", "reentry")
    active_counts = {names[i]: int(np.count_nonzero(active == i)) for i in range(NUM_CONSTRAINTS)}
    return {
        "candidate_option": int(lc),
        "nominal_option": int(ln),
        "option_switched": bool(lc != ln),
        "candidate_mode": str(candidate.option_modes[lc]),
        "nominal_mode": str(nominal.option_modes[ln]),
        "candidate_selected_active_type_counts": active_counts,
        "candidate_selected_reentry_active": bool(np.any(candidate.full_masks[lc, :, 3])),
        "candidate_field_reentry_active_options": int(candidate.diagnostics.get("reentry_active_options", 0)),
        "reserve_work_nonzero": bool(np.any(np.abs(gc[:, :, 0]) > 1.0e-12)),
        "debt_work_nonzero": bool(np.any(np.abs(gc[:, :, 1]) > 1.0e-12)),
        "work_conservation_error": work_conservation_error(candidate, nominal, lc),
        "integral_response_nonzero": bool(np.any(np.abs(gi) > 1.0e-12)),
    }


def contract_checks() -> dict[str, bool]:
    # Reuse the ECJ synthetic physical contract, then test the new temporal
    # operator on a deterministic candidate perturbation.
    ecj = ecj_contract_checks()
    sample_rate = 10.0
    T = 8
    states = np.zeros((T, 9), dtype=np.float32)
    states[:, 0] = np.linspace(0.1, 2.0, T)
    states[:, 6] = 3.0
    states[:, 7] = 4.8
    states[:, 8] = 2.0
    controls = np.zeros((T, 4), dtype=np.float32)
    hist = np.zeros((3, 3, 16), dtype=np.float32)
    valid = np.ones((3, 3), dtype=np.float32)
    hist[:, 0, 10] = 4.8
    hist[:, 0, 11] = 2.0
    hist[:, 1, 0] = 1.0
    hist[:, 1, 10] = 4.5
    hist[:, 1, 11] = 2.0
    hist[:, 2, 0] = 15.0
    hist[:, 2, 3] = -1.0
    hist[:, 2, 10] = 4.0
    hist[:, 2, 11] = 1.8
    modes = np.asarray(["brake_lane", "post_contact_stabilize", "avoid_secondary"], dtype=object)
    params = np.asarray([[-3.0, 1.0, 0.0], [0.8, 1.2, -2.0], [3.5, -3.0, 8.0]], dtype=np.float32)
    d0: dict[str, Any] = {
        "scene_id": "synthetic",
        "time_index": np.int64(3),
        "candidate_index": np.int64(0),
        "is_nominal": np.int64(1),
        "agent_history": hist,
        "agent_valid": valid,
        "ego_state": np.asarray([0, 0, 3, 0, 0, 0, 3, 4.8, 2.0], dtype=np.float32),
        "prefix_states": states,
        "prefix_controls": controls,
        "prefix_macro_id": np.int64(0),
        "prefix_param": np.zeros((3,), dtype=np.float32),
        "recovery_modes": modes,
        "recovery_params": params,
        "option_valid": np.ones((3,), dtype=np.float32),
    }
    dc = dict(d0)
    dc["candidate_index"] = np.int64(1)
    cand_states = states.copy()
    cand_states[:, 1] = np.linspace(0.0, 0.4, T)
    dc["prefix_states"] = cand_states
    cfg = {
        "sample_rate_hz": sample_rate,
        "recovery_horizon_s": 2.0,
        "d_safe0_m": 1.0,
        "safe_time_headway_s": 0.5,
        "route_dev_max_m": 2.5,
        "margin_scales": {"distance": 2.0, "stop": 5.0, "route": 1.0},
        "control_limits": {"a_min": -6.0, "a_max": 3.0, "delta_max": 0.55, "j_max": 6.0, "steer_rate_max": 0.5},
    }
    # These imports are intentionally exercised here because current runtime
    # still owns prefix/library construction in the ECJ primitive.
    assert prefix_from_sample(d0) is not None
    assert recovery_options_from_sample(d0)
    f0 = executable_constraint_field_from_sample(d0, cfg)
    fc = executable_constraint_field_from_sample(dc, cfg)
    ln, lc = best_option_index(f0), best_option_index(fc)
    z_int = integral_response_geometry(f0, f0, ln)
    z_work = constraint_work_geometry(f0, f0, ln)
    gi = integral_response_geometry(fc, f0, lc)
    gw = constraint_work_geometry(fc, f0, lc)
    u = np.arange(5 * RAW_CANDIDATE_DIM, dtype=np.float64).reshape(5, RAW_CANDIDATE_DIM) / 1000.0
    sc = fit_work_scaler(u)
    feat = matched_features(u, np.tile(gw[None, :], (5, 1)), sc)
    return {
        "ecj_prerequisite_contract": bool(ecj and all(ecj.values())),
        "work_bins_8": WORK_BINS == 8,
        "constraint_count_4": NUM_CONSTRAINTS == 4,
        "work_channels_2": WORK_CHANNELS == 2,
        "work_geometry_dim_64": WORK_GEOMETRY_DIM == 64,
        "matched_dim_220": MATCHED_DIM == 220,
        "nominal_integral_zero_exact": bool(np.count_nonzero(z_int) == 0),
        "nominal_work_zero_exact": bool(np.count_nonzero(z_work) == 0),
        "integral_response_finite": bool(np.isfinite(gi).all()),
        "constraint_work_finite": bool(np.isfinite(gw).all()),
        "work_conservation_exact": work_conservation_error(fc, f0, lc) <= 1.0e-12,
        "full_horizon_used": bool(fc.full_values.shape[1] == int(round(sample_rate * cfg["recovery_horizon_s"]))),
        "actuator_projected": bool(fc.diagnostics.get("projected_control_envelope")),
        "reentry_activity_reachable": bool(fc.diagnostics.get("reentry_active_options", 0) > 0),
        "matched_family_same_dim": feat.shape[1] == MATCHED_DIM,
        "u_scale_finite_positive": bool(np.isfinite(sc.u_scale).all() and np.all(sc.u_scale > 0.0)),
    }
