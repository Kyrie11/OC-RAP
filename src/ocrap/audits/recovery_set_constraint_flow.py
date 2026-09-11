from __future__ import annotations

"""Observation-consistent selector-free recovery-set constraint-flow audit.

V48.114 showed that full-horizon selected-option integration and signed
reserve/debt work do not form a population-stable Support/Reserve orientation,
even though executable constraint activity, option switching and post-contact
re-entry are all present.  The preregistered successor therefore removes the
*pre-readout hard option selector* rather than adding capacity.

For every valid recovery option g_l shared by candidate and nominal, compute the
same-option actuator-projected full-horizon constraint response.  The recovery
library is then treated as an empirical set/measure and aggregated *before* any
option selection:

    set_integral = E_l[ bin_mean(Delta h), bin_mean(Delta h * h0) ]
    set_work     = E_l[ positive-reserve work, negative-debt repayment work ]

where the expectation is the uniform mean over the common valid option set.
This is the canonical parameter-free first moment of the finite recovery set: it
is permutation invariant in option ordering, preserves same-option causal
correspondence inside every summand, and does not let a candidate-conditioned
argmax identity leak into the feature extractor.  Downstream OC-MERO may still
select a common option after observation-consistent evaluation; this audit asks
whether the action effect should be represented *before* that selection.

The historical frozen 156-D candidate response is retained and both set-flow
families add exactly 64 dimensions, preserving the 220-D convex capacity.  No
threshold, horizon, option-count, regime, source or encoder sweep is introduced.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from ocrap.audits.common_option_constraint_work import (
    WORK_BINS,
    WORK_CHANNELS,
    WORK_GEOMETRY_DIM,
    constraint_work_geometry,
    integral_response_geometry,
)
from ocrap.audits.constraint_native_orientation import RAW_CANDIDATE_DIM
from ocrap.audits.executable_constraint_jacobian import (
    ExecutableConstraintField,
    contract_checks as ecj_contract_checks,
    executable_constraint_field_from_sample,
    prefix_from_sample,
    recovery_options_from_sample,
)
from ocrap.audits.heterogeneous_constraint_normal_cone import NUM_CONSTRAINTS

ENGINEERING_VERSION = "v48.115.0-OC-RSCF"
SCIENTIFIC_VERSION = "v48.115-OC-RSCF"
ALGORITHM_NAME = "Observation-Consistent Recovery-Set Constraint Flow Audit"

SET_GEOMETRY_DIM = WORK_GEOMETRY_DIM
MATCHED_DIM = RAW_CANDIDATE_DIM + SET_GEOMETRY_DIM


@dataclass(frozen=True)
class SetFlowScaler:
    u_scale: np.ndarray


def _common_valid_indices(candidate: ExecutableConstraintField, nominal: ExecutableConstraintField) -> np.ndarray:
    if candidate.full_values.shape != nominal.full_values.shape:
        raise ValueError(
            f"RSCF candidate/nominal full-value mismatch {candidate.full_values.shape} vs {nominal.full_values.shape}"
        )
    if candidate.full_masks.shape != nominal.full_masks.shape:
        raise ValueError(
            f"RSCF candidate/nominal full-mask mismatch {candidate.full_masks.shape} vs {nominal.full_masks.shape}"
        )
    if candidate.option_valid.shape != nominal.option_valid.shape:
        raise ValueError("RSCF candidate/nominal option-valid shape mismatch")
    common = np.flatnonzero(candidate.option_valid & nominal.option_valid)
    if common.size == 0:
        raise ValueError("RSCF requires at least one common valid recovery option")
    return common.astype(np.int64, copy=False)


def _mean_option_geometry(
    candidate: ExecutableConstraintField,
    nominal: ExecutableConstraintField,
    *,
    work: bool,
) -> np.ndarray:
    idx = _common_valid_indices(candidate, nominal)
    fn = constraint_work_geometry if work else integral_response_geometry
    geoms = np.stack([fn(candidate, nominal, int(l)) for l in idx], axis=0).astype(np.float64)
    out = np.mean(geoms, axis=0)
    if out.shape != (SET_GEOMETRY_DIM,):
        raise ValueError(f"RSCF set geometry dim mismatch {out.shape} != {(SET_GEOMETRY_DIM,)}")
    if not np.isfinite(out).all():
        raise ValueError("RSCF non-finite recovery-set geometry")
    return out


def recovery_set_integral_geometry(
    candidate: ExecutableConstraintField,
    nominal: ExecutableConstraintField,
) -> np.ndarray:
    """Permutation-invariant mean of historical full-horizon ECJ channels over all common options."""
    return _mean_option_geometry(candidate, nominal, work=False)


def recovery_set_work_geometry(
    candidate: ExecutableConstraintField,
    nominal: ExecutableConstraintField,
) -> np.ndarray:
    """Permutation-invariant mean of signed reserve/debt work over all common options."""
    return _mean_option_geometry(candidate, nominal, work=True)


def set_work_conservation_error(
    candidate: ExecutableConstraintField,
    nominal: ExecutableConstraintField,
) -> float:
    integ = recovery_set_integral_geometry(candidate, nominal).reshape(
        WORK_BINS, NUM_CONSTRAINTS, WORK_CHANNELS
    )
    work = recovery_set_work_geometry(candidate, nominal).reshape(
        WORK_BINS, NUM_CONSTRAINTS, WORK_CHANNELS
    )
    # Work channels sum to E_l[Delta h]. Integral channel 0 is exactly the same
    # empirical-set mean response. Channel 1 intentionally remains the historical
    # Delta h * h0 control and is not part of this identity.
    return float(np.max(np.abs((work[:, :, 0] + work[:, :, 1]) - integ[:, :, 0])))


def option_permutation_invariance_error(
    candidate: ExecutableConstraintField,
    nominal: ExecutableConstraintField,
) -> float:
    """Exact numerical check that set flow is invariant to recovery-option ordering."""
    L = len(candidate.option_valid)
    perm = np.arange(L - 1, -1, -1, dtype=np.int64)

    def pfield(f: ExecutableConstraintField) -> ExecutableConstraintField:
        return ExecutableConstraintField(
            values=f.values[perm],
            masks=f.masks[perm],
            full_values=f.full_values[perm],
            full_masks=f.full_masks[perm],
            option_valid=f.option_valid[perm],
            option_scores=f.option_scores[perm],
            option_modes=tuple(f.option_modes[int(i)] for i in perm),
            diagnostics=dict(f.diagnostics),
        )

    a = recovery_set_work_geometry(candidate, nominal)
    b = recovery_set_work_geometry(pfield(candidate), pfield(nominal))
    c = recovery_set_integral_geometry(candidate, nominal)
    d = recovery_set_integral_geometry(pfield(candidate), pfield(nominal))
    return float(max(np.max(np.abs(a - b)), np.max(np.abs(c - d))))


def set_flow_diagnostics(
    candidate: ExecutableConstraintField,
    nominal: ExecutableConstraintField,
) -> dict[str, Any]:
    idx = _common_valid_indices(candidate, nominal)
    work_by_option = np.stack(
        [constraint_work_geometry(candidate, nominal, int(l)) for l in idx], axis=0
    ).astype(np.float64)
    integral_by_option = np.stack(
        [integral_response_geometry(candidate, nominal, int(l)) for l in idx], axis=0
    ).astype(np.float64)
    centered = work_by_option - np.mean(work_by_option, axis=0, keepdims=True)
    dispersion = float(np.sqrt(np.mean(centered * centered)))
    reentry_options = 0
    for l in idx:
        if bool(np.any(candidate.full_masks[int(l), :, 3] | nominal.full_masks[int(l), :, 3])):
            reentry_options += 1
    return {
        "common_valid_option_count": int(idx.size),
        "set_work_nonzero": bool(np.any(np.abs(work_by_option) > 1.0e-12)),
        "set_integral_nonzero": bool(np.any(np.abs(integral_by_option) > 1.0e-12)),
        "option_flow_dispersion": dispersion,
        "option_flow_diverse": bool(dispersion > 1.0e-12),
        "reentry_active_common_option_count": int(reentry_options),
        "reentry_available_in_set": bool(reentry_options > 0),
        "set_work_conservation_error": set_work_conservation_error(candidate, nominal),
        "option_permutation_invariance_error": option_permutation_invariance_error(candidate, nominal),
    }


def fit_set_flow_scaler(u: np.ndarray) -> SetFlowScaler:
    x = np.asarray(u, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != RAW_CANDIDATE_DIM:
        raise ValueError(f"RSCF candidate response dimension mismatch {x.shape}")
    scale = np.sqrt(np.mean(x * x, axis=0, keepdims=True))
    scale = np.where(scale > 1.0e-8, scale, 1.0)
    return SetFlowScaler(u_scale=scale)


def base_features(u: np.ndarray, scaler: SetFlowScaler) -> np.ndarray:
    return np.asarray(u, dtype=np.float64) / scaler.u_scale


def matched_features(u: np.ndarray, geometry: np.ndarray, scaler: SetFlowScaler) -> np.ndarray:
    x = np.asarray(u, dtype=np.float64)
    g = np.asarray(geometry, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != RAW_CANDIDATE_DIM:
        raise ValueError(f"RSCF candidate response batch mismatch {x.shape}")
    if g.ndim != 2 or g.shape != (len(x), SET_GEOMETRY_DIM):
        raise ValueError(f"RSCF set geometry batch mismatch u={x.shape} geometry={g.shape}")
    out = np.concatenate([base_features(x, scaler), g], axis=1)
    if out.shape[1] != MATCHED_DIM:
        raise ValueError(f"RSCF matched dim mismatch {out.shape[1]} != {MATCHED_DIM}")
    return out


def contract_checks() -> dict[str, bool]:
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
    assert prefix_from_sample(d0) is not None
    assert recovery_options_from_sample(d0)
    f0 = executable_constraint_field_from_sample(d0, cfg)
    fc = executable_constraint_field_from_sample(dc, cfg)
    z0_i = recovery_set_integral_geometry(f0, f0)
    z0_w = recovery_set_work_geometry(f0, f0)
    gi = recovery_set_integral_geometry(fc, f0)
    gw = recovery_set_work_geometry(fc, f0)
    u = np.arange(5 * RAW_CANDIDATE_DIM, dtype=np.float64).reshape(5, RAW_CANDIDATE_DIM) / 1000.0
    sc = fit_set_flow_scaler(u)
    feat = matched_features(u, np.tile(gw[None, :], (5, 1)), sc)
    diag = set_flow_diagnostics(fc, f0)
    return {
        "ecj_prerequisite_contract": bool(ecj and all(ecj.values())),
        "work_bins_8": WORK_BINS == 8,
        "constraint_count_4": NUM_CONSTRAINTS == 4,
        "set_channels_2": WORK_CHANNELS == 2,
        "set_geometry_dim_64": SET_GEOMETRY_DIM == 64,
        "matched_dim_220": MATCHED_DIM == 220,
        "nominal_set_integral_zero_exact": bool(np.count_nonzero(z0_i) == 0),
        "nominal_set_work_zero_exact": bool(np.count_nonzero(z0_w) == 0),
        "set_integral_finite": bool(np.isfinite(gi).all()),
        "set_work_finite": bool(np.isfinite(gw).all()),
        "set_work_conservation_exact": float(diag["set_work_conservation_error"]) <= 1.0e-12,
        "option_permutation_invariant": float(diag["option_permutation_invariance_error"]) <= 1.0e-12,
        "multiple_common_options": int(diag["common_valid_option_count"]) >= 2,
        "option_flow_diverse": bool(diag["option_flow_diverse"]),
        "reentry_activity_reachable": bool(diag["reentry_available_in_set"]),
        "actuator_projected": bool(fc.diagnostics.get("projected_control_envelope")),
        "matched_family_same_dim": feat.shape[1] == MATCHED_DIM,
        "u_scale_finite_positive": bool(np.isfinite(sc.u_scale).all() and np.all(sc.u_scale > 0.0)),
    }
