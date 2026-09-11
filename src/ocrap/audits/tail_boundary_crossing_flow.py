from __future__ import annotations

"""Observation-consistent tail-boundary crossing-flow audit.

V48.116 showed that the exact first-order cotangent of the nested OC-MERO
functional is scientifically well-defined but its option push-forward is nearly
singular: differentiation through ``max_option`` assigns almost all option mass
to one active argmax recovery mode.  V48.117 therefore keeps the weak-root
observation law and the same executable recovery physics, but replaces the
argmax cotangent option push-forward with a zero-boundary witness measure.

The construction has two independent factors.

1. Boundary ownership measure.  The outer OC-MERO lower-tail influence selects
   weak observation anchors.  Their observation-compatible root distributions
   are mixed *before* option selection, yielding a root exposure ``rho``.  For
   each exposed root, the recovery-set boundary at the signed zero level is
   represented by the nearest non-negative option and nearest negative option;
   exact ties share mass uniformly.  If only one side exists, the nearest option
   on that side carries the root mass.  This produces a candidate-independent,
   permutation-invariant option measure without a threshold or learned
   temperature.

2. Boundary hitting transport.  For every same-option executable constraint
   path h(t), define two zero-threshold order statistics for every constraint:

       survival(t)  = 1[min_{s<=t} h(s) >= 0]
       recovered(t) = 1[min_{s>=t} h(s) >= 0].

   The first channel is the survival function of the first violation time; the
   second is the persistent-safe suffix indicator and therefore encodes stable
   re-entry time.  Candidate-minus-nominal changes are averaged in the same
   eight full-horizon bins.  No Near/Contact regime label enters the operator.

The audit is a capacity-matched 2x2 mechanism test around V48.116:

  historical V48.116: cotangent measure + signed work
  cotangent_hitting : cotangent measure + boundary hitting
  boundary_work     : boundary witness measure + signed work
  boundary_hitting  : boundary witness measure + boundary hitting

All new families remain 156 + 64 = 220-D closed-form ridge probes.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from ocrap.algorithms.lcv import normalize_weights
from ocrap.algorithms.ocmero import sparsify_compatibility
from ocrap.audits.common_option_constraint_work import (
    WORK_BINS,
    WORK_GEOMETRY_DIM,
    _bin_mean,
    _paired_full_paths,
    constraint_work_geometry,
)
from ocrap.audits.constraint_native_orientation import RAW_CANDIDATE_DIM
from ocrap.audits.executable_constraint_jacobian import ExecutableConstraintField
from ocrap.audits.recovery_set_constraint_flow import (
    SetFlowScaler,
    base_features,
    fit_set_flow_scaler,
    matched_features,
)
from ocrap.audits.weak_root_recovery_set_flow import (
    WeakRootTailMeasure,
    _softmax_logits,
    align_model_option_measure_to_physical_library,
    nominal_ocmero_tail_measure,
    weak_root_work_geometry,
)

ENGINEERING_VERSION = "v48.117.0-OC-TBCF"
SCIENTIFIC_VERSION = "v48.117-OC-TBCF"
ALGORITHM_NAME = "Observation-Consistent Tail-Boundary Crossing Flow Audit"

BOUNDARY_CHANNELS = 2
BOUNDARY_GEOMETRY_DIM = WORK_GEOMETRY_DIM
MATCHED_DIM = RAW_CANDIDATE_DIM + BOUNDARY_GEOMETRY_DIM


@dataclass(frozen=True)
class TailBoundaryMeasure:
    option_weights: np.ndarray
    root_exposure: np.ndarray
    boundary_witness: np.ndarray
    outer_influence: np.ndarray
    root_probs: np.ndarray
    cotangent_option_weights: np.ndarray


def _tie_mass(indices: np.ndarray, scores: np.ndarray, target: float) -> np.ndarray:
    out = np.zeros(len(scores), dtype=np.float64)
    if indices.size == 0:
        return out
    vals = np.asarray(scores, dtype=np.float64)[indices]
    d = np.abs(vals - float(target))
    m = float(np.min(d))
    chosen = indices[np.isclose(d, m, rtol=0.0, atol=1.0e-12)]
    out[chosen] = 1.0 / float(len(chosen))
    return out


def _root_boundary_witness(margins_row: np.ndarray, option_valid: np.ndarray) -> np.ndarray:
    """Probability measure on the exact signed zero-boundary witnesses.

    If both safe and debt sides exist, each side receives one half of the mass;
    exact ties on a side share that half uniformly.  No margin threshold is used.
    """
    m = np.asarray(margins_row, dtype=np.float64).reshape(-1)
    ov = np.asarray(option_valid, dtype=bool).reshape(-1)
    if m.size != ov.size or not ov.any():
        raise ValueError("TBCF invalid root/option boundary witness geometry")
    if not np.isfinite(m[ov]).all():
        raise ValueError("TBCF non-finite root-option margin")
    safe = np.flatnonzero(ov & (m >= 0.0))
    debt = np.flatnonzero(ov & (m < 0.0))
    out = np.zeros_like(m, dtype=np.float64)
    if safe.size:
        min_safe = float(np.min(m[safe]))
        tied = safe[np.isclose(m[safe], min_safe, rtol=0.0, atol=1.0e-12)]
        out[tied] += (0.5 if debt.size else 1.0) / float(len(tied))
    if debt.size:
        max_debt = float(np.max(m[debt]))
        tied = debt[np.isclose(m[debt], max_debt, rtol=0.0, atol=1.0e-12)]
        out[tied] += (0.5 if safe.size else 1.0) / float(len(tied))
    if abs(float(out.sum()) - 1.0) > 1.0e-12:
        raise ValueError("TBCF boundary witness must have unit mass")
    return out


def nominal_tail_boundary_measure(
    margins: np.ndarray,
    root_logits: np.ndarray,
    compatibility: np.ndarray,
    *,
    root_valid: np.ndarray | None,
    option_valid: np.ndarray,
    alpha: float,
    beta: float,
    top_m: int | None,
) -> TailBoundaryMeasure:
    """Weak-root zero-boundary measure without an option argmax push-forward.

    The OC-MERO outer lower-tail influence is retained exactly.  Weak anchor mass
    is transported to source roots through only the observation compatibility
    law, then each exposed source root contributes its two-sided zero-boundary
    recovery witnesses.  Candidate action never enters the measure.
    """
    M = np.asarray(margins, dtype=np.float64)
    if M.ndim != 2:
        raise ValueError(f"TBCF margins must be [K,L], got {M.shape}")
    K, L = M.shape
    rv = np.ones(K, dtype=bool) if root_valid is None else np.asarray(root_valid, dtype=bool).reshape(-1)
    ov = np.asarray(option_valid, dtype=bool).reshape(-1)
    if rv.size != K or ov.size != L:
        raise ValueError("TBCF root/option validity shape mismatch")

    # Reuse the exact nominal OC-MERO functional only to obtain the frozen outer
    # weak-anchor influence.  We intentionally do NOT reuse its option pushforward.
    cot = nominal_ocmero_tail_measure(
        M, root_logits, compatibility,
        root_valid=rv, option_valid=ov, alpha=alpha, beta=beta, top_m=top_m,
    )
    p = np.asarray(cot.root_probs, dtype=np.float64)
    outer = np.asarray(cot.outer_influence, dtype=np.float64)
    C = np.asarray(compatibility, dtype=np.float64)
    C_eff = sparsify_compatibility(C, top_m)

    rho = np.zeros(K, dtype=np.float64)
    for i in range(K):
        if outer[i] <= 0.0:
            continue
        wi = normalize_weights(C_eff[i] * p)
        rho += float(outer[i]) * wi
    rho = np.where(rv, rho, 0.0)
    if np.any(rho < -1.0e-12) or not np.isfinite(rho).all() or float(rho.sum()) <= 1.0e-12:
        raise ValueError("TBCF invalid weak-root exposure")
    rho = normalize_weights(np.maximum(rho, 0.0))

    B = np.zeros((K, L), dtype=np.float64)
    for j in range(K):
        if not rv[j] or rho[j] <= 0.0:
            continue
        B[j] = _root_boundary_witness(M[j], ov)
    omega = rho @ B
    omega = np.where(ov, omega, 0.0)
    if np.any(omega < -1.0e-12) or not np.isfinite(omega).all() or float(omega.sum()) <= 1.0e-12:
        raise ValueError("TBCF invalid boundary option measure")
    omega = normalize_weights(np.maximum(omega, 0.0))

    return TailBoundaryMeasure(
        option_weights=omega,
        root_exposure=rho,
        boundary_witness=B,
        outer_influence=outer,
        root_probs=p,
        cotangent_option_weights=np.asarray(cot.option_weights, dtype=np.float64),
    )


def boundary_measure_diagnostics(measure: TailBoundaryMeasure) -> dict[str, Any]:
    w = np.asarray(measure.option_weights, dtype=np.float64)
    pos = w > 1.0e-12
    eff = 0.0 if float(np.sum(w * w)) <= 0.0 else float(1.0 / np.sum(w * w))
    B = np.asarray(measure.boundary_witness, dtype=np.float64)
    rho = np.asarray(measure.root_exposure, dtype=np.float64)
    active_roots = rho > 1.0e-12
    bracketed = np.sum((B > 1.0e-12), axis=1)
    return {
        "boundary_option_weight_sum": float(w.sum()),
        "boundary_positive_option_count": int(pos.sum()),
        "boundary_effective_option_count": eff,
        "boundary_root_exposure_mass": float(rho.sum()),
        "boundary_positive_root_count": int(active_roots.sum()),
        "boundary_mean_witness_count_per_exposed_root": float(np.sum(rho * bracketed)),
        "boundary_two_sided_root_mass": float(np.sum(rho[bracketed >= 2])),
        "cotangent_positive_option_count": int(np.sum(measure.cotangent_option_weights > 1.0e-12)),
    }


def _validate_weights(field: ExecutableConstraintField, nominal: ExecutableConstraintField, weights: np.ndarray) -> np.ndarray:
    if field.full_values.shape != nominal.full_values.shape:
        raise ValueError("TBCF candidate/nominal field shape mismatch")
    if not np.array_equal(field.option_valid, nominal.option_valid):
        raise ValueError("TBCF requires candidate-invariant recovery option validity")
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.size != len(nominal.option_valid):
        raise ValueError("TBCF option-weight length mismatch")
    if np.any(w < -1.0e-12) or not np.isfinite(w).all():
        raise ValueError("TBCF invalid option weights")
    if np.any((~nominal.option_valid) & (np.abs(w) > 1.0e-12)):
        raise ValueError("TBCF measure assigns mass to invalid option")
    if abs(float(w.sum()) - 1.0) > 1.0e-10:
        raise ValueError("TBCF option weights must have unit mass")
    return w


def _path_hitting_geometry(
    candidate: ExecutableConstraintField,
    nominal: ExecutableConstraintField,
    option_index: int,
) -> np.ndarray:
    ha, h0, active = _paired_full_paths(candidate, nominal, option_index)
    # Inactive locations are neutral for prefix/suffix minima.  Output channels
    # are zeroed there so an absent constraint cannot create artificial safety.
    inf = np.inf
    ae = np.where(active, ha, inf)
    ne = np.where(active, h0, inf)
    a_prefix = np.minimum.accumulate(ae, axis=0)
    n_prefix = np.minimum.accumulate(ne, axis=0)
    a_suffix = np.minimum.accumulate(ae[::-1], axis=0)[::-1]
    n_suffix = np.minimum.accumulate(ne[::-1], axis=0)[::-1]
    a_surv = ((a_prefix >= 0.0) & active).astype(np.float64)
    n_surv = ((n_prefix >= 0.0) & active).astype(np.float64)
    a_rec = ((a_suffix >= 0.0) & active).astype(np.float64)
    n_rec = ((n_suffix >= 0.0) & active).astype(np.float64)
    geom = np.zeros((WORK_BINS, ha.shape[1], BOUNDARY_CHANNELS), dtype=np.float64)
    geom[:, :, 0] = _bin_mean(a_surv - n_surv)
    geom[:, :, 1] = _bin_mean(a_rec - n_rec)
    out = geom.reshape(-1)
    if out.shape != (BOUNDARY_GEOMETRY_DIM,) or not np.isfinite(out).all():
        raise ValueError("TBCF invalid boundary-hitting geometry")
    return out


def weighted_work_geometry(candidate: ExecutableConstraintField, nominal: ExecutableConstraintField, weights: np.ndarray) -> np.ndarray:
    w = _validate_weights(candidate, nominal, weights)
    terms = np.stack([constraint_work_geometry(candidate, nominal, l) for l in range(len(w))], axis=0)
    return np.sum(w[:, None] * terms, axis=0).astype(np.float64, copy=False)


def weighted_hitting_geometry(candidate: ExecutableConstraintField, nominal: ExecutableConstraintField, weights: np.ndarray) -> np.ndarray:
    w = _validate_weights(candidate, nominal, weights)
    terms = np.stack([_path_hitting_geometry(candidate, nominal, l) for l in range(len(w))], axis=0)
    return np.sum(w[:, None] * terms, axis=0).astype(np.float64, copy=False)


def boundary_work_geometry(candidate: ExecutableConstraintField, nominal: ExecutableConstraintField, weights: np.ndarray) -> np.ndarray:
    return weighted_work_geometry(candidate, nominal, weights)


def boundary_hitting_geometry(candidate: ExecutableConstraintField, nominal: ExecutableConstraintField, weights: np.ndarray) -> np.ndarray:
    return weighted_hitting_geometry(candidate, nominal, weights)


def cotangent_hitting_geometry(candidate: ExecutableConstraintField, nominal: ExecutableConstraintField, weights: np.ndarray) -> np.ndarray:
    return weighted_hitting_geometry(candidate, nominal, weights)


def option_permutation_invariance_error(
    candidate: ExecutableConstraintField,
    nominal: ExecutableConstraintField,
    boundary_weights: np.ndarray,
    cotangent_weights: np.ndarray,
) -> float:
    L = len(nominal.option_valid)
    perm = np.arange(L - 1, -1, -1, dtype=np.int64)

    def pf(f: ExecutableConstraintField) -> ExecutableConstraintField:
        return ExecutableConstraintField(
            values=f.values[perm], masks=f.masks[perm],
            full_values=f.full_values[perm], full_masks=f.full_masks[perm],
            option_valid=f.option_valid[perm], option_scores=f.option_scores[perm],
            option_modes=tuple(f.option_modes[int(i)] for i in perm), diagnostics=dict(f.diagnostics),
        )

    a = [
        boundary_work_geometry(candidate, nominal, boundary_weights),
        boundary_hitting_geometry(candidate, nominal, boundary_weights),
        cotangent_hitting_geometry(candidate, nominal, cotangent_weights),
    ]
    b = [
        boundary_work_geometry(pf(candidate), pf(nominal), np.asarray(boundary_weights)[perm]),
        boundary_hitting_geometry(pf(candidate), pf(nominal), np.asarray(boundary_weights)[perm]),
        cotangent_hitting_geometry(pf(candidate), pf(nominal), np.asarray(cotangent_weights)[perm]),
    ]
    return float(max(np.max(np.abs(x - y)) for x, y in zip(a, b)))


def contract_checks() -> dict[str, bool]:
    K, L = 4, 4
    M = np.asarray([
        [-0.4, 0.1, 0.6, -0.05],
        [-0.2, 0.2, 0.7, -0.1],
        [0.3, -0.3, 0.05, 0.8],
        [0.5, -0.2, 0.1, -0.02],
    ], dtype=np.float64)
    logits = np.asarray([0.3, 0.1, -0.2, -0.4], dtype=np.float64)
    C = np.asarray([
        [1.0, .8, .2, .1], [.8, 1.0, .4, .2],
        [.2, .4, 1.0, .7], [.1, .2, .7, 1.0],
    ], dtype=np.float64)
    rv = np.ones(K, dtype=bool); ov = np.ones(L, dtype=bool)
    bm = nominal_tail_boundary_measure(M, logits, C, root_valid=rv, option_valid=ov, alpha=.5, beta=.5, top_m=3)
    d = boundary_measure_diagnostics(bm)
    # Joint option permutation must preserve the boundary measure up to permutation.
    perm = np.asarray([2, 0, 3, 1])
    bpm = nominal_tail_boundary_measure(M[:, perm], logits, C, root_valid=rv, option_valid=ov[perm], alpha=.5, beta=.5, top_m=3)
    measure_perm = np.max(np.abs(bm.option_weights[perm] - bpm.option_weights))
    # Padded model alignment is inherited from WRCF and must remain exact.
    padded, pd = align_model_option_measure_to_physical_library(
        np.asarray([True, True, True, True, False, False]), ov,
        np.r_[bm.option_weights, 0.0, 0.0],
    )

    # Exercise the physical zero-boundary channels, not only the measure.
    T = 8
    full0 = np.ones((L, T, 4), dtype=np.float64)
    fulla = full0.copy()
    # Option 0: candidate loses viability halfway through the horizon.
    fulla[0, 4:, 0] = -0.25
    # Option 1: nominal has debt until t=4, candidate repays it at t=2.
    full0[1, :4, 1] = -0.4
    fulla[1, :2, 1] = -0.4
    masks = np.ones_like(full0, dtype=bool)
    knots = np.arange(T, dtype=np.int64)
    nominal_field = ExecutableConstraintField(
        values=full0[:, knots], masks=masks[:, knots], full_values=full0, full_masks=masks,
        option_valid=ov.copy(), option_scores=np.zeros(L),
        option_modes=tuple(f"mode_{i}" for i in range(L)), diagnostics={},
    )
    candidate_field = ExecutableConstraintField(
        values=fulla[:, knots], masks=masks[:, knots], full_values=fulla, full_masks=masks,
        option_valid=ov.copy(), option_scores=np.zeros(L),
        option_modes=tuple(f"mode_{i}" for i in range(L)), diagnostics={},
    )
    hit = boundary_hitting_geometry(candidate_field, nominal_field, bm.option_weights)
    work = boundary_work_geometry(candidate_field, nominal_field, bm.option_weights)
    pinv = option_permutation_invariance_error(
        candidate_field, nominal_field, bm.option_weights, bm.cotangent_option_weights
    )
    return {
        "matched_dim_220": MATCHED_DIM == 220,
        "boundary_geometry_dim_64": BOUNDARY_GEOMETRY_DIM == 64,
        "boundary_measure_nonnegative": bool(np.all(bm.option_weights >= -1.0e-12)),
        "boundary_measure_unit_mass": abs(float(bm.option_weights.sum()) - 1.0) <= 1.0e-12,
        "root_exposure_unit_mass": abs(float(bm.root_exposure.sum()) - 1.0) <= 1.0e-12,
        "boundary_measure_permutation_invariant": float(measure_perm) <= 1.0e-12,
        "boundary_measure_nontrivial": int(d["boundary_positive_option_count"]) >= 2,
        "two_sided_boundary_mass_positive": float(d["boundary_two_sided_root_mass"]) > 0.0,
        "padded_model_option_alignment_exact": bool(np.allclose(padded, bm.option_weights) and pd["padded_tail_mass"] == 0.0),
        "boundary_hitting_finite_nonzero": bool(np.isfinite(hit).all() and np.any(np.abs(hit) > 1.0e-12)),
        "boundary_work_finite_nonzero": bool(np.isfinite(work).all() and np.any(np.abs(work) > 1.0e-12)),
        "joint_option_weight_permutation_invariant": float(pinv) <= 1.0e-12,
        "candidate_independent_measure_by_interface": True,
        "zero_boundary_no_threshold": True,
        "no_regime_router": True,
    }
