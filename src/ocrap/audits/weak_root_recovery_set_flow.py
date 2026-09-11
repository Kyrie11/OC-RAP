from __future__ import annotations

"""OC-MERO weak-root-conditioned recovery-set constraint-flow audit.

V48.115 established that removing a candidate-dependent hard recovery-option
selector improves Support and Reserve ordering in three of four unique roles,
but a uniform first moment over the executable recovery library still fails as
a population-stable absolute orientation, with certificate-Contact the dominant
reversal.  The preregistered successor changes exactly one object: the measure
used to integrate option-resolved physical constraint flow.

For a scene-time group, the *nominal* frozen OC-RAP model provides its native
root posterior, observation-compatibility kernel, and root x recovery-option
margin matrix.  We differentiate the exact nested lower-tail OC-MERO functional
at that nominal anchor.  Its deterministic LCVAR subgradient

    G[j,l] = d R_dep(M0) / d M0[j,l]

is non-negative and sums to one.  Pushing this weak-root cotangent forward over
roots gives a canonical candidate-independent recovery-option measure

    omega[l] = sum_j G[j,l].

The same omega is then used for every candidate in the group to integrate the
already-established same-option, actuator-projected full-horizon constraint
response:

    tail_integral = sum_l omega[l] * F_integral_l(a, a0)
    tail_work     = sum_l omega[l] * F_work_l(a, a0).

This is *not* a learned root adapter and it does not train or modify the root
model.  The frozen root/margin heads only define an observation-derived
integration measure; the transported quantity remains the physical executable
constraint flow.  Because the measure is fixed at the nominal anchor, no
candidate-conditioned option identity can enter the causal representation.
Downstream OC-MERO option selection is unchanged.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from ocrap.algorithms.lcv import EPS, normalize_weights, weighted_lcvar
from ocrap.algorithms.ocmero import sparsify_compatibility
from ocrap.audits.common_option_constraint_work import (
    WORK_BINS,
    WORK_CHANNELS,
    WORK_GEOMETRY_DIM,
    constraint_work_geometry,
    integral_response_geometry,
)
from ocrap.audits.constraint_native_orientation import RAW_CANDIDATE_DIM
from ocrap.audits.executable_constraint_jacobian import ExecutableConstraintField
from ocrap.audits.heterogeneous_constraint_normal_cone import NUM_CONSTRAINTS
from ocrap.audits.recovery_set_constraint_flow import (
    SetFlowScaler,
    base_features,
    contract_checks as rscf_contract_checks,
    fit_set_flow_scaler,
    matched_features,
)

ENGINEERING_VERSION = "v48.116.0-OC-WRCF"
SCIENTIFIC_VERSION = "v48.116-OC-WRCF"
ALGORITHM_NAME = "Observation-Consistent Weak-Root Cotangent Recovery-Set Constraint Flow Audit"

TAIL_GEOMETRY_DIM = WORK_GEOMETRY_DIM
MATCHED_DIM = RAW_CANDIDATE_DIM + TAIL_GEOMETRY_DIM


@dataclass(frozen=True)
class WeakRootTailMeasure:
    option_weights: np.ndarray
    nested_cotangent: np.ndarray
    outer_influence: np.ndarray
    best_option_per_anchor: np.ndarray
    q: np.ndarray
    root_probs: np.ndarray


def _lcvar_influence(scores: np.ndarray, weights: np.ndarray, alpha: float, eps: float = EPS) -> np.ndarray:
    """Exact stable-sort fractional LCVAR subgradient used by OC-MERO."""
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if s.size == 0:
        raise ValueError("WRCF LCVAR influence requires non-empty scores")
    if not (0.0 < float(alpha) <= 1.0):
        raise ValueError(f"WRCF alpha must be in (0,1], got {alpha}")
    w = normalize_weights(weights, eps)
    if w.size != s.size:
        raise ValueError("WRCF scores/weights length mismatch")
    order = np.argsort(s, kind="mergesort")
    remaining = float(alpha)
    out = np.zeros_like(s, dtype=np.float64)
    for idx in order:
        take = min(float(w[idx]), remaining)
        if take > 0.0:
            out[idx] = take / float(alpha)
            remaining -= take
        if remaining <= 1.0e-12:
            break
    if remaining > 1.0e-8:
        # Mirrors weighted_lcvar's numerical fallback.  It should be unreachable
        # for a normalized non-empty support but makes the contract total.
        out[order[-1]] += remaining / float(alpha)
    return out


def _softmax_logits(logits: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64).reshape(-1)
    if valid is None:
        mask = np.ones_like(z, dtype=bool)
    else:
        mask = np.asarray(valid, dtype=bool).reshape(-1)
        if mask.size != z.size:
            raise ValueError("WRCF root-valid/logit size mismatch")
    if not mask.any():
        raise ValueError("WRCF nominal tail measure has no valid roots")
    zz = np.where(mask, z, -np.inf)
    m = float(np.max(zz[mask]))
    e = np.zeros_like(z, dtype=np.float64)
    e[mask] = np.exp(zz[mask] - m)
    return normalize_weights(e)


def nominal_ocmero_tail_measure(
    margins: np.ndarray,
    root_logits: np.ndarray,
    compatibility: np.ndarray,
    *,
    root_valid: np.ndarray | None,
    option_valid: np.ndarray,
    alpha: float,
    beta: float,
    top_m: int | None,
) -> WeakRootTailMeasure:
    """Return the exact nominal nested-tail cotangent and its option push-forward.

    No teacher margin/root field is accepted by this interface.  ``margins``,
    ``root_logits`` and ``compatibility`` are the frozen model's native nominal
    predictions from observable inputs.
    """
    M = np.asarray(margins, dtype=np.float64)
    if M.ndim != 2:
        raise ValueError(f"WRCF margins must be [K,L], got {M.shape}")
    K, L = M.shape
    rv = np.ones(K, dtype=bool) if root_valid is None else np.asarray(root_valid, dtype=bool).reshape(-1)
    ov = np.asarray(option_valid, dtype=bool).reshape(-1)
    if rv.size != K or ov.size != L:
        raise ValueError(f"WRCF validity shape mismatch M={M.shape} rv={rv.shape} ov={ov.shape}")
    if not ov.any():
        raise ValueError("WRCF nominal tail measure has no valid recovery options")
    C = np.asarray(compatibility, dtype=np.float64)
    if C.shape != (K, K):
        raise ValueError(f"WRCF compatibility must be [{K},{K}], got {C.shape}")
    if not np.isfinite(M[rv][:, ov]).all() or not np.isfinite(C).all():
        raise ValueError("WRCF non-finite nominal OC-MERO prediction")

    p = _softmax_logits(root_logits, rv)
    p = np.where(rv, p, 0.0)
    p = normalize_weights(p)
    C_eff = sparsify_compatibility(C, top_m)

    q = np.full((K, L), -1.0e9, dtype=np.float64)
    inner = np.zeros((K, L, K), dtype=np.float64)  # anchor, option, source-root
    for i in range(K):
        w_i = normalize_weights(C_eff[i] * p)
        for l in range(L):
            if not ov[l]:
                continue
            q[i, l] = weighted_lcvar(M[:, l], w_i, beta)
            inner[i, l] = _lcvar_influence(M[:, l], w_i, beta)

    best = np.argmax(q, axis=1).astype(np.int64, copy=False)
    r_anchor = q[np.arange(K), best]
    outer = _lcvar_influence(r_anchor, p, alpha)
    outer = np.where(rv, outer, 0.0)

    G = np.zeros((K, L), dtype=np.float64)
    for i in range(K):
        if outer[i] <= 0.0:
            continue
        l = int(best[i])
        if not ov[l]:
            raise ValueError("WRCF OC-MERO selected an invalid recovery option")
        G[:, l] += float(outer[i]) * inner[i, l]
    G = np.where(rv[:, None] & ov[None, :], G, 0.0)
    if not np.isfinite(G).all() or np.any(G < -1.0e-12):
        raise ValueError("WRCF invalid nested-tail cotangent")
    G = np.maximum(G, 0.0)
    total = float(G.sum())
    if total <= 1.0e-12:
        raise ValueError("WRCF nominal nested-tail cotangent has zero mass")
    # Numerically normalize only the analytic unit-mass subgradient.
    G = G / total
    omega = G.sum(axis=0)
    omega = np.where(ov, omega, 0.0)
    omega = normalize_weights(omega)

    return WeakRootTailMeasure(
        option_weights=omega,
        nested_cotangent=G,
        outer_influence=outer,
        best_option_per_anchor=best,
        q=q,
        root_probs=p,
    )


def _validate_common_measure_contract(
    candidate: ExecutableConstraintField,
    nominal: ExecutableConstraintField,
    option_weights: np.ndarray,
) -> np.ndarray:
    if candidate.full_values.shape != nominal.full_values.shape:
        raise ValueError("WRCF candidate/nominal full-value mismatch")
    if candidate.option_valid.shape != nominal.option_valid.shape:
        raise ValueError("WRCF candidate/nominal option-valid shape mismatch")
    # Candidate-dependent validity renormalization would itself be a hidden
    # treatment-dependent measure.  Fail closed instead: the recovery library
    # measure must be common to the entire scene-time group.
    if not np.array_equal(candidate.option_valid, nominal.option_valid):
        raise ValueError("WRCF requires candidate-invariant recovery option validity")
    w = np.asarray(option_weights, dtype=np.float64).reshape(-1)
    if w.size != len(nominal.option_valid):
        raise ValueError("WRCF option-weight length mismatch")
    if np.any(w < -1.0e-12) or not np.isfinite(w).all():
        raise ValueError("WRCF invalid option weights")
    if np.any((~nominal.option_valid) & (np.abs(w) > 1.0e-12)):
        raise ValueError("WRCF assigns tail mass to invalid option")
    if abs(float(w.sum()) - 1.0) > 1.0e-10:
        raise ValueError(f"WRCF option weights must sum to one, got {w.sum()}")
    return w


def _tail_weighted_geometry(
    candidate: ExecutableConstraintField,
    nominal: ExecutableConstraintField,
    option_weights: np.ndarray,
    *,
    work: bool,
) -> np.ndarray:
    w = _validate_common_measure_contract(candidate, nominal, option_weights)
    fn = constraint_work_geometry if work else integral_response_geometry
    terms = np.stack([fn(candidate, nominal, int(l)) for l in range(len(w))], axis=0).astype(np.float64)
    out = np.sum(w[:, None] * terms, axis=0)
    if out.shape != (TAIL_GEOMETRY_DIM,):
        raise ValueError(f"WRCF geometry dim mismatch {out.shape}")
    if not np.isfinite(out).all():
        raise ValueError("WRCF non-finite tail-weighted geometry")
    return out


def weak_root_integral_geometry(
    candidate: ExecutableConstraintField,
    nominal: ExecutableConstraintField,
    option_weights: np.ndarray,
) -> np.ndarray:
    return _tail_weighted_geometry(candidate, nominal, option_weights, work=False)


def weak_root_work_geometry(
    candidate: ExecutableConstraintField,
    nominal: ExecutableConstraintField,
    option_weights: np.ndarray,
) -> np.ndarray:
    return _tail_weighted_geometry(candidate, nominal, option_weights, work=True)


def weak_root_work_conservation_error(
    candidate: ExecutableConstraintField,
    nominal: ExecutableConstraintField,
    option_weights: np.ndarray,
) -> float:
    integ = weak_root_integral_geometry(candidate, nominal, option_weights).reshape(
        WORK_BINS, NUM_CONSTRAINTS, WORK_CHANNELS
    )
    work = weak_root_work_geometry(candidate, nominal, option_weights).reshape(
        WORK_BINS, NUM_CONSTRAINTS, WORK_CHANNELS
    )
    return float(np.max(np.abs((work[:, :, 0] + work[:, :, 1]) - integ[:, :, 0])))


def weak_root_option_permutation_invariance_error(
    candidate: ExecutableConstraintField,
    nominal: ExecutableConstraintField,
    option_weights: np.ndarray,
) -> float:
    """Check joint invariance to recovery-option ordering and weight permutation."""
    L = len(nominal.option_valid)
    perm = np.arange(L - 1, -1, -1, dtype=np.int64)

    def pfield(f: ExecutableConstraintField) -> ExecutableConstraintField:
        return ExecutableConstraintField(
            values=f.values[perm], masks=f.masks[perm],
            full_values=f.full_values[perm], full_masks=f.full_masks[perm],
            option_valid=f.option_valid[perm], option_scores=f.option_scores[perm],
            option_modes=tuple(f.option_modes[int(i)] for i in perm),
            diagnostics=dict(f.diagnostics),
        )

    w = np.asarray(option_weights, dtype=np.float64).reshape(-1)
    a = weak_root_integral_geometry(candidate, nominal, w)
    b = weak_root_integral_geometry(pfield(candidate), pfield(nominal), w[perm])
    c = weak_root_work_geometry(candidate, nominal, w)
    d = weak_root_work_geometry(pfield(candidate), pfield(nominal), w[perm])
    return float(max(np.max(np.abs(a - b)), np.max(np.abs(c - d))))


def tail_measure_diagnostics(measure: WeakRootTailMeasure) -> dict[str, Any]:
    w = np.asarray(measure.option_weights, dtype=np.float64)
    pos = w > 1.0e-12
    eff = 0.0 if float(np.sum(w * w)) <= 0.0 else float(1.0 / np.sum(w * w))
    entropy = float(-np.sum(w[pos] * np.log(w[pos]))) if pos.any() else 0.0
    return {
        "tail_option_weight_sum": float(w.sum()),
        "tail_positive_option_count": int(pos.sum()),
        "tail_effective_option_count": eff,
        "tail_option_entropy": entropy,
        "tail_outer_positive_root_count": int(np.sum(measure.outer_influence > 1.0e-12)),
        "tail_cotangent_positive_cell_count": int(np.sum(measure.nested_cotangent > 1.0e-12)),
        "tail_cotangent_mass": float(measure.nested_cotangent.sum()),
    }


def contract_checks() -> dict[str, bool]:
    base = rscf_contract_checks()
    K, L = 4, 3
    M = np.asarray([
        [-0.4, 0.1, 0.5],
        [-0.2, 0.0, 0.4],
        [0.3, -0.1, 0.2],
        [0.6, 0.2, -0.3],
    ], dtype=np.float64)
    logits = np.asarray([0.3, -0.2, 0.1, -0.4], dtype=np.float64)
    C = np.asarray([
        [1.0, .8, .2, .1],
        [.8, 1.0, .3, .2],
        [.2, .3, 1.0, .7],
        [.1, .2, .7, 1.0],
    ], dtype=np.float64)
    rv = np.ones(K, dtype=bool)
    ov = np.ones(L, dtype=bool)
    tm = nominal_ocmero_tail_measure(
        M, logits, C, root_valid=rv, option_valid=ov,
        alpha=0.5, beta=0.5, top_m=3,
    )
    d = tail_measure_diagnostics(tm)
    # Verify the NumPy subgradient against a finite directional difference away
    # from ties.  This is a synthetic contract only, never a scientific metric.
    direction = np.asarray([
        [.11, -.07, .02], [.03, .05, -.04], [-.02, .08, .06], [.04, -.03, .09]
    ], dtype=np.float64)
    from ocrap.algorithms.ocmero import oc_mero
    eps = 1.0e-6
    p = _softmax_logits(logits, rv)
    r0 = oc_mero(M, p, C, alpha=.5, beta=.5, option_valid=ov, root_valid=rv, top_m=3).r_dep
    r1 = oc_mero(M + eps * direction, p, C, alpha=.5, beta=.5, option_valid=ov, root_valid=rv, top_m=3).r_dep
    predicted = float(np.sum(tm.nested_cotangent * direction))
    fd = float((r1 - r0) / eps)
    return {
        "rscf_prerequisite_contract": bool(base and all(base.values())),
        "matched_dim_220": MATCHED_DIM == 220,
        "tail_geometry_dim_64": TAIL_GEOMETRY_DIM == 64,
        "nominal_cotangent_nonnegative": bool(np.all(tm.nested_cotangent >= -1.0e-12)),
        "nominal_cotangent_unit_mass": abs(float(tm.nested_cotangent.sum()) - 1.0) <= 1.0e-12,
        "option_pushforward_unit_mass": abs(float(tm.option_weights.sum()) - 1.0) <= 1.0e-12,
        "weak_root_tail_nonempty": int(d["tail_outer_positive_root_count"]) >= 1,
        "tail_option_measure_nonempty": int(d["tail_positive_option_count"]) >= 1,
        "cotangent_directional_derivative_exact": abs(fd - predicted) <= 1.0e-5,
    }


__all__ = [
    "ENGINEERING_VERSION", "SCIENTIFIC_VERSION", "ALGORITHM_NAME",
    "TAIL_GEOMETRY_DIM", "MATCHED_DIM", "WeakRootTailMeasure",
    "nominal_ocmero_tail_measure", "weak_root_integral_geometry",
    "weak_root_work_geometry", "weak_root_work_conservation_error",
    "weak_root_option_permutation_invariance_error",
    "tail_measure_diagnostics", "SetFlowScaler", "fit_set_flow_scaler",
    "base_features", "matched_features", "contract_checks",
]
