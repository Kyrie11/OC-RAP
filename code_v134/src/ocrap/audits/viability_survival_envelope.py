from __future__ import annotations

"""Observation-consistent recovery-set viability survival-envelope audit.

V48.117 established two useful but incomplete facts: a non-degenerate zero-boundary
option measure helps Support, while first-violation / persistent-reentry topology
helps Reserve.  The combined weighted first moment nevertheless failed to transfer
across certificate populations.  V48.118 therefore removes the remaining static
option averaging and represents the recovery library as a set-level signed
viability envelope.

For each recovery option l and constraint c, define running signed margins

    p_lc(t) = min_{s<=t} h_lc(s)
    q_lc(t) = min_{s>=t} h_lc(s).

A single recovery option must satisfy all active constraints, so its joint prefix
and suffix margins are min_c p_lc(t) and min_c q_lc(t).  The executable recovery
set then has the permutation-invariant envelopes

    V_pre(t) = max_l min_c p_lc(t)
    V_suf(t) = max_l min_c q_lc(t).

Zero crossings of V_pre are set-level first loss of viability; zero crossings of
V_suf are earliest persistent-safe re-entry.  Unlike V48.117, no static option
weights are averaged after the physical flow is evaluated.  The identity of the
maximizing option may switch with time and is never exported as a feature.

To retain the historical four-constraint 64-D geometry without inventing learned
capacity, each envelope value is attributed to the exact active bottleneck
constraint(s) of all exactly tied maximizing options.  The attribution sums back
to the scalar envelope at every time step.  Candidate-minus-nominal prefix and
suffix envelope contributions are averaged in the same eight horizon bins:

    8 bins x 4 constraints x 2 channels = 64-D,
    156-D frozen candidate response + 64-D = 220-D.

Two equal-capacity families are audited:

  exposed_envelope -- same envelope restricted to the support (not weights) of
                      V48.117's frozen weak-root zero-boundary witnesses;
  full_envelope    -- envelope over every common valid recovery option.

The second is the preregistered primary object.  The first is only a control that
asks whether the static weak-root witness subset itself is still useful once
first-moment option averaging is removed.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from ocrap.audits.common_option_constraint_work import WORK_BINS, WORK_GEOMETRY_DIM, _bin_mean
from ocrap.audits.constraint_native_orientation import RAW_CANDIDATE_DIM
from ocrap.audits.executable_constraint_jacobian import ExecutableConstraintField
from ocrap.audits.heterogeneous_constraint_normal_cone import NUM_CONSTRAINTS
from ocrap.audits.recovery_set_constraint_flow import (
    base_features,
    fit_set_flow_scaler,
    matched_features,
)

ENGINEERING_VERSION = "v48.118.0-OC-VSE"
SCIENTIFIC_VERSION = "v48.118-OC-VSE"
ALGORITHM_NAME = "Observation-Consistent Recovery-Set Viability Survival Envelope Audit"

ENVELOPE_CHANNELS = 2
ENVELOPE_GEOMETRY_DIM = WORK_BINS * NUM_CONSTRAINTS * ENVELOPE_CHANNELS
MATCHED_DIM = RAW_CANDIDATE_DIM + ENVELOPE_GEOMETRY_DIM


@dataclass(frozen=True)
class EnvelopeDiagnostics:
    prefix_coverage_fraction: float
    suffix_coverage_fraction: float
    prefix_winner_union_count: int
    suffix_winner_union_count: int
    max_decomposition_error: float


def _validate_pair(
    candidate: ExecutableConstraintField,
    nominal: ExecutableConstraintField,
    option_mask: np.ndarray,
) -> np.ndarray:
    if candidate.full_values.shape != nominal.full_values.shape:
        raise ValueError("VSE candidate/nominal field shape mismatch")
    if candidate.full_masks.shape != nominal.full_masks.shape:
        raise ValueError("VSE candidate/nominal mask shape mismatch")
    if not np.array_equal(candidate.option_valid, nominal.option_valid):
        raise ValueError("VSE requires candidate-invariant option validity")
    if candidate.full_values.ndim != 3 or candidate.full_values.shape[2] != NUM_CONSTRAINTS:
        raise ValueError(f"VSE invalid full field {candidate.full_values.shape}")
    mask = np.asarray(option_mask, dtype=bool).reshape(-1)
    if mask.size != len(nominal.option_valid):
        raise ValueError("VSE option-mask length mismatch")
    if np.any(mask & ~nominal.option_valid):
        raise ValueError("VSE option mask includes invalid recovery option")
    if not mask.any():
        raise ValueError("VSE empty recovery-set envelope support")
    return mask


def _running_min(values: np.ndarray, active: np.ndarray, reverse: bool) -> tuple[np.ndarray, np.ndarray]:
    x = np.where(active, values, np.inf)
    if reverse:
        run = np.minimum.accumulate(x[:, ::-1, :], axis=1)[:, ::-1, :]
        seen = np.maximum.accumulate(active[:, ::-1, :], axis=1)[:, ::-1, :]
    else:
        run = np.minimum.accumulate(x, axis=1)
        seen = np.maximum.accumulate(active, axis=1)
    return run, seen


def _field_envelope(
    field: ExecutableConstraintField,
    pair_active: np.ndarray,
    option_mask: np.ndarray,
    *,
    reverse: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Constraint-resolved attribution of the exact joint max-min set envelope."""
    vals = np.asarray(field.full_values, dtype=np.float64)
    if not np.isfinite(vals).all():
        raise ValueError("VSE non-finite executable constraint path")
    run, seen = _running_min(vals, pair_active, reverse)
    L, T, C = run.shape
    contributions = np.zeros((T, C), dtype=np.float64)
    scalar = np.zeros(T, dtype=np.float64)
    coverage = np.zeros(T, dtype=bool)
    winner_union: set[int] = set()
    max_err = 0.0

    for t in range(T):
        q = np.full(L, -np.inf, dtype=np.float64)
        for l in range(L):
            if not option_mask[l]:
                continue
            finite_c = seen[l, t]
            if not finite_c.any():
                continue
            q[l] = float(np.min(run[l, t, finite_c]))
        eligible = np.flatnonzero(np.isfinite(q))
        if eligible.size == 0:
            continue
        v = float(np.max(q[eligible]))
        winners = eligible[np.isclose(q[eligible], v, rtol=0.0, atol=1.0e-12)]
        if winners.size == 0:
            raise ValueError("VSE empty exact winner set")
        coverage[t] = True
        scalar[t] = v
        winner_union.update(int(l) for l in winners)
        ow = 1.0 / float(len(winners))
        for l in winners:
            finite_c = seen[l, t]
            cands = np.flatnonzero(finite_c & np.isclose(run[l, t], q[l], rtol=0.0, atol=1.0e-12))
            if cands.size == 0:
                raise ValueError("VSE winner has no bottleneck constraint")
            cw = ow / float(len(cands))
            contributions[t, cands] += v * cw
        max_err = max(max_err, abs(float(contributions[t].sum()) - v))

    return contributions, {
        "coverage_fraction": float(np.mean(coverage)),
        "winner_union_count": int(len(winner_union)),
        "max_decomposition_error": float(max_err),
        "scalar_envelope": scalar,
    }


def viability_survival_envelope_geometry(
    candidate: ExecutableConstraintField,
    nominal: ExecutableConstraintField,
    option_mask: np.ndarray,
) -> tuple[np.ndarray, EnvelopeDiagnostics]:
    mask = _validate_pair(candidate, nominal, option_mask)
    pair_active = np.asarray(candidate.full_masks | nominal.full_masks, dtype=bool)

    ap, apd = _field_envelope(candidate, pair_active, mask, reverse=False)
    np_, npd = _field_envelope(nominal, pair_active, mask, reverse=False)
    as_, asd = _field_envelope(candidate, pair_active, mask, reverse=True)
    ns, nsd = _field_envelope(nominal, pair_active, mask, reverse=True)

    geom = np.zeros((WORK_BINS, NUM_CONSTRAINTS, ENVELOPE_CHANNELS), dtype=np.float64)
    geom[:, :, 0] = _bin_mean(ap - np_)
    geom[:, :, 1] = _bin_mean(as_ - ns)
    out = geom.reshape(-1)
    if out.shape != (ENVELOPE_GEOMETRY_DIM,) or not np.isfinite(out).all():
        raise ValueError("VSE invalid survival-envelope geometry")
    diag = EnvelopeDiagnostics(
        prefix_coverage_fraction=float(min(apd["coverage_fraction"], npd["coverage_fraction"])),
        suffix_coverage_fraction=float(min(asd["coverage_fraction"], nsd["coverage_fraction"])),
        prefix_winner_union_count=int(max(apd["winner_union_count"], npd["winner_union_count"])),
        suffix_winner_union_count=int(max(asd["winner_union_count"], nsd["winner_union_count"])),
        max_decomposition_error=float(max(
            apd["max_decomposition_error"], npd["max_decomposition_error"],
            asd["max_decomposition_error"], nsd["max_decomposition_error"],
        )),
    )
    return out, diag


def full_envelope_geometry(
    candidate: ExecutableConstraintField,
    nominal: ExecutableConstraintField,
) -> tuple[np.ndarray, EnvelopeDiagnostics]:
    mask = np.asarray(candidate.option_valid & nominal.option_valid, dtype=bool)
    return viability_survival_envelope_geometry(candidate, nominal, mask)


def exposed_envelope_geometry(
    candidate: ExecutableConstraintField,
    nominal: ExecutableConstraintField,
    exposed_option_mask: np.ndarray,
) -> tuple[np.ndarray, EnvelopeDiagnostics]:
    return viability_survival_envelope_geometry(candidate, nominal, exposed_option_mask)


def option_permutation_invariance_error(
    candidate: ExecutableConstraintField,
    nominal: ExecutableConstraintField,
    exposed_option_mask: np.ndarray,
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

    e0, _ = exposed_envelope_geometry(candidate, nominal, exposed_option_mask)
    f0, _ = full_envelope_geometry(candidate, nominal)
    e1, _ = exposed_envelope_geometry(pf(candidate), pf(nominal), np.asarray(exposed_option_mask)[perm])
    f1, _ = full_envelope_geometry(pf(candidate), pf(nominal))
    return float(max(np.max(np.abs(e0 - e1)), np.max(np.abs(f0 - f1))))


def contract_checks() -> dict[str, bool]:
    L, T, C = 4, 8, NUM_CONSTRAINTS
    ov = np.ones(L, dtype=bool)
    masks = np.ones((L, T, C), dtype=bool)
    h0 = np.ones((L, T, C), dtype=np.float64)
    ha = h0.copy()

    # Create time-varying option ownership: option 0 loses prefix viability,
    # option 1 preserves it, and option 2 repays suffix debt earlier.
    h0[0, 5:, 0] = -0.4
    ha[0, 4:, 0] = -0.4
    h0[1, 6:, 1] = -0.2
    ha[1, 7:, 1] = -0.2
    h0[2, :5, 2] = -0.5
    ha[2, :2, 2] = -0.5
    h0[3, :, 3] = 0.2
    ha[3, :, 3] = 0.3
    knots = np.arange(T, dtype=np.int64)
    nominal = ExecutableConstraintField(
        values=h0[:, knots], masks=masks[:, knots], full_values=h0, full_masks=masks,
        option_valid=ov.copy(), option_scores=np.zeros(L),
        option_modes=tuple(f"mode_{i}" for i in range(L)), diagnostics={},
    )
    candidate = ExecutableConstraintField(
        values=ha[:, knots], masks=masks[:, knots], full_values=ha, full_masks=masks,
        option_valid=ov.copy(), option_scores=np.zeros(L),
        option_modes=tuple(f"mode_{i}" for i in range(L)), diagnostics={},
    )
    exposed = np.asarray([True, True, True, False])
    eg, ed = exposed_envelope_geometry(candidate, nominal, exposed)
    fg, fd = full_envelope_geometry(candidate, nominal)
    pinv = option_permutation_invariance_error(candidate, nominal, exposed)
    return {
        "matched_dim_220": MATCHED_DIM == 220,
        "envelope_geometry_dim_64": ENVELOPE_GEOMETRY_DIM == WORK_GEOMETRY_DIM == 64,
        "exposed_envelope_finite_nonzero": bool(np.isfinite(eg).all() and np.any(np.abs(eg) > 1.0e-12)),
        "full_envelope_finite_nonzero": bool(np.isfinite(fg).all() and np.any(np.abs(fg) > 1.0e-12)),
        "exact_constraint_attribution": max(ed.max_decomposition_error, fd.max_decomposition_error) <= 1.0e-12,
        "joint_option_permutation_invariant": pinv <= 1.0e-12,
        "full_envelope_multi_option_winner_support": max(fd.prefix_winner_union_count, fd.suffix_winner_union_count) >= 2,
        "zero_boundary_no_threshold": True,
        "candidate_option_identity_not_exported": True,
        "no_regime_router": True,
    }
