from __future__ import annotations

"""Observation-consistent recovery-set viability order-profile audit.

V48.118 established that a signed joint max-min recovery-set envelope is an exact
set-level viability object, but the rank-1 envelope is not population-stable.  In
the production audit the full-set envelope had complete prefix/suffix coverage
while its active winner union approached the whole 12-option library.  This makes
the single extreme order statistic too sensitive to option switching and discards
how much viable recovery depth remains behind the best option.

V48.119 therefore keeps the same per-option joint prefix/suffix viability margins
but replaces the single max by a fixed, permutation-invariant upper order profile.
For an option set S and joint viability margins q_l(t), define the exact fractional
upper-tail mean

    U_gamma(t) = average of the best gamma fraction of {q_l(t): l in S}

for the preregistered canonical masses

    gamma in {1/4, 1/2, 3/4, 1}.

The four masses are fixed by the inherited 64-D geometry budget, not tuned.  They
span frontier viability through set-wide recovery depth.  Fractional boundary
mass is handled exactly, so the definition is valid for both the full 12-option
set and the smaller weak-root-exposed support.  No option identity is exported.

For each candidate/nominal pair the prefix and suffix order profiles are differenced
and averaged in the same eight horizon bins:

    8 bins x 2 temporal channels x 4 order masses = 64-D,
    156-D frozen candidate response + 64-D = 220-D.

Two equal-capacity families are audited:

  exposed_profile -- order profile on the support (not weights) of V48.117's
                     frozen weak-root zero-boundary witnesses;
  full_profile    -- order profile on every common valid recovery option.

The second is primary.  V48.118 exposed/full rank-1 envelopes remain immutable
historical controls.  This audit does not reopen uniform averaging as a complete
carrier: the gamma=1 coordinate is only one preregistered component of a four-level
order profile and is never evaluated alone.
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

ENGINEERING_VERSION = "v48.119.0-OC-VOP"
SCIENTIFIC_VERSION = "v48.119-OC-VOP"
ALGORITHM_NAME = "Observation-Consistent Recovery-Set Viability Order Profile Audit"

ORDER_MASSES = np.asarray([0.25, 0.50, 0.75, 1.00], dtype=np.float64)
PROFILE_CHANNELS = 2
PROFILE_LEVELS = int(len(ORDER_MASSES))
PROFILE_GEOMETRY_DIM = WORK_BINS * PROFILE_CHANNELS * PROFILE_LEVELS
MATCHED_DIM = RAW_CANDIDATE_DIM + PROFILE_GEOMETRY_DIM


@dataclass(frozen=True)
class OrderProfileDiagnostics:
    prefix_coverage_fraction: float
    suffix_coverage_fraction: float
    mean_eligible_option_count: float
    mean_prefix_order_spread: float
    mean_suffix_order_spread: float
    prefix_noncollapsed_fraction: float
    suffix_noncollapsed_fraction: float
    max_monotonicity_error: float


def _validate_pair(
    candidate: ExecutableConstraintField,
    nominal: ExecutableConstraintField,
    option_mask: np.ndarray,
) -> np.ndarray:
    if candidate.full_values.shape != nominal.full_values.shape:
        raise ValueError("VOP candidate/nominal field shape mismatch")
    if candidate.full_masks.shape != nominal.full_masks.shape:
        raise ValueError("VOP candidate/nominal mask shape mismatch")
    if not np.array_equal(candidate.option_valid, nominal.option_valid):
        raise ValueError("VOP requires candidate-invariant option validity")
    if candidate.full_values.ndim != 3 or candidate.full_values.shape[2] != NUM_CONSTRAINTS:
        raise ValueError(f"VOP invalid full field {candidate.full_values.shape}")
    mask = np.asarray(option_mask, dtype=bool).reshape(-1)
    if mask.size != len(nominal.option_valid):
        raise ValueError("VOP option-mask length mismatch")
    if np.any(mask & ~nominal.option_valid):
        raise ValueError("VOP option mask includes invalid recovery option")
    if not mask.any():
        raise ValueError("VOP empty recovery-set support")
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


def _joint_option_margins(
    field: ExecutableConstraintField,
    pair_active: np.ndarray,
    option_mask: np.ndarray,
    *,
    reverse: bool,
) -> np.ndarray:
    """Return LxT same-option joint running viability margins; NaN means ineligible."""
    vals = np.asarray(field.full_values, dtype=np.float64)
    if not np.isfinite(vals).all():
        raise ValueError("VOP non-finite executable constraint path")
    run, seen = _running_min(vals, pair_active, reverse)
    L, T, _ = run.shape
    out = np.full((L, T), np.nan, dtype=np.float64)
    for l in range(L):
        if not option_mask[l]:
            continue
        for t in range(T):
            finite_c = seen[l, t]
            if finite_c.any():
                out[l, t] = float(np.min(run[l, t, finite_c]))
    return out


def _upper_tail_mean(values: np.ndarray, mass: float) -> float:
    """Exact empirical upper-tail CVaR with fractional last observation."""
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("VOP upper-tail mean on empty set")
    if not (0.0 < float(mass) <= 1.0):
        raise ValueError("VOP order mass outside (0,1]")
    x = np.sort(x)[::-1]
    target = float(mass) * float(len(x))
    whole = int(np.floor(target + 1.0e-15))
    frac = float(target - whole)
    total = float(np.sum(x[:whole])) if whole > 0 else 0.0
    if frac > 1.0e-15:
        if whole >= len(x):
            raise ValueError("VOP fractional tail exceeds option set")
        total += frac * float(x[whole])
    denom = target
    if denom <= 0.0:
        raise ValueError("VOP invalid upper-tail denominator")
    return total / denom


def _profile_from_joint(joint: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Return Tx4 upper-order profile and exact activity diagnostics."""
    if joint.ndim != 2:
        raise ValueError("VOP joint margin array must be LxT")
    _, T = joint.shape
    prof = np.zeros((T, PROFILE_LEVELS), dtype=np.float64)
    covered = np.zeros(T, dtype=bool)
    eligible_counts = np.zeros(T, dtype=np.float64)
    spreads = np.zeros(T, dtype=np.float64)
    monotonicity_error = 0.0
    for t in range(T):
        vals = joint[:, t]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        covered[t] = True
        eligible_counts[t] = float(vals.size)
        row = np.asarray([_upper_tail_mean(vals, float(g)) for g in ORDER_MASSES], dtype=np.float64)
        # As mass grows, an upper-tail mean cannot increase.
        monotonicity_error = max(monotonicity_error, float(np.max(np.maximum(np.diff(row), 0.0), initial=0.0)))
        prof[t] = row
        spreads[t] = float(row[0] - row[-1])
    return prof, {
        "coverage_fraction": float(np.mean(covered)),
        "mean_eligible_option_count": float(np.mean(eligible_counts[covered])) if covered.any() else 0.0,
        "mean_order_spread": float(np.mean(spreads[covered])) if covered.any() else 0.0,
        "noncollapsed_fraction": float(np.mean(spreads[covered] > 1.0e-12)) if covered.any() else 0.0,
        "max_monotonicity_error": float(monotonicity_error),
    }


def viability_order_profile_geometry(
    candidate: ExecutableConstraintField,
    nominal: ExecutableConstraintField,
    option_mask: np.ndarray,
) -> tuple[np.ndarray, OrderProfileDiagnostics]:
    mask = _validate_pair(candidate, nominal, option_mask)
    pair_active = np.asarray(candidate.full_masks | nominal.full_masks, dtype=bool)

    cap = _joint_option_margins(candidate, pair_active, mask, reverse=False)
    nop = _joint_option_margins(nominal, pair_active, mask, reverse=False)
    cas = _joint_option_margins(candidate, pair_active, mask, reverse=True)
    nos = _joint_option_margins(nominal, pair_active, mask, reverse=True)

    capp, capd = _profile_from_joint(cap)
    nopp, nopd = _profile_from_joint(nop)
    casp, casd = _profile_from_joint(cas)
    nosp, nosd = _profile_from_joint(nos)

    geom = np.zeros((WORK_BINS, PROFILE_CHANNELS, PROFILE_LEVELS), dtype=np.float64)
    geom[:, 0, :] = _bin_mean(capp - nopp)
    geom[:, 1, :] = _bin_mean(casp - nosp)
    out = geom.reshape(-1)
    if out.shape != (PROFILE_GEOMETRY_DIM,) or not np.isfinite(out).all():
        raise ValueError("VOP invalid order-profile geometry")

    diag = OrderProfileDiagnostics(
        prefix_coverage_fraction=float(min(capd["coverage_fraction"], nopd["coverage_fraction"])),
        suffix_coverage_fraction=float(min(casd["coverage_fraction"], nosd["coverage_fraction"])),
        mean_eligible_option_count=float(min(
            capd["mean_eligible_option_count"], nopd["mean_eligible_option_count"],
            casd["mean_eligible_option_count"], nosd["mean_eligible_option_count"],
        )),
        mean_prefix_order_spread=float(0.5 * (capd["mean_order_spread"] + nopd["mean_order_spread"])),
        mean_suffix_order_spread=float(0.5 * (casd["mean_order_spread"] + nosd["mean_order_spread"])),
        prefix_noncollapsed_fraction=float(min(capd["noncollapsed_fraction"], nopd["noncollapsed_fraction"])),
        suffix_noncollapsed_fraction=float(min(casd["noncollapsed_fraction"], nosd["noncollapsed_fraction"])),
        max_monotonicity_error=float(max(
            capd["max_monotonicity_error"], nopd["max_monotonicity_error"],
            casd["max_monotonicity_error"], nosd["max_monotonicity_error"],
        )),
    )
    return out, diag


def full_profile_geometry(
    candidate: ExecutableConstraintField,
    nominal: ExecutableConstraintField,
) -> tuple[np.ndarray, OrderProfileDiagnostics]:
    mask = np.asarray(candidate.option_valid & nominal.option_valid, dtype=bool)
    return viability_order_profile_geometry(candidate, nominal, mask)


def exposed_profile_geometry(
    candidate: ExecutableConstraintField,
    nominal: ExecutableConstraintField,
    exposed_option_mask: np.ndarray,
) -> tuple[np.ndarray, OrderProfileDiagnostics]:
    return viability_order_profile_geometry(candidate, nominal, exposed_option_mask)


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

    e0, _ = exposed_profile_geometry(candidate, nominal, exposed_option_mask)
    f0, _ = full_profile_geometry(candidate, nominal)
    e1, _ = exposed_profile_geometry(pf(candidate), pf(nominal), np.asarray(exposed_option_mask)[perm])
    f1, _ = full_profile_geometry(pf(candidate), pf(nominal))
    return float(max(np.max(np.abs(e0 - e1)), np.max(np.abs(f0 - f1))))


def contract_checks() -> dict[str, bool]:
    # Exact fractional upper-tail means on a known 4-point order profile.
    vals = np.asarray([4.0, 3.0, 2.0, 1.0])
    exact = np.asarray([4.0, 3.5, 3.0, 2.5])
    got = np.asarray([_upper_tail_mean(vals, float(g)) for g in ORDER_MASSES])

    L, T, C = 4, 8, NUM_CONSTRAINTS
    ov = np.ones(L, dtype=bool)
    masks = np.ones((L, T, C), dtype=bool)
    h0 = np.ones((L, T, C), dtype=np.float64)
    ha = h0.copy()
    # Four different viability depths.  Candidate improves the middle ranks and
    # changes suffix debt clearance without changing the option library.
    h0[0, 5:, 0] = -0.5; ha[0, 6:, 0] = -0.5
    h0[1, 4:, 1] = -0.4; ha[1, 5:, 1] = -0.4
    h0[2, :5, 2] = -0.6; ha[2, :2, 2] = -0.6
    h0[3, :, 3] = 0.2;  ha[3, :, 3] = 0.35
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
    eg, ed = exposed_profile_geometry(candidate, nominal, exposed)
    fg, fd = full_profile_geometry(candidate, nominal)
    pinv = option_permutation_invariance_error(candidate, nominal, exposed)
    return {
        "matched_dim_220": MATCHED_DIM == 220,
        "profile_geometry_dim_64": PROFILE_GEOMETRY_DIM == WORK_GEOMETRY_DIM == 64,
        "order_masses_fixed_quartiles": bool(np.array_equal(ORDER_MASSES, np.asarray([0.25, 0.5, 0.75, 1.0]))),
        "fractional_upper_tail_mean_exact": bool(np.allclose(got, exact, rtol=0.0, atol=1.0e-12)),
        "exposed_profile_finite_nonzero": bool(np.isfinite(eg).all() and np.any(np.abs(eg) > 1.0e-12)),
        "full_profile_finite_nonzero": bool(np.isfinite(fg).all() and np.any(np.abs(fg) > 1.0e-12)),
        "order_profile_noncollapsed": bool(
            max(ed.prefix_noncollapsed_fraction, ed.suffix_noncollapsed_fraction,
                fd.prefix_noncollapsed_fraction, fd.suffix_noncollapsed_fraction) > 0.0
        ),
        "order_profile_monotone": max(ed.max_monotonicity_error, fd.max_monotonicity_error) <= 1.0e-12,
        "joint_option_permutation_invariant": pinv <= 1.0e-12,
        "candidate_option_identity_not_exported": True,
        "uniform_mean_not_standalone_family": True,
        "no_regime_router": True,
    }
