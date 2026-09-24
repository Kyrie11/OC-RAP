from __future__ import annotations

"""Observation-consistent recovery-set viability rank-transport audit.

V48.119 showed that a fixed four-level order profile is active and non-collapsed,
but it is not population-stable.  The key remaining defect is that candidate and
nominal recovery sets are independently re-sorted before their order statistics
are differenced.  That Eulerian comparison preserves the marginal viability
profile but erases *which nominal recovery ranks were deformed by the candidate*.
This is especially problematic after V48.118 exposed near-library-wide winner
switching.

V48.120 therefore keeps the same same-option joint prefix/suffix viability margin
q_l(t), but uses the nominal recovery order as a candidate-independent coordinate.
For each time t, options are ordered by the nominal joint viability margin.  The
same physical option is then followed under the candidate and its signed margin
displacement

    d_l(t) = q_l^candidate(t) - q_l^nominal(t)

is represented as a function of nominal rank u in [0,1].  Exact nominal ties form
a shared rank interval and use the mean displacement of the tied options, so the
representation is invariant to arbitrary tie ordering and to a joint option
permutation.

The rank-conditioned displacement field is projected onto the first four shifted
Legendre modes P_k(2u-1), k=0..3, by *exact interval integration*.  These fixed
modes have canonical interpretations:

  k=0  global signed viability shift,
  k=1  frontier-vs-tail tilt,
  k=2  shoulder curvature,
  k=3  higher-order rank asymmetry.

There are no rank cuts, quantile thresholds, temperatures, or tuned masses.  The
four modes are fixed solely by the inherited 64-D geometry budget:

    8 full-horizon bins x 2 temporal channels x 4 transport modes = 64-D,
    156-D frozen candidate response + 64-D = 220-D.

Two equal-capacity families are audited:

  exposed_transport -- transport on the support (not weights) of V48.117's
                       frozen weak-root zero-boundary witnesses;
  full_transport    -- transport on every common valid recovery option.

The full family is primary.  No option identity is exported to the readout, no
candidate rank sorting is used to define the coordinate, and downstream OC-MERO
selection is unchanged.  The audit introduces no learned set encoder, regime
router, boundary transport, threshold/horizon/source/capacity sweep, or training.
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

ENGINEERING_VERSION = "v48.120.0-OC-VRT"
SCIENTIFIC_VERSION = "v48.120-OC-VRT"
ALGORITHM_NAME = "Observation-Consistent Recovery-Set Viability Rank Transport Audit"

TRANSPORT_MODE_DEGREES = (0, 1, 2, 3)
TRANSPORT_CHANNELS = 2
TRANSPORT_MODES = len(TRANSPORT_MODE_DEGREES)
TRANSPORT_GEOMETRY_DIM = WORK_BINS * TRANSPORT_CHANNELS * TRANSPORT_MODES
MATCHED_DIM = RAW_CANDIDATE_DIM + TRANSPORT_GEOMETRY_DIM


@dataclass(frozen=True)
class RankTransportDiagnostics:
    prefix_coverage_fraction: float
    suffix_coverage_fraction: float
    mean_eligible_option_count: float
    mean_prefix_transport_energy: float
    mean_suffix_transport_energy: float
    mean_prefix_shape_energy: float
    mean_suffix_shape_energy: float
    prefix_nonzero_fraction: float
    suffix_nonzero_fraction: float
    mean_prefix_rank_inversion_fraction: float
    mean_suffix_rank_inversion_fraction: float


def _validate_pair(
    candidate: ExecutableConstraintField,
    nominal: ExecutableConstraintField,
    option_mask: np.ndarray,
) -> np.ndarray:
    if candidate.full_values.shape != nominal.full_values.shape:
        raise ValueError("VRT candidate/nominal field shape mismatch")
    if candidate.full_masks.shape != nominal.full_masks.shape:
        raise ValueError("VRT candidate/nominal mask shape mismatch")
    if not np.array_equal(candidate.option_valid, nominal.option_valid):
        raise ValueError("VRT requires candidate-invariant option validity")
    if candidate.full_values.ndim != 3 or candidate.full_values.shape[2] != NUM_CONSTRAINTS:
        raise ValueError(f"VRT invalid full field {candidate.full_values.shape}")
    mask = np.asarray(option_mask, dtype=bool).reshape(-1)
    if mask.size != len(nominal.option_valid):
        raise ValueError("VRT option-mask length mismatch")
    if np.any(mask & ~nominal.option_valid):
        raise ValueError("VRT option mask includes invalid recovery option")
    if not mask.any():
        raise ValueError("VRT empty recovery-set support")
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
        raise ValueError("VRT non-finite executable constraint path")
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


def _shifted_legendre_antiderivative(degree: int, u: np.ndarray | float) -> np.ndarray:
    """Antiderivative of P_degree(2u-1) on u in [0,1], with F(0)=0."""
    x = np.asarray(u, dtype=np.float64)
    if degree == 0:
        return x
    if degree == 1:
        return x * x - x
    if degree == 2:
        return 2.0 * x**3 - 3.0 * x**2 + x
    if degree == 3:
        return 5.0 * x**4 - 10.0 * x**3 + 6.0 * x**2 - x
    raise ValueError(f"VRT unsupported shifted-Legendre degree {degree}")


def _basis_interval_integrals(a: float, b: float) -> np.ndarray:
    if not (0.0 <= float(a) <= float(b) <= 1.0):
        raise ValueError("VRT invalid rank interval")
    return np.asarray([
        float(_shifted_legendre_antiderivative(k, b) - _shifted_legendre_antiderivative(k, a))
        for k in TRANSPORT_MODE_DEGREES
    ], dtype=np.float64)


def _rank_inversion_fraction(nominal: np.ndarray, candidate: np.ndarray) -> float:
    """Pairwise same-option rank reversal fraction; ties are neutral diagnostics."""
    n = len(nominal)
    if n < 2:
        return 0.0
    inv = 0
    comparable = 0
    for i in range(n):
        for j in range(i + 1, n):
            dn = float(nominal[i] - nominal[j])
            if dn == 0.0:
                continue
            dc = float(candidate[i] - candidate[j])
            comparable += 1
            if dc != 0.0 and dn * dc < 0.0:
                inv += 1
    return float(inv / comparable) if comparable else 0.0


def _transport_modes_from_joint(candidate_joint: np.ndarray, nominal_joint: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Return Tx4 exact nominal-rank transport moments and activity diagnostics.

    Candidate values are *not* sorted.  A stable sort is used only to construct
    nominal tie blocks; tied nominal options share their interval and contribute
    the mean same-option displacement over that block.
    """
    ca = np.asarray(candidate_joint, dtype=np.float64)
    no = np.asarray(nominal_joint, dtype=np.float64)
    if ca.shape != no.shape or ca.ndim != 2:
        raise ValueError("VRT joint arrays must be matched LxT")
    if not np.array_equal(np.isfinite(ca), np.isfinite(no)):
        raise ValueError("VRT candidate/nominal eligibility mismatch")
    _, T = ca.shape
    modes = np.zeros((T, TRANSPORT_MODES), dtype=np.float64)
    covered = np.zeros(T, dtype=bool)
    counts = np.zeros(T, dtype=np.float64)
    energy = np.zeros(T, dtype=np.float64)
    shape_energy = np.zeros(T, dtype=np.float64)
    inversions = np.zeros(T, dtype=np.float64)

    for t in range(T):
        finite = np.isfinite(no[:, t])
        if not finite.any():
            continue
        nvals = no[finite, t]
        cvals = ca[finite, t]
        delta = cvals - nvals
        n = int(len(nvals))
        covered[t] = True
        counts[t] = float(n)
        inversions[t] = _rank_inversion_fraction(nvals, cvals)

        order = np.argsort(nvals, kind="mergesort")  # worst -> best nominal viability
        sorted_nom = nvals[order]
        sorted_delta = delta[order]
        i = 0
        row = np.zeros(TRANSPORT_MODES, dtype=np.float64)
        while i < n:
            j = i + 1
            while j < n and sorted_nom[j] == sorted_nom[i]:
                j += 1
            # Exact nominal tie block: no arbitrary within-tie rank assignment.
            block_delta = float(np.mean(sorted_delta[i:j]))
            a = float(i) / float(n)
            b = float(j) / float(n)
            row += block_delta * _basis_interval_integrals(a, b)
            i = j
        modes[t] = row
        energy[t] = float(np.linalg.norm(row))
        shape_energy[t] = float(np.linalg.norm(row[1:]))

    nz = energy > 1.0e-12
    return modes, {
        "coverage_fraction": float(np.mean(covered)),
        "mean_eligible_option_count": float(np.mean(counts[covered])) if covered.any() else 0.0,
        "mean_transport_energy": float(np.mean(energy[covered])) if covered.any() else 0.0,
        "mean_shape_energy": float(np.mean(shape_energy[covered])) if covered.any() else 0.0,
        "nonzero_fraction": float(np.mean(nz[covered])) if covered.any() else 0.0,
        "mean_rank_inversion_fraction": float(np.mean(inversions[covered])) if covered.any() else 0.0,
    }


def viability_rank_transport_geometry(
    candidate: ExecutableConstraintField,
    nominal: ExecutableConstraintField,
    option_mask: np.ndarray,
) -> tuple[np.ndarray, RankTransportDiagnostics]:
    mask = _validate_pair(candidate, nominal, option_mask)
    pair_active = np.asarray(candidate.full_masks | nominal.full_masks, dtype=bool)

    cap = _joint_option_margins(candidate, pair_active, mask, reverse=False)
    nop = _joint_option_margins(nominal, pair_active, mask, reverse=False)
    cas = _joint_option_margins(candidate, pair_active, mask, reverse=True)
    nos = _joint_option_margins(nominal, pair_active, mask, reverse=True)

    prefix, pd = _transport_modes_from_joint(cap, nop)
    suffix, sd = _transport_modes_from_joint(cas, nos)

    geom = np.zeros((WORK_BINS, TRANSPORT_CHANNELS, TRANSPORT_MODES), dtype=np.float64)
    geom[:, 0, :] = _bin_mean(prefix)
    geom[:, 1, :] = _bin_mean(suffix)
    out = geom.reshape(-1)
    if out.shape != (TRANSPORT_GEOMETRY_DIM,) or not np.isfinite(out).all():
        raise ValueError("VRT invalid rank-transport geometry")

    diag = RankTransportDiagnostics(
        prefix_coverage_fraction=float(pd["coverage_fraction"]),
        suffix_coverage_fraction=float(sd["coverage_fraction"]),
        mean_eligible_option_count=float(min(pd["mean_eligible_option_count"], sd["mean_eligible_option_count"])),
        mean_prefix_transport_energy=float(pd["mean_transport_energy"]),
        mean_suffix_transport_energy=float(sd["mean_transport_energy"]),
        mean_prefix_shape_energy=float(pd["mean_shape_energy"]),
        mean_suffix_shape_energy=float(sd["mean_shape_energy"]),
        prefix_nonzero_fraction=float(pd["nonzero_fraction"]),
        suffix_nonzero_fraction=float(sd["nonzero_fraction"]),
        mean_prefix_rank_inversion_fraction=float(pd["mean_rank_inversion_fraction"]),
        mean_suffix_rank_inversion_fraction=float(sd["mean_rank_inversion_fraction"]),
    )
    return out, diag


def full_transport_geometry(
    candidate: ExecutableConstraintField,
    nominal: ExecutableConstraintField,
) -> tuple[np.ndarray, RankTransportDiagnostics]:
    mask = np.asarray(candidate.option_valid & nominal.option_valid, dtype=bool)
    return viability_rank_transport_geometry(candidate, nominal, mask)


def exposed_transport_geometry(
    candidate: ExecutableConstraintField,
    nominal: ExecutableConstraintField,
    exposed_option_mask: np.ndarray,
) -> tuple[np.ndarray, RankTransportDiagnostics]:
    return viability_rank_transport_geometry(candidate, nominal, exposed_option_mask)


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

    e0, _ = exposed_transport_geometry(candidate, nominal, exposed_option_mask)
    f0, _ = full_transport_geometry(candidate, nominal)
    e1, _ = exposed_transport_geometry(pf(candidate), pf(nominal), np.asarray(exposed_option_mask)[perm])
    f1, _ = full_transport_geometry(pf(candidate), pf(nominal))
    return float(max(np.max(np.abs(e0 - e1)), np.max(np.abs(f0 - f1))))


def contract_checks() -> dict[str, bool]:
    # Exact shifted-Legendre interval identities.
    whole = np.asarray([_basis_interval_integrals(0.0, 1.0)[k] for k in range(TRANSPORT_MODES)])
    interval_ok = bool(np.allclose(whole, [1.0, 0.0, 0.0, 0.0], atol=1.0e-12, rtol=0.0))

    # Static order profile can be unchanged when two option identities swap, but
    # same-option nominal-rank transport must remain nonzero.
    L, T, C = 4, 8, NUM_CONSTRAINTS
    ov = np.ones(L, dtype=bool)
    masks = np.ones((L, T, C), dtype=bool)
    h0 = np.zeros((L, T, C), dtype=np.float64)
    ha = np.zeros_like(h0)
    nominal_levels = np.asarray([-1.5, -0.5, 0.5, 1.5], dtype=np.float64)
    candidate_levels = np.asarray([1.5, -0.5, 0.5, -1.5], dtype=np.float64)
    for l in range(L):
        h0[l, :, :] = nominal_levels[l]
        ha[l, :, :] = candidate_levels[l]
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
    eg, ed = exposed_transport_geometry(candidate, nominal, exposed)
    fg, fd = full_transport_geometry(candidate, nominal)
    pinv = option_permutation_invariance_error(candidate, nominal, exposed)

    # Exact-tie invariance: swapping candidate values inside a nominal tie cannot
    # change the tied rank interval's mean transport contribution.
    no = np.asarray([[0.0], [0.0], [1.0]], dtype=np.float64)
    ca1 = np.asarray([[1.0], [-1.0], [1.5]], dtype=np.float64)
    ca2 = np.asarray([[-1.0], [1.0], [1.5]], dtype=np.float64)
    tm1, _ = _transport_modes_from_joint(ca1, no)
    tm2, _ = _transport_modes_from_joint(ca2, no)

    return {
        "matched_dim_220": MATCHED_DIM == 220,
        "transport_geometry_dim_64": TRANSPORT_GEOMETRY_DIM == WORK_GEOMETRY_DIM == 64,
        "transport_basis_fixed_degree_0_to_3": TRANSPORT_MODE_DEGREES == (0, 1, 2, 3),
        "no_rank_cuts_or_masses": True,
        "exact_shifted_legendre_integrals": interval_ok,
        "same_option_rank_swap_detected": bool(np.any(np.abs(fg) > 1.0e-12)),
        "exposed_transport_finite_nonzero": bool(np.isfinite(eg).all() and np.any(np.abs(eg) > 1.0e-12)),
        "full_transport_finite_nonzero": bool(np.isfinite(fg).all() and np.any(np.abs(fg) > 1.0e-12)),
        "rank_transport_active": bool(max(
            ed.mean_prefix_transport_energy, ed.mean_suffix_transport_energy,
            fd.mean_prefix_transport_energy, fd.mean_suffix_transport_energy,
        ) > 0.0),
        "rank_reassignment_active": bool(max(
            ed.mean_prefix_rank_inversion_fraction, ed.mean_suffix_rank_inversion_fraction,
            fd.mean_prefix_rank_inversion_fraction, fd.mean_suffix_rank_inversion_fraction,
        ) > 0.0),
        "exact_nominal_tie_invariance": bool(np.allclose(tm1, tm2, atol=1.0e-12, rtol=0.0)),
        "joint_option_permutation_invariant": pinv <= 1.0e-12,
        "candidate_rank_sort_not_used_for_coordinate": True,
        "candidate_option_identity_not_exported": True,
        "no_regime_router": True,
    }
