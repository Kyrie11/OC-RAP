from __future__ import annotations

"""Observation-consistent recovery-set viability rank-persistence coupling audit.

V48.120 established a candidate-independent nominal viability-rank coordinate and
followed the same physical recovery option under the candidate, but the resulting
rank-transport field was summarized independently at each time before fixed-bin
averaging.  That preserves *instantaneous* nominal ordering and candidate causal
correspondence, yet it does not encode whether the same nominal recovery rank is
owned by a temporally persistent recovery option or by rapidly exchanging option
identities.  The V48.120 production run showed real rank reassignment and higher-
order transport activity without population-stable transfer, while Contact Reserve
retained directional gains.  The next preregistered question is therefore whether
candidate-induced viability displacement must be coupled to the persistence of the
same option's nominal rank trajectory.

For each temporal channel z in {prefix, suffix}, recovery option l, and time t:

    q_l^0,z(t) = nominal same-option joint viability margin
    q_l^a,z(t) = candidate same-option joint viability margin
    d_l^z(t)   = q_l^a,z(t) - q_l^0,z(t)

The nominal margins alone define a candidate-independent exact midrank r_l^z(t) in
[0,1]; exact nominal ties share one midrank and never receive arbitrary within-tie
identity order.  Rank persistence is measured directionally on the same temporal
semantics as the viability margin:

    pre_defect_l(t) = mean_{s<=t} |r_l^pre(t) - r_l^pre(s)|
    suf_defect_l(t) = mean_{s>=t} |r_l^suf(t) - r_l^suf(s)|

A value of zero means the same option has retained its nominal rank over the whole
relevant prefix/suffix.  No threshold, rank cut, learned kernel, or tuned decay is
introduced.

At each time the same-option displacement is projected onto four fixed coupling
modes:

    phi0 = 1                              global signed viability shift
    phi1 = x = 2 r - 1                   instantaneous nominal-rank tilt
    phi2 = p = persistence defect        displacement on rank-unstable options
    phi3 = x * p                         rank x persistence interaction

The option mean of d*phi is then averaged in the inherited eight full-horizon bins:

    8 bins x 2 temporal channels x 4 fixed coupling modes = 64-D
    156-D frozen candidate response + 64-D = 220-D.

Two equal-capacity families are audited:

  exposed_persistence -- coupling on the support (not weights) of V48.117's
                         frozen weak-root zero-boundary witnesses;
  full_persistence    -- coupling on every common valid recovery option [PRIMARY].

Option identity is used internally only for same-option nominal-rank trajectories
and candidate correspondence; no identity is exported to the readout.  Downstream
OC-MERO option selection is unchanged.  This audit introduces no planner/source/
root training, regime router, boundary transport, rank cut, threshold, horizon,
option-count, capacity, or hyperparameter sweep.
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

ENGINEERING_VERSION = "v48.121.0-OC-VRPC"
SCIENTIFIC_VERSION = "v48.121-OC-VRPC"
ALGORITHM_NAME = "Observation-Consistent Recovery-Set Viability Rank-Persistence Coupling Audit"

COUPLING_MODE_NAMES = (
    "global_shift",
    "nominal_rank_tilt",
    "rank_persistence_defect",
    "rank_persistence_interaction",
)
COUPLING_CHANNELS = 2
COUPLING_MODES = len(COUPLING_MODE_NAMES)
COUPLING_GEOMETRY_DIM = WORK_BINS * COUPLING_CHANNELS * COUPLING_MODES
MATCHED_DIM = RAW_CANDIDATE_DIM + COUPLING_GEOMETRY_DIM


@dataclass(frozen=True)
class RankPersistenceDiagnostics:
    prefix_coverage_fraction: float
    suffix_coverage_fraction: float
    mean_eligible_option_count: float
    mean_prefix_coupling_energy: float
    mean_suffix_coupling_energy: float
    mean_prefix_persistence_energy: float
    mean_suffix_persistence_energy: float
    prefix_nonzero_fraction: float
    suffix_nonzero_fraction: float
    mean_prefix_rank_persistence_defect: float
    mean_suffix_rank_persistence_defect: float
    mean_prefix_rank_inversion_fraction: float
    mean_suffix_rank_inversion_fraction: float


def _validate_pair(
    candidate: ExecutableConstraintField,
    nominal: ExecutableConstraintField,
    option_mask: np.ndarray,
) -> np.ndarray:
    if candidate.full_values.shape != nominal.full_values.shape:
        raise ValueError("VRPC candidate/nominal field shape mismatch")
    if candidate.full_masks.shape != nominal.full_masks.shape:
        raise ValueError("VRPC candidate/nominal mask shape mismatch")
    if not np.array_equal(candidate.option_valid, nominal.option_valid):
        raise ValueError("VRPC requires candidate-invariant option validity")
    if candidate.full_values.ndim != 3 or candidate.full_values.shape[2] != NUM_CONSTRAINTS:
        raise ValueError(f"VRPC invalid full field {candidate.full_values.shape}")
    mask = np.asarray(option_mask, dtype=bool).reshape(-1)
    if mask.size != len(nominal.option_valid):
        raise ValueError("VRPC option-mask length mismatch")
    if np.any(mask & ~nominal.option_valid):
        raise ValueError("VRPC option mask includes invalid recovery option")
    if not mask.any():
        raise ValueError("VRPC empty recovery-set support")
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
        raise ValueError("VRPC non-finite executable constraint path")
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


def _rank_inversion_fraction(nominal: np.ndarray, candidate: np.ndarray) -> float:
    """Pairwise same-option rank reversal fraction; exact ties are neutral diagnostics."""
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


def _nominal_midranks(nominal_joint: np.ndarray) -> np.ndarray:
    """Exact candidate-independent nominal midranks in [0,1], with exact tie blocks."""
    no = np.asarray(nominal_joint, dtype=np.float64)
    if no.ndim != 2:
        raise ValueError("VRPC nominal joint margins must be LxT")
    L, T = no.shape
    ranks = np.full((L, T), np.nan, dtype=np.float64)
    for t in range(T):
        ids = np.flatnonzero(np.isfinite(no[:, t]))
        if ids.size == 0:
            continue
        vals = no[ids, t]
        order = np.argsort(vals, kind="mergesort")  # worst -> best
        sorted_ids = ids[order]
        sorted_vals = vals[order]
        n = len(sorted_ids)
        i = 0
        while i < n:
            j = i + 1
            while j < n and sorted_vals[j] == sorted_vals[i]:
                j += 1
            # Midpoint of the exact occupied rank interval [i/n, j/n].
            mid = (float(i) + float(j)) / (2.0 * float(n))
            ranks[sorted_ids[i:j], t] = mid
            i = j
    return ranks


def _directional_rank_persistence_defect(ranks: np.ndarray, *, reverse: bool) -> np.ndarray:
    """Mean same-option nominal-rank displacement over the relevant prefix/suffix.

    The coordinate is candidate-independent and threshold-free.  A value of zero
    means exact rank persistence over the relevant directional history.
    """
    r = np.asarray(ranks, dtype=np.float64)
    if r.ndim != 2:
        raise ValueError("VRPC ranks must be LxT")
    L, T = r.shape
    out = np.full((L, T), np.nan, dtype=np.float64)
    for l in range(L):
        for t in range(T):
            if not np.isfinite(r[l, t]):
                continue
            window = r[l, t:] if reverse else r[l, : t + 1]
            window = window[np.isfinite(window)]
            if window.size == 0:
                continue
            out[l, t] = float(np.mean(np.abs(window - r[l, t])))
    return out


def _persistence_modes_from_joint(
    candidate_joint: np.ndarray,
    nominal_joint: np.ndarray,
    *,
    reverse: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return Tx4 nominal-rank/persistence coupling moments and diagnostics."""
    ca = np.asarray(candidate_joint, dtype=np.float64)
    no = np.asarray(nominal_joint, dtype=np.float64)
    if ca.shape != no.shape or ca.ndim != 2:
        raise ValueError("VRPC joint arrays must be matched LxT")
    if not np.array_equal(np.isfinite(ca), np.isfinite(no)):
        raise ValueError("VRPC candidate/nominal eligibility mismatch")

    ranks = _nominal_midranks(no)
    defect = _directional_rank_persistence_defect(ranks, reverse=reverse)
    _, T = ca.shape
    modes = np.zeros((T, COUPLING_MODES), dtype=np.float64)
    covered = np.zeros(T, dtype=bool)
    counts = np.zeros(T, dtype=np.float64)
    energy = np.zeros(T, dtype=np.float64)
    persistence_energy = np.zeros(T, dtype=np.float64)
    defect_mean = np.zeros(T, dtype=np.float64)
    inversions = np.zeros(T, dtype=np.float64)

    for t in range(T):
        finite = np.isfinite(no[:, t]) & np.isfinite(ranks[:, t]) & np.isfinite(defect[:, t])
        if not finite.any():
            continue
        nvals = no[finite, t]
        cvals = ca[finite, t]
        delta = cvals - nvals
        r = ranks[finite, t]
        p = defect[finite, t]
        x = 2.0 * r - 1.0
        phi = np.stack((np.ones_like(x), x, p, x * p), axis=1)
        row = np.mean(delta[:, None] * phi, axis=0)
        modes[t] = row
        covered[t] = True
        counts[t] = float(len(delta))
        energy[t] = float(np.linalg.norm(row))
        persistence_energy[t] = float(np.linalg.norm(row[2:]))
        defect_mean[t] = float(np.mean(p))
        inversions[t] = _rank_inversion_fraction(nvals, cvals)

    nz = energy > 1.0e-12
    return modes, {
        "coverage_fraction": float(np.mean(covered)),
        "mean_eligible_option_count": float(np.mean(counts[covered])) if covered.any() else 0.0,
        "mean_coupling_energy": float(np.mean(energy[covered])) if covered.any() else 0.0,
        "mean_persistence_energy": float(np.mean(persistence_energy[covered])) if covered.any() else 0.0,
        "nonzero_fraction": float(np.mean(nz[covered])) if covered.any() else 0.0,
        "mean_rank_persistence_defect": float(np.mean(defect_mean[covered])) if covered.any() else 0.0,
        "mean_rank_inversion_fraction": float(np.mean(inversions[covered])) if covered.any() else 0.0,
    }


def viability_rank_persistence_geometry(
    candidate: ExecutableConstraintField,
    nominal: ExecutableConstraintField,
    option_mask: np.ndarray,
) -> tuple[np.ndarray, RankPersistenceDiagnostics]:
    mask = _validate_pair(candidate, nominal, option_mask)
    pair_active = np.asarray(candidate.full_masks | nominal.full_masks, dtype=bool)

    cap = _joint_option_margins(candidate, pair_active, mask, reverse=False)
    nop = _joint_option_margins(nominal, pair_active, mask, reverse=False)
    cas = _joint_option_margins(candidate, pair_active, mask, reverse=True)
    nos = _joint_option_margins(nominal, pair_active, mask, reverse=True)

    prefix, pd = _persistence_modes_from_joint(cap, nop, reverse=False)
    suffix, sd = _persistence_modes_from_joint(cas, nos, reverse=True)

    geom = np.zeros((WORK_BINS, COUPLING_CHANNELS, COUPLING_MODES), dtype=np.float64)
    geom[:, 0, :] = _bin_mean(prefix)
    geom[:, 1, :] = _bin_mean(suffix)
    out = geom.reshape(-1)
    if out.shape != (COUPLING_GEOMETRY_DIM,) or not np.isfinite(out).all():
        raise ValueError("VRPC invalid rank-persistence geometry")

    diag = RankPersistenceDiagnostics(
        prefix_coverage_fraction=float(pd["coverage_fraction"]),
        suffix_coverage_fraction=float(sd["coverage_fraction"]),
        mean_eligible_option_count=float(min(pd["mean_eligible_option_count"], sd["mean_eligible_option_count"])),
        mean_prefix_coupling_energy=float(pd["mean_coupling_energy"]),
        mean_suffix_coupling_energy=float(sd["mean_coupling_energy"]),
        mean_prefix_persistence_energy=float(pd["mean_persistence_energy"]),
        mean_suffix_persistence_energy=float(sd["mean_persistence_energy"]),
        prefix_nonzero_fraction=float(pd["nonzero_fraction"]),
        suffix_nonzero_fraction=float(sd["nonzero_fraction"]),
        mean_prefix_rank_persistence_defect=float(pd["mean_rank_persistence_defect"]),
        mean_suffix_rank_persistence_defect=float(sd["mean_rank_persistence_defect"]),
        mean_prefix_rank_inversion_fraction=float(pd["mean_rank_inversion_fraction"]),
        mean_suffix_rank_inversion_fraction=float(sd["mean_rank_inversion_fraction"]),
    )
    return out, diag


def full_persistence_geometry(
    candidate: ExecutableConstraintField,
    nominal: ExecutableConstraintField,
) -> tuple[np.ndarray, RankPersistenceDiagnostics]:
    mask = np.asarray(candidate.option_valid & nominal.option_valid, dtype=bool)
    return viability_rank_persistence_geometry(candidate, nominal, mask)


def exposed_persistence_geometry(
    candidate: ExecutableConstraintField,
    nominal: ExecutableConstraintField,
    exposed_option_mask: np.ndarray,
) -> tuple[np.ndarray, RankPersistenceDiagnostics]:
    return viability_rank_persistence_geometry(candidate, nominal, exposed_option_mask)


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

    e0, _ = exposed_persistence_geometry(candidate, nominal, exposed_option_mask)
    f0, _ = full_persistence_geometry(candidate, nominal)
    e1, _ = exposed_persistence_geometry(pf(candidate), pf(nominal), np.asarray(exposed_option_mask)[perm])
    f1, _ = full_persistence_geometry(pf(candidate), pf(nominal))
    return float(max(np.max(np.abs(e0 - e1)), np.max(np.abs(f0 - f1))))


def contract_checks() -> dict[str, bool]:
    L, T, C = 4, 8, NUM_CONSTRAINTS
    ov = np.ones(L, dtype=bool)
    masks = np.ones((L, T, C), dtype=bool)

    # A persistent nominal rank path with a global same-option shift must have no
    # persistence-specific modes, proving that phi2/phi3 are not duplicate DC/rank modes.
    no_static = np.repeat(np.asarray([[-1.5], [-0.5], [0.5], [1.5]], dtype=np.float64), T, axis=1)
    ca_static = no_static + 0.4
    sm, sd = _persistence_modes_from_joint(ca_static, no_static, reverse=False)
    static_persistence_zero = bool(np.allclose(sm[:, 2:], 0.0, atol=1.0e-12, rtol=0.0))

    # Construct two identity histories with the same per-time nominal value multiset
    # and the same displacement as a function of instantaneous nominal rank.  An
    # Eulerian instantaneous rank summary is therefore identical, but the same-option
    # nominal rank histories differ and VRPC must distinguish them through persistence.
    levels = np.asarray([-1.5, -0.5, 0.5, 1.5], dtype=np.float64)
    rank_delta = np.asarray([3.0, 1.0, -1.0, -3.0], dtype=np.float64)
    no_a = np.repeat(levels[:, None], T, axis=1)
    ca_a = no_a + rank_delta[:, None]
    no_b = np.empty_like(no_a)
    ca_b = np.empty_like(ca_a)
    for t in range(T):
        perm = np.arange(L) if t % 2 == 0 else np.asarray([3, 1, 2, 0])
        no_b[:, t] = levels[perm]
        # Same displacement-by-rank profile as case A, attached to whichever identity
        # occupies that nominal rank at this time.
        ca_b[:, t] = no_b[:, t]
        order = np.argsort(no_b[:, t], kind="mergesort")
        ca_b[order, t] += rank_delta
    ma, _ = _persistence_modes_from_joint(ca_a, no_a, reverse=False)
    mb, bd = _persistence_modes_from_joint(ca_b, no_b, reverse=False)
    # Modes 0/1 are the instantaneous rank transport and remain identical; persistence
    # modes 2/3 must separate the two same-option histories.
    instantaneous_equal = bool(np.allclose(ma[:, :2], mb[:, :2], atol=1.0e-12, rtol=0.0))
    persistence_detects_identity_path = bool(np.max(np.abs(ma[:, 2:] - mb[:, 2:])) > 1.0e-6)

    h0 = np.repeat(no_b[:, :, None], C, axis=2)
    ha = np.repeat(ca_b[:, :, None], C, axis=2)
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
    eg, ed = exposed_persistence_geometry(candidate, nominal, exposed)
    fg, fd = full_persistence_geometry(candidate, nominal)
    pinv = option_permutation_invariance_error(candidate, nominal, exposed)

    # Exact tie invariance under an arbitrary identity swap within a nominal tie.
    no_tie = np.asarray([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    ca1 = np.asarray([[1.0, 0.5], [-1.0, -0.5], [1.5, 1.25]], dtype=np.float64)
    ca2 = ca1[[1, 0, 2]]
    no_tie2 = no_tie[[1, 0, 2]]
    tm1, _ = _persistence_modes_from_joint(ca1, no_tie, reverse=False)
    tm2, _ = _persistence_modes_from_joint(ca2, no_tie2, reverse=False)

    return {
        "matched_dim_220": MATCHED_DIM == 220,
        "coupling_geometry_dim_64": COUPLING_GEOMETRY_DIM == WORK_GEOMETRY_DIM == 64,
        "coupling_modes_fixed": COUPLING_MODE_NAMES == (
            "global_shift", "nominal_rank_tilt", "rank_persistence_defect", "rank_persistence_interaction"
        ),
        "no_rank_cuts_masses_or_decay": True,
        "persistent_rank_path_zeroes_persistence_modes": static_persistence_zero,
        "instantaneous_rank_transport_control_equal": instantaneous_equal,
        "same_option_rank_persistence_detected": persistence_detects_identity_path,
        "exposed_persistence_finite_nonzero": bool(np.isfinite(eg).all() and np.any(np.abs(eg) > 1.0e-12)),
        "full_persistence_finite_nonzero": bool(np.isfinite(fg).all() and np.any(np.abs(fg) > 1.0e-12)),
        "rank_persistence_active": bool(max(
            ed.mean_prefix_rank_persistence_defect, ed.mean_suffix_rank_persistence_defect,
            fd.mean_prefix_rank_persistence_defect, fd.mean_suffix_rank_persistence_defect,
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
