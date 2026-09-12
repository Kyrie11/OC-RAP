from __future__ import annotations

"""Observation-consistent signed viability rank-state transport audit.

V48.120 preserved a candidate-independent nominal recovery ordering while following
exactly the same recovery option under the candidate. V48.121 then tested whether
candidate-induced displacement must be coupled to persistence of that nominal rank
trajectory. The V48.121 production result showed real persistence structure and a
clean exposed-support Support improvement, but persistence remained insufficient as
a population-stable complete carrier. The preregistered missing primitive is the
*absolute signed nominal viability level relative to the physical zero boundary*.

Rank alone cannot distinguish shallow reserve from deep reserve, nor shallow debt
from deep debt. For each temporal channel z in {prefix, suffix}, recovery option l,
and time t:

    q_l^0,z(t) = nominal same-option joint signed viability margin
    q_l^a,z(t) = candidate same-option joint signed viability margin
    d_l^z(t)   = q_l^a,z(t) - q_l^0,z(t)

The nominal margins alone define an exact candidate-independent midrank r_l^z(t) in
[0,1], with exact nominal ties sharing one midrank. The signed state coordinate is
the physically normalized nominal margin itself:

    s_l^z(t) = q_l^0,z(t)

Zero is therefore the actual reserve/debt boundary; no centering, learned scale,
threshold, clipping, temperature, or rank cut is introduced. The heterogeneous
constraint construction already normalizes each signed margin by fixed physical /
configuration scales while preserving zero.

At each time the same-option displacement is projected onto a fixed tensor-product
basis of nominal rank and signed nominal state:

    phi0 = 1                              global signed viability shift
    phi1 = x = 2 r - 1                   nominal-rank tilt
    phi2 = s = q^0                       signed nominal-state coupling
    phi3 = x * s                         rank x signed-state interaction

The first two modes exactly retain the V48.121 instantaneous global/rank transport
control. The latter two are the only new primitive. The option mean of d*phi is
then averaged in the inherited eight full-horizon bins:

    8 bins x 2 temporal channels x 4 fixed modes = 64-D
    156-D frozen candidate response + 64-D = 220-D.

Two equal-capacity families are audited:

  exposed_signed_state -- transport on the support (not weights) of V48.117's
                          frozen weak-root zero-boundary witnesses;
  full_signed_state    -- transport on every common valid recovery option [PRIMARY].

Option identity is used internally only for same-option candidate correspondence;
no identity is exported to the readout. Candidate values never define the rank
coordinate. Downstream OC-MERO selection is unchanged. This audit introduces no
planner/source/root training, regime router, boundary transport, persistence/rank
window, threshold, horizon, option-count, capacity, or hyperparameter sweep.
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

ENGINEERING_VERSION = "v48.122.0-OC-SVRT"
SCIENTIFIC_VERSION = "v48.122-OC-SVRT"
ALGORITHM_NAME = "Observation-Consistent Signed Viability Rank-State Transport Audit"

STATE_MODE_NAMES = (
    "global_shift",
    "nominal_rank_tilt",
    "signed_nominal_state_coupling",
    "rank_signed_nominal_state_interaction",
)
STATE_CHANNELS = 2
STATE_MODES = len(STATE_MODE_NAMES)
STATE_GEOMETRY_DIM = WORK_BINS * STATE_CHANNELS * STATE_MODES
MATCHED_DIM = RAW_CANDIDATE_DIM + STATE_GEOMETRY_DIM


@dataclass(frozen=True)
class SignedRankStateDiagnostics:
    prefix_coverage_fraction: float
    suffix_coverage_fraction: float
    mean_eligible_option_count: float
    mean_prefix_coupling_energy: float
    mean_suffix_coupling_energy: float
    mean_prefix_signed_state_energy: float
    mean_suffix_signed_state_energy: float
    prefix_nonzero_fraction: float
    suffix_nonzero_fraction: float
    mean_prefix_abs_nominal_state: float
    mean_suffix_abs_nominal_state: float
    mean_prefix_two_sided_state_fraction: float
    mean_suffix_two_sided_state_fraction: float
    mean_prefix_zero_crossing_fraction: float
    mean_suffix_zero_crossing_fraction: float
    mean_prefix_rank_inversion_fraction: float
    mean_suffix_rank_inversion_fraction: float


def _validate_pair(
    candidate: ExecutableConstraintField,
    nominal: ExecutableConstraintField,
    option_mask: np.ndarray,
) -> np.ndarray:
    if candidate.full_values.shape != nominal.full_values.shape:
        raise ValueError("SVRT candidate/nominal field shape mismatch")
    if candidate.full_masks.shape != nominal.full_masks.shape:
        raise ValueError("SVRT candidate/nominal mask shape mismatch")
    if not np.array_equal(candidate.option_valid, nominal.option_valid):
        raise ValueError("SVRT requires candidate-invariant option validity")
    if candidate.full_values.ndim != 3 or candidate.full_values.shape[2] != NUM_CONSTRAINTS:
        raise ValueError(f"SVRT invalid full field {candidate.full_values.shape}")
    mask = np.asarray(option_mask, dtype=bool).reshape(-1)
    if mask.size != len(nominal.option_valid):
        raise ValueError("SVRT option-mask length mismatch")
    if np.any(mask & ~nominal.option_valid):
        raise ValueError("SVRT option mask includes invalid recovery option")
    if not mask.any():
        raise ValueError("SVRT empty recovery-set support")
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
        raise ValueError("SVRT non-finite executable constraint path")
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
    """Pairwise same-option rank reversal fraction; exact nominal ties are neutral."""
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
        raise ValueError("SVRT nominal joint margins must be LxT")
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
            mid = (float(i) + float(j)) / (2.0 * float(n))
            ranks[sorted_ids[i:j], t] = mid
            i = j
    return ranks


def _signed_state_modes_from_joint(
    candidate_joint: np.ndarray,
    nominal_joint: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return Tx4 nominal-rank/signed-state transport moments and diagnostics."""
    ca = np.asarray(candidate_joint, dtype=np.float64)
    no = np.asarray(nominal_joint, dtype=np.float64)
    if ca.shape != no.shape or ca.ndim != 2:
        raise ValueError("SVRT joint arrays must be matched LxT")
    if not np.array_equal(np.isfinite(ca), np.isfinite(no)):
        raise ValueError("SVRT candidate/nominal eligibility mismatch")

    ranks = _nominal_midranks(no)
    _, T = ca.shape
    modes = np.zeros((T, STATE_MODES), dtype=np.float64)
    covered = np.zeros(T, dtype=bool)
    counts = np.zeros(T, dtype=np.float64)
    energy = np.zeros(T, dtype=np.float64)
    state_energy = np.zeros(T, dtype=np.float64)
    abs_state = np.zeros(T, dtype=np.float64)
    two_sided = np.zeros(T, dtype=np.float64)
    crossings = np.zeros(T, dtype=np.float64)
    inversions = np.zeros(T, dtype=np.float64)

    for t in range(T):
        finite = np.isfinite(no[:, t]) & np.isfinite(ranks[:, t])
        if not finite.any():
            continue
        nvals = no[finite, t]
        cvals = ca[finite, t]
        delta = cvals - nvals
        r = ranks[finite, t]
        x = 2.0 * r - 1.0
        s = nvals  # physical/config normalized signed nominal viability; zero is exact boundary.
        phi = np.stack((np.ones_like(x), x, s, x * s), axis=1)
        row = np.mean(delta[:, None] * phi, axis=0)
        modes[t] = row
        covered[t] = True
        counts[t] = float(len(delta))
        energy[t] = float(np.linalg.norm(row))
        state_energy[t] = float(np.linalg.norm(row[2:]))
        abs_state[t] = float(np.mean(np.abs(s)))
        two_sided[t] = float(bool(np.any(s < 0.0) and np.any(s > 0.0)))
        # Exact zero is neutral; count only strict side changes.
        crossings[t] = float(np.mean(((s < 0.0) & (cvals > 0.0)) | ((s > 0.0) & (cvals < 0.0))))
        inversions[t] = _rank_inversion_fraction(nvals, cvals)

    nz = energy > 1.0e-12
    return modes, {
        "coverage_fraction": float(np.mean(covered)),
        "mean_eligible_option_count": float(np.mean(counts[covered])) if covered.any() else 0.0,
        "mean_coupling_energy": float(np.mean(energy[covered])) if covered.any() else 0.0,
        "mean_signed_state_energy": float(np.mean(state_energy[covered])) if covered.any() else 0.0,
        "nonzero_fraction": float(np.mean(nz[covered])) if covered.any() else 0.0,
        "mean_abs_nominal_state": float(np.mean(abs_state[covered])) if covered.any() else 0.0,
        "mean_two_sided_state_fraction": float(np.mean(two_sided[covered])) if covered.any() else 0.0,
        "mean_zero_crossing_fraction": float(np.mean(crossings[covered])) if covered.any() else 0.0,
        "mean_rank_inversion_fraction": float(np.mean(inversions[covered])) if covered.any() else 0.0,
    }


def signed_viability_rank_state_geometry(
    candidate: ExecutableConstraintField,
    nominal: ExecutableConstraintField,
    option_mask: np.ndarray,
) -> tuple[np.ndarray, SignedRankStateDiagnostics]:
    mask = _validate_pair(candidate, nominal, option_mask)
    pair_active = np.asarray(candidate.full_masks | nominal.full_masks, dtype=bool)

    cap = _joint_option_margins(candidate, pair_active, mask, reverse=False)
    nop = _joint_option_margins(nominal, pair_active, mask, reverse=False)
    cas = _joint_option_margins(candidate, pair_active, mask, reverse=True)
    nos = _joint_option_margins(nominal, pair_active, mask, reverse=True)

    prefix, pd = _signed_state_modes_from_joint(cap, nop)
    suffix, sd = _signed_state_modes_from_joint(cas, nos)

    geom = np.zeros((WORK_BINS, STATE_CHANNELS, STATE_MODES), dtype=np.float64)
    geom[:, 0, :] = _bin_mean(prefix)
    geom[:, 1, :] = _bin_mean(suffix)
    out = geom.reshape(-1)
    if out.shape != (STATE_GEOMETRY_DIM,) or not np.isfinite(out).all():
        raise ValueError("SVRT invalid signed rank-state geometry")

    diag = SignedRankStateDiagnostics(
        prefix_coverage_fraction=float(pd["coverage_fraction"]),
        suffix_coverage_fraction=float(sd["coverage_fraction"]),
        mean_eligible_option_count=float(min(pd["mean_eligible_option_count"], sd["mean_eligible_option_count"])),
        mean_prefix_coupling_energy=float(pd["mean_coupling_energy"]),
        mean_suffix_coupling_energy=float(sd["mean_coupling_energy"]),
        mean_prefix_signed_state_energy=float(pd["mean_signed_state_energy"]),
        mean_suffix_signed_state_energy=float(sd["mean_signed_state_energy"]),
        prefix_nonzero_fraction=float(pd["nonzero_fraction"]),
        suffix_nonzero_fraction=float(sd["nonzero_fraction"]),
        mean_prefix_abs_nominal_state=float(pd["mean_abs_nominal_state"]),
        mean_suffix_abs_nominal_state=float(sd["mean_abs_nominal_state"]),
        mean_prefix_two_sided_state_fraction=float(pd["mean_two_sided_state_fraction"]),
        mean_suffix_two_sided_state_fraction=float(sd["mean_two_sided_state_fraction"]),
        mean_prefix_zero_crossing_fraction=float(pd["mean_zero_crossing_fraction"]),
        mean_suffix_zero_crossing_fraction=float(sd["mean_zero_crossing_fraction"]),
        mean_prefix_rank_inversion_fraction=float(pd["mean_rank_inversion_fraction"]),
        mean_suffix_rank_inversion_fraction=float(sd["mean_rank_inversion_fraction"]),
    )
    return out, diag


def full_signed_state_geometry(
    candidate: ExecutableConstraintField,
    nominal: ExecutableConstraintField,
) -> tuple[np.ndarray, SignedRankStateDiagnostics]:
    mask = np.asarray(candidate.option_valid & nominal.option_valid, dtype=bool)
    return signed_viability_rank_state_geometry(candidate, nominal, mask)


def exposed_signed_state_geometry(
    candidate: ExecutableConstraintField,
    nominal: ExecutableConstraintField,
    exposed_option_mask: np.ndarray,
) -> tuple[np.ndarray, SignedRankStateDiagnostics]:
    return signed_viability_rank_state_geometry(candidate, nominal, exposed_option_mask)


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

    e0, _ = exposed_signed_state_geometry(candidate, nominal, exposed_option_mask)
    f0, _ = full_signed_state_geometry(candidate, nominal)
    e1, _ = exposed_signed_state_geometry(pf(candidate), pf(nominal), np.asarray(exposed_option_mask)[perm])
    f1, _ = full_signed_state_geometry(pf(candidate), pf(nominal))
    return float(max(np.max(np.abs(e0 - e1)), np.max(np.abs(f0 - f1))))


def contract_checks() -> dict[str, bool]:
    L, T, C = 4, 8, NUM_CONSTRAINTS
    ov = np.ones(L, dtype=bool)
    masks = np.ones((L, T, C), dtype=bool)

    # Same ranks and same candidate displacement, but a different absolute nominal
    # signed level. Modes 0/1 must remain identical while modes 2/3 change.
    base = np.asarray([-1.5, -0.5, 0.5, 1.5], dtype=np.float64)
    delta = np.asarray([0.4, -0.1, 0.2, 0.6], dtype=np.float64)
    no_a = np.repeat(base[:, None], T, axis=1)
    ca_a = no_a + delta[:, None]
    no_b = no_a + 3.0
    ca_b = no_b + delta[:, None]
    ma, _ = _signed_state_modes_from_joint(ca_a, no_a)
    mb, _ = _signed_state_modes_from_joint(ca_b, no_b)
    instantaneous_control_equal = bool(np.allclose(ma[:, :2], mb[:, :2], atol=1.0e-12, rtol=0.0))
    signed_state_disambiguates = bool(np.max(np.abs(ma[:, 2:] - mb[:, 2:])) > 1.0e-6)

    # At the exact physical boundary q0=0, signed-state-specific modes vanish.
    no_zero = np.zeros((L, T), dtype=np.float64)
    ca_zero = np.repeat(delta[:, None], T, axis=1)
    mz, _ = _signed_state_modes_from_joint(ca_zero, no_zero)
    zero_boundary_exact = bool(np.allclose(mz[:, 2:], 0.0, atol=1.0e-12, rtol=0.0))

    h0 = np.repeat(no_a[:, :, None], C, axis=2)
    ha = np.repeat(ca_a[:, :, None], C, axis=2)
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
    eg, ed = exposed_signed_state_geometry(candidate, nominal, exposed)
    fg, fd = full_signed_state_geometry(candidate, nominal)
    pinv = option_permutation_invariance_error(candidate, nominal, exposed)

    # Exact tie invariance under arbitrary identity swap inside a nominal tie block.
    no_tie = np.asarray([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    ca1 = np.asarray([[1.0, 0.5], [-1.0, -0.5], [1.5, 1.25]], dtype=np.float64)
    ca2 = ca1[[1, 0, 2]]
    no_tie2 = no_tie[[1, 0, 2]]
    tm1, _ = _signed_state_modes_from_joint(ca1, no_tie)
    tm2, _ = _signed_state_modes_from_joint(ca2, no_tie2)

    return {
        "matched_dim_220": MATCHED_DIM == 220,
        "state_geometry_dim_64": STATE_GEOMETRY_DIM == WORK_GEOMETRY_DIM == 64,
        "state_modes_fixed": STATE_MODE_NAMES == (
            "global_shift", "nominal_rank_tilt", "signed_nominal_state_coupling", "rank_signed_nominal_state_interaction"
        ),
        "absolute_signed_state_relative_to_zero_boundary": True,
        "no_state_threshold_clip_or_learned_scale": True,
        "instantaneous_global_rank_control_equal": instantaneous_control_equal,
        "signed_nominal_state_disambiguates_equal_rank_transport": signed_state_disambiguates,
        "exact_zero_boundary_annuls_state_modes": zero_boundary_exact,
        "exposed_signed_state_finite_nonzero": bool(np.isfinite(eg).all() and np.any(np.abs(eg) > 1.0e-12)),
        "full_signed_state_finite_nonzero": bool(np.isfinite(fg).all() and np.any(np.abs(fg) > 1.0e-12)),
        "signed_state_active": bool(max(
            ed.mean_prefix_abs_nominal_state, ed.mean_suffix_abs_nominal_state,
            fd.mean_prefix_abs_nominal_state, fd.mean_suffix_abs_nominal_state,
        ) > 0.0),
        "signed_state_coupling_active": bool(max(
            ed.mean_prefix_signed_state_energy, ed.mean_suffix_signed_state_energy,
            fd.mean_prefix_signed_state_energy, fd.mean_suffix_signed_state_energy,
        ) > 0.0),
        "rank_reassignment_active": bool(_rank_inversion_fraction(
            base, np.asarray([1.6, -0.6, 0.7, -1.4], dtype=np.float64)
        ) > 0.0),
        "exact_nominal_tie_invariance": bool(np.allclose(tm1, tm2, atol=1.0e-12, rtol=0.0)),
        "joint_option_permutation_invariant": pinv <= 1.0e-12,
        "candidate_rank_sort_not_used_for_coordinate": True,
        "candidate_option_identity_not_exported": True,
        "no_regime_router": True,
    }
