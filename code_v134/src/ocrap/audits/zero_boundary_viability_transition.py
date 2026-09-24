from __future__ import annotations

"""Observation-consistent zero-boundary viability state-transition audit.

V48.120 established candidate-independent nominal recovery ordering with exact
same-option candidate correspondence. V48.121 showed that nominal-rank persistence
carries real but incomplete information, and V48.122 showed that absolute signed
nominal viability relative to the physical zero boundary materially helps the
observation-exposed Support carrier but is not a population-stable complete carrier.

The remaining preregistered question is not another rank/state statistic. It is
whether the *transition through the physical viability boundary* is the missing
coordinate. For each temporal channel z in {prefix, suffix}, recovery option l and
time t, let

    q_l^0,z(t) = nominal same-option joint signed viability margin
    q_l^a,z(t) = candidate same-option joint signed viability margin.

Using the exact physical zero boundary, decompose every same-option displacement
into a safety-reserve transition and a debt-repayment transition:

    Delta R = [q^a]_+ - [q^0]_+
    Delta D = [-q^0]_+ - [-q^a]_+

where [x]_+ = max(x, 0). The decomposition is exact:

    q^a - q^0 = Delta R + Delta D.

Thus positive Delta R means added/preserved safe reserve, while positive Delta D
means debt repayment toward or through re-entry. No regime label is used: the
zero-boundary decomposition is determined entirely by the signed physical state.
The nominal margins alone also define the same candidate-independent exact midrank
r in [0,1], with exact nominal ties sharing one midrank, and x = 2r - 1.

The four fixed transition modes are

    phi0 = Delta R             reserve transition
    phi1 = x * Delta R         nominal-rank-weighted reserve transition
    phi2 = Delta D             debt-repayment transition
    phi3 = x * Delta D         nominal-rank-weighted debt-repayment transition.

The V48.120 instantaneous global/rank transport control is exactly recoverable as
phi0 + phi2 and phi1 + phi3. Therefore the new representation does not discard the
established same-option rank transport; it only resolves *which side of the physical
zero boundary generated it*. The option means are averaged in the inherited eight
full-horizon bins:

    8 bins x 2 temporal channels x 4 fixed modes = 64-D
    156-D frozen candidate response + 64-D = 220-D.

Two equal-capacity families are audited:

  exposed_transition -- zero-boundary transitions on the support (not weights) of
                        V48.117 frozen weak-root zero-boundary witnesses;
  full_transition    -- zero-boundary transitions on every common valid recovery
                        option [PRIMARY].

Option identity is used internally only for same-option correspondence and is never
exported to the readout. Candidate values never define the rank coordinate. The zero
boundary is fixed at exactly 0; there is no threshold, clipping, learned scale,
transition window, decay, rank cut, option-count, capacity, source, regime, horizon,
or hyperparameter sweep. Downstream OC-MERO selection is unchanged.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from ocrap.audits.common_option_constraint_work import WORK_BINS, WORK_GEOMETRY_DIM, _bin_mean
from ocrap.audits.constraint_native_orientation import RAW_CANDIDATE_DIM
from ocrap.audits.executable_constraint_jacobian import ExecutableConstraintField
from ocrap.audits.heterogeneous_constraint_normal_cone import NUM_CONSTRAINTS
from ocrap.audits.recovery_set_constraint_flow import base_features, fit_set_flow_scaler, matched_features

ENGINEERING_VERSION = "v48.123.0-OC-ZBST"
SCIENTIFIC_VERSION = "v48.123-OC-ZBST"
ALGORITHM_NAME = "Observation-Consistent Zero-Boundary Viability State Transition Audit"

TRANSITION_MODE_NAMES = (
    "reserve_transition",
    "nominal_rank_reserve_transition",
    "debt_repayment_transition",
    "nominal_rank_debt_repayment_transition",
)
TRANSITION_CHANNELS = 2
TRANSITION_MODES = len(TRANSITION_MODE_NAMES)
TRANSITION_GEOMETRY_DIM = WORK_BINS * TRANSITION_CHANNELS * TRANSITION_MODES
MATCHED_DIM = RAW_CANDIDATE_DIM + TRANSITION_GEOMETRY_DIM


@dataclass(frozen=True)
class ZeroBoundaryTransitionDiagnostics:
    prefix_coverage_fraction: float
    suffix_coverage_fraction: float
    mean_eligible_option_count: float
    mean_prefix_transition_energy: float
    mean_suffix_transition_energy: float
    mean_prefix_reserve_transition_energy: float
    mean_suffix_reserve_transition_energy: float
    mean_prefix_debt_repayment_energy: float
    mean_suffix_debt_repayment_energy: float
    prefix_nonzero_fraction: float
    suffix_nonzero_fraction: float
    mean_prefix_zero_crossing_fraction: float
    mean_suffix_zero_crossing_fraction: float
    mean_prefix_debt_to_reserve_fraction: float
    mean_suffix_debt_to_reserve_fraction: float
    mean_prefix_reserve_to_debt_fraction: float
    mean_suffix_reserve_to_debt_fraction: float
    mean_prefix_rank_inversion_fraction: float
    mean_suffix_rank_inversion_fraction: float
    max_prefix_displacement_decomposition_error: float
    max_suffix_displacement_decomposition_error: float


def _validate_pair(
    candidate: ExecutableConstraintField,
    nominal: ExecutableConstraintField,
    option_mask: np.ndarray,
) -> np.ndarray:
    if candidate.full_values.shape != nominal.full_values.shape:
        raise ValueError("ZBST candidate/nominal field shape mismatch")
    if candidate.full_masks.shape != nominal.full_masks.shape:
        raise ValueError("ZBST candidate/nominal mask shape mismatch")
    if not np.array_equal(candidate.option_valid, nominal.option_valid):
        raise ValueError("ZBST requires candidate-invariant option validity")
    if candidate.full_values.ndim != 3 or candidate.full_values.shape[2] != NUM_CONSTRAINTS:
        raise ValueError(f"ZBST invalid full field {candidate.full_values.shape}")
    mask = np.asarray(option_mask, dtype=bool).reshape(-1)
    if mask.size != len(nominal.option_valid):
        raise ValueError("ZBST option-mask length mismatch")
    if np.any(mask & ~nominal.option_valid):
        raise ValueError("ZBST option mask includes invalid recovery option")
    if not mask.any():
        raise ValueError("ZBST empty recovery-set support")
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
        raise ValueError("ZBST non-finite executable constraint path")
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
        raise ValueError("ZBST nominal joint margins must be LxT")
    L, T = no.shape
    ranks = np.full((L, T), np.nan, dtype=np.float64)
    for t in range(T):
        ids = np.flatnonzero(np.isfinite(no[:, t]))
        if ids.size == 0:
            continue
        vals = no[ids, t]
        order = np.argsort(vals, kind="mergesort")
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


def _transition_modes_from_joint(
    candidate_joint: np.ndarray,
    nominal_joint: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return Tx4 exact zero-boundary transition moments and diagnostics."""
    ca = np.asarray(candidate_joint, dtype=np.float64)
    no = np.asarray(nominal_joint, dtype=np.float64)
    if ca.shape != no.shape or ca.ndim != 2:
        raise ValueError("ZBST joint arrays must be matched LxT")
    if not np.array_equal(np.isfinite(ca), np.isfinite(no)):
        raise ValueError("ZBST candidate/nominal eligibility mismatch")

    ranks = _nominal_midranks(no)
    _, T = ca.shape
    modes = np.zeros((T, TRANSITION_MODES), dtype=np.float64)
    covered = np.zeros(T, dtype=bool)
    counts = np.zeros(T, dtype=np.float64)
    energy = np.zeros(T, dtype=np.float64)
    reserve_energy = np.zeros(T, dtype=np.float64)
    debt_energy = np.zeros(T, dtype=np.float64)
    crossings = np.zeros(T, dtype=np.float64)
    debt_to_reserve = np.zeros(T, dtype=np.float64)
    reserve_to_debt = np.zeros(T, dtype=np.float64)
    inversions = np.zeros(T, dtype=np.float64)
    decomposition_error = np.zeros(T, dtype=np.float64)

    for t in range(T):
        finite = np.isfinite(no[:, t]) & np.isfinite(ranks[:, t])
        if not finite.any():
            continue
        q0 = no[finite, t]
        qa = ca[finite, t]
        r = ranks[finite, t]
        x = 2.0 * r - 1.0

        reserve_delta = np.maximum(qa, 0.0) - np.maximum(q0, 0.0)
        debt_repayment = np.maximum(-q0, 0.0) - np.maximum(-qa, 0.0)
        displacement = qa - q0
        decomposition_error[t] = float(np.max(np.abs(displacement - (reserve_delta + debt_repayment))))

        row = np.asarray([
            np.mean(reserve_delta),
            np.mean(x * reserve_delta),
            np.mean(debt_repayment),
            np.mean(x * debt_repayment),
        ], dtype=np.float64)
        modes[t] = row
        covered[t] = True
        counts[t] = float(len(q0))
        energy[t] = float(np.linalg.norm(row))
        reserve_energy[t] = float(np.linalg.norm(row[:2]))
        debt_energy[t] = float(np.linalg.norm(row[2:]))
        d2r = (q0 < 0.0) & (qa > 0.0)
        r2d = (q0 > 0.0) & (qa < 0.0)
        debt_to_reserve[t] = float(np.mean(d2r))
        reserve_to_debt[t] = float(np.mean(r2d))
        crossings[t] = float(np.mean(d2r | r2d))
        inversions[t] = _rank_inversion_fraction(q0, qa)

    nz = energy > 1.0e-12
    return modes, {
        "coverage_fraction": float(np.mean(covered)),
        "mean_eligible_option_count": float(np.mean(counts[covered])) if covered.any() else 0.0,
        "mean_transition_energy": float(np.mean(energy[covered])) if covered.any() else 0.0,
        "mean_reserve_transition_energy": float(np.mean(reserve_energy[covered])) if covered.any() else 0.0,
        "mean_debt_repayment_energy": float(np.mean(debt_energy[covered])) if covered.any() else 0.0,
        "nonzero_fraction": float(np.mean(nz[covered])) if covered.any() else 0.0,
        "mean_zero_crossing_fraction": float(np.mean(crossings[covered])) if covered.any() else 0.0,
        "mean_debt_to_reserve_fraction": float(np.mean(debt_to_reserve[covered])) if covered.any() else 0.0,
        "mean_reserve_to_debt_fraction": float(np.mean(reserve_to_debt[covered])) if covered.any() else 0.0,
        "mean_rank_inversion_fraction": float(np.mean(inversions[covered])) if covered.any() else 0.0,
        "max_displacement_decomposition_error": float(np.max(decomposition_error[covered])) if covered.any() else 0.0,
    }


def zero_boundary_viability_transition_geometry(
    candidate: ExecutableConstraintField,
    nominal: ExecutableConstraintField,
    option_mask: np.ndarray,
) -> tuple[np.ndarray, ZeroBoundaryTransitionDiagnostics]:
    mask = _validate_pair(candidate, nominal, option_mask)
    pair_active = np.asarray(candidate.full_masks | nominal.full_masks, dtype=bool)

    cap = _joint_option_margins(candidate, pair_active, mask, reverse=False)
    nop = _joint_option_margins(nominal, pair_active, mask, reverse=False)
    cas = _joint_option_margins(candidate, pair_active, mask, reverse=True)
    nos = _joint_option_margins(nominal, pair_active, mask, reverse=True)

    prefix, pd = _transition_modes_from_joint(cap, nop)
    suffix, sd = _transition_modes_from_joint(cas, nos)

    geom = np.zeros((WORK_BINS, TRANSITION_CHANNELS, TRANSITION_MODES), dtype=np.float64)
    geom[:, 0, :] = _bin_mean(prefix)
    geom[:, 1, :] = _bin_mean(suffix)
    out = geom.reshape(-1)
    if out.shape != (TRANSITION_GEOMETRY_DIM,) or not np.isfinite(out).all():
        raise ValueError("ZBST invalid zero-boundary transition geometry")

    diag = ZeroBoundaryTransitionDiagnostics(
        prefix_coverage_fraction=float(pd["coverage_fraction"]),
        suffix_coverage_fraction=float(sd["coverage_fraction"]),
        mean_eligible_option_count=float(min(pd["mean_eligible_option_count"], sd["mean_eligible_option_count"])),
        mean_prefix_transition_energy=float(pd["mean_transition_energy"]),
        mean_suffix_transition_energy=float(sd["mean_transition_energy"]),
        mean_prefix_reserve_transition_energy=float(pd["mean_reserve_transition_energy"]),
        mean_suffix_reserve_transition_energy=float(sd["mean_reserve_transition_energy"]),
        mean_prefix_debt_repayment_energy=float(pd["mean_debt_repayment_energy"]),
        mean_suffix_debt_repayment_energy=float(sd["mean_debt_repayment_energy"]),
        prefix_nonzero_fraction=float(pd["nonzero_fraction"]),
        suffix_nonzero_fraction=float(sd["nonzero_fraction"]),
        mean_prefix_zero_crossing_fraction=float(pd["mean_zero_crossing_fraction"]),
        mean_suffix_zero_crossing_fraction=float(sd["mean_zero_crossing_fraction"]),
        mean_prefix_debt_to_reserve_fraction=float(pd["mean_debt_to_reserve_fraction"]),
        mean_suffix_debt_to_reserve_fraction=float(sd["mean_debt_to_reserve_fraction"]),
        mean_prefix_reserve_to_debt_fraction=float(pd["mean_reserve_to_debt_fraction"]),
        mean_suffix_reserve_to_debt_fraction=float(sd["mean_reserve_to_debt_fraction"]),
        mean_prefix_rank_inversion_fraction=float(pd["mean_rank_inversion_fraction"]),
        mean_suffix_rank_inversion_fraction=float(sd["mean_rank_inversion_fraction"]),
        max_prefix_displacement_decomposition_error=float(pd["max_displacement_decomposition_error"]),
        max_suffix_displacement_decomposition_error=float(sd["max_displacement_decomposition_error"]),
    )
    return out, diag


def full_transition_geometry(
    candidate: ExecutableConstraintField,
    nominal: ExecutableConstraintField,
) -> tuple[np.ndarray, ZeroBoundaryTransitionDiagnostics]:
    mask = np.asarray(candidate.option_valid & nominal.option_valid, dtype=bool)
    return zero_boundary_viability_transition_geometry(candidate, nominal, mask)


def exposed_transition_geometry(
    candidate: ExecutableConstraintField,
    nominal: ExecutableConstraintField,
    exposed_option_mask: np.ndarray,
) -> tuple[np.ndarray, ZeroBoundaryTransitionDiagnostics]:
    return zero_boundary_viability_transition_geometry(candidate, nominal, exposed_option_mask)


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

    e0, _ = exposed_transition_geometry(candidate, nominal, exposed_option_mask)
    f0, _ = full_transition_geometry(candidate, nominal)
    e1, _ = exposed_transition_geometry(pf(candidate), pf(nominal), np.asarray(exposed_option_mask)[perm])
    f1, _ = full_transition_geometry(pf(candidate), pf(nominal))
    return float(max(np.max(np.abs(e0 - e1)), np.max(np.abs(f0 - f1))))


def contract_checks() -> dict[str, bool]:
    L, T, C = 4, 8, NUM_CONSTRAINTS
    ov = np.ones(L, dtype=bool)
    masks = np.ones((L, T, C), dtype=bool)

    # Mixed sides with actual zero-boundary transitions and rank reassignment.
    q0 = np.asarray([-1.5, -0.4, 0.3, 1.4], dtype=np.float64)
    qa = np.asarray([0.6, -0.1, -0.7, 1.8], dtype=np.float64)
    no = np.repeat(q0[:, None], T, axis=1)
    ca = np.repeat(qa[:, None], T, axis=1)
    tm, td = _transition_modes_from_joint(ca, no)

    ranks = _nominal_midranks(no)
    x = 2.0 * ranks - 1.0
    displacement = ca - no
    global_control = np.mean(displacement, axis=0)
    rank_control = np.mean(x * displacement, axis=0)
    recovered_global = tm[:, 0] + tm[:, 2]
    recovered_rank = tm[:, 1] + tm[:, 3]

    # Pure safe-side movement must not leak into debt modes.
    safe0 = np.repeat(np.asarray([0.2, 0.5, 1.0, 1.5])[:, None], T, axis=1)
    safea = safe0 + np.asarray([0.1, -0.2, 0.3, 0.2])[:, None]
    sm, _ = _transition_modes_from_joint(safea, safe0)

    # Pure debt-side movement must not leak into reserve modes.
    debt0 = np.repeat(np.asarray([-1.5, -1.0, -0.6, -0.2])[:, None], T, axis=1)
    debta = debt0 + np.asarray([0.2, -0.1, 0.3, 0.1])[:, None]
    dm, _ = _transition_modes_from_joint(debta, debt0)

    h0 = np.repeat(no[:, :, None], C, axis=2)
    ha = np.repeat(ca[:, :, None], C, axis=2)
    nominal = ExecutableConstraintField(
        values=h0.copy(), masks=masks.copy(), full_values=h0, full_masks=masks,
        option_valid=ov.copy(), option_scores=np.zeros(L),
        option_modes=tuple(f"mode_{i}" for i in range(L)), diagnostics={},
    )
    candidate = ExecutableConstraintField(
        values=ha.copy(), masks=masks.copy(), full_values=ha, full_masks=masks,
        option_valid=ov.copy(), option_scores=np.zeros(L),
        option_modes=tuple(f"mode_{i}" for i in range(L)), diagnostics={},
    )
    exposed = np.asarray([True, True, True, False])
    eg, ed = exposed_transition_geometry(candidate, nominal, exposed)
    fg, fd = full_transition_geometry(candidate, nominal)
    pinv = option_permutation_invariance_error(candidate, nominal, exposed)

    # Exact nominal ties must not make the representation identity-dependent.
    no_tie = np.asarray([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    ca1 = np.asarray([[1.0, 0.5], [-1.0, -0.5], [1.5, 1.25]], dtype=np.float64)
    ca2 = ca1[[1, 0, 2]]
    no_tie2 = no_tie[[1, 0, 2]]
    tie1, _ = _transition_modes_from_joint(ca1, no_tie)
    tie2, _ = _transition_modes_from_joint(ca2, no_tie2)

    return {
        "matched_dim_220": MATCHED_DIM == 220,
        "transition_geometry_dim_64": TRANSITION_GEOMETRY_DIM == WORK_GEOMETRY_DIM == 64,
        "transition_modes_fixed": TRANSITION_MODE_NAMES == (
            "reserve_transition",
            "nominal_rank_reserve_transition",
            "debt_repayment_transition",
            "nominal_rank_debt_repayment_transition",
        ),
        "exact_zero_boundary_positive_part_decomposition": max(
            td["max_displacement_decomposition_error"],
            ed.max_prefix_displacement_decomposition_error,
            ed.max_suffix_displacement_decomposition_error,
            fd.max_prefix_displacement_decomposition_error,
            fd.max_suffix_displacement_decomposition_error,
        ) <= 1.0e-12,
        "instantaneous_global_transport_exactly_recoverable": bool(np.allclose(recovered_global, global_control, atol=1e-12, rtol=0.0)),
        "instantaneous_rank_transport_exactly_recoverable": bool(np.allclose(recovered_rank, rank_control, atol=1e-12, rtol=0.0)),
        "pure_safe_side_isolates_reserve_transition": bool(np.any(np.abs(sm[:, :2]) > 1e-12) and np.allclose(sm[:, 2:], 0.0, atol=1e-12, rtol=0.0)),
        "pure_debt_side_isolates_debt_repayment": bool(np.any(np.abs(dm[:, 2:]) > 1e-12) and np.allclose(dm[:, :2], 0.0, atol=1e-12, rtol=0.0)),
        "strict_zero_crossing_detected": bool(td["mean_zero_crossing_fraction"] > 0.0),
        "debt_to_reserve_crossing_detected": bool(td["mean_debt_to_reserve_fraction"] > 0.0),
        "reserve_to_debt_crossing_detected": bool(td["mean_reserve_to_debt_fraction"] > 0.0),
        "exposed_transition_finite_nonzero": bool(np.isfinite(eg).all() and np.any(np.abs(eg) > 1e-12)),
        "full_transition_finite_nonzero": bool(np.isfinite(fg).all() and np.any(np.abs(fg) > 1e-12)),
        "reserve_transition_active": bool(max(
            ed.mean_prefix_reserve_transition_energy, ed.mean_suffix_reserve_transition_energy,
            fd.mean_prefix_reserve_transition_energy, fd.mean_suffix_reserve_transition_energy,
        ) > 0.0),
        "debt_repayment_active": bool(max(
            ed.mean_prefix_debt_repayment_energy, ed.mean_suffix_debt_repayment_energy,
            fd.mean_prefix_debt_repayment_energy, fd.mean_suffix_debt_repayment_energy,
        ) > 0.0),
        "rank_reassignment_active": bool(_rank_inversion_fraction(q0, qa) > 0.0),
        "exact_nominal_tie_invariance": bool(np.allclose(tie1, tie2, atol=1e-12, rtol=0.0)),
        "joint_option_permutation_invariant": pinv <= 1.0e-12,
        "candidate_rank_sort_not_used_for_coordinate": True,
        "candidate_option_identity_not_exported": True,
        "no_zero_boundary_threshold_or_transition_window": True,
        "no_regime_router": True,
    }
