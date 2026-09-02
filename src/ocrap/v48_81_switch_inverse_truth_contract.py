from __future__ import annotations

"""V48.81 switch-aware inverse structural truth contract.

V48.80 propagated valid but deliberately loose one-sided bounds whenever a
root-option cell *could* be touched by a structural floor/cap.  This module
uses the exact monotone operator order implemented by ``teacher_margin`` and
inverts the observed stored cell value.  If a floor/cap is inactive, the
pre-structural physical value is point-identified; if it is active, only the
mathematically valid one-sided preimage is retained.  Hidden/artifact hard
replacements remain unidentifiable.

No teacher metadata is exposed to the model.  The result is a training-only
interval sidecar and does not modify the dataset or teacher labels.
"""

from dataclasses import asdict, dataclass
from typing import Any
import numpy as np

from ocrap.algorithms.ocmero import oc_mero
from ocrap.v48_79_truth_contract import (
    STRUCT_HIDDEN_BRANCH,
    STRUCT_RECOVERY_FLOOR,
    STRUCT_ROUTE_OVERRIDE,
    STRUCT_SECONDARY_FLOOR,
    nested_tail_truth_contract,
    structural_root_option_reason_bits,
)

_BOUND = 1.0e6
_RECOVERY_FLOOR = 0.6
_ROUTE_CAP = -0.8
_SECONDARY_FLOOR = 0.9
_TOL = 1.0e-6


@dataclass(frozen=True)
class SwitchInverseTruthContractRecord:
    valid: bool
    physical_lower: float
    physical_upper: float
    exact_physical: bool
    informative: bool
    interval_width: float | None
    lower_finite: bool
    upper_finite: bool
    structural_exposure_mass: float
    r_dep_stored: float
    r_dep_recomputed: float
    r_dep_abs_error: float
    alpha: float
    beta: float
    top_m: int
    exact_cell_fraction: float
    inactive_structural_cell_fraction: float
    mixed_structural_cell_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)




def structural_root_option_reason_profile(sample: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return (any_bits, all_bits) for structural rules over futures in each root.

    ``m_star`` is an *intra-root aggregate* of future-option margins.  Therefore
    a rule that is active for only some futures in a root cannot be inverted from
    the aggregated scalar.  ``any_bits`` records potential exposure, while
    ``all_bits`` records rules that apply uniformly to every assigned future for
    that root-option cell.  Exact switch inversion is only valid for uniform
    rules.  Mixed exposure is fail-closed as unidentifiable.
    """
    m = np.asarray(sample.get("m_star", np.zeros((0, 0))), dtype=np.float64)
    if m.ndim != 2:
        raise ValueError("m_star must be [K,L]")
    K, L = m.shape
    assignments = np.asarray(sample.get("root_assignments", []), dtype=np.int64).reshape(-1)
    metas = sample.get("future_metadata", [])
    # Reuse the V48.79 parser through its public bit helper for ANY exposure,
    # but reconstruct per-future bits here to compute the uniform-ALL mask.
    from ocrap.v48_79_truth_contract import _json_scalar, _string_list, _HIDDEN_VALUES, _FLOORED_MODES
    metas = _json_scalar(metas, [])
    if not isinstance(metas, list):
        metas = []
    modes = _string_list(sample.get("recovery_modes", []))
    if len(modes) < L:
        modes = modes + [""] * (L - len(modes))
    any_bits = np.zeros((K, L), dtype=np.int64)
    all_bits = np.zeros((K, L), dtype=np.int64)
    n = min(len(metas), assignments.size)
    for k in range(K):
        fis = [fi for fi in range(n) if int(assignments[fi]) == k]
        if not fis:
            continue
        for l in range(L):
            vals = []
            mode = modes[l]
            for fi in fis:
                meta = metas[fi] if isinstance(metas[fi], dict) else {}
                hidden = meta.get("artifact_branch") in _HIDDEN_VALUES or meta.get("hidden_intent") in _HIDDEN_VALUES
                b = 0
                if hidden:
                    b |= STRUCT_HIDDEN_BRANCH
                if (not hidden) and mode in _FLOORED_MODES:
                    b |= STRUCT_RECOVERY_FLOOR
                if bool(meta.get("route_blocked", False)) and mode == "yield_rejoin":
                    b |= STRUCT_ROUTE_OVERRIDE
                if bool(meta.get("secondary_threat", False)) and mode == "avoid_secondary":
                    b |= STRUCT_SECONDARY_FLOOR
                vals.append(int(b))
            a = 0
            for b in vals:
                a |= b
            c = vals[0]
            for b in vals[1:]:
                c &= b
            any_bits[k, l] = a
            all_bits[k, l] = c
    return any_bits, all_bits

def _inverse_floor(lo: float, hi: float, c: float) -> tuple[float, float]:
    """Preimage of [lo,hi] through z=max(x,c)."""
    if hi < c - _TOL:
        raise ValueError(f"floor inverse empty: [{lo},{hi}] below {c}")
    if lo > c + _TOL:
        return lo, hi
    return -_BOUND, hi


def _inverse_cap(lo: float, hi: float, c: float) -> tuple[float, float]:
    """Preimage of [lo,hi] through z=min(x,c)."""
    if lo > c + _TOL:
        raise ValueError(f"cap inverse empty: [{lo},{hi}] above {c}")
    if hi < c - _TOL:
        return lo, hi
    return lo, _BOUND


def _cell_preimage(y: float, any_bits: int, all_bits: int) -> tuple[float, float, bool, bool]:
    # A root-level m_star cell is an intra-root LCVAR aggregate.  If a
    # structural rule applies to only a subset of futures in that root, its
    # inverse cannot be recovered from the aggregate scalar without the
    # future-level margins.  Treat such mixed exposure as unidentifiable.
    mixed = bool(any_bits != all_bits)
    if (any_bits & STRUCT_HIDDEN_BRANCH) or mixed:
        return -_BOUND, _BOUND, False, mixed
    bits = int(all_bits)
    lo = hi = float(y)
    # Reverse the exact forward order in simulation.teacher.margins.teacher_margin:
    # recovery floor -> route cap -> secondary floor.
    if bits & STRUCT_SECONDARY_FLOOR:
        lo, hi = _inverse_floor(lo, hi, _SECONDARY_FLOOR)
    if bits & STRUCT_ROUTE_OVERRIDE:
        lo, hi = _inverse_cap(lo, hi, _ROUTE_CAP)
    if bits & STRUCT_RECOVERY_FLOOR:
        lo, hi = _inverse_floor(lo, hi, _RECOVERY_FLOOR)
    exact = (lo > -0.5 * _BOUND) and (hi < 0.5 * _BOUND) and abs(hi - lo) <= 1e-10
    return float(lo), float(hi), bool(exact), mixed


def nested_tail_switch_inverse_interval(
    sample: dict[str, Any], *, alpha: float = 0.2, beta: float = 0.2,
    top_m: int = 8, recompute_tolerance: float = 1.0e-5,
) -> SwitchInverseTruthContractRecord:
    base = nested_tail_truth_contract(
        sample, alpha=alpha, beta=beta, top_m=top_m,
        recompute_tolerance=recompute_tolerance,
    )
    M = np.asarray(sample.get("m_star"), dtype=np.float64)
    if M.ndim != 2 or M.size == 0:
        raise ValueError("m_star must be a non-empty [K,L] matrix")
    K, L = M.shape
    bits = structural_root_option_reason_bits(sample)
    any_bits, all_bits = structural_root_option_reason_profile(sample)
    if bits.shape != M.shape or any_bits.shape != M.shape or all_bits.shape != M.shape:
        raise ValueError("structural bitmask shape mismatch")
    if not np.array_equal(bits, any_bits):
        raise ValueError("V48.79 ANY structural exposure profile mismatch")

    lower = np.empty_like(M)
    upper = np.empty_like(M)
    exact_cells = 0
    structural_cells = 0
    inactive_structural_cells = 0
    mixed_structural_cells = 0
    for k in range(K):
        for l in range(L):
            b = int(bits[k, l])
            lo, hi, exact, mixed = _cell_preimage(float(M[k, l]), int(any_bits[k,l]), int(all_bits[k,l]))
            lower[k, l], upper[k, l] = lo, hi
            exact_cells += int(exact)
            if b:
                structural_cells += 1
                inactive_structural_cells += int(exact)
                mixed_structural_cells += int(mixed)

    p = np.asarray(sample.get("root_probs", np.zeros(K)), dtype=np.float64).reshape(-1)[:K]
    rv = np.asarray(sample.get("root_valid", np.ones(K)), dtype=bool).reshape(-1)[:K]
    C = np.asarray(sample.get("c_star", np.eye(K)), dtype=np.float64)
    ov = np.asarray(sample.get("option_valid", np.ones(L)), dtype=bool).reshape(-1)[:L]
    kwargs = dict(alpha=float(alpha), beta=float(beta), option_valid=ov,
                  root_valid=rv, use_lcvar=True, use_obs_kernel=True, top_m=int(top_m))
    lo = float(oc_mero(lower, p, C, **kwargs).r_dep)
    hi = float(oc_mero(upper, p, C, **kwargs).r_dep)
    if lo > hi + 1e-6:
        raise ValueError(f"invalid propagated interval: lower={lo} upper={hi}")
    lower_finite = lo > -0.5 * _BOUND
    upper_finite = hi < +0.5 * _BOUND
    exact = bool(lower_finite and upper_finite and abs(hi - lo) <= 1e-10)
    informative = bool(lower_finite or upper_finite)
    width = float(hi - lo) if lower_finite and upper_finite else None
    return SwitchInverseTruthContractRecord(
        valid=bool(base.valid), physical_lower=lo, physical_upper=hi,
        exact_physical=exact, informative=informative, interval_width=width,
        lower_finite=lower_finite, upper_finite=upper_finite,
        structural_exposure_mass=float(base.structural_exposure_mass),
        r_dep_stored=float(base.r_dep_stored), r_dep_recomputed=float(base.r_dep_recomputed),
        r_dep_abs_error=float(base.r_dep_abs_error), alpha=float(alpha), beta=float(beta), top_m=int(top_m),
        exact_cell_fraction=float(exact_cells / max(1, K * L)),
        inactive_structural_cell_fraction=float(inactive_structural_cells / max(1, structural_cells)),
        mixed_structural_cell_fraction=float(mixed_structural_cells / max(1, structural_cells)),
    )
