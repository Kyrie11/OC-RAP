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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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


def _cell_preimage(y: float, bits: int) -> tuple[float, float, bool]:
    if bits & STRUCT_HIDDEN_BRANCH:
        return -_BOUND, _BOUND, False
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
    return float(lo), float(hi), bool(exact)


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
    if bits.shape != M.shape:
        raise ValueError("structural bitmask shape mismatch")

    lower = np.empty_like(M)
    upper = np.empty_like(M)
    exact_cells = 0
    structural_cells = 0
    inactive_structural_cells = 0
    for k in range(K):
        for l in range(L):
            b = int(bits[k, l])
            lo, hi, exact = _cell_preimage(float(M[k, l]), b)
            lower[k, l], upper[k, l] = lo, hi
            exact_cells += int(exact)
            if b:
                structural_cells += 1
                inactive_structural_cells += int(exact)

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
    )
