"""V48.94 OC-SRCA: support--reserve complementarity admission.

This module intentionally contains no learned parameter and no regime logic.
The state switch is shared by every candidate in one scene/time group and is
computed from the frozen nominal hard DRS coordinate of the native OC-MERO
certificate:

* no nominal shared-recovery support -> test whether a candidate establishes it;
* nominal support already present -> require the candidate to preserve support
  and remain on the non-negative deployability side of R_dep.

The two boundaries are historical semantic boundaries, not fitted thresholds:
hard DRS has exact zero as the support-existence frontier and
sigmoid(R_dep) has 0.5 as the already-frozen R_dep=0 frontier.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import math

ENGINEERING_VERSION = "v48.94.0-OC-SRCA"
SCHEMA = "ocrap-v48.94-support-reserve-admission-v1"


@dataclass(frozen=True)
class SupportReserveAdmission:
    state: str
    score: float
    passed: bool
    nominal_hard_drs: float
    candidate_hard_drs: float
    candidate_deployability: float
    candidate_gap_quality: float


def _certificate(x: Sequence[float], *, name: str) -> tuple[float, float, float, float]:
    if x is None or len(x) < 4:
        raise ValueError(f"{name} native certificate must contain 4 coordinates")
    vals = tuple(float(v) for v in x[:4])
    if not all(math.isfinite(v) for v in vals):
        raise ValueError(f"{name} native certificate contains non-finite values")
    if any(v < -1e-7 or v > 1.0 + 1e-7 for v in vals):
        raise ValueError(f"{name} native certificate must lie in [0,1]")
    return tuple(min(1.0, max(0.0, v)) for v in vals)  # type: ignore[return-value]


def support_reserve_admission(
    candidate_native_certificate: Sequence[float],
    nominal_native_certificate: Sequence[float],
    *,
    deployability_threshold: float = 0.5,
) -> SupportReserveAdmission:
    """Return the fixed OC-SRCA absolute source for one candidate.

    Coordinate convention (frozen since the native certificate was introduced):
      0 hard shared-recovery feasible root mass (DRS-like support),
      1 sigmoid(R_dep),
      2 smooth shared-feasible mass,
      3 exp(-gap) quality.

    ``candidate_gap_quality`` is exposed for diagnostics only.  V48.93 showed
    GAP is not a positive mediator, so it is deliberately not allowed to rescue
    or veto a candidate in this experiment.
    """
    if not math.isfinite(float(deployability_threshold)) or abs(float(deployability_threshold) - 0.5) > 1e-12:
        raise ValueError("OC-SRCA keeps the historical R_dep=0 boundary: threshold must be 0.5")
    c_drs, c_dep, _c_smooth, c_gap = _certificate(candidate_native_certificate, name="candidate")
    n_drs, _n_dep, _n_smooth, _n_gap = _certificate(nominal_native_certificate, name="nominal")

    # Exact zero is not a tuned threshold: native_hard_drs is a probability mass
    # over roots whose selected common recovery option is on the physical
    # non-negative side.  Zero therefore means no such support exists.
    support_exists = c_drs > 0.0
    if n_drs <= 0.0:
        state = "support_establishment"
        score = c_drs
        passed = support_exists
    else:
        state = "reserve_debt"
        # Non-compensatory: losing shared support cannot be compensated by a
        # positive deployability score.  The product is used only as a continuous
        # ordering score; the pass boundary remains the two exact predicates.
        score = c_drs * c_dep
        passed = support_exists and c_dep >= 0.5

    return SupportReserveAdmission(
        state=state,
        score=float(score),
        passed=bool(passed),
        nominal_hard_drs=float(n_drs),
        candidate_hard_drs=float(c_drs),
        candidate_deployability=float(c_dep),
        candidate_gap_quality=float(c_gap),
    )
