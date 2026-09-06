"""V48.95 OC-NROA: native recovery observability audit primitives.

Audit-only utilities.  No planner parameters, no regime logic, no thresholds are
introduced into the deployed planner.  The purpose is to decide whether the
frozen native OC-MERO certificate exposes the support-state and action-response
information required by the V48.93 support/reserve complementarity result.
"""
from __future__ import annotations

from typing import Iterable
import math

ENGINEERING_VERSION = "v48.95.0-OC-NROA"
SCHEMA = "ocrap-v48.95-native-recovery-observability-v1"


def tie_auc(positive: Iterable[float], negative: Iterable[float]) -> float | None:
    p = [float(x) for x in positive if math.isfinite(float(x))]
    n = [float(x) for x in negative if math.isfinite(float(x))]
    if not p or not n:
        return None
    wins = 0.0
    for x in p:
        for y in n:
            wins += 1.0 if x > y else 0.5 if x == y else 0.0
    return wins / float(len(p) * len(n))


def cert4(x: object, *, name: str) -> tuple[float, float, float, float]:
    if not isinstance(x, list) or len(x) < 4:
        raise ValueError(f"{name} must contain >=4 native-certificate coordinates")
    vals = tuple(float(v) for v in x[:4])
    if not all(math.isfinite(v) for v in vals):
        raise ValueError(f"{name} contains non-finite values")
    if any(v < -1e-7 or v > 1.0 + 1e-7 for v in vals):
        raise ValueError(f"{name} must lie in [0,1]")
    return tuple(min(1.0, max(0.0, v)) for v in vals)  # type: ignore[return-value]


def frozen_native_features(candidate: object, nominal: object) -> dict[str, float]:
    """Return only already-frozen observation-derived certificate coordinates."""
    ch, cd, cs, cg = cert4(candidate, name="candidate")
    nh, nd, ns, ng = cert4(nominal, name="nominal")
    return {
        "nominal_hard_support": nh,
        "nominal_deployability": nd,
        "nominal_smooth_support": ns,
        "nominal_gap_quality": ng,
        "candidate_hard_support": ch,
        "candidate_deployability": cd,
        "candidate_smooth_support": cs,
        "candidate_gap_quality": cg,
        "delta_hard_support": ch - nh,
        "delta_deployability": cd - nd,
        "delta_smooth_support": cs - ns,
        "delta_gap_quality": cg - ng,
    }
