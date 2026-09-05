from __future__ import annotations

"""V48.92 OC-FRAD: exact factorization of the registered recovery advantage.

This module is intentionally audit-only.  It introduces no planner parameter and
never changes OC-MERO, the teacher labels, candidate proposals, or deployment
logic.  The only learned-object question it answers is *what decision-semantic
factor currently carries the registered candidate-vs-nominal PCD advantage*.
"""

from dataclasses import asdict, dataclass
from itertools import combinations
from math import exp, factorial
from typing import Mapping

import numpy as np

ENGINEERING_VERSION = "v48.92.0-OC-FRAD"
SCHEMA = "ocrap-v48.92-factorized-recovery-advantage-v1"
FACTOR_NAMES = ("drs", "deployability_gate", "gap_discount")


def _sigmoid(x: float) -> float:
    x = float(x)
    if x >= 0.0:
        z = exp(-x)
        return 1.0 / (1.0 + z)
    z = exp(x)
    return z / (1.0 + z)


def pcd_factors(*, drs: float, r_dep: float, gap: float) -> dict[str, float]:
    """Return the three multiplicative PCD factors in their deployed units."""
    d = float(np.clip(float(drs), 0.0, 1.0))
    dep = _sigmoid(float(r_dep))
    gd = exp(-float(np.clip(float(gap), 0.0, 20.0)))
    return {"drs": d, "deployability_gate": dep, "gap_discount": gd}


def pcd_from_factors(factors: Mapping[str, float]) -> float:
    out = 1.0
    for name in FACTOR_NAMES:
        out *= float(factors[name])
    return float(np.clip(out, 0.0, 1.0))


def shapley_product_delta(
    nominal: Mapping[str, float], candidate: Mapping[str, float]
) -> dict[str, float]:
    """Exact Shapley decomposition of candidate-minus-nominal PCD.

    The decomposition is parameter-free and additive in the *same numerical
    coordinate as teacher_adv*.  Averaging over all factor replacement orders
    avoids choosing an arbitrary causal/serialization order for the product.
    """
    n = len(FACTOR_NAMES)
    contrib = {name: 0.0 for name in FACTOR_NAMES}
    names = list(FACTOR_NAMES)
    for j, name in enumerate(names):
        others = [x for x in names if x != name]
        for r in range(len(others) + 1):
            weight = factorial(r) * factorial(n - r - 1) / factorial(n)
            for subset in combinations(others, r):
                s = set(subset)
                base = {k: float(candidate[k]) if k in s else float(nominal[k]) for k in names}
                with_j = dict(base)
                with_j[name] = float(candidate[name])
                contrib[name] += weight * (pcd_from_factors(with_j) - pcd_from_factors(base))
    return {k: float(v) for k, v in contrib.items()}


@dataclass(frozen=True)
class FactorizedRecoveryAdvantage:
    teacher_adv_reconstructed: float
    candidate_pcd: float
    nominal_pcd: float
    candidate_drs: float
    nominal_drs: float
    candidate_r_dep: float
    nominal_r_dep: float
    candidate_gap: float
    nominal_gap: float
    candidate_deployability_gate: float
    nominal_deployability_gate: float
    candidate_gap_discount: float
    nominal_gap_discount: float
    delta_drs: float
    delta_r_dep: float
    delta_gap: float
    shapley_drs: float
    shapley_deployability_gate: float
    shapley_gap_discount: float
    shapley_sum_error: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def factorize_recovery_advantage(
    *,
    candidate_drs: float,
    nominal_drs: float,
    candidate_r_dep: float,
    nominal_r_dep: float,
    candidate_gap: float,
    nominal_gap: float,
) -> FactorizedRecoveryAdvantage:
    cand = pcd_factors(drs=candidate_drs, r_dep=candidate_r_dep, gap=candidate_gap)
    nom = pcd_factors(drs=nominal_drs, r_dep=nominal_r_dep, gap=nominal_gap)
    cpcd = pcd_from_factors(cand)
    npcd = pcd_from_factors(nom)
    adv = float(cpcd - npcd)
    phi = shapley_product_delta(nom, cand)
    err = abs(float(sum(phi.values()) - adv))
    return FactorizedRecoveryAdvantage(
        teacher_adv_reconstructed=adv,
        candidate_pcd=cpcd,
        nominal_pcd=npcd,
        candidate_drs=float(candidate_drs),
        nominal_drs=float(nominal_drs),
        candidate_r_dep=float(candidate_r_dep),
        nominal_r_dep=float(nominal_r_dep),
        candidate_gap=float(candidate_gap),
        nominal_gap=float(nominal_gap),
        candidate_deployability_gate=float(cand["deployability_gate"]),
        nominal_deployability_gate=float(nom["deployability_gate"]),
        candidate_gap_discount=float(cand["gap_discount"]),
        nominal_gap_discount=float(nom["gap_discount"]),
        delta_drs=float(candidate_drs - nominal_drs),
        delta_r_dep=float(candidate_r_dep - nominal_r_dep),
        delta_gap=float(candidate_gap - nominal_gap),
        shapley_drs=float(phi["drs"]),
        shapley_deployability_gate=float(phi["deployability_gate"]),
        shapley_gap_discount=float(phi["gap_discount"]),
        shapley_sum_error=float(err),
    )
