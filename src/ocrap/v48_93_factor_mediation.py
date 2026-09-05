from __future__ import annotations

"""V48.93 OC-FMCA: exact factor-mediation complementarity adjudication.

This module is audit-only.  V48.92 showed that several PCD-derived quantities all
pass the same marginal safe-positive-vs-harmful screen.  Because those quantities
are algebraically coupled to the registered teacher advantage, marginal AUC alone
cannot identify a *unique* action-benefit mediator.  V48.93 therefore performs
exact counterfactual factor interventions on the three deployed PCD factors and
asks a stricter question: which candidate factor changes are necessary and/or
sufficient for the registered candidate-minus-nominal advantage?

No planner parameter, model input, dataset row, teacher label, OC-MERO quantity,
relative ranker, or deployment threshold is changed.
"""

from dataclasses import asdict, dataclass
from math import exp
from typing import Mapping

import numpy as np

ENGINEERING_VERSION = "v48.93.0-OC-FMCA"
SCHEMA = "ocrap-v48.93-factor-mediation-v1"
FACTOR_NAMES = ("drs", "deployability_gate", "gap_discount")
POSITIVE_GAIN = 0.015


def _clip01(x: float) -> float:
    return float(np.clip(float(x), 0.0, 1.0))


def pcd_product(factors: Mapping[str, float]) -> float:
    out = 1.0
    for name in FACTOR_NAMES:
        out *= _clip01(float(factors[name]))
    return float(np.clip(out, 0.0, 1.0))


def exact_factor_counterfactuals(
    nominal: Mapping[str, float],
    candidate: Mapping[str, float],
    *,
    positive_gain: float = POSITIVE_GAIN,
) -> dict[str, float | bool]:
    """Return exact one-factor knockout/single-change PCD counterfactuals.

    * ``knockout_j`` keeps all candidate factors except factor ``j``, which is
      reset to its nominal value.  A safe-positive row is ``necessary_j`` when
      that exact intervention removes the registered positive advantage.
    * ``single_j`` keeps the nominal point and changes only factor ``j`` to the
      candidate value.  It is ``sufficient_j`` when that change alone creates
      the registered positive advantage.

    This is not a learned attribution and does not use Shapley additivity.  It is
    an exact intervention on the production PCD product.
    """
    n = {k: _clip01(float(nominal[k])) for k in FACTOR_NAMES}
    c = {k: _clip01(float(candidate[k])) for k in FACTOR_NAMES}
    p_nom = pcd_product(n)
    p_cand = pcd_product(c)
    full = float(p_cand - p_nom)
    out: dict[str, float | bool] = {
        "nominal_pcd": p_nom,
        "candidate_pcd": p_cand,
        "full_advantage": full,
    }
    thr = float(positive_gain)
    for name in FACTOR_NAMES:
        ko = dict(c)
        ko[name] = n[name]
        sg = dict(n)
        sg[name] = c[name]
        ko_adv = float(pcd_product(ko) - p_nom)
        sg_adv = float(pcd_product(sg) - p_nom)
        out[f"knockout_{name}_advantage"] = ko_adv
        out[f"single_{name}_advantage"] = sg_adv
        out[f"necessary_{name}"] = bool(full >= thr and ko_adv < thr)
        out[f"sufficient_{name}"] = bool(sg_adv >= thr)
    return out


def mediation_mode(cf: Mapping[str, float | bool]) -> str:
    necessary = [name for name in FACTOR_NAMES if bool(cf[f"necessary_{name}"])]
    if len(necessary) == 1:
        return {
            "drs": "drs_activation",
            "deployability_gate": "deployability_gain",
            "gap_discount": "gap_gain",
        }[necessary[0]]
    if len(necessary) > 1:
        return "multi_factor_necessary"
    return "redundant_or_interaction"


@dataclass(frozen=True)
class FactorMediationRow:
    nominal_pcd: float
    candidate_pcd: float
    full_advantage: float
    knockout_drs_advantage: float
    knockout_deployability_gate_advantage: float
    knockout_gap_discount_advantage: float
    single_drs_advantage: float
    single_deployability_gate_advantage: float
    single_gap_discount_advantage: float
    necessary_drs: bool
    necessary_deployability_gate: bool
    necessary_gap_discount: bool
    sufficient_drs: bool
    sufficient_deployability_gate: bool
    sufficient_gap_discount: bool
    mediation_mode: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def adjudicate_factor_mediation(
    *,
    nominal_drs: float,
    candidate_drs: float,
    nominal_deployability_gate: float,
    candidate_deployability_gate: float,
    nominal_gap_discount: float,
    candidate_gap_discount: float,
    positive_gain: float = POSITIVE_GAIN,
) -> FactorMediationRow:
    n = {
        "drs": nominal_drs,
        "deployability_gate": nominal_deployability_gate,
        "gap_discount": nominal_gap_discount,
    }
    c = {
        "drs": candidate_drs,
        "deployability_gate": candidate_deployability_gate,
        "gap_discount": candidate_gap_discount,
    }
    x = exact_factor_counterfactuals(n, c, positive_gain=positive_gain)
    return FactorMediationRow(
        nominal_pcd=float(x["nominal_pcd"]),
        candidate_pcd=float(x["candidate_pcd"]),
        full_advantage=float(x["full_advantage"]),
        knockout_drs_advantage=float(x["knockout_drs_advantage"]),
        knockout_deployability_gate_advantage=float(x["knockout_deployability_gate_advantage"]),
        knockout_gap_discount_advantage=float(x["knockout_gap_discount_advantage"]),
        single_drs_advantage=float(x["single_drs_advantage"]),
        single_deployability_gate_advantage=float(x["single_deployability_gate_advantage"]),
        single_gap_discount_advantage=float(x["single_gap_discount_advantage"]),
        necessary_drs=bool(x["necessary_drs"]),
        necessary_deployability_gate=bool(x["necessary_deployability_gate"]),
        necessary_gap_discount=bool(x["necessary_gap_discount"]),
        sufficient_drs=bool(x["sufficient_drs"]),
        sufficient_deployability_gate=bool(x["sufficient_deployability_gate"]),
        sufficient_gap_discount=bool(x["sufficient_gap_discount"]),
        mediation_mode=mediation_mode(x),
    )
