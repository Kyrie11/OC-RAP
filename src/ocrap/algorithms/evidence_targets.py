"""Teacher targets for factorized recovery admission evidence.

The admission benefit target is the signed OC-RAP deployability-score advantage.
The harm target is deliberately non-compensatory: a recovery candidate is vetoed
when any safety-relevant component degrades beyond its tolerance relative to the
nominal prefix, even when other components improve enough to yield positive total
PCD advantage.  Keeping this logic in one module prevents train/certificate drift.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class ComponentVetoTolerances:
    """Nominal-relative tolerances in normalized component space."""

    drs: float = 0.05
    deployability_gate: float = 0.05
    gap_discount: float = 0.05
    hard_violation: float = 0.05
    harm_proxy: float = 0.05


def _sigmoid_np(x: float | np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    a = np.clip(a, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-a))


def component_veto_terms_numpy(
    *,
    candidate_drs: float,
    nominal_drs: float,
    candidate_r_dep: float,
    nominal_r_dep: float,
    candidate_gap: float,
    nominal_gap: float,
    candidate_hard: float = 0.0,
    nominal_hard: float = 0.0,
    candidate_harm_proxy: float = 0.0,
    nominal_harm_proxy: float = 0.0,
    tolerances: ComponentVetoTolerances | None = None,
) -> np.ndarray:
    """Return ordered, non-compensatory component degradation margins.

    The component order is ``DRS, deployability, gap, hard, harm_proxy``.
    Keeping the vector available lets a shared evidence model supervise the
    active risk mechanisms directly instead of asking one scalar tail to infer
    which physical component failed.
    """

    t = tolerances or ComponentVetoTolerances()
    candidate_dep = float(_sigmoid_np(candidate_r_dep))
    nominal_dep = float(_sigmoid_np(nominal_r_dep))
    candidate_gap_quality = math.exp(-max(0.0, min(float(candidate_gap), 20.0)))
    nominal_gap_quality = math.exp(-max(0.0, min(float(nominal_gap), 20.0)))
    return np.asarray(
        [
            float(nominal_drs) - float(candidate_drs) - float(t.drs),
            nominal_dep - candidate_dep - float(t.deployability_gate),
            nominal_gap_quality - candidate_gap_quality - float(t.gap_discount),
            float(candidate_hard) - float(nominal_hard) - float(t.hard_violation),
            float(candidate_harm_proxy) - float(nominal_harm_proxy) - float(t.harm_proxy),
        ],
        dtype=np.float64,
    )


def component_veto_margin_numpy(
    *,
    candidate_drs: float,
    nominal_drs: float,
    candidate_r_dep: float,
    nominal_r_dep: float,
    candidate_gap: float,
    nominal_gap: float,
    candidate_hard: float = 0.0,
    nominal_hard: float = 0.0,
    candidate_harm_proxy: float = 0.0,
    nominal_harm_proxy: float = 0.0,
    tolerances: ComponentVetoTolerances | None = None,
) -> float:
    """Return the largest normalized safety-component degradation.

    Positive values indicate that at least one component exceeds its allowed
    nominal-relative degradation.  The maximum implements a non-compensatory
    veto: an improvement in one component cannot hide a regression in another.
    """

    terms = component_veto_terms_numpy(
        candidate_drs=candidate_drs,
        nominal_drs=nominal_drs,
        candidate_r_dep=candidate_r_dep,
        nominal_r_dep=nominal_r_dep,
        candidate_gap=candidate_gap,
        nominal_gap=nominal_gap,
        candidate_hard=candidate_hard,
        nominal_hard=nominal_hard,
        candidate_harm_proxy=candidate_harm_proxy,
        nominal_harm_proxy=nominal_harm_proxy,
        tolerances=tolerances,
    )
    return float(np.max(terms))


def component_veto_terms_torch(
    *,
    candidate_drs: torch.Tensor,
    nominal_drs: torch.Tensor,
    candidate_r_dep: torch.Tensor,
    nominal_r_dep: torch.Tensor,
    candidate_gap: torch.Tensor,
    nominal_gap: torch.Tensor,
    candidate_hard: torch.Tensor | None = None,
    nominal_hard: torch.Tensor | None = None,
    candidate_harm_proxy: torch.Tensor | None = None,
    nominal_harm_proxy: torch.Tensor | None = None,
    tolerances: ComponentVetoTolerances | None = None,
) -> torch.Tensor:
    """Torch component margins with final dimension ordered as in NumPy."""

    t = tolerances or ComponentVetoTolerances()
    zeros = torch.zeros_like(candidate_drs)
    candidate_hard = zeros if candidate_hard is None else candidate_hard
    nominal_hard = torch.zeros_like(candidate_hard) if nominal_hard is None else nominal_hard
    candidate_harm_proxy = zeros if candidate_harm_proxy is None else candidate_harm_proxy
    nominal_harm_proxy = (
        torch.zeros_like(candidate_harm_proxy)
        if nominal_harm_proxy is None
        else nominal_harm_proxy
    )
    candidate_gap_quality = torch.exp(-candidate_gap.float().clamp(0.0, 20.0))
    nominal_gap_quality = torch.exp(-nominal_gap.float().clamp(0.0, 20.0))
    return torch.stack(
        [
            nominal_drs.float() - candidate_drs.float() - float(t.drs),
            torch.sigmoid(nominal_r_dep.float())
            - torch.sigmoid(candidate_r_dep.float())
            - float(t.deployability_gate),
            nominal_gap_quality - candidate_gap_quality - float(t.gap_discount),
            candidate_hard.float() - nominal_hard.float() - float(t.hard_violation),
            candidate_harm_proxy.float()
            - nominal_harm_proxy.float()
            - float(t.harm_proxy),
        ],
        dim=-1,
    )


def component_veto_margin_torch(
    *,
    candidate_drs: torch.Tensor,
    nominal_drs: torch.Tensor,
    candidate_r_dep: torch.Tensor,
    nominal_r_dep: torch.Tensor,
    candidate_gap: torch.Tensor,
    nominal_gap: torch.Tensor,
    candidate_hard: torch.Tensor | None = None,
    nominal_hard: torch.Tensor | None = None,
    candidate_harm_proxy: torch.Tensor | None = None,
    nominal_harm_proxy: torch.Tensor | None = None,
    tolerances: ComponentVetoTolerances | None = None,
) -> torch.Tensor:
    """Torch analogue of :func:`component_veto_margin_numpy`."""

    terms = component_veto_terms_torch(
        candidate_drs=candidate_drs,
        nominal_drs=nominal_drs,
        candidate_r_dep=candidate_r_dep,
        nominal_r_dep=nominal_r_dep,
        candidate_gap=candidate_gap,
        nominal_gap=nominal_gap,
        candidate_hard=candidate_hard,
        nominal_hard=nominal_hard,
        candidate_harm_proxy=candidate_harm_proxy,
        nominal_harm_proxy=nominal_harm_proxy,
        tolerances=tolerances,
    )
    return terms.max(dim=-1).values


def component_veto_soft_target(margin: torch.Tensor, *, temperature: float = 0.05) -> torch.Tensor:
    """Map a component-veto margin to a stable soft binary target."""

    return torch.sigmoid(margin / max(float(temperature), 1.0e-4))
