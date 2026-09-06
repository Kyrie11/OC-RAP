from __future__ import annotations

"""V48.96 OC-SRROA: support/reserve root-observability audit helpers.

Audit-only.  This module does not alter planner execution.  It derives the
V48.93 support-establishment / deployability-gain semantics from the historical
teacher PCD index and exposes deterministic helpers used by the frozen Stage-I
root-token probe.
"""

from math import exp
from typing import Any, Mapping
import copy

import numpy as np
import torch

from ocrap.v48_93_factor_mediation import POSITIVE_GAIN, adjudicate_factor_mediation

ENGINEERING_VERSION = "v48.96.1-OC-SRROA-ENGFIX"
SCHEMA = "ocrap-v48.96-support-reserve-root-observability-v1"
DEPLOYABLE_MACROS = frozenset({2, 3, 5, 6, 7})
VALID_MODES = frozenset({"drs_activation", "deployability_gain"})


def sigmoid(x: float) -> float:
    z = float(x)
    if z >= 0:
        e = float(np.exp(-z))
        return 1.0 / (1.0 + e)
    e = float(np.exp(z))
    return e / (1.0 + e)


def teacher_factor_tuple(row: Mapping[str, Any]) -> dict[str, float]:
    return {
        "drs": float(np.clip(float(row["teacher_drs"]), 0.0, 1.0)),
        "deployability_gate": sigmoid(float(row["teacher_r_dep"])),
        "gap_discount": float(np.clip(exp(-max(0.0, float(row["teacher_gap"]))), 0.0, 1.0)),
    }


def derive_candidate_semantics(
    nominal: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    positive_gain: float = POSITIVE_GAIN,
) -> dict[str, Any]:
    n = teacher_factor_tuple(nominal)
    c = teacher_factor_tuple(candidate)
    med = adjudicate_factor_mediation(
        nominal_drs=n["drs"],
        candidate_drs=c["drs"],
        nominal_deployability_gate=n["deployability_gate"],
        candidate_deployability_gate=c["deployability_gate"],
        nominal_gap_discount=n["gap_discount"],
        candidate_gap_discount=c["gap_discount"],
        positive_gain=float(positive_gain),
    )
    teacher_adv = float(candidate["teacher_pcd"]) - float(nominal["teacher_pcd"])
    if abs(float(med.full_advantage) - teacher_adv) > 2.0e-6:
        raise ValueError(f"PCD advantage mismatch: {med.full_advantage} vs {teacher_adv}")
    harmful = bool(candidate.get("component_harmful", False))
    safe_positive = bool(teacher_adv >= float(positive_gain) and not harmful)
    mode = med.mediation_mode if safe_positive else None
    return {
        "teacher_adv": teacher_adv,
        "teacher_harmful": harmful,
        "safe_positive": safe_positive,
        "mediation_mode": mode,
        "nominal_drs": n["drs"],
        "candidate_drs": c["drs"],
        "nominal_deployability_gate": n["deployability_gate"],
        "candidate_deployability_gate": c["deployability_gate"],
    }


FEATURE_ONLY_TRUTH_CONTRACT = "legacy_full"
FEATURE_ONLY_SUPERVISION_OBJECTIVE = "binary_sign"


def feature_only_dataset_cfg(
    cfg: Mapping[str, Any],
    *,
    cache_dir: str,
    workers: int = 8,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a dataset config for frozen-feature extraction only.

    Checkpoints such as L80 intentionally serialize *training-only* truth-sidecar
    contracts (for example ``structural_interval_bounds`` plus a train/dev truth
    index).  A representation audit must not ask ``OCRAPSampleDataset`` to attach
    those supervision records when reading a different registered split such as
    the certificate pool.  Doing so both causes a false fail-closed error and,
    more importantly, couples feature extraction to a supervision artifact that
    is not an input to the frozen planner.

    Only supervision-sidecar keys are neutralized.  Model/feature geometry and
    every input-affecting model flag remain untouched.
    """
    out = copy.deepcopy(dict(cfg))
    training = out.setdefault("training", {})
    if not isinstance(training, dict):
        raise ValueError("checkpoint training config must be a mapping")
    before = {
        "truth_contract": str(training.get("direct_value_absolute_feasibility_truth_contract", "legacy_full")),
        "truth_index": str(training.get("direct_value_absolute_feasibility_truth_index", "") or ""),
        "supervision_objective": str(training.get("direct_value_absolute_feasibility_supervision_objective", "binary_sign")),
        "action_response_truth_index": str(training.get("direct_value_action_response_truth_index", "") or ""),
    }
    training["direct_value_absolute_feasibility_truth_contract"] = FEATURE_ONLY_TRUTH_CONTRACT
    training["direct_value_absolute_feasibility_truth_index"] = ""
    training["direct_value_absolute_feasibility_supervision_objective"] = FEATURE_ONLY_SUPERVISION_OBJECTIVE
    training["direct_value_action_response_truth_index"] = ""
    training["persistent_tensor_cache"] = True
    training["persistent_tensor_cache_dir"] = str(cache_dir)
    training["persistent_tensor_cache_build_workers"] = max(1, int(workers))
    event = {
        "feature_only_dataset": True,
        "checkpoint_supervision": before,
        "effective_truth_contract": FEATURE_ONLY_TRUTH_CONTRACT,
        "effective_supervision_objective": FEATURE_ONLY_SUPERVISION_OBJECTIVE,
        "truth_sidecars_attached": False,
    }
    return out, event


def _weighted_root_stats(
    values: torch.Tensor,
    weights: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Permutation-invariant probability-weighted root distribution statistics."""
    w = weights.float() * valid.float()
    w = w / w.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
    values = values.float()
    mean = (w.unsqueeze(-1) * values).sum(dim=1)
    var = (w.unsqueeze(-1) * (values - mean.unsqueeze(1)).pow(2)).sum(dim=1)
    std = var.clamp_min(0.0).sqrt()
    inf = torch.tensor(float("inf"), device=values.device, dtype=values.dtype)
    ninf = torch.tensor(float("-inf"), device=values.device, dtype=values.dtype)
    vmax = torch.where(valid.unsqueeze(-1), values, ninf).amax(dim=1)
    vmin = torch.where(valid.unsqueeze(-1), values, inf).amin(dim=1)
    any_valid = valid.any(dim=1, keepdim=True)
    vmax = torch.where(any_valid, vmax, torch.zeros_like(vmax))
    vmin = torch.where(any_valid, vmin, torch.zeros_like(vmin))
    return torch.cat([mean, std, vmax, vmin], dim=-1)


def root_observability_features(
    root_tokens: torch.Tensor,
    root_probs: torch.Tensor,
    root_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build state/delta/context without assuming root-slot correspondence.

    Row 0 is the nominal action and rows 1.. are candidates.  Each action's
    latent-root distribution is summarized *independently* with its own root
    probabilities and valid mask.  The action response is the difference of
    these permutation-invariant distribution summaries.  This respects the
    V48.90 conclusion that action-dependent root partitions must not be treated
    as a slot-wise bijection.
    """
    if root_tokens.ndim != 3 or root_probs.ndim != 2 or root_valid.ndim != 2:
        raise ValueError("expected root_tokens[B,K,D], root_probs[B,K], root_valid[B,K]")
    if root_tokens.shape[:2] != root_probs.shape or root_probs.shape != root_valid.shape:
        raise ValueError("root token/probability/valid shapes are inconsistent")
    if root_tokens.shape[0] < 2:
        raise ValueError("root observability requires one nominal and at least one candidate")
    stats = _weighted_root_stats(root_tokens, root_probs, root_valid.bool())
    nominal = stats[0:1]
    candidate = stats[1:]
    delta = candidate - nominal
    state = nominal.expand(candidate.shape[0], -1)
    context = delta * (1.0 + torch.tanh(state))
    return state, delta, context
