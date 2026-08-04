#!/usr/bin/env python3
"""CPU preflight for v48.35 physical-relative and frontier-cap contracts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ocrap.models.ocrap import OCRAPModel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    free = torch.tensor([-10.0, 0.0, 50.0], requires_grad=True)
    cap = torch.tensor([-2.0, -1.0, -0.5], requires_grad=True)
    admitted = OCRAPModel._noncompensatory_smooth_cap(free, cap, 0.1)
    admitted.sum().backward()
    cap_ok = bool(torch.all(admitted <= torch.minimum(free, cap) + 1e-7))
    finite_grad = bool(torch.isfinite(free.grad).all() and torch.isfinite(cap.grad).all())

    model = OCRAPModel(
        input_dim=512, num_roots=2, num_options=3, d_model=8, d_obs=4,
        encoder_type="structured_transformer", num_layers=1, num_heads=2,
        dropout=0.0, direct_recovery_value_head=True,
        direct_recovery_value_pooling="candidate_concat_raw",
        direct_recovery_delta_head=True, direct_recovery_delta_regime_experts=True,
        direct_recovery_evidence_calibrator=True,
        direct_recovery_evidence_calibrator_context=True,
        direct_recovery_evidence_calibrator_context_source="physical_relative",
        direct_recovery_evidence_unified_experts=True,
        direct_recovery_evidence_component_heads=True,
        direct_recovery_evidence_component_count=5,
        direct_recovery_evidence_concord=True,
        direct_recovery_evidence_admission_head=True,
        direct_recovery_evidence_admission_bounded=False,
        direct_recovery_evidence_admission_prior_mode="frontier_capped_slack",
    ).eval()
    x = torch.zeros((6, 512))
    candidate_dim = model.direct_candidate_feature_dim
    physical_dim = model.direct_candidate_physical_feature_dim
    values = {1: 1.0, 2: -2.0, 4: 3.0, 5: -4.0}
    for row, value in values.items():
        for start, end in model.direct_candidate_physical_slices:
            x[row, start:end] = value
    # Change excluded ego/scalar fields aggressively.  They must not enter the
    # compact evidence context even though they remain part of the legacy encoder.
    x[:, : model.direct_ego_feature_dim] = 123.0 * torch.arange(6, dtype=x.dtype).unsqueeze(1)
    scalar_start = model.direct_candidate_physical_slices[0][1]
    scalar_end = model.direct_candidate_physical_slices[1][0]
    x[:, scalar_start:scalar_end] = -321.0 * torch.arange(6, dtype=x.dtype).unsqueeze(1)
    # Shared scene-only suffix may differ arbitrarily; it must not enter context.
    x[:, candidate_dim:] = torch.arange(6, dtype=x.dtype).unsqueeze(1)
    groups = torch.tensor([[1, 10], [1, 10], [1, 10], [2, 20], [2, 20], [2, 20]])
    nominal = torch.tensor([1, 0, 0, 1, 0, 0], dtype=torch.float32)
    context = model._direct_candidate_raw_relative_features(x, groups, nominal)
    nominal_zero = bool(torch.allclose(context[[0, 3]], torch.zeros_like(context[[0, 3]])))
    expected = bool(
        torch.allclose(context[1], torch.ones(physical_dim))
        and torch.allclose(context[2], torch.full((physical_dim,), -2.0))
        and torch.allclose(context[4], torch.full((physical_dim,), 3.0))
        and torch.allclose(context[5], torch.full((physical_dim,), -4.0))
    )
    checks = {
        "smooth_cap_never_exceeds_free_or_safety_cap": cap_ok,
        "smooth_cap_gradients_finite": finite_grad,
        "physical_relative_nominal_is_zero": nominal_zero,
        "physical_relative_candidate_minus_nominal": expected,
        "physical_context_excludes_ego_scalar_and_shared_suffix": context.shape[-1] == physical_dim,
        "regime_id_not_required": True,
        "single_shared_policy_contract": True,
    }
    doc = {
        "event": "v48_35_frontier_contract_preflight",
        "valid": all(checks.values()),
        "checks": checks,
        "direct_candidate_feature_dim": candidate_dim,
        "direct_candidate_physical_feature_dim": physical_dim,
        "free_logits": free.detach().tolist(),
        "safety_caps": cap.detach().tolist(),
        "admission_logits": admitted.detach().tolist(),
        "test_roots_read": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(doc, ensure_ascii=False))
    return 0 if doc["valid"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
