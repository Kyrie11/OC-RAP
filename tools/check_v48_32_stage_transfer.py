#!/usr/bin/env python3
"""Verify v48.32 factor -> identity -> calibration parameter transfer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

IDENTITY_ALLOWED_PREFIXES = (
    "direct_evidence_concord_benefit_calibrator.",
    "direct_evidence_concord_harm_calibrator.",
    "direct_evidence_concord_admission_calibrator.",
)
FINAL_ALLOWED_PREFIXES = ("direct_evidence_concord_admission_calibrator.",)


def _load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _compare(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
    allowed: tuple[str, ...],
) -> tuple[list[str], list[tuple[str, float]], list[str]]:
    changed_allowed: list[str] = []
    changed_disallowed: list[tuple[str, float]] = []
    missing: list[str] = []
    for key in sorted(set(before) | set(after)):
        if key not in after:
            missing.append(key)
            continue
        if key not in before:
            if key.startswith(allowed):
                changed_allowed.append(key)
            else:
                changed_disallowed.append((key, float("inf")))
            continue
        b=before[key]; a=after[key]
        if tuple(b.shape) != tuple(a.shape):
            diff=float("inf")
        else:
            diff = float((b.detach().float() - a.detach().float()).abs().max().item())
        if diff <= 0.0:
            continue
        if key.startswith(allowed):
            changed_allowed.append(key)
        else:
            changed_disallowed.append((key, diff))
    return changed_allowed, changed_disallowed, missing


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--factor", type=Path, required=True)
    ap.add_argument("--identity", type=Path, required=True)
    ap.add_argument("--final", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--final-stage-disabled", action="store_true")
    args = ap.parse_args()

    factor = _load(args.factor)["model_state"]
    identity = _load(args.identity)["model_state"]
    final = _load(args.final)["model_state"]
    failures: list[str] = []

    identity_changed, identity_disallowed, identity_missing = _compare(
        factor, identity, IDENTITY_ALLOWED_PREFIXES
    )
    if identity_missing:
        failures.append("identity checkpoint missing parameters: " + ", ".join(identity_missing[:10]))
    if identity_disallowed:
        failures.append(
            "identity stage changed frozen parameters: "
            + ", ".join(f"{k}={d:.3g}" for k, d in identity_disallowed[:10])
        )
    # Identity training also evaluates its initial factor checkpoint. Selecting
    # epoch zero is an algorithmic no-improvement result, not parameter-transfer
    # corruption, provided that no frozen parameter changed and no key vanished.
    identity_selected_initial = (
        not identity_changed and not identity_disallowed and not identity_missing
    )

    final_changed, final_disallowed, final_missing = _compare(
        identity, final, FINAL_ALLOWED_PREFIXES
    )
    if final_missing:
        failures.append("final checkpoint missing parameters: " + ", ".join(final_missing[:10]))
    if final_disallowed:
        failures.append(
            "final calibration stage changed non-admission parameters: "
            + ", ".join(f"{k}={d:.3g}" for k, d in final_disallowed[:10])
        )
    if args.final_stage_disabled and final_changed:
        failures.append("final stage was disabled but final checkpoint differs from identity checkpoint")

    # A selected epoch-0 checkpoint is a valid fail-safe fallback. It means the
    # calibration stage was attempted but its initial identity checkpoint won.
    final_selected_initial = not final_changed and not final_disallowed and not final_missing
    doc = {
        "version": "v48.32-IDENTITY-UTILITY-BRIDGE",
        "valid": not failures,
        "factor": str(args.factor),
        "identity": str(args.identity),
        "final": str(args.final),
        "identity_allowed_changed_parameter_count": len(identity_changed),
        "identity_disallowed_changed_parameter_count": len(identity_disallowed),
        "identity_selected_initial_checkpoint": identity_selected_initial,
        "no_op_identity_selection_is_valid": True,
        "final_allowed_changed_parameter_count": len(final_changed),
        "final_disallowed_changed_parameter_count": len(final_disallowed),
        "final_stage_disabled": bool(args.final_stage_disabled),
        "final_selected_initial_checkpoint": final_selected_initial,
        "no_op_final_selection_is_valid": True,
        "failure_reasons": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(doc))
    return 0 if not failures else 31


if __name__ == "__main__":
    raise SystemExit(main())
