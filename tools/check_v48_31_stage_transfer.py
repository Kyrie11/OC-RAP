#!/usr/bin/env python3
"""Verify the intended three-stage parameter-transfer contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

ALLOWED_FINAL_PREFIXES = (
    "direct_evidence_concord_benefit_calibrator.",
    "direct_evidence_concord_harm_calibrator.",
    "direct_evidence_concord_admission_calibrator.",
)
FROZEN_FACTOR_PREFIXES = (
    "direct_evidence_concord_benefit_calibrator.",
    "direct_evidence_concord_harm_calibrator.",
)


def _load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _max_diff(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor], prefixes: tuple[str, ...]) -> float:
    diffs = []
    for key, value in a.items():
        if key in b and key.startswith(prefixes):
            diffs.append(float((value.detach().float() - b[key].detach().float()).abs().max().item()))
    return max(diffs) if diffs else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--factor", type=Path, required=True)
    ap.add_argument("--admission", type=Path, required=True)
    ap.add_argument("--final", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--allow-no-joint", action="store_true")
    args = ap.parse_args()

    factor = _load(args.factor)
    admission = _load(args.admission)
    final = _load(args.final)
    fs = factor["model_state"]
    ads = admission["model_state"]
    fins = final["model_state"]
    failures = []

    factor_to_admission_frozen_diff = _max_diff(fs, ads, FROZEN_FACTOR_PREFIXES)
    if factor_to_admission_frozen_diff > 0.0:
        failures.append("stage2 changed frozen benefit/component factor parameters")

    disallowed_final_diffs = []
    changed_allowed = []
    for key, value in ads.items():
        if key not in fins:
            failures.append(f"final checkpoint missing parameter {key}")
            continue
        diff = float((value.detach().float() - fins[key].detach().float()).abs().max().item())
        if key.startswith(ALLOWED_FINAL_PREFIXES):
            if diff > 0.0:
                changed_allowed.append(key)
        elif diff > 0.0:
            disallowed_final_diffs.append((key, diff))
    if disallowed_final_diffs:
        failures.append(
            "stage3 changed frozen parameters: "
            + ", ".join(f"{k}={d:.3g}" for k, d in disallowed_final_diffs[:10])
        )
    if not changed_allowed and not args.allow_no_joint:
        failures.append("stage3 did not update any allowed evidence calibrator parameter")

    doc = {
        "version": "v48.31-CONTRACT-SLACK-RANK",
        "valid": not failures,
        "factor": str(args.factor),
        "admission": str(args.admission),
        "final": str(args.final),
        "factor_to_admission_frozen_max_abs_diff": factor_to_admission_frozen_diff,
        "stage3_allowed_changed_parameter_count": len(changed_allowed),
        "joint_refinement_required": not args.allow_no_joint,
        "stage3_disallowed_changed_parameter_count": len(disallowed_final_diffs),
        "failure_reasons": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, indent=2) + "\n")
    print(json.dumps(doc))
    return 0 if not failures else 31


if __name__ == "__main__":
    raise SystemExit(main())
