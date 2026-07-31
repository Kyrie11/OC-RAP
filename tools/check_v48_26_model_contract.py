#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ocrap.models.inference import load_model_bundle


def main() -> int:
    ap = argparse.ArgumentParser(description="Fail-closed v48.26 checkpoint/inference contract audit")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--expect-frontier", choices=("true", "false"), default="true")
    ap.add_argument("--expect-admission-bounded", choices=("true", "false"), default="false")
    ap.add_argument("--expect-component-prior-logit", type=float, default=-2.0)
    args = ap.parse_args()

    bundle = load_model_bundle(args.checkpoint)
    if bundle is None:
        raise SystemExit(f"unable to load checkpoint: {args.checkpoint}")
    model = bundle.model
    actual = {
        "direct_recovery_evidence_frontier": bool(model.direct_recovery_evidence_frontier),
        "direct_recovery_evidence_admission_bounded": bool(model.direct_recovery_evidence_admission_bounded),
        "direct_recovery_evidence_component_prior_logit": float(model.direct_recovery_evidence_component_prior_logit),
        "inference_evidence_contract_verified": bool(
            (bundle.cfg.get("model", {}) or {}).get("inference_evidence_contract_verified", False)
        ),
    }
    expected = {
        "direct_recovery_evidence_frontier": args.expect_frontier == "true",
        "direct_recovery_evidence_admission_bounded": args.expect_admission_bounded == "true",
        "direct_recovery_evidence_component_prior_logit": float(args.expect_component_prior_logit),
        "inference_evidence_contract_verified": True,
    }
    mismatches = {
        key: {"expected": expected[key], "actual": actual[key]}
        for key in expected
        if actual[key] != expected[key]
    }
    doc = {
        "event": "v48_26_checkpoint_inference_contract_audit",
        "checkpoint": str(args.checkpoint),
        "expected": expected,
        "actual": actual,
        "valid": not mismatches,
        "mismatches": mismatches,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(doc, ensure_ascii=False))
    return 0 if not mismatches else 4


if __name__ == "__main__":
    raise SystemExit(main())
