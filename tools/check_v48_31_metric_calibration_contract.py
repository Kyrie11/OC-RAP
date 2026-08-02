#!/usr/bin/env python3
"""Fail closed when training validation and adaptation-dev calibration differ.

The model is still unified across regimes.  Near/Contact are used only as
reporting strata to verify that the exact same eligible scene-time groups and
proposal-contained safe opportunities reached checkpoint selection and rule
fitting.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-summary", type=Path, required=True)
    ap.add_argument("--near-rule", type=Path, required=True)
    ap.add_argument("--contact-rule", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    summary = json.loads(args.train_summary.read_text())
    best_epoch = int(summary["best_epoch"])
    history = summary.get("history", [])
    best = next((x for x in history if int(x.get("epoch", -1)) == best_epoch), None)
    if best is None:
        raise SystemExit(f"best epoch {best_epoch} not found in train summary")
    val = best.get("val", {})
    rules = {
        "near": json.loads(args.near_rule.read_text()),
        "contact": json.loads(args.contact_rule.read_text()),
    }
    checks = {}
    failures = []
    for regime, rule in rules.items():
        train_groups = int(round(float(val.get(f"direct_group_count_{regime}", -1))))
        train_safe = int(round(float(val.get(f"direct_safe_opportunity_group_count_{regime}", -1))))
        calibration_groups = int(rule.get("num_groups", -2))
        oracle_fit = ((rule.get("proposal_constrained_oracle_gate") or {}).get("fit") or {})
        calibration_safe = int(oracle_fit.get("proposal_safe_positive_groups", -2))
        group_match = train_groups == calibration_groups
        safe_match = train_safe == calibration_safe
        checks[regime] = {
            "train_exact_eligible_groups": train_groups,
            "calibration_exact_eligible_groups": calibration_groups,
            "train_proposal_safe_opportunities": train_safe,
            "calibration_proposal_safe_opportunities": calibration_safe,
            "group_match": group_match,
            "safe_opportunity_match": safe_match,
        }
        if not group_match:
            failures.append(f"{regime}: eligible group count mismatch")
        if not safe_match:
            failures.append(f"{regime}: proposal safe-opportunity count mismatch")

    doc = {
        "version": "v48.31-CONTRACT-SLACK-RANK",
        "valid": not failures,
        "best_epoch": best_epoch,
        "best_metric": summary.get("best_metric"),
        "checks": checks,
        "failure_reasons": failures,
        "test_roots_read": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(doc, ensure_ascii=False))
    return 0 if not failures else 31


if __name__ == "__main__":
    raise SystemExit(main())
