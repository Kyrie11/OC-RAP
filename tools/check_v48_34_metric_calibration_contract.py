#!/usr/bin/env python3
"""Fail-closed identity check for v48.34 train, calibration and gate contracts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_env(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def _same_float(a: Any, b: Any, tol: float = 1.0e-12) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-summary", type=Path, required=True)
    ap.add_argument("--near-rule", type=Path, required=True)
    ap.add_argument("--contact-rule", type=Path, required=True)
    ap.add_argument("--gate-spec", type=Path, required=True)
    ap.add_argument("--policy-contract", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    summary = json.loads(args.train_summary.read_text(encoding="utf-8"))
    best_epoch = int(summary["best_epoch"])
    best = next((x for x in summary.get("history", []) if int(x.get("epoch", -1)) == best_epoch), None)
    if best is None:
        raise SystemExit(f"best epoch {best_epoch} not found in train summary")
    val = best.get("val", {})
    gate_doc = json.loads(args.gate_spec.read_text(encoding="utf-8"))
    protocol = gate_doc.get("protocol") or {}
    policy = _read_env(args.policy_contract)
    expected_topk = int(policy.get("PROPOSAL_TOP_K", "-1"))
    expected_rerank = policy.get("EVIDENCE_RERANK_TOP_K", "").lower() == "true"
    expected_order = "rank_topk_then_filter_then_evidence_rerank"
    policy_order = policy.get("SELECTION_SEMANTICS", "")
    gate_policy = protocol.get("policy") or {}
    gate_topk = int(gate_policy.get("proposal_top_k", -1))
    gate_order = str(gate_policy.get("selection_semantics", ""))
    rules = {
        "near": json.loads(args.near_rule.read_text(encoding="utf-8")),
        "contact": json.loads(args.contact_rule.read_text(encoding="utf-8")),
    }

    checks: dict[str, Any] = {}
    failures: list[str] = []
    for regime, rule in rules.items():
        constraints = rule.get("constraints") or {}
        gate_fit = ((protocol.get(regime) or {}).get("fit") or {})
        train_groups = int(round(float(val.get(f"direct_group_count_{regime}", -1))))
        train_safe = int(round(float(val.get(f"direct_safe_opportunity_group_count_{regime}", -1))))
        calibration_groups = int(rule.get("num_groups", -2))
        oracle_fit = ((rule.get("proposal_constrained_oracle_gate") or {}).get("fit") or {})
        calibration_safe = int(oracle_fit.get("proposal_safe_positive_groups", -2))
        threshold_map = {
            "min_selected": "min_fit_selected",
            "min_precision_lcb": "min_fit_precision_lcb",
            "max_harmful_group_ucb": "max_fit_harmful_group_ucb",
            "max_harmful_selected_ucb": "max_fit_harmful_selected_ucb",
        }
        threshold_checks = {
            gate_name: _same_float(constraints.get(rule_name), gate_fit.get(gate_name))
            for gate_name, rule_name in threshold_map.items()
        }
        proposal_topk_match = (
            expected_topk > 0
            and gate_topk == expected_topk
            and int(rule.get("proposal_top_k", -1)) == expected_topk
            and int(constraints.get("proposal_top_k", -1)) == expected_topk
            and int(oracle_fit.get("proposal_top_k", -1)) == expected_topk
        )
        rerank_match = bool(constraints.get("evidence_rerank_top_k", False)) == expected_rerank
        selection_order_match = (
            expected_rerank
            and policy_order == expected_order
            and gate_order == expected_order
        )
        group_match = train_groups == calibration_groups
        safe_match = train_safe == calibration_safe
        fit_feasible = bool(oracle_fit.get("feasible", False))
        checks[regime] = {
            "train_exact_eligible_groups": train_groups,
            "calibration_exact_eligible_groups": calibration_groups,
            "train_proposal_safe_opportunities": train_safe,
            "calibration_proposal_safe_opportunities": calibration_safe,
            "group_match": group_match,
            "safe_opportunity_match": safe_match,
            "fit_threshold_match": threshold_checks,
            "proposal_top_k": expected_topk,
            "proposal_top_k_match": proposal_topk_match,
            "evidence_rerank_match": rerank_match,
            "selection_order_match": selection_order_match,
            "proposal_oracle_fit_feasible": fit_feasible,
            "selection_semantics": expected_order,
        }
        if not group_match:
            failures.append(f"{regime}: eligible group count mismatch")
        if not safe_match:
            failures.append(f"{regime}: proposal safe-opportunity count mismatch")
        if not all(threshold_checks.values()):
            failures.append(f"{regime}: development-fit thresholds differ from preregistered GATE_SPEC")
        if not proposal_topk_match:
            failures.append(f"{regime}: proposal_top_k mismatch")
        if not rerank_match:
            failures.append(f"{regime}: evidence-rerank policy mismatch")
        if not selection_order_match:
            failures.append(f"{regime}: policy/GATE_SPEC selection-order mismatch")
        if not fit_feasible:
            failures.append(f"{regime}: proposal-constrained oracle cannot satisfy preregistered fit gate")

    doc = {
        "version": "v48.34-BARRIER-CROSSFIT",
        "valid": not failures,
        "best_epoch": best_epoch,
        "best_metric": summary.get("best_metric"),
        "gate_spec_sha256": gate_doc.get("protocol_sha256"),
        "policy_contract": {
            "proposal_top_k": expected_topk,
            "evidence_rerank_top_k": expected_rerank,
            "selection_semantics": expected_order,
            "policy_contract_selection_semantics": policy_order,
            "gate_spec_selection_semantics": gate_order,
        },
        "checks": checks,
        "failure_reasons": failures,
        "test_roots_read": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(doc, ensure_ascii=False))
    return 0 if not failures else 31


if __name__ == "__main__":
    raise SystemExit(main())
