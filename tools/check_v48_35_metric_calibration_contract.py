#!/usr/bin/env python3
"""Fail-closed identity check for the v48.35 one-rule protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _env(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def _same(a: Any, b: Any, tol: float = 1e-12) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-summary", type=Path, required=True)
    ap.add_argument("--near-dev", type=Path, required=True)
    ap.add_argument("--contact-dev", type=Path, required=True)
    ap.add_argument("--shared-rule", type=Path, required=True)
    ap.add_argument("--gate-spec", type=Path, required=True)
    ap.add_argument("--policy-contract", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    summary = json.loads(args.train_summary.read_text(encoding="utf-8"))
    best_epoch = int(summary["best_epoch"])
    best = next((x for x in summary.get("history", []) if int(x.get("epoch", -1)) == best_epoch), None)
    if best is None:
        raise SystemExit(f"best epoch {best_epoch} not found")
    val = best.get("val") or {}
    dev = {
        "near": json.loads(args.near_dev.read_text(encoding="utf-8")),
        "contact": json.loads(args.contact_dev.read_text(encoding="utf-8")),
    }
    shared = json.loads(args.shared_rule.read_text(encoding="utf-8"))
    gate_doc = json.loads(args.gate_spec.read_text(encoding="utf-8"))
    protocol = gate_doc.get("protocol") or {}
    policy = _env(args.policy_contract)

    failures: list[str] = []
    checks: dict[str, Any] = {}
    expected_order = "rank_topk_then_filter_then_evidence_rerank"
    expected_topk = int(policy.get("PROPOSAL_TOP_K", "-1"))
    checks["single_shared_rule"] = (
        int(shared.get("shared_rule_count", 0)) == 1
        and shared.get("strategy_regime_conditioning") is False
        and sorted(shared.get("audit_strata_only") or []) == ["contact", "near"]
    )
    selected_rule = shared.get("rule") or shared.get("diagnostic_fit_rule") or {}
    semantic_domain = ((shared.get("constraints") or {}).get("semantic_rule_domain") or {})
    checks["noncompensatory_semantic_rule_domain"] = (
        float(selected_rule.get("opportunity_threshold", -1.0)) >= float(semantic_domain.get("min_opportunity_threshold", 0.5))
        and float(selected_rule.get("harm_threshold", 2.0)) <= float(semantic_domain.get("max_harm_threshold", 0.5))
        and float(selected_rule.get("score_threshold", -1.0)) >= float(semantic_domain.get("min_score_threshold", 0.0))
    )
    checks["selection_semantics"] = (
        policy.get("SELECTION_SEMANTICS") == expected_order
        and ((protocol.get("policy") or {}).get("selection_semantics")) == expected_order
    )
    checks["proposal_top_k"] = (
        expected_topk > 0
        and int((protocol.get("policy") or {}).get("proposal_top_k", -1)) == expected_topk
        and all(int(d.get("proposal_top_k", -1)) == expected_topk for d in dev.values())
    )
    source_meta = shared.get("sources") or {}
    for stratum, path in (("near", args.near_dev), ("contact", args.contact_dev)):
        # shared fitter consumes proposal rows, not the dev summary. Verify its
        # declared stratum population against the dev certificate population.
        train_groups = int(round(float(val.get(f"direct_group_count_{stratum}", -1))))
        dev_groups = int(dev[stratum].get("num_groups", -2))
        source_groups = int((source_meta.get(stratum) or {}).get("group_count", -3))
        safe_train = int(round(float(val.get(f"direct_safe_opportunity_group_count_{stratum}", -1))))
        safe_dev = int((((dev[stratum].get("proposal_constrained_oracle_gate") or {}).get("fit") or {}).get("proposal_safe_positive_groups", -2)))
        fit_spec = ((protocol.get(stratum) or {}).get("fit") or {})
        shared_constraints = shared.get("constraints") or {}
        threshold_checks = {
            "min_selected": int((shared_constraints.get("min_selected") or {}).get(stratum, -1)) == int(fit_spec.get("min_selected", -2)),
            "min_precision_lcb": _same((shared_constraints.get("min_precision_lcb") or {}).get(stratum), fit_spec.get("min_precision_lcb")),
            "max_harmful_group_ucb": _same((shared_constraints.get("max_harmful_group_ucb") or {}).get(stratum), fit_spec.get("max_harmful_group_ucb")),
            "max_harmful_selected_ucb": _same((shared_constraints.get("max_harmful_selected_ucb") or {}).get(stratum), fit_spec.get("max_harmful_selected_ucb")),
        }
        checks[stratum] = {
            "train_groups": train_groups,
            "dev_groups": dev_groups,
            "shared_source_groups": source_groups,
            "train_safe_opportunity_groups": safe_train,
            "dev_safe_opportunity_groups": safe_dev,
            "group_identity": train_groups == dev_groups == source_groups,
            "safe_opportunity_identity": safe_train == safe_dev,
            "fit_threshold_identity": threshold_checks,
            "proposal_oracle_fit_feasible": bool((((dev[stratum].get("proposal_constrained_oracle_gate") or {}).get("fit") or {}).get("feasible", False))),
        }
    for name, ok in checks.items():
        if isinstance(ok, bool) and not ok:
            failures.append(name)
    for stratum in ("near", "contact"):
        row = checks[stratum]
        if not row["group_identity"]:
            failures.append(f"{stratum}:group_identity")
        if not row["safe_opportunity_identity"]:
            failures.append(f"{stratum}:safe_opportunity_identity")
        if not all(row["fit_threshold_identity"].values()):
            failures.append(f"{stratum}:fit_threshold_identity")
        if not row["proposal_oracle_fit_feasible"]:
            failures.append(f"{stratum}:proposal_oracle_fit_infeasible")

    doc = {
        "version": "v48.35-CONTINUOUS-FRONTIER",
        "valid": not failures,
        "best_epoch": best_epoch,
        "best_metric": summary.get("best_metric"),
        "gate_spec_sha256": hashlib.sha256(args.gate_spec.read_bytes()).hexdigest(),
        "shared_rule_sha256": hashlib.sha256(args.shared_rule.read_bytes()).hexdigest(),
        "single_deployment_rule": True,
        "strategy_regime_conditioning": False,
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
