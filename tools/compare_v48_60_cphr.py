#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any


def load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def dep_block(d: dict) -> tuple[str, dict]:
    phase = "fit" if bool(d.get("development_fit_only")) else "verify"
    return phase, (d.get(phase) or {})


def metric(run: Path, variant: str, kind: str) -> dict[str, Any]:
    fn = {
        "dev_near": "dev_diagnostic_near_v48.json",
        "dev_contact": "dev_diagnostic_contact_v48.json",
        "certificate_near": "direct_value_risk_near_v48.json",
        "certificate_contact": "direct_value_risk_contact_v48.json",
    }[kind]
    p = run / "candidates" / variant / "calibration" / fn
    if not p.is_file():
        return {"missing": str(p)}
    d = load(p); phase, dep = dep_block(d)
    dk = (
        "num_groups", "num_selected", "selection_rate", "num_positive_selected",
        "positive_recall", "precision", "precision_wilson_lcb90",
        "num_harmful_selected", "harmful_selected_rate", "harmful_selected_ucb90",
        "harmful_group_exposure", "harmful_group_exposure_ucb90", "num_opportunities",
    )
    qk = (
        "candidate_safe_positive_auc", "candidate_harm_auc",
        "candidate_pred_teacher_correlation", "candidate_rank_teacher_correlation",
        "proposal_evidence_top1_correlation", "proposal_evidence_top1_safe_positive_auc",
        "proposal_evidence_top1_harm_auc", "proposal_deployed_rule_abstention_rate",
        "proposal_deployed_rule_top1_safe_positive_auc", "proposal_top_k",
        "proposal_positive_group_count", "proposal_oracle_best_hit_rate_positive_groups",
        "proposal_any_positive_hit_rate_positive_groups",
    )
    return {
        "path": str(p), "phase": phase, "valid_for_deployment": d.get("valid_for_deployment"),
        "rejection_kind": d.get("rejection_kind"),
        "absolute_feasibility_mode": d.get("absolute_feasibility_mode"),
        "absolute_feasibility_threshold": d.get("absolute_feasibility_threshold"),
        "deployment": {k: dep.get(k) for k in dk if k in dep},
        "ranking_and_selector_diagnostics": {k: d.get(k) for k in qk if k in d},
        "proposal_constrained_oracle_gate": d.get("proposal_constrained_oracle_gate"),
        "proposal_support_curve": d.get("proposal_support_curve"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="v48.60 CPHR controlled attribution comparison")
    for n in ("a", "b", "c", "d", "e"):
        ap.add_argument("--" + n, type=Path, required=True)
    ap.add_argument("--feasibility-audit", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    x = ap.parse_args()
    arms = {"A": x.a, "B_native": x.b, "C_AFE": x.c, "D_ORFC": x.d, "E_Main_CPHR": x.e}
    doc = {
        "schema": "ocrap-v48.60-cphr-comparison-v1",
        "arms": {},
        "attribution_order": [
            "reuse V48.58/V48.59 A-B-C-D evidence chain",
            "E-B: contextual physical source correction vs raw native (primary)",
            "E-C: contextual physical source vs compressed AFE",
            "E-D: contextual physical source vs context-free option bias",
            "state isolation and fixed-proposal contract",
            "E-A deployment propagation",
        ],
        "scientific_contract": {
            "primary_hypothesis": (
                "after AFE compression and global option-bias failures, residual absolute-source error is "
                "context dependent and can be corrected by deployable signed physical headroom without a regime id"
            ),
            "GO_requires": [
                "Near AND Contact source AUC/order improves over B and does not show the V48.59 cross-severity tradeoff",
                "teacher-feasible/safe-positive rejection decreases while teacher-infeasible/harmful pass is materially below C/D",
                "the learned six weights are non-negative, bounded, no-bias, and Stage-I remains bitwise frozen",
                "source gains replicate in balanced and precision variants and propagate to dev/certificate deployment",
                "Safe remains a shared-policy non-interference evaluation; no Safe supervision or regime input is used",
            ],
            "STOP_if": [
                "Near/Contact AUC trade off again",
                "only the 0.5 operating point/pass rate moves without discrimination gain",
                "infeasible/harmful pass remains at the V48.58-C/V48.59-D level",
                "deployment does not respond to source gain",
                "state isolation/provenance fails",
            ],
            "centering_authorized_only_after_GO": True,
            "if_GO_residual_question": (
                "separately audit relative opportunity/harm/pred_adv/evidence-margin; only then test a relative physical-headroom anchor"
            ),
            "forbidden_next_sweeps": [
                "AFE feature-stack/width/class-weight/threshold",
                "ORFC option-bias/LR/threshold grid",
                "proposal expansion",
                "regime router/policy/threshold/budget",
                "broad root or margin-head retraining",
                "teacher privileged margin distillation",
            ],
        },
    }
    for name, run in arms.items():
        doc["arms"][name] = {
            v: {k: metric(run, v, k) for k in ("dev_near", "dev_contact", "certificate_near", "certificate_contact")}
            for v in ("balanced", "precision")
        }
    doc["feasibility_role_audit"] = load(x.feasibility_audit)
    x.output.parent.mkdir(parents=True, exist_ok=True)
    x.output.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "v48_60_cphr_comparison", "output": str(x.output)}))


if __name__ == "__main__":
    main()
