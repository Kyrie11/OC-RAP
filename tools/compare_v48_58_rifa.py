#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any


def load(p: Path) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _deployment_block(d: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    # adaptation-dev diagnostic is the fit population; dedicated certificate is
    # external verification.  Do not read obsolete top-level aliases.
    phase = "fit" if bool(d.get("development_fit_only")) else "verify"
    return phase, (d.get(phase) or {})


def metric(run: Path, variant: str, kind: str) -> dict[str, Any]:
    base = run / "candidates" / variant / "calibration"
    filename = {
        "dev_near": "dev_diagnostic_near_v48.json",
        "dev_contact": "dev_diagnostic_contact_v48.json",
        "certificate_near": "direct_value_risk_near_v48.json",
        "certificate_contact": "direct_value_risk_contact_v48.json",
    }[kind]
    p = base / filename
    if not p.is_file():
        return {"missing": str(p)}
    d = load(p)
    phase, dep = _deployment_block(d)
    deployment_keys = (
        "num_groups", "num_selected", "selection_rate", "num_positive_selected",
        "positive_recall", "precision", "precision_wilson_lcb90",
        "num_harmful_selected", "harmful_selected_rate", "harmful_selected_ucb90",
        "harmful_group_exposure", "harmful_group_exposure_ucb90",
        "num_opportunities", "teacher_advantage_mean", "teacher_advantage_min",
        "max_selected_macro_share", "selected_macro_counts",
    )
    diagnostic_keys = (
        "candidate_safe_positive_auc", "candidate_harm_auc",
        "candidate_positive_auc", "candidate_pred_teacher_correlation",
        "candidate_rank_teacher_correlation",
        "proposal_evidence_top1_correlation", "proposal_evidence_top1_positive_auc",
        "proposal_evidence_top1_safe_positive_auc", "proposal_evidence_top1_harm_auc",
        "proposal_evidence_top1_conditional_harm_auc",
        "proposal_evidence_nonpositive_false_switch_rate", "proposal_evidence_harmful_switch_rate",
        "proposal_deployed_rule_selected_count", "proposal_deployed_rule_abstention_count",
        "proposal_deployed_rule_abstention_rate", "proposal_deployed_rule_top1_correlation",
        "proposal_deployed_rule_top1_positive_auc", "proposal_deployed_rule_top1_safe_positive_auc",
        "proposal_deployed_rule_top1_harm_auc", "proposal_deployed_rule_nonpositive_false_switch_rate",
        "proposal_deployed_rule_harmful_switch_rate", "proposal_deployed_rule_positive_top1_regret_mean",
        "proposal_top_k", "proposal_positive_group_count", "proposal_oracle_best_hit_rate",
        "proposal_oracle_best_hit_rate_positive_groups", "proposal_any_positive_hit_rate_positive_groups",
    )
    return {
        "path": str(p),
        "development_or_certificate_phase": phase,
        "certificate_mode": d.get("certificate_mode"),
        "valid_for_deployment": d.get("valid_for_deployment"),
        "rejection_kind": d.get("rejection_kind"),
        "selection_rule": d.get("selection_rule"),
        "absolute_feasibility_mode": d.get("absolute_feasibility_mode"),
        "absolute_feasibility_threshold": d.get("absolute_feasibility_threshold"),
        "deployment": {k: dep.get(k) for k in deployment_keys if k in dep},
        "ranking_and_selector_diagnostics": {k: d.get(k) for k in diagnostic_keys if k in d},
        "proposal_constrained_oracle_gate": d.get("proposal_constrained_oracle_gate"),
        "proposal_support_curve": d.get("proposal_support_curve"),
        "warnings": d.get("warnings"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--b", type=Path, required=True)
    ap.add_argument("--c", type=Path, required=True)
    ap.add_argument("--feasibility-audit", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    x = ap.parse_args()
    arms = {"A": x.a, "B": x.b, "C_Main": x.c}
    d: dict[str, Any] = {
        "schema": "ocrap-v48.58-rifa-comparison-v2",
        "arms": {},
        "attribution_order": [
            "B-A: structural two-stage placement using raw native absolute feasibility",
            "C-B: isolated absolute-feasibility source correction with Stage-I frozen",
            "C-A: full RIFA effect",
        ],
    }
    for name, r in arms.items():
        d["arms"][name] = {
            v: {k: metric(r, v, k) for k in ("dev_near", "dev_contact", "certificate_near", "certificate_contact")}
            for v in ("balanced", "precision")
        }
    d["feasibility_role_audit"] = load(x.feasibility_audit)
    d["decision_contract"] = {
        "retain_CMRI": False,
        "retain_RIFA_only_if": [
            "B-A demonstrates structural safety benefit without catastrophic safe-positive loss OR C recovers that loss",
            "C-B improves absolute feasibility source geometry on Near and Contact",
            "Stage-I state isolation is bitwise valid",
            "certificate/dev deployment moves consistently with source geometry",
        ],
        "authorize_centering_next_only_if": [
            "RIFA source/native geometry forms Near+Contact Pareto improvement",
            "remaining dominant error is final relative score sign/centering rather than feasibility source",
        ],
        "stop_if": [
            "learned AFE does not beat raw native source on boundary discrimination",
            "safe-positive false veto remains dominant",
            "improvements are regime-specific with cross-severity regression",
            "state isolation fails",
        ],
    }
    x.output.parent.mkdir(parents=True, exist_ok=True)
    x.output.write_text(json.dumps(d, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "v48_58_rifa_comparison", "output": str(x.output)}))


if __name__ == "__main__":
    main()
