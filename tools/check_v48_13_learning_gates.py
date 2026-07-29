#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _f(v: Any) -> float | None:
    try:
        return None if v is None else float(v)
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Audit v48.13 TERRA proposal, proposal-conditioned evidence, and Natural gate."
    )
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--proposal-best-hit-min", type=float, default=0.75)
    ap.add_argument("--proposal-positive-hit-min", type=float, default=0.90)
    ap.add_argument("--near-benefit-auc-min", type=float, default=0.70)
    ap.add_argument("--contact-benefit-auc-min", type=float, default=0.75)
    ap.add_argument("--harm-auc-min", type=float, default=0.60)
    ap.add_argument("--proposal-evidence-corr-min", type=float, default=0.10)
    args = ap.parse_args()

    variants: dict[str, dict[str, Any]] = {}
    for variant in ("balanced", "precision"):
        base = args.run / "candidates" / variant
        row: dict[str, Any] = {"proposal": {}, "evidence": {}, "certificate": {}}
        for regime in ("near", "contact"):
            final = _load(base / "calibration" / f"direct_value_risk_{regime}_v48.json")
            pref = _load(
                base / "stages" / "set_tournament" / "preference_audit" / f"preference_{regime}.json"
            )
            best_hit = _f(final.get("proposal_oracle_best_hit_rate_positive_groups"))
            positive_hit = _f(final.get("proposal_any_positive_hit_rate_positive_groups"))
            row["proposal"][regime] = {
                "proposal_top_k": final.get("proposal_top_k"),
                "positive_group_count": final.get("proposal_positive_group_count"),
                "oracle_best_hit_rate_positive_groups": best_hit,
                "any_positive_hit_rate_positive_groups": positive_hit,
                "exact_top1_correlation": final.get("unconstrained_group_top1_correlation"),
                "exact_top1_accuracy": final.get("positive_group_top1_accuracy"),
                "positive_top1_regret": final.get("positive_group_top1_regret_mean"),
                "stage_p_audit_top1_correlation": pref.get("unconstrained_group_top1_correlation"),
                "passed": bool(
                    best_hit is not None
                    and best_hit >= args.proposal_best_hit_min
                    and positive_hit is not None
                    and positive_hit >= args.proposal_positive_hit_min
                ),
            }
            benefit_auc = _f(final.get("proposal_evidence_top1_positive_auc"))
            harm_auc = _f(final.get("proposal_evidence_top1_harm_auc"))
            evidence_corr = _f(final.get("proposal_evidence_top1_correlation"))
            benefit_min = args.near_benefit_auc_min if regime == "near" else args.contact_benefit_auc_min
            row["evidence"][regime] = {
                "proposal_evidence_top1_benefit_auc": benefit_auc,
                "proposal_evidence_top1_harm_auc": harm_auc,
                "proposal_evidence_top1_correlation": evidence_corr,
                "legacy_rank_top1_benefit_auc": final.get("policy_top1_positive_auc"),
                "legacy_rank_top1_harm_auc": final.get("policy_top1_harm_auc"),
                "candidate_benefit_auc": final.get("candidate_positive_auc"),
                "candidate_harm_auc": final.get("candidate_risk_harm_auc"),
                "passed": bool(
                    benefit_auc is not None
                    and benefit_auc >= benefit_min
                    and harm_auc is not None
                    and harm_auc >= args.harm_auc_min
                    and evidence_corr is not None
                    and evidence_corr >= args.proposal_evidence_corr_min
                ),
            }
            verify = final.get("verify") or {}
            row["certificate"][regime] = {
                "valid_for_deployment": bool(final.get("valid_for_deployment", False)),
                "verify_selected": verify.get("num_selected"),
                "precision": verify.get("precision"),
                "precision_lcb90": verify.get("precision_wilson_lcb90"),
                "harmful_selected_rate": verify.get("harmful_selected_rate"),
                "harmful_selected_ucb90": verify.get("harmful_selected_ucb90"),
                "positive_recall": verify.get("positive_recall"),
                "teacher_advantage_mean": verify.get("teacher_advantage_mean"),
                "macro_excess_share": verify.get("selected_macro_excess_share"),
                "near_miss_verify_frontier": final.get("near_miss_verify_frontier", [])[:5],
                "warnings": final.get("warnings", []),
                "passed": bool(final.get("valid_for_deployment", False)),
            }
        row["stage_p_proposal_passed"] = all(x["passed"] for x in row["proposal"].values())
        row["stage_e_proposal_evidence_passed"] = all(x["passed"] for x in row["evidence"].values())
        row["natural_gate_passed"] = all(x["passed"] for x in row["certificate"].values())
        variants[variant] = row

    doc = {
        "version": "v48.13",
        "algorithm": "OC-TRAC-TERRA",
        "run": str(args.run),
        "thresholds": {
            "proposal_best_hit_min": args.proposal_best_hit_min,
            "proposal_positive_hit_min": args.proposal_positive_hit_min,
            "near_benefit_auc_min": args.near_benefit_auc_min,
            "contact_benefit_auc_min": args.contact_benefit_auc_min,
            "harm_auc_min": args.harm_auc_min,
            "proposal_evidence_corr_min": args.proposal_evidence_corr_min,
        },
        "variants": variants,
        "decision": {
            "continue_to_multiseed": any(
                v["stage_p_proposal_passed"] and v["stage_e_proposal_evidence_passed"]
                for v in variants.values()
            ),
            "continue_to_stress_closed_loop": any(v["natural_gate_passed"] for v in variants.values()),
        },
        "note": (
            "Proposal/evidence thresholds are diagnostic and do not authorize deployment. "
            "Only the unchanged scene-disjoint Natural gate authorizes stress closed-loop."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(doc, ensure_ascii=False, indent=2))
    if doc["decision"]["continue_to_stress_closed_loop"]:
        return 0
    if doc["decision"]["continue_to_multiseed"]:
        return 10
    return 20


if __name__ == "__main__":
    raise SystemExit(main())
