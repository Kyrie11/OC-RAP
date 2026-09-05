#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ocrap.v48_93_factor_mediation import ENGINEERING_VERSION

ROLES = ("dev_near", "dev_contact", "certificate_near", "certificate_contact")
FACTORS = ("drs", "deployability_gate", "gap_discount")


def _has_near_contact(roles: list[str]) -> bool:
    return any("near" in r for r in roles) and any("contact" in r for r in roles)


def _single_gate(summary: dict[str, Any], factor: str) -> dict[str, Any]:
    necessity_roles: list[str] = []
    sufficiency_roles: list[str] = []
    false_rescue_ok: list[str] = []
    per: dict[str, Any] = {}
    for role in ROLES:
        s = summary["roles"][role]["factor_stats"][factor]
        n = s.get("safe_necessity_fraction")
        u = s.get("safe_sufficiency_fraction")
        f = s.get("harmful_single_factor_false_rescue_fraction")
        per[role] = {"necessity": n, "sufficiency": u, "harmful_false_rescue": f}
        if n is not None and float(n) >= 0.70:
            necessity_roles.append(role)
        if u is not None and float(u) >= 0.60:
            sufficiency_roles.append(role)
        if f is not None and float(f) <= 0.10:
            false_rescue_ok.append(role)
    n_go = len(necessity_roles) >= 3 and _has_near_contact(necessity_roles)
    u_go = len(sufficiency_roles) >= 3 and _has_near_contact(sufficiency_roles)
    f_go = len(false_rescue_ok) == 4
    return {
        "per_role": per,
        "necessity_roles": necessity_roles,
        "sufficiency_roles": sufficiency_roles,
        "false_rescue_ok_roles": false_rescue_ok,
        "necessity_gate": n_go,
        "sufficiency_gate": u_go,
        "false_rescue_gate": f_go,
        "go": bool(n_go and u_go and f_go),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--v48-92-comparison", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    s = json.loads(args.summary.read_text())
    c92 = json.loads(args.v48_92_comparison.read_text())
    q92 = c92.get("preregistered_decision") or {}
    errors: list[str] = []
    if not (s.get("valid") and s.get("attribution_ready")):
        errors.append("invalid V48.93 summary")
    if str(s.get("engineering_version")) != ENGINEERING_VERSION:
        errors.append("V48.93 version mismatch")
    if not (c92.get("valid") and q92.get("status") == "SHARED_RECOVERY_ADVANTAGE_MEDIATOR_GO"):
        errors.append("V48.92 shared-mediator screening prerequisite missing")
    if len(q92.get("shared_mediator_winners") or []) < 2:
        errors.append("V48.93 requires multi-winner V48.92 screening")
    if float(s.get("max_advantage_identity_error", 1.0)) > 2.0e-6:
        errors.append("factor-mediation PCD identity failed")

    single = {f: _single_gate(s, f) for f in FACTORS}
    single_winners = [f for f, g in single.items() if g["go"]]

    coverage_roles = [
        role for role in ROLES
        if float(s["roles"][role].get("drs_or_deployability_necessity_coverage") if s["roles"][role].get("drs_or_deployability_necessity_coverage") is not None else 0.0) >= 0.90
    ]
    gap_ok_roles = [
        role for role in ROLES
        if float(s["roles"][role].get("gap_necessity_fraction") if s["roles"][role].get("gap_necessity_fraction") is not None else 1.0) <= 0.10
    ]
    residual_ok_roles = [
        role for role in ROLES
        if float(s["roles"][role].get("multi_or_unexplained_fraction") if s["roles"][role].get("multi_or_unexplained_fraction") is not None else 1.0) <= 0.10
    ]
    drs_power_roles = [
        role for role in ROLES
        if int(s["roles"][role]["safe_mode_counts"].get("drs_activation", 0)) >= 5
    ]
    dep_power_roles = [
        role for role in ROLES
        if int(s["roles"][role]["safe_mode_counts"].get("deployability_gain", 0)) >= 5
    ]
    complementarity_go = bool(
        not single_winners
        and len(coverage_roles) == 4
        and len(gap_ok_roles) == 4
        and len(residual_ok_roles) == 4
        and len(drs_power_roles) >= 3 and _has_near_contact(drs_power_roles)
        and len(dep_power_roles) >= 3 and _has_near_contact(dep_power_roles)
    )

    if single_winners:
        status = "SINGLE_PCD_MEDIATOR_GO"
        next_branch = "one_single_factor_is_necessary_and_sufficient_then_only_that_fixed_capacity_mediator_specific_experiment_is_authorized"
    elif complementarity_go:
        status = "PCD_FACTOR_COMPLEMENTARITY_GO"
        next_branch = "drs_activation_plus_deployability_gain_complementarity_identified_then_design_one_fixed_capacity_complementarity_aligned_experiment_no_weight_or_regime_sweep"
    else:
        status = "PCD_FACTOR_MEDIATION_STOP"
        next_branch = "no_single_or_two_mode_factor_mediation_then_audit_teacher_action_benefit_semantics_before_new_capacity"

    decision = {
        "single_factor_gates": single,
        "single_factor_winners": single_winners,
        "drs_or_deployability_coverage_roles": coverage_roles,
        "gap_nonmediator_roles": gap_ok_roles,
        "multi_or_unexplained_small_roles": residual_ok_roles,
        "drs_mode_power_roles": drs_power_roles,
        "deployability_mode_power_roles": dep_power_roles,
        "factor_complementarity_go": complementarity_go,
        "source_training_authorized": False,
        "boundary_transport_authorized": False,
        "regime_conditioned_policy_authorized": False,
        "dataset_reconstruction_authorized": False,
        "next_branch": next_branch,
        "status": status,
    }
    out = {
        "schema": "ocrap-v48.93-fmca-comparison-v1",
        "engineering_version": ENGINEERING_VERSION,
        "valid": not errors,
        "attribution_ready": not errors,
        "errors": errors,
        "experiment_type": "audit_only_exact_pcd_factor_mediation_complementarity",
        "planner_parameters_trained": 0,
        "dataset_reconstruction": False,
        "dataset_reselection": False,
        "teacher_labels_changed": False,
        "teacher_metadata_input_to_model": False,
        "boundary_transport": False,
        "relative_ranker_modified": False,
        "regime_conditioning": False,
        "womd_replay_performed": False,
        "preregistered_decision": decision,
        "test_roots_read": False,
    }
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"valid": out["valid"], "status": status, "single_winners": single_winners, "complementarity_go": complementarity_go}))
    return 0 if out["valid"] else 30


if __name__ == "__main__":
    raise SystemExit(main())
