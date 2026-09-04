#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROLES = ("dev_near", "certificate_near", "dev_contact", "certificate_contact")


def _v(role: dict[str, Any], path: str, default=None):
    cur: Any = role
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-summary", type=Path, required=True)
    ap.add_argument("--v48-88-comparison", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    audit = json.loads(args.audit_summary.read_text())
    prev = json.loads(args.v48_88_comparison.read_text())
    errors: list[str] = []
    if not audit.get("valid") or not audit.get("attribution_ready"):
        errors.append("V48.89 audit summary invalid")
    pd = prev.get("preregistered_decision") or {}
    if not (
        prev.get("valid")
        and pd.get("status") == "QUOTIENT_TAIL_RESPONSE_STOP"
        and not pd.get("quotient_tail_identifiability_go")
        and "root_correspondence" in str(pd.get("next_branch", ""))
    ):
        errors.append("V48.88 STOP/root-correspondence prerequisite missing")

    roles = audit.get("roles") or {}
    missing = [r for r in ROLES if r not in roles]
    if missing:
        errors.append(f"missing roles {missing}")
    label_identity = audit.get("label_identity") or {}
    for role in ROLES:
        ident = label_identity.get(role) or {}
        if not bool(ident.get("teacher_value_identity_on_overlap")):
            errors.append(f"teacher label overlap identity invalid for {role}")
        if int(ident.get("mismatches", 1)) != 0:
            errors.append(f"teacher label overlap mismatch for {role}")
        if int(ident.get("shared_rows", 0)) <= 0:
            errors.append(f"balanced/precision proposal support has no shared rows for {role}")
        if ident.get("cohort_policy") != "union_of_registered_balanced_precision_l80_proposals":
            errors.append(f"unexpected V48.89 label cohort policy for {role}")

    power = {r: int(_v(roles.get(r, {}), "labeled_rows", 0)) >= 100 for r in ROLES}
    shared = {
        r: float(_v(roles.get(r, {}), "shared_future_mass_candidate.median", 0.0) or 0.0) >= 0.95
        and float(_v(roles.get(r, {}), "shared_future_mass_nominal.median", 0.0) or 0.0) >= 0.95
        for r in ROLES
    }
    weak_identity = {
        r: float(_v(roles.get(r, {}), "semantic_identity_fallback_mass.q90", 1.0) or 0.0) <= 0.05
        for r in ROLES
    }
    soft_tail = {
        r: float(_v(roles.get(r, {}), "nested_tail_soft_correspondence_mass.median", 0.0) or 0.0) >= 0.85
        for r in ROLES
    }
    exact_tail = {
        r: float(_v(roles.get(r, {}), "nested_tail_exact_correspondence_mass.median", 0.0) or 0.0) >= 0.75
        for r in ROLES
    }
    root_correspondence_go = bool(
        not errors
        and all(power.values())
        and all(shared.values())
        and all(weak_identity.values())
        and all(soft_tail.values())
        and sum(exact_tail.values()) >= 3
    )

    sign_mass = {
        r: float(_v(roles.get(r, {}), "matched_tail_sign_identifiable_mass.median", 0.0) or 0.0) >= 0.50
        for r in ROLES
    }
    safe_power = {r: int(_v(roles.get(r, {}), "safe_positive_rows", 0)) >= 10 for r in ROLES}
    auc60 = {
        r: (_v(roles.get(r, {}), "safe_positive_vs_harmful_auc") is not None)
        and float(_v(roles.get(r, {}), "safe_positive_vs_harmful_auc")) >= 0.60
        for r in ROLES
    }
    top1_lift = {
        r: (_v(roles.get(r, {}), "safe_positive_top1_lift") is not None)
        and float(_v(roles.get(r, {}), "safe_positive_top1_lift")) >= 0.10
        for r in ROLES
    }
    near_roles = ("dev_near", "certificate_near")
    contact_roles = ("dev_contact", "certificate_contact")
    directional = bool(
        sum(auc60.values()) >= 3
        and any(auc60[r] for r in near_roles)
        and any(auc60[r] for r in contact_roles)
        and sum(top1_lift.values()) >= 2
        and any(top1_lift[r] for r in near_roles)
        and any(top1_lift[r] for r in contact_roles)
    )
    response_identifiability_go = bool(
        root_correspondence_go
        and all(safe_power.values())
        and sum(sign_mass.values()) >= 3
        and directional
    )
    training_authorized = bool(root_correspondence_go and response_identifiability_go)

    if training_authorized:
        status = "ROOT_CORRESPONDENCE_PHYSICAL_RESPONSE_IDENTIFIABILITY_GO"
        next_branch = (
            "train_matched_root_signed_response_operator_with_exact_correspondence_supervision_"
            "fixed_capacity_no_boundary_transport"
        )
    elif root_correspondence_go:
        status = "ROOT_CORRESPONDENCE_GO_LOCAL_PHYSICAL_RESPONSE_UNDERIDENTIFIED"
        next_branch = (
            "do_not_train_new_source_under_current_targets_then_audit_or_add_non_input_"
            "future_root_physical_margin_sidecar_without_dataset_reselection_no_capacity_sweep"
        )
    else:
        status = "COUNTERFACTUAL_ROOT_CORRESPONDENCE_STOP"
        next_branch = (
            "do_not_train_new_source_then_audit_counterfactual_future_identity_and_root_partition_"
            "stability_no_encoder_or_adapter_sweep"
        )

    decision = {
        "status": status,
        "next_branch": next_branch,
        "root_correspondence_go": root_correspondence_go,
        "root_local_physical_response_identifiability_go": response_identifiability_go,
        "matched_root_response_training_authorized": training_authorized,
        "powered_roles": power,
        "shared_future_mass_gate": shared,
        "weak_identity_fallback_gate": weak_identity,
        "soft_tail_correspondence_gate": soft_tail,
        "exact_tail_correspondence_gate": exact_tail,
        "sign_identifiable_mass_gate": sign_mass,
        "safe_positive_power_gate": safe_power,
        "safe_positive_vs_harmful_auc_gate": auc60,
        "safe_positive_top1_lift_gate": top1_lift,
        "directional_relevance_gate": directional,
    }
    doc = {
        "schema": "ocrap-v48.89-rcpi-comparison-v2",
        "engineering_version": "v48.89.1-OC-RCPI-ENGFIX",
        "valid": not errors,
        "attribution_ready": not errors,
        "errors": errors,
        "preregistered_decision": decision,
        "scientific_contract": {
            "experiment_type": "audit_only_identifiability_adjudication",
            "planner_parameters_trained": 0,
            "teacher_labels_changed": False,
            "teacher_metadata_input_to_model": False,
            "dataset_reconstruction": False,
            "regime_conditioning": False,
            "boundary_transport": "OFF",
            "relative_ranker_modified": False,
            "root_slot_identity_assumed": False,
            "correspondence_source": "shared counterfactual-future semantic branch identity",
            "physical_response_target": "matched-root pre-structural interval difference on exact nested OC-MERO tail",
            "teacher_label_cohort": "union of registered balanced/precision L80 top-K proposal supports after overlap-value identity",
            "capacity_sweep": False,
        },
        "audit_summary": audit,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"valid": doc["valid"], "status": status, "training_authorized": training_authorized}))
    return 0 if doc["valid"] else 30


if __name__ == "__main__":
    raise SystemExit(main())
