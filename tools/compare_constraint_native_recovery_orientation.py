#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ocrap.audits.executable_constraint_jacobian import (
    ENGINEERING_VERSION,
    JACOBIAN_GEOMETRY_DIM,
    MATCHED_DIM,
    SCIENTIFIC_VERSION,
)

AUTHORITATIVE_V112_COMPARISON_SHA256 = "0f07aed5ad1d52572a91b3491dbb63d21f1e2c231c1321bb52a66e5cbe351d64"
AUTHORITATIVE_V112_PIPELINE_SHA256 = "426329407549b0408c4d6a225fa40239c13115f49b05d834a6c631f1e460e6ea"
AUTHORITATIVE_V112_BALANCED_SHA256 = "20bc2c049f4793ca8ff74fe428ffeaf1284fad74ce03036e1f45ff20bf212a1b"
AUTHORITATIVE_V112_PRECISION_SHA256 = "16129135e0509566e31604e6a80c5a5a82788d2452eca1d00c19cd0459953995"
ROLES = ("dev_near", "dev_contact", "certificate_near", "certificate_contact")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ok(v: Any, threshold: float) -> bool:
    return v is not None and float(v) >= threshold


def _cross(roles: set[str], n: int) -> bool:
    return len(roles) >= n and any("near" in x for x in roles) and any("contact" in x for x in roles)


def _action_gate(docs: dict[str, Any], space: str, metric: str) -> dict[str, Any]:
    positive: list[list[str]] = []
    top: list[list[str]] = []
    roles: set[str] = set()
    top_roles: set[str] = set()
    for variant, d in docs.items():
        for role in ROLES:
            m = d[f"{space}_cells"][role][f"{metric}_true"]
            if _ok(m.get("auc"), 0.65) and _ok(m.get("auc_vs_shuffled"), 0.05):
                positive.append([variant, role])
                roles.add(role)
            if _ok(m.get("top1_vs_shuffled"), 0.10):
                top.append([variant, role])
                top_roles.add(role)
    go = len(positive) >= 6 and _cross(roles, 3) and len(top) >= 4 and _cross(top_roles, 2)
    return {
        "go": bool(go),
        "local_order": bool(len(top) >= 4 and _cross(top_roles, 2)),
        "positive_cells": positive,
        "top1_material_cells": top,
        "roles": sorted(roles),
        "top1_roles": sorted(top_roles),
    }


def _matched_increment_gate(docs: dict[str, Any], metric: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    positive: list[list[str]] = []
    material: list[list[str]] = []
    roles: set[str] = set()
    for variant, d in docs.items():
        for role in ROLES:
            ca = d["candidate_option_cells"][role][f"{metric}_true"].get("auc")
            no = d["nominal_option_cells"][role][f"{metric}_true"].get("auc")
            delta = None if ca is None or no is None else float(ca) - float(no)
            rows.append({
                "variant": variant, "role": role,
                "candidate_option_auc": ca, "nominal_option_auc": no,
                "candidate_minus_nominal_option": delta,
            })
            if delta is not None and delta > 0.0:
                positive.append([variant, role]); roles.add(role)
            if delta is not None and delta >= 0.01:
                material.append([variant, role])
    go = len(positive) >= 6 and _cross(roles, 3) and len(material) >= 4
    return {
        "go": bool(go), "positive_cells": positive, "material_cells": material,
        "roles": sorted(roles), "rows": rows,
    }


def _continuation_increment_gate(docs: dict[str, Any], hist: dict[str, Any], metric: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    positive: list[list[str]] = []
    material: list[list[str]] = []
    roles: set[str] = set()
    for variant in ("balanced", "precision"):
        for role in ROLES:
            now = docs[variant]["candidate_option_cells"][role][f"{metric}_true"].get("auc")
            old = hist[variant]["candidate_cone_cells"][role][f"{metric}_true"].get("auc")
            delta = None if now is None or old is None else float(now) - float(old)
            rows.append({
                "variant": variant, "role": role,
                "candidate_option_auc": now, "v48_112_candidate_cone_auc": old,
                "executable_continuation_minus_prefix_cone": delta,
            })
            if delta is not None and delta > 0.0:
                positive.append([variant, role]); roles.add(role)
            if delta is not None and delta >= 0.01:
                material.append([variant, role])
    go = len(positive) >= 6 and _cross(roles, 3) and len(material) >= 4
    return {
        "go": bool(go), "positive_cells": positive, "material_cells": material,
        "roles": sorted(roles), "rows": rows,
    }


def _variant_identity(docs: dict[str, Any]) -> dict[str, Any]:
    diffs: list[str] = []
    for space in ("base", "nominal_option", "candidate_option"):
        for role in ROLES:
            for metric in ("support_true", "support_shuffled", "reserve_true", "reserve_shuffled"):
                a = docs["balanced"][f"{space}_cells"][role][metric]
                b = docs["precision"][f"{space}_cells"][role][metric]
                for key in ("auc", "top1", "rows", "positive_rows", "negative_rows", "powered_groups"):
                    if a.get(key) != b.get(key):
                        diffs.append(f"{space}:{role}:{metric}:{key}")
    return {"exact": not diffs, "differences": diffs, "effective_unique_roles_if_exact": 4}


def _power(docs: dict[str, Any], space: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for variant in ("balanced", "precision"):
        for role in ROLES:
            for axis in ("support", "reserve"):
                m = docs[variant][f"{space}_cells"][role][f"{axis}_true"]
                out.append({
                    "variant": variant, "role": role, "axis": axis,
                    "rows": m.get("rows"), "positive_rows": m.get("positive_rows"),
                    "negative_rows": m.get("negative_rows"), "powered_groups": m.get("powered_groups"),
                    "underpowered": bool(
                        (m.get("powered_groups") or 0) < 3
                        or (m.get("positive_rows") or 0) < 5
                        or (m.get("negative_rows") or 0) < 5
                    ),
                })
    return out


def _activity_gate(docs: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    option_switch_roles: set[str] = set()
    mode_diverse_roles: set[str] = set()
    constraint_diverse_roles: set[str] = set()
    reentry_contact_roles: set[str] = set()
    for variant in ("balanced", "precision"):
        d = docs[variant]
        for role in ROLES:
            diag = ((d.get("events") or {}).get(role, {}).get("pair_diagnostics") or {})
            switch = float(diag.get("option_switch_fraction", 0.0) or 0.0)
            modes = diag.get("candidate_selected_mode_counts", {}) or {}
            active = diag.get("candidate_selected_active_type_counts", {}) or {}
            reentry = float(diag.get("candidate_selected_reentry_active_fraction", 0.0) or 0.0)
            rows.append({
                "variant": variant, "role": role,
                "option_switch_fraction": switch,
                "candidate_selected_mode_counts": modes,
                "candidate_selected_active_type_counts": active,
                "candidate_selected_reentry_active_fraction": reentry,
                "field_reentry_available_fraction": float(diag.get("field_reentry_available_fraction", 0.0) or 0.0),
            })
            if switch > 0.0:
                option_switch_roles.add(role)
            if sum(int(v) > 0 for v in modes.values()) >= 2:
                mode_diverse_roles.add(role)
            if sum(int(v) > 0 for v in active.values()) >= 2:
                constraint_diverse_roles.add(role)
            if "contact" in role and reentry > 0.0:
                reentry_contact_roles.add(role)
    # Core ECJ activity is about whether the executable continuation actually
    # exposes heterogeneous binding constraints.  Recovery-mode diversity and
    # candidate-vs-nominal option switching are prerequisites only for claiming
    # an adaptive selector effect.  Re-entry coverage is reported separately:
    # the paper-level Contact bucket may be a counterfactual contact-surrogate
    # target and therefore does not imply an observed overlap at the audit
    # anchor.  Requiring re-entry for the *core* ECJ gate would conflate dataset
    # role semantics with observable physical contact.  A full post-contact
    # promotion still requires explicit re-entry evidence.
    core_go = _cross(constraint_diverse_roles, 3)
    adaptive_go = _cross(option_switch_roles, 3) and _cross(mode_diverse_roles, 3)
    reentry_go = len(reentry_contact_roles) >= 1
    return {
        "go": bool(core_go),
        "adaptive_selector_activity_go": bool(adaptive_go),
        "reentry_contact_coverage_go": bool(reentry_go),
        "option_switch_roles": sorted(option_switch_roles),
        "mode_diverse_roles": sorted(mode_diverse_roles),
        "constraint_diverse_roles": sorted(constraint_diverse_roles),
        "reentry_contact_roles": sorted(reentry_contact_roles),
        "rows": rows,
    }


def _result_errors(d: dict[str, Any], variant: str) -> list[str]:
    errors: list[str] = []
    required = {
        "valid": True,
        "variant": variant,
        "audit_only": True,
        "convex_closed_form_ridge": True,
        "strictly_convex_unique_solution": True,
        "iterative_optimizer_used": False,
        "capacity_matched_nominal_vs_candidate_option": True,
        "actuator_projection": True,
        "teacher_npz_fields_loaded_into_feature_path": False,
        "planner_parameters_trained": 0,
        "stage_i_parameters_trained": 0,
        "root_decoder_parameters_trained": 0,
        "source_parameters_trained": 0,
        "relative_ranker_modified": False,
        "regime_conditioning": False,
        "boundary_transport": False,
        "teacher_metadata_input_to_model": False,
        "test_roots_read": False,
        "posthoc_feature_selection": False,
    }
    for k, v in required.items():
        if d.get(k) != v:
            errors.append(f"{variant}:{k}")
    if d.get("engineering_version") != ENGINEERING_VERSION:
        errors.append(f"{variant}:engineering_version")
    if d.get("scientific_version") != SCIENTIFIC_VERSION:
        errors.append(f"{variant}:scientific_version")
    if int(d.get("jacobian_geometry_dimension", -1)) != JACOBIAN_GEOMETRY_DIM:
        errors.append(f"{variant}:geometry_dim")
    if int(d.get("matched_family_dimension", -1)) != MATCHED_DIM:
        errors.append(f"{variant}:matched_dim")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser()
    for key in ("balanced", "precision", "v112_pipeline", "v112_comparison", "v112_balanced", "v112_precision"):
        ap.add_argument("--" + key.replace("_", "-"), dest=key, type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    errors: list[str] = []

    docs = {v: json.loads(getattr(a, v).read_text()) for v in ("balanced", "precision")}
    for v, obj in docs.items():
        errors += _result_errors(obj, v)
        if obj.get("run_instance_id") != a.run_id:
            errors.append(f"{v}:run_instance_id")

    p112 = json.loads(a.v112_pipeline.read_text())
    c112 = json.loads(a.v112_comparison.read_text())
    b112 = json.loads(a.v112_balanced.read_text())
    p112r = json.loads(a.v112_precision.read_text())
    d112 = c112.get("preregistered_decision") or {}

    if _sha(a.v112_pipeline) != AUTHORITATIVE_V112_PIPELINE_SHA256:
        errors.append("v112_pipeline_sha")
    if _sha(a.v112_comparison) != AUTHORITATIVE_V112_COMPARISON_SHA256:
        errors.append("v112_comparison_sha")
    if _sha(a.v112_balanced) != AUTHORITATIVE_V112_BALANCED_SHA256:
        errors.append("v112_balanced_sha")
    if _sha(a.v112_precision) != AUTHORITATIVE_V112_PRECISION_SHA256:
        errors.append("v112_precision_sha")
    if not (
        p112.get("valid") and p112.get("attribution_ready")
        and p112.get("scientific_version") == "v48.112-OC-HCNC"
        and p112.get("preregistered_status") == "HETEROGENEOUS_CONSTRAINT_NORMAL_CONE_STOP"
    ):
        errors.append("v112_pipeline")
    if not (
        c112.get("valid") and c112.get("attribution_ready")
        and d112.get("status") == "HETEROGENEOUS_CONSTRAINT_NORMAL_CONE_STOP"
        and d112.get("next_branch")
        == "close_prefix_level_first_order_constraint_cone_then_preregister_candidate_option_executable_constraint_jacobian_audit_no_training_or_source_sweep"
    ):
        errors.append("v112_branch")

    historical = {"balanced": b112, "precision": p112r}
    # Preserve the exact V48.112 base owner/cohort/null before attributing the
    # executable-continuation intervention.
    for metric in ("support", "reserve"):
        for variant in ("balanced", "precision"):
            for role in ROLES:
                got = docs[variant]["base_cells"][role][f"{metric}_true"].get("auc")
                exp = historical[variant]["base_cells"][role][f"{metric}_true"].get("auc")
                if got is None or exp is None or abs(float(got) - float(exp)) > 1.0e-12:
                    errors.append(f"{variant}:{role}:{metric}:v112_base_identity")

    ngs = _action_gate(docs, "nominal_option", "support") if not errors else {"go": False, "local_order": False}
    ngr = _action_gate(docs, "nominal_option", "reserve") if not errors else {"go": False, "local_order": False}
    cgs = _action_gate(docs, "candidate_option", "support") if not errors else {"go": False, "local_order": False}
    cgr = _action_gate(docs, "candidate_option", "reserve") if not errors else {"go": False, "local_order": False}
    sis = _matched_increment_gate(docs, "support") if not errors else {"go": False, "rows": []}
    sir = _matched_increment_gate(docs, "reserve") if not errors else {"go": False, "rows": []}
    cis = _continuation_increment_gate(docs, historical, "support") if not errors else {"go": False, "rows": []}
    cir = _continuation_increment_gate(docs, historical, "reserve") if not errors else {"go": False, "rows": []}
    activity = _activity_gate(docs) if not errors else {
        "go": False,
        "adaptive_selector_activity_go": False,
        "reentry_contact_coverage_go": False,
        "rows": [],
    }

    core_go = bool(cgs.get("go") and cgr.get("go") and cis.get("go") and cir.get("go") and activity.get("go"))
    adaptive_go = bool(core_go and sis.get("go") and sir.get("go") and activity.get("adaptive_selector_activity_go"))
    reentry_coverage_go = bool(activity.get("reentry_contact_coverage_go"))
    unified_post_contact_go = bool(core_go and reentry_coverage_go)
    if errors:
        status = "V48_113_ENGINEERING_STOP"
        branch = "fix_v48_113_engineering_and_rerun_same_executable_constraint_jacobian_audit"
    elif adaptive_go and reentry_coverage_go:
        status = "EXECUTABLE_CONSTRAINT_JACOBIAN_ADAPTIVE_GO"
        branch = "promote_same_option_executable_constraint_jacobian_and_candidate_option_selector_then_preregister_one_nominal_invariant_carrier_no_source_or_boundary_sweep"
    elif adaptive_go:
        status = "EXECUTABLE_CONSTRAINT_JACOBIAN_ADAPTIVE_GO_REENTRY_COVERAGE_PENDING"
        branch = "retain_same_option_ecj_and_candidate_option_selector_but_require_observed_contact_reentry_coverage_before_unified_carrier_or_main_promotion"
    elif core_go and reentry_coverage_go:
        status = "EXECUTABLE_CONSTRAINT_JACOBIAN_CORE_GO_SELECTOR_STOP"
        branch = "promote_same_option_executable_constraint_jacobian_only_keep_option_selector_frozen_then_preregister_one_nominal_invariant_carrier"
    elif core_go:
        status = "EXECUTABLE_CONSTRAINT_JACOBIAN_CORE_GO_REENTRY_COVERAGE_PENDING"
        branch = "retain_same_option_ecj_but_require_observed_contact_reentry_coverage_before_unified_carrier_or_main_promotion_keep_selector_frozen"
    elif cgr.get("go") and cir.get("go") and not (cgs.get("go") and cis.get("go")):
        status = "EXECUTABLE_CONSTRAINT_JACOBIAN_RESERVE_ONLY"
        branch = "retain_debt_side_executable_jacobian_diagnostic_then_audit_support_establishment_on_same_option_no_new_capacity"
    elif cgs.get("go") and cis.get("go") and not (cgr.get("go") and cir.get("go")):
        status = "EXECUTABLE_CONSTRAINT_JACOBIAN_SUPPORT_ONLY"
        branch = "retain_support_side_executable_jacobian_diagnostic_then_audit_signed_debt_on_same_option_no_new_capacity"
    elif cgs.get("local_order") and cgr.get("local_order"):
        status = "EXECUTABLE_CONSTRAINT_JACOBIAN_LOCAL_ORDER_ONLY"
        branch = "same_executable_jacobian_features_then_one_convex_pairwise_audit_no_feature_or_source_change"
    else:
        status = "EXECUTABLE_CONSTRAINT_JACOBIAN_STOP"
        branch = "close_selected_option_first_order_jacobian_then_preregister_common_option_constraint_work_audit_no_capacity_or_regime_sweep"

    ident = _variant_identity(docs)
    decision = {
        "status": status,
        "next_branch": branch,
        "nominal_option_support_gate": ngs,
        "nominal_option_reserve_gate": ngr,
        "candidate_option_support_gate": cgs,
        "candidate_option_reserve_gate": cgr,
        "candidate_minus_nominal_option_support_gate": sis,
        "candidate_minus_nominal_option_reserve_gate": sir,
        "executable_continuation_vs_v48_112_prefix_support_gate": cis,
        "executable_continuation_vs_v48_112_prefix_reserve_gate": cir,
        "executable_option_activity_gate": activity,
        "core_executable_constraint_jacobian_go": core_go,
        "adaptive_candidate_option_selector_go": adaptive_go,
        "reentry_contact_coverage_go": reentry_coverage_go,
        "unified_post_contact_executable_jacobian_go": unified_post_contact_go,
        "capacity_matched_nominal_vs_candidate_option": True,
        "matched_dimension": MATCHED_DIM,
        "geometry_dimension": JACOBIAN_GEOMETRY_DIM,
        "constraint_names": ["clearance", "stopping", "route", "reentry"],
        "balanced_precision_metric_identity": ident,
        "power_diagnostics": _power(docs, "candidate_option"),
        "scientific_note": (
            "candidate and nominal option families share the same 220-D linear class; every response compares "
            "candidate and nominal under the same actuator-projected recovery option. The candidate family changes "
            "only which observation-only max-min executable option is selected. Cross-version continuation gates "
            "compare the 220-D executable readout to the authoritative 220-D V48.112 prefix-cone readout."
        ),
        "source_training_authorized": False,
        "broad_encoder_training_authorized": False,
        "boundary_transport_authorized": False,
        "dataset_reconstruction_authorized": False,
        "regime_conditioned_policy_authorized": False,
    }
    out = {
        "schema": "ocrap-v48.113-ecj-comparison-v1",
        "engineering_version": ENGINEERING_VERSION,
        "scientific_version": SCIENTIFIC_VERSION,
        "run_instance_id": a.run_id,
        "valid": not errors,
        "attribution_ready": not errors,
        "errors": errors,
        "experiment_type": "audit_only_candidate_option_executable_constraint_jacobian",
        "preregistered_decision": decision,
        "planner_parameters_trained": 0,
        "stage_i_parameters_trained": 0,
        "root_decoder_parameters_trained": 0,
        "source_parameters_trained": 0,
        "relative_ranker_modified": False,
        "regime_conditioning": False,
        "boundary_transport": False,
        "teacher_metadata_input_to_model": False,
        "test_roots_read": False,
        "v48_112_pipeline_sha256": _sha(a.v112_pipeline),
        "v48_112_comparison_sha256": _sha(a.v112_comparison),
        "v48_112_balanced_sha256": _sha(a.v112_balanced),
        "v48_112_precision_sha256": _sha(a.v112_precision),
        "authoritative_v48_112_comparison_sha256": AUTHORITATIVE_V112_COMPARISON_SHA256,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"valid": out["valid"], "status": status, "errors": errors}))
    return 0 if out["valid"] else 30


if __name__ == "__main__":
    raise SystemExit(main())
