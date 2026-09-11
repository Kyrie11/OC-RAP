#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ocrap.audits.common_option_constraint_work import (
    ENGINEERING_VERSION,
    MATCHED_DIM,
    SCIENTIFIC_VERSION,
    WORK_GEOMETRY_DIM,
)

AUTHORITATIVE_V113_PIPELINE_SHA256 = "141164bc3881f734ec64963cf32c1f1b07ac0180e4835cc27b078b02b923930a"
AUTHORITATIVE_V113_COMPARISON_SHA256 = "6ca24f95aaca4198eb565547abdcdd4d7d37aebc2bdf9758b6b003653411cc00"
AUTHORITATIVE_V113_BALANCED_SHA256 = "544ff65a97ccb54cb90e081e437dfc165f6fb3cda08932c67656e7671efa7baf"
AUTHORITATIVE_V113_PRECISION_SHA256 = "ee709ea955731e05277709f0622a9d8cf6bfab58962f01372eb392c3073e4048"
ROLES = ("dev_near", "dev_contact", "certificate_near", "certificate_contact")
SPACES = ("base", "nominal_integral", "candidate_integral", "nominal_work", "candidate_work")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ok(v: Any, threshold: float) -> bool:
    return v is not None and float(v) >= threshold


def _cross(roles: set[str], n: int) -> bool:
    return len(roles) >= n and any("near" in x for x in roles) and any("contact" in x for x in roles)


def _action_gate(docs: dict[str, Any], space: str, metric: str) -> dict[str, Any]:
    """Absolute orientation vs within-group whole-row candidate shuffle.

    The legacy 6/8 count is retained for continuity, but balanced/precision
    exact identity is explicitly reported and the gate additionally requires
    at least 3/4 unique roles with Near+Contact coverage.
    """
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


def _within_increment_gate(
    docs: dict[str, Any],
    treatment: str,
    control: str,
    metric: str,
    *,
    label: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    positive: list[list[str]] = []
    material: list[list[str]] = []
    roles: set[str] = set()
    for variant, d in docs.items():
        for role in ROLES:
            ta = d[f"{treatment}_cells"][role][f"{metric}_true"].get("auc")
            co = d[f"{control}_cells"][role][f"{metric}_true"].get("auc")
            delta = None if ta is None or co is None else float(ta) - float(co)
            rows.append({
                "variant": variant,
                "role": role,
                f"{treatment}_auc": ta,
                f"{control}_auc": co,
                label: delta,
            })
            if delta is not None and delta > 0.0:
                positive.append([variant, role])
                roles.add(role)
            if delta is not None and delta >= 0.01:
                material.append([variant, role])
    go = len(positive) >= 6 and _cross(roles, 3) and len(material) >= 4
    return {
        "go": bool(go),
        "positive_cells": positive,
        "material_cells": material,
        "roles": sorted(roles),
        "rows": rows,
    }


def _historical_increment_gate(
    docs: dict[str, Any], hist: dict[str, Any], space: str, metric: str, *, label: str
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    positive: list[list[str]] = []
    material: list[list[str]] = []
    roles: set[str] = set()
    for variant in ("balanced", "precision"):
        for role in ROLES:
            now = docs[variant][f"{space}_cells"][role][f"{metric}_true"].get("auc")
            old = hist[variant]["candidate_option_cells"][role][f"{metric}_true"].get("auc")
            delta = None if now is None or old is None else float(now) - float(old)
            rows.append({
                "variant": variant,
                "role": role,
                f"v48_114_{space}_auc": now,
                "v48_113_candidate_option_auc": old,
                label: delta,
            })
            if delta is not None and delta > 0.0:
                positive.append([variant, role])
                roles.add(role)
            if delta is not None and delta >= 0.01:
                material.append([variant, role])
    go = len(positive) >= 6 and _cross(roles, 3) and len(material) >= 4
    return {
        "go": bool(go),
        "positive_cells": positive,
        "material_cells": material,
        "roles": sorted(roles),
        "rows": rows,
    }


def _variant_identity(docs: dict[str, Any]) -> dict[str, Any]:
    diffs: list[str] = []
    for space in SPACES:
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
                    "variant": variant,
                    "role": role,
                    "axis": axis,
                    "rows": m.get("rows"),
                    "positive_rows": m.get("positive_rows"),
                    "negative_rows": m.get("negative_rows"),
                    "powered_groups": m.get("powered_groups"),
                    "underpowered": bool(
                        (m.get("powered_groups") or 0) < 3
                        or (m.get("positive_rows") or 0) < 5
                        or (m.get("negative_rows") or 0) < 5
                    ),
                })
    return out


def _activity_gate(docs: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    constraint_diverse_roles: set[str] = set()
    response_roles: set[str] = set()
    reserve_work_roles: set[str] = set()
    debt_work_contact_roles: set[str] = set()
    reentry_contact_roles: set[str] = set()
    option_switch_roles: set[str] = set()
    mode_diverse_roles: set[str] = set()
    conservation_ok = True
    for variant in ("balanced", "precision"):
        for role in ROLES:
            diag = ((docs[variant].get("events") or {}).get(role, {}).get("pair_diagnostics") or {})
            active = diag.get("candidate_selected_active_type_counts", {}) or {}
            switch = float(diag.get("option_switch_fraction", 0.0) or 0.0)
            modes = diag.get("candidate_selected_mode_counts", {}) or {}
            reentry = float(diag.get("candidate_selected_reentry_active_fraction", 0.0) or 0.0)
            response = float(diag.get("integral_response_nonzero_fraction", 0.0) or 0.0)
            reserve = float(diag.get("reserve_work_nonzero_fraction", 0.0) or 0.0)
            debt = float(diag.get("debt_work_nonzero_fraction", 0.0) or 0.0)
            cerr = float(diag.get("max_work_conservation_error", 0.0) or 0.0)
            conservation_ok = conservation_ok and cerr <= 1.0e-10
            rows.append({
                "variant": variant,
                "role": role,
                "option_switch_fraction": switch,
                "candidate_selected_mode_counts": modes,
                "candidate_selected_active_type_counts": active,
                "candidate_selected_reentry_active_fraction": reentry,
                "field_reentry_available_fraction": float(diag.get("field_reentry_available_fraction", 0.0) or 0.0),
                "integral_response_nonzero_fraction": response,
                "reserve_work_nonzero_fraction": reserve,
                "debt_work_nonzero_fraction": debt,
                "max_work_conservation_error": cerr,
            })
            if sum(int(v) > 0 for v in active.values()) >= 2:
                constraint_diverse_roles.add(role)
            if response > 0.0:
                response_roles.add(role)
            if reserve > 0.0:
                reserve_work_roles.add(role)
            if "contact" in role and debt > 0.0:
                debt_work_contact_roles.add(role)
            if "contact" in role and reentry > 0.0:
                reentry_contact_roles.add(role)
            if switch > 0.0:
                option_switch_roles.add(role)
            if sum(int(v) > 0 for v in modes.values()) >= 2:
                mode_diverse_roles.add(role)
    core = (
        conservation_ok
        and _cross(constraint_diverse_roles, 3)
        and _cross(response_roles, 3)
        and _cross(reserve_work_roles, 3)
        and len(debt_work_contact_roles) >= 1
    )
    selector = _cross(option_switch_roles, 3) and _cross(mode_diverse_roles, 3)
    reentry = len(reentry_contact_roles) >= 1
    return {
        "go": bool(core),
        "selector_activity_go": bool(selector),
        "reentry_contact_coverage_go": bool(reentry),
        "work_conservation_go": bool(conservation_ok),
        "constraint_diverse_roles": sorted(constraint_diverse_roles),
        "integral_response_roles": sorted(response_roles),
        "reserve_work_roles": sorted(reserve_work_roles),
        "debt_work_contact_roles": sorted(debt_work_contact_roles),
        "reentry_contact_roles": sorted(reentry_contact_roles),
        "option_switch_roles": sorted(option_switch_roles),
        "mode_diverse_roles": sorted(mode_diverse_roles),
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
        "capacity_matched_all_work_families": True,
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
    if int(d.get("work_geometry_dimension", -1)) != WORK_GEOMETRY_DIM:
        errors.append(f"{variant}:geometry_dim")
    if int(d.get("matched_family_dimension", -1)) != MATCHED_DIM:
        errors.append(f"{variant}:matched_dim")
    if d.get("constraint_work_channels") != ["positive_reserve_work", "negative_debt_repayment_work"]:
        errors.append(f"{variant}:work_channels")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser()
    for key in ("balanced", "precision", "v113_pipeline", "v113_comparison", "v113_balanced", "v113_precision"):
        ap.add_argument("--" + key.replace("_", "-"), dest=key, type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    errors: list[str] = []

    docs = {v: json.loads(getattr(a, v).read_text()) for v in ("balanced", "precision")}
    for variant, obj in docs.items():
        errors += _result_errors(obj, variant)
        if obj.get("run_instance_id") != a.run_id:
            errors.append(f"{variant}:run_instance_id")

    p113 = json.loads(a.v113_pipeline.read_text())
    c113 = json.loads(a.v113_comparison.read_text())
    b113 = json.loads(a.v113_balanced.read_text())
    q113 = json.loads(a.v113_precision.read_text())
    d113 = c113.get("preregistered_decision") or {}

    for path, want, name in (
        (a.v113_pipeline, AUTHORITATIVE_V113_PIPELINE_SHA256, "v113_pipeline_sha"),
        (a.v113_comparison, AUTHORITATIVE_V113_COMPARISON_SHA256, "v113_comparison_sha"),
        (a.v113_balanced, AUTHORITATIVE_V113_BALANCED_SHA256, "v113_balanced_sha"),
        (a.v113_precision, AUTHORITATIVE_V113_PRECISION_SHA256, "v113_precision_sha"),
    ):
        if _sha(path) != want:
            errors.append(name)
    if not (
        p113.get("valid") and p113.get("attribution_ready")
        and p113.get("scientific_version") == "v48.113-OC-ECJ"
        and p113.get("preregistered_status") == "EXECUTABLE_CONSTRAINT_JACOBIAN_STOP"
    ):
        errors.append("v113_pipeline")
    if not (
        c113.get("valid") and c113.get("attribution_ready")
        and d113.get("status") == "EXECUTABLE_CONSTRAINT_JACOBIAN_STOP"
        and d113.get("next_branch")
        == "close_selected_option_first_order_jacobian_then_preregister_common_option_constraint_work_audit_no_capacity_or_regime_sweep"
    ):
        errors.append("v113_branch")

    historical = {"balanced": b113, "precision": q113}
    # Exact cohort/base/checkpoint identity is required before cross-version
    # attribution.  This turns the V48.113 pointwise ECJ into an immutable
    # equal-capacity historical control rather than silently refitting it.
    for variant in ("balanced", "precision"):
        if docs[variant].get("checkpoint_sha256") != historical[variant].get("checkpoint_sha256"):
            errors.append(f"{variant}:checkpoint_identity")
        for metric in ("support", "reserve"):
            for role in ROLES:
                got = docs[variant]["base_cells"][role][f"{metric}_true"].get("auc")
                exp = historical[variant]["base_cells"][role][f"{metric}_true"].get("auc")
                if got is None or exp is None or abs(float(got) - float(exp)) > 1.0e-12:
                    errors.append(f"{variant}:{role}:{metric}:v113_base_identity")

    if errors:
        action = lambda *args, **kwargs: {"go": False, "local_order": False, "rows": []}
        increment = lambda *args, **kwargs: {"go": False, "rows": []}
        iw_s = action(); iw_r = action(); cw_s = action(); cw_r = action(); nw_s = action(); nw_r = action()
        hist_i_s = increment(); hist_i_r = increment(); hist_w_s = increment(); hist_w_r = increment()
        decomp_s = increment(); decomp_r = increment(); selector_s = increment(); selector_r = increment()
        activity = {"go": False, "selector_activity_go": False, "reentry_contact_coverage_go": False, "rows": []}
    else:
        iw_s = _action_gate(docs, "candidate_integral", "support")
        iw_r = _action_gate(docs, "candidate_integral", "reserve")
        cw_s = _action_gate(docs, "candidate_work", "support")
        cw_r = _action_gate(docs, "candidate_work", "reserve")
        nw_s = _action_gate(docs, "nominal_work", "support")
        nw_r = _action_gate(docs, "nominal_work", "reserve")
        hist_i_s = _historical_increment_gate(docs, historical, "candidate_integral", "support", label="integral_minus_v48_113_pointwise")
        hist_i_r = _historical_increment_gate(docs, historical, "candidate_integral", "reserve", label="integral_minus_v48_113_pointwise")
        hist_w_s = _historical_increment_gate(docs, historical, "candidate_work", "support", label="work_minus_v48_113_pointwise")
        hist_w_r = _historical_increment_gate(docs, historical, "candidate_work", "reserve", label="work_minus_v48_113_pointwise")
        decomp_s = _within_increment_gate(docs, "candidate_work", "candidate_integral", "support", label="work_minus_integral")
        decomp_r = _within_increment_gate(docs, "candidate_work", "candidate_integral", "reserve", label="work_minus_integral")
        selector_s = _within_increment_gate(docs, "candidate_work", "nominal_work", "support", label="candidate_minus_nominal_work")
        selector_r = _within_increment_gate(docs, "candidate_work", "nominal_work", "reserve", label="candidate_minus_nominal_work")
        activity = _activity_gate(docs)

    integral_core_go = bool(iw_s.get("go") and iw_r.get("go") and hist_i_s.get("go") and hist_i_r.get("go") and activity.get("go"))
    work_core_go = bool(cw_s.get("go") and cw_r.get("go") and hist_w_s.get("go") and hist_w_r.get("go") and activity.get("go"))
    signed_decomposition_go = bool(work_core_go and decomp_s.get("go") and decomp_r.get("go"))
    adaptive_selector_go = bool(work_core_go and selector_s.get("go") and selector_r.get("go") and activity.get("selector_activity_go"))
    reentry_coverage_go = bool(activity.get("reentry_contact_coverage_go"))
    unified_work_go = bool(work_core_go and reentry_coverage_go)

    if errors:
        status = "V48_114_ENGINEERING_STOP"
        branch = "fix_v48_114_engineering_and_rerun_same_common_option_constraint_work_audit"
    elif signed_decomposition_go and reentry_coverage_go:
        status = "COMMON_OPTION_CONSTRAINT_WORK_GO"
        branch = (
            "promote_full_horizon_signed_constraint_work"
            + ("_and_candidate_option_selector" if adaptive_selector_go else "_keep_selector_frozen")
            + "_then_preregister_one_nominal_invariant_carrier_no_source_or_boundary_sweep"
        )
    elif signed_decomposition_go:
        status = "COMMON_OPTION_CONSTRAINT_WORK_GO_REENTRY_COVERAGE_PENDING"
        branch = "retain_signed_constraint_work_but_require_observed_contact_reentry_coverage_before_unified_carrier_promotion"
    elif integral_core_go and not signed_decomposition_go:
        status = "FULL_HORIZON_INTEGRAL_RESPONSE_GO_SIGNED_DECOMPOSITION_STOP"
        branch = "promote_temporal_integral_response_only_keep_reserve_debt_decomposition_out_of_main_then_preregister_one_carrier"
    elif cw_r.get("go") and hist_w_r.get("go") and not (cw_s.get("go") and hist_w_s.get("go")):
        status = "COMMON_OPTION_CONSTRAINT_WORK_RESERVE_ONLY"
        branch = "retain_debt_repayment_work_diagnostic_then_audit_support_establishment_as_set_valued_common_option_survival_no_new_capacity"
    elif cw_s.get("go") and hist_w_s.get("go") and not (cw_r.get("go") and hist_w_r.get("go")):
        status = "COMMON_OPTION_CONSTRAINT_WORK_SUPPORT_ONLY"
        branch = "retain_reserve_preservation_work_diagnostic_then_audit_contact_debt_occupation_measure_no_regime_specific_policy"
    elif cw_s.get("local_order") and cw_r.get("local_order"):
        status = "COMMON_OPTION_CONSTRAINT_WORK_LOCAL_ORDER_ONLY"
        branch = "same_constraint_work_features_then_one_convex_pairwise_audit_no_feature_or_source_change"
    else:
        status = "COMMON_OPTION_CONSTRAINT_WORK_STOP"
        branch = "close_selected_option_fixed_bin_work_family_then_preregister_selector_free_recovery_set_constraint_flow_audit_no_capacity_or_regime_sweep"

    ident = _variant_identity(docs)
    decision = {
        "status": status,
        "next_branch": branch,
        "candidate_integral_support_gate": iw_s,
        "candidate_integral_reserve_gate": iw_r,
        "candidate_work_support_gate": cw_s,
        "candidate_work_reserve_gate": cw_r,
        "nominal_work_support_gate": nw_s,
        "nominal_work_reserve_gate": nw_r,
        "integral_vs_v48_113_pointwise_support_gate": hist_i_s,
        "integral_vs_v48_113_pointwise_reserve_gate": hist_i_r,
        "work_vs_v48_113_pointwise_support_gate": hist_w_s,
        "work_vs_v48_113_pointwise_reserve_gate": hist_w_r,
        "signed_work_vs_integral_support_gate": decomp_s,
        "signed_work_vs_integral_reserve_gate": decomp_r,
        "candidate_vs_nominal_work_support_gate": selector_s,
        "candidate_vs_nominal_work_reserve_gate": selector_r,
        "constraint_work_activity_gate": activity,
        "full_horizon_integral_core_go": integral_core_go,
        "common_option_constraint_work_core_go": work_core_go,
        "signed_reserve_debt_decomposition_go": signed_decomposition_go,
        "adaptive_candidate_option_selector_go": adaptive_selector_go,
        "reentry_contact_coverage_go": reentry_coverage_go,
        "unified_post_contact_constraint_work_go": unified_work_go,
        "capacity_matched_all_families": True,
        "matched_dimension": MATCHED_DIM,
        "geometry_dimension": WORK_GEOMETRY_DIM,
        "work_bins": 8,
        "constraint_names": ["clearance", "stopping", "route", "reentry"],
        "work_channels": ["positive_reserve_work", "negative_debt_repayment_work"],
        "balanced_precision_metric_identity": ident,
        "power_diagnostics": _power(docs, "candidate_work"),
        "scientific_note": (
            "V48.114 keeps the V48.113 selector, constraints, horizon, base 156-D response, and 220-D convex capacity fixed. "
            "The integral arm isolates full-horizon accumulation from sparse point samples using the historical ECJ channels. "
            "The work arm then changes only the two temporal channels to positive-reserve work and negative-debt repayment, whose sum equals the signed response exactly. "
            "All candidate-minus-nominal comparisons use one common actuator-projected recovery option identity; no teacher future or regime label enters the feature path."
        ),
        "source_training_authorized": False,
        "broad_encoder_training_authorized": False,
        "boundary_transport_authorized": False,
        "dataset_reconstruction_authorized": False,
        "regime_conditioned_policy_authorized": False,
    }
    out = {
        "schema": "ocrap-v48.114-ccw-comparison-v1",
        "engineering_version": ENGINEERING_VERSION,
        "scientific_version": SCIENTIFIC_VERSION,
        "run_instance_id": a.run_id,
        "valid": not errors,
        "attribution_ready": not errors,
        "errors": errors,
        "experiment_type": "audit_only_common_option_full_horizon_constraint_work",
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
        "v48_113_pipeline_sha256": _sha(a.v113_pipeline),
        "v48_113_comparison_sha256": _sha(a.v113_comparison),
        "v48_113_balanced_sha256": _sha(a.v113_balanced),
        "v48_113_precision_sha256": _sha(a.v113_precision),
        "authoritative_v48_113_comparison_sha256": AUTHORITATIVE_V113_COMPARISON_SHA256,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"valid": out["valid"], "status": status, "errors": errors}))
    return 0 if out["valid"] else 30


if __name__ == "__main__":
    raise SystemExit(main())
