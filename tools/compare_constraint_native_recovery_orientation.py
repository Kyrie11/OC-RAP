#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ocrap.audits.weak_root_recovery_set_flow import (
    ENGINEERING_VERSION,
    MATCHED_DIM,
    SCIENTIFIC_VERSION,
    TAIL_GEOMETRY_DIM,
)

AUTHORITATIVE_V115_PIPELINE_SHA256 = "eaf196f55c8b5d9ab32111e7c2ea27eacd7a01ce123fa50237d51562da15c4e2"
AUTHORITATIVE_V115_COMPARISON_SHA256 = "70d5fe0ed97ad08f1d571ba73add12a152e0b1115312b420ff481a233e037b42"
AUTHORITATIVE_V115_BALANCED_SHA256 = "9be23e261c8ba0f5e494eb136f663c3e4e960e2400d15af10b78e3ea41e50248"
AUTHORITATIVE_V115_PRECISION_SHA256 = "4a0a53eea966e8e9c2791b75e12ab520045d9a5b0d033a648833a8bd1c361118"
ROLES = ("dev_near", "dev_contact", "certificate_near", "certificate_contact")
SPACES = ("base", "tail_integral", "tail_work")


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
                positive.append([variant, role]); roles.add(role)
            if _ok(m.get("top1_vs_shuffled"), 0.10):
                top.append([variant, role]); top_roles.add(role)
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
    docs: dict[str, Any], treatment: str, control: str, metric: str, *, label: str,
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
            rows.append({"variant": variant, "role": role, f"{treatment}_auc": ta, f"{control}_auc": co, label: delta})
            if delta is not None and delta > 0.0:
                positive.append([variant, role]); roles.add(role)
            if delta is not None and delta >= 0.01:
                material.append([variant, role])
    go = len(positive) >= 6 and _cross(roles, 3) and len(material) >= 4
    return {"go": bool(go), "rows": rows, "positive_cells": positive, "material_cells": material, "roles": sorted(roles)}


def _historical_increment_gate(
    docs: dict[str, Any], historical: dict[str, Any], treatment: str, historical_space: str,
    metric: str, *, label: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    positive: list[list[str]] = []
    material: list[list[str]] = []
    roles: set[str] = set()
    for variant, d in docs.items():
        old = historical[variant]
        for role in ROLES:
            ta = d[f"{treatment}_cells"][role][f"{metric}_true"].get("auc")
            co = old[f"{historical_space}_cells"][role][f"{metric}_true"].get("auc")
            delta = None if ta is None or co is None else float(ta) - float(co)
            rows.append({
                "variant": variant, "role": role,
                f"v48_116_{treatment}_auc": ta,
                f"v48_115_{historical_space}_auc": co,
                label: delta,
            })
            if delta is not None and delta > 0.0:
                positive.append([variant, role]); roles.add(role)
            if delta is not None and delta >= 0.01:
                material.append([variant, role])
    go = len(positive) >= 6 and _cross(roles, 3) and len(material) >= 4
    return {"go": bool(go), "rows": rows, "positive_cells": positive, "material_cells": material, "roles": sorted(roles)}


def _activity_gate(docs: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    tail_work: set[str] = set()
    tail_integral: set[str] = set()
    physical_diverse: set[str] = set()
    multi_option: set[str] = set()
    nontrivial_tail: set[str] = set()
    weak_root_tail: set[str] = set()
    reentry_contact: set[str] = set()
    exact = True
    for variant, d in docs.items():
        ev = d.get("events") or {}
        for role in ROLES:
            event = ev.get(role) or {}
            diag = event.get("pair_diagnostics") or {}
            tail = event.get("tail_measure_diagnostics") or {}
            row = {
                "variant": variant,
                "role": role,
                "mean_common_valid_option_count": float(diag.get("mean_common_valid_option_count", 0.0)),
                "min_common_valid_option_count": int(diag.get("min_common_valid_option_count", 0)),
                "tail_work_nonzero_fraction": float(diag.get("tail_work_nonzero_fraction", 0.0)),
                "tail_integral_nonzero_fraction": float(diag.get("tail_integral_nonzero_fraction", 0.0)),
                "option_flow_diverse_fraction": float(diag.get("option_flow_diverse_fraction", 0.0)),
                "reentry_set_available_fraction": float(diag.get("reentry_set_available_fraction", 0.0)),
                "max_tail_work_conservation_error": float(diag.get("max_tail_work_conservation_error", 1.0)),
                "max_tail_option_permutation_invariance_error": float(diag.get("max_tail_option_permutation_invariance_error", 1.0)),
                "tail_group_count": int(tail.get("group_count", 0)),
                "mean_tail_positive_option_count": float(tail.get("mean_positive_option_count", 0.0)),
                "min_tail_positive_option_count": int(tail.get("min_positive_option_count", 0)),
                "mean_tail_effective_option_count": float(tail.get("mean_effective_option_count", 0.0)),
                "mean_tail_outer_positive_root_count": float(tail.get("mean_outer_positive_root_count", 0.0)),
                "max_option_weight_sum_error": float(tail.get("max_option_weight_sum_error", 1.0)),
                "max_cotangent_mass_error": float(tail.get("max_cotangent_mass_error", 1.0)),
            }
            rows.append(row)
            if row["tail_work_nonzero_fraction"] > 0.0: tail_work.add(role)
            if row["tail_integral_nonzero_fraction"] > 0.0: tail_integral.add(role)
            if row["option_flow_diverse_fraction"] > 0.0: physical_diverse.add(role)
            if row["min_common_valid_option_count"] >= 2: multi_option.add(role)
            if row["mean_tail_positive_option_count"] >= 2.0: nontrivial_tail.add(role)
            if row["mean_tail_outer_positive_root_count"] >= 1.0: weak_root_tail.add(role)
            if "contact" in role and row["reentry_set_available_fraction"] > 0.0: reentry_contact.add(role)
            if (
                row["max_tail_work_conservation_error"] > 1.0e-10
                or row["max_tail_option_permutation_invariance_error"] > 1.0e-10
                or row["max_option_weight_sum_error"] > 1.0e-10
                or row["max_cotangent_mass_error"] > 1.0e-10
            ):
                exact = False
    core = bool(
        exact and _cross(tail_work, 3) and _cross(tail_integral, 3)
        and _cross(physical_diverse, 3) and _cross(multi_option, 3)
        and _cross(weak_root_tail, 3) and len(reentry_contact) >= 1
    )
    nontrivial = bool(_cross(nontrivial_tail, 3))
    return {
        "go": bool(core and nontrivial),
        "core_activity_go": core,
        "nontrivial_tail_measure_go": nontrivial,
        "exact_tail_contract_go": exact,
        "tail_work_nonzero_roles": sorted(tail_work),
        "tail_integral_nonzero_roles": sorted(tail_integral),
        "physical_option_flow_diverse_roles": sorted(physical_diverse),
        "multi_option_roles": sorted(multi_option),
        "nontrivial_tail_option_roles": sorted(nontrivial_tail),
        "weak_root_tail_roles": sorted(weak_root_tail),
        "reentry_contact_roles": sorted(reentry_contact),
        "reentry_contact_coverage_go": bool(reentry_contact),
        "rows": rows,
    }


def _variant_identity(docs: dict[str, Any]) -> dict[str, Any]:
    a, b = docs["balanced"], docs["precision"]
    diffs: list[str] = []
    for space in SPACES:
        for role in ROLES:
            for metric in ("support_true", "support_shuffled", "reserve_true", "reserve_shuffled"):
                ma = a[f"{space}_cells"][role][metric]
                mb = b[f"{space}_cells"][role][metric]
                for k in ("auc", "top1", "auc_vs_shuffled", "top1_vs_shuffled", "rows", "positive_rows", "negative_rows", "powered_groups"):
                    if ma.get(k) != mb.get(k): diffs.append(f"{space}:{role}:{metric}:{k}")
    return {"exact": not diffs, "differences": diffs, "effective_unique_roles_if_exact": 4 if not diffs else 4}


def _power(docs: dict[str, Any], space: str) -> list[dict[str, Any]]:
    out = []
    for variant, d in docs.items():
        for role in ROLES:
            for axis in ("support", "reserve"):
                m = d[f"{space}_cells"][role][f"{axis}_true"]
                out.append({
                    "variant": variant, "role": role, "axis": axis,
                    "rows": m.get("rows"), "positive_rows": m.get("positive_rows"),
                    "negative_rows": m.get("negative_rows"), "powered_groups": m.get("powered_groups"),
                    "underpowered": bool((m.get("powered_groups") or 0) < 2),
                })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--balanced", type=Path, required=True)
    ap.add_argument("--precision", type=Path, required=True)
    ap.add_argument("--v115-pipeline", type=Path, required=True)
    ap.add_argument("--v115-comparison", type=Path, required=True)
    ap.add_argument("--v115-balanced", type=Path, required=True)
    ap.add_argument("--v115-precision", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    errors: list[str] = []

    def read(p: Path, name: str) -> dict[str, Any]:
        try:
            return json.loads(p.read_text())
        except Exception as exc:
            errors.append(f"{name}:json:{type(exc).__name__}")
            return {}

    docs = {"balanced": read(a.balanced, "balanced"), "precision": read(a.precision, "precision")}
    p115 = read(a.v115_pipeline, "v115_pipeline")
    c115 = read(a.v115_comparison, "v115_comparison")
    b115 = read(a.v115_balanced, "v115_balanced")
    q115 = read(a.v115_precision, "v115_precision")

    expected_sha = {
        a.v115_pipeline: AUTHORITATIVE_V115_PIPELINE_SHA256,
        a.v115_comparison: AUTHORITATIVE_V115_COMPARISON_SHA256,
        a.v115_balanced: AUTHORITATIVE_V115_BALANCED_SHA256,
        a.v115_precision: AUTHORITATIVE_V115_PRECISION_SHA256,
    }
    for path, want in expected_sha.items():
        if not path.is_file() or _sha(path) != want:
            errors.append(f"v115_sha:{path.name}")

    for variant, d in docs.items():
        if not (
            d.get("valid") and d.get("engineering_version") == ENGINEERING_VERSION
            and d.get("scientific_version") == SCIENTIFIC_VERSION and d.get("variant") == variant
            and d.get("run_instance_id") == a.run_id and d.get("audit_only")
            and d.get("convex_closed_form_ridge") and d.get("strictly_convex_unique_solution")
            and d.get("capacity_matched_all_tail_families") and d.get("matched_family_dimension") == MATCHED_DIM
            and d.get("tail_geometry_dimension") == TAIL_GEOMETRY_DIM
            and d.get("candidate_independent_tail_measure") is True
            and d.get("frozen_root_decoder_read_only") is True
            and d.get("frozen_margin_head_read_only") is True
            and d.get("same_option_inside_each_weighted_summand") is True
            and d.get("teacher_npz_fields_loaded_into_feature_path") == ["root_valid"]
            and d.get("teacher_margin_probability_compatibility_fields_used") is False
            and d.get("teacher_future_fields_used") is False
            and d.get("root_decoder_parameters_trained") == 0
            and d.get("regime_conditioning") is False and d.get("boundary_transport") is False
            and d.get("test_roots_read") is False
        ):
            errors.append(f"{variant}:contract")

    d115 = c115.get("preregistered_decision") or {}
    if not (
        p115.get("valid") and p115.get("attribution_ready")
        and p115.get("engineering_version") == "v48.115.0-OC-RSCF"
        and p115.get("preregistered_status") == "RECOVERY_SET_CONSTRAINT_FLOW_STOP"
    ):
        errors.append("v115_pipeline")
    if not (
        c115.get("valid") and c115.get("attribution_ready")
        and d115.get("status") == "RECOVERY_SET_CONSTRAINT_FLOW_STOP"
        and d115.get("next_branch") == "close_observation_only_option_set_mean_flow_then_preregister_ocmero_weak_root_conditioned_recovery_set_flow_audit_frozen_roots_no_training_or_capacity_sweep"
    ):
        errors.append("v115_branch")

    historical = {"balanced": b115, "precision": q115}
    for variant in ("balanced", "precision"):
        if docs[variant].get("checkpoint_sha256") != historical[variant].get("checkpoint_sha256"):
            errors.append(f"{variant}:checkpoint_identity")
        for metric in ("support", "reserve"):
            for role in ROLES:
                got = docs[variant]["base_cells"][role][f"{metric}_true"].get("auc")
                exp = historical[variant]["base_cells"][role][f"{metric}_true"].get("auc")
                if got is None or exp is None or abs(float(got) - float(exp)) > 1.0e-12:
                    errors.append(f"{variant}:{role}:{metric}:v115_base_identity")

    if errors:
        fake = {"go": False, "local_order": False, "rows": []}
        ti_s = ti_r = tw_s = tw_r = fake
        hist_i_s = hist_i_r = hist_w_s = hist_w_r = fake
        decomp_s = decomp_r = fake
        activity = {"go": False, "reentry_contact_coverage_go": False, "rows": []}
    else:
        ti_s = _action_gate(docs, "tail_integral", "support")
        ti_r = _action_gate(docs, "tail_integral", "reserve")
        tw_s = _action_gate(docs, "tail_work", "support")
        tw_r = _action_gate(docs, "tail_work", "reserve")
        hist_i_s = _historical_increment_gate(docs, historical, "tail_integral", "set_integral", "support", label="tail_minus_uniform_integral")
        hist_i_r = _historical_increment_gate(docs, historical, "tail_integral", "set_integral", "reserve", label="tail_minus_uniform_integral")
        hist_w_s = _historical_increment_gate(docs, historical, "tail_work", "set_work", "support", label="tail_minus_uniform_work")
        hist_w_r = _historical_increment_gate(docs, historical, "tail_work", "set_work", "reserve", label="tail_minus_uniform_work")
        decomp_s = _within_increment_gate(docs, "tail_work", "tail_integral", "support", label="tail_work_minus_tail_integral")
        decomp_r = _within_increment_gate(docs, "tail_work", "tail_integral", "reserve", label="tail_work_minus_tail_integral")
        activity = _activity_gate(docs)

    integral_core = bool(ti_s.get("go") and ti_r.get("go") and hist_i_s.get("go") and hist_i_r.get("go") and activity.get("go"))
    work_core = bool(tw_s.get("go") and tw_r.get("go") and hist_w_s.get("go") and hist_w_r.get("go") and activity.get("go"))
    decomposition = bool(work_core and decomp_s.get("go") and decomp_r.get("go"))
    reentry = bool(activity.get("reentry_contact_coverage_go"))

    if errors:
        status = "V48_116_ENGINEERING_STOP"
        branch = "fix_v48_116_engineering_and_rerun_same_weak_root_cotangent_flow_audit"
    elif work_core and decomposition and reentry:
        status = "WEAK_ROOT_RECOVERY_SET_FLOW_GO"
        branch = "promote_nominal_ocmero_tail_weighted_signed_constraint_flow_then_preregister_exactly_one_main_carrier_integration_no_source_or_boundary_sweep"
    elif work_core and reentry:
        status = "WEAK_ROOT_RECOVERY_SET_FLOW_GO_SIGNED_DECOMPOSITION_NOT_REQUIRED"
        branch = "promote_nominal_ocmero_tail_weighted_recovery_flow_then_preregister_exactly_one_main_carrier_integration"
    elif integral_core and reentry:
        status = "WEAK_ROOT_RECOVERY_SET_INTEGRAL_GO"
        branch = "promote_nominal_ocmero_tail_weighted_integral_flow_only_then_preregister_exactly_one_main_carrier_integration"
    elif tw_s.get("go") and hist_w_s.get("go") and not (tw_r.get("go") and hist_w_r.get("go")):
        status = "WEAK_ROOT_RECOVERY_SET_FLOW_SUPPORT_ONLY"
        branch = "retain_weak_root_support_flow_only_then_audit_tail_boundary_debt_hitting_functional_no_capacity_or_regime_sweep"
    elif tw_r.get("go") and hist_w_r.get("go") and not (tw_s.get("go") and hist_w_s.get("go")):
        status = "WEAK_ROOT_RECOVERY_SET_FLOW_RESERVE_ONLY"
        branch = "retain_weak_root_debt_flow_only_then_audit_tail_boundary_support_hitting_functional_no_capacity_or_regime_sweep"
    elif tw_s.get("local_order") and tw_r.get("local_order"):
        status = "WEAK_ROOT_RECOVERY_SET_FLOW_LOCAL_ORDER_ONLY"
        branch = "one_convex_pairwise_audit_on_exact_tail_weighted_features_no_feature_or_source_change"
    else:
        status = "WEAK_ROOT_RECOVERY_SET_FLOW_STOP"
        branch = "close_first_order_nominal_ocmero_cotangent_option_pushforward_then_preregister_tail_boundary_crossing_flow_audit_no_training_capacity_regime_or_source_sweep"

    ident = _variant_identity(docs)
    decision = {
        "status": status,
        "next_branch": branch,
        "tail_integral_support_gate": ti_s,
        "tail_integral_reserve_gate": ti_r,
        "tail_work_support_gate": tw_s,
        "tail_work_reserve_gate": tw_r,
        "tail_integral_vs_v48_115_uniform_integral_support_gate": hist_i_s,
        "tail_integral_vs_v48_115_uniform_integral_reserve_gate": hist_i_r,
        "tail_work_vs_v48_115_uniform_work_support_gate": hist_w_s,
        "tail_work_vs_v48_115_uniform_work_reserve_gate": hist_w_r,
        "tail_work_vs_tail_integral_support_gate": decomp_s,
        "tail_work_vs_tail_integral_reserve_gate": decomp_r,
        "weak_root_tail_activity_gate": activity,
        "weak_root_integral_core_go": integral_core,
        "weak_root_work_core_go": work_core,
        "signed_reserve_debt_tail_decomposition_go": decomposition,
        "reentry_contact_coverage_go": reentry,
        "capacity_matched_all_families": True,
        "matched_dimension": MATCHED_DIM,
        "geometry_dimension": TAIL_GEOMETRY_DIM,
        "work_bins": 8,
        "constraint_names": ["clearance", "stopping", "route", "reentry"],
        "tail_measure": "nominal_native_ocmero_nested_lcvar_cotangent_pushforward_over_recovery_options",
        "balanced_precision_metric_identity": ident,
        "power_diagnostics": _power(docs, "tail_work"),
        "scientific_note": (
            "V48.116 changes only the recovery-option integration measure.  The physical same-option full-horizon constraint flow, 220-D convex capacity, cohorts, horizon and signed work semantics are held fixed. "
            "A frozen nominal native OC-MERO root/margin prediction defines the exact nested-LCVAR cotangent dR_dep/dM; its root mass is pushed forward to a candidate-independent option measure and used to integrate executable flow. "
            "No teacher m_star/root_probs/c_star/future value enters the feature path; only the frozen root-validity support mask is reused.  No learned root adapter, root retraining, regime label or candidate-conditioned option selector is introduced."
        ),
        "source_training_authorized": False,
        "broad_encoder_training_authorized": False,
        "boundary_transport_authorized": False,
        "dataset_reconstruction_authorized": False,
        "regime_conditioned_policy_authorized": False,
    }
    out = {
        "schema": "ocrap-v48.116-wrcf-comparison-v1",
        "engineering_version": ENGINEERING_VERSION,
        "scientific_version": SCIENTIFIC_VERSION,
        "run_instance_id": a.run_id,
        "valid": not errors,
        "attribution_ready": not errors,
        "errors": errors,
        "experiment_type": "audit_only_nominal_ocmero_weak_root_cotangent_recovery_set_flow",
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
        "v48_115_pipeline_sha256": _sha(a.v115_pipeline),
        "v48_115_comparison_sha256": _sha(a.v115_comparison),
        "v48_115_balanced_sha256": _sha(a.v115_balanced),
        "v48_115_precision_sha256": _sha(a.v115_precision),
        "authoritative_v48_115_comparison_sha256": AUTHORITATIVE_V115_COMPARISON_SHA256,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"valid": out["valid"], "status": status, "errors": errors}))
    return 0 if out["valid"] else 30


if __name__ == "__main__":
    raise SystemExit(main())
