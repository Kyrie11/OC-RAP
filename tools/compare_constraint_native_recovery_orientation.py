#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ocrap.audits.recovery_set_constraint_flow import (
    ENGINEERING_VERSION,
    MATCHED_DIM,
    SCIENTIFIC_VERSION,
    SET_GEOMETRY_DIM,
)

AUTHORITATIVE_V114_PIPELINE_SHA256 = "4887300ad1af2232c36ce4d8101ca3526ce7eeae056be846b114f91b14e43ece"
AUTHORITATIVE_V114_COMPARISON_SHA256 = "7ce7d09809da348fb3229c0d89338858fa05fe461301088a9a0e6392bf8c0319"
AUTHORITATIVE_V114_BALANCED_SHA256 = "432dce3a32a5c4a52ad10f77ca4ff823a9d2ded14502efff9c6eb689dc5e2b21"
AUTHORITATIVE_V114_PRECISION_SHA256 = "7b6915c09502485d06ddc6b08f6f30e0876ead70f3a6ad85fffd020a7e7039ce"
ROLES = ("dev_near", "dev_contact", "certificate_near", "certificate_contact")
SPACES = ("base", "set_integral", "set_work")


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
            rows.append({"variant": variant, "role": role, f"v48_115_{treatment}_auc": ta,
                         f"v48_114_{historical_space}_auc": co, label: delta})
            if delta is not None and delta > 0.0:
                positive.append([variant, role]); roles.add(role)
            if delta is not None and delta >= 0.01:
                material.append([variant, role])
    go = len(positive) >= 6 and _cross(roles, 3) and len(material) >= 4
    return {"go": bool(go), "rows": rows, "positive_cells": positive, "material_cells": material, "roles": sorted(roles)}


def _activity_gate(docs: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    set_nonzero: set[str] = set()
    integral_nonzero: set[str] = set()
    diverse: set[str] = set()
    multi_option: set[str] = set()
    reentry_contact: set[str] = set()
    exact = True
    for variant, d in docs.items():
        ev = d.get("events") or {}
        for role in ROLES:
            diag = ((ev.get(role) or {}).get("pair_diagnostics") or {})
            row = {
                "variant": variant,
                "role": role,
                "mean_common_valid_option_count": float(diag.get("mean_common_valid_option_count", 0.0)),
                "min_common_valid_option_count": int(diag.get("min_common_valid_option_count", 0)),
                "set_work_nonzero_fraction": float(diag.get("set_work_nonzero_fraction", 0.0)),
                "set_integral_nonzero_fraction": float(diag.get("set_integral_nonzero_fraction", 0.0)),
                "option_flow_diverse_fraction": float(diag.get("option_flow_diverse_fraction", 0.0)),
                "reentry_set_available_fraction": float(diag.get("reentry_set_available_fraction", 0.0)),
                "max_set_work_conservation_error": float(diag.get("max_set_work_conservation_error", 1.0)),
                "max_option_permutation_invariance_error": float(diag.get("max_option_permutation_invariance_error", 1.0)),
            }
            rows.append(row)
            if row["set_work_nonzero_fraction"] > 0.0: set_nonzero.add(role)
            if row["set_integral_nonzero_fraction"] > 0.0: integral_nonzero.add(role)
            if row["option_flow_diverse_fraction"] > 0.0: diverse.add(role)
            if row["min_common_valid_option_count"] >= 2: multi_option.add(role)
            if "contact" in role and row["reentry_set_available_fraction"] > 0.0: reentry_contact.add(role)
            if row["max_set_work_conservation_error"] > 1.0e-10 or row["max_option_permutation_invariance_error"] > 1.0e-10:
                exact = False
    go = bool(
        exact and _cross(set_nonzero, 3) and _cross(integral_nonzero, 3)
        and _cross(diverse, 3) and _cross(multi_option, 3) and len(reentry_contact) >= 1
    )
    return {
        "go": go,
        "exact_set_contract_go": exact,
        "set_work_nonzero_roles": sorted(set_nonzero),
        "set_integral_nonzero_roles": sorted(integral_nonzero),
        "option_flow_diverse_roles": sorted(diverse),
        "multi_option_roles": sorted(multi_option),
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
    return {"exact": not diffs, "differences": diffs, "effective_unique_roles_if_exact": 4 if not diffs else 8}


def _power(docs: dict[str, Any], space: str) -> list[dict[str, Any]]:
    out = []
    for variant, d in docs.items():
        for role in ROLES:
            for axis in ("support", "reserve"):
                m = d[f"{space}_cells"][role][f"{axis}_true"]
                out.append({"variant": variant, "role": role, "axis": axis,
                            "rows": m.get("rows"), "positive_rows": m.get("positive_rows"),
                            "negative_rows": m.get("negative_rows"), "powered_groups": m.get("powered_groups"),
                            "underpowered": bool((m.get("powered_groups") or 0) < 2)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--balanced", type=Path, required=True)
    ap.add_argument("--precision", type=Path, required=True)
    ap.add_argument("--v114-pipeline", type=Path, required=True)
    ap.add_argument("--v114-comparison", type=Path, required=True)
    ap.add_argument("--v114-balanced", type=Path, required=True)
    ap.add_argument("--v114-precision", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    errors: list[str] = []

    def read(p: Path, name: str) -> dict[str, Any]:
        try: return json.loads(p.read_text())
        except Exception as exc:
            errors.append(f"{name}:json:{type(exc).__name__}"); return {}

    docs = {"balanced": read(a.balanced, "balanced"), "precision": read(a.precision, "precision")}
    p114 = read(a.v114_pipeline, "v114_pipeline")
    c114 = read(a.v114_comparison, "v114_comparison")
    b114 = read(a.v114_balanced, "v114_balanced")
    q114 = read(a.v114_precision, "v114_precision")

    expected_sha = {
        a.v114_pipeline: AUTHORITATIVE_V114_PIPELINE_SHA256,
        a.v114_comparison: AUTHORITATIVE_V114_COMPARISON_SHA256,
        a.v114_balanced: AUTHORITATIVE_V114_BALANCED_SHA256,
        a.v114_precision: AUTHORITATIVE_V114_PRECISION_SHA256,
    }
    for path, want in expected_sha.items():
        if not path.is_file() or _sha(path) != want: errors.append(f"v114_sha:{path.name}")

    for variant, d in docs.items():
        if not (
            d.get("valid") and d.get("engineering_version") == ENGINEERING_VERSION
            and d.get("scientific_version") == SCIENTIFIC_VERSION and d.get("variant") == variant
            and d.get("run_instance_id") == a.run_id and d.get("audit_only")
            and d.get("convex_closed_form_ridge") and d.get("strictly_convex_unique_solution")
            and d.get("capacity_matched_all_set_families") and d.get("matched_family_dimension") == MATCHED_DIM
            and d.get("set_geometry_dimension") == SET_GEOMETRY_DIM and d.get("selector_free_feature_path") is True
            and d.get("same_option_inside_each_set_summand") is True
            and d.get("teacher_npz_fields_loaded_into_feature_path") is False
            and d.get("regime_conditioning") is False and d.get("boundary_transport") is False
            and d.get("test_roots_read") is False
        ):
            errors.append(f"{variant}:contract")

    d114 = c114.get("preregistered_decision") or {}
    if not (
        p114.get("valid") and p114.get("attribution_ready")
        and p114.get("engineering_version") == "v48.114.0-OC-CCW"
        and p114.get("preregistered_status") == "COMMON_OPTION_CONSTRAINT_WORK_STOP"
    ): errors.append("v114_pipeline")
    if not (
        c114.get("valid") and c114.get("attribution_ready")
        and d114.get("status") == "COMMON_OPTION_CONSTRAINT_WORK_STOP"
        and d114.get("next_branch") == "close_selected_option_fixed_bin_work_family_then_preregister_selector_free_recovery_set_constraint_flow_audit_no_capacity_or_regime_sweep"
    ): errors.append("v114_branch")

    historical = {"balanced": b114, "precision": q114}
    for variant in ("balanced", "precision"):
        if docs[variant].get("checkpoint_sha256") != historical[variant].get("checkpoint_sha256"):
            errors.append(f"{variant}:checkpoint_identity")
        for metric in ("support", "reserve"):
            for role in ROLES:
                got = docs[variant]["base_cells"][role][f"{metric}_true"].get("auc")
                exp = historical[variant]["base_cells"][role][f"{metric}_true"].get("auc")
                if got is None or exp is None or abs(float(got) - float(exp)) > 1.0e-12:
                    errors.append(f"{variant}:{role}:{metric}:v114_base_identity")

    if errors:
        fake = {"go": False, "local_order": False, "rows": []}
        si_s = si_r = sw_s = sw_r = fake
        sel_i_s = sel_i_r = sel_w_s = sel_w_r = fake
        decomp_s = decomp_r = fake
        activity = {"go": False, "reentry_contact_coverage_go": False, "rows": []}
    else:
        si_s = _action_gate(docs, "set_integral", "support")
        si_r = _action_gate(docs, "set_integral", "reserve")
        sw_s = _action_gate(docs, "set_work", "support")
        sw_r = _action_gate(docs, "set_work", "reserve")
        sel_i_s = _historical_increment_gate(docs, historical, "set_integral", "candidate_integral", "support", label="set_minus_selected_integral")
        sel_i_r = _historical_increment_gate(docs, historical, "set_integral", "candidate_integral", "reserve", label="set_minus_selected_integral")
        sel_w_s = _historical_increment_gate(docs, historical, "set_work", "candidate_work", "support", label="set_minus_selected_work")
        sel_w_r = _historical_increment_gate(docs, historical, "set_work", "candidate_work", "reserve", label="set_minus_selected_work")
        decomp_s = _within_increment_gate(docs, "set_work", "set_integral", "support", label="set_work_minus_set_integral")
        decomp_r = _within_increment_gate(docs, "set_work", "set_integral", "reserve", label="set_work_minus_set_integral")
        activity = _activity_gate(docs)

    integral_core = bool(si_s.get("go") and si_r.get("go") and sel_i_s.get("go") and sel_i_r.get("go") and activity.get("go"))
    work_core = bool(sw_s.get("go") and sw_r.get("go") and sel_w_s.get("go") and sel_w_r.get("go") and activity.get("go"))
    decomposition = bool(work_core and decomp_s.get("go") and decomp_r.get("go"))
    reentry = bool(activity.get("reentry_contact_coverage_go"))

    if errors:
        status = "V48_115_ENGINEERING_STOP"
        branch = "fix_v48_115_engineering_and_rerun_same_selector_free_recovery_set_flow_audit"
    elif work_core and decomposition and reentry:
        status = "RECOVERY_SET_CONSTRAINT_FLOW_GO"
        branch = "promote_selector_free_signed_recovery_set_flow_then_preregister_one_main_carrier_integration_no_source_or_boundary_sweep"
    elif work_core and reentry:
        status = "RECOVERY_SET_FLOW_GO_SIGNED_DECOMPOSITION_NOT_REQUIRED"
        branch = "promote_selector_free_recovery_set_flow_but_not_signed_decomposition_then_preregister_one_main_carrier_integration"
    elif integral_core and reentry:
        status = "RECOVERY_SET_INTEGRAL_FLOW_GO"
        branch = "promote_selector_free_integral_recovery_set_flow_only_then_preregister_one_main_carrier_integration"
    elif sw_s.get("go") and sel_w_s.get("go") and not (sw_r.get("go") and sel_w_r.get("go")):
        status = "RECOVERY_SET_FLOW_SUPPORT_ONLY"
        branch = "retain_selector_free_support_set_flow_only_then_audit_weak_root_debt_flow_no_capacity_or_regime_sweep"
    elif sw_r.get("go") and sel_w_r.get("go") and not (sw_s.get("go") and sel_w_s.get("go")):
        status = "RECOVERY_SET_FLOW_RESERVE_ONLY"
        branch = "retain_selector_free_debt_set_flow_only_then_audit_weak_root_support_flow_no_capacity_or_regime_sweep"
    elif sw_s.get("local_order") and sw_r.get("local_order"):
        status = "RECOVERY_SET_FLOW_LOCAL_ORDER_ONLY"
        branch = "one_convex_pairwise_audit_on_exact_selector_free_set_features_no_feature_or_source_change"
    else:
        status = "RECOVERY_SET_CONSTRAINT_FLOW_STOP"
        branch = "close_observation_only_option_set_mean_flow_then_preregister_ocmero_weak_root_conditioned_recovery_set_flow_audit_frozen_roots_no_training_or_capacity_sweep"

    ident = _variant_identity(docs)
    decision = {
        "status": status,
        "next_branch": branch,
        "set_integral_support_gate": si_s,
        "set_integral_reserve_gate": si_r,
        "set_work_support_gate": sw_s,
        "set_work_reserve_gate": sw_r,
        "set_integral_vs_v48_114_selected_integral_support_gate": sel_i_s,
        "set_integral_vs_v48_114_selected_integral_reserve_gate": sel_i_r,
        "set_work_vs_v48_114_selected_work_support_gate": sel_w_s,
        "set_work_vs_v48_114_selected_work_reserve_gate": sel_w_r,
        "set_work_vs_set_integral_support_gate": decomp_s,
        "set_work_vs_set_integral_reserve_gate": decomp_r,
        "recovery_set_activity_gate": activity,
        "selector_free_integral_core_go": integral_core,
        "selector_free_work_core_go": work_core,
        "signed_reserve_debt_set_decomposition_go": decomposition,
        "reentry_contact_coverage_go": reentry,
        "capacity_matched_all_families": True,
        "matched_dimension": MATCHED_DIM,
        "geometry_dimension": SET_GEOMETRY_DIM,
        "work_bins": 8,
        "constraint_names": ["clearance", "stopping", "route", "reentry"],
        "set_aggregation": "uniform_empirical_mean_over_all_common_valid_recovery_options",
        "balanced_precision_metric_identity": ident,
        "power_diagnostics": _power(docs, "set_work"),
        "scientific_note": (
            "V48.115 removes the prereadout hard max-min option selector while preserving same-option causal correspondence inside each library element. "
            "The finite executable recovery library is represented as a permutation-invariant empirical set, using the parameter-free first moment of full-horizon integral response or signed reserve/debt work. "
            "All set families remain 220-D closed-form convex probes on the exact V48.114 cohorts/checkpoints; option selection is intentionally deferred to downstream OC-MERO semantics rather than embedded in the causal feature extractor."
        ),
        "source_training_authorized": False,
        "broad_encoder_training_authorized": False,
        "boundary_transport_authorized": False,
        "dataset_reconstruction_authorized": False,
        "regime_conditioned_policy_authorized": False,
    }
    out = {
        "schema": "ocrap-v48.115-rscf-comparison-v1",
        "engineering_version": ENGINEERING_VERSION,
        "scientific_version": SCIENTIFIC_VERSION,
        "run_instance_id": a.run_id,
        "valid": not errors,
        "attribution_ready": not errors,
        "errors": errors,
        "experiment_type": "audit_only_selector_free_recovery_set_constraint_flow",
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
        "v48_114_pipeline_sha256": _sha(a.v114_pipeline),
        "v48_114_comparison_sha256": _sha(a.v114_comparison),
        "v48_114_balanced_sha256": _sha(a.v114_balanced),
        "v48_114_precision_sha256": _sha(a.v114_precision),
        "authoritative_v48_114_comparison_sha256": AUTHORITATIVE_V114_COMPARISON_SHA256,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"valid": out["valid"], "status": status, "errors": errors}))
    return 0 if out["valid"] else 30


if __name__ == "__main__":
    raise SystemExit(main())
