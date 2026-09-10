#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ocrap.audits.heterogeneous_constraint_normal_cone import (
    CONE_GEOMETRY_DIM,
    ENGINEERING_VERSION,
    MATCHED_DIM,
    SCIENTIFIC_VERSION,
)

AUTHORITATIVE_V111_COMPARISON_SHA256 = "ee2a3f13f2793dd8d0a4a1bdf73192a188d21bfac31c1549b3d6d0ae63cb8373"
AUTHORITATIVE_V111_PIPELINE_SHA256 = "c155ac8277b2fe690be030eaaf4031e75873e1e22d2521ce56d10aabebb56187"
AUTHORITATIVE_V111_BALANCED_SHA256 = "df5dd1a3a590774255f09e92de3e5d4a9762272fb871a3e64abfe6c3d8eb7e19"
AUTHORITATIVE_V111_PRECISION_SHA256 = "ec431e27086a2fe2f30f278a1c3fbfbafca5e4d62094a87dc838e9b52e627b66"
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


def _switch_gate(docs: dict[str, Any], metric: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    positive: list[list[str]] = []
    material: list[list[str]] = []
    roles: set[str] = set()
    for variant in ("balanced", "precision"):
        for role in ROLES:
            n = docs[variant]["nominal_cone_cells"][role][f"{metric}_true"].get("auc")
            c = docs[variant]["candidate_cone_cells"][role][f"{metric}_true"].get("auc")
            delta = None if n is None or c is None else float(c) - float(n)
            rows.append({
                "variant": variant,
                "role": role,
                "nominal_cone_auc": n,
                "candidate_cone_auc": c,
                "candidate_minus_nominal_cone": delta,
            })
            if delta is not None and delta > 0:
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


def _result_errors(obj: dict[str, Any], variant: str) -> list[str]:
    e: list[str] = []
    checks = [
        (obj.get("valid") is True, "valid"),
        (obj.get("engineering_version") == ENGINEERING_VERSION, "version"),
        (obj.get("scientific_version") == SCIENTIFIC_VERSION, "scientific_version"),
        (obj.get("variant") == variant, "variant"),
        (obj.get("audit_only") is True, "audit"),
        (obj.get("convex_closed_form_ridge") is True, "convex"),
        (obj.get("strictly_convex_unique_solution") is True, "unique"),
        (obj.get("iterative_optimizer_used") is False, "no_iterative_optimizer"),
        (obj.get("score_family") == "linear_on_fixed_heterogeneous_constraint_normal_cone_response_features", "score_family"),
        (obj.get("nominal_zero_score_by_construction") is True, "nominal_zero"),
        (obj.get("capacity_matched_nominal_vs_candidate_cone") is True, "capacity_match"),
        (int(obj.get("matched_family_dimension", -1)) == MATCHED_DIM, "matched_dim"),
        (int(obj.get("cone_geometry_dimension", -1)) == CONE_GEOMETRY_DIM, "geometry_dim"),
        (float(obj.get("max_normal_equation_residual", 1.0)) <= 1.0e-7, "normal_residual"),
        (obj.get("constraint_names") == ["clearance", "stopping", "route", "reentry"], "constraint_names"),
        (int(obj.get("stage_i_parameters_trained", -1)) == 0, "stage_i"),
        (int(obj.get("root_decoder_parameters_trained", -1)) == 0, "root"),
        (int(obj.get("source_parameters_trained", -1)) == 0, "source"),
        (int(obj.get("planner_parameters_trained", -1)) == 0, "planner"),
        (obj.get("regime_conditioning") is False, "regime"),
        (obj.get("boundary_transport") is False, "boundary"),
        (obj.get("teacher_metadata_input_to_model") is False, "teacher"),
        (obj.get("test_roots_read") is False, "test_roots"),
    ]
    for ok, name in checks:
        if not ok:
            e.append(f"{variant}:{name}")
    for space in ("base", "nominal_cone", "candidate_cone"):
        for role in ROLES:
            if role not in obj.get(f"{space}_cells", {}):
                e.append(f"{variant}:{space}:{role}")
    for ev, evo in obj.get("events", {}).items():
        if float(evo.get("agent_set_candidate_delta_max_abs", 1.0)) > 1.0e-6:
            e.append(f"{variant}:{ev}:agent_delta")
        if int(evo.get("agent_mask_candidate_delta_count", 1)) != 0:
            e.append(f"{variant}:{ev}:agent_mask")
    return e


def _variant_identity(docs: dict[str, Any]) -> dict[str, Any]:
    diffs: list[list[Any]] = []
    for space in ("base", "nominal_cone", "candidate_cone"):
        for role in ROLES:
            for metric in ("support", "reserve"):
                for suffix in ("auc", "auc_vs_shuffled", "top1", "top1_vs_shuffled"):
                    a = docs["balanced"][f"{space}_cells"][role][f"{metric}_true"].get(suffix)
                    b = docs["precision"][f"{space}_cells"][role][f"{metric}_true"].get(suffix)
                    if a is None and b is None:
                        continue
                    if a is None or b is None or abs(float(a) - float(b)) > 1.0e-12:
                        diffs.append([space, role, metric, suffix, a, b])
    return {"exact": not diffs, "differences": diffs, "effective_unique_roles_if_exact": 4 if not diffs else 8}


def _power(docs: dict[str, Any], space: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for variant in ("balanced", "precision"):
        for role in ROLES:
            for metric in ("support", "reserve"):
                m = docs[variant][f"{space}_cells"][role][f"{metric}_true"]
                out.append({
                    "variant": variant,
                    "role": role,
                    "axis": metric,
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
    switch_roles: set[str] = set()
    heterogeneous_roles: set[str] = set()
    reentry_contact_roles: set[str] = set()
    # balanced/precision raw feature extraction should be identical, but keep
    # both in the record for protocol visibility.
    for variant in ("balanced", "precision"):
        d = docs[variant]
        for role in ROLES:
            ev = d.get("events", {}).get(role, {})
            diag = ev.get("selector_diagnostics", {})
            switch_fraction = float(diag.get("selector_switch_fraction", 0.0) or 0.0)
            counts = diag.get("candidate_active_type_counts", {}) or {}
            active_types = sorted(k for k, v in counts.items() if int(v) > 0)
            any_switch_fraction = float(diag.get("candidate_any_switch_fraction", 0.0) or 0.0)
            rows.append({
                "variant": variant,
                "role": role,
                "selector_switch_fraction": switch_fraction,
                "candidate_any_switch_fraction": any_switch_fraction,
                "active_types": active_types,
                "candidate_active_type_counts": counts,
            })
            if any_switch_fraction > 0.0:
                switch_roles.add(role)
            if len(active_types) >= 2:
                heterogeneous_roles.add(role)
            if "contact" in role and int(counts.get("reentry", 0)) > 0:
                reentry_contact_roles.add(role)
    go = (
        _cross(switch_roles, 3)
        and _cross(heterogeneous_roles, 3)
        and len(reentry_contact_roles) >= 1
    )
    return {
        "go": bool(go),
        "switch_roles": sorted(switch_roles),
        "heterogeneous_roles": sorted(heterogeneous_roles),
        "reentry_contact_roles": sorted(reentry_contact_roles),
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    for key in ("balanced", "precision", "v111_pipeline", "v111_comparison", "v111_balanced", "v111_precision"):
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

    p111 = json.loads(a.v111_pipeline.read_text())
    c111 = json.loads(a.v111_comparison.read_text())
    b111 = json.loads(a.v111_balanced.read_text())
    p111r = json.loads(a.v111_precision.read_text())
    d111 = c111.get("preregistered_decision") or {}

    if _sha(a.v111_pipeline) != AUTHORITATIVE_V111_PIPELINE_SHA256:
        errors.append("v111_pipeline_sha")
    if _sha(a.v111_comparison) != AUTHORITATIVE_V111_COMPARISON_SHA256:
        errors.append("v111_comparison_sha")
    if _sha(a.v111_balanced) != AUTHORITATIVE_V111_BALANCED_SHA256:
        errors.append("v111_balanced_sha")
    if _sha(a.v111_precision) != AUTHORITATIVE_V111_PRECISION_SHA256:
        errors.append("v111_precision_sha")
    if not (
        p111.get("valid")
        and p111.get("attribution_ready")
        and p111.get("scientific_version") == "v48.111-OC-CNRO"
        and p111.get("preregistered_status") == "CONSTRAINT_NATIVE_ACTIVE_GEOMETRY_STOP"
    ):
        errors.append("v111_pipeline")
    if not (
        c111.get("valid")
        and c111.get("attribution_ready")
        and d111.get("status") == "CONSTRAINT_NATIVE_ACTIVE_GEOMETRY_STOP"
        and d111.get("next_branch")
        == "close_fixed_cv_circle_agent_geometry_then_preregister_heterogeneous_active_constraint_normal_cone_audit_no_training_or_source_sweep"
    ):
        errors.append("v111_branch")

    historical = {"balanced": b111, "precision": p111r}
    # Exact V48.111 base identity: same cohort, response coordinate, scaler rule,
    # convex ridge owner and deterministic shuffle.
    for metric in ("support", "reserve"):
        for variant in ("balanced", "precision"):
            for role in ROLES:
                got = docs[variant]["base_cells"][role][f"{metric}_true"].get("auc")
                exp = historical[variant]["base_cells"][role][f"{metric}_true"].get("auc")
                if got is None or exp is None or abs(float(got) - float(exp)) > 1.0e-12:
                    errors.append(f"{variant}:{role}:{metric}:v111_base_identity")

    ngs = _action_gate(docs, "nominal_cone", "support") if not errors else {"go": False, "local_order": False}
    ngr = _action_gate(docs, "nominal_cone", "reserve") if not errors else {"go": False, "local_order": False}
    cgs = _action_gate(docs, "candidate_cone", "support") if not errors else {"go": False, "local_order": False}
    cgr = _action_gate(docs, "candidate_cone", "reserve") if not errors else {"go": False, "local_order": False}
    sis = _switch_gate(docs, "support") if not errors else {"go": False, "rows": []}
    sir = _switch_gate(docs, "reserve") if not errors else {"go": False, "rows": []}
    activity = _activity_gate(docs) if not errors else {"go": False, "rows": []}

    if errors:
        status = "V48_112_ENGINEERING_STOP"
        branch = "fix_v48_112_engineering_and_rerun_same_heterogeneous_constraint_normal_cone_audit"
    elif cgs.get("go") and cgr.get("go") and sis.get("go") and sir.get("go") and activity.get("go"):
        status = "HETEROGENEOUS_CONSTRAINT_NORMAL_CONE_BOTH_AXES_GO"
        branch = "promote_candidate_conditioned_heterogeneous_constraint_normal_cone_then_preregister_one_nominal_invariant_carrier_no_source_or_transformer_sweep"
    elif cgs.get("go") and sis.get("go") and not (cgr.get("go") and sir.get("go")):
        status = "HETEROGENEOUS_CONSTRAINT_NORMAL_CONE_SUPPORT_ONLY"
        branch = "support_normal_cone_go_reserve_stop_then_audit_signed_debt_constraint_response_only"
    elif cgr.get("go") and sir.get("go") and not (cgs.get("go") and sis.get("go")):
        status = "HETEROGENEOUS_CONSTRAINT_NORMAL_CONE_RESERVE_ONLY"
        branch = "reserve_normal_cone_go_support_stop_then_audit_support_establishment_constraint_response_only"
    elif cgs.get("local_order") and cgr.get("local_order"):
        status = "HETEROGENEOUS_CONSTRAINT_NORMAL_CONE_LOCAL_ORDER_ONLY"
        branch = "same_heterogeneous_normal_cone_features_then_one_convex_pairwise_audit_no_feature_or_source_change"
    else:
        status = "HETEROGENEOUS_CONSTRAINT_NORMAL_CONE_STOP"
        branch = "close_prefix_level_first_order_constraint_cone_then_preregister_candidate_option_executable_constraint_jacobian_audit_no_training_or_source_sweep"

    ident = _variant_identity(docs)
    decision = {
        "status": status,
        "next_branch": branch,
        "nominal_cone_support_gate": ngs,
        "nominal_cone_reserve_gate": ngr,
        "candidate_cone_support_gate": cgs,
        "candidate_cone_reserve_gate": cgr,
        "candidate_minus_nominal_cone_support_gate": sis,
        "candidate_minus_nominal_cone_reserve_gate": sir,
        "constraint_activity_gate": activity,
        "capacity_matched_nominal_vs_candidate_cone": True,
        "matched_dimension": MATCHED_DIM,
        "geometry_dimension": CONE_GEOMETRY_DIM,
        "constraint_names": ["clearance", "stopping", "route", "reentry"],
        "balanced_precision_metric_identity": ident,
        "power_diagnostics": _power(docs, "candidate_cone"),
        "scientific_note": (
            "candidate and nominal cone families share the same 220-D linear function class; "
            "their only relational difference is whether active constraint type is selected by "
            "the candidate or nominal signed heterogeneous constraint state"
        ),
        "source_training_authorized": False,
        "broad_encoder_training_authorized": False,
        "boundary_transport_authorized": False,
        "dataset_reconstruction_authorized": False,
        "regime_conditioned_policy_authorized": False,
    }
    out = {
        "schema": "ocrap-v48.112-hcnc-comparison-v1",
        "engineering_version": ENGINEERING_VERSION,
        "scientific_version": SCIENTIFIC_VERSION,
        "run_instance_id": a.run_id,
        "valid": not errors,
        "attribution_ready": not errors,
        "errors": errors,
        "experiment_type": "audit_only_heterogeneous_active_constraint_normal_cone",
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
        "v48_111_pipeline_sha256": _sha(a.v111_pipeline),
        "v48_111_comparison_sha256": _sha(a.v111_comparison),
        "v48_111_balanced_sha256": _sha(a.v111_balanced),
        "v48_111_precision_sha256": _sha(a.v111_precision),
        "authoritative_v48_111_comparison_sha256": AUTHORITATIVE_V111_COMPARISON_SHA256,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"valid": out["valid"], "status": status, "errors": errors}))
    return 0 if out["valid"] else 30


if __name__ == "__main__":
    raise SystemExit(main())
