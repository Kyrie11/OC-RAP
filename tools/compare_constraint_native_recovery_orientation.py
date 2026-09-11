#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ocrap.audits.viability_survival_envelope import (
    ENGINEERING_VERSION,
    SCIENTIFIC_VERSION,
    MATCHED_DIM,
    ENVELOPE_GEOMETRY_DIM,
)

AUTHORITATIVE_V117_PIPELINE_SHA256 = "674902c61b68c05387798f011d5e0b9639efd1eff5f3e1b08b3222d819f7c399"
AUTHORITATIVE_V117_COMPARISON_SHA256 = "8a4e54a71dfc77821842041a053002ef8b4b10c9b6a778270384b7da219fc99b"
AUTHORITATIVE_V117_BALANCED_SHA256 = "9574b2d30a772c6952f3c5ffd601b2bc15ae5703892c51d8162ced9fc08a8d24"
AUTHORITATIVE_V117_PRECISION_SHA256 = "836d45e7cf3f8f32692847bde8aefdd985523fe63bee333a16a031d92bee9ea8"

ROLES = ("dev_near", "dev_contact", "certificate_near", "certificate_contact")
SPACES = ("base", "exposed_envelope", "full_envelope")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ok(v: Any, threshold: float) -> bool:
    return v is not None and float(v) >= threshold


def _cross(roles: set[str], n: int) -> bool:
    return len(roles) >= n and any("near" in r for r in roles) and any("contact" in r for r in roles)


def _action_gate(docs: dict[str, Any], space: str, axis: str) -> dict[str, Any]:
    pos: list[list[str]] = []
    top: list[list[str]] = []
    roles: set[str] = set()
    top_roles: set[str] = set()
    for variant, d in docs.items():
        for role in ROLES:
            m = d[f"{space}_cells"][role][f"{axis}_true"]
            if _ok(m.get("auc"), 0.65) and _ok(m.get("auc_vs_shuffled"), 0.05):
                pos.append([variant, role]); roles.add(role)
            if _ok(m.get("top1_vs_shuffled"), 0.10):
                top.append([variant, role]); top_roles.add(role)
    return {
        "go": bool(len(pos) >= 6 and _cross(roles, 3) and len(top) >= 4 and _cross(top_roles, 2)),
        "local_order": bool(len(top) >= 4 and _cross(top_roles, 2)),
        "positive_cells": pos,
        "top1_material_cells": top,
        "roles": sorted(roles),
        "top1_roles": sorted(top_roles),
    }


def _within(docs: dict[str, Any], treatment: str, control: str, axis: str, label: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    pos: list[list[str]] = []
    material: list[list[str]] = []
    roles: set[str] = set()
    for variant, d in docs.items():
        for role in ROLES:
            a = d[f"{treatment}_cells"][role][f"{axis}_true"].get("auc")
            b = d[f"{control}_cells"][role][f"{axis}_true"].get("auc")
            delta = None if a is None or b is None else float(a) - float(b)
            rows.append({"variant": variant, "role": role, f"{treatment}_auc": a, f"{control}_auc": b, label: delta})
            if delta is not None and delta > 0:
                pos.append([variant, role]); roles.add(role)
            if delta is not None and delta >= 0.01:
                material.append([variant, role])
    return {
        "go": bool(len(pos) >= 6 and _cross(roles, 3) and len(material) >= 4),
        "rows": rows,
        "positive_cells": pos,
        "material_cells": material,
        "roles": sorted(roles),
    }


def _historical(
    docs: dict[str, Any], hist: dict[str, Any], treatment: str, old_space: str,
    axis: str, label: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    pos: list[list[str]] = []
    material: list[list[str]] = []
    roles: set[str] = set()
    for variant, d in docs.items():
        for role in ROLES:
            a = d[f"{treatment}_cells"][role][f"{axis}_true"].get("auc")
            b = hist[variant][f"{old_space}_cells"][role][f"{axis}_true"].get("auc")
            delta = None if a is None or b is None else float(a) - float(b)
            rows.append({
                "variant": variant, "role": role,
                f"v48_118_{treatment}_auc": a,
                f"v48_117_{old_space}_auc": b,
                label: delta,
            })
            if delta is not None and delta > 0:
                pos.append([variant, role]); roles.add(role)
            if delta is not None and delta >= 0.01:
                material.append([variant, role])
    return {
        "go": bool(len(pos) >= 6 and _cross(roles, 3) and len(material) >= 4),
        "rows": rows,
        "positive_cells": pos,
        "material_cells": material,
        "roles": sorted(roles),
    }


def _activity(docs: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    exposed_nonzero: set[str] = set()
    full_nonzero: set[str] = set()
    diverse: set[str] = set()
    multi: set[str] = set()
    full_dynamic: set[str] = set()
    exposed_support: set[str] = set()
    reentry: set[str] = set()
    exact = True
    for variant, d in docs.items():
        for role in ROLES:
            e = (d.get("events") or {}).get(role) or {}
            p = e.get("pair_diagnostics") or {}
            b = e.get("boundary_support_diagnostics") or {}
            row = {
                "variant": variant,
                "role": role,
                "mean_common_valid_option_count": float(p.get("mean_common_valid_option_count", 0.0)),
                "min_common_valid_option_count": int(p.get("min_common_valid_option_count", 0)),
                "exposed_envelope_nonzero_fraction": float(p.get("exposed_envelope_nonzero_fraction", 0.0)),
                "full_envelope_nonzero_fraction": float(p.get("full_envelope_nonzero_fraction", 0.0)),
                "option_flow_diverse_fraction": float(p.get("option_flow_diverse_fraction", 0.0)),
                "reentry_set_available_fraction": float(p.get("reentry_set_available_fraction", 0.0)),
                "max_option_permutation_invariance_error": float(p.get("max_option_permutation_invariance_error", 1.0)),
                "max_envelope_decomposition_error": float(p.get("max_envelope_decomposition_error", 1.0)),
                "mean_exposed_winner_union_count": float(p.get("mean_exposed_winner_union_count", 0.0)),
                "mean_full_winner_union_count": float(p.get("mean_full_winner_union_count", 0.0)),
                "mean_full_prefix_coverage_fraction": float(p.get("mean_full_prefix_coverage_fraction", 0.0)),
                "mean_full_suffix_coverage_fraction": float(p.get("mean_full_suffix_coverage_fraction", 0.0)),
                "mean_boundary_positive_option_count": float(b.get("mean_positive_option_count", 0.0)),
                "max_option_weight_sum_error": float(b.get("max_option_weight_sum_error", 1.0)),
                "max_root_exposure_mass_error": float(b.get("max_root_exposure_mass_error", 1.0)),
                "all_model_padding_invalid": bool(b.get("all_model_padding_invalid", False)),
                "all_model_physical_valid_prefix_match": bool(b.get("all_model_physical_valid_prefix_match", False)),
                "max_padded_tail_mass": float(b.get("max_padded_tail_mass", 1.0)),
                "max_physical_tail_mass_error": float(b.get("max_physical_tail_mass_error", 1.0)),
            }
            rows.append(row)
            if row["exposed_envelope_nonzero_fraction"] > 0: exposed_nonzero.add(role)
            if row["full_envelope_nonzero_fraction"] > 0: full_nonzero.add(role)
            if row["option_flow_diverse_fraction"] > 0: diverse.add(role)
            if row["min_common_valid_option_count"] >= 2: multi.add(role)
            if row["mean_full_winner_union_count"] >= 2: full_dynamic.add(role)
            if row["mean_boundary_positive_option_count"] >= 2: exposed_support.add(role)
            if "contact" in role and row["reentry_set_available_fraction"] > 0: reentry.add(role)
            if (
                row["max_option_permutation_invariance_error"] > 1e-10
                or row["max_envelope_decomposition_error"] > 1e-10
                or row["max_option_weight_sum_error"] > 1e-10
                or row["max_root_exposure_mass_error"] > 1e-10
                or not row["all_model_padding_invalid"]
                or not row["all_model_physical_valid_prefix_match"]
                or row["max_padded_tail_mass"] > 1e-10
                or row["max_physical_tail_mass_error"] > 1e-10
            ):
                exact = False
    core = bool(
        exact and _cross(exposed_nonzero, 3) and _cross(full_nonzero, 3)
        and _cross(diverse, 3) and _cross(multi, 3) and len(reentry) >= 1
    )
    dynamic = bool(_cross(full_dynamic, 3))
    support = bool(_cross(exposed_support, 3))
    return {
        "go": bool(core and dynamic and support),
        "core_activity_go": core,
        "nontrivial_full_set_envelope_go": dynamic,
        "nontrivial_exposed_support_go": support,
        "exact_envelope_contract_go": exact,
        "exposed_envelope_nonzero_roles": sorted(exposed_nonzero),
        "full_envelope_nonzero_roles": sorted(full_nonzero),
        "physical_option_flow_diverse_roles": sorted(diverse),
        "multi_option_roles": sorted(multi),
        "dynamic_full_envelope_roles": sorted(full_dynamic),
        "nontrivial_exposed_support_roles": sorted(exposed_support),
        "reentry_contact_roles": sorted(reentry),
        "reentry_contact_coverage_go": bool(reentry),
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
                    if ma.get(k) != mb.get(k):
                        diffs.append(f"{space}:{role}:{metric}:{k}")
    return {"exact": not diffs, "differences": diffs, "effective_unique_roles_if_exact": 4}


def _power(docs: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for variant, d in docs.items():
        for role in ROLES:
            for axis in ("support", "reserve"):
                m = d["base_cells"][role][f"{axis}_true"]
                out.append({
                    "variant": variant, "role": role, "axis": axis,
                    "rows": m.get("rows", 0),
                    "positive_rows": m.get("positive_rows", 0),
                    "negative_rows": m.get("negative_rows", 0),
                    "powered_groups": m.get("powered_groups", 0),
                    "underpowered": bool(
                        int(m.get("powered_groups", 0)) < 2
                        or min(int(m.get("positive_rows", 0)), int(m.get("negative_rows", 0))) < 3
                    ),
                })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--balanced", type=Path, required=True)
    ap.add_argument("--precision", type=Path, required=True)
    ap.add_argument("--v117-pipeline", type=Path, required=True)
    ap.add_argument("--v117-comparison", type=Path, required=True)
    ap.add_argument("--v117-balanced", type=Path, required=True)
    ap.add_argument("--v117-precision", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()

    docs = {"balanced": json.loads(a.balanced.read_text()), "precision": json.loads(a.precision.read_text())}
    hist = {"balanced": json.loads(a.v117_balanced.read_text()), "precision": json.loads(a.v117_precision.read_text())}
    p117 = json.loads(a.v117_pipeline.read_text())
    c117 = json.loads(a.v117_comparison.read_text())
    errors: list[str] = []

    if _sha(a.v117_pipeline) != AUTHORITATIVE_V117_PIPELINE_SHA256: errors.append("v117_pipeline_sha")
    if _sha(a.v117_comparison) != AUTHORITATIVE_V117_COMPARISON_SHA256: errors.append("v117_comparison_sha")
    if _sha(a.v117_balanced) != AUTHORITATIVE_V117_BALANCED_SHA256: errors.append("v117_balanced_sha")
    if _sha(a.v117_precision) != AUTHORITATIVE_V117_PRECISION_SHA256: errors.append("v117_precision_sha")
    d117 = c117.get("preregistered_decision") or {}
    if not (
        p117.get("valid") and p117.get("attribution_ready")
        and p117.get("preregistered_status") == "TAIL_BOUNDARY_CROSSING_FLOW_STOP"
        and c117.get("valid") and c117.get("attribution_ready")
        and d117.get("status") == "TAIL_BOUNDARY_CROSSING_FLOW_STOP"
        and d117.get("next_branch") == "close_static_boundary_witness_hitting_flow_then_preregister_recovery_set_viability_survival_envelope_audit_no_training_capacity_regime_source_horizon_or_threshold_sweep"
    ):
        errors.append("v117_prerequisite")

    for variant, d in docs.items():
        if not (
            d.get("valid") and d.get("engineering_version") == ENGINEERING_VERSION
            and d.get("scientific_version") == SCIENTIFIC_VERSION
            and d.get("variant") == variant and d.get("run_instance_id") == a.run_id
            and d.get("audit_only") and d.get("capacity_matched_all_envelope_families")
            and d.get("matched_family_dimension") == MATCHED_DIM
            and d.get("envelope_geometry_dimension") == ENVELOPE_GEOMETRY_DIM
            and d.get("root_decoder_parameters_trained") == 0
            and d.get("source_parameters_trained") == 0
            and d.get("regime_conditioning") is False
            and d.get("horizon_sweep") is False and d.get("threshold_sweep") is False
        ):
            errors.append(variant)

    gates: dict[str, Any] = {}
    if errors:
        activity = {"go": False, "rows": [], "reentry_contact_coverage_go": False}
    else:
        for space in ("exposed_envelope", "full_envelope"):
            for axis in ("support", "reserve"):
                gates[f"{space}_{axis}"] = _action_gate(docs, space, axis)
                gates[f"{space}_vs_v117_{axis}"] = _historical(
                    docs, hist, space, "boundary_hitting", axis,
                    f"{space}_minus_v117_boundary_hitting",
                )
        for axis in ("support", "reserve"):
            gates[f"full_set_effect_{axis}"] = _within(
                docs, "full_envelope", "exposed_envelope", axis,
                "full_minus_exposed_envelope",
            )
        activity = _activity(docs)

    if errors:
        status = "V48_118_ENGINEERING_STOP"
        branch = "fix_v48_118_engineering_and_rerun_same_viability_survival_envelope_audit"
    else:
        full_core = bool(
            gates["full_envelope_support"]["go"] and gates["full_envelope_reserve"]["go"]
            and gates["full_envelope_vs_v117_support"]["go"] and gates["full_envelope_vs_v117_reserve"]["go"]
            and activity["go"] and activity.get("reentry_contact_coverage_go")
        )
        exposed_core = bool(
            gates["exposed_envelope_support"]["go"] and gates["exposed_envelope_reserve"]["go"]
            and gates["exposed_envelope_vs_v117_support"]["go"] and gates["exposed_envelope_vs_v117_reserve"]["go"]
            and activity["go"] and activity.get("reentry_contact_coverage_go")
        )
        if full_core:
            status = "VIABILITY_SURVIVAL_ENVELOPE_GO"
            branch = "promote_full_recovery_set_viability_survival_envelope_then_preregister_exactly_one_main_carrier_integration_no_source_boundary_regime_or_capacity_sweep"
        elif exposed_core:
            status = "WEAK_EXPOSED_VIABILITY_ENVELOPE_GO"
            branch = "promote_weak_exposed_viability_survival_envelope_then_preregister_exactly_one_main_carrier_integration_no_source_boundary_regime_or_capacity_sweep"
        elif gates["full_envelope_support"]["go"] and gates["full_envelope_vs_v117_support"]["go"]:
            status = "VIABILITY_SURVIVAL_ENVELOPE_SUPPORT_ONLY"
            branch = "retain_full_set_prefix_viability_support_then_audit_viability_order_profile_for_reserve_no_capacity_regime_source_horizon_or_threshold_sweep"
        elif gates["full_envelope_reserve"]["go"] and gates["full_envelope_vs_v117_reserve"]["go"]:
            status = "VIABILITY_SURVIVAL_ENVELOPE_RESERVE_ONLY"
            branch = "retain_full_set_suffix_reentry_reserve_then_audit_viability_order_profile_for_support_no_capacity_regime_source_horizon_or_threshold_sweep"
        elif gates["full_envelope_support"].get("local_order") and gates["full_envelope_reserve"].get("local_order"):
            status = "VIABILITY_SURVIVAL_ENVELOPE_LOCAL_ORDER_ONLY"
            branch = "one_convex_pairwise_audit_on_exact_viability_envelope_features_no_feature_source_or_capacity_change"
        else:
            status = "VIABILITY_SURVIVAL_ENVELOPE_STOP"
            branch = "close_signed_joint_max_min_survival_envelope_then_preregister_recovery_set_viability_order_profile_audit_no_training_capacity_regime_source_horizon_or_threshold_sweep"

    ident = _variant_identity(docs)
    full_set_attribution_go = bool(
        not errors
        and gates.get("full_set_effect_support", {}).get("go")
        and gates.get("full_set_effect_reserve", {}).get("go")
    )
    decision = {
        "status": status,
        "next_branch": branch,
        "balanced_precision_metric_identity": ident,
        "power_diagnostics": _power(docs),
        "viability_envelope_activity_gate": activity,
        "full_set_vs_exposed_support_attribution_go": full_set_attribution_go,
        "reentry_contact_coverage_go": bool(activity.get("reentry_contact_coverage_go")),
        "boundary_transport_authorized": False,
        "broad_encoder_training_authorized": False,
        "source_training_authorized": False,
        "regime_conditioned_policy_authorized": False,
        "dataset_reconstruction_authorized": False,
        "matched_dimension": MATCHED_DIM,
        "geometry_dimension": ENVELOPE_GEOMETRY_DIM,
        "constraint_names": ["clearance", "stopping", "route", "reentry"],
        "scientific_note": (
            "V48.118 replaces V48.117's static option-weighted first moment with an exact permutation-invariant "
            "signed joint max-min viability envelope. Prefix envelopes encode set-level first-loss margin and suffix "
            "envelopes encode persistent-safe re-entry margin. The primary family uses every common valid recovery "
            "option; an equal-capacity control restricts the envelope to the support of the frozen V48.117 weak-root "
            "zero-boundary witnesses but discards their weights. No learned set encoder, threshold, horizon, source, "
            "capacity or regime sweep is introduced."
        ),
    }
    decision.update({k + "_gate": v for k, v in gates.items()})

    out = {
        "schema": "ocrap-v48.118-vse-comparison-v1",
        "engineering_version": ENGINEERING_VERSION,
        "scientific_version": SCIENTIFIC_VERSION,
        "run_instance_id": a.run_id,
        "valid": not errors,
        "attribution_ready": not errors,
        "errors": errors,
        "experiment_type": "audit_only_recovery_set_viability_survival_envelope",
        "preregistered_decision": decision,
        "authoritative_v48_117_comparison_sha256": AUTHORITATIVE_V117_COMPARISON_SHA256,
        "v48_117_pipeline_sha256": _sha(a.v117_pipeline),
        "v48_117_comparison_sha256": _sha(a.v117_comparison),
        "v48_117_balanced_sha256": _sha(a.v117_balanced),
        "v48_117_precision_sha256": _sha(a.v117_precision),
        "stage_i_parameters_trained": 0,
        "root_decoder_parameters_trained": 0,
        "source_parameters_trained": 0,
        "planner_parameters_trained": 0,
        "relative_ranker_modified": False,
        "regime_conditioning": False,
        "boundary_transport": False,
        "teacher_metadata_input_to_model": False,
        "test_roots_read": False,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"valid": out["valid"], "status": status, "errors": errors}))
    return 0 if out["valid"] else 30


if __name__ == "__main__":
    raise SystemExit(main())
