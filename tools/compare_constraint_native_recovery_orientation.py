#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ocrap.audits.viability_order_profile import (
    ENGINEERING_VERSION,
    SCIENTIFIC_VERSION,
    MATCHED_DIM,
    PROFILE_GEOMETRY_DIM,
    ORDER_MASSES,
)

AUTHORITATIVE_V118_PIPELINE_SHA256 = "8ded1afa8a7fc002fc39a19c11da8669dc702f2704e8141e742182f241af05ee"
AUTHORITATIVE_V118_COMPARISON_SHA256 = "ebe8b2bc18baa33db9f62c809d22eb7b4c9d28f0fcd8f6f4140f66287e3a3184"
AUTHORITATIVE_V118_BALANCED_SHA256 = "f5125fded56405945ddc1ffe6aa6c54f3349c7e85235e6e377ba546fb0d45128"
AUTHORITATIVE_V118_PRECISION_SHA256 = "3126b02c91cdccb22ef0baa810b424e9e6b78e2de55d4acff9a314f9e076f275"

ROLES = ("dev_near", "dev_contact", "certificate_near", "certificate_contact")
SPACES = ("base", "exposed_profile", "full_profile")


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
                f"v48_119_{treatment}_auc": a,
                f"v48_118_{old_space}_auc": b,
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
    exposed_noncollapsed: set[str] = set()
    full_noncollapsed: set[str] = set()
    diverse: set[str] = set()
    multi: set[str] = set()
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
                "exposed_profile_nonzero_fraction": float(p.get("exposed_profile_nonzero_fraction", 0.0)),
                "full_profile_nonzero_fraction": float(p.get("full_profile_nonzero_fraction", 0.0)),
                "option_flow_diverse_fraction": float(p.get("option_flow_diverse_fraction", 0.0)),
                "reentry_set_available_fraction": float(p.get("reentry_set_available_fraction", 0.0)),
                "max_option_permutation_invariance_error": float(p.get("max_option_permutation_invariance_error", 1.0)),
                "max_order_monotonicity_error": float(p.get("max_order_monotonicity_error", 1.0)),
                "mean_exposed_prefix_noncollapsed_fraction": float(p.get("mean_exposed_prefix_noncollapsed_fraction", 0.0)),
                "mean_exposed_suffix_noncollapsed_fraction": float(p.get("mean_exposed_suffix_noncollapsed_fraction", 0.0)),
                "mean_full_prefix_noncollapsed_fraction": float(p.get("mean_full_prefix_noncollapsed_fraction", 0.0)),
                "mean_full_suffix_noncollapsed_fraction": float(p.get("mean_full_suffix_noncollapsed_fraction", 0.0)),
                "mean_exposed_prefix_order_spread": float(p.get("mean_exposed_prefix_order_spread", 0.0)),
                "mean_exposed_suffix_order_spread": float(p.get("mean_exposed_suffix_order_spread", 0.0)),
                "mean_full_prefix_order_spread": float(p.get("mean_full_prefix_order_spread", 0.0)),
                "mean_full_suffix_order_spread": float(p.get("mean_full_suffix_order_spread", 0.0)),
                "mean_full_eligible_option_count": float(p.get("mean_full_eligible_option_count", 0.0)),
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
            if row["exposed_profile_nonzero_fraction"] > 0: exposed_nonzero.add(role)
            if row["full_profile_nonzero_fraction"] > 0: full_nonzero.add(role)
            if max(row["mean_exposed_prefix_noncollapsed_fraction"], row["mean_exposed_suffix_noncollapsed_fraction"]) > 0: exposed_noncollapsed.add(role)
            if max(row["mean_full_prefix_noncollapsed_fraction"], row["mean_full_suffix_noncollapsed_fraction"]) > 0: full_noncollapsed.add(role)
            if row["option_flow_diverse_fraction"] > 0: diverse.add(role)
            if row["min_common_valid_option_count"] >= 2 and row["mean_full_eligible_option_count"] >= 2: multi.add(role)
            if "contact" in role and row["reentry_set_available_fraction"] > 0: reentry.add(role)
            if (
                row["max_option_permutation_invariance_error"] > 1e-10
                or row["max_order_monotonicity_error"] > 1e-10
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
        and _cross(diverse, 3) and _cross(multi, 3)
        and _cross(exposed_noncollapsed, 3) and _cross(full_noncollapsed, 3)
        and len(reentry) == 2
    )
    return {
        "go": core,
        "core_activity_go": core,
        "exact_order_contract_go": exact,
        "exposed_profile_nonzero_roles": sorted(exposed_nonzero),
        "full_profile_nonzero_roles": sorted(full_nonzero),
        "exposed_profile_noncollapsed_roles": sorted(exposed_noncollapsed),
        "full_profile_noncollapsed_roles": sorted(full_noncollapsed),
        "physical_option_flow_diverse_roles": sorted(diverse),
        "multi_option_roles": sorted(multi),
        "reentry_contact_roles": sorted(reentry),
        "reentry_contact_coverage_go": len(reentry) == 2,
        "rows": rows,
    }


def _variant_identity(docs: dict[str, Any]) -> dict[str, Any]:
    if set(docs) != {"balanced", "precision"}:
        return {"exact": False, "differences": ["missing_variant"], "effective_unique_roles_if_exact": 4}
    a = docs["balanced"]; b = docs["precision"]
    diffs: list[str] = []
    for space in SPACES:
        for role in ROLES:
            for axis in ("support", "reserve"):
                for kind in ("true", "shuffled"):
                    x = a[f"{space}_cells"][role][f"{axis}_{kind}"]
                    y = b[f"{space}_cells"][role][f"{axis}_{kind}"]
                    for metric in ("auc", "top1", "auc_vs_shuffled", "top1_vs_shuffled"):
                        if x.get(metric) != y.get(metric): diffs.append(f"{space}:{role}:{axis}:{kind}:{metric}")
    return {"exact": not diffs, "differences": diffs, "effective_unique_roles_if_exact": 4}


def _power(docs: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for variant, d in docs.items():
        for role in ROLES:
            for axis in ("support", "reserve"):
                m = d["base_cells"][role][f"{axis}_true"]
                rows.append({
                    "variant": variant, "role": role, "axis": axis,
                    "rows": m.get("rows"), "positive_rows": m.get("positive_rows"),
                    "negative_rows": m.get("negative_rows"), "powered_groups": m.get("powered_groups"),
                    "underpowered": bool(role == "dev_near" and axis == "reserve"),
                })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--balanced", type=Path, required=True)
    ap.add_argument("--precision", type=Path, required=True)
    ap.add_argument("--v118-pipeline", type=Path, required=True)
    ap.add_argument("--v118-comparison", type=Path, required=True)
    ap.add_argument("--v118-balanced", type=Path, required=True)
    ap.add_argument("--v118-precision", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()

    errors: list[str] = []
    want = {
        a.v118_pipeline: AUTHORITATIVE_V118_PIPELINE_SHA256,
        a.v118_comparison: AUTHORITATIVE_V118_COMPARISON_SHA256,
        a.v118_balanced: AUTHORITATIVE_V118_BALANCED_SHA256,
        a.v118_precision: AUTHORITATIVE_V118_PRECISION_SHA256,
    }
    for p, digest in want.items():
        if not p.is_file(): errors.append(f"missing_authoritative_input:{p.name}")
        elif _sha(p) != digest: errors.append(f"authoritative_sha_mismatch:{p.name}")

    docs: dict[str, Any] = {}
    for variant, path in (("balanced", a.balanced), ("precision", a.precision)):
        try:
            d = json.loads(path.read_text())
            docs[variant] = d
            if not d.get("valid"): errors.append(f"invalid_{variant}")
            if d.get("engineering_version") != ENGINEERING_VERSION: errors.append(f"engineering_version_{variant}")
            if d.get("scientific_version") != SCIENTIFIC_VERSION: errors.append(f"scientific_version_{variant}")
            if d.get("run_instance_id") != a.run_id: errors.append(f"run_id_{variant}")
            if int(d.get("matched_family_dimension", -1)) != MATCHED_DIM: errors.append(f"matched_dim_{variant}")
            if int(d.get("profile_geometry_dimension", -1)) != PROFILE_GEOMETRY_DIM: errors.append(f"profile_dim_{variant}")
        except Exception as exc:
            errors.append(f"read_{variant}:{exc}")

    hist: dict[str, Any] = {}
    if not errors:
        try:
            hp = json.loads(a.v118_pipeline.read_text())
            hc = json.loads(a.v118_comparison.read_text())
            if not (hp.get("valid") and hp.get("attribution_ready") and hp.get("preregistered_status") == "VIABILITY_SURVIVAL_ENVELOPE_STOP"):
                errors.append("v118_pipeline_stop_prerequisite")
            hd = hc.get("preregistered_decision") or {}
            if not (hc.get("valid") and hc.get("attribution_ready") and hd.get("status") == "VIABILITY_SURVIVAL_ENVELOPE_STOP"):
                errors.append("v118_comparison_stop_prerequisite")
            if hd.get("next_branch") != "close_signed_joint_max_min_survival_envelope_then_preregister_recovery_set_viability_order_profile_audit_no_training_capacity_regime_source_horizon_or_threshold_sweep":
                errors.append("v118_next_branch_mismatch")
            hist = {
                "balanced": json.loads(a.v118_balanced.read_text()),
                "precision": json.loads(a.v118_precision.read_text()),
            }
        except Exception as exc:
            errors.append(f"historical_read:{exc}")

    gates: dict[str, Any] = {}
    activity: dict[str, Any] = {"go": False}
    if not errors:
        for space in ("exposed_profile", "full_profile"):
            for axis in ("support", "reserve"):
                gates[f"{space}_{axis}"] = _action_gate(docs, space, axis)
        for axis in ("support", "reserve"):
            gates[f"full_profile_vs_v118_{axis}"] = _historical(
                docs, hist, "full_profile", "full_envelope", axis,
                f"full_profile_minus_v118_full_envelope_{axis}",
            )
            gates[f"exposed_profile_vs_v118_{axis}"] = _historical(
                docs, hist, "exposed_profile", "exposed_envelope", axis,
                f"exposed_profile_minus_v118_exposed_envelope_{axis}",
            )
            gates[f"full_set_profile_effect_{axis}"] = _within(
                docs, "full_profile", "exposed_profile", axis,
                f"full_minus_exposed_profile_{axis}",
            )
        activity = _activity(docs)

    status = "V48_119_ENGINEERING_STOP"
    branch = "fix_v48_119_engineering_and_rerun_same_viability_order_profile_audit"
    if not errors:
        full_core = bool(
            gates["full_profile_support"]["go"] and gates["full_profile_reserve"]["go"]
            and gates["full_profile_vs_v118_support"]["go"] and gates["full_profile_vs_v118_reserve"]["go"]
            and activity.get("go")
        )
        exposed_core = bool(
            gates["exposed_profile_support"]["go"] and gates["exposed_profile_reserve"]["go"]
            and gates["exposed_profile_vs_v118_support"]["go"] and gates["exposed_profile_vs_v118_reserve"]["go"]
            and activity.get("go")
        )
        if full_core:
            status = "VIABILITY_ORDER_PROFILE_GO"
            branch = "authorize_exactly_one_full_viability_order_profile_main_carrier_integration_no_source_boundary_regime_or_capacity_cochange"
        elif exposed_core:
            status = "EXPOSED_VIABILITY_ORDER_PROFILE_GO"
            branch = "authorize_exactly_one_exposed_viability_order_profile_main_carrier_integration_no_source_boundary_regime_or_capacity_cochange"
        elif gates["full_profile_support"]["go"] and gates["full_profile_vs_v118_support"]["go"]:
            status = "VIABILITY_ORDER_PROFILE_SUPPORT_ONLY"
            branch = "retain_full_viability_order_profile_support_axis_then_audit_missing_reserve_persistence_without_training_capacity_regime_source_horizon_or_threshold_sweep"
        elif gates["full_profile_reserve"]["go"] and gates["full_profile_vs_v118_reserve"]["go"]:
            status = "VIABILITY_ORDER_PROFILE_RESERVE_ONLY"
            branch = "retain_full_viability_order_profile_reserve_axis_then_audit_missing_support_exposure_without_training_capacity_regime_source_horizon_or_threshold_sweep"
        elif gates["full_profile_support"].get("local_order") and gates["full_profile_reserve"].get("local_order"):
            status = "VIABILITY_ORDER_PROFILE_LOCAL_ORDER_ONLY"
            branch = "preregister_one_pairwise_audit_on_exact_same_viability_order_profile_features"
        else:
            status = "VIABILITY_ORDER_PROFILE_STOP"
            branch = "close_fixed_quartile_viability_order_profile_then_preregister_recovery_set_viability_rank_transport_audit_no_training_capacity_regime_source_horizon_or_threshold_sweep"

    ident = _variant_identity(docs) if docs else {"exact": False, "differences": ["no_docs"], "effective_unique_roles_if_exact": 4}
    full_set_attribution_go = bool(
        not errors and gates.get("full_set_profile_effect_support", {}).get("go")
        and gates.get("full_set_profile_effect_reserve", {}).get("go")
    )
    decision = {
        "status": status,
        "next_branch": branch,
        "balanced_precision_metric_identity": ident,
        "power_diagnostics": _power(docs) if docs else [],
        "viability_order_profile_activity_gate": activity,
        "full_set_vs_exposed_profile_attribution_go": full_set_attribution_go,
        "reentry_contact_coverage_go": bool(activity.get("reentry_contact_coverage_go")),
        "boundary_transport_authorized": False,
        "broad_encoder_training_authorized": False,
        "source_training_authorized": False,
        "regime_conditioned_policy_authorized": False,
        "dataset_reconstruction_authorized": False,
        "matched_dimension": MATCHED_DIM,
        "geometry_dimension": PROFILE_GEOMETRY_DIM,
        "order_masses": [float(x) for x in ORDER_MASSES],
        "constraint_names": ["clearance", "stopping", "route", "reentry"],
        "scientific_note": (
            "V48.119 keeps V48.118's same-option joint prefix/suffix viability margins but replaces the single rank-1 "
            "max envelope with a fixed four-level upper order profile at masses 1/4, 1/2, 3/4 and 1. The profile "
            "therefore retains frontier viability and recovery-set depth without exporting option identity. Equal-capacity "
            "exposed-support and full-set families isolate whether weak-root observation exposure or full executable "
            "redundancy supplies the transferable order structure. The gamma=1 coordinate is not a standalone uniform-mean "
            "family. No learned set encoder, threshold, horizon, source, capacity or regime sweep is introduced."
        ),
    }
    decision.update({k + "_gate": v for k, v in gates.items()})

    out = {
        "schema": "ocrap-v48.119-vop-comparison-v1",
        "engineering_version": ENGINEERING_VERSION,
        "scientific_version": SCIENTIFIC_VERSION,
        "run_instance_id": a.run_id,
        "valid": not errors,
        "attribution_ready": not errors,
        "errors": errors,
        "experiment_type": "audit_only_recovery_set_viability_order_profile",
        "preregistered_decision": decision,
        "authoritative_v48_118_comparison_sha256": AUTHORITATIVE_V118_COMPARISON_SHA256,
        "v48_118_pipeline_sha256": _sha(a.v118_pipeline) if a.v118_pipeline.is_file() else None,
        "v48_118_comparison_sha256": _sha(a.v118_comparison) if a.v118_comparison.is_file() else None,
        "v48_118_balanced_sha256": _sha(a.v118_balanced) if a.v118_balanced.is_file() else None,
        "v48_118_precision_sha256": _sha(a.v118_precision) if a.v118_precision.is_file() else None,
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
