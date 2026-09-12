#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ocrap.audits.viability_rank_persistence import (
    ENGINEERING_VERSION,
    SCIENTIFIC_VERSION,
    MATCHED_DIM,
    COUPLING_GEOMETRY_DIM,
    COUPLING_MODE_NAMES,
)

AUTHORITATIVE_V120_PIPELINE_SHA256 = "2839be06885f1b05cf6064b934fbe6ed55eda9ff0db6ffe46bf50e89c21cb471"
AUTHORITATIVE_V120_COMPARISON_SHA256 = "9a7827d769b3abe4cda0802fa4aa95b879f38c0ffe8392df65c21bc017cae3c4"
AUTHORITATIVE_V120_BALANCED_SHA256 = "a507bedd5f61cdfc6086dce7e1a1b12955fa62b1525d1e865d2fc574e993a263"
AUTHORITATIVE_V120_PRECISION_SHA256 = "57aafc304b0ec29d04b902d6dd5cec7680a380f69350614f3546fc0c9fc7f647"
V120_NEXT = "close_nominal_rank_viability_transport_then_preregister_recovery_set_rank_persistence_coupling_audit_no_training_capacity_regime_source_horizon_or_threshold_sweep"

ROLES = ("dev_near", "dev_contact", "certificate_near", "certificate_contact")
SPACES = ("base", "exposed_persistence", "full_persistence")


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
                f"v48_121_{treatment}_auc": a,
                f"v48_120_{old_space}_auc": b,
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
    exposed_persistence: set[str] = set()
    full_persistence: set[str] = set()
    persistence_defect: set[str] = set()
    reassignment: set[str] = set()
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
                "exposed_persistence_nonzero_fraction": float(p.get("exposed_persistence_nonzero_fraction", 0.0)),
                "full_persistence_nonzero_fraction": float(p.get("full_persistence_nonzero_fraction", 0.0)),
                "option_flow_diverse_fraction": float(p.get("option_flow_diverse_fraction", 0.0)),
                "reentry_set_available_fraction": float(p.get("reentry_set_available_fraction", 0.0)),
                "max_option_permutation_invariance_error": float(p.get("max_option_permutation_invariance_error", 1.0)),
                "mean_exposed_prefix_coupling_energy": float(p.get("mean_exposed_prefix_coupling_energy", 0.0)),
                "mean_exposed_suffix_coupling_energy": float(p.get("mean_exposed_suffix_coupling_energy", 0.0)),
                "mean_full_prefix_coupling_energy": float(p.get("mean_full_prefix_coupling_energy", 0.0)),
                "mean_full_suffix_coupling_energy": float(p.get("mean_full_suffix_coupling_energy", 0.0)),
                "mean_exposed_prefix_persistence_energy": float(p.get("mean_exposed_prefix_persistence_energy", 0.0)),
                "mean_exposed_suffix_persistence_energy": float(p.get("mean_exposed_suffix_persistence_energy", 0.0)),
                "mean_full_prefix_persistence_energy": float(p.get("mean_full_prefix_persistence_energy", 0.0)),
                "mean_full_suffix_persistence_energy": float(p.get("mean_full_suffix_persistence_energy", 0.0)),
                "mean_exposed_prefix_rank_persistence_defect": float(p.get("mean_exposed_prefix_rank_persistence_defect", 0.0)),
                "mean_exposed_suffix_rank_persistence_defect": float(p.get("mean_exposed_suffix_rank_persistence_defect", 0.0)),
                "mean_full_prefix_rank_persistence_defect": float(p.get("mean_full_prefix_rank_persistence_defect", 0.0)),
                "mean_full_suffix_rank_persistence_defect": float(p.get("mean_full_suffix_rank_persistence_defect", 0.0)),
                "mean_exposed_prefix_rank_inversion_fraction": float(p.get("mean_exposed_prefix_rank_inversion_fraction", 0.0)),
                "mean_exposed_suffix_rank_inversion_fraction": float(p.get("mean_exposed_suffix_rank_inversion_fraction", 0.0)),
                "mean_full_prefix_rank_inversion_fraction": float(p.get("mean_full_prefix_rank_inversion_fraction", 0.0)),
                "mean_full_suffix_rank_inversion_fraction": float(p.get("mean_full_suffix_rank_inversion_fraction", 0.0)),
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
            if row["exposed_persistence_nonzero_fraction"] > 0: exposed_nonzero.add(role)
            if row["full_persistence_nonzero_fraction"] > 0: full_nonzero.add(role)
            if max(row["mean_exposed_prefix_persistence_energy"], row["mean_exposed_suffix_persistence_energy"]) > 0: exposed_persistence.add(role)
            if max(row["mean_full_prefix_persistence_energy"], row["mean_full_suffix_persistence_energy"]) > 0: full_persistence.add(role)
            if max(row["mean_full_prefix_rank_persistence_defect"], row["mean_full_suffix_rank_persistence_defect"]) > 0: persistence_defect.add(role)
            if max(row["mean_full_prefix_rank_inversion_fraction"], row["mean_full_suffix_rank_inversion_fraction"]) > 0: reassignment.add(role)
            if row["option_flow_diverse_fraction"] > 0: diverse.add(role)
            if row["min_common_valid_option_count"] >= 2 and row["mean_full_eligible_option_count"] >= 2: multi.add(role)
            if "contact" in role and row["reentry_set_available_fraction"] > 0: reentry.add(role)
            if (
                row["max_option_permutation_invariance_error"] > 1e-10
                or row["max_option_weight_sum_error"] > 1e-10
                or row["max_root_exposure_mass_error"] > 1e-10
                or not row["all_model_padding_invalid"]
                or not row["all_model_physical_valid_prefix_match"]
                or row["max_padded_tail_mass"] > 1e-10
                or row["max_physical_tail_mass_error"] > 1e-10
            ):
                exact = False
    core = bool(
        exact
        and _cross(exposed_nonzero, 3) and _cross(full_nonzero, 3)
        and _cross(exposed_persistence, 3) and _cross(full_persistence, 3)
        and _cross(persistence_defect, 3) and _cross(reassignment, 3)
        and _cross(diverse, 3) and _cross(multi, 3)
        and len(reentry) == 2
    )
    return {
        "go": core,
        "core_activity_go": core,
        "exact_persistence_contract_go": exact,
        "exposed_persistence_nonzero_roles": sorted(exposed_nonzero),
        "full_persistence_nonzero_roles": sorted(full_nonzero),
        "exposed_persistence_energy_roles": sorted(exposed_persistence),
        "full_persistence_energy_roles": sorted(full_persistence),
        "rank_persistence_defect_roles": sorted(persistence_defect),
        "rank_reassignment_roles": sorted(reassignment),
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
                        if x.get(metric) != y.get(metric):
                            diffs.append(f"{space}:{role}:{axis}:{kind}:{metric}")
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
    ap.add_argument("--v120-pipeline", type=Path, required=True)
    ap.add_argument("--v120-comparison", type=Path, required=True)
    ap.add_argument("--v120-balanced", type=Path, required=True)
    ap.add_argument("--v120-precision", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()

    errors: list[str] = []
    want = {
        a.v120_pipeline: AUTHORITATIVE_V120_PIPELINE_SHA256,
        a.v120_comparison: AUTHORITATIVE_V120_COMPARISON_SHA256,
        a.v120_balanced: AUTHORITATIVE_V120_BALANCED_SHA256,
        a.v120_precision: AUTHORITATIVE_V120_PRECISION_SHA256,
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
            if int(d.get("coupling_geometry_dimension", -1)) != COUPLING_GEOMETRY_DIM: errors.append(f"coupling_dim_{variant}")
        except Exception as exc:
            errors.append(f"read_{variant}:{exc}")

    hist: dict[str, Any] = {}
    if not errors:
        try:
            hp = json.loads(a.v120_pipeline.read_text())
            hc = json.loads(a.v120_comparison.read_text())
            if not (hp.get("valid") and hp.get("attribution_ready") and hp.get("preregistered_status") == "VIABILITY_RANK_TRANSPORT_STOP"):
                errors.append("v120_pipeline_stop_prerequisite")
            hd = hc.get("preregistered_decision") or {}
            if not (hc.get("valid") and hc.get("attribution_ready") and hd.get("status") == "VIABILITY_RANK_TRANSPORT_STOP"):
                errors.append("v120_comparison_stop_prerequisite")
            if hd.get("next_branch") != V120_NEXT:
                errors.append("v120_next_branch_mismatch")
            hist = {
                "balanced": json.loads(a.v120_balanced.read_text()),
                "precision": json.loads(a.v120_precision.read_text()),
            }
        except Exception as exc:
            errors.append(f"historical_read:{exc}")

    gates: dict[str, Any] = {}
    activity: dict[str, Any] = {"go": False}
    if not errors:
        for space in ("exposed_persistence", "full_persistence"):
            for axis in ("support", "reserve"):
                gates[f"{space}_{axis}"] = _action_gate(docs, space, axis)
        for axis in ("support", "reserve"):
            gates[f"full_persistence_vs_v120_{axis}"] = _historical(
                docs, hist, "full_persistence", "full_transport", axis,
                f"full_persistence_minus_v120_full_transport_{axis}",
            )
            gates[f"exposed_persistence_vs_v120_{axis}"] = _historical(
                docs, hist, "exposed_persistence", "exposed_transport", axis,
                f"exposed_persistence_minus_v120_exposed_transport_{axis}",
            )
            gates[f"full_set_persistence_effect_{axis}"] = _within(
                docs, "full_persistence", "exposed_persistence", axis,
                f"full_minus_exposed_persistence_{axis}",
            )
        activity = _activity(docs)

    status = "V48_121_ENGINEERING_STOP"
    branch = "fix_v48_121_engineering_and_rerun_same_viability_rank_persistence_coupling_audit"
    if not errors:
        full_core = bool(
            gates["full_persistence_support"]["go"] and gates["full_persistence_reserve"]["go"]
            and gates["full_persistence_vs_v120_support"]["go"] and gates["full_persistence_vs_v120_reserve"]["go"]
            and activity.get("go")
        )
        exposed_core = bool(
            gates["exposed_persistence_support"]["go"] and gates["exposed_persistence_reserve"]["go"]
            and gates["exposed_persistence_vs_v120_support"]["go"] and gates["exposed_persistence_vs_v120_reserve"]["go"]
            and activity.get("go")
        )
        if full_core:
            status = "VIABILITY_RANK_PERSISTENCE_COUPLING_GO"
            branch = "authorize_exactly_one_full_viability_rank_persistence_coupling_main_carrier_integration_no_source_boundary_regime_or_capacity_cochange"
        elif exposed_core:
            status = "EXPOSED_VIABILITY_RANK_PERSISTENCE_COUPLING_GO"
            branch = "authorize_exactly_one_exposed_viability_rank_persistence_coupling_main_carrier_integration_no_source_boundary_regime_or_capacity_cochange"
        elif gates["full_persistence_support"]["go"] and gates["full_persistence_vs_v120_support"]["go"]:
            status = "VIABILITY_RANK_PERSISTENCE_COUPLING_SUPPORT_ONLY"
            branch = "retain_full_rank_persistence_support_axis_then_audit_missing_reserve_signed_state_without_training_capacity_regime_source_horizon_or_threshold_sweep"
        elif gates["full_persistence_reserve"]["go"] and gates["full_persistence_vs_v120_reserve"]["go"]:
            status = "VIABILITY_RANK_PERSISTENCE_COUPLING_RESERVE_ONLY"
            branch = "retain_full_rank_persistence_reserve_axis_then_audit_missing_support_signed_state_without_training_capacity_regime_source_horizon_or_threshold_sweep"
        elif gates["full_persistence_support"].get("local_order") and gates["full_persistence_reserve"].get("local_order"):
            status = "VIABILITY_RANK_PERSISTENCE_COUPLING_LOCAL_ORDER_ONLY"
            branch = "preregister_one_pairwise_audit_on_exact_same_viability_rank_persistence_features"
        else:
            status = "VIABILITY_RANK_PERSISTENCE_COUPLING_STOP"
            branch = "close_nominal_rank_persistence_coupling_then_preregister_signed_viability_rank_state_transport_audit_no_training_capacity_regime_source_horizon_or_threshold_sweep"

    ident = _variant_identity(docs) if docs else {"exact": False, "differences": ["no_docs"], "effective_unique_roles_if_exact": 4}
    full_set_attribution_go = bool(
        not errors and gates.get("full_set_persistence_effect_support", {}).get("go")
        and gates.get("full_set_persistence_effect_reserve", {}).get("go")
    )
    decision = {
        "status": status,
        "next_branch": branch,
        "balanced_precision_metric_identity": ident,
        "power_diagnostics": _power(docs) if docs else [],
        "viability_rank_persistence_coupling_activity_gate": activity,
        "full_set_vs_exposed_persistence_attribution_go": full_set_attribution_go,
        "reentry_contact_coverage_go": bool(activity.get("reentry_contact_coverage_go")),
        "boundary_transport_authorized": False,
        "broad_encoder_training_authorized": False,
        "source_training_authorized": False,
        "regime_conditioned_policy_authorized": False,
        "dataset_reconstruction_authorized": False,
        "matched_dimension": MATCHED_DIM,
        "geometry_dimension": COUPLING_GEOMETRY_DIM,
        "coupling_mode_names": [str(x) for x in COUPLING_MODE_NAMES],
        "persistence_basis": "fixed_global_rank_persistence_and_rank_x_persistence_modes",
        "rank_coordinate": "candidate_independent_nominal_same_option_viability_midranks",
        "constraint_names": ["clearance", "stopping", "route", "reentry"],
        "scientific_note": (
            "V48.121 retains V48.120 candidate-independent nominal-rank same-option causal correspondence, but couples "
            "candidate-minus-nominal signed viability displacement to the temporal persistence of each same option's nominal "
            "rank over the relevant prefix or suffix. Four fixed modes encode global shift, nominal-rank tilt, rank-persistence "
            "defect, and rank-by-persistence interaction. This directly tests whether persistent rank ownership, rather than "
            "instantaneous rank transport alone, is the missing population-stable carrier. No candidate rank coordinate, rank "
            "cut, tuned decay, option identity export, learned set encoder, regime router, boundary transport, or capacity/source/"
            "horizon/threshold change is introduced."
        ),
    }
    decision.update({k + "_gate": v for k, v in gates.items()})

    out = {
        "schema": "ocrap-v48.121-vrpc-comparison-v1",
        "engineering_version": ENGINEERING_VERSION,
        "scientific_version": SCIENTIFIC_VERSION,
        "run_instance_id": a.run_id,
        "valid": not errors,
        "attribution_ready": not errors,
        "errors": errors,
        "experiment_type": "audit_only_recovery_set_viability_rank_persistence_coupling",
        "preregistered_decision": decision,
        "authoritative_v48_120_comparison_sha256": AUTHORITATIVE_V120_COMPARISON_SHA256,
        "v48_120_pipeline_sha256": _sha(a.v120_pipeline) if a.v120_pipeline.is_file() else None,
        "v48_120_comparison_sha256": _sha(a.v120_comparison) if a.v120_comparison.is_file() else None,
        "v48_120_balanced_sha256": _sha(a.v120_balanced) if a.v120_balanced.is_file() else None,
        "v48_120_precision_sha256": _sha(a.v120_precision) if a.v120_precision.is_file() else None,
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
