#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ocrap.v48_99_recovery_jacobian import ENGINEERING_VERSION

ROLES = ("dev_near", "dev_contact", "certificate_near", "certificate_contact")


def _ok(v: Any, t: float) -> bool:
    try:
        return v is not None and float(v) >= float(t)
    except Exception:
        return False


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _result_errors(obj: dict[str, Any], variant: str) -> list[str]:
    e: list[str] = []
    if not obj.get("valid") or obj.get("engineering_version") != ENGINEERING_VERSION:
        e.append(f"{variant}:contract")
    if int(obj.get("root_jacobian_parameters_trained", 0)) <= 0:
        e.append(f"{variant}:no_root_jacobian_training")
    for k in ("planner_parameters_trained", "source_parameters_trained", "erss_parameters_trained", "stage_i_parameters_trained"):
        if int(obj.get(k, -1)) != 0:
            e.append(f"{variant}:{k}")
    if int(obj.get("semantic_rank", -1)) != 2 or not obj.get("state_conditioned_control_affine"):
        e.append(f"{variant}:semantic_operator_contract")
    ident = obj.get("nominal_identity") or {}
    for split in ("dev", "certificate"):
        d = ident.get(split) or {}
        if max(float(d.get("support_max_abs_error", 1.0)), float(d.get("reserve_max_abs_error", 1.0))) > 1e-7:
            e.append(f"{variant}:{split}:nominal_identity")
    for role in ROLES:
        c = (obj.get("cells") or {}).get(role) or {}
        for name in ("state", "support_true", "reserve_true"):
            m = c.get(name) or {}
            if int(m.get("rows", 0)) <= 0 or m.get("auc") is None:
                e.append(f"{variant}:{role}:{name}:empty_or_null")
        rc = (obj.get("evaluation_contracts") or {}).get(role) or {}
        if not rc.get("valid"):
            e.append(f"{variant}:{role}:evaluation_contract")
    scales = obj.get("semantic_metric_scales") or {}
    if min(float(scales.get("support", 0.0)), float(scales.get("reserve", 0.0))) <= 0.0:
        e.append(f"{variant}:semantic_metric_scales")
    return e


def _population_errors(obj: dict[str, Any], ref: dict[str, Any], variant: str) -> list[str]:
    e: list[str] = []
    for role in ROLES:
        a = (obj.get("cells") or {}).get(role) or {}
        b = (ref.get("cells") or {}).get(role) or {}
        for name, fields in (
            ("state", ("rows", "drs_state_rows", "dep_state_rows")),
            ("support_true", ("rows", "positive_rows", "negative_rows", "powered_groups")),
            ("support_shuffled", ("rows", "positive_rows", "negative_rows", "powered_groups")),
            ("reserve_true", ("rows", "positive_rows", "negative_rows", "powered_groups")),
            ("reserve_shuffled", ("rows", "positive_rows", "negative_rows", "powered_groups")),
        ):
            for f in fields:
                if int((a.get(name) or {}).get(f, -1)) != int((b.get(name) or {}).get(f, -2)):
                    e.append(f"{variant}:{role}:{name}:{f}:v99={(a.get(name) or {}).get(f)}!=v98={(b.get(name) or {}).get(f)}")
    return e


def _state_errors(obj: dict[str, Any], ref: dict[str, Any], variant: str) -> list[str]:
    e: list[str] = []
    for role in ROLES:
        a = ((obj.get("cells") or {}).get(role) or {}).get("state") or {}
        b = ((ref.get("cells") or {}).get(role) or {}).get("state") or {}
        aa, bb = a.get("auc"), b.get("auc")
        if aa is None or bb is None or abs(float(aa) - float(bb)) > 1e-12:
            e.append(f"{variant}:{role}:state_auc_identity")
    return e


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--balanced", type=Path, required=True)
    ap.add_argument("--precision", type=Path, required=True)
    ap.add_argument("--v48-98-balanced", type=Path, required=True)
    ap.add_argument("--v48-98-precision", type=Path, required=True)
    ap.add_argument("--v48-98-comparison", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    b = json.loads(a.balanced.read_text()); p = json.loads(a.precision.read_text())
    rb = json.loads(a.v48_98_balanced.read_text()); rp = json.loads(a.v48_98_precision.read_text())
    rcmp = json.loads(a.v48_98_comparison.read_text())
    errors: list[str] = []
    rd = rcmp.get("preregistered_decision") or {}
    if not (rcmp.get("valid") and rcmp.get("attribution_ready") and rd.get("status") == "EXECUTABLE_RECOVERY_TANGENT_ALIGNMENT_STOP"):
        errors.append("v48_98_stop_prerequisite_missing")
    if not (rd.get("state_chart_preserved") is True and rd.get("support_tangent_go") is False and rd.get("reserve_debt_tangent_go") is False):
        errors.append("v48_98_branch_shape_mismatch")
    errors += _result_errors(b, "balanced") + _result_errors(p, "precision")
    errors += _population_errors(b, rb, "balanced") + _population_errors(p, rp, "precision")
    errors += _state_errors(b, rb, "balanced") + _state_errors(p, rp, "precision")

    sup_cells: list[list[str]] = []; res_cells: list[list[str]] = []
    sup_top: list[list[str]] = []; res_top: list[list[str]] = []
    sup_roles: set[str] = set(); res_roles: set[str] = set(); sup_top_roles: set[str] = set(); res_top_roles: set[str] = set()
    if not errors:
        for variant, obj in (("balanced", b), ("precision", p)):
            for role in ROLES:
                c = obj["cells"][role]
                s, r = c["support_true"], c["reserve_true"]
                if _ok(s.get("auc"), 0.65) and _ok(s.get("auc_vs_shuffled"), 0.05):
                    sup_cells.append([variant, role]); sup_roles.add(role)
                if _ok(s.get("top1_vs_shuffled"), 0.10):
                    sup_top.append([variant, role]); sup_top_roles.add(role)
                if _ok(r.get("auc"), 0.65) and _ok(r.get("auc_vs_shuffled"), 0.05):
                    res_cells.append([variant, role]); res_roles.add(role)
                if _ok(r.get("top1_vs_shuffled"), 0.10):
                    res_top.append([variant, role]); res_top_roles.add(role)

    def cross(x: set[str]) -> bool:
        return any("near" in r for r in x) and any("contact" in r for r in x)

    support_go = bool(not errors and len(sup_cells) >= 6 and len(sup_roles) >= 3 and cross(sup_roles) and len(sup_top) >= 4 and cross(sup_top_roles))
    reserve_go = bool(not errors and len(res_cells) >= 6 and len(res_roles) >= 3 and cross(res_roles) and len(res_top) >= 4 and cross(res_top_roles))
    state_preserved = not errors
    full_go = bool(state_preserved and support_go and reserve_go)

    if errors:
        status = "V48_99_ENGINEERING_STOP"
        next_branch = "fix_v48_99_engineering_and_rerun_same_recovery_jacobian_experiment"
    elif full_go:
        status = "RECOVERY_JACOBIAN_ALIGNMENT_GO"
        next_branch = "one_final_fixed_capacity_observation_aligned_source_then_freeze_if_source_gate_passes"
    elif reserve_go and not support_go:
        status = "RECOVERY_JACOBIAN_SUPPORT_GUARD_STOP"
        next_branch = "close_smooth_support_jacobian_then_adjudicate_hybrid_support_guard_no_source_or_threshold_sweep"
    elif support_go and not reserve_go:
        status = "RECOVERY_JACOBIAN_RESERVE_FLOW_STOP"
        next_branch = "retain_support_jacobian_then_adjudicate_supported_reserve_flow_objective_no_source_sweep"
    else:
        status = "RECOVERY_JACOBIAN_ALIGNMENT_STOP"
        next_branch = "close_state_conditioned_root_jacobian_then_preregister_joint_root_decoder_semantic_objective_no_source_sweep"

    decision = {
        "state_chart_preserved": state_preserved,
        "support_jacobian_go": support_go,
        "reserve_debt_jacobian_go": reserve_go,
        "recovery_jacobian_alignment_go": full_go,
        "support_positive_cells": sup_cells,
        "support_roles": sorted(sup_roles),
        "support_top1_material_cells": sup_top,
        "reserve_positive_cells": res_cells,
        "reserve_roles": sorted(res_roles),
        "reserve_top1_material_cells": res_top,
        "final_source_experiment_authorized": bool(full_go and not errors),
        "boundary_transport_authorized": False,
        "regime_conditioned_policy_authorized": False,
        "dataset_reconstruction_authorized": False,
        "status": status,
        "next_branch": next_branch,
    }
    out = {
        "schema": "ocrap-v48.99-ocrj-comparison-v1",
        "engineering_version": ENGINEERING_VERSION,
        "valid": not errors,
        "attribution_ready": not errors,
        "errors": errors,
        "experiment_type": "state_conditioned_control_affine_root_recovery_jacobian",
        "root_jacobian_parameters_trained": int(b.get("root_jacobian_parameters_trained", 0)),
        "planner_parameters_trained": 0,
        "source_parameters_trained": 0,
        "erss_parameters_trained": 0,
        "stage_i_parameters_trained": 0,
        "semantic_rank": 2,
        "state_conditioned_control_affine": True,
        "boundary_transport": False,
        "regime_conditioning": False,
        "relative_ranker_modified": False,
        "teacher_metadata_input_to_model": False,
        "dataset_reconstruction": False,
        "dataset_reselection": False,
        "evaluation_population_contract": "exact_v48_98_equals_v48_97_2_equals_v48_96_strata",
        "v48_98_comparison_sha256": _sha(a.v48_98_comparison),
        "v48_98_balanced_sha256": _sha(a.v48_98_balanced),
        "v48_98_precision_sha256": _sha(a.v48_98_precision),
        "preregistered_decision": decision,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"valid": out["valid"], "status": status, "errors": errors}))
    return 0 if out["valid"] else 30


if __name__ == "__main__":
    raise SystemExit(main())
