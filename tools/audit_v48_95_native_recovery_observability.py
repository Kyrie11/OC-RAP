#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from pathlib import Path
from typing import Any

from ocrap.v48_95_native_recovery_observability import ENGINEERING_VERSION, SCHEMA, frozen_native_features, tie_auc

ROLES = ("dev_near", "dev_contact", "certificate_near", "certificate_contact")
VARIANTS = ("balanced", "precision")
ROLE_FILES = {
    "dev_near": "dev_diagnostic_near_v48.proposal_rows.jsonl",
    "dev_contact": "dev_diagnostic_contact_v48.proposal_rows.jsonl",
    "certificate_near": "direct_value_risk_near_v48.proposal_rows.jsonl",
    "certificate_contact": "direct_value_risk_contact_v48.proposal_rows.jsonl",
}

STATE_CHANNELS = (
    "nominal_hard_support",
    "nominal_smooth_support",
    "nominal_deployability",
)
SUPPORT_CHANNELS = (
    "delta_hard_support",
    "delta_smooth_support",
    "candidate_smooth_support",
    "candidate_deployability",
)
RESERVE_CHANNELS = (
    "delta_deployability",
    "delta_smooth_support",
    "candidate_deployability",
    "candidate_smooth_support",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open() as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except Exception as e:
                raise ValueError(f"invalid JSONL {path}:{i}: {e}")
    return out


def prop_key(r: dict[str, Any]) -> tuple[str, int, int]:
    return str(r["scene"]), int(r["time"]), int(r["candidate"])


def label_key(r: dict[str, Any]) -> tuple[str, int, int]:
    return str(r["scene_id"]), int(r["time_index"]), int(r["candidate_index"])


def finite_auc(pos: list[float], neg: list[float]) -> float | None:
    return tie_auc(pos, neg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v94-main", type=Path, required=True)
    ap.add_argument("--v93-audit", type=Path, required=True)
    ap.add_argument("--v94-comparison", type=Path, required=True)
    ap.add_argument("--v94-pipeline", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()

    errors: list[str] = []
    cmp94 = json.loads(a.v94_comparison.read_text())
    pipe94 = json.loads(a.v94_pipeline.read_text())
    if not bool(cmp94.get("valid")) or not bool(pipe94.get("valid")):
        errors.append("V48.94 prerequisite is not valid")
    if str(cmp94.get("preregistered_decision", {}).get("status")) != "SUPPORT_RESERVE_COMPLEMENTARITY_STOP":
        errors.append("V48.95 requires V48.94 SUPPORT_RESERVE_COMPLEMENTARITY_STOP branch")
    if bool(cmp94.get("preregistered_decision", {}).get("absolute_source_freeze_authorized")):
        errors.append("V48.95 must not run after absolute-source freeze authorization")

    labels_by_role: dict[str, dict[tuple[str, int, int], dict[str, Any]]] = {r: {} for r in ROLES}
    for y in load_jsonl(a.v93_audit):
        role = str(y.get("dataset_role"))
        if role not in labels_by_role:
            continue
        k = label_key(y)
        if k in labels_by_role[role]:
            errors.append(f"duplicate V48.93 label {role}/{k}")
        labels_by_role[role][k] = y

    cells: dict[str, dict[str, Any]] = {}
    dedup: dict[str, dict[tuple[str, int, int], dict[str, Any]]] = {r: {} for r in ROLES}

    for variant in VARIANTS:
        cells[variant] = {}
        root = a.v94_main / "candidates" / variant / "evaluation"
        for role in ROLES:
            path = root / ROLE_FILES[role]
            if not path.is_file():
                errors.append(f"missing V48.94 proposal rows {path}")
                continue
            joined: list[tuple[dict[str, Any], dict[str, Any], dict[str, float]]] = []
            for r in load_jsonl(path):
                k = prop_key(r)
                y = labels_by_role[role].get(k)
                if y is None:
                    errors.append(f"missing V48.93 label {variant}/{role}/{k}")
                    continue
                try:
                    feat = frozen_native_features(r.get("native_candidate_certificate"), r.get("native_nominal_certificate"))
                except Exception as e:
                    errors.append(f"native certificate invalid {variant}/{role}/{k}: {e}")
                    continue
                joined.append((r, y, feat))
                rec = {
                    "label": y,
                    "features": feat,
                    "teacher_nominal_drs": float(y.get("nominal_drs", float("nan"))),
                }
                old = dedup[role].get(k)
                if old is not None:
                    # balanced/precision must expose execution-identical frozen certificate diagnostics.
                    for fk, fv in feat.items():
                        if abs(float(old["features"][fk]) - float(fv)) > 1e-6:
                            errors.append(f"cross-variant native feature mismatch {role}/{k}/{fk}")
                            break
                else:
                    dedup[role][k] = rec

            safe = [(r, y, f) for r, y, f in joined if bool(y.get("safe_positive"))]
            drs_safe = [(r, y, f) for r, y, f in safe if y.get("mediation_mode") == "drs_activation"]
            dep_safe = [(r, y, f) for r, y, f in safe if y.get("mediation_mode") == "deployability_gain"]
            harmful_support = [(r, y, f) for r, y, f in joined if bool(y.get("teacher_harmful")) and abs(float(y.get("nominal_drs", 0.0))) < 1e-9]
            harmful_reserve = [(r, y, f) for r, y, f in joined if bool(y.get("teacher_harmful")) and abs(float(y.get("nominal_drs", 0.0)) - 1.0) < 1e-9]

            state_auc = {}
            for ch in STATE_CHANNELS:
                state_auc[ch] = finite_auc(
                    [f[ch] for _, y, f in dep_safe],
                    [f[ch] for _, y, f in drs_safe],
                )
            support_auc = {}
            for ch in SUPPORT_CHANNELS:
                support_auc[ch] = finite_auc(
                    [f[ch] for _, y, f in drs_safe],
                    [f[ch] for _, y, f in harmful_support],
                )
            reserve_auc = {}
            for ch in RESERVE_CHANNELS:
                reserve_auc[ch] = finite_auc(
                    [f[ch] for _, y, f in dep_safe],
                    [f[ch] for _, y, f in harmful_reserve],
                )
            cells[variant][role] = {
                "rows": len(joined),
                "safe_positive_rows": len(safe),
                "drs_activation_safe_rows": len(drs_safe),
                "deployability_gain_safe_rows": len(dep_safe),
                "harmful_support_state_rows": len(harmful_support),
                "harmful_reserve_state_rows": len(harmful_reserve),
                "state_auc": state_auc,
                "support_action_auc": support_auc,
                "reserve_action_auc": reserve_auc,
            }

    # Unique role diagnostics: exact-zero boundary accuracy and certificate saturation.
    role_diag: dict[str, Any] = {}
    for role in ROLES:
        recs = list(dedup[role].values())
        safe = [x for x in recs if bool(x["label"].get("safe_positive")) and x["label"].get("mediation_mode") in {"drs_activation", "deployability_gain"}]
        state_good = []
        for x in safe:
            pred = "support_establishment" if float(x["features"]["nominal_hard_support"]) <= 0.0 else "reserve_debt"
            expected = "support_establishment" if x["label"].get("mediation_mode") == "drs_activation" else "reserve_debt"
            state_good.append(pred == expected)
        role_diag[role] = {
            "safe_mode_rows": len(safe),
            "hard_zero_state_accuracy": (sum(state_good) / len(state_good) if state_good else None),
            "nominal_hard_zero_fraction": (sum(float(x["features"]["nominal_hard_support"]) <= 0.0 for x in safe) / len(safe) if safe else None),
            "candidate_hard_positive_safe_fraction": (
                sum(float(x["features"]["candidate_hard_support"]) > 0.0 for x in safe) / len(safe) if safe else None
            ),
        }

    out = {
        "schema": SCHEMA,
        "engineering_version": ENGINEERING_VERSION,
        "valid": not errors,
        "attribution_ready": not errors,
        "errors": errors,
        "experiment_type": "audit_only_frozen_native_support_reserve_observability",
        "planner_parameters_trained": 0,
        "dataset_reconstruction": False,
        "dataset_reselection": False,
        "teacher_labels_changed": False,
        "teacher_metadata_input_to_model": False,
        "boundary_transport": False,
        "relative_ranker_modified": False,
        "regime_conditioning": False,
        "womd_replay_performed": False,
        "same_v48_94_proposal_rows": True,
        "state_channels": list(STATE_CHANNELS),
        "support_channels": list(SUPPORT_CHANNELS),
        "reserve_channels": list(RESERVE_CHANNELS),
        "role_diagnostics": role_diag,
        "cells": cells,
        "test_roots_read": False,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"valid": out["valid"], "errors": errors[:5], "roles": role_diag}, sort_keys=True))
    return 0 if out["valid"] else 30


if __name__ == "__main__":
    raise SystemExit(main())
