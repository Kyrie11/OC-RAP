#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ocrap.audits.fixed_main_stability import (
    NEAR_BENEFIT,
    NEAR_HARD_NO_HARM,
    _gate_benefit,
    _gate_no_harm,
    target_keys,
)

HISTORICAL_STATUS = "FIXED_MAIN_NEAR_VALIDITY_STOP"
ENGINEERING_VERSION = "v48.124.10-OC-FMSA-NEAR-RIFA-SYSTEM-AXIS-AUDIT"
SECONDARY_NO_HARM = {
    "closed_loop_bounded_NUP": "higher",
    "route_progression_m": "higher",
    "near_contact_exposure_duration_s": "lower",
}


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def arm_gate(comps: dict[str, dict[str, Any]]) -> dict[str, Any]:
    variants = ("balanced", "precision")
    common: set[str] | None = None
    rows: dict[str, Any] = {}
    for v in variants:
        hard = _gate_no_harm(comps[v], NEAR_HARD_NO_HARM)
        benefit = _gate_benefit(comps[v], NEAR_BENEFIT)
        secondary = _gate_no_harm(comps[v], SECONDARY_NO_HARM)
        bset = set(benefit["beneficial_metrics"])
        common = bset if common is None else common & bset
        rows[v] = {
            "primary_near_go": bool(hard["go"] and benefit["go"]),
            "hard_no_harm": hard,
            "benefit": benefit,
            "secondary_system_no_harm": secondary,
        }
    common = common or set()
    primary_go = bool(all(rows[v]["primary_near_go"] for v in variants) and common)
    system_clean = bool(all(rows[v]["secondary_system_no_harm"]["go"] for v in variants))
    return {
        "primary_near_gate_go": primary_go,
        "common_beneficial_metrics": sorted(common),
        "secondary_system_no_harm_go": system_clean,
        "promotion_ready": bool(primary_go and system_clean),
        "variants": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Adjudicate V48.124.10 diagnostic-only Near RIFA system-axis arms.")
    ap.add_argument("--historical-adjudication", type=Path, required=True)
    ap.add_argument("--cohort-audit", type=Path, required=True)
    ap.add_argument("--runtime-contract", type=Path, required=True)
    ap.add_argument("--reference-contract", type=Path, required=True)
    for arm in ("delta", "nested"):
        for variant in ("balanced", "precision"):
            ap.add_argument(f"--{arm}-{variant}-full", type=Path, required=True)
            ap.add_argument(f"--{arm}-{variant}-vs-nominal", type=Path, required=True)
            ap.add_argument(f"--{arm}-{variant}-vs-historical", type=Path, required=True)
    ap.add_argument("--historical-balanced-full", type=Path, required=True)
    ap.add_argument("--historical-precision-full", type=Path, required=True)
    ap.add_argument("--nominal-full", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()

    errors: list[str] = []
    hist = load(a.historical_adjudication)
    if not (hist.get("valid") and hist.get("attribution_ready") and (hist.get("preregistered_decision") or {}).get("status") == HISTORICAL_STATUS):
        # Some bundles expose decision fields at top level; accept only the same status.
        if not (hist.get("valid") and hist.get("attribution_ready") and hist.get("status") == HISTORICAL_STATUS):
            errors.append("historical_v48_124_9_near_stop_not_established")
    runtime = load(a.runtime_contract)
    reference = load(a.reference_contract)
    if not (runtime.get("valid") and runtime.get("attribution_ready") and runtime.get("engineering_version") == ENGINEERING_VERSION):
        errors.append("near_axis_runtime_contract_invalid")
    if not reference.get("valid"):
        errors.append("historical_reference_contract_invalid")
    cohort = load(a.cohort_audit)
    if not cohort.get("valid"):
        errors.append("cohort_audit_invalid")
    if int(cohort.get("intervention_scene_count") or 0) <= 0:
        errors.append("empty_intervention_cohort")

    nominal = load(a.nominal_full)
    hfull = {
        "balanced": load(a.historical_balanced_full),
        "precision": load(a.historical_precision_full),
    }
    nkeys = target_keys(nominal)
    if len(nkeys) != 250:
        errors.append(f"unexpected_nominal_near_population:{len(nkeys)}")
    for v in ("balanced", "precision"):
        if target_keys(hfull[v]) != nkeys:
            errors.append(f"historical_target_mismatch:{v}")

    arm_docs: dict[str, Any] = {}
    expected_selector = {
        "delta": "lcb_constrained_relative_delta",
        "nested": "lcb_constrained_nested_evidence",
    }
    for arm in ("delta", "nested"):
        comps: dict[str, dict[str, Any]] = {}
        variants: dict[str, Any] = {}
        for v in ("balanced", "precision"):
            full = load(getattr(a, f"{arm}_{v}_full"))
            comp_nom = load(getattr(a, f"{arm}_{v}_vs_nominal"))
            comp_hist = load(getattr(a, f"{arm}_{v}_vs_historical"))
            if target_keys(full) != nkeys:
                errors.append(f"diagnostic_target_mismatch:{arm}:{v}")
            recon = full.get("exact_monotone_subset_reconstruction") or {}
            if not recon.get("internal_diagnostic_only"):
                errors.append(f"missing_monotone_reconstruction_contract:{arm}:{v}")
            scfg = full.get("selector_config") or {}
            selector = str(scfg.get("ocrap_selector") or "")
            if selector != expected_selector[arm]:
                errors.append(f"selector_contract:{arm}:{v}:{selector}")
            try:
                if abs(float(scfg.get("rifa_relative_min_advantage")) - 0.0) > 1e-12:
                    errors.append(f"relative_min_advantage_contract:{arm}:{v}")
                if arm == "nested":
                    if int(scfg.get("rifa_relative_proposal_top_k")) != 5:
                        errors.append(f"relative_topk_contract:{arm}:{v}")
                    if abs(float(scfg.get("rifa_relative_opportunity_threshold")) - 0.65) > 1e-12:
                        errors.append(f"relative_opportunity_contract:{arm}:{v}")
                    if abs(float(scfg.get("rifa_relative_harm_threshold")) - 0.30) > 1e-12:
                        errors.append(f"relative_harm_contract:{arm}:{v}")
            except Exception:
                errors.append(f"relative_metadata_contract:{arm}:{v}")
            if int(comp_nom.get("num_paired_scenes") or -1) != len(nkeys):
                errors.append(f"nominal_pair_coverage:{arm}:{v}")
            if int(comp_hist.get("num_paired_scenes") or -1) != len(nkeys):
                errors.append(f"historical_pair_coverage:{arm}:{v}")
            comps[v] = comp_nom
            variants[v] = {
                "selector": selector,
                "num_scenes": full.get("num_scenes"),
                "intervention_rate": full.get("intervention_rate"),
                "comparison_vs_nominal": comp_nom,
                "comparison_vs_historical": comp_hist,
            }
        arm_docs[arm] = {"gate": arm_gate(comps), "variants": variants}

    if errors:
        status = "NEAR_RIFA_SYSTEM_AXIS_ATTRIBUTION_NOT_ENTERED"
        next_branch = "fix_diagnostic_engineering_only"
        promoted = None
    else:
        delta_go = bool(arm_docs["delta"]["gate"]["promotion_ready"])
        nested_go = bool(arm_docs["nested"]["gate"]["promotion_ready"])
        # Minimal-sufficient intervention is preferred.  If the sign-only arm
        # already closes the Near gate cleanly, the stricter nested arm is not
        # promoted merely because it also works.  This prevents unnecessary
        # deployment complexity and post-hoc mechanism accretion.
        if delta_go:
            status = "NEAR_RIFA_RELATIVE_SIGN_GO"
            promoted = "delta"
            next_branch = "freeze_relative_sign_system_fix_and_run_fresh_full_250_near_confirmation_no_mechanism_search"
        elif nested_go:
            status = "NEAR_RIFA_NESTED_EVIDENCE_GO_DELTA_STOP"
            promoted = "nested"
            next_branch = "freeze_nested_role_isolation_system_fix_and_run_fresh_full_250_near_confirmation_no_mechanism_search"
        else:
            status = "NEAR_RIFA_SYSTEM_AXIS_STOP"
            promoted = None
            next_branch = (
                "keep_recovery_mechanism_family_frozen_and_audit_the_35_intervention_scenes_for_"
                "absolute_admission_candidate_availability_and_counterfactual_action_quality_no_threshold_or_capacity_sweep"
            )

    out = {
        "schema": "ocrap-v48.124.10-near-rifa-system-axis-v2",
        "engineering_version": ENGINEERING_VERSION,
        "valid": not errors,
        "attribution_ready": not errors,
        "historical_status": HISTORICAL_STATUS,
        "diagnostic_scope": "Near deployed-selector/system-integration axis only; no training, recalibration, recovery-mechanism, capacity, regime, source, horizon, or threshold sweep",
        "errors": errors,
        "runtime_contract": {"path": str(a.runtime_contract), "snapshot": runtime.get("snapshot")},
        "historical_reference_contract": {"path": str(a.reference_contract), "source": (reference.get("reference_provenance") or {}).get("source")},
        "status": status,
        "promoted_diagnostic_arm": promoted,
        "next_branch": next_branch,
        "arms": arm_docs,
        "publication_rule": (
            "A reconstructed 35-scene diagnostic can authorize only the next selector candidate. "
            "Paper main results require a fresh complete 250-scene Near run after the selector is frozen, "
            "followed by the unchanged Contact gate and a new scene-disjoint confirmatory evaluation."
        ),
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"valid": out["valid"], "status": status, "next_branch": next_branch, "output": str(a.output)}, indent=2))
    return 0 if not errors else 30


if __name__ == "__main__":
    raise SystemExit(main())
