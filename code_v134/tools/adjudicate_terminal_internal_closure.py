#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

ONE_SHOT_JSON = "OC-RAP-v48.124.10.7.1-ONE-SHOT-ACTION-REALIZATION.json"
ONE_SHOT_MANIFEST = "OC-RAP-v48.124.10.7.1-result-bundle-manifest.json"
PREDECESSOR_JSON = "reference/OC-RAP-v48.124.10.6-NONFLOOR-ADMISSION-SEED-SCREEN.json"
EXPECTED_ONE_SHOT_STATUS = "ONE_SHOT_ACTION_REALIZATION_MIXED"
EXPECTED_ONE_SHOT_NEXT = (
    "candidate_physical_effect_is_not_uniform_under_near_endpoints_"
    "close_absolute_admission_repair_keep_mechanism_frozen"
)
EXPECTED_PREDECESSOR_STATUS = "NONFLOOR_ADMISSION_SEED_SCREEN_NOT_PROMISING"


def _loads(raw: bytes) -> dict[str, Any]:
    return json.loads(raw.decode("utf-8"))


def _verify_bundle(path: Path) -> tuple[dict[str, Any], dict[str, Any], list[str], str]:
    errors: list[str] = []
    if not path.is_file():
        return {}, {}, [f"missing_bundle:{path}"], ""
    bundle_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    with zipfile.ZipFile(path, "r") as zf:
        names = set(zf.namelist())
        for required in (ONE_SHOT_JSON, ONE_SHOT_MANIFEST, PREDECESSOR_JSON):
            if required not in names:
                errors.append(f"missing_member:{required}")
        if errors:
            return {}, {}, errors, bundle_sha
        manifest = _loads(zf.read(ONE_SHOT_MANIFEST))
        files = manifest.get("files") or {}
        if int(manifest.get("pipeline_exit_code", -1)) != 0 or not bool(manifest.get("complete_exit_zero")):
            errors.append("pipeline_not_exit_zero")
        if int(manifest.get("num_files", -1)) != len(files):
            errors.append("manifest_num_files_mismatch")
        for rel, ent in files.items():
            if rel not in names:
                errors.append(f"manifest_missing_member:{rel}")
                continue
            raw = zf.read(rel)
            if hashlib.sha256(raw).hexdigest() != str((ent or {}).get("sha256")):
                errors.append(f"manifest_sha_mismatch:{rel}")
            if len(raw) != int((ent or {}).get("size", -1)):
                errors.append(f"manifest_size_mismatch:{rel}")
        one_shot = _loads(zf.read(ONE_SHOT_JSON))
        predecessor = _loads(zf.read(PREDECESSOR_JSON))
    return one_shot, predecessor, errors, bundle_sha


def _variant_contract_errors(one_shot: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    variants = one_shot.get("variants") or {}
    for variant in ("balanced", "precision"):
        row = variants.get(variant) or {}
        if not bool(row.get("exact_one_shot_contract")):
            errors.append(f"{variant}:exact_one_shot_contract_false")
        if not bool(row.get("hard_no_harm")):
            errors.append(f"{variant}:hard_no_harm_false")
        seed = row.get("seed_execution") or {}
        if len(seed) != 2:
            errors.append(f"{variant}:seed_scene_count_not_two")
        effects = row.get("effects") or {}
        scene_effects = effects.get("scene_effects") or {}
        classes = sorted(str((v or {}).get("classification")) for v in scene_effects.values())
        if classes != ["locally_positive", "worsened"]:
            errors.append(f"{variant}:expected_one_positive_one_worsened:{classes}")
    if (variants.get("balanced") or {}).get("effects") != (variants.get("precision") or {}).get("effects"):
        errors.append("balanced_precision_effects_not_identical")
    if (variants.get("balanced") or {}).get("seed_execution") != (variants.get("precision") or {}).get("seed_execution"):
        errors.append("balanced_precision_seed_execution_not_identical")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Close the V48.124.10.x internal Near diagnostic chain after the terminal 10.7.1 one-shot result. "
            "This tool performs no training, no GPU simulation, and no deployable algorithm change."
        )
    )
    ap.add_argument("--one-shot-results", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    bundle = Path(args.one_shot_results)
    one_shot, predecessor, errors, bundle_sha = _verify_bundle(bundle)

    if one_shot:
        if not bool(one_shot.get("valid")) or not bool(one_shot.get("attribution_ready")):
            errors.append("one_shot_not_attribution_ready")
        if bool(one_shot.get("algorithm_modified")):
            errors.append("one_shot_unexpected_algorithm_modified")
        if bool(one_shot.get("publication_evidence")):
            errors.append("one_shot_unexpected_publication_evidence")
        if one_shot.get("status") != EXPECTED_ONE_SHOT_STATUS:
            errors.append(f"unexpected_one_shot_status:{one_shot.get('status')}")
        if one_shot.get("next_branch") != EXPECTED_ONE_SHOT_NEXT:
            errors.append("unexpected_one_shot_next_branch")
        errors.extend(_variant_contract_errors(one_shot))

    if predecessor:
        if not bool(predecessor.get("valid")) or not bool(predecessor.get("attribution_ready")):
            errors.append("predecessor_not_attribution_ready")
        if predecessor.get("status") != EXPECTED_PREDECESSOR_STATUS:
            errors.append(f"unexpected_predecessor_status:{predecessor.get('status')}")

    valid = not errors
    status = (
        "TERMINAL_INTERNAL_CLOSURE_COMPLETE_MAIN_NOT_FROZEN"
        if valid
        else "TERMINAL_INTERNAL_CLOSURE_ATTRIBUTION_NOT_ENTERED"
    )
    out: dict[str, Any] = {
        "schema": "ocrap-v48.124.10.7.2-terminal-internal-closure-v1",
        "engineering_version": "v48.124.10.7.2-TERMINAL-INTERNAL-CLOSURE",
        "scientific_version": "v48.124-OC-FMSA",
        "algorithm_modified": False,
        "gpu_experiment": False,
        "publication_evidence": False,
        "valid": valid,
        "attribution_ready": valid,
        "errors": errors,
        "status": status,
        "inputs": {
            "one_shot_results": str(bundle),
            "one_shot_results_sha256": bundle_sha,
            "one_shot_status": one_shot.get("status") if one_shot else None,
            "predecessor_status": predecessor.get("status") if predecessor else None,
        },
        "scientific_closure": {
            "repeated_privileged_trajectory_confounding_removed": True if valid else None,
            "one_shot_candidate_physical_effect_uniform": False if valid else None,
            "absolute_admission_repair_hypothesis": "CLOSED" if valid else "NOT_ADJUDICATED",
            "recovery_mechanism_search": "FROZEN" if valid else "UNCHANGED",
            "internal_mechanism_convergence": True if valid else False,
            "new_internal_algorithm_iteration_authorized": False,
            "new_threshold_or_capacity_sweep_authorized": False,
            "new_recovery_mechanism_authorized": False,
        },
        "deployment_closure": {
            "deployed_main_changed_by_10_7": False,
            "deployed_main_freeze_authorized": False,
            "reason": (
                "V48.124 Main freeze still requires Coverage+Determinism+Safe+Near+Contact GO; "
                "10.7.1 is diagnostic-only and the unresolved Near system gate is not repaired."
            ),
            "final_three_regime_submission_test_authorized": False,
            "external_baseline_submission_comparison_authorized": False,
            "additional_deployed_realization_gpu_closure_authorized": False,
        },
        "next_branch": (
            "stop_internal_algorithm_iteration_keep_mechanism_frozen_do_not_run_another_realization_diagnostic_"
            "retain_current_deployed_main_as_not_frozen_and_move_to_paper_limitation_reporting_unless_new_independent_evidence_arrives"
            if valid
            else "fix_engineering_or_provenance_only"
        ),
        "interpretation_rule": (
            "This is bookkeeping closure of the preregistered terminal diagnostic chain, not a new experiment. "
            "A complete closure freezes internal mechanism search while explicitly withholding deployed-Main freeze."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"valid": valid, "status": status, "next_branch": out["next_branch"], "output": str(output)}, indent=2))
    return 0 if valid else 30


if __name__ == "__main__":
    raise SystemExit(main())
