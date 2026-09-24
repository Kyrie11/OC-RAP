#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

EXPECTED_CLOSURE_STATUS = "TERMINAL_INTERNAL_CLOSURE_COMPLETE_MAIN_NOT_FROZEN"
ENGINEERING_VERSION = "v48.124.10.7.3-FINAL-EVALUATION-LOCK"
SCIENTIFIC_VERSION = "v48.124-OC-FMSA"

RUNTIME_FILES = [
    "src/ocrap/planning/selector.py",
    "src/ocrap/simulation/closed_loop_runner.py",
    "src/ocrap/simulation/waymax_rollout.py",
    "src/ocrap/evaluation/metrics.py",
    "scripts/run_ocrap_closed_loop.sh",
    "scripts/run_ocrap_three_regime_evaluation.sh",
    "scripts/run_nominal_three_regime_control.sh",
    "scripts/run_external_baselines.sh",
    "scripts/run_external_baselines_safe.sh",
    "scripts/run_external_baselines_near.sh",
    "scripts/run_external_baselines_contact.sh",
    "tools/build_regime_comparison_tables.py",
    "tools/build_submission_external_baseline_tables.py",
    "configs/default.yaml",
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(path: Path, root: Path | None = None) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "relative_path": str(path.resolve().relative_to(root.resolve())) if root is not None else None,
        "sha256": sha(path),
        "size": path.stat().st_size,
    }


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def resolve_variant(model_run: Path, variant: str) -> tuple[Path, Path]:
    candidates = model_run / "candidates" / variant
    dedicated = model_run / "dedicated_candidates" / variant
    root = candidates
    if not (candidates / "model_v48_trac_sr" / "best.pt").is_file() and (dedicated / "model_v48_trac_sr" / "best.pt").is_file():
        root = dedicated
    ckpt = root / "model_v48_trac_sr" / "best.pt"
    calib = root / "calibration" / "gamma_rec_by_bucket_v48.json"
    return ckpt, calib


def main() -> int:
    ap = argparse.ArgumentParser(description="Create an immutable final-evaluation lock after terminal internal convergence without relabeling the failed V48.124 deployment-acceptance gate as GO.")
    ap.add_argument("--terminal-closure", type=Path, required=True)
    ap.add_argument("--model-run", type=Path, required=True)
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    errors: list[str] = []
    closure: dict[str, Any] = {}
    try:
        closure = load(args.terminal_closure)
    except Exception as exc:
        errors.append(f"closure_parse:{type(exc).__name__}")

    sc = closure.get("scientific_closure") or {}
    dc = closure.get("deployment_closure") or {}
    if closure:
        if not closure.get("valid") or not closure.get("attribution_ready"):
            errors.append("terminal_closure_not_attribution_ready")
        if closure.get("status") != EXPECTED_CLOSURE_STATUS:
            errors.append(f"unexpected_terminal_closure_status:{closure.get('status')}")
        if sc.get("internal_mechanism_convergence") is not True:
            errors.append("internal_mechanism_convergence_not_closed")
        if sc.get("recovery_mechanism_search") != "FROZEN":
            errors.append("recovery_mechanism_search_not_frozen")
        if sc.get("absolute_admission_repair_hypothesis") != "CLOSED":
            errors.append("absolute_admission_repair_not_closed")
        if sc.get("new_internal_algorithm_iteration_authorized") is not False:
            errors.append("new_internal_iteration_not_forbidden")
        if dc.get("deployed_main_freeze_authorized") is not False:
            errors.append("terminal_closure_unexpected_acceptance_freeze")

    repo = args.repo.resolve()
    runtime: dict[str, Any] = {}
    for rel in RUNTIME_FILES:
        p = repo / rel
        if not p.is_file():
            errors.append(f"missing_runtime_file:{rel}")
        else:
            runtime[rel] = record(p, repo)

    variants: dict[str, Any] = {}
    for variant in ("balanced", "precision"):
        ckpt, calib = resolve_variant(args.model_run.resolve(), variant)
        if not ckpt.is_file():
            errors.append(f"missing_checkpoint:{variant}:{ckpt}")
        if not calib.is_file():
            errors.append(f"missing_calibration:{variant}:{calib}")
        variants[variant] = {
            "checkpoint": record(ckpt) if ckpt.is_file() else {"path": str(ckpt)},
            "calibration": record(calib) if calib.is_file() else {"path": str(calib)},
        }

    valid = not errors
    out = {
        "schema": "ocrap-v48.124.10.7.3-final-evaluation-lock-v1",
        "engineering_version": ENGINEERING_VERSION,
        "scientific_version": SCIENTIFIC_VERSION,
        "algorithm_modified": False,
        "gpu_experiment": False,
        "valid": valid,
        "attribution_ready": valid,
        "errors": errors,
        "status": "FINAL_EVALUATION_LOCKED_ACCEPTANCE_GATE_REMAINS_STOP" if valid else "FINAL_EVALUATION_LOCK_NOT_ENTERED",
        "terminal_closure": record(args.terminal_closure) if args.terminal_closure.is_file() else {"path": str(args.terminal_closure)},
        "model_run": str(args.model_run.resolve()),
        "variants": variants,
        "runtime_files": runtime,
        "freeze_semantics": {
            "internal_algorithm_search_frozen": True if valid else None,
            "no_further_model_or_threshold_tuning_after_lock": True if valid else None,
            "immutable_submission_evaluation_snapshot_authorized": True if valid else False,
            "final_three_regime_characterization_authorized": True if valid else False,
            "paired_external_baseline_characterization_authorized": True if valid else False,
            "deployed_main_acceptance_freeze_authorized": False,
            "acceptance_gate_reason": (
                "The preregistered V48.124 deployment-acceptance contract still requires Coverage+Determinism+Safe+Near+Contact GO. "
                "Terminal diagnostics closed internal mechanism search but did not repair the deployed Near gate."
            ),
            "claim_scope": (
                "Final locked characterization for paper reporting. This lock freezes artifacts and forbids further tuning; "
                "it does not relabel the historical V48.124 Near STOP as GO and must not be cited as deployment-acceptance success."
            ),
            "contact_reporting_rule": (
                "External Contact tables may compare generic physical metrics on the exact paired target cohort. "
                "post_contact_* metrics are fully paired post-impact evidence only if every compared method has observed-contact eligibility 1.0."
            ),
        },
        "next_branch": (
            "run_final_locked_three_regime_characterization_then_paired_external_baselines_and_build_tables_no_v48_125_no_retuning"
            if valid else "fix_engineering_or_provenance_only"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"valid": valid, "status": out["status"], "output": str(args.output)}, indent=2))
    return 0 if valid else 30


if __name__ == "__main__":
    raise SystemExit(main())
