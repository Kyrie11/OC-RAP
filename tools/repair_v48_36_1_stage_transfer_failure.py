#!/usr/bin/env python3
"""Repair the exact v48.36.1 false RC=30 stage-transfer failure without retraining.

This tool is intentionally narrow.  It accepts only the uploaded failure shape:
both variants completed factor/identity training, both stopped at the legacy
v48.32 transfer checker with RC=31, and the only allegedly frozen parameters
were OCAF interaction-bridge parameters that the identity-stage architecture
explicitly registered as trainable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

BASE_ALGORITHM_VERSION = "v48.36-OCAF"
IMPLEMENTATION_VERSION = "v48.36.2-STAGE-TRANSFER-HOTFIX"
IDENTITY_PREFIXES = (
    "direct_evidence_concord_benefit_calibrator",
    "direct_evidence_concord_harm_calibrator",
    "direct_evidence_concord_admission_calibrator",
    "direct_evidence_interaction_bridge",
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object in {path}")
    return value


def _atomic_json(path: Path, doc: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(doc, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _trainable(arch: Mapping[str, Any]) -> tuple[str, ...]:
    raw = arch.get("trainable")
    if isinstance(raw, str):
        text = raw
    elif isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], str):
        text = raw[0]
    else:
        raise TypeError("malformed trainable metadata")
    return tuple(item.strip() for item in text.split(",") if item.strip())


def _run(command: list[str], *, cwd: Path) -> None:
    completed = subprocess.run(command, cwd=cwd, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed with rc={completed.returncode}: "
            + " ".join(command)
        )


def _same_path(recorded: Any, expected: Path) -> bool:
    return Path(str(recorded)).resolve() == expected.resolve()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Repair the exact v48.36.1 OCAF stage-transfer false failure without retraining"
    )
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--source-run", type=Path, required=True)
    ap.add_argument("--protocol-root", type=Path, required=True)
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    root = args.run
    repo = args.repo.resolve()
    output = args.output or root / "V48_36_2_STAGE_TRANSFER_REPAIR.json"
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    errors: list[str] = []

    def record(name: str, condition: bool, detail: Any = None) -> None:
        checks[name] = bool(condition)
        if detail is not None:
            details[name] = detail
        if not condition:
            errors.append(name)

    try:
        failed = _json(root / "PIPELINE_FAILED.json")
        complete = _json(root / "V48_36_COMPLETE.json")
        attempt = _json(root / "ATTEMPT_STARTED.json")
    except Exception as exc:
        doc = {
            "event": "v48_36_2_stage_transfer_repair",
            "version": BASE_ALGORITHM_VERSION,
            "implementation_version": IMPLEMENTATION_VERSION,
            "created_unix": time.time(),
            "run": str(root),
            "valid": False,
            "errors": [f"status_read_error: {type(exc).__name__}: {exc}"],
            "test_roots_read": False,
        }
        _atomic_json(output, doc)
        print(json.dumps(doc, ensure_ascii=False))
        return 4

    adaptation = failed.get("adaptation_exit_codes")
    adaptation = adaptation if isinstance(adaptation, Mapping) else {}
    record("pipeline_failure_event", failed.get("event") == "v48_36_pipeline_failed")
    record("pipeline_failure_stage", failed.get("stage") == "adaptation", failed.get("stage"))
    record("pipeline_raw_rc", int(failed.get("raw_exit_code", -1)) == 30)
    record("pipeline_normalized_rc", int(failed.get("normalized_exit_code", -1)) == 30)
    record(
        "both_variants_stopped_at_transfer_checker",
        adaptation.get("balanced") == 31 and adaptation.get("precision") == 31,
        dict(adaptation),
    )
    record(
        "certificate_and_gate_never_executed",
        failed.get("certificate_executed") is False
        and failed.get("gate_evaluated") is False
        and complete.get("certificate_executed") is False
        and complete.get("gate_evaluated") is False,
    )
    record(
        "test_roots_sealed",
        failed.get("test_roots_read") is False
        and complete.get("test_roots_read") is False
        and attempt.get("test_roots_read") is False,
    )
    record("source_run_matches", _same_path(complete.get("source_run"), args.source_run))
    record("protocol_root_matches", _same_path(complete.get("protocol_root"), args.protocol_root))
    record("attempt_protocol_matches", _same_path(attempt.get("protocol_root"), args.protocol_root))

    stale_certificate_artifacts: list[str] = []
    for variant in ("balanced", "precision"):
        calibration = root / "candidates" / variant / "calibration"
        if calibration.exists():
            stale_certificate_artifacts.append(str(calibration))
    for rel in ("NEXT_COMMANDS.txt", "GATE_FAILED.json", "CALIBRATION_FAILED.json", "GATE_SPEC.json"):
        path = root / rel
        if path.exists():
            stale_certificate_artifacts.append(str(path))
    record("no_certificate_artifacts", not stale_certificate_artifacts, stale_certificate_artifacts)

    controller_variants = complete.get("variants")
    controller_variants = controller_variants if isinstance(controller_variants, Mapping) else {}
    prepared: dict[str, dict[str, Any]] = {}
    for variant in ("balanced", "precision"):
        run = root / "candidates" / variant
        variant_checks: dict[str, bool] = {}
        try:
            stage_failure = _json(run / "VARIANT_STAGE_FAILED.json")
            old_transfer = _json(run / "STAGE_TRANSFER_INTEGRITY.json")
            factor_arch = _json(run / "factor_stage" / "STAGE_ARCHITECTURE.json")
            identity_arch = _json(run / "identity_stage" / "STAGE_ARCHITECTURE.json")
            final_arch = _json(run / "STAGE_ARCHITECTURE.json")
            factor = run / "factor_stage" / "model_v48_trac_sr" / "best.pt"
            identity = run / "identity_stage" / "model_v48_trac_sr" / "best.pt"
            final = run / "model_v48_trac_sr" / "best.pt"
            support = run / "FACTOR_SUPPORT_CONTRACT.json"
            source = args.source_run / "candidates" / variant / "model_v48_trac_sr" / "best.pt"
            required = [
                factor,
                identity,
                final,
                support,
                source,
                run / "factor_stage" / "TRAINING_COMPLETE.json",
                run / "identity_stage" / "TRAINING_COMPLETE.json",
                run / "TRAINING_COMPLETE.json",
            ]
            variant_checks["required_artifacts_exist"] = all(path.is_file() for path in required)
            variant_checks["failure_stage_exact"] = (
                stage_failure.get("stage") == "stage_transfer_integrity"
                and int(stage_failure.get("exit_code", -1)) == 31
            )
            variant_checks["old_checker_rejected"] = old_transfer.get("valid") is False
            reasons = old_transfer.get("failure_reasons") or []
            variant_checks["only_interaction_bridge_was_misclassified"] = bool(reasons) and all(
                "identity stage changed frozen parameters" in str(reason)
                and "direct_evidence_interaction_bridge." in str(reason)
                for reason in reasons
            )
            variant_checks["legacy_disallowed_count_is_bridge_count"] = int(
                old_transfer.get("identity_disallowed_changed_parameter_count", -1)
            ) > 0
            variant_checks["identity_architecture_exact"] = set(_trainable(identity_arch)) == set(
                IDENTITY_PREFIXES
            )
            variant_checks["final_stage_is_disabled_copy"] = (
                set(_trainable(final_arch)) == set(IDENTITY_PREFIXES)
                and old_transfer.get("final_stage_disabled") is True
                and old_transfer.get("final_disallowed_changed_parameter_count") == 0
            )
            variant_checks["ocaf_semantics_registered"] = all(
                arch.get("context_source") == "physical_interaction"
                and arch.get("observation_conditioned_action_frontier") is True
                and arch.get("regime_id_exposed_to_evidence_model") is False
                and arch.get("shared_deployment_rule_required") is True
                and arch.get("test_roots_read") is False
                for arch in (factor_arch, identity_arch, final_arch)
            )
            final_record = controller_variants.get(variant)
            final_record = final_record if isinstance(final_record, Mapping) else {}
            variant_checks["controller_final_hash_matches"] = (
                final.is_file() and _sha256(final) == final_record.get("sha256")
            )
        except Exception as exc:
            errors.append(f"{variant}_read_error")
            details[f"{variant}_read_error"] = repr(exc)
            continue

        for name, condition in variant_checks.items():
            record(f"{variant}_{name}", condition)
        prepared[variant] = {
            "run": run,
            "source": source,
            "factor": factor,
            "identity": identity,
            "final": final,
            "support": support,
            "identity_arch": identity_arch,
            "old_transfer": run / "STAGE_TRANSFER_INTEGRITY.json",
            "stage_failure": run / "VARIANT_STAGE_FAILED.json",
            "temporary_transfer": run / ".STAGE_TRANSFER_INTEGRITY.v48.36.2.tmp.json",
        }

    if errors:
        doc = {
            "event": "v48_36_2_stage_transfer_repair",
            "version": BASE_ALGORITHM_VERSION,
            "implementation_version": IMPLEMENTATION_VERSION,
            "created_unix": time.time(),
            "run": str(root),
            "source_run": str(args.source_run),
            "protocol_root": str(args.protocol_root),
            "valid": False,
            "checks": checks,
            "details": details,
            "errors": errors,
            "test_roots_read": False,
        }
        _atomic_json(output, doc)
        print(json.dumps(doc, ensure_ascii=False))
        return 4

    try:
        # Run the corrected checker into temporary files first so the old failure
        # evidence remains active unless both variants pass.
        for variant, item in prepared.items():
            temp = item["temporary_transfer"]
            temp.unlink(missing_ok=True)
            _run(
                [
                    sys.executable,
                    str(repo / "tools" / "check_v48_36_stage_transfer.py"),
                    "--factor",
                    str(item["factor"]),
                    "--identity",
                    str(item["identity"]),
                    "--final",
                    str(item["final"]),
                    "--identity-architecture",
                    str(item["run"] / "identity_stage" / "STAGE_ARCHITECTURE.json"),
                    "--final-architecture",
                    str(item["run"] / "STAGE_ARCHITECTURE.json"),
                    "--identity-allowed-prefixes",
                    ",".join(IDENTITY_PREFIXES),
                    "--final-allowed-prefixes",
                    "direct_evidence_concord_admission_calibrator",
                    "--final-stage-disabled",
                    "--implementation-version",
                    IMPLEMENTATION_VERSION,
                    "--output",
                    str(temp),
                ],
                cwd=repo,
            )
            corrected = _json(temp)
            if corrected.get("valid") is not True:
                raise RuntimeError(f"corrected stage-transfer contract failed for {variant}")
    
        repair_history = root / "repair_history" / f"v48.36.2-{time.time_ns()}"
        repair_history.mkdir(parents=True, exist_ok=False)
        variant_records: dict[str, Any] = {}
        for variant, item in prepared.items():
            old_transfer = item["old_transfer"]
            stage_failure = item["stage_failure"]
            shutil.copy2(old_transfer, repair_history / f"{variant}-STAGE_TRANSFER_INTEGRITY.v48.36.1.json")
            shutil.copy2(stage_failure, repair_history / f"{variant}-VARIANT_STAGE_FAILED.v48.36.1.json")
            os.replace(item["temporary_transfer"], old_transfer)
    
            architecture = item["identity_arch"]
            _run(
                [
                    sys.executable,
                    str(repo / "tools" / "finalize_v48_36_adaptation_variant.py"),
                    "--run",
                    str(item["run"]),
                    "--source",
                    str(item["source"]),
                    "--factor",
                    str(item["factor"]),
                    "--identity",
                    str(item["identity"]),
                    "--final",
                    str(item["final"]),
                    "--support",
                    str(item["support"]),
                    "--support-reliability-enabled",
                    "true",
                    "--identity-train-all",
                    "true",
                    "--prior-coupled",
                    str(architecture.get("admission_prior_detach") is False).lower(),
                    "--adaptive-margin",
                    str(float(architecture.get("safe_hard_negative_teacher_scale", 0.0)) > 0.0).lower(),
                    "--final-enabled",
                    "false",
                    "--context-source",
                    str(architecture.get("context_source")),
                    "--consensus-prior-scale",
                    str(architecture.get("consensus_prior_scale", 0.50)),
                    "--interaction-hidden",
                    str(architecture.get("interaction_hidden", 64)),
                    "--interaction-dropout",
                    str(architecture.get("interaction_dropout", 0.05)),
                    "--admission-prior-mode",
                    str(architecture.get("admission_prior_mode", "frontier_capped_slack")),
                    "--implementation-version",
                    IMPLEMENTATION_VERSION,
                ],
                cwd=repo,
            )
            stage_failure.unlink(missing_ok=True)
            three = item["run"] / "THREE_STAGE_TRAINING_COMPLETE.json"
            variant_records[variant] = {
                "factor_checkpoint": str(item["factor"]),
                "factor_sha256": _sha256(item["factor"]),
                "identity_checkpoint": str(item["identity"]),
                "identity_sha256": _sha256(item["identity"]),
                "final_checkpoint": str(item["final"]),
                "final_sha256": _sha256(item["final"]),
                "stage_transfer": str(old_transfer),
                "stage_transfer_sha256": _sha256(old_transfer),
                "three_stage_completion": str(three),
                "three_stage_completion_sha256": _sha256(three),
            }
    
        doc = {
            "event": "v48_36_2_stage_transfer_repair",
            "version": BASE_ALGORITHM_VERSION,
            "implementation_version": IMPLEMENTATION_VERSION,
            "created_unix": time.time(),
            "run": str(root),
            "source_run": str(args.source_run),
            "protocol_root": str(args.protocol_root),
            "valid": True,
            "algorithm_changed": False,
            "retraining_performed": False,
            "known_failure_signature": "both variants RC31 at legacy stage-transfer checker; only OCAF interaction bridge misclassified",
            "repair_history": str(repair_history),
            "checks": checks,
            "variants": variant_records,
            "authorized_next_action": "rerun main controller with RESUME_AFTER_ADAPTATION=1",
            "test_roots_read": False,
        }
        _atomic_json(output, doc)
        print(json.dumps(doc, ensure_ascii=False))
        return 0
    except Exception as exc:
        rejected_contracts: dict[str, Any] = {}
        for variant, item in prepared.items():
            temp = item["temporary_transfer"]
            if temp.is_file():
                rejected = item["run"] / "STAGE_TRANSFER_REPAIR_REJECTED.v48.36.2.json"
                os.replace(temp, rejected)
                try:
                    rejected_contracts[variant] = _json(rejected)
                except Exception as read_exc:
                    rejected_contracts[variant] = {"path": str(rejected), "read_error": repr(read_exc)}
        doc = {
            "event": "v48_36_2_stage_transfer_repair",
            "version": BASE_ALGORITHM_VERSION,
            "implementation_version": IMPLEMENTATION_VERSION,
            "created_unix": time.time(),
            "run": str(root),
            "source_run": str(args.source_run),
            "protocol_root": str(args.protocol_root),
            "valid": False,
            "algorithm_changed": False,
            "retraining_performed": False,
            "checks": checks,
            "details": details,
            "rejected_stage_transfer_contracts": rejected_contracts,
            "errors": [f"repair_execution_error: {type(exc).__name__}: {exc}"],
            "test_roots_read": False,
        }
        _atomic_json(output, doc)
        print(json.dumps(doc, ensure_ascii=False))
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
