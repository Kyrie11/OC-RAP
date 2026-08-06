#!/usr/bin/env python3
"""Authorize a no-retraining resume for exact, audited v48.36 RC=30 signatures.

Supported signatures are deliberately narrow: the earlier checkpoint-proven
training-contract metadata failure, and the v48.36.1 stage-transfer false
failure after a valid v48.36.2 repair contract.  Algorithmic RC=20, prior
certificate access, changed checkpoint bytes, changed data/source identity, or
any unregistered failure remains rejected.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected object in {path}")
    return value


def _checkpoint_cfg(path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError(f"checkpoint is not a mapping: {path}")
    cfg = payload.get("cfg")
    if not isinstance(cfg, Mapping):
        raise TypeError(f"checkpoint cfg missing: {path}")
    return cfg


def _path_equal(recorded: Any, expected: str) -> bool:
    if expected == "":
        return True
    return str(recorded) == expected


def _int_equal(value: Any, expected: int) -> bool:
    try:
        return int(value) == expected
    except (TypeError, ValueError):
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Fail-closed v48.36 RC30 no-retraining resume authorization")
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--expect-source-run", default="")
    ap.add_argument("--expect-protocol-root", default="")
    args = ap.parse_args()

    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    errors: list[str] = []

    def record(name: str, condition: bool, detail: Any = None) -> None:
        checks[name] = bool(condition)
        if detail is not None:
            details[name] = detail
        if not condition:
            errors.append(name)

    root = args.run
    try:
        failed = _json(root / "PIPELINE_FAILED.json")
        complete = _json(root / "V48_36_COMPLETE.json")
    except Exception as exc:
        failed = {}
        complete = {}
        record("prior_status_readable", False, repr(exc))
    else:
        record("prior_status_readable", True)

    repair_path = root / "V48_36_2_STAGE_TRANSFER_REPAIR.json"
    try:
        repair = _json(repair_path) if repair_path.is_file() else {}
    except Exception as exc:
        repair = {"read_error": repr(exc)}
    adaptation = failed.get("adaptation_exit_codes") if isinstance(failed.get("adaptation_exit_codes"), Mapping) else {}
    legacy_training_contract_failure = (
        failed.get("stage") == "training_contract"
        and _int_equal(failed.get("raw_exit_code"), 4)
        and adaptation.get("balanced") == 0
        and adaptation.get("precision") == 0
    )
    repaired_stage_transfer_failure = (
        failed.get("stage") == "adaptation"
        and _int_equal(failed.get("raw_exit_code"), 30)
        and adaptation.get("balanced") == 31
        and adaptation.get("precision") == 31
        and repair.get("event") == "v48_36_2_stage_transfer_repair"
        and repair.get("valid") is True
        and repair.get("algorithm_changed") is False
        and repair.get("retraining_performed") is False
        and repair.get("test_roots_read") is False
    )
    failure_mode = (
        "legacy_training_contract" if legacy_training_contract_failure
        else "repaired_stage_transfer" if repaired_stage_transfer_failure
        else "unsupported"
    )
    record("known_failure_event", failed.get("event") == "v48_36_pipeline_failed", failed)
    record("known_failure_signature", failure_mode != "unsupported", {"mode": failure_mode, "stage": failed.get("stage"), "raw_exit_code": failed.get("raw_exit_code"), "adaptation_exit_codes": adaptation})
    record("known_failure_normalized_rc", _int_equal(failed.get("normalized_exit_code"), 30), failed.get("normalized_exit_code"))
    record("adaptation_artifacts_authorized", legacy_training_contract_failure or repaired_stage_transfer_failure, adaptation)
    if repaired_stage_transfer_failure:
        record("stage_transfer_repair_source_matches", _path_equal(repair.get("source_run"), args.expect_source_run), repair.get("source_run"))
        record("stage_transfer_repair_protocol_matches", _path_equal(repair.get("protocol_root"), args.expect_protocol_root), repair.get("protocol_root"))
    record("certificate_not_executed", failed.get("certificate_executed") is False and complete.get("certificate_executed") is False)
    record("gate_not_evaluated", failed.get("gate_evaluated") is False and complete.get("gate_evaluated") is False)
    record("test_roots_sealed_in_status", failed.get("test_roots_read") is False and complete.get("test_roots_read") is False)
    record("source_run_unchanged", _path_equal(complete.get("source_run"), args.expect_source_run), complete.get("source_run"))
    record("protocol_root_unchanged", _path_equal(complete.get("protocol_root"), args.expect_protocol_root), complete.get("protocol_root"))

    stale_certificate_markers = []
    for variant in ("balanced", "precision"):
        calibration_dir = root / "candidates" / variant / "calibration"
        if calibration_dir.exists():
            stale_certificate_markers.append(str(calibration_dir))
    for rel in ("NEXT_COMMANDS.txt", "GATE_FAILED.json", "CALIBRATION_FAILED.json", "GATE_SPEC.json"):
        p = root / rel
        if p.exists():
            stale_certificate_markers.append(str(p))
    record("no_prior_certificate_artifacts", not stale_certificate_markers, stale_certificate_markers)

    controller_variants = complete.get("variants") if isinstance(complete.get("variants"), Mapping) else {}
    variant_details: dict[str, Any] = {}
    for variant in ("balanced", "precision"):
        run = root / "candidates" / variant
        v: dict[str, Any] = {"run": str(run)}
        variant_ok = True
        try:
            three = _json(run / "THREE_STAGE_TRAINING_COMPLETE.json")
            transfer = _json(run / "STAGE_TRANSFER_INTEGRITY.json")
            support = run / "FACTOR_SUPPORT_CONTRACT.json"
            paths = {
                "factor": run / "factor_stage" / "model_v48_trac_sr" / "best.pt",
                "identity": run / "identity_stage" / "model_v48_trac_sr" / "best.pt",
                "final": run / "model_v48_trac_sr" / "best.pt",
            }
            architectures = {
                "factor": _json(run / "factor_stage" / "STAGE_ARCHITECTURE.json"),
                "identity": _json(run / "identity_stage" / "STAGE_ARCHITECTURE.json"),
                "final": _json(run / "STAGE_ARCHITECTURE.json"),
            }
            completions = {
                "factor": _json(run / "factor_stage" / "TRAINING_COMPLETE.json"),
                "identity": _json(run / "identity_stage" / "TRAINING_COMPLETE.json"),
                "final": _json(run / "TRAINING_COMPLETE.json"),
            }
        except Exception as exc:
            v["read_error"] = repr(exc)
            variant_ok = False
        else:
            v["stages"] = {}
            for stage, path in paths.items():
                stage_doc: dict[str, Any] = {"checkpoint": str(path)}
                try:
                    digest = _sha256(path)
                    cfg = _checkpoint_cfg(path)
                    training = cfg.get("training") if isinstance(cfg.get("training"), Mapping) else {}
                    arch = architectures[stage]
                    completion = completions[stage]
                    expected_hash = three.get(f"{stage}_sha256")
                    if stage == "final":
                        expected_hash = three.get("final_sha256")
                    controller_variant = controller_variants.get(variant) if isinstance(controller_variants.get(variant), Mapping) else {}
                    exact_metadata_supported = (
                        arch.get("exact_deployment_eligibility_metric") is True
                        or (
                            "exact_deployment_eligibility_metric" not in arch
                            and arch.get("semantic_frontier_eligibility_metric") is True
                        )
                    )
                    stage_checks = {
                        "checkpoint_exists": path.is_file(),
                        "three_stage_hash_matches": digest == expected_hash,
                        "training_complete_hash_matches": digest == completion.get("checkpoint_sha256"),
                        "controller_completion_final_hash_matches": stage != "final" or digest == controller_variant.get("sha256"),
                        "checkpoint_exact_eligibility": training.get("direct_policy_metric_exact_eligibility") is True,
                        "exact_eligibility_metadata_supported": exact_metadata_supported,
                        "semantic_frontier_metadata": arch.get("semantic_frontier_eligibility_metric") is True,
                        "no_regime_routing": arch.get("regime_id_exposed_to_evidence_model") is False,
                        "physical_interaction_context": (
                            arch.get("context_source") == "physical_interaction"
                            and arch.get("observation_conditioned_action_frontier") is True
                        ),
                        "noncompensatory_frontier": arch.get("noncompensatory_frontier_cap") is True,
                        "shared_rule_required": arch.get("shared_deployment_rule_required") is True,
                        "test_roots_sealed": arch.get("test_roots_read") is False,
                    }
                    if repaired_stage_transfer_failure:
                        repair_variants = repair.get("variants") if isinstance(repair.get("variants"), Mapping) else {}
                        repair_variant = repair_variants.get(variant) if isinstance(repair_variants.get(variant), Mapping) else {}
                        repair_hash_key = f"{stage}_sha256"
                        stage_checks["stage_transfer_repair_hash_matches"] = digest == repair_variant.get(repair_hash_key)
                    stage_doc.update({"sha256": digest, "expected_sha256": expected_hash, "checks": stage_checks})
                    variant_ok &= all(stage_checks.values())
                except Exception as exc:
                    stage_doc["error"] = repr(exc)
                    variant_ok = False
                v["stages"][stage] = stage_doc

            support_ok = support.is_file() and _sha256(support) == three.get("factor_support_sha256")
            variant_contract = {
                "support_hash_matches": support_ok,
                "stage_transfer_valid": transfer.get("valid") is True,
                "unified_model_semantics": three.get("model_regime_routing") is False
                and three.get("shared_deployment_rule_required") is True
                and three.get("evidence_context_source") == "physical_interaction",
                "certificate_never_used_for_fit": three.get("test_roots_read") is False,
            }
            v["checks"] = variant_contract
            variant_ok &= all(variant_contract.values())
        v["valid"] = bool(variant_ok)
        variant_details[variant] = v
        record(f"{variant}_adaptation_reusable", variant_ok)

    valid = all(checks.values())
    doc = {
        "event": "v48_36_rc30_resume_authorization",
        "version": "v48.36-OCAF",
        "implementation_version": "v48.36.2-STAGE-TRANSFER-HOTFIX",
        "failure_mode": failure_mode,
        "run": str(root),
        "valid": valid,
        "authorized_action": "reuse_byte_identical_balanced_and_precision_adaptation_checkpoints_then_rerun_contracts_and_certificate" if valid else None,
        "retraining_authorized": False,
        "known_signature_only": True,
        "stage_transfer_repair": str(repair_path) if repaired_stage_transfer_failure else None,
        "checks": checks,
        "details": details,
        "variants": variant_details,
        "errors": errors,
        "test_roots_read": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(doc, ensure_ascii=False))
    return 0 if valid else 4


if __name__ == "__main__":
    raise SystemExit(main())
