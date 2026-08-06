from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
IDENTITY_PREFIXES = (
    "direct_evidence_concord_benefit_calibrator",
    "direct_evidence_concord_harm_calibrator",
    "direct_evidence_concord_admission_calibrator",
    "direct_evidence_interaction_bridge",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save(path: Path, state: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": state,
            "cfg": {
                "training": {
                    "direct_policy_metric_exact_eligibility": True,
                    "direct_policy_metric_risk_source": "proposal_deployed_rule",
                    "direct_policy_metric_proposal_top_k": 5,
                    "direct_policy_metric_evidence_rerank_top_k": 5,
                }
            },
        },
        path,
    )


def _architecture(trainable: tuple[str, ...] = IDENTITY_PREFIXES) -> dict:
    return {
        "version": "v48.36-OCAF",
        "trainable": [",".join(trainable)],
        "context_source": "physical_interaction",
        "observation_conditioned_action_frontier": True,
        "semantic_frontier_eligibility_metric": True,
        "exact_deployment_eligibility_metric": True,
        "noncompensatory_frontier_cap": True,
        "regime_id_exposed_to_evidence_model": False,
        "shared_deployment_rule_required": True,
        "test_roots_read": False,
        "admission_prior_detach": False,
        "safe_hard_negative_teacher_scale": 0.0,
        "consensus_prior_scale": 0.5,
        "interaction_hidden": 64,
        "interaction_dropout": 0.05,
        "admission_prior_mode": "frontier_capped_slack",
    }


def _states() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    factor = {
        "direct_evidence_concord_benefit_calibrator.0.weight": torch.zeros(2),
        "direct_evidence_concord_harm_calibrator.0.weight": torch.zeros(2),
        "direct_evidence_concord_admission_calibrator.0.weight": torch.zeros(2),
        "direct_evidence_interaction_bridge.action_raw.weight": torch.zeros(2),
        "encoder.weight": torch.ones(2),
    }
    identity = {key: value.clone() for key, value in factor.items()}
    identity["direct_evidence_concord_benefit_calibrator.0.weight"] += 1
    identity["direct_evidence_concord_harm_calibrator.0.weight"] += 2
    identity["direct_evidence_concord_admission_calibrator.0.weight"] += 3
    identity["direct_evidence_interaction_bridge.action_raw.weight"] += 4
    return factor, identity


def _run_checker(
    tmp_path: Path,
    *,
    identity_allowed: tuple[str, ...] = IDENTITY_PREFIXES,
    identity_arch_trainable: tuple[str, ...] = IDENTITY_PREFIXES,
    mutate_encoder: bool = False,
) -> tuple[subprocess.CompletedProcess, dict]:
    factor, identity = _states()
    if mutate_encoder:
        identity["encoder.weight"] = torch.zeros(2)
    factor_path = tmp_path / "factor.pt"
    identity_path = tmp_path / "identity.pt"
    final_path = tmp_path / "final.pt"
    _save(factor_path, factor)
    _save(identity_path, identity)
    _save(final_path, {key: value.clone() for key, value in identity.items()})
    identity_arch = tmp_path / "identity_arch.json"
    final_arch = tmp_path / "final_arch.json"
    identity_arch.write_text(json.dumps(_architecture(identity_arch_trainable)))
    final_arch.write_text(json.dumps(_architecture(identity_arch_trainable)))
    output = tmp_path / "transfer.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "check_v48_36_stage_transfer.py"),
            "--factor",
            str(factor_path),
            "--identity",
            str(identity_path),
            "--final",
            str(final_path),
            "--identity-architecture",
            str(identity_arch),
            "--final-architecture",
            str(final_arch),
            "--identity-allowed-prefixes",
            ",".join(identity_allowed),
            "--final-stage-disabled",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=False,
    )
    return completed, json.loads(output.read_text())


def test_new_stage_transfer_accepts_registered_ocaf_bridge(tmp_path: Path) -> None:
    completed, doc = _run_checker(tmp_path)
    assert completed.returncode == 0
    assert doc["valid"] is True
    assert doc["identity_disallowed_changed_parameter_count"] == 0
    assert doc["identity_allowed_changed_parameter_count"] == 4
    assert "direct_evidence_interaction_bridge" in doc["identity_allowed_prefixes"]
    assert doc["final_selected_initial_checkpoint"] is True


def test_legacy_v48_32_checker_reproduces_uploaded_false_failure(tmp_path: Path) -> None:
    factor, identity = _states()
    factor_path, identity_path, final_path = [tmp_path / name for name in ("factor.pt", "identity.pt", "final.pt")]
    _save(factor_path, factor)
    _save(identity_path, identity)
    _save(final_path, identity)
    output = tmp_path / "legacy.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "check_v48_32_stage_transfer.py"),
            "--factor",
            str(factor_path),
            "--identity",
            str(identity_path),
            "--final",
            str(final_path),
            "--final-stage-disabled",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=False,
    )
    doc = json.loads(output.read_text())
    assert completed.returncode == 31
    assert doc["valid"] is False
    assert "direct_evidence_interaction_bridge" in doc["failure_reasons"][0]


def test_stage_transfer_still_rejects_true_frozen_drift(tmp_path: Path) -> None:
    completed, doc = _run_checker(tmp_path, mutate_encoder=True)
    assert completed.returncode == 31
    assert doc["valid"] is False
    assert doc["identity_disallowed_changed_parameter_count"] == 1
    assert doc["identity_diff"]["disallowed_changed"][0]["name"] == "encoder.weight"


def test_stage_transfer_rejects_architecture_contract_mismatch(tmp_path: Path) -> None:
    completed, doc = _run_checker(
        tmp_path,
        identity_allowed=IDENTITY_PREFIXES,
        identity_arch_trainable=IDENTITY_PREFIXES[:-1],
    )
    assert completed.returncode == 31
    assert doc["valid"] is False
    assert any("architecture/trainable-prefix mismatch" in reason for reason in doc["failure_reasons"])


def _write_completion(path: Path, checkpoint: Path, best_epoch: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "event": "variant_training_complete",
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha(checkpoint),
                "best_epoch": best_epoch,
                "epochs_completed": best_epoch + 1,
                "best_metric": "direct_contract_lexicographic",
            }
        )
    )


def test_exact_uploaded_failure_can_be_repaired_without_retraining(tmp_path: Path) -> None:
    run = tmp_path / "run"
    source_run = tmp_path / "source"
    protocol = tmp_path / "protocol"
    protocol.mkdir()
    attempt = "attempt-1"
    (run).mkdir()
    (run / "ATTEMPT_STARTED.json").write_text(
        json.dumps(
            {
                "event": "v48_36_attempt_started",
                "attempt_id": attempt,
                "protocol_root": str(protocol),
                "test_roots_read": False,
            }
        )
    )
    (run / "PIPELINE_FAILED.json").write_text(
        json.dumps(
            {
                "event": "v48_36_pipeline_failed",
                "stage": "adaptation",
                "raw_exit_code": 30,
                "normalized_exit_code": 30,
                "adaptation_exit_codes": {"balanced": 31, "precision": 31},
                "certificate_executed": False,
                "gate_evaluated": False,
                "test_roots_read": False,
            }
        )
    )
    controller_variants = {}
    factor_state, identity_state = _states()
    for variant in ("balanced", "precision"):
        candidate = run / "candidates" / variant
        factor = candidate / "factor_stage" / "model_v48_trac_sr" / "best.pt"
        identity = candidate / "identity_stage" / "model_v48_trac_sr" / "best.pt"
        final = candidate / "model_v48_trac_sr" / "best.pt"
        source = source_run / "candidates" / variant / "model_v48_trac_sr" / "best.pt"
        _save(factor, factor_state)
        _save(identity, identity_state)
        _save(final, identity_state)
        _save(source, factor_state)
        arch = _architecture()
        for path in (
            candidate / "factor_stage" / "STAGE_ARCHITECTURE.json",
            candidate / "identity_stage" / "STAGE_ARCHITECTURE.json",
            candidate / "STAGE_ARCHITECTURE.json",
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(arch))
        _write_completion(candidate / "factor_stage" / "TRAINING_COMPLETE.json", factor, 2)
        _write_completion(candidate / "identity_stage" / "TRAINING_COMPLETE.json", identity, 3)
        _write_completion(candidate / "TRAINING_COMPLETE.json", final, 3)
        (candidate / "FACTOR_SUPPORT_CONTRACT.json").write_text(
            json.dumps({"reliability": [1, 1, 1, 0, 0], "independent_measured_hard_veto_preserved": True})
        )
        (candidate / "VARIANT_STAGE_FAILED.json").write_text(
            json.dumps({"stage": "stage_transfer_integrity", "exit_code": 31})
        )
        (candidate / "STAGE_TRANSFER_INTEGRITY.json").write_text(
            json.dumps(
                {
                    "version": "v48.32-IDENTITY-UTILITY-BRIDGE",
                    "valid": False,
                    "identity_disallowed_changed_parameter_count": 1,
                    "final_disallowed_changed_parameter_count": 0,
                    "final_stage_disabled": True,
                    "failure_reasons": [
                        "identity stage changed frozen parameters: direct_evidence_interaction_bridge.action_raw.weight=4"
                    ],
                }
            )
        )
        controller_variants[variant] = {"checkpoint": str(final), "sha256": _sha(final)}
    (run / "V48_36_COMPLETE.json").write_text(
        json.dumps(
            {
                "event": "v48_36_ocaf_controller_complete",
                "source_run": str(source_run),
                "protocol_root": str(protocol),
                "variants": controller_variants,
                "certificate_executed": False,
                "gate_evaluated": False,
                "test_roots_read": False,
            }
        )
    )

    output = run / "V48_36_2_STAGE_TRANSFER_REPAIR.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "repair_v48_36_1_stage_transfer_failure.py"),
            "--run",
            str(run),
            "--source-run",
            str(source_run),
            "--protocol-root",
            str(protocol),
            "--repo",
            str(ROOT),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=False,
    )
    doc = json.loads(output.read_text())
    assert completed.returncode == 0
    assert doc["valid"] is True
    assert doc["algorithm_changed"] is False
    assert doc["retraining_performed"] is False
    for variant in ("balanced", "precision"):
        candidate = run / "candidates" / variant
        assert json.loads((candidate / "STAGE_TRANSFER_INTEGRITY.json").read_text())["valid"] is True
        assert (candidate / "THREE_STAGE_TRAINING_COMPLETE.json").is_file()
        assert not (candidate / "VARIANT_STAGE_FAILED.json").exists()

    resume_output = run / "V48_36_RESUME_CONTRACT.json"
    resume = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "check_v48_36_resume_contract.py"),
            "--run",
            str(run),
            "--output",
            str(resume_output),
            "--expect-source-run",
            str(source_run),
            "--expect-protocol-root",
            str(protocol),
        ],
        cwd=ROOT,
        check=False,
    )
    resume_doc = json.loads(resume_output.read_text())
    assert resume.returncode == 0, resume_doc
    assert resume_doc["valid"] is True
    assert resume_doc["failure_mode"] == "repaired_stage_transfer"


def test_controller_wires_versioned_transfer_failure_and_resume_tools() -> None:
    variant = (ROOT / "scripts" / "adapt_ocrap_v48_36_ocaf_variant.sh").read_text()
    controller = (ROOT / "scripts" / "run_v48_36_ocaf_dedicated.sh").read_text()
    repair = (ROOT / "scripts" / "repair_v48_36_1_stage_transfer_with_v48_36_2.sh").read_text()
    assert "check_v48_36_stage_transfer.py" in variant
    assert "check_v48_32_stage_transfer.py" not in variant
    assert "finalize_v48_36_adaptation_variant.py" in variant
    assert "extract_v48_36_failure_signature.py" in controller
    assert "RESUME_AFTER_ADAPTATION=1" in repair
    assert "repair_v48_36_1_stage_transfer_failure.py" in repair
    assert "REPAIR_ONLY" in repair
