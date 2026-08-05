from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(args, cwd=ROOT, env=env, text=True, capture_output=True, check=False)


def _load_training_contract_module():
    path = ROOT / "tools" / "check_v48_35_training_contract.py"
    spec = importlib.util.spec_from_file_location("v4835_training_contract", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_checkpoint(path: Path, *, exact: bool, tag: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "cfg": {
                "training": {
                    "direct_policy_metric_exact_eligibility": exact,
                    "direct_policy_metric_risk_source": "ordinal_evidence",
                    "direct_policy_metric_proposal_top_k": 5,
                    "direct_policy_metric_evidence_rerank_top_k": True,
                }
            },
            "tag": tag,
        },
        path,
    )


def _legacy_architecture() -> dict:
    return {
        "semantic_frontier_eligibility_metric": True,
        "regime_id_exposed_to_evidence_model": False,
        "context_source": "physical_relative",
        "noncompensatory_frontier_cap": True,
        "shared_deployment_rule_required": True,
        "test_roots_read": False,
    }


def _make_resume_fixture(root: Path, *, precision_exact: bool = True) -> None:
    _write_json(
        root / "PIPELINE_FAILED.json",
        {
            "event": "v48_35_pipeline_failed",
            "stage": "training_contract",
            "raw_exit_code": 4,
            "normalized_exit_code": 30,
            "adaptation_exit_codes": {"balanced": 0, "precision": 0},
            "certificate_executed": False,
            "gate_evaluated": False,
            "test_roots_read": False,
        },
    )
    completion_doc = {
        "event": "v48_35_continuous_frontier_controller_complete",
        "source_run": "runs/source",
        "protocol_root": "/data/protocol",
        "certificate_executed": False,
        "gate_evaluated": False,
        "test_roots_read": False,
        "variants": {},
    }
    for variant in ("balanced", "precision"):
        run = root / "candidates" / variant
        exact = precision_exact if variant == "precision" else True
        paths = {
            "factor": run / "factor_stage" / "model_v48_trac_sr" / "best.pt",
            "identity": run / "identity_stage" / "model_v48_trac_sr" / "best.pt",
            "final": run / "model_v48_trac_sr" / "best.pt",
        }
        for stage, path in paths.items():
            _write_checkpoint(path, exact=exact, tag=f"{variant}-{stage}")
            arch_path = run / "STAGE_ARCHITECTURE.json"
            complete_path = run / "TRAINING_COMPLETE.json"
            if stage == "factor":
                arch_path = run / "factor_stage" / "STAGE_ARCHITECTURE.json"
                complete_path = run / "factor_stage" / "TRAINING_COMPLETE.json"
            elif stage == "identity":
                arch_path = run / "identity_stage" / "STAGE_ARCHITECTURE.json"
                complete_path = run / "identity_stage" / "TRAINING_COMPLETE.json"
            _write_json(arch_path, _legacy_architecture())
            _write_json(complete_path, {"checkpoint_sha256": _sha(path)})
        support = run / "FACTOR_SUPPORT_CONTRACT.json"
        _write_json(support, {"reliability": [1, 1, 1, 1, 1]})
        _write_json(run / "STAGE_TRANSFER_INTEGRITY.json", {"valid": True})
        _write_json(
            run / "THREE_STAGE_TRAINING_COMPLETE.json",
            {
                "factor_sha256": _sha(paths["factor"]),
                "identity_sha256": _sha(paths["identity"]),
                "final_sha256": _sha(paths["final"]),
                "factor_support_sha256": _sha(support),
                "model_regime_routing": False,
                "shared_deployment_rule_required": True,
                "evidence_context_source": "physical_relative",
                "test_roots_read": False,
            },
        )
        completion_doc["variants"][variant] = {"sha256": _sha(paths["final"])}
    _write_json(root / "V48_35_COMPLETE.json", completion_doc)


def test_new_stage_metadata_names_exact_deployment_contract() -> None:
    text = (ROOT / "scripts" / "adapt_ocrap_v48_35_continuous_frontier_single_stage.sh").read_text(encoding="utf-8")
    assert '"exact_deployment_eligibility_metric": true' in text
    assert '"exact_deployment_eligibility_provenance": "checkpoint_cfg.training.direct_policy_metric_exact_eligibility"' in text
    assert "POLICY_METRIC_EXACT_ELIGIBILITY=true" in text


def test_legacy_metadata_is_repairable_only_with_checkpoint_proof(tmp_path: Path) -> None:
    module = _load_training_contract_module()
    paths = []
    for i in range(3):
        path = tmp_path / f"stage-{i}.pt"
        _write_checkpoint(path, exact=True, tag=str(i))
        paths.append(path)
    result = module._exact_eligibility_contract([_legacy_architecture()] * 3, paths)
    assert result["valid"] is True
    assert result["legacy_metadata_repair_used"] is True
    _write_checkpoint(paths[1], exact=False, tag="bad")
    rejected = module._exact_eligibility_contract([_legacy_architecture()] * 3, paths)
    assert rejected["valid"] is False
    assert rejected["checkpoint_exact_all_stages"] is False


def test_explicit_false_exact_metadata_is_not_legacy_repairable(tmp_path: Path) -> None:
    module = _load_training_contract_module()
    paths = []
    architectures = []
    for i in range(3):
        path = tmp_path / f"stage-{i}.pt"
        _write_checkpoint(path, exact=True, tag=str(i))
        paths.append(path)
        arch = _legacy_architecture()
        arch["exact_deployment_eligibility_metric"] = False
        architectures.append(arch)
    result = module._exact_eligibility_contract(architectures, paths)
    assert result["valid"] is False
    assert result["metadata_contradiction_present"] is True


def test_resume_contract_accepts_only_known_rc30_without_retraining(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _make_resume_fixture(run)
    out = run / "V48_35_RESUME_CONTRACT.json"
    proc = _run(
        sys.executable,
        "tools/check_v48_35_resume_contract.py",
        "--run", str(run),
        "--output", str(out),
        "--expect-source-run", "runs/source",
        "--expect-protocol-root", "/data/protocol",
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["valid"] is True
    assert doc["retraining_authorized"] is False
    assert doc["authorized_action"].startswith("reuse_byte_identical")


def test_resume_contract_rejects_checkpoint_without_exact_eligibility(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _make_resume_fixture(run, precision_exact=False)
    out = run / "V48_35_RESUME_CONTRACT.json"
    proc = _run(
        sys.executable,
        "tools/check_v48_35_resume_contract.py",
        "--run", str(run),
        "--output", str(out),
        "--expect-source-run", "runs/source",
        "--expect-protocol-root", "/data/protocol",
    )
    assert proc.returncode == 4
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["valid"] is False
    assert doc["checks"]["precision_adaptation_reusable"] is False


def test_resume_contract_rejects_controller_final_hash_mismatch(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _make_resume_fixture(run)
    complete_path = run / "V48_35_COMPLETE.json"
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    complete["variants"]["balanced"]["sha256"] = "0" * 64
    _write_json(complete_path, complete)
    out = run / "V48_35_RESUME_CONTRACT.json"
    proc = _run(
        sys.executable,
        "tools/check_v48_35_resume_contract.py",
        "--run", str(run),
        "--output", str(out),
        "--expect-source-run", "runs/source",
        "--expect-protocol-root", "/data/protocol",
    )
    assert proc.returncode == 4
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["checks"]["balanced_adaptation_reusable"] is False


def test_controller_checks_resume_before_cleanup_and_skips_adaptation() -> None:
    text = (ROOT / "scripts" / "run_v48_35_continuous_frontier_dedicated.sh").read_text(encoding="utf-8")
    check_pos = text.index("check_v48_35_resume_contract.py")
    cleanup_pos = text.index('rm -f "$OUTPUTDIR"/ADAPTATION_FAILED_*.json')
    assert check_pos < cleanup_pos
    assert 'RESUME_AFTER_ADAPTATION="${RESUME_AFTER_ADAPTATION:-0}"' in text
    assert "resume_after_adaptation=1 retraining=0" in text
    assert 'if [[ "$RESUME_AFTER_ADAPTATION" == 1 ]]; then\n  s0=0; s1=0' in text
    assert "resume_training_index_contract" in text
    assert "resume_validation_index_contract" in text
    assert "adaptation_reused_without_retraining" in text


def test_repair_wrapper_is_no_retraining_and_known_signature_only() -> None:
    text = (ROOT / "scripts" / "repair_v48_35_rc30_training_contract_with_v48_35_1.sh").read_text(encoding="utf-8")
    assert "RESUME_AFTER_ADAPTATION=1" in text
    assert "unset REBUILD_ADAPT_INDEX REBUILD_ADAPT_DEV_INDEX" in text
    assert "run_v48_35_continuous_frontier_dedicated.sh" in text
    assert "adapt_ocrap_v48_35" not in text
    help_proc = _run("bash", "scripts/repair_v48_35_rc30_training_contract_with_v48_35_1.sh", "--help")
    assert help_proc.returncode == 0
    assert "never retrains" in help_proc.stdout
