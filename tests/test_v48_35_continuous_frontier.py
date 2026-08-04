from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import torch

from ocrap.models.ocrap import OCRAPModel

ROOT = Path(__file__).resolve().parents[1]


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        args, cwd=ROOT, env=env, text=True, capture_output=True, check=False,
    )


def _physical_model() -> OCRAPModel:
    return OCRAPModel(
        input_dim=512,
        num_roots=2,
        num_options=3,
        d_model=8,
        d_obs=4,
        encoder_type="structured_transformer",
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        direct_recovery_value_head=True,
        direct_recovery_value_pooling="candidate_concat_raw",
        direct_recovery_delta_head=True,
        direct_recovery_delta_regime_experts=True,
        direct_recovery_evidence_calibrator=True,
        direct_recovery_evidence_calibrator_context=True,
        direct_recovery_evidence_calibrator_context_source="physical_relative",
        direct_recovery_evidence_unified_experts=True,
        direct_recovery_evidence_component_heads=True,
        direct_recovery_evidence_component_count=5,
        direct_recovery_evidence_concord=True,
        direct_recovery_evidence_admission_head=True,
        direct_recovery_evidence_admission_bounded=False,
        direct_recovery_evidence_admission_prior_mode="frontier_capped_slack",
    ).eval()


def test_noncompensatory_frontier_cannot_be_overridden_by_large_benefit() -> None:
    free = torch.tensor([-3.0, 0.0, 10.0, 1_000.0], requires_grad=True)
    safety_cap = torch.tensor([-2.0, -1.0, -0.5, -0.25], requires_grad=True)
    admitted = OCRAPModel._noncompensatory_smooth_cap(free, safety_cap, 0.1)
    assert torch.all(admitted <= free + 1.0e-7)
    assert torch.all(admitted <= safety_cap + 1.0e-7)
    admitted.sum().backward()
    assert torch.isfinite(free.grad).all()
    assert torch.isfinite(safety_cap.grad).all()


def test_physical_relative_context_excludes_ego_scalar_and_scene_shortcuts() -> None:
    model = _physical_model()
    assert model.direct_candidate_feature_dim == 156
    assert model.direct_candidate_physical_feature_dim == 141

    x = torch.zeros((3, 512))
    # Candidate 1 differs from nominal only in executable prefix geometry/control.
    for start, end in model.direct_candidate_physical_slices:
        x[1, start:end] = 2.0
    # Candidate 2 changes excluded ego, scalar/audit labels, and shared scene suffix.
    x[2, : model.direct_ego_feature_dim] = 17.0
    scalar_start = model.direct_candidate_physical_slices[0][1]
    scalar_end = model.direct_candidate_physical_slices[1][0]
    x[2, scalar_start:scalar_end] = -19.0
    x[2, model.direct_candidate_feature_dim :] = 23.0

    group = torch.tensor([[7, 11], [7, 11], [7, 11]])
    nominal = torch.tensor([1.0, 0.0, 0.0])
    context = model._direct_candidate_raw_relative_features(x, group, nominal)
    assert torch.allclose(context[0], torch.zeros_like(context[0]))
    assert torch.allclose(context[1], torch.full_like(context[1], 2.0))
    assert torch.allclose(context[2], torch.zeros_like(context[2]))


def _write_rows(path: Path, stratum: str, start_scene: int) -> None:
    rows = []
    for i in range(4):
        common = {
            "scene": f"{stratum}-{start_scene + i}",
            "time": 10 + i,
            "fold": i % 2,
            "macro": 2,
            "has_safe_opportunity": True,
        }
        rows.append(common | {
            "candidate": 1,
            "proposal_rank": 1,
            "opportunity": 0.9,
            "harm": 0.05,
            "pred_adv": 0.8,
            "teacher_adv": 0.2,
            "teacher_harmful": False,
        })
        rows.append(common | {
            "candidate": 2,
            "proposal_rank": 2,
            "opportunity": 0.8,
            "harm": 0.9,
            "pred_adv": 0.6,
            "teacher_adv": -0.2,
            "teacher_harmful": True,
        })
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_shared_rule_fitter_emits_one_rule_for_all_audit_strata(tmp_path: Path) -> None:
    near = tmp_path / "near.jsonl"
    contact = tmp_path / "contact.jsonl"
    out = tmp_path / "shared.json"
    _write_rows(near, "near", 0)
    _write_rows(contact, "contact", 100)
    proc = _run(
        sys.executable,
        "tools/calibrate_shared_continuous_rule_v48_35.py",
        "--stratum", f"near={near}",
        "--stratum", f"contact={contact}",
        "--output", str(out),
        "--grid-size", "3",
        "--min-selected", "near=2,contact=2",
        "--min-precision-lcb", "near=0,contact=0",
        "--max-harmful-group-ucb", "near=1,contact=1",
        "--max-harmful-selected-ucb", "near=1,contact=1",
        "--max-macro-share", "1",
    )
    assert proc.returncode == 0, proc.stderr
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["valid"] is True
    assert doc["shared_rule_count"] == 1
    assert doc["strategy_regime_conditioning"] is False
    assert sorted(doc["audit_strata_only"]) == ["contact", "near"]
    assert set(doc["rule"]) == {
        "opportunity_threshold", "harm_threshold", "score_threshold", "rank_margin_threshold",
    }
    assert doc["fit"]["by_stratum"]["near"]["num_selected"] >= 2
    assert doc["fit"]["by_stratum"]["contact"]["num_selected"] >= 2
    assert doc["rule"]["opportunity_threshold"] >= 0.5
    assert doc["rule"]["harm_threshold"] <= 0.5
    assert doc["rule"]["score_threshold"] >= 0.0


def test_calibration_diagnostics_follow_the_deployed_rule_not_fixed_thresholds() -> None:
    text = (ROOT / "tools/calibrate_policy_risk_v48.py").read_text(encoding="utf-8")
    block = text[text.index("proposal_deployed_rule_top1"):text.index("def _proposal_oracle_partition")]
    assert "deployed_rule_selected" in block
    assert 'rule["opportunity_threshold"]' in block
    assert 'rule["harm_threshold"]' in block
    assert "diagnostic_opportunity_threshold" not in block
    assert "diagnostic_harm_threshold" not in block
    assert '"proposal_exact_eligible_semantics": "deprecated_alias_of_deployed_rule"' in text


def test_pipeline_uses_byte_identical_shared_rule_and_preserves_algorithm_rc3() -> None:
    calibration = (ROOT / "scripts/calibrate_v48_35_shared_certificate_pool.sh").read_text(encoding="utf-8")
    assert calibration.count('dev_frozen_shared_rule_v48.json') >= 6
    assert "dev_frozen_near" not in calibration
    assert "dev_frozen_contact" not in calibration
    assert '( "$dn" != 0 && "$dn" != 3 )' in calibration
    assert '( "$dc" != 0 && "$dc" != 3 )' in calibration
    assert "byte-identical shared rule" in calibration
    fitter = (ROOT / "tools/calibrate_shared_continuous_rule_v48_35.py").read_text(encoding="utf-8")
    assert "min-opportunity-threshold" in fitter
    assert "max-harm-threshold" in fitter
    assert "min-score-threshold" in fitter
    staged = (ROOT / "scripts/adapt_ocrap_v48_35_continuous_frontier_variant.sh").read_text(encoding="utf-8")
    assert staged.count("EVIDENCE_ADMISSION_PRIOR_MODE=frontier_capped_slack") >= 2
    assert "EVIDENCE_ADMISSION_PRIOR_MODE=safety_slack" not in staged
    assert "train_metric_uses_final_fitted_thresholds':False" in staged
    assert "exact_eligibility_metric':True" not in staged


def test_pytest_import_contract_is_repository_local() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "[tool.pytest.ini_options]" in text
    assert 'pythonpath = ["src"]' in text


def test_v48_35_generated_command_dependencies_exist() -> None:
    import re

    paths = [
        ROOT / "scripts/calibrate_v48_35_shared_certificate_pool.sh",
        ROOT / "scripts/run_v48_35_continuous_frontier_dedicated.sh",
        ROOT / "scripts/adapt_ocrap_v48_35_continuous_frontier_variant.sh",
        ROOT / "scripts/adapt_ocrap_v48_35_continuous_frontier_single_stage.sh",
        ROOT / "scripts/run_v48_35_safe_noninferiority.sh",
        ROOT / "scripts/run_v48_35_stress_if_authorized.sh",
        ROOT / "scripts/run_v48_35_continuous_frontier_ablations.sh",
    ]
    missing: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for rel in re.findall(r"(?:bash|python)\s+((?:scripts|tools)/[A-Za-z0-9_.-]+)", text):
            if not (ROOT / rel).is_file():
                missing.append(f"{path.name}: {rel}")
    assert not missing, "missing command dependencies: " + ", ".join(missing)
    assert (ROOT / "tools/materialize_v48_35_factor_cache.py").is_file()


def test_post_gate_wrappers_fail_closed_on_v48_35_authorization() -> None:
    safe = (ROOT / "scripts/run_v48_35_safe_noninferiority.sh").read_text(encoding="utf-8")
    stress = (ROOT / "scripts/run_v48_35_stress_if_authorized.sh").read_text(encoding="utf-8")
    for text in (safe, stress):
        assert "V48_35_COMPLETE.json" in text
        assert "certificate_exit_code') == 0" in text
        assert "next_commands_generated" in text
    assert "V48_34_COMPLETE.json" not in safe + stress
    assert "shared frozen rule" in stress


def test_ablation_is_a_shared_rule_2x2_not_regime_specific_policy_forks() -> None:
    text = (ROOT / "scripts/run_v48_35_continuous_frontier_ablations.sh").read_text(encoding="utf-8")
    assert "A_legacy_context_soft_slack" in text
    assert "B_physical_context_soft_slack" in text
    assert "C_legacy_context_frontier_cap" in text
    assert "D_physical_context_frontier_cap_main" in text
    assert "calibrate_v48_35_shared_certificate_pool.sh" not in text  # controller owns the single-rule gate
    controller = (ROOT / "scripts/run_v48_35_continuous_frontier_dedicated.sh").read_text(encoding="utf-8")
    assert 'EVIDENCE_CONTEXT_SOURCE="${EVIDENCE_CALIBRATOR_CONTEXT_SOURCE:-physical_relative}"' in controller
    assert 'ADMISSION_PRIOR_MODE="${EVIDENCE_ADMISSION_PRIOR_MODE:-frontier_capped_slack}"' in controller
