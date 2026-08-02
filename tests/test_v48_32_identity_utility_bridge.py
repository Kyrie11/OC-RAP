from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from ocrap.models.ocrap import OCRAPModel

ROOT = Path(__file__).resolve().parents[1]


def _model(*, detach: bool) -> OCRAPModel:
    return OCRAPModel(
        input_dim=12,
        num_roots=2,
        num_options=3,
        d_model=8,
        d_obs=4,
        encoder_type="mlp",
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        direct_recovery_value_head=True,
        direct_recovery_value_output="score",
        direct_recovery_relative_features_include_absolute=False,
        direct_recovery_set_tournament=True,
        direct_recovery_set_tournament_hidden=16,
        direct_recovery_set_tournament_heads=2,
        direct_recovery_set_tournament_dropout=0.0,
        direct_recovery_set_tournament_replace_base=True,
        direct_recovery_delta_head=True,
        direct_recovery_delta_regime_experts=True,
        direct_recovery_delta_policy_features=True,
        direct_recovery_delta_hidden=16,
        direct_recovery_delta_dropout=0.0,
        direct_recovery_delta_mode="ordinal_evidence",
        direct_recovery_evidence_calibrator=True,
        direct_recovery_evidence_calibrator_hidden=12,
        direct_recovery_evidence_calibrator_scale=0.75,
        direct_recovery_evidence_calibrator_mode="dual_tail_context",
        direct_recovery_evidence_calibrator_context=True,
        direct_recovery_evidence_calibrator_context_detach=True,
        direct_recovery_evidence_calibrator_context_source="tournament",
        direct_recovery_evidence_unified_experts=True,
        direct_recovery_evidence_component_heads=True,
        direct_recovery_evidence_component_count=5,
        direct_recovery_evidence_component_scale=6.0,
        direct_recovery_evidence_component_reliability="1,1,1,0,0",
        direct_recovery_evidence_concord=True,
        direct_recovery_evidence_admission_head=True,
        direct_recovery_evidence_admission_scale=2.0,
        direct_recovery_evidence_admission_bounded=True,
        direct_recovery_evidence_admission_prior_mode="safety_slack",
        direct_recovery_evidence_admission_prior_detach=detach,
        direct_recovery_evidence_slack_temperature=0.025,
        direct_recovery_evidence_slack_penalty=1.0,
        direct_recovery_evidence_frontier=True,
        direct_recovery_evidence_component_prior_logit=-2.0,
    ).eval()


def _save(path: Path, state: dict[str, torch.Tensor]) -> None:
    torch.save({"model_state": state}, path)


def test_safe_utility_prior_has_explicit_coupled_gradient_contract() -> None:
    assert _model(detach=False).direct_recovery_evidence_admission_prior_detach is False
    assert _model(detach=True).direct_recovery_evidence_admission_prior_detach is True
    source = (ROOT / "src" / "ocrap" / "models" / "ocrap.py").read_text()
    block = source[source.index("prior_benefit = (") : source.index("if self.direct_recovery_evidence_admission_prior_mode", source.index("prior_benefit = ("))]
    assert "if self.direct_recovery_evidence_admission_prior_detach" in block
    assert "else unified_benefit_logit" in block
    assert "else effective_component_harm_logits" in block
    assert "bucket_id" not in block and "regime_id" not in block


def test_teacher_gap_margin_is_continuous_and_regime_agnostic() -> None:
    source = (ROOT / "src" / "ocrap" / "models" / "losses.py").read_text()
    block = source[source.index("teacher_scale = max(") : source.index("# Directly train the high-benefit", source.index("teacher_scale = max("))]
    assert "teacher_gap" in block
    assert "required_margin = margin + teacher_scale * teacher_gap" in block
    assert "teacher_noop_depth" in block
    assert "regime" not in block.lower()


def test_stage_transfer_accepts_epoch_zero_final_and_added_allowed_heads(tmp_path: Path) -> None:
    factor = {
        "direct_evidence_concord_benefit_calibrator.0.weight": torch.zeros(1),
        "direct_evidence_concord_harm_calibrator.0.weight": torch.zeros(1),
        "encoder.weight": torch.ones(1),
    }
    identity = {
        **factor,
        "direct_evidence_concord_benefit_calibrator.0.weight": torch.ones(1),
        "direct_evidence_concord_admission_calibrator.0.weight": torch.ones(1),
    }
    final = {k: v.clone() for k, v in identity.items()}
    paths = [tmp_path / x for x in ("factor.pt", "identity.pt", "final.pt")]
    for path, state in zip(paths, (factor, identity, final)):
        _save(path, state)
    output = tmp_path / "transfer.json"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "check_v48_32_stage_transfer.py"),
         "--factor", str(paths[0]), "--identity", str(paths[1]), "--final", str(paths[2]),
         "--output", str(output)],
        check=False,
    )
    doc = json.loads(output.read_text())
    assert proc.returncode == 0
    assert doc["valid"] is True
    assert doc["identity_allowed_changed_parameter_count"] >= 2
    assert doc["final_selected_initial_checkpoint"] is True


def test_stage_transfer_rejects_frozen_parameter_change(tmp_path: Path) -> None:
    factor = {
        "direct_evidence_concord_benefit_calibrator.0.weight": torch.zeros(1),
        "encoder.weight": torch.ones(1),
    }
    identity = {
        "direct_evidence_concord_benefit_calibrator.0.weight": torch.ones(1),
        "encoder.weight": torch.zeros(1),
    }
    paths = [tmp_path / x for x in ("factor.pt", "identity.pt", "final.pt")]
    for path, state in zip(paths, (factor, identity, identity)):
        _save(path, state)
    output = tmp_path / "transfer.json"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "check_v48_32_stage_transfer.py"),
         "--factor", str(paths[0]), "--identity", str(paths[1]), "--final", str(paths[2]),
         "--output", str(output)],
        check=False,
    )
    assert proc.returncode == 31
    assert json.loads(output.read_text())["valid"] is False


def test_variant_copies_complete_metadata_and_trains_identity_heads() -> None:
    text = (ROOT / "scripts" / "adapt_ocrap_v48_32_identity_utility_variant.sh").read_text()
    assert "direct_evidence_concord_benefit_calibrator,direct_evidence_concord_harm_calibrator,direct_evidence_concord_admission_calibrator" in text
    assert "EVIDENCE_ADMISSION_PRIOR_DETACH=\"$prior_detach\"" in text
    assert "ORDINAL_EVIDENCE_SAFE_HARD_NEGATIVE_TEACHER_SCALE=\"$teacher_scale\"" in text
    assert "TRAINING_COMPLETE.json EVIDENCE_CORRECTION_COMPLETE.json" in text
    assert "GROUP_BATCH_STRATIFIED=false GROUP_BATCHING_REPLACEMENT=false" in text


def test_next_commands_state_is_explicit_and_stress_is_fail_closed() -> None:
    cal = (ROOT / "scripts" / "calibrate_v48_32_certificate_pool.sh").read_text()
    assert "NEXT_COMMANDS_BLOCKED.json" in cal
    assert "NEXT_COMMANDS_STATUS.json" in cal
    assert "certificate_artifact_or_protocol_failure" in cal
    assert "natural_gate_failed" in cal
    controller = (ROOT / "scripts" / "run_v48_32_identity_utility_bridge_dedicated.sh").read_text()
    assert "certificate/NEXT_COMMANDS contract mismatch" in controller
    assert "certificate_exit_code':30 if stage=='certificate' else None" in controller
    stress = (ROOT / "scripts" / "run_v48_32_stress_if_authorized.sh").read_text()
    assert "certificate_exit_code')==0" in stress
    assert "next_commands_generated" in stress


def test_ablation_reuses_factor_stage_and_records_every_failure() -> None:
    text = (ROOT / "scripts" / "run_v48_32_identity_utility_bridge_ablations.sh").read_text()
    for name in (
        "A_admission_only_detached_fixed_margin",
        "B_joint_identity_detached_fixed_margin",
        "C_joint_identity_coupled_fixed_margin",
        "D_full_identity_utility_bridge",
    ):
        assert name in text
    assert "V4832_FACTOR_CACHE_RUN" in text
    assert "V4832_ABLATION_FACTOR_CACHE_BALANCED" in text
    assert "V4832_ABLATION_FACTOR_CACHE_PRECISION" in text
    assert "factor_stage_cache" in text
    assert "write_task_failed" in text
    assert "max_concurrent_tasks':2" in text


def test_shadow_is_blocked_before_waymax_when_main_pipeline_is_invalid() -> None:
    text = (ROOT / "scripts" / "run_v48_32_dev_shadow_closed_loop.sh").read_text()
    assert "SHADOW_BLOCKED.json" in text
    assert "certificate_not_evaluated" in text
    assert "missing_gamma" in text
    assert text.index("SHADOW_BLOCKED.json") < text.index("run_ocrap_v48_trac_sr.sh")


def test_stage_transfer_accepts_epoch_zero_identity_and_final(tmp_path: Path) -> None:
    factor = {
        "direct_evidence_concord_benefit_calibrator.0.weight": torch.zeros(1),
        "encoder.weight": torch.ones(1),
    }
    identity = {k: v.clone() for k, v in factor.items()}
    final = {k: v.clone() for k, v in identity.items()}
    paths = [tmp_path / x for x in ("factor.pt", "identity.pt", "final.pt")]
    for path, state in zip(paths, (factor, identity, final)):
        _save(path, state)
    output = tmp_path / "transfer.json"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "check_v48_32_stage_transfer.py"),
         "--factor", str(paths[0]), "--identity", str(paths[1]), "--final", str(paths[2]),
         "--output", str(output)],
        check=False,
    )
    doc = json.loads(output.read_text())
    assert proc.returncode == 0
    assert doc["identity_selected_initial_checkpoint"] is True
    assert doc["no_op_identity_selection_is_valid"] is True
    assert doc["final_selected_initial_checkpoint"] is True


def test_factor_cache_contract_rejects_changed_training_input(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    group = tmp_path / "group.jsonl"
    val = tmp_path / "val.jsonl"
    support = tmp_path / "support.json"
    source.write_bytes(b"source")
    group.write_text("{}\n")
    val.write_text("{}\n")
    support.write_text(json.dumps({"index": "/run/a/index.jsonl", "reliability": [1, 1, 1, 0, 0]}))
    contract = tmp_path / "contract.json"
    common = [
        sys.executable, str(ROOT / "tools" / "manage_v48_32_factor_cache.py"),
        "--source-checkpoint", str(source), "--group-index", str(group),
        "--validation-group-index", str(val), "--support-contract", str(support),
        "--train-mix", "train", "--validation-mix", "val", "--variant", "balanced",
        "--setting", "epochs=20", "--contract", str(contract),
    ]
    assert subprocess.run(common + ["--mode", "create"], check=False).returncode == 0
    assert subprocess.run(common + ["--mode", "verify"], check=False).returncode == 0
    relocated_group = tmp_path / "other_group.jsonl"
    relocated_val = tmp_path / "other_val.jsonl"
    relocated_support = tmp_path / "other_support.json"
    relocated_group.write_bytes(group.read_bytes())
    relocated_val.write_bytes(val.read_bytes())
    relocated_support.write_text(json.dumps({"index": "/run/b/index.jsonl", "reliability": [1, 1, 1, 0, 0]}))
    relocated = common.copy()
    relocated[relocated.index(str(group))] = str(relocated_group)
    relocated[relocated.index(str(val))] = str(relocated_val)
    relocated[relocated.index(str(support))] = str(relocated_support)
    assert subprocess.run(relocated + ["--mode", "verify"], check=False).returncode == 0
    changed = common.copy()
    i = changed.index("epochs=20")
    changed[i] = "epochs=21"
    assert subprocess.run(changed + ["--mode", "verify"], check=False).returncode == 30


def test_single_stage_forwards_prior_gradient_flag_explicitly() -> None:
    text = (ROOT / "scripts" / "adapt_ocrap_v48_32_identity_utility_single_stage.sh").read_text()
    assert 'EVIDENCE_ADMISSION_PRIOR_DETACH="${EVIDENCE_ADMISSION_PRIOR_DETACH:-true}"' in text


def test_coupled_prior_routes_admission_gradient_into_identity_heads() -> None:
    totals: dict[bool, dict[str, float]] = {}
    for detach in (True, False):
        torch.manual_seed(1)
        model = _model(detach=detach).train()
        with torch.no_grad():
            # Activate a positive slack barrier so the harm path has a defined
            # deployment-utility gradient in the coupled model.
            model.direct_evidence_concord_harm_calibrator[3].bias.fill_(5.0)
        x = torch.randn(4, 12)
        group = torch.zeros(4, dtype=torch.long)
        nominal = torch.tensor([1, 0, 0, 0], dtype=torch.float32)
        out = model(x, group_index=group, is_nominal=nominal, direct_only=True)
        out["direct_recovery_admission_logit"][1:].sum().backward()
        totals[detach] = {}
        for prefix in (
            "direct_evidence_concord_benefit_calibrator",
            "direct_evidence_concord_harm_calibrator",
            "direct_evidence_concord_admission_calibrator",
        ):
            totals[detach][prefix] = sum(
                float(param.grad.abs().sum())
                for name, param in model.named_parameters()
                if name.startswith(prefix) and param.grad is not None
            )
    assert totals[True]["direct_evidence_concord_benefit_calibrator"] == 0.0
    assert totals[True]["direct_evidence_concord_harm_calibrator"] == 0.0
    assert totals[False]["direct_evidence_concord_benefit_calibrator"] > 0.0
    assert totals[False]["direct_evidence_concord_harm_calibrator"] > 0.0
    assert totals[False]["direct_evidence_concord_admission_calibrator"] > 0.0
