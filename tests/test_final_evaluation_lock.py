from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _closure(path: Path) -> None:
    path.write_text(json.dumps({
        "valid": True,
        "attribution_ready": True,
        "status": "TERMINAL_INTERNAL_CLOSURE_COMPLETE_MAIN_NOT_FROZEN",
        "scientific_closure": {
            "internal_mechanism_convergence": True,
            "recovery_mechanism_search": "FROZEN",
            "absolute_admission_repair_hypothesis": "CLOSED",
            "new_internal_algorithm_iteration_authorized": False,
        },
        "deployment_closure": {"deployed_main_freeze_authorized": False},
    }))


def _model_run(root: Path) -> Path:
    run = root / "model"
    for v in ("balanced", "precision"):
        p = run / "candidates" / v / "model_v48_trac_sr" / "best.pt"
        p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes((v + "-ckpt").encode())
        c = run / "candidates" / v / "calibration" / "gamma_rec_by_bucket_v48.json"
        c.parent.mkdir(parents=True, exist_ok=True); c.write_text('{"gamma_rec_by_bucket":{}}')
    return run


def test_final_lock_preserves_acceptance_stop_but_authorizes_characterization(tmp_path: Path) -> None:
    closure = tmp_path / "closure.json"; _closure(closure)
    run = _model_run(tmp_path)
    out = tmp_path / "lock.json"
    subprocess.run([
        sys.executable, str(ROOT / "tools/create_final_evaluation_lock.py"),
        "--terminal-closure", str(closure), "--model-run", str(run),
        "--repo", str(ROOT), "--output", str(out),
    ], check=True, cwd=ROOT)
    d = json.loads(out.read_text())
    assert d["valid"] is True
    fs = d["freeze_semantics"]
    assert fs["immutable_submission_evaluation_snapshot_authorized"] is True
    assert fs["final_three_regime_characterization_authorized"] is True
    assert fs["paired_external_baseline_characterization_authorized"] is True
    assert fs["deployed_main_acceptance_freeze_authorized"] is False
    assert fs["no_further_model_or_threshold_tuning_after_lock"] is True


def test_final_characterization_is_explicit_mode_and_freezes_paired_target_keys() -> None:
    launcher = (ROOT / "scripts/run_constraint_native_orientation_audit.sh").read_text()
    assert 'OCRAP_CONSTRAINT_AUDIT_MODE:-terminal_internal_closure' in launcher
    assert 'final_characterization' in launcher
    assert 'run_final_locked_three_regime_characterization.sh' in launcher
    text = (ROOT / "scripts/run_final_locked_three_regime_characterization.sh").read_text()
    assert 'MODEL_VARIANT="$variant"' in text
    assert 'WOMD_ROLE="$WOMD_ROLE"' in text
    assert 'export_paired_target_keys.py' in text
    assert 'MAX_STEPS="${MAX_STEPS:-40}"' in text


def test_final_external_wrapper_forces_exact_targets_and_isolated_latency() -> None:
    text = (ROOT / "scripts/run_final_external_baselines.sh").read_text()
    assert 'CL_TARGET_KEYS_FILE="$keyfile"' in text
    assert '--womd-role "$WOMD_ROLE"' in text
    assert 'RUN_SUPPLEMENTARY_SAFE=true' in text
    assert 'RUN_SUPPLEMENTARY_NEAR=true' in text
    assert 'profile_external_baselines_latency.sh' in text
    assert '--jobs-per-gpu "$JOBS_PER_GPU"' in text


def test_external_publication_replay_is_observation_legal() -> None:
    for rel in (
        "scripts/run_external_baselines_safe.sh",
        "scripts/run_external_baselines_near.sh",
        "scripts/run_external_baselines_contact.sh",
    ):
        text = (ROOT / rel).read_text()
        assert '--set closed_loop.use_sdc_paths=true' in text
        assert '--set closed_loop.require_observation_legal_route=true' in text
        assert '--set closed_loop.allow_future_route_proxy=false' in text
        assert '--set waymax.dataloader_include_sdc_paths=true' in text
        assert '--set waymax.allow_logged_sdc_route_fallback=false' in text


def test_submission_table_builder_includes_default_supplementary_methods() -> None:
    text = (ROOT / "tools/build_submission_external_baseline_tables.py").read_text()
    assert '"diffusion_planner"' in text
    assert '"flow_planner"' in text
    assert '"plan_r1"' in text
    assert '"betopnet"' in text


def test_final_characterization_builds_method_independent_route_locks_before_planners():
    text = (ROOT / "scripts/run_final_locked_three_regime_characterization.sh").read_text()
    assert "build_final_observation_legal_target_locks.sh" in text
    assert text.index("build_final_observation_legal_target_locks.sh") < text.index("run_variant balanced")
    assert "paired_target_keys" in text
    assert "check_target_key_lock.py" in text


def test_final_baselines_can_build_target_locks_without_ocrap_results():
    text = (ROOT / "scripts/run_final_external_baselines.sh").read_text()
    assert "ensure_target_locks" in text
    assert "build_final_observation_legal_target_locks.sh" in text
    assert "PRETRAINED_BASELINE_ROOT" in text


def test_final_ablation_launcher_uses_current_main_and_common_target_locks():
    text = (ROOT / "scripts/run_submission_ablations.sh").read_text()
    assert "submission_no_nominal_abstention.yaml" in text
    assert "no_rifa_absolute_admission" in text
    assert "FINAL_TARGET_LOCK_ROOT" in text
    assert '--target-keys-file "$keyfile"' in text


def test_one_command_ocrap_final_wrapper_preserves_legacy_output_roots_and_contracts() -> None:
    wrapper = (ROOT / "scripts/run_final_ocrap_evaluation.sh").read_text()
    assert 'BUILD_TARGET_LOCKS=true PROFILE_LATENCY=true' in wrapper
    assert 'OCRAP_FINAL_CHARACTERIZATION_OUT="${OCRAP_FINAL_CHARACTERIZATION_OUT:-$BASE_OUT/ocrap_v48_124_final_characterization}"' in wrapper
    assert 'OCRAP_LATENCY_OUT="${OCRAP_LATENCY_OUT:-$BASE_OUT/ocrap_v48_124_latency_isolated}"' in wrapper
    assert 'run_final_locked_three_regime_characterization.sh' in wrapper


def test_final_characterization_profiles_latency_to_legacy_standalone_root_by_default() -> None:
    text = (ROOT / "scripts/run_final_locked_three_regime_characterization.sh").read_text()
    assert 'BUILD_TARGET_LOCKS="${BUILD_TARGET_LOCKS:-true}"' in text
    assert 'OCRAP_LATENCY_OUT="${OCRAP_LATENCY_OUT:-$BASE_OUT/ocrap_v48_124_latency_isolated}"' in text
    assert 'OCRAP_LATENCY_OUT="$OCRAP_LATENCY_OUT"' in text
    assert "latency_path=latency_root/'LATENCY_INDEX.json'" in text
