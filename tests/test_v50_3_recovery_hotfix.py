from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _scene(i: int, bucket: str = "test_safe") -> dict:
    return {
        "scene_id": f"scene-{i}",
        "bucket_name": bucket,
        "canonical_regime": "safe",
        "target_key": f"{bucket}:scene-{i}:t10",
        "target_time_index": 10,
        "num_decisions": 2,
        "num_metric_steps": 2,
        "method": "ocrap",
        "gamma_rec": 1.0,
        "label_mode": "fast",
        "intervention_rate": 0.5,
        "intervention_episode_count": 1,
        "metric_summary": {
            "overlap_any": 0.0,
            "overlap_mean": 0.0,
            "offroad_any": 0.0,
            "offroad_mean": 0.0,
            "min_clearance_m_min": 1.0,
            "ttc_s_min": 2.0,
        },
        "macro_counts": {"nominal": 1},
        "selection_reason_counts": {},
        "timing": {"wall_s": 1.0, "totals_s": {"policy": 0.5}},
        "decisions": [{"payload": "x" * 1000}],
        "render_trace": [{"frame": 1}],
        "state_xy_trace": [[0.0, 0.0]],
    }


def _write_complete_artifact(path: Path, count: int = 2) -> None:
    journal = path.with_suffix(path.suffix + ".scenes.jsonl")
    progress = path.with_suffix(path.suffix + ".progress.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with journal.open("w", encoding="utf-8") as f:
        for i in range(count):
            f.write(json.dumps({"version": 1, "run_fingerprint": "fp", "scene": _scene(i)}) + "\n")
    progress.write_text(json.dumps({"status": "complete", "requested_rollouts": count, "completed_rollouts": count, "run_fingerprint": "fp"}))
    path.write_text(json.dumps({
        "method": "ocrap", "source": "model", "num_scenes": count,
        "bucket_target_count": count, "num_decisions": 2 * count,
        "run_fingerprint": "fp",
    }))


def test_metric_scene_storage_removes_step_payloads() -> None:
    from ocrap.simulation.closed_loop_runner import _scene_storage_view

    compact = _scene_storage_view(_scene(0), "metrics")
    assert "decisions" not in compact
    assert "render_trace" not in compact
    assert "state_xy_trace" not in compact
    assert compact["metric_summary"]["ttc_s_min"] == 2.0
    assert compact["target_key"].startswith("test_safe:")


def test_journal_finalizer_reconstructs_compact_result(tmp_path: Path) -> None:
    output = tmp_path / "closed_loop_ocrap.json"
    _write_complete_artifact(output, 3)
    output.unlink()
    output.with_suffix(output.suffix + ".progress.json").write_text(json.dumps({
        "status": "running_scene", "requested_rollouts": 3,
        "completed_rollouts": 3, "run_fingerprint": "fp",
    }))
    proc = subprocess.run(
        [sys.executable, "tools/finalize_closed_loop_from_journal.py", "--output", str(output)],
        cwd=ROOT, env=_env(), text=True, capture_output=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    result = json.loads(output.read_text())
    progress = json.loads(output.with_suffix(output.suffix + ".progress.json").read_text())
    assert result["num_scenes"] == 3
    assert result["bucket_target_count"] == 3
    assert result["scenes_embedded"] is False
    assert "scenes" not in result
    assert progress["status"] == "complete"


def test_journal_finalizer_refuses_unknown_expected_count(tmp_path: Path) -> None:
    output = tmp_path / "closed_loop_ocrap.json"
    journal = output.with_suffix(output.suffix + ".scenes.jsonl")
    journal.write_text(json.dumps({"version": 1, "run_fingerprint": "fp", "scene": _scene(0)}) + "\n")
    proc = subprocess.run(
        [sys.executable, "tools/finalize_closed_loop_from_journal.py", "--output", str(output)],
        cwd=ROOT, env=_env(), text=True, capture_output=True, check=False,
    )
    assert proc.returncode == 5
    assert "unknown_expected_count" in proc.stdout


def test_external_index_uses_complete_artifacts_over_stale_failed_phase(tmp_path: Path) -> None:
    root = tmp_path / "external"
    expected = {
        "safe": ["nominal_replay", "wayformer_bc", "gameformer_lite", "betopnet_lite"],
        "near": ["gameformer_lite", "marc_lite", "racp_lite", "predictive_safety_filter", "dro_cvar_filter", "cvar_risk_filter", "expected_risk_filter"],
        "contact": ["postimpact_mpc_lite", "post_crash_braking", "post_collision_restoration", "severity_minimization"],
    }
    for regime, methods in expected.items():
        (root / f"{regime}.phase.json").parent.mkdir(parents=True, exist_ok=True)
        (root / f"{regime}.phase.json").write_text(json.dumps({"status": "failed", "exit_code": 1}))
        for method in methods:
            _write_complete_artifact(root / regime / f"closed_loop_{method}.json", 1)
    proc = subprocess.run(
        [sys.executable, "tools/build_external_baseline_run_index.py", "--root", str(root), "--launcher-exit-code", "0"],
        cwd=ROOT, env=_env(), text=True, capture_output=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    doc = json.loads((root / "EXTERNAL_BASELINE_RUN_INDEX.json").read_text())
    assert doc["complete"] is True
    assert all(v["phase_effective_status"] == "complete_from_artifacts" for v in doc["regimes"].values())


def test_launcher_contains_portable_scheduler_alias_and_skip_guards() -> None:
    safe = (ROOT / "scripts/run_safe_regime_external_baselines.sh").read_text()
    near = (ROOT / "scripts/run_near_contact_external_baselines_2gpu_optimized.sh").read_text()
    contact = (ROOT / "scripts/run_contact_external_baselines.sh").read_text()
    assert "runtime_method=nominal" in safe
    assert "help wait" in near and "USE_DYNAMIC_SCHEDULER" in near
    assert "SKIP_COMPLETE_METHODS" in safe
    assert "SKIP_COMPLETE_METHODS" in near
    assert "SKIP_COMPLETE_METHODS" in contact
