from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return env


def test_womd_at_suffix_resolves_complete_shard_set(tmp_path: Path) -> None:
    from ocrap.data.womd.sharded_path import resolve_womd_spec, sharded_filename

    prefix = str(tmp_path / "validation_tfexample.tfrecord")
    for i in range(3):
        Path(sharded_filename(prefix, i, 3)).write_bytes(b"")
    resolved = resolve_womd_spec(prefix + "@3")
    assert resolved.valid
    assert len(resolved.files) == 3
    assert resolved.declared_shard_count == 3
    assert not resolved.missing_files
    bare = resolve_womd_spec(prefix)
    assert not bare.valid
    assert "bare shard prefix" in " ".join(bare.errors)


def test_closed_loop_preflight_treats_at_as_shards_not_scenario_limit(tmp_path: Path) -> None:
    from ocrap.data.womd.sharded_path import sharded_filename

    dataset = tmp_path / "test_safe" / "samples"
    dataset.mkdir(parents=True)
    np.savez_compressed(
        dataset / "a.npz",
        split_id=np.asarray("test"),
        scene_id=np.asarray("waymax_deadbeef__wx00001234"),
        time_index=np.asarray(10, dtype=np.int64),
    )
    prefix = str(tmp_path / "validation" / "validation_tfexample.tfrecord")
    Path(prefix).parent.mkdir(parents=True)
    for i in range(3):
        Path(sharded_filename(prefix, i, 3)).write_bytes(b"")
    output = tmp_path / "support.json"
    proc = subprocess.run(
        [sys.executable, "tools/check_closed_loop_dataset_support.py", "--dataset", str(dataset.parent),
         "--split", "test", "--womd-pattern", prefix + "@3", "--output", str(output)],
        cwd=ROOT, env=_env(), text=True, capture_output=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    doc = json.loads(output.read_text())
    assert doc["schema_supports_closed_loop"] is True
    assert doc["num_resolved_womd_files"] == 3
    assert doc["raw_scenario_scan_limit"] is None
    assert doc["max_source_scenario_index"] == 1234


def test_target_keys_file_filters_before_per_scene_cap(tmp_path: Path) -> None:
    from ocrap.simulation.closed_loop_runner import _load_closed_loop_targets

    samples = tmp_path / "test_near_contact" / "samples"
    samples.mkdir(parents=True)
    # Same scene with two target times: the selected later target must not be
    # shadowed by max_targets_per_scene=1.
    for t in (5, 9):
        np.savez_compressed(
            samples / f"scene-a-{t}.npz",
            split_id=np.asarray("test"), scene_id=np.asarray("scene-a"),
            time_index=np.asarray(t, dtype=np.int64),
        )
    selected = tmp_path / "keys.json"
    selected.write_text(json.dumps({"target_keys": ["test_near_contact:scene-a:t9"]}))
    cfg = {"closed_loop": {"bucket_split": "test", "max_targets_per_scene": 1,
                            "target_keys_file": str(selected), "require_target_keys": True}}
    targets = _load_closed_loop_targets(str(samples.parent), cfg)
    assert [x["target_key"] for x in targets] == ["test_near_contact:scene-a:t9"]


def test_external_index_is_written_for_early_failure(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    for regime in ("safe", "near", "contact"):
        (root / f"{regime}.phase.json").write_text(json.dumps({"status": "failed", "exit_code": 3}))
    proc = subprocess.run(
        [sys.executable, "tools/build_external_baseline_run_index.py", "--root", str(root),
         "--launcher-exit-code", "3"], cwd=ROOT, env=_env(), text=True, capture_output=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    doc = json.loads((root / "EXTERNAL_BASELINE_RUN_INDEX.json").read_text())
    assert doc["complete"] is False
    assert set(doc["failed_or_incomplete_regimes"]) == {"safe", "near", "contact"}


def test_preflight_rejects_missing_selected_target_key(tmp_path: Path) -> None:
    from ocrap.data.womd.sharded_path import sharded_filename

    samples = tmp_path / "test_contact" / "samples"
    samples.mkdir(parents=True)
    np.savez_compressed(
        samples / "a.npz",
        split_id=np.asarray("test"),
        scene_id=np.asarray("scene-a"),
        time_index=np.asarray(7, dtype=np.int64),
    )
    keys = tmp_path / "keys.json"
    keys.write_text(json.dumps({"target_keys": ["test_contact:scene-missing:t7"]}))
    prefix = str(tmp_path / "validation_interactive_tfexample.tfrecord")
    for i in range(2):
        Path(sharded_filename(prefix, i, 2)).write_bytes(b"")
    output = tmp_path / "support.json"
    proc = subprocess.run(
        [sys.executable, "tools/check_closed_loop_dataset_support.py", "--dataset", str(samples.parent),
         "--split", "test", "--womd", prefix + "@2", "--target-keys-file", str(keys),
         "--require-target-keys", "--output", str(output)],
        cwd=ROOT, env=_env(), text=True, capture_output=True, check=False,
    )
    assert proc.returncode == 3
    doc = json.loads(output.read_text())
    assert doc["target_keys_valid"] is False
    assert doc["missing_requested_target_keys"] == ["test_contact:scene-missing:t7"]


def _contact_scene(key: str, *, intervention: float, terminal: float, auc: float,
                   clearance_gain: float, overlap_duration: float,
                   escape: float, recontact: float, new_stable: float,
                   offroad: float = 0.0) -> dict:
    return {
        "target_key": key,
        "scene_id": key.split(":")[1] if ":" in key else key,
        "target_time_index": 10,
        "intervention_rate": intervention,
        "closed_loop_bounded_NUP": 0.8,
        "metric_summary": {
            "post_contact_terminal_clearance_m": terminal,
            "post_contact_free_space_auc_normalized_m": auc,
            "post_contact_clearance_gain_m": clearance_gain,
            "post_contact_overlap_duration_s": overlap_duration,
            "post_contact_escape_event": escape,
            "recontact_event": recontact,
            "new_stable_stop_quality_event": new_stable,
            # The initiating contact is expected in this bucket and must not
            # disqualify an otherwise improved recovery rollout.
            "overlap_any": 1.0,
            "offroad_any": offroad,
            "ttc_recovery_gain_s": 0.0,
        },
    }


def test_contact_video_selection_uses_post_contact_recovery_not_initial_overlap(tmp_path: Path) -> None:
    control_path = tmp_path / "control.json.scenes.jsonl"
    method_path = tmp_path / "method.json.scenes.jsonl"
    key = "test_contact:scene-a:t10"
    control = _contact_scene(
        key, intervention=0.0, terminal=0.1, auc=0.2, clearance_gain=0.0,
        overlap_duration=0.8, escape=0.0, recontact=1.0, new_stable=0.0,
    )
    method = _contact_scene(
        key, intervention=0.4, terminal=1.0, auc=1.2, clearance_gain=0.6,
        overlap_duration=0.1, escape=1.0, recontact=0.0, new_stable=1.0,
    )
    control_path.write_text(json.dumps({"scene": control}) + "\n")
    method_path.write_text(json.dumps({"scene": method}) + "\n")
    output = tmp_path / "selection.json"
    proc = subprocess.run(
        [sys.executable, "tools/select_critical_scenes_v48_34.py",
         "--method-scenes", str(method_path), "--control-scenes", str(control_path),
         "--regime", "contact", "--num-positive", "1", "--num-failure", "0",
         "--require-exact-positive-count", "--output", str(output)],
        cwd=ROOT, env=_env(), text=True, capture_output=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    selected = json.loads(output.read_text())["selected"]
    assert len(selected) == 1  # --num-failure 0 must not append one failure case.
    assert selected[0]["target_key"] == key
    assert "overlap_duration_reduced" in selected[0]["material_improvements"]
    assert "new_stable_stop" in selected[0]["material_improvements"]


def test_contact_renderer_marks_causal_anchor_when_rollout_starts_separated() -> None:
    import importlib.util

    module_path = ROOT / "tools" / "render_critical_scenes_v48_34.py"
    spec = importlib.util.spec_from_file_location("render_critical_scenes_v48_34", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    trace = [{
        "metrics": {"overlap": 0.0},
        "agents": [{"is_sdc": True, "x": 3.0, "y": -2.0}],
    }]
    xy, label = module._contact_marker(trace, "contact")
    assert xy == (3.0, -2.0)
    assert label == "post-contact rollout start"



def test_preflight_accepts_runner_exact_bucket_target_key(tmp_path: Path) -> None:
    from ocrap.data.womd.sharded_path import sharded_filename

    samples = tmp_path / "test_near_contact" / "samples"
    samples.mkdir(parents=True)
    np.savez_compressed(
        samples / "a.npz",
        split_id=np.asarray("test"),
        scene_id=np.asarray("scene-a"),
        time_index=np.asarray(7, dtype=np.int64),
    )
    keys = tmp_path / "keys.json"
    keys.write_text(json.dumps({"target_keys": ["test_near_contact:scene-a:t7"]}))
    prefix = str(tmp_path / "validation_interactive_tfexample.tfrecord")
    for i in range(2):
        Path(sharded_filename(prefix, i, 2)).write_bytes(b"")
    output = tmp_path / "support.json"
    proc = subprocess.run(
        [sys.executable, "tools/check_closed_loop_dataset_support.py", "--dataset", str(samples.parent),
         "--split", "test", "--womd", prefix + "@2", "--target-keys-file", str(keys),
         "--require-target-keys", "--output", str(output)],
        cwd=ROOT, env=_env(), text=True, capture_output=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    doc = json.loads(output.read_text())
    assert doc["target_keys_valid"] is True
    assert doc["bucket_counts_after_split"] == {"test_near_contact": 1}
    assert doc["regime_counts_after_split"] == {"near_contact": 1}


def test_selected_target_resolver_migrates_legacy_hash_by_source_index(tmp_path: Path) -> None:
    samples = tmp_path / "test_contact" / "samples"
    samples.mkdir(parents=True)
    np.savez_compressed(
        samples / "a.npz",
        split_id=np.asarray("test"),
        scene_id=np.asarray("new_scene_hash__wx00002123"),
        source_scenario_index=np.asarray(2123, dtype=np.int64),
        time_index=np.asarray(11, dtype=np.int64),
    )
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({
        "regime": "contact",
        "selected": [{
            "target_key": "test_contact:waymax_old_hash:t11",
            "scene_id": "old_official_hash__wx00002123",
            "target_time_index": 11,
            "category": "positive_toy_example",
        }],
    }))
    keys = tmp_path / "resolved_keys.json"
    resolved = tmp_path / "resolved_selection.json"
    report = tmp_path / "resolution_report.json"
    proc = subprocess.run(
        [sys.executable, "tools/resolve_selected_targets_v50.py",
         "--dataset", str(samples.parent), "--split", "test",
         "--selection", str(selection), "--target-keys-output", str(keys),
         "--selection-output", str(resolved), "--report-output", str(report)],
        cwd=ROOT, env=_env(), text=True, capture_output=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    resolved_doc = json.loads(resolved.read_text())
    item = resolved_doc["selected"][0]
    assert item["target_key"] == "test_contact:new_scene_hash:t11"
    assert item["source_target_key"] == "test_contact:waymax_old_hash:t11"
    assert item["target_resolution_method"] == "source_scenario_index_and_time"
    assert json.loads(report.read_text())["valid"] is True

def test_render_context_captures_nearby_roadgraph_once() -> None:
    from types import SimpleNamespace
    from ocrap.simulation.closed_loop_runner import _capture_render_context

    roadgraph = SimpleNamespace(
        x=np.asarray([0.0, 1.0, 2.0, 100.0], dtype=np.float32),
        y=np.asarray([0.0, 0.0, 0.0, 100.0], dtype=np.float32),
        valid=np.asarray([True, True, True, True]),
        ids=np.asarray([7, 7, 7, 8], dtype=np.int64),
        types=np.asarray([1, 1, 1, 2], dtype=np.int64),
    )
    trajectory = SimpleNamespace(
        x=np.asarray([[0.0]], dtype=np.float32),
        y=np.asarray([[0.0]], dtype=np.float32),
    )
    state = SimpleNamespace(roadgraph_points=roadgraph, sim_trajectory=trajectory, timestep=np.asarray(0))
    context = _capture_render_context(state, 0, radius_m=20.0)
    assert len(context["roadgraph_polylines"]) == 1
    assert context["roadgraph_polylines"][0]["id"] == 7
    assert len(context["roadgraph_polylines"][0]["xy"]) == 3
