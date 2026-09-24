from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ocrap.models.data import sample_to_feature, samples_to_feature_matrix
from ocrap.simulation import closed_loop_runner as clr


def _feature_sample(candidate_index: int) -> dict:
    rng = np.random.default_rng(123)
    sample = {
        "time_index": np.int64(17),
        "candidate_index": np.int64(candidate_index),
        "is_nominal": np.int64(candidate_index == 0),
        "ego_state": rng.normal(size=(9,)).astype(np.float32),
        "prefix_param": (rng.normal(size=(5,)) + candidate_index).astype(np.float32),
        "prefix_macro_id": np.int64(candidate_index),
        "prefix_macro_type_id": np.int64(candidate_index % 4),
        "utility": np.float32(2.5 - 0.1 * candidate_index),
        "hard_violation": np.float32(0.2 * candidate_index),
        "harm_proxy": np.float32(0.05 * candidate_index),
        "feasible": np.int64(1),
        "prefix_states": (rng.normal(size=(10, 8)) + candidate_index).astype(np.float32),
        "prefix_controls": (rng.normal(size=(10, 4)) + candidate_index).astype(np.float32),
        "agent_history": rng.normal(size=(11, 9, 16)).astype(np.float32),
        "agent_valid": np.ones((11, 9), dtype=np.float32),
        "bev_occ": rng.normal(size=(7, 32, 32)).astype(np.float32),
        "route": rng.normal(size=(20, 2)).astype(np.float32),
        "map_polylines": rng.normal(size=(8, 12, 4)).astype(np.float32),
        "dynamic_map": rng.normal(size=(11, 6, 4)).astype(np.float32),
    }
    return sample


def test_shared_scene_feature_matrix_is_exactly_equal() -> None:
    samples = [_feature_sample(i) for i in range(6)]
    # Candidate builders share these arrays at a replan. Make that identity
    # explicit so the optimized path is tested under its actual precondition.
    for key in ("agent_history", "agent_valid", "bev_occ", "route", "map_polylines", "dynamic_map", "ego_state"):
        for sample in samples[1:]:
            sample[key] = samples[0][key]
    expected = np.stack([sample_to_feature(sample, {}) for sample in samples], axis=0)
    actual = samples_to_feature_matrix(samples, {}, shared_scene=True)
    np.testing.assert_array_equal(actual, expected)


def _fake_scene_result(scene_id: str, rank: int) -> dict:
    return {
        "scene_id": scene_id,
        "bucket_name": None,
        "target_key": None,
        "target_time_index": None,
        "num_decisions": 0,
        "num_metric_steps": 0,
        "method": "nominal",
        "label_mode": "fast",
        "rank": rank,
        "metric_summary": {},
        "macro_counts": {},
        "selection_reason_counts": {},
    }


def test_closed_loop_resumes_completed_scenes_from_journal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raws = [SimpleNamespace(scenario_id=f"s{i}", metadata={}) for i in range(4)]
    monkeypatch.setattr(clr, "iter_waymax_womd_scenarios", lambda *args, **kwargs: iter(raws))

    first_calls: list[str] = []

    def interrupted_rollout(raw, rank, *args, **kwargs):
        first_calls.append(raw.scenario_id)
        if raw.scenario_id == "s2":
            raise RuntimeError("simulated interruption")
        return _fake_scene_result(raw.scenario_id, rank)

    monkeypatch.setattr(clr, "_rollout_one_scene", interrupted_rollout)
    output = tmp_path / "closed.json"
    cfg = {
        "closed_loop": {
            "max_scenarios": 4,
            "method": "nominal",
            "resume": True,
            "save_partial": True,
            "partial_write_every_scenes": 4,
            "progress": False,
        },
        "selection": {"gamma_rec": 0.0},
        "waymax": {},
        "artifact": {},
    }
    with pytest.raises(RuntimeError, match="simulated interruption"):
        clr.closed_loop_evaluate("dummy.tfrecord", None, output, cfg)
    assert first_calls == ["s0", "s1", "s2"]
    journal = output.with_suffix(output.suffix + ".scenes.jsonl")
    assert journal.exists()
    assert len(journal.read_text().strip().splitlines()) == 2

    resumed_calls: list[str] = []

    def resumed_rollout(raw, rank, *args, **kwargs):
        resumed_calls.append(raw.scenario_id)
        return _fake_scene_result(raw.scenario_id, rank)

    monkeypatch.setattr(clr, "_rollout_one_scene", resumed_rollout)
    result = clr.closed_loop_evaluate("dummy.tfrecord", None, output, cfg)
    assert resumed_calls == ["s2", "s3"]
    assert result["num_scenes"] == 4
    assert result["resume"]["resumed_rollouts"] == 2
    assert json.loads(output.read_text())["num_scenes"] == 4
    progress = json.loads(output.with_suffix(output.suffix + ".progress.json").read_text())
    assert progress["status"] == "complete"
    assert progress["completed_rollouts"] == 4


def test_legacy_partial_is_resumable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raws = [SimpleNamespace(scenario_id=f"s{i}", metadata={}) for i in range(3)]
    monkeypatch.setattr(clr, "iter_waymax_womd_scenarios", lambda *args, **kwargs: iter(raws))
    output = tmp_path / "legacy.json"
    partial = output.with_suffix(output.suffix + ".partial")
    partial.write_text(json.dumps({
        "method": "nominal",
        "bucket_dataset": None,
        "raw_scenarios_seen": 2,
        "scenes": [_fake_scene_result("s0", 0), _fake_scene_result("s1", 1)],
    }))
    calls: list[str] = []

    def rollout(raw, rank, *args, **kwargs):
        calls.append(raw.scenario_id)
        return _fake_scene_result(raw.scenario_id, rank)

    monkeypatch.setattr(clr, "_rollout_one_scene", rollout)
    cfg = {
        "closed_loop": {"max_scenarios": 3, "method": "nominal", "resume": True, "progress": False},
        "selection": {"gamma_rec": 0.0},
        "waymax": {},
        "artifact": {},
    }
    result = clr.closed_loop_evaluate("dummy.tfrecord", None, output, cfg)
    assert calls == ["s2"]
    assert result["num_scenes"] == 3
    assert result["resume"]["legacy_sources"] == ["partial"]


def test_targeted_zero_rollout_cap_means_all_loaded_targets() -> None:
    assert clr._closed_loop_rollout_limit(target_count=175, max_rollouts=0, max_scenes=0) == 175
    assert clr._closed_loop_rollout_limit(target_count=175, max_rollouts=20, max_scenes=0) == 20
    assert clr._closed_loop_rollout_limit(target_count=0, max_rollouts=0, max_scenes=8) == 8


def test_bucket_closed_loop_materializes_only_selected_target_source_indices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-target WOMD records must not be subjected to route fail-close.

    The selected replay iterator parses past unrelated records but only constructs
    SimulatorState/RawScenario for source indices owned by the fixed bucket.
    """
    target = {
        "bucket_name": "near_contact_test",
        "scene_id": "official_target",
        "scene_aliases": ["official_target"],
        "saved_scene_id": "official_target__wx00000017",
        "original_scenario_id": "official_target",
        "official_scenario_id": "official_target",
        "legacy_scenario_id": "waymax_legacy_target",
        "source_scenario_index": 17,
        "womd_source_role": "validation",
        "waymax_max_num_objects": 64,
        "time_index": 10,
        "target_key": "near_contact_test:official_target:t10",
    }
    monkeypatch.setattr(clr, "_load_closed_loop_targets", lambda *_: [target])

    calls = {"selected": 0, "full": 0, "indices": None}
    raw = SimpleNamespace(
        scenario_id="official_target__wx00000017",
        metadata={
            "_waymax_scenario_index": 17,
            "official_scenario_id": "official_target",
            "original_scenario_id": "official_target",
            "legacy_scenario_id": "waymax_legacy_target",
        },
    )

    def selected(
        _patterns,
        indices,
        parser_cfg=None,
        skip_observation_legal_route_unavailable=False,
        eligibility_diagnostics=None,
    ):
        calls["selected"] += 1
        calls["indices"] = list(indices)
        calls["skip_route_unavailable"] = bool(skip_observation_legal_route_unavailable)
        calls["eligibility_diagnostics"] = eligibility_diagnostics is not None
        yield raw

    def full(*_args, **_kwargs):
        calls["full"] += 1
        raise AssertionError(
            "bucket-targeted evaluation must not materialize unrelated WOMD records"
        )

    monkeypatch.setattr(clr, "iter_waymax_womd_scenarios_selected", selected)
    monkeypatch.setattr(clr, "iter_waymax_womd_scenarios", full)
    monkeypatch.setattr(
        clr,
        "_rollout_one_scene",
        lambda raw, rank, *args, **kwargs: {
            **_fake_scene_result(raw.scenario_id, rank),
            "bucket_name": kwargs.get("bucket_name"),
            "target_key": kwargs.get("target_key"),
            "target_time_index": kwargs.get("start_time_index_override"),
        },
    )

    output = tmp_path / "targeted.json"
    cfg = {
        "closed_loop": {
            "max_scenarios": 0,
            "max_rollouts": 0,
            "method": "nominal",
            "bucket_dataset": "dummy_bucket",
            "bucket_split": "test",
            "require_bucket_targets": True,
            "resume": False,
            "save_partial": False,
            "progress": False,
        },
        "selection": {"gamma_rec": 0.0},
        "waymax": {},
        "artifact": {},
    }
    result = clr.closed_loop_evaluate("/data/validation/validation.tfrecord@150", None, output, cfg)
    assert calls == {
        "selected": 1,
        "full": 0,
        "indices": [17],
        "skip_route_unavailable": False,
        "eligibility_diagnostics": True,
    }
    assert result["num_scenes"] == 1
    assert result["bucket_matched_rollouts"] == 1
    assert result["raw_scan_bound_source"] == "selected_target_source_indices"
    assert result["raw_scenarios_seen_this_run"] == 18


def test_resume_sparse_target_replay_excludes_completed_canonical_targets() -> None:
    targets = [
        {"target_key": "near:s0:t10", "source_scenario_index": 3},
        {"target_key": "near:s1:t10", "source_scenario_index": 8},
        # Same WOMD record, different unfinished target: materialize once.
        {"target_key": "near:s1:t20", "source_scenario_index": 8},
        # Legacy target has no target_key, so it must stay conservative/unfinished.
        {"target_key": "", "source_scenario_index": 11},
    ]
    got = clr._unfinished_selected_target_source_indices(
        targets, {"target:near:s0:t10", "target:some-other-key"}
    )
    assert got == [8, 11]


def test_resume_fingerprint_ignores_persistence_knobs_but_legacy_hash_does_not() -> None:
    """Regression for interrupted publication jobs.

    Requirement: changing only resume/persistence/serialization controls must not
    invalidate an otherwise identical closed-loop scientific run.
    """
    base = {
        "closed_loop": {
            "method": "marc_lite",
            "max_steps": 40,
            "resume": True,
            "resume_force": False,
            "save_partial": True,
            "include_scenes_in_partial": False,
            "partial_write_every_scenes": 32,
        },
        "waymax": {"use_jit_scan_rollouts": True},
    }
    changed = json.loads(json.dumps(base))
    changed["closed_loop"].update({
        "resume_force": True,
        "save_partial": False,
        "partial_write_every_scenes": 1,
        "include_scenes_in_partial": True,
    })

    fp_a = clr._closed_loop_fingerprint("dummy.tfrecord", None, "marc_lite", "near", base)
    fp_b = clr._closed_loop_fingerprint("dummy.tfrecord", None, "marc_lite", "near", changed)
    assert fp_a == fp_b

    legacy_a = clr._closed_loop_legacy_full_config_fingerprint(
        "dummy.tfrecord", None, "marc_lite", "near", base
    )
    legacy_b = clr._closed_loop_legacy_full_config_fingerprint(
        "dummy.tfrecord", None, "marc_lite", "near", changed
    )
    assert legacy_a != legacy_b


def test_metric_only_partial_cannot_veto_compatible_scene_journal(tmp_path: Path) -> None:
    """Publication partials omit scenes; the journal is the resume authority."""
    output = tmp_path / "closed_loop_marc_lite.json"
    partial = output.with_suffix(output.suffix + ".partial")
    journal = output.with_suffix(output.suffix + ".scenes.jsonl")

    # This deliberately has a foreign bookkeeping fingerprint but no scenes.
    # Before the repair it raised before the compatible journal was inspected.
    partial.write_text(json.dumps({
        "run_fingerprint": "stale-metric-only-partial",
        "method": "marc_lite",
        "bucket_dataset": "near",
        "scenes": [],
    }))
    scene = _fake_scene_result("s0", 0)
    scene["method"] = "marc_lite"
    journal.write_text(json.dumps({
        "version": 1,
        "run_fingerprint": "legacy-compatible",
        "resume_key": clr._scene_resume_key(scene),
        "scene": scene,
    }) + "\n")

    scenes, meta = clr._load_resume_scene_results(
        output_path=output,
        partial_path=partial,
        journal_path=journal,
        fingerprint="current",
        compatible_fingerprints={"legacy-compatible"},
        method="marc_lite",
        target_spec="near",
        force=False,
        allow_legacy=True,
    )
    assert [s["scene_id"] for s in scenes] == ["s0"]
    assert meta["sources"] == ["journal"]
    assert meta["fingerprint_migrated_from"] == ["legacy-compatible"]


def test_resume_fingerprint_migration_keeps_journal_single_fingerprint(tmp_path: Path) -> None:
    """Forced/legacy resume must never leave a mixed-fingerprint scene journal."""
    output = tmp_path / "closed_loop_racp_lite.json"
    partial = output.with_suffix(output.suffix + ".partial")
    journal = output.with_suffix(output.suffix + ".scenes.jsonl")
    progress = output.with_suffix(output.suffix + ".progress.json")

    old_fp = "old-fingerprint"
    new_fp = "new-fingerprint"
    partial.write_text(json.dumps({"run_fingerprint": old_fp, "scenes": []}))
    progress.write_text(json.dumps({"run_fingerprint": old_fp, "status": "running"}))
    rows = []
    for i in range(2):
        scene = _fake_scene_result(f"s{i}", i)
        scene["method"] = "racp_lite"
        rows.append(json.dumps({
            "version": 1,
            "run_fingerprint": old_fp,
            "resume_key": clr._scene_resume_key(scene),
            "scene": scene,
        }))
    journal.write_text("\n".join(rows) + "\n")

    clr._migrate_resume_fingerprints(
        output_path=output,
        partial_path=partial,
        journal_path=journal,
        progress_path=progress,
        new_fingerprint=new_fp,
        old_fingerprints=[old_fp],
    )

    assert json.loads(partial.read_text())["run_fingerprint"] == new_fp
    assert json.loads(progress.read_text())["run_fingerprint"] == new_fp
    fps = {
        json.loads(line)["run_fingerprint"]
        for line in journal.read_text().splitlines()
        if line.strip()
    }
    assert fps == {new_fp}


def test_forced_resume_still_rejects_wrong_method_or_target_lock() -> None:
    """Fingerprint force must not bypass the frozen scientific cohort contract."""
    targets = [{"target_key": "near:s0:t10"}, {"target_key": "near:s1:t10"}]
    scene = _fake_scene_result("s0", 0)
    scene.update({"method": "racp_lite", "target_key": "near:s0:t10"})
    with pytest.raises(ValueError, match="contains method"):
        clr._validate_resumed_scene_contract(
            [scene], method="marc_lite", targets=targets, require_target_keys=True
        )

    scene["method"] = "marc_lite"
    scene["target_key"] = "near:foreign:t10"
    with pytest.raises(ValueError, match="outside the current frozen target lock"):
        clr._validate_resumed_scene_contract(
            [scene], method="marc_lite", targets=targets, require_target_keys=True
        )
