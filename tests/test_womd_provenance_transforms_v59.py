from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import merge_dataset_roots as merger
import resolve_womd_replay_source as resolver


def _write_legacy_shard(root: Path, scene: str, pattern: str) -> None:
    (root / "samples").mkdir(parents=True)
    np.savez_compressed(
        root / "samples" / f"{scene}.npz",
        split_id=np.asarray("calibration"),
        scene_id=np.asarray(scene),
        original_scenario_id=np.asarray(scene),
    )
    with (root / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["path", "scene_id", "original_scenario_id", "split_id", "custom_future_field"],
        )
        w.writeheader()
        w.writerow(
            {
                "path": f"samples/{scene}.npz",
                "scene_id": scene,
                "original_scenario_id": scene,
                "split_id": "calibration",
                "custom_future_field": "keep_me",
            }
        )
    (root / "resume_contract.json").write_text(
        json.dumps({"semantic_config": {"womd_patterns": pattern}}), encoding="utf-8"
    )


def _make_shards(root: Path, n: int = 2) -> None:
    d = root / "validation"
    d.mkdir(parents=True)
    for i in range(n):
        (d / f"validation_tfexample.tfrecord-{i:05d}-of-{n:05d}").write_bytes(b"x")


def test_merge_and_filter_preserve_and_backfill_womd_replay_provenance(tmp_path: Path) -> None:
    pattern = "/archive/tf_example/validation/validation_tfexample.tfrecord@2"
    s0, s1 = tmp_path / "s0", tmp_path / "s1"
    _write_legacy_shard(s0, "scene_a", pattern)
    _write_legacy_shard(s1, "scene_b", pattern)

    merged = tmp_path / "merged"
    merger.merge([s0, s1], merged, copy=True)

    with (merged / "manifest.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["womd_source_role"] for r in rows] == ["validation", "validation"]
    assert all(r["custom_future_field"] == "keep_me" for r in rows)
    assert "official_scenario_id" in rows[0]  # current schema retained even for legacy inputs
    contract = json.loads((merged / "womd_replay_contract.json").read_text(encoding="utf-8"))
    assert contract["womd_source_role"] == "validation"
    assert contract["womd_pattern"] == pattern

    filtered = tmp_path / "filtered"
    subprocess.run(
        [
            sys.executable,
            str(TOOLS / "filter_dataset_scenes_v48.py"),
            "--input",
            str(merged),
            "--output",
            str(filtered),
            "--link-mode",
            "copy",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    filtered_contract = json.loads((filtered / "womd_replay_contract.json").read_text(encoding="utf-8"))
    assert filtered_contract["womd_source_role"] == "validation"

    womd = tmp_path / "tf_example"
    _make_shards(womd)
    doc = resolver.resolve_for_dataset(
        filtered, split="calibration", womd_root=womd, shards=2, role="auto"
    )
    assert doc["resolved_role"] == "validation"
    assert doc["role_counts"] == {"validation": 2}


def test_publication_repair_harness_pins_paper_womd_role_and_refits_cpsf_if_needed() -> None:
    repo = Path(__file__).resolve().parents[1]
    text = (repo / "scripts/rerun_corrected_closed_loop_metrics_v55.sh").read_text(encoding="utf-8")
    assert ': "${PRIMARY_WOMD_ROLE:=validation}"' in text
    assert 'calibration_near_contact" calibration "$NEAR_CALIB_WOMD_ROLE"' in text
    assert 'CL_WOMD="$near_womd" CALIB_WOMD="$near_calib_womd"' in text
    assert 'DO_TRAIN=false DO_CALIBRATE=true FORCE_RECALIBRATE="$NEAR_FORCE_RECALIBRATE"' in text


def test_merge_does_not_promote_unknown_input_from_other_known_root(tmp_path: Path) -> None:
    pattern = "/archive/tf_example/validation/validation_tfexample.tfrecord@2"
    known, unknown = tmp_path / "known", tmp_path / "unknown"
    _write_legacy_shard(known, "scene_known", pattern)
    _write_legacy_shard(unknown, "scene_unknown", pattern)
    (unknown / "resume_contract.json").unlink()

    merged = tmp_path / "mixed"
    merger.merge([known, unknown], merged, copy=True)
    assert not (merged / "womd_replay_contract.json").exists()
    summary = json.loads((merged / "merged_dataset_summary.json").read_text(encoding="utf-8"))
    assert summary["womd_source_role"] == "unknown"
