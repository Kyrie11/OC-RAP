from __future__ import annotations

import csv
from pathlib import Path

from ocrap.config.defaults import DEFAULT_CONFIG
from ocrap.data.build.builder import MANIFEST_FIELDS, build_dataset
from ocrap.data.build.manifest import rebuild_manifest


def test_rebuild_manifest_from_existing_samples(tmp_path: Path):
    cfg = DEFAULT_CONFIG.copy()
    cfg.update(
        {
            "progress": False,
            "num_synthetic_scenarios": 1,
            "num_candidate_prefixes": 3,
            "num_reactive_futures": 1,
            "num_targeted_futures": 0,
            "max_times_per_scenario": 1,
            "max_biased_times_per_scenario": 0,
            "bev_resolution_m": 4.0,
            "io": {"compress_npz": False, "fsync_npz": False},
        }
    )
    root = tmp_path / "ds"
    built = build_dataset(root, cfg, skip_existing=True)
    assert built["num_samples"] > 0
    (root / "manifest.csv").unlink()

    summary = rebuild_manifest(root, require_complete=True)

    assert summary["num_manifest_rows"] == built["num_samples"]
    assert summary["safe_for_direct_reader_scan"] is True
    with (root / "manifest.csv").open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert reader.fieldnames == MANIFEST_FIELDS
    assert len(rows) == built["num_samples"]


def test_rebuild_manifest_detects_and_quarantines_uncommitted_group(tmp_path: Path):
    import shutil
    import pytest

    cfg = DEFAULT_CONFIG.copy()
    cfg.update(
        {
            "progress": False,
            "num_synthetic_scenarios": 1,
            "num_candidate_prefixes": 2,
            "num_reactive_futures": 1,
            "num_targeted_futures": 0,
            "max_times_per_scenario": 1,
            "max_biased_times_per_scenario": 0,
            "bev_resolution_m": 4.0,
            "io": {"compress_npz": False, "fsync_npz": False},
        }
    )
    root = tmp_path / "partial"
    build_dataset(root, cfg, skip_existing=True)
    source = next((root / "samples").glob("*.npz"))
    fake_partial = root / "samples" / "fake_scene_t9999_a00.npz"
    shutil.copy2(source, fake_partial)

    with pytest.raises(RuntimeError, match="uncommitted_scene_time_groups=1"):
        rebuild_manifest(root, require_complete=True)

    summary = rebuild_manifest(root, quarantine_uncommitted=True)
    assert summary["num_uncommitted_scene_time_groups"] == 1
    assert fake_partial.name in summary["uncommitted_npz_quarantined"]
    assert not fake_partial.exists()
    assert (root / "samples" / "incomplete_scene_times" / fake_partial.name).exists()
    assert summary["safe_for_direct_reader_scan"] is True
