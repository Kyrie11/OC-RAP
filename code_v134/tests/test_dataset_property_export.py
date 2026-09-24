from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


def _load_tool():
    repo = Path(__file__).resolve().parents[1]
    path = repo / "tools" / "export_dataset_properties.py"
    spec = importlib.util.spec_from_file_location("export_dataset_properties", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_construction_snapshot_preserves_build_provenance(tmp_path):
    mod = _load_tool()
    semantic = {
        "data_source": "womd",
        "simulation_backend": "waymax_closed_loop",
        "womd_patterns": "/womd/validation/validation_tfexample.tfrecord@150",
        "split": {"force_id": "test"},
        "num_candidate_prefixes": 24,
        "num_reactive_futures": 2,
        "num_targeted_futures": 8,
        "targeted_future_kinds": ["hidden_vehicle_yields", "control_delay_noise"],
        "num_roots": 8,
        "num_recovery_options": 12,
        "dataset_quality": {"max_accepted_prefixes_per_scene_time": 8},
        "artifact": {"force_mine": True},
        "waymax": {"teacher_backend": "hybrid"},
    }
    (tmp_path / "resume_contract.json").write_text(
        json.dumps({"generator_version": "x", "fingerprint": "abc", "semantic_config": semantic}),
        encoding="utf-8",
    )
    (tmp_path / "dataset_summary.json").write_text(
        json.dumps({"generation": {"max_scenarios": 160, "scenario_start_index": 100}}),
        encoding="utf-8",
    )
    with (tmp_path / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["path", "split_id", "original_scenario_id", "time_index", "womd_source_role", "scenario_id_source"],
        )
        w.writeheader()
        w.writerow({"path": "a.npz", "split_id": "test", "original_scenario_id": "s1", "time_index": 1,
                    "womd_source_role": "validation", "scenario_id_source": "official"})
        w.writerow({"path": "b.npz", "split_id": "test", "original_scenario_id": "s1", "time_index": 2,
                    "womd_source_role": "validation", "scenario_id_source": "official"})
    snapshot, cfg = mod._construction_snapshot(tmp_path)
    assert cfg == semantic
    recon = snapshot["reconstructed_build_parameters"]
    assert recon["womd_patterns"].endswith("validation_tfexample.tfrecord@150")
    assert recon["num_candidate_prefixes"] == 24
    assert recon["num_roots"] == 8
    assert recon["num_recovery_options"] == 12
    assert snapshot["manifest"]["womd_source_role_counts"] == {"validation": 2}
    assert snapshot["manifest"]["unique_scenes"] == 1
    assert snapshot["manifest"]["unique_scene_time_groups"] == 2


def test_batch_dataset_audit_defaults_to_all_twelve_publication_buckets():
    repo = Path(__file__).resolve().parents[1]
    text = (repo / "scripts" / "analyze_dataset_properties.sh").read_text(encoding="utf-8")
    for name in (
        "train_safe", "train_near_contact", "train_contact",
        "val_safe", "val_near_contact", "val_contact",
        "calibration_safe", "calibration_near_contact", "calibration_contact",
        "test_safe", "test_near_contact", "test_contact",
    ):
        assert name in text
