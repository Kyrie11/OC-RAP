from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import resolve_womd_replay_source as resolver


def _make_dataset(root: Path, role: str) -> Path:
    samples = root / "samples"
    samples.mkdir(parents=True)
    p = samples / "sample_000.npz"
    np.savez_compressed(p, split_id=np.asarray("test"), womd_source_role=np.asarray(role))
    # Manifest-first path mirrors production metadata access.
    with (root / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "split_id", "womd_source_role"])
        w.writeheader()
        w.writerow({"path": "samples/sample_000.npz", "split_id": "test", "womd_source_role": role})
    return root


def _make_shards(root: Path, role: str, n: int = 2) -> None:
    dirname, prefix = resolver.ROLE_TO_PREFIX[role]
    d = root / dirname
    d.mkdir(parents=True)
    for i in range(n):
        (d / f"{prefix}-{i:05d}-of-{n:05d}").write_bytes(b"x")


def test_auto_resolves_validation_under_tf_example_root(tmp_path: Path):
    data = _make_dataset(tmp_path / "dataset", "validation")
    womd = tmp_path / "tf_example"
    _make_shards(womd, "validation")
    doc = resolver.resolve_for_dataset(data, split="test", womd_root=womd, shards=2, role="auto")
    assert doc["resolved_role"] == "validation"
    assert doc["womd_spec"].endswith("/validation/validation_tfexample.tfrecord@2")
    assert doc["num_resolved_womd_files"] == 2


def test_auto_resolves_validation_interactive_under_same_root(tmp_path: Path):
    data = _make_dataset(tmp_path / "dataset", "validation_interactive")
    womd = tmp_path / "tf_example"
    _make_shards(womd, "validation_interactive")
    doc = resolver.resolve_for_dataset(data, split="test", womd_root=womd, shards=2, role="auto")
    assert doc["resolved_role"] == "validation_interactive"
    assert doc["womd_spec"].endswith(
        "/validation_interactive/validation_interactive_tfexample.tfrecord@2"
    )


def test_explicit_wrong_role_fails_closed(tmp_path: Path):
    data = _make_dataset(tmp_path / "dataset", "validation_interactive")
    womd = tmp_path / "tf_example"
    _make_shards(womd, "validation_interactive")
    with pytest.raises(RuntimeError, match="disagrees with dataset provenance"):
        resolver.resolve_for_dataset(data, split="test", womd_root=womd, shards=2, role="validation")


def test_launchers_use_dataset_owned_auto_replay():
    repo = Path(__file__).resolve().parents[1]
    direct = (
        "scripts/run_safe_regime_external_baselines.sh",
        "scripts/run_near_contact_external_baselines_2gpu_optimized.sh",
        "scripts/run_contact_external_baselines.sh",
        "scripts/run_ocrap_three_regime_closed_loop.sh",
    )
    for rel in direct:
        text = (repo / rel).read_text(encoding="utf-8")
        assert "WOMD_ROOT" in text
        assert "v50_resolve_bucket_womd_spec" in text
        assert "auto" in text
    wrapper = (repo / "scripts/run_all_regime_external_baselines_optimized.sh").read_text(encoding="utf-8")
    assert "WOMD_ROOT" in wrapper
    assert ': "${SAFE_CL_WOMD:=auto}"' in wrapper
    assert ': "${NEAR_CL_WOMD:=auto}"' in wrapper
    assert ': "${CONTACT_CL_WOMD:=auto}"' in wrapper


def _make_legacy_dataset_with_resume_contract(root: Path, pattern: str, *, adopted_legacy: bool = False) -> Path:
    samples = root / "samples"
    samples.mkdir(parents=True)
    p = samples / "sample_000.npz"
    # Deliberately emulate the user's historical dataset: no womd_source_role.
    np.savez_compressed(p, split_id=np.asarray("test"), scene_id=np.asarray("scene_0"), time_index=np.asarray(10))
    with (root / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "split_id", "scene_id", "time_index"])
        w.writeheader()
        w.writerow({"path": "samples/sample_000.npz", "split_id": "test", "scene_id": "scene_0", "time_index": 10})
    (root / "resume_contract.json").write_text(
        __import__("json").dumps({
            "generator_version": "fixture",
            "fingerprint": "abc",
            "adopted_legacy": adopted_legacy,
            "semantic_config": {"womd_patterns": pattern},
        }),
        encoding="utf-8",
    )
    return root


def test_legacy_unknown_rows_fall_back_to_resume_contract_validation(tmp_path: Path):
    data = _make_legacy_dataset_with_resume_contract(
        tmp_path / "dataset",
        "/archive/tf_example/validation/validation_tfexample.tfrecord@2",
    )
    womd = tmp_path / "tf_example"
    _make_shards(womd, "validation")
    doc = resolver.resolve_for_dataset(data, split="test", womd_root=womd, shards=2, role="auto")
    assert doc["dataset_source_role"] == "validation"
    assert doc["resolved_role"] == "validation"
    assert doc["role_counts"] == {"unknown": 1}
    assert "resume_contract.json:semantic_config.womd_patterns" == doc["dataset_source_role_source"]
    assert doc["explicit_role_for_unprovenanced_legacy_dataset"] is False


def test_legacy_unknown_rows_fall_back_to_resume_contract_interactive(tmp_path: Path):
    data = _make_legacy_dataset_with_resume_contract(
        tmp_path / "dataset",
        "/archive/tf_example/validation_interactive/validation_interactive_tfexample.tfrecord@2",
    )
    womd = tmp_path / "tf_example"
    _make_shards(womd, "validation_interactive")
    doc = resolver.resolve_for_dataset(data, split="test", womd_root=womd, shards=2, role="auto")
    assert doc["resolved_role"] == "validation_interactive"
    assert doc["role_counts"] == {"unknown": 1}


def test_stored_row_role_and_resume_contract_conflict_fails_closed(tmp_path: Path):
    data = _make_dataset(tmp_path / "dataset", "validation")
    (data / "resume_contract.json").write_text(
        __import__("json").dumps({
            "semantic_config": {
                "womd_patterns": "/archive/tf_example/validation_interactive/validation_interactive_tfexample.tfrecord@2"
            }
        }),
        encoding="utf-8",
    )
    womd = tmp_path / "tf_example"
    _make_shards(womd, "validation")
    with pytest.raises(RuntimeError, match="disagrees with dataset-level provenance"):
        resolver.resolve_for_dataset(data, split="test", womd_root=womd, shards=2, role="auto")


def test_explicit_role_can_declare_fully_unprovenanced_legacy_dataset(tmp_path: Path):
    samples = tmp_path / "dataset" / "samples"
    samples.mkdir(parents=True)
    np.savez_compressed(samples / "sample_000.npz", split_id=np.asarray("test"))
    womd = tmp_path / "tf_example"
    _make_shards(womd, "validation")
    doc = resolver.resolve_for_dataset(
        tmp_path / "dataset", split="test", womd_root=womd, shards=2, role="validation"
    )
    assert doc["dataset_source_role"] == "unknown"
    assert doc["resolved_role"] == "validation"
    assert doc["explicit_role_for_unprovenanced_legacy_dataset"] is True
