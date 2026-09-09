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
