from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ocrap.simulation.closed_loop_runner import _state_geometry_metrics


def _overlap_state() -> SimpleNamespace:
    tr = SimpleNamespace(
        x=np.asarray([[0.0], [1.0]], dtype=float),
        y=np.asarray([[0.0], [0.0]], dtype=float),
        vel_x=np.zeros((2, 1), dtype=float),
        vel_y=np.zeros((2, 1), dtype=float),
        yaw=np.zeros((2, 1), dtype=float),
        length=np.full((2, 1), 4.0, dtype=float),
        width=np.full((2, 1), 2.0, dtype=float),
        height=np.full((2, 1), 1.5, dtype=float),
        valid=np.ones((2, 1), dtype=bool),
    )
    return SimpleNamespace(timestep=np.asarray(0), sim_trajectory=tr)


def test_publication_clearance_is_signed_under_penetration() -> None:
    metrics = _state_geometry_metrics(_overlap_state(), 0, publication_exact=True)
    assert metrics["min_clearance_m"] < 0.0
    assert metrics["min_clearance_m"] == metrics["signed_clearance_m"]
    assert np.isclose(metrics["penetration_depth_m"], -metrics["min_clearance_m"])


def test_paired_comparison_fails_closed_on_target_set_mismatch(tmp_path: Path) -> None:
    control = tmp_path / "control.json"
    method = tmp_path / "method.json"
    output = tmp_path / "comparison.json"
    control.write_text(json.dumps({"scenes": [
        {"target_key": "a", "metric_summary": {"overlap_any": 0.0}},
        {"target_key": "b", "metric_summary": {"overlap_any": 0.0}},
    ]}))
    method.write_text(json.dumps({"scenes": [
        {"target_key": "a", "metric_summary": {"overlap_any": 0.0}},
    ]}))
    proc = subprocess.run(
        [sys.executable, "tools/compare_paired_closed_loop.py", str(control), str(method), "--output", str(output), "--bootstrap", "10"],
        text=True, capture_output=True,
    )
    assert proc.returncode != 0
    assert "Target-key sets differ" in (proc.stdout + proc.stderr)
    assert not output.exists()


def test_paired_comparison_marks_time_to_minimum_as_descriptive(tmp_path: Path) -> None:
    control = tmp_path / "control.json"
    method = tmp_path / "method.json"
    output = tmp_path / "comparison.json"
    control.write_text(json.dumps({"scenes": [
        {"target_key": "a", "metric_summary": {"time_to_min_clearance_s": 1.0}},
    ]}))
    method.write_text(json.dumps({"scenes": [
        {"target_key": "a", "metric_summary": {"time_to_min_clearance_s": 2.0}},
    ]}))
    subprocess.run(
        [sys.executable, "tools/compare_paired_closed_loop.py", str(control), str(method), "--output", str(output), "--bootstrap", "10"],
        check=True, text=True, capture_output=True,
    )
    row = json.loads(output.read_text())["metrics"]["time_to_min_clearance_s"]
    assert row["direction"] == "descriptive_only"
    assert "fraction_improved" not in row


def test_publication_contact_table_requires_full_post_contact_eligibility(tmp_path: Path) -> None:
    def doc(eligibility: float) -> dict:
        return {
            "clearance_metric_full_coverage_scene_rate": 1.0,
            "ttc_metric_full_coverage_scene_rate": 1.0,
            "overlap_metric_full_coverage_scene_rate": 1.0,
            "offroad_metric_full_coverage_scene_rate": 1.0,
            "post_contact_metric_eligible_scene_rate": eligibility,
            "contact_anchor_protocol": "exact_a0_pretreatment_prelude_v1",
            "contact_anchor_manifest_sha256": "a" * 64,
            "contact_anchor_state_fingerprint_required": True,
            "evaluation_contract": {
                "metric_semantics_version": "publication_v55_signed_clearance_unclipped_v1",
                "clearance_is_signed": True,
            },
        }

    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps(doc(1.0)))
    b.write_text(json.dumps(doc(0.5)))
    proc = subprocess.run(
        [
            sys.executable,
            "tools/build_regime_comparison_tables.py",
            "--regime", "contact",
            "--input", f"a={a}",
            "--input", f"b={b}",
            "--output-dir", str(tmp_path / "out"),
        ],
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": str(Path.cwd() / "src")},
    )
    assert proc.returncode != 0
    assert "100% post-contact metric eligibility" in (proc.stdout + proc.stderr)
