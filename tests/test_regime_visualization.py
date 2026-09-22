from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import render_regime_visualization_videos as renderer
import select_regime_visualization_scenes as selector


def _args():
    return SimpleNamespace(
        **selector.DEFAULT_THRESHOLDS,
        scenario_horizon_steps=91,
        metric_dt_s=0.1,
    )


def _near_scene(*, target="s:t10", ttc=2.5, clearance=1.8, exposure=0.2, near_zero=0.01):
    return {
        "target_key": target,
        "scene_id": target.split(":", 1)[0],
        "source_scenario_index": 3,
        "target_time_index": 10,
        "intervention_rate": 0.1,
        "closed_loop_bounded_NUP": 0.98,
        "ttc_s_p05": ttc,
        "terminal_ttc_s": 4.5,
        "min_clearance_m_p05": clearance,
        "terminal_clearance_m": 2.8,
        "critical_ttc_exposure_duration_s": exposure,
        "near_zero_clearance_exposure_rate": near_zero,
        "overlap_any": 0.0,
        "offroad_any": 0.0,
    }


def test_duration_contract_uses_womd_future_horizon():
    assert selector._duration_available_s({"target_time_index": 40}, 91, 0.1) == 5.0
    assert selector._duration_available_s({"target_time_index": 60}, 91, 0.1) == 3.0
    assert selector._duration_available_s({"target_time_index": 61}, 91, 0.1) == pytest.approx(2.9)


def test_near_selector_compares_all_baselines_and_records_best_worst():
    ocrap = {"s:t10": _near_scene()}
    baseline_a = _near_scene(ttc=1.2, clearance=0.7, exposure=1.2, near_zero=0.10)
    baseline_b = _near_scene(ttc=1.8, clearance=1.2, exposure=0.7, near_zero=0.05)
    rows = selector._paired_rows("near", ocrap, {"a": {"s:t10": baseline_a}, "b": {"s:t10": baseline_b}}, _args())
    assert len(rows) == 1
    row = rows[0]
    assert row["best_external_method"] == "b"
    assert row["worst_external_method"] == "a"
    assert set(row["per_baseline"]) == {"a", "b"}
    assert row["selection_tier"] == "beats_hardest_external_strict"
    assert row["primary_external_method"] == "b"
    assert row["primary_external_method"] == row["hardest_external_method"]
    assert set(row["external_metrics"]) == {"a", "b"}


def test_oriented_box_clearance_identifies_true_minimum_box():
    frame = {
        "agents": [
            {"is_sdc": True, "x": 0.0, "y": 0.0, "length": 4.0, "width": 2.0, "yaw": 0.0},
            # Center is farther than the next vehicle, but its long box reaches closer.
            {"is_sdc": False, "x": 7.0, "y": 0.0, "length": 8.0, "width": 2.0, "yaw": 0.0, "name": "long"},
            {"is_sdc": False, "x": 5.5, "y": 3.0, "length": 2.0, "width": 2.0, "yaw": 0.0, "name": "short"},
        ]
    }
    pair = renderer._minimum_box_pair(frame)
    assert pair is not None
    _, other, distance = pair
    assert other["name"] == "long"
    assert abs(distance - 1.0) < 1e-9


def test_video_sampling_is_time_based_not_trace_length_based():
    assert renderer._sample_indices(50, fps=10, metric_dt_s=0.1)[:4] == [0, 1, 2, 3]
    assert renderer._sample_indices(50, fps=20, metric_dt_s=0.1)[:5] == [0, 0, 1, 1, 2]

def test_primary_comparator_uses_hardest_paired_score_not_absolute_winner(monkeypatch):
    ocrap = {"s:t10": _near_scene()}
    # Make b the stronger absolute baseline (larger absolute quality), while the
    # paired critical-safety evaluator says a is harder for OC-RAP to beat.
    baseline_a = _near_scene(ttc=1.2, clearance=0.7, exposure=1.2, near_zero=0.10)
    baseline_b = _near_scene(ttc=2.4, clearance=1.75, exposure=0.25, near_zero=0.012)

    def fake_eval(regime, method_scene, external_scene, thresholds):
        hard = external_scene is baseline_a
        return {
            "score": 0.05 if hard else 0.50,
            "material": ["ttc"] if hard else ["ttc", "clearance"],
            "regressions": [],
            "missing": [],
            "evidence_profile": "ttc",
            "terms": {},
        }

    monkeypatch.setattr(selector, "_pair_evaluate", fake_eval)
    rows = selector._paired_rows("near", ocrap, {"a": {"s:t10": baseline_a}, "b": {"s:t10": baseline_b}}, _args())
    row = rows[0]
    assert row["best_external_method"] == "b"
    assert row["primary_external_method"] == "a"
    assert row["hardest_external_method"] == "a"


def test_favorable_delta_respects_metric_direction_and_missing_values():
    assert renderer._favorable_delta(3.0, 2.0, "higher") == 1.0
    assert renderer._favorable_delta(1.0, 2.0, "lower") == 1.0
    assert renderer._favorable_delta(None, 2.0, "higher") is None


def test_submission_video_uses_compact_paper_display_names():
    expected = {
        "gameformer_lite": "GameFormer",
        "plantf": "PlanTF",
        "pluto": "PLUTO",
        "pdm_closed": "PDM-Closed",
        "pdm_hybrid": "PDM-Hybrid",
        "idm": "IDM",
        "marc_lite": "MARC",
        "racp_lite": "RACP",
        "robust_scenario_mpc": "Scenario MPC",
        "predictive_safety_filter": "PSF",
        "dr_cvar_safety_filter": "DR-CVaR",
        "conformal_predictive_safety_filter": "CPSF",
        "postimpact_mpc_lite": "MPC + PSO",
        "post_crash_braking": "PIB",
        "postimpact_motion_tvlqr": "APF + TVLQR",
        "post_collision_restoration": "Heuristic restoration",
        "compensatory_postimpact_mpc": "FCC-MPC",
        "robust_postimpact_control": "SMC + QP",
    }
    for method, label in expected.items():
        assert renderer._display_name(method) == label
    assert renderer._display_name("ocrap") == "OC-RAP"
    assert renderer._short_comparator_role("lowest paired critical-safety score across all external baselines (hardest to beat)", "near") == "Hardest paired external"


def test_selector_cli_registers_allowed_target_keys_file(tmp_path, monkeypatch):
    key = "test_safe:scene1:t10"
    ocrap_scene = {
        "target_key": key,
        "scene_id": "scene1",
        "target_time_index": 10,
        "closed_loop_bounded_NUP": 0.99,
        "intervention_rate": 0.02,
        "min_clearance_m_p05": 2.0,
        "ttc_s_p05": 5.0,
        "overlap_any": 0.0,
        "offroad_any": 0.0,
        "route_progression_m": 12.0,
        "jerk_p95": 1.0,
        "yaw_rate_p95": 0.1,
    }
    baseline_scene = dict(ocrap_scene)
    baseline_scene.update({"closed_loop_bounded_NUP": 0.95, "intervention_rate": 0.05})

    ocrap_path = tmp_path / "ocrap.jsonl"
    baseline_path = tmp_path / "baseline.jsonl"
    allowed_path = tmp_path / "allowed.json"
    output_path = tmp_path / "selection.json"
    keys_path = tmp_path / "keys.json"
    ocrap_path.write_text(__import__("json").dumps({"scene": ocrap_scene}) + "\n")
    baseline_path.write_text(__import__("json").dumps({"scene": baseline_scene}) + "\n")
    allowed_path.write_text(__import__("json").dumps([key]) + "\n")

    monkeypatch.setattr(sys, "argv", [
        "select_regime_visualization_scenes.py",
        "--regime", "safe",
        "--ocrap-scenes", str(ocrap_path),
        "--baseline", f"dummy={baseline_path}",
        "--allowed-target-keys-file", str(allowed_path),
        "--output", str(output_path),
        "--target-keys-output", str(keys_path),
        "--num-scenes", "1",
        "--min-duration-s", "5",
        "--fallback-min-duration-s", "3",
        "--max-selected-tier-rank", "1",
    ])
    assert selector.main() == 0
    assert __import__("json").loads(keys_path.read_text())["target_keys"] == [key]
