from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_tool(name: str):
    path = Path(__file__).resolve().parents[1] / "tools" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scene(key: str, metrics: dict, *, intervention: float = 0.0, macro: str = "nominal") -> dict:
    return {
        "scene_id": key,
        "target_key": f"test:{key}:t10",
        "target_time_index": 10,
        "bucket_name": "test_near_contact",
        "canonical_regime": "near_contact",
        "intervention_rate": intervention,
        "macro_counts": {macro: 4},
        "metric_summary": metrics,
        "decisions": [
            {
                "selected_candidate_index": 1 if intervention else 0,
                "selected_macro": macro,
                "metrics_after_step": {
                    "min_clearance_m": metrics.get("min_clearance_m_min", 1.0),
                    "ttc_s": metrics.get("ttc_s_min", 1.0),
                    "ego_speed_mps": 2.0,
                    "overlap": metrics.get("overlap_any", 0.0),
                    "offroad": metrics.get("offroad_any", 0.0),
                    "ego_yaw_rad": 0.0,
                },
            }
        ],
    }


def test_near_score_rewards_clearance_and_deficit_reduction():
    tool = _load_tool("select_critical_closed_loop_scenes.py")
    control = _scene("a", {
        "min_clearance_m_min": 0.5,
        "ttc_s_min": 1.0,
        "clearance_deficit_auc_m_s": 2.0,
        "ttc_deficit_auc_s2": 2.0,
        "overlap_any": 0.0,
        "offroad_any": 0.0,
    })
    method = _scene("a", {
        "min_clearance_m_min": 1.2,
        "ttc_s_min": 2.0,
        "clearance_deficit_auc_m_s": 0.7,
        "ttc_deficit_auc_s2": 0.5,
        "overlap_any": 0.0,
        "offroad_any": 0.0,
    }, intervention=0.25, macro="stabilize")
    row = tool._score_pair(control, method, "near_contact")
    assert row["improvement_score"] > 0
    assert row["method_intervened"] is True
    assert row["deltas_method_minus_control"]["min_clearance_m_min"] > 0


def test_new_overlap_is_ranked_as_hard_regression():
    tool = _load_tool("select_critical_closed_loop_scenes.py")
    control = _scene("b", {"min_clearance_m_min": 1.0, "ttc_s_min": 2.0, "overlap_any": 0.0, "offroad_any": 0.0})
    method = _scene("b", {"min_clearance_m_min": 1.5, "ttc_s_min": 2.5, "overlap_any": 1.0, "offroad_any": 0.0}, intervention=0.25, macro="brake")
    row = tool._score_pair(control, method, "near_contact")
    assert row["safety_flags"]["new_overlap"] is True
    assert row["hard_safety_penalty"] >= 12.0
    assert row["regression_score"] > 0


def test_contact_score_rewards_free_space_and_no_recontact():
    tool = _load_tool("select_critical_closed_loop_scenes.py")
    control = _scene("c", {
        "overlap_any": 1.0,
        "overlap_duration_s": 0.8,
        "recontact_event": 1.0,
        "post_contact_free_space_auc_m_s": 0.5,
        "post_contact_clearance_m_mean": 0.2,
        "offroad_any": 0.0,
    })
    control["post_contact_target"] = True
    method = _scene("c", {
        "overlap_any": 1.0,
        "overlap_duration_s": 0.3,
        "recontact_event": 0.0,
        "post_contact_free_space_auc_m_s": 2.0,
        "post_contact_clearance_m_mean": 0.8,
        "offroad_any": 0.0,
    }, intervention=0.25, macro="stabilize")
    method["post_contact_target"] = True
    row = tool._score_pair(control, method, "contact")
    assert row["improvement_score"] > 0
    assert row["deltas_method_minus_control"]["recontact_event"] < 0


def test_selector_writes_two_sided_report(tmp_path: Path, monkeypatch):
    tool = _load_tool("select_critical_closed_loop_scenes.py")
    control = {
        "scenes": [
            _scene("improve", {"min_clearance_m_min": 0.5, "ttc_s_min": 1.0, "clearance_deficit_auc_m_s": 2.0, "overlap_any": 0.0, "offroad_any": 0.0}),
            _scene("regress", {"min_clearance_m_min": 1.0, "ttc_s_min": 2.0, "overlap_any": 0.0, "offroad_any": 0.0}),
        ]
    }
    method = {
        "scenes": [
            _scene("improve", {"min_clearance_m_min": 1.4, "ttc_s_min": 2.2, "clearance_deficit_auc_m_s": 0.2, "overlap_any": 0.0, "offroad_any": 0.0}, intervention=0.25, macro="stabilize"),
            _scene("regress", {"min_clearance_m_min": 1.1, "ttc_s_min": 2.1, "overlap_any": 1.0, "offroad_any": 0.0}, intervention=0.25, macro="brake"),
        ]
    }
    cp, mp, out = tmp_path / "c.json", tmp_path / "m.json", tmp_path / "critical.json"
    cp.write_text(json.dumps(control)); mp.write_text(json.dumps(method))
    monkeypatch.setattr("sys.argv", ["tool", str(cp), str(mp), "--regime", "near_contact", "--output", str(out), "--top-k-each", "2"])
    assert tool.main() == 0
    doc = json.loads(out.read_text())
    assert doc["categories"]["largest_improvements"]
    assert doc["categories"]["largest_regressions"]
    assert doc["selection_policy"]["two_sided"] is True
