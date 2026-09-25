from __future__ import annotations

from tools.contact_scene_diagnostics import analyze_contact_trace, criticality_score
from tools.contact_clip_metrics import recompute_contact_metric_summary
from tools.synthesize_contact_reference import _profile_set


def _agent(i, x, y=0.0, *, sdc=False):
    return {"object_index": i, "x": float(x), "y": float(y), "yaw": 0.0,
            "length": 4.0, "width": 2.0, "is_sdc": bool(sdc)}


def _frame(sdc_x, overlap, clearance):
    return {
        "agents": [
            _agent(0, sdc_x, sdc=True),
            _agent(1, -1.0),          # initial contact partner
            _agent(2, 9.0),          # later secondary partner
            _agent(3, 5.0, 4.0),     # extra crowd actors
            _agent(4, 7.0, -4.0),
        ],
        "metrics": {"overlap": float(overlap), "min_clearance_m": float(clearance),
                    "offroad": 0.0, "sdc_off_route": 0.0,
                    "ego_speed_mps": 3.0, "ego_yaw_rad": 0.0},
    }


def test_secondary_collision_and_crowding_are_detected():
    trace = [_frame(0.0, 1, -1.0), _frame(4.0, 0, 1.0), _frame(8.0, 1, -1.0), _frame(10.0, 0, 1.0)]
    d = analyze_contact_trace(trace, dt=0.1)
    assert d["secondary_collision_event"] is True
    assert d["post_separation_secondary_collision_event"] is True
    assert d["distinct_collision_partner_count"] >= 2
    assert "secondary_collision" in d["critical_tags"]
    assert "crowded" in d["critical_tags"]


def test_display_metric_summary_recomputes_secondary_collision_fields():
    trace = [_frame(0.0, 1, -1.0), _frame(4.0, 0, 1.0), _frame(8.0, 1, -1.0), _frame(10.0, 0, 1.0)]
    m = recompute_contact_metric_summary(trace, 0.1)
    assert m["secondary_collision_event"] == 1.0
    assert m["post_separation_secondary_collision_event"] == 1.0
    assert m["distinct_collision_partner_count"] >= 2.0


def test_source_aware_profiles_add_dense_and_secondary_recovery_modes():
    names = {x["name"] for x in _profile_set(source_diag={
        "critical_tags": ["crowded", "secondary_collision", "source_offroad"],
        "nearby_agents_peak_12m": 4,
    })}
    assert "crowded_escape" in names
    assert "secondary_avoidance" in names
    assert "lane_recovery_escape" in names


def test_criticality_rewards_secondary_collision():
    base = {"usable": True, "critical_tags": [], "nearby_agents_peak_12m": 2,
            "multi_actor_conflict_peak": 1, "crowded_fraction": 0.0,
            "offroad_fraction": 0.0, "distinct_collision_partner_count": 1,
            "terminal_clearance_m": 2.0}
    hard = dict(base, secondary_collision_event=True,
                post_separation_secondary_collision_event=True,
                distinct_collision_partner_count=2)
    assert criticality_score(hard) > criticality_score(base)
