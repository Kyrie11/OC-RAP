from tools.contact_clip_metrics import recompute_contact_metric_summary


def _frame(clearance, overlap, penetration, offroad=0.0, speed=2.0, yaw=0.0):
    return {
        "metrics": {
            "min_clearance_m": clearance,
            "overlap": overlap,
            "penetration_depth_m": penetration,
            "offroad": offroad,
            "ego_speed_mps": speed,
            "ego_yaw_rad": yaw,
        }
    }


def test_contact_clip_metrics_use_state_and_interval_support():
    # Four states -> three duration intervals.  The final state owns no
    # following interval, matching closed_loop_runner semantics.
    trace = [
        _frame(-0.5, 1.0, 0.5),
        _frame(-0.2, 1.0, 0.2),
        _frame(0.6, 0.0, 0.0),
        _frame(1.0, 0.0, 0.0),
    ]
    m = recompute_contact_metric_summary(trace, 0.1, {})
    assert m["terminal_clearance_m"] == 1.0
    assert m["clearance_recovery_gain_m"] == 1.5
    assert m["overlap_duration_s"] == 0.2
    assert m["penetration_duration_s"] == 0.2
    assert m["penetration_depth_m_max"] == 0.5
    assert m["offroad_any"] == 0.0
    assert m["post_contact_terminal_clearance_m"] == 1.0
