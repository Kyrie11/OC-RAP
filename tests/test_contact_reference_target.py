import copy

from tools.synthesize_contact_reference import _generate_profile, _profile_set


def _agent(x, y, yaw=0.0, *, sdc=False):
    return {
        "x": float(x), "y": float(y), "yaw": float(yaw),
        "length": 4.8, "width": 2.0, "is_sdc": bool(sdc),
        "object_index": 0 if sdc else 1,
    }


def _scene():
    trace = []
    # Empirical nominal path moves straight; another actor sits close enough to
    # create a Contact-style initial overlap.  The test validates provenance and
    # metric consistency rather than requiring a specific optimized path.
    for i in range(16):
        x = i * 0.35
        sdc = _agent(x, 0.0, sdc=True)
        other = _agent(1.2, 0.0, sdc=False)
        trace.append({
            "time_index": 20 + i,
            "agents": [sdc, other],
            "metrics": {
                "ego_speed_mps": 3.5,
                "ego_yaw_rad": 0.0,
                "min_clearance_m": -0.5 if i < 4 else 0.5,
                "signed_clearance_m": -0.5 if i < 4 else 0.5,
                "overlap": 1.0 if i < 4 else 0.0,
                "penetration_depth_m": 0.5 if i < 4 else 0.0,
                "offroad": 0.0,
                "ttc_s": 0.0 if i < 4 else 100.0,
            },
            "selected_candidate_index": 0,
            "selected_macro": "nominal",
            "selection_reason": "fixture",
        })
    # Simple straight vehicle-lane centerline.
    ctx = {"roadgraph_polylines": [{"id": 1, "type": 1, "xy": [[-10.0, 0.0], [30.0, 0.0]]}]}
    return {
        "target_key": "test_contact:fixture:t0",
        "scene_id": "fixture",
        "target_time_index": 0,
        "contact_anchor_time_index": 20,
        "render_trace": trace,
        "render_context": ctx,
        "metric_summary": {},
        "method": "ocrap",
    }


def test_reference_synthesis_preserves_t0_and_other_agents_and_recomputes_metrics():
    src = _scene()
    original = copy.deepcopy(src)
    out, quality = _generate_profile(src, 0.1, _profile_set()[0])
    assert out["reference_trajectory"] is True
    assert out["method"] == "ocrap_reference"
    # exact-a0 state is untouched
    assert out["render_trace"][0]["agents"] == original["render_trace"][0]["agents"]
    # non-SDC actors remain the recorded actors at every step
    for got, exp in zip(out["render_trace"], original["render_trace"]):
        got_other = [a for a in got["agents"] if not a.get("is_sdc")]
        exp_other = [a for a in exp["agents"] if not a.get("is_sdc")]
        assert got_other == exp_other
    # panel metrics are recomputed from the displayed generated states
    terminal_frame = out["render_trace"][-1]
    assert out["metric_summary"]["terminal_clearance_m"] == terminal_frame["metrics"]["min_clearance_m"]
    assert "deviation_max_m" in quality
