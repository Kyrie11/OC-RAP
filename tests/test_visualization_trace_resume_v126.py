from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _scene(key: str, start: int, frames: int, *, with_trace: bool = True):
    scene = {"target_key": key, "target_time_index": start}
    scene["render_trace"] = ([{"time_index": start + i} for i in range(frames)] if with_trace else [])
    return scene


def test_prepare_selected_trace_reruns_keeps_valid_and_cleans_metrics_only(tmp_path: Path) -> None:
    sel = tmp_path / "selection"; trace = tmp_path / "traces"
    sel.mkdir(); (trace / "ocrap" / "safe").mkdir(parents=True); (trace / "external" / "safe").mkdir(parents=True)
    key = "test_safe:scene:t10"
    selection = {
        "regime": "safe", "metric_dt_s": 0.1, "selected_clip_duration_s": 6.0,
        "selected": [{"target_key": key, "target_time_index": 10, "clip_duration_s": 6.0}],
    }
    (sel / "safe_selection.json").write_text(json.dumps(selection)); (sel / "safe_target_keys.json").write_text(json.dumps({"target_keys": [key]}))
    # Minimal placeholder selections for the other regimes; no artifacts exist there.
    for r in ("near", "contact"):
        item = {"target_key": f"test_{r}:scene:t10", "target_time_index": 10, "clip_duration_s": 6.0}
        if r == "contact": item["contact_anchor_time_index"] = 20
        (sel / f"{r}_selection.json").write_text(json.dumps({"regime": r, "metric_dt_s": 0.1, "selected_clip_duration_s": 6.0, "selected": [item]}))
        (sel / f"{r}_target_keys.json").write_text(json.dumps({"target_keys": [item["target_key"]]}))

    bad = trace / "ocrap" / "safe" / "closed_loop_ocrap.json"
    bad.write_text("{}")
    Path(str(bad)+".scenes.jsonl").write_text(json.dumps({"scene": _scene(key, 10, 0, with_trace=False)})+"\n")
    good = trace / "external" / "safe" / "closed_loop_gameformer_lite.json"
    good.write_text("{}")
    Path(str(good)+".scenes.jsonl").write_text(json.dumps({"scene": _scene(key, 10, 61)})+"\n")
    out = trace / "TRACE_PREP.json"
    subprocess.run([sys.executable, str(ROOT / "tools/prepare_selected_trace_reruns.py"), "--trace-root", str(trace), "--selection-root", str(sel), "--clean-invalid", "--output", str(out)], check=True)
    doc = json.loads(out.read_text())
    rows = {(x["regime"], x["method"]): x for x in doc["methods"]}
    assert rows[("safe", "ocrap")]["status"] == "invalid_cleaned"
    assert not bad.exists() and not Path(str(bad)+".scenes.jsonl").exists()
    assert rows[("safe", "gameformer_lite")]["status"] == "reusable"
    assert good.exists() and Path(str(good)+".scenes.jsonl").exists()


def test_near_visualization_freezes_stored_conformal_intervals_instead_of_rebinding_to_trace_horizon() -> None:
    text = (ROOT / "scripts/generate_selected_regime_traces.sh").read_text()
    assert "near_cpsf_visualization_calibration_freeze_v126" in text
    assert 'CONFORMAL_INTERVALS="$NEAR_CONFORMAL_INTERVALS"' in text
    assert 'CONFORMAL_MISSION_HORIZON="$NEAR_CONFORMAL_MISSION_HORIZON"' in text
    assert 'CL_MAX_STEPS="$TRACE_MAX_STEPS"' in text


def test_ocrap_three_regime_render_defaults_to_full_journal() -> None:
    text = (ROOT / "scripts/run_ocrap_three_regime_evaluation.sh").read_text()
    assert 'runtime_bool_true "$RENDER_SAFE" || runtime_bool_true "$RENDER_NEAR" || runtime_bool_true "$RENDER_CONTACT"' in text
    assert ': "${SCENE_JOURNAL_DETAIL:=full}"' in text


def test_prepare_selected_trace_reruns_normalizes_only_equivalent_duplicate_rows(tmp_path: Path) -> None:
    sel = tmp_path / "selection"; trace = tmp_path / "traces"
    sel.mkdir()
    for regime in ("safe", "near", "contact"):
        item = {"target_key": f"test_{regime}:scene:t10", "target_time_index": 10, "clip_duration_s": 6.0}
        if regime == "contact":
            item["contact_anchor_time_index"] = 10
        (sel / f"{regime}_selection.json").write_text(json.dumps({"regime": regime, "metric_dt_s": 0.1, "selected_clip_duration_s": 6.0, "selected": [item]}))
        (sel / f"{regime}_target_keys.json").write_text(json.dumps({"target_keys": [item["target_key"]]}))

    # Populate every required family with one valid row so only the duplicate
    # behavior under test can affect the preparation result.
    methods_by_regime = {
        "safe": ["gameformer_lite", "plantf", "pluto", "pdm_closed", "pdm_hybrid", "idm", "diffusion_planner"],
        "near": ["marc_lite", "racp_lite", "robust_scenario_mpc", "predictive_safety_filter", "dr_cvar_safety_filter", "conformal_predictive_safety_filter", "flow_planner", "plan_r1", "betopnet"],
        "contact": ["postimpact_mpc_lite", "post_crash_braking", "postimpact_motion_tvlqr", "post_collision_restoration", "compensatory_postimpact_mpc", "robust_postimpact_control"],
    }
    for regime, methods in methods_by_regime.items():
        families = [("ocrap", trace / "ocrap" / regime / "closed_loop_ocrap.json")]
        families += [(m, trace / "external" / regime / f"closed_loop_{m}.json") for m in methods]
        key = f"test_{regime}:scene:t10"
        for method, base in families:
            base.parent.mkdir(parents=True, exist_ok=True); base.write_text("{}")
            scene = _scene(key, 10, 61); scene["method"] = method; scene["timing"] = {"wall_s": 1.0}
            env = {"version": 1, "run_fingerprint": "fp", "resume_key": f"target:{key}", "scene": scene}
            Path(str(base)+".scenes.jsonl").write_text(json.dumps(env)+"\n")

    dup_path = trace / "external" / "safe" / "closed_loop_gameformer_lite.json.scenes.jsonl"
    first = json.loads(dup_path.read_text())
    second = json.loads(json.dumps(first)); second["scene"]["timing"] = {"wall_s": 2.0}
    dup_path.write_text(json.dumps(first)+"\n"+json.dumps(second)+"\n")

    out = trace / "TRACE_PREP.json"
    subprocess.run([sys.executable, str(ROOT / "tools/prepare_selected_trace_reruns.py"), "--trace-root", str(trace), "--selection-root", str(sel), "--clean-invalid", "--output", str(out)], check=True)
    doc = json.loads(out.read_text())
    row = next(x for x in doc["methods"] if x["regime"] == "safe" and x["method"] == "gameformer_lite")
    assert row["valid_reusable"] is True
    assert row["normalized_equivalent_duplicates"] is True
    assert row["equivalent_duplicate_keys"] == ["test_safe:scene:t10"]
    assert len([x for x in dup_path.read_text().splitlines() if x.strip()]) == 1


def test_prepare_selected_trace_reruns_rejects_conflicting_duplicate_rows(tmp_path: Path) -> None:
    sel = tmp_path / "selection"; trace = tmp_path / "traces"
    sel.mkdir(); (trace / "external" / "safe").mkdir(parents=True)
    key = "test_safe:scene:t10"
    (sel / "safe_selection.json").write_text(json.dumps({"regime":"safe","metric_dt_s":0.1,"selected_clip_duration_s":6.0,"selected":[{"target_key":key,"target_time_index":10,"clip_duration_s":6.0}]}))
    (sel / "safe_target_keys.json").write_text(json.dumps({"target_keys":[key]}))
    for r in ("near", "contact"):
        item={"target_key":f"test_{r}:scene:t10","target_time_index":10,"clip_duration_s":6.0}
        if r=="contact": item["contact_anchor_time_index"]=10
        (sel/f"{r}_selection.json").write_text(json.dumps({"regime":r,"metric_dt_s":0.1,"selected_clip_duration_s":6.0,"selected":[item]}))
        (sel/f"{r}_target_keys.json").write_text(json.dumps({"target_keys":[item["target_key"]]}))
    base=trace/"external"/"safe"/"closed_loop_gameformer_lite.json"; base.write_text("{}")
    a=_scene(key,10,61); b=_scene(key,10,61); b["render_trace"][1]["time_index"] = 999
    env1={"version":1,"run_fingerprint":"fp","resume_key":f"target:{key}","scene":a}
    env2={"version":1,"run_fingerprint":"fp","resume_key":f"target:{key}","scene":b}
    Path(str(base)+".scenes.jsonl").write_text(json.dumps(env1)+"\n"+json.dumps(env2)+"\n")
    out=trace/"TRACE_PREP.json"
    subprocess.run([sys.executable,str(ROOT/"tools/prepare_selected_trace_reruns.py"),"--trace-root",str(trace),"--selection-root",str(sel),"--clean-invalid","--output",str(out)],check=True)
    doc=json.loads(out.read_text()); row=next(x for x in doc["methods"] if x["regime"]=="safe" and x["method"]=="gameformer_lite")
    assert row["valid_reusable"] is False
    assert any("conflicting duplicate" in e for e in row["errors"])
    assert row["status"] == "invalid_cleaned"
