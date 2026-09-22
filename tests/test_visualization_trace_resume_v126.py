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
