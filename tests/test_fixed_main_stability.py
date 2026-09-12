from __future__ import annotations

import copy

from ocrap.audits.fixed_main_stability import (
    STATUS_CONTACT_STOP,
    STATUS_COVERAGE_STOP,
    STATUS_DETERMINISM_STOP,
    STATUS_GO,
    STATUS_NEAR_STOP,
    STATUS_SAFE_STOP,
    adjudicate,
    coverage_gate,
    sentinel_determinism,
)


def metric(delta=0.0, lo=0.0, hi=0.0, n=8):
    return {"paired_delta": delta, "bootstrap_95ci": [lo, hi], "n": n}


def report():
    # Zero is exact non-interference. Benefits are made strict only for the
    # preregistered Near/Contact positive-effect metrics.
    names = [
        "overlap_any", "offroad_any", "critical_ttc_exposure_duration_s",
        "clearance_deficit_auc_m_s", "ttc_deficit_auc_s2", "closed_loop_bounded_NUP",
        "route_progression_m", "intervention_rate", "min_clearance_m_min", "ttc_s_min",
        "recontact_event", "secondary_overlap_event", "post_contact_overlap_duration_s",
        "post_contact_clearance_gain_m", "post_contact_free_space_auc_normalized_m",
        "post_contact_escape_event", "post_contact_terminal_clearance_m", "new_stable_stop_quality_event",
    ]
    out = {"metrics": {n: metric() for n in names}, "bootstrap_draws": 5000, "bootstrap_seed": 2027}
    out["metrics"]["min_clearance_m_min"] = metric(0.2, 0.1, 0.3)
    out["metrics"]["post_contact_clearance_gain_m"] = metric(0.3, 0.1, 0.4)
    return out


def result(keys=("a", "b"), source="model", bucket="/bucket"):
    scenes = []
    for i, k in enumerate(keys):
        scenes.append({"target_key": k, "x": float(i), "metric_summary": {"m": float(i)}})
    return {
        "num_scenes": len(keys), "bucket_target_count": len(keys), "scenes_embedded": True,
        "source": source, "bucket_dataset": bucket, "scenes": scenes,
    }




def support(bucket="/bucket", pattern="/womd/validation/validation_tfexample.tfrecord@150", role="validation"):
    return {
        "schema_supports_closed_loop": True,
        "raw_source_role": role,
        "womd_pattern": pattern,
        "dataset": bucket,
        "target_keys_valid": True,
        "num_requested_target_keys": 1,
        "num_matching_requested_target_keys": 1,
    }

def sentinel(full, key="a"):
    s = next(x for x in full["scenes"] if x["target_key"] == key)
    return {"scenes": [copy.deepcopy(s)]}


def fixture():
    results = {v: {r: result(bucket=f"/{r}") for r in ("safe", "near", "contact")} for v in ("nominal", "balanced", "precision")}
    comparisons = {v: {r: report() for r in ("safe", "near", "contact")} for v in ("balanced", "precision")}
    sentinels = {v: {r: sentinel(results[v][r]) for r in ("safe", "near", "contact")} for v in ("balanced", "precision")}
    supports = {v: {r: support(bucket=f"/{r}") for r in ("safe", "near", "contact")} for v in ("nominal", "balanced", "precision")}
    return comparisons, results, sentinels, supports


def test_coverage_requires_same_targets_source_and_bucket():
    n = result(); b = result(); p = result()
    supports = {v: support() for v in ("nominal", "balanced", "precision")}
    assert coverage_gate(n, b, p, support_docs=supports)["go"]
    # result["source"] is a policy/result label, not WOMD provenance.
    p["source"] = "anything"
    assert coverage_gate(n, b, p, support_docs=supports)["go"]
    supports = {v: support(pattern="/womd/validation_interactive/validation_interactive_tfexample.tfrecord@150", role="validation_interactive") for v in ("nominal", "balanced", "precision")}
    gate = coverage_gate(n, b, p, support_docs=supports)
    assert gate["same_womd_source"] and not gate["standard_validation_source"] and not gate["go"]


def test_sentinel_determinism_ignores_timing_but_not_science():
    full = result(keys=("a",))
    full["scenes"][0]["timing"] = {"wall_s": 9.0}
    full["scenes"][0]["metric_summary"]["undefined"] = float("nan")
    sent = sentinel(full)
    sent["scenes"][0]["timing"] = {"wall_s": 1.0}
    assert sentinel_determinism(full, sent)["go"]
    sent["scenes"][0]["metric_summary"]["m"] = 1.0
    assert not sentinel_determinism(full, sent)["go"]


def test_adjudicate_go_and_failure_order():
    c, r, s, u = fixture()
    assert adjudicate(comparisons=c, results=r, sentinel_results=s, support_docs=u)["status"] == STATUS_GO

    c2, r2, s2, u2 = fixture(); u2["precision"]["safe"]["raw_source_role"] = "validation_interactive"
    assert adjudicate(comparisons=c2, results=r2, sentinel_results=s2, support_docs=u2)["status"] == STATUS_COVERAGE_STOP

    c2, r2, s2, u2 = fixture(); s2["balanced"]["safe"]["scenes"][0]["x"] = 99.0
    assert adjudicate(comparisons=c2, results=r2, sentinel_results=s2, support_docs=u2)["status"] == STATUS_DETERMINISM_STOP

    c2, r2, s2, u2 = fixture(); c2["balanced"]["safe"]["metrics"]["overlap_any"] = metric(0.1, 0.05, 0.2)
    assert adjudicate(comparisons=c2, results=r2, sentinel_results=s2, support_docs=u2)["status"] == STATUS_SAFE_STOP

    c2, r2, s2, u2 = fixture();
    for v in ("balanced", "precision"):
        for n in ("critical_ttc_exposure_duration_s", "clearance_deficit_auc_m_s", "ttc_deficit_auc_s2", "min_clearance_m_min", "ttc_s_min"):
            c2[v]["near"]["metrics"][n] = metric()
    assert adjudicate(comparisons=c2, results=r2, sentinel_results=s2, support_docs=u2)["status"] == STATUS_NEAR_STOP

    c2, r2, s2, u2 = fixture();
    for v in ("balanced", "precision"):
        for n in ("post_contact_clearance_gain_m", "post_contact_free_space_auc_normalized_m", "post_contact_escape_event", "post_contact_terminal_clearance_m", "new_stable_stop_quality_event"):
            c2[v]["contact"]["metrics"][n] = metric()
    assert adjudicate(comparisons=c2, results=r2, sentinel_results=s2, support_docs=u2)["status"] == STATUS_CONTACT_STOP


def test_journal_finalizer_preserves_embedded_scene_contract(tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    output = tmp_path / "closed_loop_ocrap.json"
    progress = output.with_suffix(output.suffix + ".progress.json")
    journal = output.with_suffix(output.suffix + ".scenes.jsonl")
    bucket = str(tmp_path / "test_near_contact")
    output.write_text(json.dumps({
        "method": "ocrap",
        "source": "model",
        "bucket_dataset": bucket,
        "bucket_target_count": 2,
        "target_keys_file": "/tmp/keys.json",
        "gamma_rec": 0.2,
        "run_fingerprint": "fp",
    }))
    progress.write_text(json.dumps({
        "status": "complete", "requested_rollouts": 2, "run_fingerprint": "fp"
    }))
    scenes = [
        {"target_key": "a", "scene_id": "s1", "method": "ocrap", "bucket_name": "test_near_contact", "gamma_rec": 0.2, "num_decisions": 1, "num_metric_steps": 1, "metric_summary": {}},
        {"target_key": "b", "scene_id": "s2", "method": "ocrap", "bucket_name": "test_near_contact", "gamma_rec": 0.2, "num_decisions": 1, "num_metric_steps": 1, "metric_summary": {}},
    ]
    journal.write_text("".join(json.dumps({"version": 1, "run_fingerprint": "fp", "resume_key": s["target_key"], "scene": s}) + "\n" for s in scenes))
    env = dict(__import__("os").environ)
    env["PYTHONPATH"] = str(repo / "src") + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run([
        sys.executable, str(repo / "tools/finalize_closed_loop_from_journal.py"),
        "--output", str(output), "--include-scenes-in-result", "--result-scene-detail", "metrics",
    ], cwd=repo, check=True, env=env)
    doc = json.loads(output.read_text())
    assert doc["bucket_dataset"] == bucket
    assert doc["scenes_embedded"] is True
    assert [s["target_key"] for s in doc["scenes"]] == ["a", "b"]
    subprocess.run([
        sys.executable, str(repo / "tools/check_closed_loop_artifact.py"),
        "--output", str(output), "--require-scenes", "--quiet",
    ], cwd=repo, check=True, env=env)


def test_sentinel_builder_and_paired_compare_fallback_to_scene_journal(tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    paths = {}
    for variant in ("nominal", "balanced", "precision"):
        for regime in ("safe", "near", "contact"):
            p = tmp_path / variant / regime / ("closed_loop_nominal.json" if variant == "nominal" else "closed_loop_ocrap.json")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"num_scenes": 2, "bucket_target_count": 2, "scenes_embedded": False}))
            scenes = [
                {"target_key": f"{regime}:a", "method": "nominal" if variant == "nominal" else "ocrap", "num_decisions": 1, "num_metric_steps": 1, "metric_summary": {"overlap_any": 0.0}},
                {"target_key": f"{regime}:b", "method": "nominal" if variant == "nominal" else "ocrap", "num_decisions": 1, "num_metric_steps": 1, "metric_summary": {"overlap_any": 0.0}},
            ]
            p.with_suffix(p.suffix + ".scenes.jsonl").write_text("".join(json.dumps({"version":1,"run_fingerprint":"fp","resume_key":s["target_key"],"scene":s})+"\n" for s in scenes))
            paths[(variant, regime)] = p
    index = tmp_path / "sentinel-index.json"
    key_dir = tmp_path / "keys"
    cmd = [sys.executable, str(repo / "tools/build_fixed_main_sentinel_keys.py")]
    for regime in ("safe", "near", "contact"):
        for variant in ("nominal", "balanced", "precision"):
            cmd += [f"--{variant}-{regime}", str(paths[(variant, regime)])]
    cmd += ["--key-dir", str(key_dir), "--output", str(index)]
    env = dict(__import__("os").environ)
    env["PYTHONPATH"] = str(repo / "src") + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run(cmd, cwd=repo, check=True, env=env)
    idx = json.loads(index.read_text())
    assert idx["valid"] is True
    assert idx["regimes"]["near"]["num_balanced"] == 2
    assert idx["regimes"]["near"]["scene_sources"]["balanced"] == "journal"

    comparison = tmp_path / "compare.json"
    subprocess.run([
        sys.executable, str(repo / "tools/compare_paired_closed_loop.py"),
        str(paths[("nominal", "safe")]), str(paths[("balanced", "safe")]),
        "--bootstrap", "20", "--seed", "2027", "--output", str(comparison),
    ], cwd=repo, check=True, env=env)
    comp = json.loads(comparison.read_text())
    assert comp["num_paired_scenes"] == 2
    assert comp["control_scene_source"] == "journal"
    assert comp["method_scene_source"] == "journal"


def test_v48124_launcher_enforces_rifa_absolute_admission_before_intervention():
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    text = (repo / "scripts/run_ocrap_closed_loop.sh").read_text(encoding="utf-8")
    assert "selection.require_absolute_admission_for_intervention=true" in text
