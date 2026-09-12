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


def result(keys=("a", "b"), source="womd/validation@150", bucket="/bucket"):
    scenes = []
    for i, k in enumerate(keys):
        scenes.append({"target_key": k, "x": float(i), "metric_summary": {"m": float(i)}})
    return {
        "num_scenes": len(keys), "bucket_target_count": len(keys), "scenes_embedded": True,
        "source": source, "bucket_dataset": bucket, "scenes": scenes,
    }


def sentinel(full, key="a"):
    s = next(x for x in full["scenes"] if x["target_key"] == key)
    return {"scenes": [copy.deepcopy(s)]}


def fixture():
    results = {v: {r: result(bucket=f"/{r}") for r in ("safe", "near", "contact")} for v in ("nominal", "balanced", "precision")}
    comparisons = {v: {r: report() for r in ("safe", "near", "contact")} for v in ("balanced", "precision")}
    sentinels = {v: {r: sentinel(results[v][r]) for r in ("safe", "near", "contact")} for v in ("balanced", "precision")}
    return comparisons, results, sentinels


def test_coverage_requires_same_targets_source_and_bucket():
    n = result(); b = result(); p = result()
    assert coverage_gate(n, b, p)["go"]
    p["source"] = "womd/interactive@150"
    assert not coverage_gate(n, b, p)["go"]


def test_sentinel_determinism_ignores_timing_but_not_science():
    full = result(keys=("a",))
    full["scenes"][0]["timing"] = {"wall_s": 9.0}
    sent = sentinel(full)
    sent["scenes"][0]["timing"] = {"wall_s": 1.0}
    assert sentinel_determinism(full, sent)["go"]
    sent["scenes"][0]["metric_summary"]["m"] = 1.0
    assert not sentinel_determinism(full, sent)["go"]


def test_adjudicate_go_and_failure_order():
    c, r, s = fixture()
    assert adjudicate(comparisons=c, results=r, sentinel_results=s)["status"] == STATUS_GO

    c2, r2, s2 = fixture(); r2["precision"]["safe"]["source"] = "wrong"
    assert adjudicate(comparisons=c2, results=r2, sentinel_results=s2)["status"] == STATUS_COVERAGE_STOP

    c2, r2, s2 = fixture(); s2["balanced"]["safe"]["scenes"][0]["x"] = 99.0
    assert adjudicate(comparisons=c2, results=r2, sentinel_results=s2)["status"] == STATUS_DETERMINISM_STOP

    c2, r2, s2 = fixture(); c2["balanced"]["safe"]["metrics"]["overlap_any"] = metric(0.1, 0.05, 0.2)
    assert adjudicate(comparisons=c2, results=r2, sentinel_results=s2)["status"] == STATUS_SAFE_STOP

    c2, r2, s2 = fixture();
    for v in ("balanced", "precision"):
        for n in ("critical_ttc_exposure_duration_s", "clearance_deficit_auc_m_s", "ttc_deficit_auc_s2", "min_clearance_m_min", "ttc_s_min"):
            c2[v]["near"]["metrics"][n] = metric()
    assert adjudicate(comparisons=c2, results=r2, sentinel_results=s2)["status"] == STATUS_NEAR_STOP

    c2, r2, s2 = fixture();
    for v in ("balanced", "precision"):
        for n in ("post_contact_clearance_gain_m", "post_contact_free_space_auc_normalized_m", "post_contact_escape_event", "post_contact_terminal_clearance_m", "new_stable_stop_quality_event"):
            c2[v]["contact"]["metrics"][n] = metric()
    assert adjudicate(comparisons=c2, results=r2, sentinel_results=s2)["status"] == STATUS_CONTACT_STOP
