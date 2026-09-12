from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable
import math

ENGINEERING_VERSION = "v48.124.1-OC-FMSA"
SCIENTIFIC_VERSION = "v48.124-OC-FMSA"
ALGORITHM_NAME = "Observation-Consistent Fixed-Main Stability and Non-Interference Adjudication"

STATUS_GO = "FIXED_MAIN_STABILITY_NONINTERFERENCE_GO"
STATUS_DETERMINISM_STOP = "FIXED_MAIN_DETERMINISM_STOP"
STATUS_SAFE_STOP = "FIXED_MAIN_SAFE_NONINTERFERENCE_STOP"
STATUS_NEAR_STOP = "FIXED_MAIN_NEAR_VALIDITY_STOP"
STATUS_CONTACT_STOP = "FIXED_MAIN_CONTACT_VALIDITY_STOP"
STATUS_COVERAGE_STOP = "FIXED_MAIN_COVERAGE_STOP"

GO_NEXT_BRANCH = (
    "freeze_main_and_authorize_external_baseline_comparison_"
    "no_internal_mechanism_or_threshold_search"
)
STOP_NEXT_BRANCH = (
    "keep_recovery_mechanism_family_frozen_and_diagnose_only_failed_stability_or_closed_loop_axis_"
    "no_new_recovery_mechanism_capacity_regime_source_horizon_or_threshold_sweep"
)

# These are not tuned margins. A zero boundary is used deliberately: V48.124 is
# an adjudication of a frozen Main, so statistically supported harm is disallowed.
SAFE_NO_HARM = {
    "overlap_any": "lower",
    "offroad_any": "lower",
    "critical_ttc_exposure_duration_s": "lower",
    "clearance_deficit_auc_m_s": "lower",
    "ttc_deficit_auc_s2": "lower",
    "closed_loop_bounded_NUP": "higher",
    "route_progression_m": "higher",
    "intervention_rate": "lower",
}

NEAR_HARD_NO_HARM = {
    "overlap_any": "lower",
    "offroad_any": "lower",
}
NEAR_BENEFIT = {
    "critical_ttc_exposure_duration_s": "lower",
    "clearance_deficit_auc_m_s": "lower",
    "ttc_deficit_auc_s2": "lower",
    "min_clearance_m_min": "higher",
    "ttc_s_min": "higher",
}

CONTACT_HARD_NO_HARM = {
    "offroad_any": "lower",
    "recontact_event": "lower",
    "secondary_overlap_event": "lower",
    "post_contact_overlap_duration_s": "lower",
}
CONTACT_BENEFIT = {
    "post_contact_clearance_gain_m": "higher",
    "post_contact_free_space_auc_normalized_m": "higher",
    "post_contact_escape_event": "higher",
    "post_contact_terminal_clearance_m": "higher",
    "new_stable_stop_quality_event": "higher",
}

SCIENTIFIC_SCENE_DROP_KEYS = {
    "timing",
    "render_trace",
    "render_context",
    "render_trace_schema",
    "state_xy_trace",
}


def _finite(v: Any) -> float | None:
    try:
        x = float(v)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def _metric_row(report: dict[str, Any], name: str) -> dict[str, Any] | None:
    row = (report.get("metrics") or {}).get(name)
    return row if isinstance(row, dict) else None


def _supported_no_harm(row: dict[str, Any] | None, direction: str, *, atol: float = 1e-12) -> bool:
    if not row:
        return False
    ci = row.get("bootstrap_95ci")
    if not isinstance(ci, (list, tuple)) or len(ci) != 2:
        return False
    lo, hi = _finite(ci[0]), _finite(ci[1])
    if lo is None or hi is None:
        return False
    if direction == "lower":
        return hi <= atol
    if direction == "higher":
        return lo >= -atol
    raise ValueError(direction)


def _supported_benefit(row: dict[str, Any] | None, direction: str, *, atol: float = 1e-12) -> bool:
    if not row:
        return False
    ci = row.get("bootstrap_95ci")
    if not isinstance(ci, (list, tuple)) or len(ci) != 2:
        return False
    lo, hi = _finite(ci[0]), _finite(ci[1])
    if lo is None or hi is None:
        return False
    if direction == "lower":
        return hi < -atol
    if direction == "higher":
        return lo > atol
    raise ValueError(direction)


def _gate_no_harm(report: dict[str, Any], metrics: dict[str, str]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for name, direction in metrics.items():
        row = _metric_row(report, name)
        rows[name] = {
            "direction": direction,
            "available": row is not None,
            "pass": _supported_no_harm(row, direction),
            "paired_delta": None if row is None else row.get("paired_delta"),
            "bootstrap_95ci": None if row is None else row.get("bootstrap_95ci"),
            "n": None if row is None else row.get("n"),
        }
    return {"go": all(r["pass"] for r in rows.values()), "metrics": rows}


def _gate_benefit(report: dict[str, Any], metrics: dict[str, str]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    beneficial: list[str] = []
    for name, direction in metrics.items():
        row = _metric_row(report, name)
        ok = _supported_benefit(row, direction)
        if ok:
            beneficial.append(name)
        rows[name] = {
            "direction": direction,
            "available": row is not None,
            "benefit": ok,
            "paired_delta": None if row is None else row.get("paired_delta"),
            "bootstrap_95ci": None if row is None else row.get("bootstrap_95ci"),
            "n": None if row is None else row.get("n"),
        }
    return {"go": bool(beneficial), "beneficial_metrics": beneficial, "metrics": rows}


def target_keys(result: dict[str, Any]) -> list[str]:
    out = []
    for scene in result.get("scenes") or []:
        key = str(scene.get("target_key") or "")
        if key:
            out.append(key)
    return sorted(set(out))


def _is_standard_validation_source(source: Any) -> bool:
    s = str(source or "").strip().lower().replace("\\", "/").replace("-", "_")
    if not s or "validation_interactive" in s:
        return False
    return "/validation/" in s or "validation@" in s or "validation_tfexample" in s


def coverage_gate(nominal: dict[str, Any], balanced: dict[str, Any], precision: dict[str, Any]) -> dict[str, Any]:
    nk, bk, pk = target_keys(nominal), target_keys(balanced), target_keys(precision)
    expected = int(nominal.get("bucket_target_count") or len(nk))
    complete = all(
        int(x.get("num_scenes") or 0) == int(x.get("bucket_target_count") or -1) > 0
        and bool(x.get("scenes_embedded"))
        for x in (nominal, balanced, precision)
    )
    same = nk == bk == pk and len(nk) == expected
    same_bucket = len({str(x.get("bucket_dataset") or "") for x in (nominal, balanced, precision)}) == 1
    sources = [str(x.get("source") or "") for x in (nominal, balanced, precision)]
    same_source = len(set(sources)) == 1
    standard_validation = all(_is_standard_validation_source(x) for x in sources)
    return {
        "go": bool(complete and same and same_bucket and same_source and standard_validation),
        "complete": bool(complete),
        "same_target_keys": bool(same),
        "same_bucket_dataset": bool(same_bucket),
        "same_womd_source": bool(same_source),
        "standard_validation_source": bool(standard_validation),
        "source": nominal.get("source"),
        "bucket_dataset": nominal.get("bucket_dataset"),
        "num_target_keys": len(nk),
        "bucket_target_count": expected,
    }


def _numeric_close(a: Any, b: Any, atol: float) -> bool:
    fa, fb = _finite(a), _finite(b)
    if fa is not None or fb is not None:
        if fa is None or fb is None:
            return False
        return abs(fa - fb) <= atol
    return a == b


def _compare_tree(a: Any, b: Any, *, path: str, atol: float, differences: list[dict[str, Any]]) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        keys = sorted(set(a) | set(b))
        for k in keys:
            if k in SCIENTIFIC_SCENE_DROP_KEYS:
                continue
            if k not in a or k not in b:
                differences.append({"path": f"{path}.{k}", "a": a.get(k), "b": b.get(k)})
            else:
                _compare_tree(a[k], b[k], path=f"{path}.{k}", atol=atol, differences=differences)
        return
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            differences.append({"path": path, "a_len": len(a), "b_len": len(b)})
            return
        for i, (x, y) in enumerate(zip(a, b)):
            _compare_tree(x, y, path=f"{path}[{i}]", atol=atol, differences=differences)
        return
    if not _numeric_close(a, b, atol):
        differences.append({"path": path, "a": a, "b": b})


def sentinel_determinism(full_result: dict[str, Any], sentinel_result: dict[str, Any], *, atol: float = 1e-9) -> dict[str, Any]:
    full_scenes = {str(s.get("target_key")): s for s in (full_result.get("scenes") or []) if s.get("target_key")}
    sent_scenes = [s for s in (sentinel_result.get("scenes") or []) if s.get("target_key")]
    if len(sent_scenes) != 1:
        return {"go": False, "errors": [f"sentinel_scene_count={len(sent_scenes)}"], "differences": []}
    sent = sent_scenes[0]
    key = str(sent.get("target_key"))
    base = full_scenes.get(key)
    if base is None:
        return {"go": False, "errors": ["sentinel_key_absent_from_full_run"], "target_key": key, "differences": []}
    differences: list[dict[str, Any]] = []
    _compare_tree(base, sent, path="scene", atol=atol, differences=differences)
    return {
        "go": not differences,
        "target_key": key,
        "atol": atol,
        "differences": differences[:50],
        "difference_count": len(differences),
    }


def adjudicate(
    *,
    comparisons: dict[str, dict[str, dict[str, Any]]],
    results: dict[str, dict[str, dict[str, Any]]],
    sentinel_results: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    variants = ("balanced", "precision")
    regimes = ("safe", "near", "contact")

    coverage: dict[str, Any] = {}
    for regime in regimes:
        coverage[regime] = coverage_gate(
            results["nominal"][regime], results["balanced"][regime], results["precision"][regime]
        )
    coverage_go = all(v["go"] for v in coverage.values())

    determinism: dict[str, Any] = {}
    for variant in variants:
        determinism[variant] = {}
        for regime in regimes:
            determinism[variant][regime] = sentinel_determinism(
                results[variant][regime], sentinel_results[variant][regime]
            )
    determinism_go = all(determinism[v][r]["go"] for v in variants for r in regimes)

    safe: dict[str, Any] = {}
    for variant in variants:
        safe[variant] = _gate_no_harm(comparisons[variant]["safe"], SAFE_NO_HARM)
    safe_go = all(safe[v]["go"] for v in variants)

    near: dict[str, Any] = {}
    near_common: set[str] | None = None
    for variant in variants:
        no_harm = _gate_no_harm(comparisons[variant]["near"], NEAR_HARD_NO_HARM)
        benefit = _gate_benefit(comparisons[variant]["near"], NEAR_BENEFIT)
        bset = set(benefit["beneficial_metrics"])
        near_common = bset if near_common is None else near_common & bset
        near[variant] = {"go": bool(no_harm["go"] and benefit["go"]), "hard_no_harm": no_harm, "benefit": benefit}
    near_common = near_common or set()
    near_go = all(near[v]["go"] for v in variants) and bool(near_common)

    contact: dict[str, Any] = {}
    contact_common: set[str] | None = None
    for variant in variants:
        no_harm = _gate_no_harm(comparisons[variant]["contact"], CONTACT_HARD_NO_HARM)
        benefit = _gate_benefit(comparisons[variant]["contact"], CONTACT_BENEFIT)
        bset = set(benefit["beneficial_metrics"])
        contact_common = bset if contact_common is None else contact_common & bset
        contact[variant] = {"go": bool(no_harm["go"] and benefit["go"]), "hard_no_harm": no_harm, "benefit": benefit}
    contact_common = contact_common or set()
    contact_go = all(contact[v]["go"] for v in variants) and bool(contact_common)

    if not coverage_go:
        status = STATUS_COVERAGE_STOP
    elif not determinism_go:
        status = STATUS_DETERMINISM_STOP
    elif not safe_go:
        status = STATUS_SAFE_STOP
    elif not near_go:
        status = STATUS_NEAR_STOP
    elif not contact_go:
        status = STATUS_CONTACT_STOP
    else:
        status = STATUS_GO

    return {
        "status": status,
        "go": status == STATUS_GO,
        "next_branch": GO_NEXT_BRANCH if status == STATUS_GO else STOP_NEXT_BRANCH,
        "coverage_gate": {"go": coverage_go, "regimes": coverage},
        "determinism_gate": {"go": determinism_go, "variants": determinism},
        "safe_noninterference_gate": {"go": safe_go, "variants": safe},
        "near_closed_loop_validity_gate": {
            "go": near_go,
            "common_beneficial_metrics": sorted(near_common),
            "variants": near,
        },
        "contact_closed_loop_validity_gate": {
            "go": contact_go,
            "common_beneficial_metrics": sorted(contact_common),
            "variants": contact,
        },
    }
