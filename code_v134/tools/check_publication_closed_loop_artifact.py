#!/usr/bin/env python3
"""Fail-closed validator for one paper-facing closed-loop artifact.

The validator is intentionally method-agnostic.  It checks the evaluation and
metric contract, exact frozen target set, complete publication-metric coverage,
and (for Contact) exact reproduction of the shared pre-treatment anchor state.
It is used by the submission ablation launcher so stale pre-fix artifacts cannot
be silently reused under the final publication protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

GEOMETRY = "exact_oriented_box_signed_clearance+penetration+swept_sat_constant_velocity_ttc_v55"
DURATION_SUPPORT = "left_endpoint_t0_to_tN_minus_1_no_fictitious_terminal_interval"
COVERAGE_KEYS = (
    "clearance_metric_full_coverage_scene_rate",
    "ttc_metric_full_coverage_scene_rate",
    "overlap_metric_full_coverage_scene_rate",
    "offroad_metric_full_coverage_scene_rate",
)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_one(value: Any) -> bool:
    try:
        return math.isfinite(float(value)) and abs(float(value) - 1.0) <= 1.0e-12
    except Exception:
        return False


def _target_keys(path: Path) -> set[str]:
    doc = _json(path)
    values = doc.get("target_keys") or []
    if not isinstance(values, list):
        raise ValueError(f"target_keys must be a list: {path}")
    keys = {str(x).strip() for x in values if str(x).strip()}
    if not keys:
        raise ValueError(f"empty target lock: {path}")
    return keys


def _journal_path(path: Path) -> Path:
    return Path(str(path) + ".scenes.jsonl")


def _scene_rows(path: Path) -> list[dict[str, Any]]:
    journal = _journal_path(path)
    if not journal.is_file():
        raise FileNotFoundError(f"missing scene journal: {journal}")
    rows: list[dict[str, Any]] = []
    with journal.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            raw = json.loads(line)
            scene = raw.get("scene", raw) if isinstance(raw, dict) else None
            if isinstance(scene, dict):
                rows.append(scene)
    return rows


def _scene_key(scene: dict[str, Any]) -> str:
    key = str(scene.get("target_key") or "").strip()
    if key:
        return key
    sid = str(scene.get("scene_id") or "").strip()
    t = scene.get("target_time_index")
    return f"{sid}:t{t}" if sid and t is not None else sid


def _fail(errors: list[dict[str, Any]], check: str, detail: Any) -> None:
    errors.append({"check": check, "detail": detail})


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate one final-publication closed-loop artifact.")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--regime", choices=("safe", "near", "contact"), required=True)
    ap.add_argument("--target-keys-file", type=Path, required=True)
    ap.add_argument("--metric-semantics-version", default="publication_v55_signed_clearance_unclipped_v1")
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--replan-interval", type=int, default=1)
    ap.add_argument("--metric-dt-s", type=float, default=0.1)
    ap.add_argument("--num-candidates", type=int, default=24)
    ap.add_argument("--num-recovery-options", type=int, default=12)
    ap.add_argument("--womd-role", default="validation")
    ap.add_argument("--contact-anchor-manifest", type=Path, default=None)
    ap.add_argument("--require-latency-contract", default=None)
    ap.add_argument("--require-finite-timing", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    errors: list[dict[str, Any]] = []
    try:
        doc = _json(args.output)
        expected_keys = _target_keys(args.target_keys_file)
        scenes = _scene_rows(args.output)
    except Exception as exc:
        report = {"valid": False, "output": str(args.output), "errors": [{"check": "readable", "detail": str(exc)}]}
        if not args.quiet:
            print(json.dumps(report, ensure_ascii=False))
        return 30

    observed_keys = {_scene_key(s) for s in scenes if _scene_key(s)}
    if observed_keys != expected_keys:
        _fail(errors, "exact_target_set", {
            "expected": len(expected_keys), "observed": len(observed_keys),
            "missing": sorted(expected_keys - observed_keys)[:10],
            "extra": sorted(observed_keys - expected_keys)[:10],
        })
    if int(doc.get("num_scenes") or -1) != len(expected_keys):
        _fail(errors, "num_scenes_matches_lock", {"got": doc.get("num_scenes"), "want": len(expected_keys)})
    if doc.get("metrics_valid") is not True:
        _fail(errors, "metrics_valid", doc.get("empty_reason"))
    if int(doc.get("route_ineligible_target_count") or 0) != 0:
        _fail(errors, "route_ineligible_target_count", doc.get("route_ineligible_target_count"))

    contract = doc.get("evaluation_contract") or {}
    expected_contract = {
        "metric_semantics_version": args.metric_semantics_version,
        "max_steps": int(args.max_steps),
        "replan_interval_steps": int(args.replan_interval),
        "metric_dt_s": float(args.metric_dt_s),
        "num_candidate_prefixes": int(args.num_candidates),
        "num_recovery_options": int(args.num_recovery_options),
        "use_sdc_paths": True,
        "require_observation_legal_route": True,
        "allow_future_route_proxy": False,
        "dataloader_include_sdc_paths": True,
        "allow_logged_sdc_route_fallback": False,
        "compute_future_metrics": False,
        "publication_geometry_metric": GEOMETRY,
        "clearance_is_signed": True,
        "duration_auc_support": DURATION_SUPPORT,
        "womd_source_role": args.womd_role,
    }
    got_contract = {k: contract.get(k) for k in expected_contract}
    if got_contract != expected_contract:
        _fail(errors, "evaluation_contract", {"got": got_contract, "want": expected_contract})

    runtime = doc.get("runtime_contract") or {}
    if runtime.get("publication_geometry_metric") != GEOMETRY:
        _fail(errors, "runtime_publication_geometry_metric", runtime.get("publication_geometry_metric"))
    if runtime.get("publication_metrics_include_target_state_t0") is not True:
        _fail(errors, "publication_metrics_include_target_state_t0", runtime.get("publication_metrics_include_target_state_t0"))

    for key in COVERAGE_KEYS:
        if not _is_one(doc.get(key)):
            _fail(errors, key, doc.get(key))

    timing = doc.get("timing") or {}
    if args.require_latency_contract is not None:
        got = timing.get("execution_contract") if isinstance(timing, dict) else None
        if got != args.require_latency_contract:
            _fail(errors, "latency_execution_contract", {"got": got, "want": args.require_latency_contract})
    if args.require_finite_timing:
        steady = timing.get("steady_state_deployed_planner_s") if isinstance(timing, dict) else None
        finite = False
        if isinstance(steady, dict):
            try:
                finite = math.isfinite(float(steady.get("mean")))
            except Exception:
                finite = False
        if not finite:
            per = timing.get("per_decision_s") if isinstance(timing, dict) else None
            try:
                finite = isinstance(per, dict) and math.isfinite(float(per.get("deployed_planner")))
            except Exception:
                finite = False
        if not finite:
            _fail(errors, "finite_deployed_planner_timing", {"steady": steady, "per_decision_s": timing.get("per_decision_s") if isinstance(timing, dict) else None})

    if args.regime == "contact":
        manifest_path = args.contact_anchor_manifest
        if manifest_path is None or not manifest_path.is_file():
            _fail(errors, "contact_anchor_manifest_exists", str(manifest_path) if manifest_path else None)
        else:
            try:
                manifest = _json(manifest_path)
                manifest_keys = {str(x) for x in (manifest.get("target_keys") or [])}
                manifest_sha = _sha256(manifest_path)
                valid_manifest = (
                    manifest.get("schema") == "ocrap-contact-anchor-manifest-v1"
                    and manifest.get("valid") is True
                    and manifest.get("pre_treatment_policy") == "exact_a0"
                    and int(manifest.get("min_post_steps") or 0) >= int(args.max_steps)
                    and int(manifest.get("num_selected_anchors") or 0) == len(expected_keys)
                    and int(manifest.get("num_selected_scenes") or -1) == len(expected_keys)
                    and manifest_keys == expected_keys
                )
                if not valid_manifest:
                    _fail(errors, "contact_anchor_manifest_contract", {
                        "schema": manifest.get("schema"), "valid": manifest.get("valid"),
                        "pre_treatment_policy": manifest.get("pre_treatment_policy"),
                        "min_post_steps": manifest.get("min_post_steps"),
                        "num_selected_anchors": manifest.get("num_selected_anchors"),
                        "num_selected_scenes": manifest.get("num_selected_scenes"),
                        "manifest_target_count": len(manifest_keys), "expected_target_count": len(expected_keys),
                    })
                if doc.get("contact_anchor_protocol") != "exact_a0_pretreatment_prelude_v1":
                    _fail(errors, "contact_anchor_protocol", doc.get("contact_anchor_protocol"))
                if doc.get("contact_anchor_manifest_sha256") != manifest_sha:
                    _fail(errors, "contact_anchor_manifest_sha256", {"got": doc.get("contact_anchor_manifest_sha256"), "want": manifest_sha})
                if doc.get("contact_anchor_state_fingerprint_required") is not True:
                    _fail(errors, "contact_anchor_state_fingerprint_required", doc.get("contact_anchor_state_fingerprint_required"))
                if not _is_one(doc.get("observed_contact_scene_rate")):
                    _fail(errors, "observed_contact_scene_rate", doc.get("observed_contact_scene_rate"))
                if not _is_one(doc.get("post_contact_metric_eligible_scene_rate")):
                    _fail(errors, "post_contact_metric_eligible_scene_rate", doc.get("post_contact_metric_eligible_scene_rate"))

                anchors = {
                    str(row.get("target_key")): row
                    for row in (manifest.get("anchors") or [])
                    if isinstance(row, dict) and row.get("target_key")
                }
                mismatches: list[dict[str, Any]] = []
                for scene in scenes:
                    key = _scene_key(scene)
                    want = anchors.get(key)
                    if want is None:
                        mismatches.append({"target_key": key, "reason": "absent_from_manifest"})
                        continue
                    bad: dict[str, Any] = {}
                    for field in (
                        "contact_anchor_fingerprint",
                        "contact_anchor_time_index",
                        "contact_anchor_prelude_env_steps",
                    ):
                        if str(scene.get(field)) != str(want.get(field)):
                            bad[field] = {"got": scene.get(field), "want": want.get(field)}
                    if scene.get("contact_anchor_found") is not True:
                        bad["contact_anchor_found"] = {"got": scene.get("contact_anchor_found"), "want": True}
                    if bad:
                        mismatches.append({"target_key": key, "fields": bad})
                if mismatches:
                    _fail(errors, "per_scene_contact_anchor_reproduction", mismatches[:10])
            except Exception as exc:
                _fail(errors, "contact_anchor_manifest_readable", str(exc))

    report = {
        "valid": not errors,
        "output": str(args.output),
        "regime": args.regime,
        "num_targets": len(expected_keys),
        "metric_semantics_version": args.metric_semantics_version,
        "errors": errors,
    }
    if not args.quiet:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not errors else 30


if __name__ == "__main__":
    raise SystemExit(main())
