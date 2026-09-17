#!/usr/bin/env python3
"""Fail-closed audit of the final OC-RAP/external-baseline publication artifacts.

This does not recompute planner metrics.  It proves that already-produced
artifacts were evaluated on the same frozen targets and publication metric
contract, and that Contact starts from the same pre-treatment observed-contact
state for every method.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

REGIMES = ("safe", "near", "contact")
OWN_VARIANTS = ("balanced", "precision")
MAIN_EXTERNAL = {
    "safe": (
        "gameformer_lite", "plantf", "pluto", "pdm_closed", "pdm_hybrid", "idm",
    ),
    "near": (
        "marc_lite", "racp_lite", "robust_scenario_mpc", "predictive_safety_filter",
        "dr_cvar_safety_filter", "conformal_predictive_safety_filter",
    ),
    "contact": (
        "postimpact_mpc_lite", "post_crash_braking", "postimpact_motion_tvlqr",
        "post_collision_restoration", "compensatory_postimpact_mpc", "robust_postimpact_control",
    ),
}
SUPPLEMENTARY_EXTERNAL = {
    "safe": ("diffusion_planner",),
    "near": ("flow_planner", "plan_r1", "betopnet"),
    "contact": (),
}
PUBLICATION_METRIC_SEMANTICS_VERSION = "publication_v55_signed_clearance_unclipped_v1"
CONTRACT_KEYS = (
    "metric_semantics_version", "max_steps", "replan_interval_steps", "metric_dt_s",
    "num_candidate_prefixes", "num_recovery_options", "use_sdc_paths",
    "require_observation_legal_route", "allow_future_route_proxy",
    "dataloader_include_sdc_paths", "allow_logged_sdc_route_fallback",
    "compute_future_metrics", "publication_geometry_metric", "clearance_is_signed",
    "duration_auc_support", "womd_source_role",
)
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


def _target_keys_from_lock(path: Path) -> set[str]:
    doc = _json(path)
    values = doc.get("target_keys") or []
    if not isinstance(values, list):
        raise ValueError(f"target lock target_keys must be a list: {path}")
    keys = {str(x).strip() for x in values if str(x).strip()}
    if not keys:
        raise ValueError(f"target lock contains no target keys: {path}")
    return keys


def _journal(path: Path) -> Path:
    return Path(str(path) + ".scenes.jsonl")


def _scene_rows(path: Path) -> list[dict[str, Any]]:
    doc = _json(path)
    embedded = doc.get("scenes")
    if isinstance(embedded, list) and embedded:
        return [x for x in embedded if isinstance(x, dict)]
    j = _journal(path)
    if not j.is_file():
        raise FileNotFoundError(f"missing scene journal: {j}")
    rows: list[dict[str, Any]] = []
    with j.open(encoding="utf-8") as f:
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


def _contract(doc: dict[str, Any]) -> dict[str, Any]:
    raw = doc.get("evaluation_contract") or {}
    return {k: raw.get(k) for k in CONTRACT_KEYS} if isinstance(raw, dict) else {k: None for k in CONTRACT_KEYS}


def _is_one(value: Any) -> bool:
    try:
        return math.isfinite(float(value)) and abs(float(value) - 1.0) <= 1.0e-12
    except Exception:
        return False


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _record(checks: list[dict[str, Any]], *, ok: bool, scope: str, check: str, detail: Any = None) -> None:
    row: dict[str, Any] = {"ok": bool(ok), "scope": scope, "check": check}
    if detail is not None:
        row["detail"] = detail
    checks.append(row)


def _audit_artifact(
    path: Path,
    *,
    label: str,
    regime: str,
    target_keys: set[str],
    checks: list[dict[str, Any]],
    contact_manifest: dict[str, Any] | None,
    contact_manifest_sha: str | None,
    latency: bool,
) -> tuple[dict[str, Any] | None, set[str] | None]:
    scope = f"{label}:{regime}:{'latency' if latency else 'accuracy'}"
    if not path.is_file():
        _record(checks, ok=False, scope=scope, check="artifact_exists", detail=str(path))
        return None, None
    _record(checks, ok=True, scope=scope, check="artifact_exists", detail=str(path))
    try:
        doc = _json(path)
        scenes = _scene_rows(path)
    except Exception as exc:
        _record(checks, ok=False, scope=scope, check="artifact_readable", detail=str(exc))
        return None, None
    keys = {_scene_key(s) for s in scenes if _scene_key(s)}
    _record(
        checks,
        ok=keys == target_keys,
        scope=scope,
        check="exact_frozen_target_set",
        detail={"expected": len(target_keys), "observed": len(keys), "missing": sorted(target_keys - keys)[:10], "extra": sorted(keys - target_keys)[:10]},
    )
    _record(checks, ok=int(doc.get("num_scenes") or -1) == len(target_keys), scope=scope, check="num_scenes_matches_lock", detail=doc.get("num_scenes"))
    _record(checks, ok=bool(doc.get("metrics_valid")), scope=scope, check="metrics_valid", detail=doc.get("empty_reason"))
    _record(checks, ok=int(doc.get("route_ineligible_target_count") or 0) == 0, scope=scope, check="no_route_ineligible_targets", detail=doc.get("route_ineligible_target_count"))

    contract = _contract(doc)
    _record(
        checks,
        ok=contract.get("metric_semantics_version") == PUBLICATION_METRIC_SEMANTICS_VERSION,
        scope=scope,
        check="publication_metric_semantics",
        detail=contract.get("metric_semantics_version"),
    )
    _record(checks, ok=contract.get("clearance_is_signed") is True, scope=scope, check="signed_clearance_contract", detail=contract.get("clearance_is_signed"))
    for key in COVERAGE_KEYS:
        _record(checks, ok=_is_one(doc.get(key)), scope=scope, check=key, detail=doc.get(key))

    if latency:
        timing = doc.get("timing") or {}
        execution_contract = timing.get("execution_contract") if isinstance(timing, dict) else None
        _record(
            checks,
            ok=execution_contract == "isolated_single_process_single_gpu",
            scope=scope,
            check="isolated_latency_execution_contract",
            detail=execution_contract,
        )
        # Either top-level decision_latency_ms or steady-state timing must be finite.
        steady = timing.get("steady_state_deployed_planner_s") if isinstance(timing, dict) else None
        _record(checks, ok=_finite(doc.get("decision_latency_ms")) or _finite(steady), scope=scope, check="finite_latency_measurement", detail={"decision_latency_ms": doc.get("decision_latency_ms"), "steady_state_s": steady})

    if regime == "contact":
        _record(checks, ok=doc.get("contact_anchor_protocol") == "exact_a0_pretreatment_prelude_v1", scope=scope, check="contact_anchor_protocol", detail=doc.get("contact_anchor_protocol"))
        _record(checks, ok=doc.get("contact_anchor_state_fingerprint_required") is True, scope=scope, check="contact_anchor_fingerprint_required", detail=doc.get("contact_anchor_state_fingerprint_required"))
        _record(checks, ok=bool(contact_manifest_sha) and doc.get("contact_anchor_manifest_sha256") == contact_manifest_sha, scope=scope, check="shared_contact_anchor_manifest_sha256", detail=doc.get("contact_anchor_manifest_sha256"))
        _record(checks, ok=_is_one(doc.get("observed_contact_scene_rate")), scope=scope, check="observed_contact_scene_rate_is_one", detail=doc.get("observed_contact_scene_rate"))
        _record(checks, ok=_is_one(doc.get("post_contact_metric_eligible_scene_rate")), scope=scope, check="post_contact_metric_eligibility_is_one", detail=doc.get("post_contact_metric_eligible_scene_rate"))
        if contact_manifest is not None:
            anchors = {str(x.get("target_key")): x for x in (contact_manifest.get("anchors") or []) if isinstance(x, dict)}
            mismatches: list[dict[str, Any]] = []
            for scene in scenes:
                key = _scene_key(scene)
                want = anchors.get(key)
                if want is None:
                    mismatches.append({"target_key": key, "reason": "absent_from_manifest"})
                    continue
                fields = {
                    "contact_anchor_fingerprint": (scene.get("contact_anchor_fingerprint"), want.get("contact_anchor_fingerprint")),
                    "contact_anchor_time_index": (scene.get("contact_anchor_time_index"), want.get("contact_anchor_time_index")),
                    "contact_anchor_prelude_env_steps": (scene.get("contact_anchor_prelude_env_steps"), want.get("contact_anchor_prelude_env_steps")),
                }
                bad = {name: {"got": got, "want": expected} for name, (got, expected) in fields.items() if str(got) != str(expected)}
                if scene.get("contact_anchor_found") is not True:
                    bad["contact_anchor_found"] = {"got": scene.get("contact_anchor_found"), "want": True}
                if bad:
                    mismatches.append({"target_key": key, "fields": bad})
            _record(checks, ok=not mismatches and len(scenes) == len(target_keys), scope=scope, check="per_scene_contact_anchor_fingerprint_matches_manifest", detail=mismatches[:10] if mismatches else {"verified": len(scenes)})
    return doc, keys


def main() -> int:
    base_out = Path(os.environ.get("BASE_OUT", "/home/senzeyu2/code/OC-RAP/runs"))
    default_ocrap = Path(os.environ.get("OCRAP_FINAL_CHARACTERIZATION_OUT", str(base_out / "ocrap_v48_124_final_characterization")))
    default_external = Path(os.environ.get("FINAL_EXTERNAL_BASELINE_OUT", str(base_out / "external_baselines_v48_124_final_v2")))
    ap = argparse.ArgumentParser(description="Audit final publication metric fairness across OC-RAP and all external baselines.")
    ap.add_argument("--ocrap-root", type=Path, default=default_ocrap)
    ap.add_argument("--external-root", type=Path, default=default_external)
    ap.add_argument("--ocrap-latency-root", type=Path, default=None)
    ap.add_argument("--external-latency-root", type=Path, default=None)
    ap.add_argument("--main-only", action="store_true", help="Audit only main-table external baselines, not Safe/Near supplementary ports.")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    ocrap_latency = args.ocrap_latency_root or (args.ocrap_root / "latency_isolated")
    external_latency = args.external_latency_root or Path(str(args.external_root) + "_latency_isolated")

    checks: list[dict[str, Any]] = []
    target_by_regime: dict[str, set[str]] = {}
    for regime in REGIMES:
        lock = args.ocrap_root / "target_keys" / f"{regime}.json"
        try:
            target_by_regime[regime] = _target_keys_from_lock(lock)
            _record(checks, ok=True, scope=f"lock:{regime}", check="target_lock_readable", detail={"path": str(lock), "num_targets": len(target_by_regime[regime])})
        except Exception as exc:
            _record(checks, ok=False, scope=f"lock:{regime}", check="target_lock_readable", detail=str(exc))
            target_by_regime[regime] = set()

    manifest_path = args.ocrap_root / "contact_anchor" / "contact_anchor_manifest.json"
    contact_manifest: dict[str, Any] | None = None
    contact_manifest_sha: str | None = None
    if manifest_path.is_file():
        try:
            contact_manifest = _json(manifest_path)
            contact_manifest_sha = _sha256(manifest_path)
            contact_targets = target_by_regime.get("contact", set())
            manifest_targets = {str(x) for x in (contact_manifest.get("target_keys") or [])}
            valid = (
                contact_manifest.get("schema") == "ocrap-contact-anchor-manifest-v1"
                and contact_manifest.get("valid") is True
                and manifest_targets == contact_targets
                and int(contact_manifest.get("num_selected_anchors") or 0) == len(contact_targets)
                and int(contact_manifest.get("num_selected_scenes") or -1) == len(contact_targets)
                and int(contact_manifest.get("min_post_steps") or 0) >= 40
                and contact_manifest.get("pre_treatment_policy") == "exact_a0"
            )
            _record(checks, ok=valid, scope="contact_manifest", check="valid_scene_disjoint_full_horizon_anchor_manifest", detail={"path": str(manifest_path), "sha256": contact_manifest_sha, "min_post_steps": contact_manifest.get("min_post_steps"), "num_targets": len(manifest_targets)})
        except Exception as exc:
            _record(checks, ok=False, scope="contact_manifest", check="valid_scene_disjoint_full_horizon_anchor_manifest", detail=str(exc))
    else:
        _record(checks, ok=False, scope="contact_manifest", check="manifest_exists", detail=str(manifest_path))

    accuracy_docs: dict[str, dict[str, dict[str, Any]]] = {r: {} for r in REGIMES}
    accuracy_keys: dict[tuple[str, str], set[str] | None] = {}
    # Proposed method variants.
    for variant in OWN_VARIANTS:
        label = f"ocrap_{variant}"
        for regime in REGIMES:
            path = args.ocrap_root / "ocrap" / variant / regime / "closed_loop_ocrap.json"
            doc, keys = _audit_artifact(path, label=label, regime=regime, target_keys=target_by_regime[regime], checks=checks, contact_manifest=contact_manifest, contact_manifest_sha=contact_manifest_sha, latency=False)
            if doc is not None:
                accuracy_docs[regime][label] = doc
            accuracy_keys[(label, regime)] = keys
            lp = ocrap_latency / "ocrap" / variant / regime / "closed_loop_ocrap.json"
            ldoc, lkeys = _audit_artifact(lp, label=label, regime=regime, target_keys=target_by_regime[regime], checks=checks, contact_manifest=contact_manifest, contact_manifest_sha=contact_manifest_sha, latency=True)
            if doc is not None and ldoc is not None:
                _record(checks, ok=_contract(doc) == _contract(ldoc), scope=f"{label}:{regime}", check="accuracy_latency_evaluation_contract_equal", detail=None if _contract(doc) == _contract(ldoc) else {"accuracy": _contract(doc), "latency": _contract(ldoc)})
                _record(checks, ok=keys == lkeys == target_by_regime[regime], scope=f"{label}:{regime}", check="accuracy_latency_target_set_equal")

    # External methods.
    for regime in REGIMES:
        methods = list(MAIN_EXTERNAL[regime])
        if not args.main_only:
            methods.extend(SUPPLEMENTARY_EXTERNAL[regime])
        for method in methods:
            path = args.external_root / regime / f"closed_loop_{method}.json"
            doc, keys = _audit_artifact(path, label=method, regime=regime, target_keys=target_by_regime[regime], checks=checks, contact_manifest=contact_manifest, contact_manifest_sha=contact_manifest_sha, latency=False)
            if doc is not None:
                accuracy_docs[regime][method] = doc
            accuracy_keys[(method, regime)] = keys
            lp = external_latency / regime / f"closed_loop_{method}.json"
            ldoc, lkeys = _audit_artifact(lp, label=method, regime=regime, target_keys=target_by_regime[regime], checks=checks, contact_manifest=contact_manifest, contact_manifest_sha=contact_manifest_sha, latency=True)
            if doc is not None and ldoc is not None:
                _record(checks, ok=_contract(doc) == _contract(ldoc), scope=f"{method}:{regime}", check="accuracy_latency_evaluation_contract_equal", detail=None if _contract(doc) == _contract(ldoc) else {"accuracy": _contract(doc), "latency": _contract(ldoc)})
                _record(checks, ok=keys == lkeys == target_by_regime[regime], scope=f"{method}:{regime}", check="accuracy_latency_target_set_equal")

    # Strongest cross-method condition: one identical publication contract in each regime.
    for regime, docs in accuracy_docs.items():
        sigs = {label: _contract(doc) for label, doc in docs.items()}
        values = list(sigs.values())
        equal = bool(values) and all(v == values[0] for v in values[1:])
        _record(checks, ok=equal, scope=f"cross_method:{regime}", check="identical_publication_evaluation_contract", detail=None if equal else sigs)

    # Nominal is an audit/control, not an external baseline; check it when present.
    for regime in REGIMES:
        p = args.ocrap_root / "nominal" / regime / "closed_loop_nominal.json"
        if p.is_file():
            _audit_artifact(p, label="nominal", regime=regime, target_keys=target_by_regime[regime], checks=checks, contact_manifest=contact_manifest, contact_manifest_sha=contact_manifest_sha, latency=False)

    failures = [c for c in checks if not c["ok"]]
    report = {
        "schema": "ocrap-final-metric-fairness-audit-v1",
        "status": "PASS" if not failures else "FAIL",
        "publication_metric_semantics_version": PUBLICATION_METRIC_SEMANTICS_VERSION,
        "ocrap_root": str(args.ocrap_root),
        "external_root": str(args.external_root),
        "ocrap_latency_root": str(ocrap_latency),
        "external_latency_root": str(external_latency),
        "main_only": bool(args.main_only),
        "num_checks": len(checks),
        "num_failures": len(failures),
        "failures": failures,
        "checks": checks,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if not failures else 30


if __name__ == "__main__":
    raise SystemExit(main())
