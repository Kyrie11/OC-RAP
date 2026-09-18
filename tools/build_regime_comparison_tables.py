#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from ocrap.external_baselines.provenance import find_provenance


SCHEMA: dict[str, list[tuple[str, str, str]]] = {
    "safe": [
        ("num_scenes", "Scenes", "count"),
        ("collision_scene_rate", "Collision scene rate ↓", "rate"),
        ("offroad_scene_rate", "Off-road scene rate ↓", "rate"),
        ("minimum_clearance_m", "Minimum clearance [m] ↑", "float"),
        ("scene_min_clearance_m_p05", "Scene clearance p05 [m] ↑", "float"),
        ("scene_min_clearance_m_median", "Scene clearance median [m] ↑", "float"),
        ("minimum_ttc_s", "Minimum TTC [s] ↑", "float"),
        ("scene_ttc_s_p05", "Scene TTC p05 [s] ↑", "float"),
        ("closed_loop_bounded_NUP", "Bounded NUP ↑", "float"),
        ("intervention_rate", "Intervention rate", "rate"),
        ("decision_latency_ms", "Decision latency [ms] ↓", "float"),
    ],
    "near": [
        ("num_scenes", "Scenes", "count"),
        ("collision_scene_rate", "Collision scene rate ↓", "rate"),
        ("offroad_scene_rate", "Off-road scene rate ↓", "rate"),
        ("scene_min_clearance_m_p05", "Scene clearance p05 [m] ↑", "float"),
        ("scene_min_clearance_noncollision_m_p05", "Non-collision scene clearance p05 [m] ↑", "float"),
        ("scene_ttc_s_p05", "Scene TTC p05 [s] ↑", "float"),
        ("terminal_clearance_m", "Terminal clearance [m] ↑", "float"),
        ("terminal_ttc_s", "Terminal TTC [s] ↑", "float"),
        ("critical_ttc_exposure_duration_s", "Critical-TTC exposure [s] ↓", "float"),
        ("ttc_deficit_auc_s2", "TTC-deficit AUC [s²] ↓", "float"),
        ("near_zero_clearance_exposure_rate", "Near-zero-clearance exposure ↓", "rate"),
        ("closed_loop_bounded_NUP", "Bounded NUP ↑", "float"),
        ("intervention_rate", "Intervention rate", "rate"),
        ("decision_latency_ms", "Decision latency [ms] ↓", "float"),
    ],
    "contact": [
        ("num_scenes", "Scenes", "count"),
        ("counterfactual_contact_target_scene_rate", "Counterfactual-contact target rate", "rate"),
        ("observed_contact_scene_rate", "Observed-contact scene rate", "rate"),
        ("post_contact_metric_eligible_scene_rate", "Post-contact metric eligibility", "rate"),
        ("collision_scene_rate", "Any-overlap scene rate (anchor audit)", "rate"),
        ("scene_min_clearance_m_p05", "Scene clearance p05 [m] ↑", "float"),
        ("scene_min_clearance_noncollision_m_p05", "Non-collision scene clearance p05 [m] ↑", "float"),
        ("terminal_clearance_m", "Terminal clearance [m] ↑", "float"),
        ("clearance_recovery_gain_m", "Clearance recovery gain [m] ↑", "float"),
        ("overlap_duration_s", "Overlap duration [s] ↓", "float"),
        ("penetration_scene_rate", "Any-penetration scene rate (anchor-sensitive audit)", "rate"),
        ("penetration_duration_s", "OBB penetration duration [s] ↓", "float"),
        ("scene_max_penetration_depth_m_mean", "Mean scene max penetration [m] ↓", "float"),
        ("penetration_depth_auc_m_s", "Penetration-depth AUC [m·s] ↓", "float"),
        ("post_contact_terminal_clearance_m", "Observed-contact terminal clearance [m] ↑", "float"),
        ("post_contact_free_space_auc_normalized_m", "Observed-contact free-space AUC [m] ↑", "float"),
        ("post_contact_clearance_gain_m", "Observed-contact clearance gain [m] ↑", "float"),
        ("post_contact_escape_scene_rate", "Observed-contact escape rate ↑", "rate"),
        ("recontact_scene_rate", "Observed-contact re-contact rate ↓", "rate"),
        ("secondary_overlap_identity_available_scene_rate", "Secondary-overlap identity eligibility", "rate"),
        ("secondary_overlap_scene_rate", "Observed-contact secondary-overlap rate ↓", "rate"),
        ("stable_stop_eligible_scene_rate", "New-stop eligibility", "rate"),
        ("new_stable_stop_quality_conditional_scene_rate", "Conditional stable-stop-quality rate ↑", "rate"),
        ("offroad_scene_rate", "Off-road scene rate ↓", "rate"),
        ("post_contact_overlap_duration_s", "Observed-contact overlap [s] ↓", "float"),
        ("decision_latency_ms", "Decision latency [ms] ↓", "float"),
    ],
}


def _scene_journal(path: Path) -> Path | None:
    candidates = [Path(str(path) + ".scenes.jsonl"), path.with_suffix(path.suffix + ".scenes.jsonl")]
    return next((x for x in candidates if x.is_file()), None)


def _scene_keys(path: Path) -> set[str]:
    out: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            e = json.loads(line); s = e.get("scene", e)
            key = str(s.get("target_key") or e.get("resume_key") or "")
            if not key:
                scene_id = str(s.get("scene_id") or ""); t = s.get("target_time_index")
                key = f"{scene_id}:t{t}" if scene_id and t is not None else scene_id
            if key:
                out.add(key)
    return out


def _finite(x: Any) -> float | int | None:
    try:
        v = float(x)
        if not math.isfinite(v):
            return None
        return int(v) if v.is_integer() else v
    except Exception:
        return None


def _get(doc: dict[str, Any], key: str) -> float | int | None:
    if key == "decision_latency_ms":
        timing = doc.get("timing", {}) or {}
        per = timing.get("per_decision_s", {}) or {}
        # Publication latency is observation -> deployable action selection.
        # Prefer synchronized steady-state samples with per-scene warmup removed;
        # never include teacher/audit labels or simulator/metric bookkeeping.
        steady = timing.get("steady_state_deployed_planner_s", {}) or {}
        if steady.get("mean") is not None and math.isfinite(float(steady.get("mean"))):
            return _finite(1000.0 * float(steady["mean"]))
        if "deployed_planner" in per:
            return _finite(1000.0 * float(per["deployed_planner"]))
        deployed = ["state_history", "candidate_features", "policy_selection"]
        values = [float(per[k]) for k in deployed if k in per and per[k] is not None and math.isfinite(float(per[k]))]
        if values:
            return _finite(1000.0 * sum(values))
        # Very old artifacts exposed only a single total and cannot be safely
        # decomposed; retain compatibility as a last-resort fallback.
        if "total" in per:
            return _finite(1000.0 * float(per["total"]))
        return None
    if key in doc:
        return _finite(doc.get(key))
    wm = doc.get("waymax_metrics", {}) or {}
    return _finite(wm.get(key))


def _format(value: Any, kind: str) -> str:
    if value is None:
        return "—"
    v = float(value)
    if kind == "count":
        return str(int(round(v)))
    if kind == "rate":
        return f"{v:.4f}"
    return f"{v:.4f}"


PUBLICATION_METRIC_SEMANTICS_VERSION = "publication_v55_signed_clearance_unclipped_v1"
PUBLICATION_CONTRACT_KEYS = (
    "metric_semantics_version", "max_steps", "replan_interval_steps", "metric_dt_s",
    "num_candidate_prefixes", "num_recovery_options", "use_sdc_paths",
    "require_observation_legal_route", "allow_future_route_proxy",
    "dataloader_include_sdc_paths", "allow_logged_sdc_route_fallback",
    "compute_future_metrics", "publication_geometry_metric", "clearance_is_signed",
    "duration_auc_support", "womd_source_role",
)


def _evaluation_contract(doc: dict[str, Any]) -> dict[str, Any]:
    value = doc.get("evaluation_contract") or {}
    return value if isinstance(value, dict) else {}


def _publication_contract_signature(doc: dict[str, Any]) -> dict[str, Any]:
    c = _evaluation_contract(doc)
    return {k: c.get(k) for k in PUBLICATION_CONTRACT_KEYS}


def main() -> int:
    ap = argparse.ArgumentParser(description="Build regime-specific OC-RAP/external-baseline comparison tables.")
    ap.add_argument("--regime", choices=tuple(SCHEMA), required=True)
    ap.add_argument("--input", action="append", required=True, metavar="METHOD=RESULT.json")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--allow-unpaired", action="store_true", help="Do not fail when scene journals exist but target sets differ.")
    ap.add_argument("--allow-unanchored-contact", action="store_true", help="Diagnostic compatibility only. Publication Contact comparison requires one shared pre-treatment observed-contact anchor manifest.")
    ap.add_argument("--latency-input", action="append", default=[], metavar="METHOD=RESULT.json", help="Optional isolated-latency artifact for one method; metrics still come from --input.")
    ap.add_argument("--omit-latency", action="store_true", help="Omit latency from the table when no isolated publication-latency rerun is available; never substitute throughput timing silently.")
    args = ap.parse_args()
    metric_schema = [x for x in SCHEMA[args.regime] if not (args.omit_latency and x[0] == "decision_latency_ms")]

    latency_docs: dict[str, tuple[Path, dict[str, Any]]] = {}
    for spec in args.latency_input:
        if "=" not in spec:
            raise SystemExit(f"invalid --latency-input {spec!r}; expected METHOD=PATH")
        method, raw = spec.split("=", 1); lp = Path(raw)
        if not lp.is_file():
            raise SystemExit(f"missing isolated latency result: {lp}")
        ldoc = json.loads(lp.read_text(encoding="utf-8"))
        contract = str(((ldoc.get("timing") or {}).get("execution_contract") or ""))
        if contract != "isolated_single_process_single_gpu":
            raise SystemExit(f"latency artifact for {method} is not isolated: {contract!r} ({lp})")
        latency_docs[method.strip()] = (lp, ldoc)

    entries: list[tuple[str, Path, dict[str, Any]]] = []
    for spec in args.input:
        if "=" not in spec:
            raise SystemExit(f"invalid --input {spec!r}; expected METHOD=PATH")
        method, raw = spec.split("=", 1); path = Path(raw)
        if not path.is_file():
            raise SystemExit(f"missing result: {path}")
        entries.append((method.strip(), path, json.loads(path.read_text(encoding="utf-8"))))

    # When an isolated latency artifact is provided, cohort identity is a
    # prerequisite for using its timing value.  Check this before other artifact
    # metadata so a target mismatch is never masked by a secondary validation.
    for method, path, _ in entries:
        latency_source = latency_docs.get(method)
        if latency_source is None:
            continue
        acc_journal = _scene_journal(path)
        lat_journal = _scene_journal(latency_source[0])
        if acc_journal is None or lat_journal is None:
            raise SystemExit(
                f"accuracy/latency scene journals are required for isolated latency pairing: {method}"
            )
        acc_keys = _scene_keys(acc_journal)
        lat_keys = _scene_keys(lat_journal)
        if acc_keys != lat_keys:
            mismatch = {
                "only_accuracy": sorted(acc_keys - lat_keys)[:10],
                "only_latency": sorted(lat_keys - acc_keys)[:10],
                "accuracy_count": len(acc_keys),
                "latency_count": len(lat_keys),
            }
            raise SystemExit(
                f"latency target set does not match accuracy target set for {method}: "
                + json.dumps(mismatch, ensure_ascii=False)
            )

    # Fail closed if any scene lost a publication metric observation.  Missing
    # clearance/TTC/Waymax overlap/offroad samples must never silently become a
    # favorable zero exposure/event for one method.
    coverage_keys = (
        "clearance_metric_full_coverage_scene_rate",
        "ttc_metric_full_coverage_scene_rate",
        "overlap_metric_full_coverage_scene_rate",
        "offroad_metric_full_coverage_scene_rate",
    )
    coverage_by_method = {
        m: {k: _get(doc, k) for k in coverage_keys}
        for m, _, doc in entries
    }
    bad_coverage = {
        m: vals for m, vals in coverage_by_method.items()
        if any(v is None or abs(float(v) - 1.0) > 1.0e-12 for v in vals.values())
    }
    if bad_coverage and not args.allow_unpaired:
        raise SystemExit(
            "publication metric coverage is incomplete; refusing to compare potentially biased endpoints: "
            + json.dumps(bad_coverage, ensure_ascii=False)
        )

    # Fail closed on metric/rollout contract mismatches.  Equal target keys are
    # not enough if one method used clipped clearance, a different horizon, a
    # future-route fallback, or another WOMD source role.
    contract_by_method = {m: _publication_contract_signature(doc) for m, _, doc in entries}
    missing_contract = [m for m, sig in contract_by_method.items() if sig.get("metric_semantics_version") is None]
    if missing_contract and not args.allow_unpaired:
        raise SystemExit("missing publication evaluation_contract: " + ", ".join(missing_contract))
    bad_semantics = {m: sig.get("metric_semantics_version") for m, sig in contract_by_method.items()
                     if sig.get("metric_semantics_version") not in {None, PUBLICATION_METRIC_SEMANTICS_VERSION}}
    if bad_semantics and not args.allow_unpaired:
        raise SystemExit("non-publication metric semantics: " + json.dumps(bad_semantics, ensure_ascii=False))
    if contract_by_method and not args.allow_unpaired:
        ref_method = entries[0][0]
        ref_contract = contract_by_method[ref_method]
        mismatched_contract = {m: sig for m, sig in contract_by_method.items() if sig != ref_contract}
        if mismatched_contract:
            raise SystemExit(
                "evaluation contracts differ across methods; target pairing alone is insufficient: "
                + json.dumps({"reference": {ref_method: ref_contract}, "mismatch": mismatched_contract}, ensure_ascii=False)
            )

    if args.regime == "contact" and not args.allow_unanchored_contact:
        anchor_rows = {
            m: {
                "protocol": doc.get("contact_anchor_protocol"),
                "manifest_sha256": doc.get("contact_anchor_manifest_sha256"),
                "fingerprint_required": doc.get("contact_anchor_state_fingerprint_required"),
            }
            for m, _, doc in entries
        }
        if any(v.get("protocol") != "exact_a0_pretreatment_prelude_v1" or not v.get("manifest_sha256")
               or v.get("fingerprint_required") is not True for v in anchor_rows.values()):
            raise SystemExit(
                "publication Contact comparison requires the shared exact-a0 pre-treatment observed-contact anchor protocol: "
                + json.dumps(anchor_rows, ensure_ascii=False)
            )
        manifest_shas = {str(v["manifest_sha256"]) for v in anchor_rows.values()}
        if len(manifest_shas) != 1:
            raise SystemExit("Contact methods do not share one frozen anchor manifest: " + json.dumps(anchor_rows, ensure_ascii=False))

        # Final post-impact endpoints are only a fully paired treatment comparison
        # when every retained scene is observed-contact eligible for every method.
        # The exact-a0 anchor contract is designed to make this 100%; fail closed
        # rather than silently mixing conditional denominators in a main table.
        eligibility_rows = {
            m: _get(doc, "post_contact_metric_eligible_scene_rate")
            for m, _, doc in entries
        }
        bad_eligibility = {
            m: v for m, v in eligibility_rows.items()
            if v is None or abs(float(v) - 1.0) > 1.0e-12
        }
        if bad_eligibility:
            raise SystemExit(
                "publication Contact requires 100% post-contact metric eligibility on the shared anchor cohort: "
                + json.dumps(bad_eligibility, ensure_ascii=False)
            )

    journals = [(m, _scene_journal(p)) for m, p, _ in entries]
    present = [(m, j) for m, j in journals if j is not None]
    missing_journals = [m for m, j in journals if j is None]
    if missing_journals and not args.allow_unpaired:
        raise SystemExit(
            "missing scene journals required for paired comparison: "
            + ", ".join(missing_journals)
        )
    paired = False
    paired_count = None
    if len(present) == len(entries) and present:
        key_sets = {m: _scene_keys(j) for m, j in present if j is not None}
        reference_method = entries[0][0]
        reference = key_sets[reference_method]
        mismatch = {
            m: {
                "only_reference": sorted(reference - ks)[:10],
                "only_method": sorted(ks - reference)[:10],
            }
            for m, ks in key_sets.items() if ks != reference
        }
        if mismatch and not args.allow_unpaired:
            raise SystemExit(f"unpaired closed-loop target sets: {json.dumps(mismatch, ensure_ascii=False)}")
        paired = not mismatch
        paired_count = len(reference) if paired else None

    rows: list[dict[str, Any]] = []
    for method, path, doc in entries:
        prov = find_provenance(method)
        latency_source = latency_docs.get(method)
        if latency_source is not None:
            acc_contract = _publication_contract_signature(doc)
            lat_contract = _publication_contract_signature(latency_source[1])
            if acc_contract != lat_contract:
                raise SystemExit(
                    f"latency evaluation contract does not match accuracy contract for {method}: "
                    + json.dumps({"accuracy": acc_contract, "latency": lat_contract}, ensure_ascii=False)
                )
            if args.regime == "contact" and not args.allow_unanchored_contact:
                if doc.get("contact_anchor_manifest_sha256") != latency_source[1].get("contact_anchor_manifest_sha256"):
                    raise SystemExit(f"Contact latency anchor manifest does not match accuracy artifact for {method}")
        row: dict[str, Any] = {
            "method": method,
            "reporting_name": prov.reporting_name if prov else method,
            "implementation_kind": prov.implementation_kind if prov else ("OC-RAP" if method.lower().startswith("ocrap") else "unknown"),
            "fidelity": prov.fidelity if prov else ("proposed method" if method.lower().startswith("ocrap") else "unknown"),
            "source_result": str(path),
            "source_latency_result": str(latency_source[0]) if latency_source else None,
        }
        for key, _, _ in metric_schema:
            row[key] = _get(latency_source[1], key) if key == "decision_latency_ms" and latency_source else _get(doc, key)
        rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.regime}_comparison"
    contact_eligibility = {
        method: _get(doc, "post_contact_metric_eligible_scene_rate")
        for method, _, doc in entries
    } if args.regime == "contact" else None
    contact_post_metrics_fully_paired = (
        bool(contact_eligibility)
        and all(v is not None and abs(float(v) - 1.0) <= 1.0e-12 for v in contact_eligibility.values())
    ) if args.regime == "contact" else None
    json_doc = {
        "schema_version": 2, "regime": args.regime, "paired_scene_set": paired,
        "paired_scene_count": paired_count,
        "metric_protocol": (
            "shared exact-a0 pre-treatment observed-contact anchor cohort; all physical and post_contact_* endpoints start from the same verified treatment-boundary simulator state" if args.regime == "contact" and not args.allow_unanchored_contact
            else "counterfactual-contact target cohort; generic physical metrics use all paired scenes, while post_contact_* metrics are strictly anchored only on observed Waymax overlap" if args.regime == "contact"
            else "deployable physical closed-loop metrics; expensive selected/all-candidate teacher certificate audits are excluded from the main table" if args.regime == "near"
            else "deployable physical closed-loop metrics"
        ),
        "contact_protocol": (
            "Publication Contact uses one scene-disjoint treatment-independent exact-a0 prelude manifest. Every compared method independently reproduces the same observed-overlap simulator-state fingerprint before its first evaluated action; the frozen anchor lock also requires the full evaluation horizon to remain."
            if args.regime == "contact" and not args.allow_unanchored_contact
            else "Legacy diagnostic mode: test_contact is a counterfactual contact-surrogate cohort; post_contact_* values are conditional on policy-dependent observed contact."
            if args.regime == "contact" else None
        ),
        "contact_post_metrics_fully_paired": contact_post_metrics_fully_paired,
        "post_contact_metric_eligibility_by_method": contact_eligibility,
        "publication_metric_coverage_by_method": coverage_by_method,
        "metrics": [{"key": k, "label": label, "kind": kind} for k, label, kind in metric_schema],
        "rows": rows,
    }
    (args.output_dir / f"{stem}.json").write_text(json.dumps(json_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fields = ["method", "reporting_name", "implementation_kind", "fidelity", "source_result", "source_latency_result"] + [x[0] for x in metric_schema]
    with (args.output_dir / f"{stem}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows([{k: r.get(k) for k in fields} for r in rows])

    headers = ["Method"] + [x[1] for x in metric_schema]
    lines = ["# " + args.regime.capitalize() + " regime comparison", "", f"Paired target set: **{paired}**" + (f" ({paired_count} scenes)" if paired_count is not None else ""), ""]
    if args.regime == "contact":
        if not args.allow_unanchored_contact:
            lines += [
                "> Publication Contact is evaluated from one shared exact-a0 pre-treatment observed-contact anchor manifest. The first evaluated action for every method sees the same verified simulator-state fingerprint, and the anchor cohort retains the full treatment horizon.",
                "",
            ]
        else:
            lines += [
                "> Legacy diagnostic mode: `post_contact_*` columns are conditional on policy-dependent observed contact and are not a fully paired post-impact comparison unless eligibility is 100% for every method.",
                "",
            ]
    elif args.regime == "near":
        lines += ["> The main Near table uses deployable physical closed-loop metrics. Exact teacher-label FRA/DRS/ODG audits are optional diagnostics and are not mixed into the runtime comparison.", ""]
    lines += ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for r in rows:
        cells = [str(r["reporting_name"])] + [_format(r.get(k), kind) for k, _, kind in metric_schema]
        lines.append("| " + " | ".join(cells) + " |")
    (args.output_dir / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"event": "regime_comparison_table", "regime": args.regime, "methods": len(rows), "paired": paired, "output_dir": str(args.output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
