#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

METHODS = {
    "safe": ["gameformer_lite", "plantf", "pluto", "pdm_closed", "pdm_hybrid", "idm", "diffusion_planner"],
    "near": ["marc_lite", "racp_lite", "robust_scenario_mpc", "predictive_safety_filter", "dr_cvar_safety_filter", "conformal_predictive_safety_filter", "flow_planner", "plan_r1", "betopnet"],
    "contact": ["postimpact_mpc_lite", "post_crash_braking", "postimpact_motion_tvlqr", "post_collision_restoration", "compensatory_postimpact_mpc", "robust_postimpact_control"],
}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def journal_trace_summary(path: Path) -> dict:
    if not path.is_file():
        return {"exists": False, "rows": 0, "render_trace_rows": 0}
    rows = traces = 0
    keys = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows += 1
            x = json.loads(line); scene = x.get("scene", x)
            k = str(scene.get("target_key") or x.get("resume_key") or "")
            if k.startswith("target:"):
                k = k[len("target:"):]
            if k:
                keys.add(k)
            if scene.get("render_trace"):
                traces += 1
    return {"exists": True, "rows": rows, "unique_keys": len(keys), "render_trace_rows": traces}


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit the staged regime-visualization pipeline and explain the first incomplete stage.")
    ap.add_argument("--root", type=Path, required=True, help="regime_visualization_v48_124_final directory")
    ap.add_argument("--expected-scenes", type=int, default=3)
    ap.add_argument("--expected-safe-scenes", type=int, default=None)
    ap.add_argument("--expected-near-scenes", type=int, default=None)
    ap.add_argument("--expected-contact-scenes", type=int, default=None)
    args = ap.parse_args()
    expected_by_regime = {
        "safe": int(args.expected_safe_scenes if args.expected_safe_scenes is not None else args.expected_scenes),
        "near": int(args.expected_near_scenes if args.expected_near_scenes is not None else args.expected_scenes),
        "contact": int(args.expected_contact_scenes if args.expected_contact_scenes is not None else args.expected_scenes),
    }
    root = args.root
    errors: list[str] = []
    stages: dict[str, object] = {}

    contract = root / "provenance" / "VISUALIZATION_INPUT_CONTRACT.json"
    if not contract.is_file():
        errors.append(f"missing input contract: {contract}")
    else:
        d = load(contract)
        stages["input_contract"] = {"valid": d.get("valid"), "errors": d.get("errors")}
        if d.get("valid") is not True:
            errors.append("visualization input contract is invalid")

    horizon = root / "provenance" / "CONTACT_VISUALIZATION_HORIZON.json"
    if horizon.is_file():
        h = load(horizon)
        stages["contact_horizon"] = {k: h.get(k) for k in (
            "locked_targets", "preferred_duration_s", "num_preferred_eligible",
            "fallback_duration_s", "num_fallback_eligible")}

    selections = {}
    selection_docs = {}
    for regime in ("safe", "near", "contact"):
        p = root / "selection" / f"{regime}_selection.json"
        if not p.is_file():
            errors.append(f"missing {regime} selection: {p}")
            continue
        d = load(p); n = len(d.get("selected") or [])
        selection_docs[regime] = d
        realism_failures = []
        for row in d.get("selected") or []:
            q = row.get("visualization_trace_quality") or {}
            key = str(row.get("target_key") or "<unknown>")
            if regime == "safe" and q.get("accepted") is not True:
                realism_failures.append(f"{key}:safe_trace_gate")
            elif regime == "near" and q.get("ocrap_realistic_safe") is not True:
                realism_failures.append(f"{key}:near_realism_gate")
            elif regime == "contact" and q.get("ocrap_controlled_recovery") is not True:
                realism_failures.append(f"{key}:contact_controlled_recovery_gate")
        selections[regime] = {
            "selected": n, "clip_duration_s": d.get("selected_clip_duration_s"),
            "duration_mode": d.get("duration_selection_mode"), "duration_source": d.get("duration_source"),
            "preferred_candidates": d.get("num_preferred_duration_candidates"),
            "fallback_candidates": d.get("num_fallback_duration_candidates"),
            "realism_gate_failures": realism_failures,
        }
        if n < expected_by_regime[regime]:
            errors.append(f"{regime} selection has {n} scenes, expected {expected_by_regime[regime]}")
        if realism_failures:
            errors.append(f"{regime} selected scenes failed realism gate: " + ", ".join(realism_failures))
    stages["selection"] = selections

    prep = root / "selective_traces" / "TRACE_PREP.json"
    if prep.is_file():
        d = load(prep)
        stages["trace_preparation"] = {
            "num_reusable": d.get("num_reusable"),
            "num_invalid_or_missing": d.get("num_invalid_or_missing"),
        }

    rerun = root / "selective_traces" / "TRACE_RERUN_STATUS.json"
    if rerun.is_file():
        d = load(rerun)
        stages["trace_rerun_status"] = d
        if d.get("valid") is not True:
            errors.append(f"selected trace rerun stage failed: {d.get('failures')}")

    trace_inventory = {}
    for regime, ext in METHODS.items():
        paths = {"ocrap": root / "selective_traces" / "ocrap" / regime / "closed_loop_ocrap.json.scenes.jsonl"}
        paths.update({m: root / "selective_traces" / "external" / regime / f"closed_loop_{m}.json.scenes.jsonl" for m in ext})
        trace_inventory[regime] = {m: journal_trace_summary(p) for m, p in paths.items()}
    stages["trace_inventory"] = trace_inventory

    trace_contract = root / "selective_traces" / "TRACE_CONTRACT.json"
    if trace_contract.is_file():
        t = load(trace_contract)
        stages["trace_contract"] = {"valid": t.get("valid"), "errors": t.get("errors"), "regimes": t.get("regimes")}
        if t.get("valid") is not True:
            errors.append("selective trace contract is invalid")
    elif len(selections) == 3:
        errors.append(f"selection completed but trace contract is missing: {trace_contract}")
        # Give actionable hints even when the producer exited before writing the contract.
        no_trace = []
        for r, methods in trace_inventory.items():
            for m, x in methods.items():
                if not x.get("exists") or int(x.get("render_trace_rows") or 0) < expected_by_regime[r]:
                    no_trace.append(f"{r}/{m}")
        if no_trace:
            errors.append("missing/incomplete render traces: " + ", ".join(no_trace))

    fig_index = root / "paper_figures" / "PAPER_FIGURE_INDEX.json"
    if fig_index.is_file():
        f = load(fig_index)
        n_records = sum(len((x or {}).get("records") or []) for x in (f.get("regimes") or {}).values())
        stages["paper_figures"] = {"num_scene_records": n_records}
        if n_records <= 0:
            errors.append("paper figure index exists but contains no scene records")
        for regime, sdoc in selection_docs.items():
            frecs = (((f.get("regimes") or {}).get(regime) or {}).get("records") or [])
            expected = [
                (str(x.get("target_key") or ""), int(x.get("category_rank") or 0), str(x.get("primary_external_method") or ""))
                for x in (sdoc.get("selected") or [])
            ]
            actual = [
                (str(x.get("target_key") or ""), int(x.get("rank") or 0), str(x.get("primary_external_method") or ""))
                for x in frecs
            ]
            if actual != expected:
                errors.append(f"{regime} paper figure index does not match current selection/comparator ordering")
    elif trace_contract.is_file() and load(trace_contract).get("valid") is True:
        errors.append(f"trace contract passed but paper figure index is missing: {fig_index}")

    video_index = root / "videos" / "REGIME_VIDEO_INDEX.json"
    if video_index.is_file():
        v = load(video_index); stages["videos"] = {"num_videos": v.get("num_videos")}
        if int(v.get("num_videos") or 0) <= 0:
            errors.append("video index exists but contains no videos")
        for regime, sdoc in selection_docs.items():
            vrecs = (((v.get("regimes") or {}).get(regime) or {}).get("records") or [])
            expected = [
                (str(x.get("target_key") or ""), int(x.get("category_rank") or 0), str(x.get("primary_external_method") or ""))
                for x in (sdoc.get("selected") or [])
            ]
            actual = [
                (str(x.get("target_key") or ""), int(x.get("rank") or 0), str(x.get("primary_external_method") or ""))
                for x in vrecs
            ]
            if actual != expected:
                errors.append(f"{regime} video index does not match current selection/comparator ordering")
    elif trace_contract.is_file() and load(trace_contract).get("valid") is True:
        errors.append(f"trace contract passed but video index is missing: {video_index}")

    doc = {"event": "regime_visualization_output_audit_v128_realism", "root": str(root), "valid": not errors, "errors": errors, "stages": stages}
    print(json.dumps(doc, ensure_ascii=False, indent=2))
    return 0 if not errors else 30


if __name__ == "__main__":
    raise SystemExit(main())
