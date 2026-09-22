#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit the staged regime-visualization pipeline and explain the first incomplete stage.")
    ap.add_argument("--root", type=Path, required=True, help="regime_visualization_v48_124_final directory")
    ap.add_argument("--expected-scenes", type=int, default=3)
    args = ap.parse_args()
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
        stages["contact_horizon"] = {
            "locked_targets": h.get("locked_targets"),
            "preferred_duration_s": h.get("preferred_duration_s"),
            "num_preferred_eligible": h.get("num_preferred_eligible"),
            "fallback_duration_s": h.get("fallback_duration_s"),
            "num_fallback_eligible": h.get("num_fallback_eligible"),
        }

    selections = {}
    for regime in ("safe", "near", "contact"):
        p = root / "selection" / f"{regime}_selection.json"
        if not p.is_file():
            errors.append(f"missing {regime} selection: {p}")
            continue
        d = load(p)
        n = len(d.get("selected") or [])
        selections[regime] = {
            "selected": n,
            "clip_duration_s": d.get("selected_clip_duration_s"),
            "duration_mode": d.get("duration_selection_mode"),
            "duration_source": d.get("duration_source"),
            "preferred_candidates": d.get("num_preferred_duration_candidates"),
            "fallback_candidates": d.get("num_fallback_duration_candidates"),
        }
        if n < args.expected_scenes:
            errors.append(f"{regime} selection has {n} scenes, expected {args.expected_scenes}")
    stages["selection"] = selections

    trace_contract = root / "selective_traces" / "TRACE_CONTRACT.json"
    if trace_contract.is_file():
        t = load(trace_contract)
        stages["trace_contract"] = {"valid": t.get("valid"), "errors": t.get("errors"), "regimes": t.get("regimes")}
        if t.get("valid") is not True:
            errors.append("selective trace contract is invalid")
    elif len(selections) == 3:
        errors.append(f"selection completed but trace contract is missing: {trace_contract}")

    video_index = root / "videos" / "REGIME_VIDEO_INDEX.json"
    if video_index.is_file():
        v = load(video_index)
        stages["videos"] = {"num_videos": v.get("num_videos")}
        if int(v.get("num_videos") or 0) <= 0:
            errors.append("video index exists but contains no videos")
    elif trace_contract.is_file() and load(trace_contract).get("valid") is True:
        errors.append(f"trace contract passed but video index is missing: {video_index}")

    fig_index = root / "paper_figures" / "PAPER_FIGURE_INDEX.json"
    if fig_index.is_file():
        f = load(fig_index)
        n_records = sum(len((x or {}).get("records") or []) for x in (f.get("regimes") or {}).values())
        stages["paper_figures"] = {"num_scene_records": n_records}
        if n_records <= 0:
            errors.append("paper figure index exists but contains no scene records")
    elif video_index.is_file():
        errors.append(f"videos exist but paper figure index is missing: {fig_index}")

    doc = {
        "event": "regime_visualization_output_audit_v125",
        "root": str(root),
        "valid": not errors,
        "errors": errors,
        "stages": stages,
    }
    print(json.dumps(doc, ensure_ascii=False, indent=2))
    return 0 if not errors else 30


if __name__ == "__main__":
    raise SystemExit(main())
