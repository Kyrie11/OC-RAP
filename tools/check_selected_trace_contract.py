#!/usr/bin/env python3
"""Strict contract for already-generated regime visualization traces."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

METHODS = {
    "safe": ["gameformer_lite", "plantf", "pluto", "pdm_closed", "pdm_hybrid", "idm", "diffusion_planner"],
    "near": ["marc_lite", "racp_lite", "robust_scenario_mpc", "predictive_safety_filter", "dr_cvar_safety_filter", "conformal_predictive_safety_filter", "flow_planner", "plan_r1", "betopnet"],
    "contact": ["postimpact_mpc_lite", "post_crash_braking", "postimpact_motion_tvlqr", "post_collision_restoration", "compensatory_postimpact_mpc", "robust_postimpact_control"],
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace-root", type=Path, required=True)
    ap.add_argument("--selection-root", type=Path, required=True)
    ap.add_argument("--trace-max-steps", type=int, default=60)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--allow-extra-targets", action="store_true", help="Allow journals to contain an over-selected candidate pool while validating only the final selected subset.")
    args = ap.parse_args()

    errors: list[str] = []
    regimes: dict[str, object] = {}
    for regime, methods in METHODS.items():
        selection = json.loads((args.selection_root / f"{regime}_selection.json").read_text())
        requested = [str(x) for x in json.loads((args.selection_root / f"{regime}_target_keys.json").read_text())["target_keys"]]
        requested_set = set(requested)
        dt = float(selection.get("metric_dt_s", 0.1) or 0.1)
        selected_by_key = {str(x["target_key"]): x for x in (selection.get("selected") or [])}
        req_steps: dict[str, int] = {}
        for key in requested:
            item = selected_by_key.get(key) or {}
            clip = float(item.get("clip_duration_s", selection.get("selected_clip_duration_s", 0.0)) or 0.0)
            steps = int(math.ceil(clip / dt - 1e-9))
            req_steps[key] = steps
            if steps <= 0:
                errors.append(f"invalid selected clip duration {regime}/{key}: clip={clip} dt={dt}")
            if steps > args.trace_max_steps:
                errors.append(f"selected clip exceeds trace cap {regime}/{key}: required_steps={steps} cap={args.trace_max_steps}")

        paths = {"ocrap": args.trace_root / "ocrap" / regime / "closed_loop_ocrap.json.scenes.jsonl"}
        paths.update({m: args.trace_root / "external" / regime / f"closed_loop_{m}.json.scenes.jsonl" for m in methods})
        counts: dict[str, int] = {}
        trace_starts = {k: {} for k in requested}
        for method, path in paths.items():
            if not path.is_file():
                errors.append(f"missing journal {regime}/{method}: {path}")
                continue
            seen: dict[str, int] = {}
            with path.open(encoding="utf-8") as f:
                for lineno, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    env = json.loads(line)
                    scene = env.get("scene", env)
                    key = str(scene.get("target_key") or env.get("resume_key") or "")
                    if key.startswith("target:"):
                        key = key[len("target:"):]
                    if key in seen:
                        errors.append(f"duplicate selected target {regime}/{method}/{key}: lines {seen[key]},{lineno}")
                    seen[key] = lineno
                    if key not in requested_set:
                        continue
                    trace = scene.get("render_trace") or []
                    required_frames = req_steps[key] + 1
                    if not trace:
                        errors.append(f"no render_trace {regime}/{method}/{key}; selected journal must be full")
                    else:
                        try:
                            trace_starts[key][method] = int(trace[0]["time_index"])
                        except Exception:
                            errors.append(f"render_trace lacks integer start time {regime}/{method}/{key}")
                        if len(trace) < required_frames:
                            errors.append(f"short render_trace {regime}/{method}/{key}: frames={len(trace)} required>={required_frames}")
            missing = sorted(requested_set - set(seen))
            extra = sorted(set(seen) - requested_set)
            if missing:
                errors.append(f"unresolved selected targets {regime}/{method}: {missing}")
            if extra and not args.allow_extra_targets:
                errors.append(f"unexpected selected-trace targets {regime}/{method}: {extra}")
            counts[method] = len(seen)

        for key in requested:
            starts = trace_starts.get(key, {})
            if starts and len(set(starts.values())) != 1:
                errors.append(f"model trace starts not synchronized {regime}/{key}: {starts}")
            item = selected_by_key.get(key) or {}
            field = "contact_anchor_time_index" if regime == "contact" else "target_time_index"
            expected = item.get(field)
            if starts and expected is not None and next(iter(starts.values())) != int(expected):
                errors.append(f"trace start mismatch {regime}/{key}: got={next(iter(starts.values()))} expected_{field}={expected}")
        regimes[regime] = {
            "requested": len(requested),
            "journal_counts": counts,
            "required_rollout_steps_by_target": req_steps,
            "max_required_rollout_steps": max(req_steps.values()) if req_steps else 0,
            "selected_clip_duration_s": selection.get("selected_clip_duration_s"),
            "duration_selection_mode": selection.get("duration_selection_mode"),
            "duration_source": selection.get("duration_source"),
        }

    doc = {
        "event": "selected_regime_trace_contract_v127",
        "valid": not errors,
        "errors": errors,
        "trace_max_steps": args.trace_max_steps,
        "allow_extra_targets": bool(args.allow_extra_targets),
        "regimes": regimes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(doc, ensure_ascii=False, indent=2))
    return 0 if not errors else 30


if __name__ == "__main__":
    raise SystemExit(main())
