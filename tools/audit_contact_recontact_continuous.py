#!/usr/bin/env python3
"""Compute continuous re-contact diagnostics from full Contact render traces.

This is an *audit* metric family.  It does not replace the historical
Bernoulli ``recontact_scene_rate`` used in the main table.  It adds conditional
post-separation burden metrics that are more informative on small exact-a0
cohorts and can be recomputed offline from already-generated full traces.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def _scene_key(scene: dict[str, Any], env: dict[str, Any]) -> str:
    key = str(scene.get("target_key") or env.get("resume_key") or "")
    if key.startswith("target:"):
        key = key[len("target:"):]
    if key:
        return key
    sid = str(scene.get("scene_id") or "")
    ti = scene.get("target_time_index")
    return f"{sid}:t{ti}" if sid and ti is not None else sid


def _load_keys(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(d, list):
        rows = d
    elif isinstance(d, dict):
        rows = d.get("target_keys") or []
    else:
        rows = []
    keys = {str(x) for x in rows if str(x)}
    if not keys:
        raise SystemExit(f"target-key file is empty or invalid: {path}")
    return keys


def _load_journal(path: Path, allowed: set[str] | None) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise SystemExit(f"missing trace journal: {path}")
    out: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            env = json.loads(line)
            scene = env.get("scene", env)
            if not isinstance(scene, dict):
                continue
            key = _scene_key(scene, env)
            if not key or (allowed is not None and key not in allowed):
                continue
            if key in out:
                raise SystemExit(f"duplicate target {key}: {path}")
            out[key] = scene
    if not out:
        raise SystemExit(f"no matching scenes in {path}")
    return out


def _metric(frame: dict[str, Any], key: str) -> float | None:
    try:
        v = float((frame.get("metrics") or {}).get(key))
    except Exception:
        return None
    return v if math.isfinite(v) else None


def _flag(frame: dict[str, Any], key: str) -> bool:
    return (_metric(frame, key) or 0.0) > 0.5


def _trace_metrics(scene: dict[str, Any], dt: float) -> dict[str, Any]:
    trace = scene.get("render_trace") or []
    if len(trace) < 2:
        return {"eligible": False, "reason": "missing_render_trace"}
    overlap = [_flag(f, "overlap") for f in trace]
    first_contact = next((i for i, x in enumerate(overlap) if x), None)
    if first_contact is None or first_contact >= len(overlap) - 1:
        return {"eligible": False, "reason": "no_observed_contact"}

    # First state after the initial continuous contact episode.  Re-contact is
    # meaningful only after the rollout has actually separated once.
    first_sep = next((i for i in range(first_contact + 1, len(overlap)) if not overlap[i]), None)
    if first_sep is None or first_sep >= len(overlap) - 1:
        return {
            "eligible": False,
            "reason": "no_initial_separation",
            "observed_contact": True,
            "initial_contact_duration_s": float((len(overlap) - 1 - first_contact) * dt),
        }

    # Intervals are represented by their left state, matching the runner's
    # duration convention.  Exclude the terminal state because it owns no
    # following interval.
    post_flags = overlap[first_sep:-1]
    starts = []
    prev = False
    for j, flag in enumerate(post_flags):
        if flag and not prev:
            starts.append(j)
        prev = flag
    n = len(post_flags)
    event = bool(starts)
    duration_steps = int(sum(post_flags))
    first_recontact_offset = starts[0] if starts else None
    free_fraction = 1.0 if not event else float(first_recontact_offset / max(n, 1))
    return {
        "eligible": True,
        "reason": "ok",
        "observed_contact": True,
        "first_contact_index": int(first_contact),
        "first_separation_index": int(first_sep),
        "initial_contact_duration_s": float((first_sep - first_contact) * dt),
        "post_separation_horizon_s": float(n * dt),
        "recontact_event": float(event),
        "recontact_episode_count": float(len(starts)),
        "recontact_duration_s": float(duration_steps * dt),
        "recontact_exposure_rate": float(duration_steps / n) if n else None,
        "recontact_free_fraction": free_fraction,
        "time_to_first_recontact_s": (float(first_recontact_offset * dt) if event else None),
    }


def _mean(vals: list[float]) -> float | None:
    return sum(vals) / len(vals) if vals else None


def _aggregate(method: str, rows: dict[str, dict[str, Any]], dt: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    per_scene = []
    for key, scene in sorted(rows.items()):
        x = _trace_metrics(scene, dt)
        per_scene.append({"method": method, "target_key": key, **x})
    eligible = [r for r in per_scene if r.get("eligible")]
    no_sep = [r for r in per_scene if r.get("reason") == "no_initial_separation"]
    observed = [r for r in per_scene if r.get("observed_contact")]
    event_vals = [float(r["recontact_event"]) for r in eligible]
    ep_vals = [float(r["recontact_episode_count"]) for r in eligible]
    dur_vals = [float(r["recontact_duration_s"]) for r in eligible]
    exp_vals = [float(r["recontact_exposure_rate"]) for r in eligible if r.get("recontact_exposure_rate") is not None]
    free_vals = [float(r["recontact_free_fraction"]) for r in eligible]
    ttr_vals = [float(r["time_to_first_recontact_s"]) for r in eligible if r.get("time_to_first_recontact_s") is not None]
    agg = {
        "method": method,
        "num_scenes": len(per_scene),
        "observed_contact_scenes": len(observed),
        "separation_eligible_scenes": len(eligible),
        "persistent_initial_contact_scenes": len(no_sep),
        "separation_eligible_rate": len(eligible) / len(per_scene) if per_scene else None,
        "recontact_after_separation_scene_rate": _mean(event_vals),
        "no_recontact_after_separation_scene_rate": (1.0 - _mean(event_vals)) if event_vals else None,
        "mean_recontact_episode_count": _mean(ep_vals),
        "mean_recontact_duration_s": _mean(dur_vals),
        "mean_recontact_exposure_rate": _mean(exp_vals),
        "mean_recontact_free_fraction": _mean(free_vals),
        "mean_time_to_first_recontact_s_conditional": _mean(ttr_vals),
        "scientific_note": (
            "Re-contact burden is conditional on achieving an initial separation. "
            "Persistent initial contact is reported separately so a method cannot look good merely because it never separated."
        ),
    }
    return agg, per_scene


def main() -> int:
    ap = argparse.ArgumentParser(description="Offline continuous re-contact audit from full Contact render traces.")
    ap.add_argument("--trace", action="append", default=[], help="METHOD=path/to/scenes.jsonl")
    ap.add_argument("--target-keys-file", type=Path, default=None)
    ap.add_argument("--metric-dt-s", type=float, default=0.1)
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--output-csv", type=Path, required=True)
    ap.add_argument("--per-scene-csv", type=Path, default=None)
    args = ap.parse_args()
    if args.metric_dt_s <= 0:
        raise SystemExit("metric-dt-s must be positive")
    specs = []
    for spec in args.trace:
        if "=" not in spec:
            raise SystemExit(f"invalid --trace {spec!r}; expected METHOD=PATH")
        name, raw = spec.split("=", 1)
        specs.append((name.strip(), Path(raw)))
    if not specs:
        raise SystemExit("at least one --trace is required")
    allowed = _load_keys(args.target_keys_file)
    summaries = []
    per_scene_all = []
    target_sets = {}
    for method, path in specs:
        rows = _load_journal(path, allowed)
        target_sets[method] = set(rows)
        agg, per_scene = _aggregate(method, rows, float(args.metric_dt_s))
        summaries.append(agg)
        per_scene_all.extend(per_scene)
    ref_name, ref_set = next(iter(target_sets.items()))
    mismatch = {
        name: {"missing_vs_reference": sorted(ref_set - keys), "extra_vs_reference": sorted(keys - ref_set)}
        for name, keys in target_sets.items() if keys != ref_set
    }
    if mismatch:
        raise SystemExit("unpaired trace target sets: " + json.dumps(mismatch, ensure_ascii=False))
    doc = {
        "event": "contact_recontact_continuous_audit_v1",
        "num_paired_targets": len(ref_set),
        "target_keys_file": str(args.target_keys_file) if args.target_keys_file else None,
        "metric_dt_s": float(args.metric_dt_s),
        "methods": summaries,
        "interpretation": (
            "Keep the historical Bernoulli recontact_scene_rate for backward compatibility. "
            "Use these conditional continuous metrics as an additional robustness analysis, not as a post-hoc replacement chosen for better-looking numbers."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    fields = [
        "method", "num_scenes", "observed_contact_scenes", "separation_eligible_scenes",
        "persistent_initial_contact_scenes", "separation_eligible_rate",
        "recontact_after_separation_scene_rate", "no_recontact_after_separation_scene_rate",
        "mean_recontact_episode_count", "mean_recontact_duration_s", "mean_recontact_exposure_rate",
        "mean_recontact_free_fraction", "mean_time_to_first_recontact_s_conditional",
    ]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in summaries:
            w.writerow({k: row.get(k) for k in fields})
    if args.per_scene_csv is not None:
        per_fields = [
            "method", "target_key", "eligible", "reason", "observed_contact", "initial_contact_duration_s",
            "post_separation_horizon_s", "recontact_event", "recontact_episode_count", "recontact_duration_s",
            "recontact_exposure_rate", "recontact_free_fraction", "time_to_first_recontact_s",
        ]
        args.per_scene_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.per_scene_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=per_fields)
            w.writeheader()
            for row in per_scene_all:
                w.writerow({k: row.get(k) for k in per_fields})
    print(json.dumps({"event": doc["event"], "num_paired_targets": len(ref_set), "output_csv": str(args.output_csv)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
