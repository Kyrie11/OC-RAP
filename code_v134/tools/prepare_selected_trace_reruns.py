#!/usr/bin/env python3
"""Prepare resumable selected-trace reruns for regime visualization.

A selected-trace artifact is reusable only when every requested target has a
full render_trace long enough for the selected clip and synchronized to the
regime's treatment boundary.  Missing/incomplete artifacts can optionally be
removed as a *family* so the normal closed-loop launchers regenerate only the
invalid method/regime instead of deleting all already-valid traces.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

METHODS = {
    "safe": ["gameformer_lite", "plantf", "pluto", "pdm_closed", "pdm_hybrid", "idm", "diffusion_planner"],
    "near": ["marc_lite", "racp_lite", "robust_scenario_mpc", "predictive_safety_filter", "dr_cvar_safety_filter", "conformal_predictive_safety_filter", "flow_planner", "plan_r1", "betopnet"],
    "contact": ["postimpact_mpc_lite", "post_crash_braking", "postimpact_motion_tvlqr", "post_collision_restoration", "compensatory_postimpact_mpc", "robust_postimpact_control"],
}


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _key(envelope: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    scene = envelope.get("scene", envelope)
    key = str(scene.get("target_key") or envelope.get("resume_key") or "")
    if key.startswith("target:"):
        key = key[len("target:"):]
    if not key:
        sid = str(scene.get("scene_id") or "")
        ti = scene.get("target_time_index")
        key = f"{sid}:t{ti}" if sid and ti is not None else sid
    return key, scene


def _artifact_family(base: Path) -> list[Path]:
    return [
        base,
        Path(str(base) + ".partial"),
        Path(str(base) + ".progress.json"),
        Path(str(base) + ".scenes.jsonl"),
        base.with_suffix(".log"),
    ]


def _scientific_scene_signature(scene: dict[str, Any]) -> str:
    """Canonical scene payload excluding execution-only timing measurements."""
    scientific = {k: v for k, v in scene.items() if k != "timing"}
    return json.dumps(scientific, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=True)


def _check_journal(
    path: Path,
    selection: dict[str, Any],
    requested: list[str],
    *,
    repair_equivalent_duplicates: bool = False,
) -> tuple[bool, list[str], dict[str, Any]]:
    errors: list[str] = []
    if not path.is_file():
        return False, [f"missing journal: {path}"], {"rows": 0, "unique_keys": 0}
    selected = {str(x["target_key"]): x for x in (selection.get("selected") or [])}
    dt = float(selection.get("metric_dt_s", 0.1) or 0.1)
    rows = 0
    scenes: dict[str, dict[str, Any]] = {}
    envelopes: dict[str, dict[str, Any]] = {}
    kept_raw_lines: list[str] = []
    equivalent_duplicate_keys: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            rows += 1
            env = json.loads(line)
            k, scene = _key(env)
            if not k:
                errors.append(f"line {lineno}: scene without target key")
                continue
            if k in scenes:
                first_env = envelopes[k]
                same_fp = str(first_env.get("run_fingerprint") or "") == str(env.get("run_fingerprint") or "")
                same_science = _scientific_scene_signature(scenes[k]) == _scientific_scene_signature(scene)
                if same_fp and same_science:
                    equivalent_duplicate_keys.add(k)
                    continue
                errors.append(f"conflicting duplicate target key {k} in selected trace journal")
                continue
            scenes[k] = scene
            envelopes[k] = env
            kept_raw_lines.append(line.rstrip("\n"))
    req = set(requested)
    missing = sorted(req - set(scenes))
    extra = sorted(set(scenes) - req)
    if missing:
        errors.append(f"missing selected targets: {missing}")
    if extra:
        errors.append(f"unexpected targets in selected trace journal: {extra}")
    per_target: dict[str, Any] = {}
    regime = str(selection.get("regime") or "")
    for k in requested:
        scene = scenes.get(k)
        item = selected.get(k) or {}
        if scene is None:
            continue
        clip = float(item.get("clip_duration_s", selection.get("selected_clip_duration_s", 0.0)) or 0.0)
        steps = int(math.ceil(clip / dt - 1e-9))
        trace = list(scene.get("render_trace") or [])
        expected_field = "contact_anchor_time_index" if regime == "contact" else "target_time_index"
        expected_start = item.get(expected_field)
        start = None
        if trace:
            try:
                start = int(trace[0]["time_index"])
            except Exception:
                pass
        if len(trace) < steps + 1:
            errors.append(f"{k}: render_trace frames={len(trace)} required>={steps+1}")
        if expected_start is not None and start != int(expected_start):
            errors.append(f"{k}: trace start={start} expected {expected_field}={expected_start}")
        per_target[k] = {
            "clip_duration_s": clip,
            "required_rollout_steps": steps,
            "trace_frames": len(trace),
            "trace_start_time_index": start,
            "expected_start_field": expected_field,
            "expected_start_time_index": expected_start,
        }
    normalized = False
    if not errors and equivalent_duplicate_keys and repair_equivalent_duplicates:
        tmp = path.with_suffix(path.suffix + ".dedupe.tmp")
        tmp.write_text("\n".join(kept_raw_lines) + "\n", encoding="utf-8")
        tmp.replace(path)
        rows = len(kept_raw_lines)
        normalized = True
    return not errors, errors, {
        "rows": rows,
        "unique_keys": len(scenes),
        "per_target": per_target,
        "equivalent_duplicate_keys": sorted(equivalent_duplicate_keys),
        "normalized_equivalent_duplicates": normalized,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace-root", type=Path, required=True)
    ap.add_argument("--selection-root", type=Path, required=True)
    ap.add_argument("--clean-invalid", action="store_true")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    methods: list[dict[str, Any]] = []
    for regime in ("safe", "near", "contact"):
        selection_path = args.selection_root / f"{regime}_selection.json"
        keys_path = args.selection_root / f"{regime}_target_keys.json"
        selection = _load(selection_path)
        requested = [str(x) for x in (_load(keys_path).get("target_keys") or [])]
        specs = [("ocrap", args.trace_root / "ocrap" / regime / "closed_loop_ocrap.json")]
        specs += [(m, args.trace_root / "external" / regime / f"closed_loop_{m}.json") for m in METHODS[regime]]
        for method, base in specs:
            journal = Path(str(base) + ".scenes.jsonl")
            valid, errs, detail = _check_journal(
                journal, selection, requested, repair_equivalent_duplicates=args.clean_invalid
            )
            existed = any(p.exists() for p in _artifact_family(base))
            removed: list[str] = []
            status = "reusable" if valid else ("invalid" if existed else "missing")
            if not valid and existed and args.clean_invalid:
                for p in _artifact_family(base):
                    if p.exists():
                        p.unlink()
                        removed.append(str(p))
                status = "invalid_cleaned"
            methods.append({
                "regime": regime,
                "method": method,
                "artifact": str(base),
                "journal": str(journal),
                "status": status,
                "valid_reusable": valid,
                "errors": errs,
                "removed": removed,
                **detail,
            })

    doc = {
        "event": "selected_trace_rerun_preparation_v126",
        "trace_root": str(args.trace_root),
        "selection_root": str(args.selection_root),
        "clean_invalid": args.clean_invalid,
        "num_reusable": sum(bool(x["valid_reusable"]) for x in methods),
        "num_invalid_or_missing": sum(not bool(x["valid_reusable"]) for x in methods),
        "methods": methods,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: doc[k] for k in ("event", "num_reusable", "num_invalid_or_missing", "clean_invalid")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
