#!/usr/bin/env python3
"""Choose a small deterministic Contact subset for reference synthesis.

The expensive reference controller should not be run on every expanded Contact
anchor when the final artifact needs only a handful of ranks.  This preselector
uses *existing empirical traces only* to keep:

* a configurable number of already-good preferred scenes (e.g. current rank01);
* critical but still fixable scenes where external baselines have substantial
  post-contact difficulty and the empirical source is not already a gross
  off-road/runaway case.

No recovery/safety gate is weakened here.  The downstream reference planner and
trace-aware selector still apply the full hard realism contract.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

CONTACT_METHODS = (
    "postimpact_mpc_lite",
    "post_crash_braking",
    "postimpact_motion_tvlqr",
    "post_collision_restoration",
    "compensatory_postimpact_mpc",
    "robust_postimpact_control",
)


def _scene_key(scene: dict[str, Any], env: dict[str, Any]) -> str:
    k = str(scene.get("target_key") or env.get("resume_key") or "")
    return k[7:] if k.startswith("target:") else k


def _load(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            env = json.loads(line)
            scene = env.get("scene", env)
            if not isinstance(scene, dict):
                continue
            key = _scene_key(scene, env)
            if key:
                rows[key] = scene
    return rows


def _flag(frame: dict[str, Any], key: str) -> bool:
    try:
        return float((frame.get("metrics") or {}).get(key) or 0.0) > 0.5
    except Exception:
        return False


def _metric(frame: dict[str, Any], key: str) -> float | None:
    try:
        v = float((frame.get("metrics") or {}).get(key))
    except Exception:
        return None
    return v if math.isfinite(v) else None


def _quick_quality(scene: dict[str, Any], dt: float) -> dict[str, Any]:
    tr = list(scene.get("render_trace") or [])
    if not tr:
        return {"usable": False}
    overlap = [_flag(f, "overlap") for f in tr]
    offroad = [_flag(f, "offroad") for f in tr]
    clear = [_metric(f, "min_clearance_m") for f in tr]
    first_contact = next((i for i, x in enumerate(overlap) if x), None)
    first_sep = None
    if first_contact is not None:
        first_sep = next((i for i in range(first_contact + 1, len(overlap)) if not overlap[i]), None)
    recontact = bool(first_sep is not None and any(overlap[first_sep + 1 :]))
    overlap_s = float(sum(overlap[:-1]) * dt) if len(overlap) > 1 else 0.0
    term = clear[-1] if clear else None
    return {
        "usable": bool(first_contact is not None),
        "first_separation_s": None if first_sep is None else float(first_sep * dt),
        "recontact": recontact,
        "overlap_duration_s": overlap_s,
        "terminal_clearance_m": term,
        "offroad_fraction": float(sum(offroad) / len(offroad)) if offroad else 0.0,
        "persistent_initial_contact": first_sep is None,
    }


def _preferred_keys(path: Path | None, n: int) -> list[str]:
    if path is None or not path.is_file() or n <= 0:
        return []
    d = json.loads(path.read_text(encoding="utf-8"))
    return [str(x.get("target_key")) for x in (d.get("selected") or []) if isinstance(x, dict) and x.get("target_key")][:n]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ocrap-trace", type=Path, required=True)
    ap.add_argument("--baseline", action="append", default=[])
    ap.add_argument("--anchor-manifest", type=Path, required=True)
    ap.add_argument("--preferred-selection", type=Path, default=None)
    ap.add_argument("--preserve-preferred-count", type=int, default=1)
    ap.add_argument("--max-scenes", type=int, default=14)
    ap.add_argument("--max-source-offroad-fraction", type=float, default=0.15)
    ap.add_argument("--metric-dt-s", type=float, default=0.1)
    ap.add_argument("--output-target-keys", type=Path, required=True)
    ap.add_argument("--output-anchor-manifest", type=Path, required=True)
    ap.add_argument("--output-audit", type=Path, required=True)
    args = ap.parse_args()
    if args.max_scenes <= 0:
        raise SystemExit("--max-scenes must be positive")

    ocrap = _load(args.ocrap_trace)
    baseline_maps: dict[str, dict[str, dict[str, Any]]] = {}
    for spec in args.baseline:
        if "=" not in spec:
            raise SystemExit(f"invalid --baseline={spec!r}")
        name, raw = spec.split("=", 1)
        baseline_maps[name.strip()] = _load(Path(raw))
    manifest = json.loads(args.anchor_manifest.read_text(encoding="utf-8"))
    anchors = list(manifest.get("anchors") or [])
    anchor_by_key = {str(a.get("target_key")): a for a in anchors if a.get("target_key")}
    preferred = _preferred_keys(args.preferred_selection, int(args.preserve_preferred_count))
    preferred = [k for k in preferred if k in anchor_by_key and k in ocrap]

    rows: list[dict[str, Any]] = []
    dt = float(args.metric_dt_s)
    for key, anchor in anchor_by_key.items():
        if key not in ocrap:
            continue
        oq = _quick_quality(ocrap[key], dt)
        if not oq.get("usable"):
            continue
        bq = {m: _quick_quality(mp[key], dt) for m, mp in baseline_maps.items() if key in mp}
        # Criticality rewards cases where several empirical baselines remain in
        # contact, re-contact, or separate slowly.  It is intentionally based on
        # *source* results only; no synthesized outcome is consulted.
        persistent = sum(bool(q.get("persistent_initial_contact")) for q in bq.values())
        recontact = sum(bool(q.get("recontact")) for q in bq.values())
        slow_sep = sum(
            q.get("first_separation_s") is None or float(q.get("first_separation_s")) >= 1.0
            for q in bq.values()
        )
        overlap_med = 0.0
        ovs = sorted(float(q.get("overlap_duration_s") or 0.0) for q in bq.values())
        if ovs:
            overlap_med = ovs[len(ovs) // 2]
        # Fixability is a conservative source filter only.  Hard lane/recontact
        # constraints are still applied downstream after synthesis.
        gross_offroad = float(oq.get("offroad_fraction") or 0.0) > float(args.max_source_offroad_fraction)
        criticality = 8.0 * persistent + 5.0 * recontact + 3.0 * slow_sep + 4.0 * overlap_med
        if oq.get("recontact"):
            criticality += 2.0
        if oq.get("first_separation_s") is None:
            criticality += 1.0
        rows.append({
            "target_key": key,
            "scene_id": anchor.get("scene_id"),
            "preferred": key in preferred,
            "gross_source_offroad": gross_offroad,
            "criticality_score": float(criticality),
            "ocrap_source": oq,
            "baseline_source": bq,
        })

    # Keep forced preferred scenes first.  For synthesis candidates reject gross
    # source off-road cases, because they tend to need route-level rather than
    # local post-contact correction.
    selected: list[str] = []
    for key in preferred:
        if key not in selected:
            selected.append(key)
    eligible = [r for r in rows if not r["gross_source_offroad"] and r["target_key"] not in selected]
    eligible.sort(key=lambda r: (-float(r["criticality_score"]), str(r["target_key"])))
    for row in eligible:
        if len(selected) >= int(args.max_scenes):
            break
        selected.append(str(row["target_key"]))

    if not selected:
        raise SystemExit("reference synthesis prefilter selected no scenes")
    selected_set = set(selected)
    filtered_anchors = [a for a in anchors if str(a.get("target_key")) in selected_set]
    filtered = dict(manifest)
    filtered["anchors"] = filtered_anchors
    filtered["target_keys"] = sorted(selected_set)
    filtered["num_selected_anchors"] = len(filtered_anchors)
    filtered["num_selected_scenes"] = len({str(a.get("scene_id")) for a in filtered_anchors})
    filtered["selection_policy"] = str(filtered.get("selection_policy") or "") + "+reference_synthesis_prefilter"
    filtered["reference_synthesis_prefilter_only"] = True
    filtered["source_anchor_manifest"] = str(args.anchor_manifest)

    args.output_target_keys.parent.mkdir(parents=True, exist_ok=True)
    args.output_target_keys.write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
    args.output_anchor_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_anchor_manifest.write_text(json.dumps(filtered, indent=2) + "\n", encoding="utf-8")
    audit = {
        "event": "contact_reference_synthesis_subset_v1",
        "max_scenes": int(args.max_scenes),
        "preserve_preferred_count": int(args.preserve_preferred_count),
        "preferred_keys": preferred,
        "selected_keys": selected,
        "rows": rows,
        "scientific_note": "This is a compute prefilter only. Final target-display realism/comparative gates remain unchanged.",
    }
    args.output_audit.parent.mkdir(parents=True, exist_ok=True)
    args.output_audit.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": audit["event"], "selected": len(selected), "preferred": preferred, "output": str(args.output_target_keys)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
