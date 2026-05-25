#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

# Allow running scripts directly from a source checkout without pip install -e .
import sys as _sys
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from recap.envs.scenario_description_adapter import (
    extract_map_features_from_scenario,
    extract_root_state_from_scenario,
    extract_route_info_from_scenario,
    history_from_scenario,
    load_scenarionet_summary,
    read_scenario_description,
    scenario_current_time_index,
    scenario_file_path,
)
from recap.envs.scenario_regimes import REGIME_RATIOS
from recap.utils.datatypes import dataclass_to_jsonable
from recap.utils.serialization import write_json
from recap.utils.progress import tqdm



def _actor_clearance(ego, actor) -> float:
    center = float(np.hypot(actor.x - ego.x, actor.y - ego.y))
    ego_radius = 0.5 * max(float(getattr(ego, "length", 4.7)), float(getattr(ego, "width", 1.9)))
    actor_radius = 0.5 * max(float(getattr(actor, "length", 4.7)), float(getattr(actor, "width", 1.9)))
    return center - ego_radius - actor_radius


def _min_actor_clearance(ego, actors) -> float:
    if not actors:
        return float("inf")
    return float(min(_actor_clearance(ego, a) for a in actors))


def _closing_ttc(ego, actor) -> float:
    rel_pos = np.asarray([actor.x - ego.x, actor.y - ego.y], dtype=np.float32)
    d = float(np.linalg.norm(rel_pos))
    if d < 1e-6:
        return 0.0
    ego_v = np.asarray([ego.v * np.cos(ego.heading), ego.v * np.sin(ego.heading)], dtype=np.float32)
    actor_v = np.asarray([actor.vx, actor.vy], dtype=np.float32)
    rel_v = actor_v - ego_v
    closing = -float(np.dot(rel_pos / d, rel_v))
    clearance = max(_actor_clearance(ego, actor), 0.0)
    if closing <= 1e-3:
        return float("inf")
    return clearance / closing


def _min_ttc(ego, actors) -> float:
    if not actors:
        return float("inf")
    return float(min(_closing_ttc(ego, a) for a in actors))


def _future_min_clearance(scenario: dict, summary: dict, t: int, horizon: int) -> tuple[float, float]:
    T = _track_length(scenario, summary)
    if T <= 0:
        T = t + 1
    hi = min(int(T) - 1, int(t) + max(0, int(horizon)))
    best_clearance = float("inf")
    best_ttc = float("inf")
    for j in range(max(0, int(t)), hi + 1):
        ego_j, actors_j = extract_root_state_from_scenario(scenario, t=j, summary=summary)
        best_clearance = min(best_clearance, _min_actor_clearance(ego_j, actors_j))
        best_ttc = min(best_ttc, _min_ttc(ego_j, actors_j))
    return float(best_clearance), float(best_ttc)


def _classify_regime(ego, actors, summary: dict, scenario: dict | None = None, t: int | None = None, lookahead_steps: int = 30) -> str:
    # Use approximate actor-body clearance, not center distance.  Center-distance
    # thresholds miss most vehicle contacts because two car centers can remain
    # 4--6 m apart at body overlap.
    clearance = _min_actor_clearance(ego, actors)
    ttc = _min_ttc(ego, actors)
    if scenario is not None and t is not None:
        future_clearance, future_ttc = _future_min_clearance(scenario, summary, int(t), lookahead_steps)
        clearance = min(clearance, future_clearance)
        ttc = min(ttc, future_ttc)
    moving = int((summary.get("number_summary", {}) or {}).get("num_moving_objects", 0) or 0)
    if clearance <= 0.25:
        return "contact_post_contact"
    if clearance <= 2.0 or ttc <= 1.0:
        return "near_contact"
    if clearance <= 6.0 or ttc <= 2.5 or moving >= 32:
        return "low_headroom"
    return "normal_high_headroom"


def _root_replay_history_ego(scenario: dict, summary: dict, t: int) -> list[dict]:
    # Store the ego prefix from ScenarioEnv reset time to the root tick.  This is
    # needed when roots are event-aligned later than current_time_index=10; the
    # previous code only kept the last BEV history window, which is insufficient
    # for replaying MetaDrive to the requested root time.
    out = []
    for j in range(0, max(0, int(t)) + 1):
        e, _ = extract_root_state_from_scenario(scenario, t=j, summary=summary)
        out.append(dataclass_to_jsonable(e))
    return out


def _tick_event_score(scenario: dict, summary: dict, t: int, lookahead_steps: int) -> tuple[int, float, float, float]:
    ego, actors = extract_root_state_from_scenario(scenario, t=t, summary=summary)
    clearance, ttc = _future_min_clearance(scenario, summary, int(t), lookahead_steps)
    regime = _classify_regime(ego, actors, summary, scenario=scenario, t=int(t), lookahead_steps=lookahead_steps)
    priority = {"contact_post_contact": 3, "near_contact": 2, "low_headroom": 1, "normal_high_headroom": 0}[regime]
    # Higher is better: prioritize harder regimes, then smaller clearance/TTC.
    severity = -min(clearance, 20.0) - 0.5 * min(ttc, 10.0)
    return priority, severity, float(clearance), float(ttc)

def _moving_objects(summary: dict) -> int:
    return int((summary.get("number_summary", {}) or {}).get("num_moving_objects", 0) or 0)


def _num_objects(summary: dict) -> int:
    return int((summary.get("number_summary", {}) or {}).get("num_objects", 0) or 0)


def _parse_regime_counts(spec: str | None) -> Dict[str, int]:
    if not spec:
        return {}
    out: Dict[str, int] = {}
    aliases = {
        "normal": "normal_high_headroom",
        "low": "low_headroom",
        "near": "near_contact",
        "contact": "contact_post_contact",
    }
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Bad --target-regime-counts item {part!r}; expected name=count")
        k, v = [x.strip() for x in part.split("=", 1)]
        k = aliases.get(k, k)
        if k not in REGIME_RATIOS:
            raise ValueError(f"Unknown regime {k!r}; valid regimes are {sorted(REGIME_RATIOS)}")
        out[k] = int(v)
    return out


def _root_json(root_id: str, scenario_dir: Path, scenario_index: int, scenario_id: str, scenario_pkl: Path, scenario: dict, summary: dict, history_steps: int, root_tick: int | None = None, event_lookahead_steps: int = 30) -> dict:
    t = scenario_current_time_index(scenario, summary) if root_tick is None else int(root_tick)
    ego, actors = extract_root_state_from_scenario(scenario, t=t, summary=summary)
    mf = extract_map_features_from_scenario(scenario)
    route = extract_route_info_from_scenario(scenario, ego, summary=summary, t=t)
    hist = history_from_scenario(scenario, t, history_steps, summary=summary)
    regime = _classify_regime(ego, actors, summary, scenario=scenario, t=t, lookahead_steps=event_lookahead_steps)
    # Python's built-in hash() is intentionally randomized across processes,
    # which would make root-shared mode seeds non-reproducible.  Include the root
    # tick so temporal roots from the same WOMD log remain distinct when
    # --max-samples-per-log > 1 is used for diagnostics.
    seed_key = f"{scenario_id}:{int(t)}"
    seed = int.from_bytes(hashlib.blake2b(seed_key.encode("utf-8"), digest_size=8).digest(), "little") % (2**31 - 1)
    return {
        "root_id": root_id,
        "seed": int(seed),
        "regime": regime,
        "root_tick": int(t),
        "ego_state": dataclass_to_jsonable(ego),
        "actor_states": dataclass_to_jsonable(actors),
        "map_features": {
            "drivable_polygons": [p.tolist() for p in mf.drivable_polygons],
            "lane_centerlines": [p.tolist() for p in mf.lane_centerlines],
            "lane_boundaries": [p.tolist() for p in mf.lane_boundaries],
            "static_obstacles": [p.tolist() for p in mf.static_obstacles],
            "speed_limit_mps": mf.speed_limit_mps,
        },
        "route_info": {
            "waypoints": route.waypoints.tolist(),
            "command_ids": route.command_ids.tolist() if route.command_ids is not None else [],
            "speed_limit_mps": route.speed_limit_mps,
        },
        "history": [
            {"ego_state": dataclass_to_jsonable(e), "actor_states": dataclass_to_jsonable(a)} for e, a in hist
        ],
        "replay_history_ego": _root_replay_history_ego(scenario, summary, int(t)),
        "scenario_data": {
            "data_directory": str(scenario_dir),
            "scenario_index": int(scenario_index),
            "scenario_id": str(scenario_id),
            "scenario_file_name": str(Path(scenario_pkl).name),
            "waymo_scenario_id": str(summary.get("scenario_id", (scenario.get("metadata", {}) or {}).get("scenario_id", ""))),
            "scenario_pkl": str(scenario_pkl),
            "current_time_index": int(t),
            "source_dataset": (summary.get("dataset") or (scenario.get("metadata", {}) or {}).get("dataset") or "waymo"),
            "source_file": summary.get("source_file") or (scenario.get("metadata", {}) or {}).get("source_file"),
            "sdc_id": summary.get("sdc_id") or (scenario.get("metadata", {}) or {}).get("sdc_id"),
            "coordinate_frame": "metadrive_centralized_sdc_initial",
        },
        "map_config": {"source": "scenarionet", "coordinate": (summary.get("coordinate") or (scenario.get("metadata", {}) or {}).get("coordinate"))},
        "traffic_config": {"source": "scenarionet_waymo", "num_objects": _num_objects(summary), "num_moving_objects": _moving_objects(summary)},
    }


def _track_length(scenario: dict, summary: dict) -> int:
    from recap.envs.scenario_description_adapter import scenario_sdc_id, _state_series, _get_by_flexible_key
    tracks = scenario.get("tracks", {}) or {}
    sdc_id = scenario_sdc_id(scenario, summary)
    track = _get_by_flexible_key(tracks, sdc_id) if sdc_id is not None else None
    if track is None and tracks:
        track = next(iter(tracks.values()))
    state = (track or {}).get("state", {}) or {}
    pos = _state_series(state, ["position", "pos", "center"])
    if pos is not None and getattr(pos, "ndim", 0) >= 2:
        return int(pos.shape[0])
    return int(summary.get("track_length", summary.get("scenario_length", 0)) or 0)


def _sample_root_ticks(scenario: dict, summary: dict, history_steps: int, max_samples: int, stride: int, event_aligned: bool = False, event_lookahead_steps: int = 30) -> list[int]:
    current = int(scenario_current_time_index(scenario, summary))
    max_samples = max(1, int(max_samples))
    T = _track_length(scenario, summary)
    if T <= 0:
        return [current]
    lo = max(0, int(history_steps) - 1)
    hi = max(lo + 1, T - 2)
    stride = max(1, int(stride))
    ticks = list(range(lo, hi + 1, stride))
    if current not in ticks and lo <= current <= hi:
        ticks.append(current)
    ticks = sorted(set(ticks))
    if not ticks:
        return [current]
    if event_aligned:
        ranked = sorted(
            ticks,
            key=lambda tt: _tick_event_score(scenario, summary, int(tt), int(event_lookahead_steps))[:2],
            reverse=True,
        )
        chosen = sorted(ranked[:max_samples])
        return [int(x) for x in chosen]
    if max_samples == 1:
        return [current]
    if len(ticks) <= max_samples:
        return ticks
    # Deterministic uniform coverage of the log rather than adjacent highly-correlated frames.
    idx = np.linspace(0, len(ticks) - 1, max_samples).round().astype(int)
    return [ticks[int(i)] for i in idx]


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect ReCAP root JSON files from a MetaDrive/ScenarioNet real-world database.")
    ap.add_argument("--scenario-dir", required=True, help="ScenarioNet database containing dataset_summary.pkl.")
    ap.add_argument("--output", default="data/recap/roots_raw")
    ap.add_argument("--split-name", choices=["train", "val", "calib", "test", "debug"], default="train")
    ap.add_argument("--max-roots", type=int, default=None)
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--min-moving-objects", type=int, default=1)
    ap.add_argument("--min-objects", type=int, default=2)
    ap.add_argument("--history-steps", type=int, default=10)
    ap.add_argument("--max-samples-per-log", type=int, default=1, help="Number of temporal roots to sample from each scenario/log. Use 1 with --event-aligned-root for one event-rich root per log.")
    ap.add_argument("--sample-stride", type=int, default=5, help="Minimum frame stride between temporal roots when max_samples_per_log > 1 or event-aligned scanning is enabled.")
    ap.add_argument("--event-aligned-root", action="store_true", help="Choose the root tick with highest contact/near-contact/low-headroom score instead of always using current_time_index.")
    ap.add_argument("--event-lookahead-steps", type=int, default=30, help="Future frames used to classify/scored event-aligned roots.")
    ap.add_argument("--target-regime-counts", default="", help="Optional balanced collection target, e.g. normal=2000,low=2000,near=2000,contact=1000. When set, scanning continues until requested per-regime counts or --max-scenarios-to-scan is reached.")
    ap.add_argument("--max-scenarios-to-scan", type=int, default=None, help="Optional cap on scenarios scanned when using --target-regime-counts.")
    ap.add_argument("--append", action="store_true", help="Append to existing root directory/splits instead of replacing split list.")
    args = ap.parse_args()
    target_counts = _parse_regime_counts(args.target_regime_counts)
    scenario_dir = Path(args.scenario_dir)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    summary, scenario_ids, mapping = load_scenarionet_summary(scenario_dir)
    split_path = out / "splits.json"
    existing_split_map = json.loads(split_path.read_text()) if args.append and split_path.exists() else {}
    existing_scenario_ids = set()
    for ids0 in existing_split_map.values():
        for rid0 in ids0 or []:
            p0 = out / f"{rid0}.json"
            if not p0.exists():
                continue
            try:
                sd0 = (json.loads(p0.read_text()).get("scenario_data", {}) or {})
                if sd0.get("scenario_id"):
                    existing_scenario_ids.add(str(sd0.get("scenario_id")))
            except Exception:
                continue
    candidates: List[tuple[int, str]] = []
    for idx, sid in enumerate(tqdm(scenario_ids, desc="scan_scenarios", unit="scenario")):
        if idx < args.start_index:
            continue
        if str(sid) in existing_scenario_ids:
            continue
        if args.max_scenarios_to_scan is not None and len(candidates) >= int(args.max_scenarios_to_scan):
            break
        sm = summary.get(sid, {}) or {}
        if _moving_objects(sm) < args.min_moving_objects:
            continue
        if _num_objects(sm) < args.min_objects:
            continue
        candidates.append((idx, sid))
        # max_roots is a cap on output root JSON files, not scenarios.  For multi-tick sampling, apply again while writing.
        if not target_counts and args.max_roots is not None and len(candidates) * max(1, args.max_samples_per_log) >= args.max_roots:
            break
    if args.append and split_path.exists():
        split_map = existing_split_map
    else:
        split_map = {"train": [], "val": [], "calib": [], "test": [], "debug": []}
    regime_counts: Dict[str, int] = {k: 0 for k in REGIME_RATIOS}
    written = []
    split_entries = split_map.setdefault(args.split_name, [])
    split_seen = set(split_entries)
    for j, (scenario_index, sid) in enumerate(tqdm(candidates, desc="write_roots", unit="scenario")):
        if args.max_roots is not None and len(written) >= args.max_roots:
            break
        if target_counts and all(regime_counts.get(k, 0) >= v for k, v in target_counts.items()):
            break
        pkl = scenario_file_path(scenario_dir, sid, mapping)
        # Must match MetaDrive ScenarioEnv's data-loading convention.  ScenarioEnv
        # centralizes ScenarioNet/WOMD coordinates to the SDC initial pose; root
        # JSON extracted from raw global coordinates will be kilometers away from
        # env.reset() and will fail teacher rollout alignment.
        scenario = read_scenario_description(pkl, centralize=True)
        sm = summary.get(sid, {}) or {}
        ticks = _sample_root_ticks(scenario, sm, args.history_steps, args.max_samples_per_log, args.sample_stride, args.event_aligned_root, args.event_lookahead_steps)
        for sample_j, tick in enumerate(ticks):
            if args.max_roots is not None and len(written) >= args.max_roots:
                break
            if args.max_samples_per_log == 1 and not args.event_aligned_root:
                root_id = f"{args.split_name}_{scenario_index:08d}"
            else:
                root_id = f"{args.split_name}_{scenario_index:08d}_t{int(tick):03d}"
            root = _root_json(root_id, scenario_dir, scenario_index, sid, pkl, scenario, sm, args.history_steps, root_tick=int(tick), event_lookahead_steps=args.event_lookahead_steps)
            if target_counts and regime_counts.get(root["regime"], 0) >= target_counts.get(root["regime"], 0):
                continue
            write_json(out / f"{root_id}.json", root)
            if root_id not in split_seen:
                split_entries.append(root_id)
                split_seen.add(root_id)
            regime_counts[root["regime"]] = regime_counts.get(root["regime"], 0) + 1
            written.append(root_id)
    write_json(split_path, split_map)
    meta_path = out / "metadata.json"
    old_meta = json.loads(meta_path.read_text()) if args.append and meta_path.exists() else {}
    metadata = {
        **old_meta,
        "backend": "metadrive_scenarionet_waymo",
        "scenario_dir": str(scenario_dir),
        "num_roots": sum(len(v) for v in split_map.values()),
        "split_by": "root_scene_id",
        "is_synthetic": False,
        "paper_final_ready": False,
        "requires_teacher_rollout": True,
        "max_samples_per_log_last_run": int(args.max_samples_per_log),
        "sample_stride_last_run": int(args.sample_stride),
        "event_aligned_root_last_run": bool(args.event_aligned_root),
        "event_lookahead_steps_last_run": int(args.event_lookahead_steps),
        "scenario_coordinate_frame": "metadrive_centralized_sdc_initial",
        "read_scenario_data_centralize": True,
        "temporal_roots_require_state_restore_for_metadrive_rollout": bool(args.max_samples_per_log > 1 or args.event_aligned_root),
        "regime_counts_last_run": regime_counts,
        "target_regime_counts_last_run": target_counts,
    }
    write_json(meta_path, metadata)
    print(json.dumps({"written": len(written), "split": args.split_name, "output": str(out), "regime_counts": regime_counts}, indent=2), flush=True)


if __name__ == "__main__":
    main()
