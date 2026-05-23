#!/usr/bin/env python
from __future__ import annotations

import argparse
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


def _min_actor_distance(ego, actors) -> float:
    if not actors:
        return float("inf")
    return float(min(np.hypot(a.x - ego.x, a.y - ego.y) for a in actors))


def _classify_regime(ego, actors, summary: dict) -> str:
    d = _min_actor_distance(ego, actors)
    moving = int((summary.get("number_summary", {}) or {}).get("num_moving_objects", 0) or 0)
    if d < 2.5:
        return "contact_post_contact"
    if d < 8.0:
        return "near_contact"
    if d < 18.0 or moving >= 32:
        return "low_headroom"
    return "normal_high_headroom"


def _moving_objects(summary: dict) -> int:
    return int((summary.get("number_summary", {}) or {}).get("num_moving_objects", 0) or 0)


def _num_objects(summary: dict) -> int:
    return int((summary.get("number_summary", {}) or {}).get("num_objects", 0) or 0)


def _root_json(root_id: str, scenario_dir: Path, scenario_index: int, scenario_id: str, scenario_pkl: Path, scenario: dict, summary: dict, history_steps: int, root_tick: int | None = None) -> dict:
    t = scenario_current_time_index(scenario, summary) if root_tick is None else int(root_tick)
    ego, actors = extract_root_state_from_scenario(scenario, t=t, summary=summary)
    mf = extract_map_features_from_scenario(scenario)
    route = extract_route_info_from_scenario(scenario, ego, summary=summary, t=t)
    hist = history_from_scenario(scenario, t, history_steps)
    regime = _classify_regime(ego, actors, summary)
    seed = abs(hash(str(scenario_id))) % (2**31 - 1)
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
        "scenario_data": {
            "data_directory": str(scenario_dir),
            "scenario_index": int(scenario_index),
            "scenario_id": str(scenario_id),
            "scenario_pkl": str(scenario_pkl),
            "current_time_index": int(t),
            "source_dataset": (summary.get("dataset") or (scenario.get("metadata", {}) or {}).get("dataset") or "waymo"),
            "source_file": summary.get("source_file") or (scenario.get("metadata", {}) or {}).get("source_file"),
            "sdc_id": summary.get("sdc_id") or (scenario.get("metadata", {}) or {}).get("sdc_id"),
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


def _sample_root_ticks(scenario: dict, summary: dict, history_steps: int, max_samples: int, stride: int) -> list[int]:
    current = int(scenario_current_time_index(scenario, summary))
    max_samples = max(1, int(max_samples))
    if max_samples == 1:
        return [current]
    T = _track_length(scenario, summary)
    if T <= 0:
        return [current]
    lo = max(0, int(history_steps) - 1)
    hi = max(lo + 1, T - 2)
    ticks = list(range(lo, hi + 1, max(1, int(stride))))
    if current not in ticks and lo <= current <= hi:
        ticks.append(current)
        ticks = sorted(set(ticks))
    if len(ticks) <= max_samples:
        return ticks
    # Deterministic uniform coverage of the log rather than adjacent highly-correlated frames.
    idx = np.linspace(0, len(ticks) - 1, max_samples).round().astype(int)
    return [ticks[int(i)] for i in idx]


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect ReCAP root JSON files from a MetaDrive/ScenarioNet real-world database.")
    ap.add_argument("--scenario-dir", required=True, help="ScenarioNet database containing dataset_summary.pkl.")
    ap.add_argument("--output", default="data/recap/roots_raw")
    ap.add_argument("--split-name", choices=["train", "calib", "test", "debug"], default="train")
    ap.add_argument("--max-roots", type=int, default=None)
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--min-moving-objects", type=int, default=1)
    ap.add_argument("--min-objects", type=int, default=2)
    ap.add_argument("--history-steps", type=int, default=10)
    ap.add_argument("--max-samples-per-log", type=int, default=1, help="Number of temporal roots to sample from each scenario/log. Use 1 for ScenarioEnv-compatible paper-final closed-loop rollouts unless you have verified root-time restore.")
    ap.add_argument("--sample-stride", type=int, default=5, help="Minimum frame stride between temporal roots when max-samples-per-log > 1.")
    ap.add_argument("--append", action="store_true", help="Append to existing root directory/splits instead of replacing split list.")
    args = ap.parse_args()
    scenario_dir = Path(args.scenario_dir)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    summary, scenario_ids, mapping = load_scenarionet_summary(scenario_dir)
    candidates: List[tuple[int, str]] = []
    for idx, sid in enumerate(tqdm(scenario_ids, desc="scan_scenarios", unit="scenario")):
        if idx < args.start_index:
            continue
        sm = summary.get(sid, {}) or {}
        if _moving_objects(sm) < args.min_moving_objects:
            continue
        if _num_objects(sm) < args.min_objects:
            continue
        candidates.append((idx, sid))
        # max_roots is a cap on output root JSON files, not scenarios.  For multi-tick sampling, apply again while writing.
        if args.max_roots is not None and len(candidates) * max(1, args.max_samples_per_log) >= args.max_roots:
            break
    split_path = out / "splits.json"
    if args.append and split_path.exists():
        split_map = json.loads(split_path.read_text())
    else:
        split_map = {"train": [], "calib": [], "test": [], "debug": []}
    regime_counts: Dict[str, int] = {k: 0 for k in REGIME_RATIOS}
    written = []
    split_entries = split_map.setdefault(args.split_name, [])
    split_seen = set(split_entries)
    for j, (scenario_index, sid) in enumerate(tqdm(candidates, desc="write_roots", unit="scenario")):
        if args.max_roots is not None and len(written) >= args.max_roots:
            break
        pkl = scenario_file_path(scenario_dir, sid, mapping)
        scenario = read_scenario_description(pkl)
        sm = summary.get(sid, {}) or {}
        ticks = _sample_root_ticks(scenario, sm, args.history_steps, args.max_samples_per_log, args.sample_stride)
        for sample_j, tick in enumerate(ticks):
            if args.max_roots is not None and len(written) >= args.max_roots:
                break
            if args.max_samples_per_log == 1:
                root_id = f"{args.split_name}_{scenario_index:08d}"
            else:
                root_id = f"{args.split_name}_{scenario_index:08d}_t{int(tick):03d}"
            root = _root_json(root_id, scenario_dir, scenario_index, sid, pkl, scenario, sm, args.history_steps, root_tick=int(tick))
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
        "regime_counts_last_run": regime_counts,
    }
    write_json(meta_path, metadata)
    print(json.dumps({"written": len(written), "split": args.split_name, "output": str(out), "regime_counts": regime_counts}, indent=2), flush=True)


if __name__ == "__main__":
    main()
