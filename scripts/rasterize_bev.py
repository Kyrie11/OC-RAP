#!/usr/bin/env python
from __future__ import annotations

# Allow running scripts directly from a source checkout without pip install -e .
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))


import argparse
from pathlib import Path
import json
import numpy as np
import yaml

from recap.raster.bev_builder import BEVBuilder, HistoryBuffer
from recap.raster.affordance_maps import AffordanceProvider
from recap.raster.debug_draw import write_channel_pngs
from recap.utils.datatypes import BEVSpec, EgoState, ActorState, MapFeatures, RouteInfo
from recap.teacher.dataset_writer import write_dataset
from scripts._common import load_config


def _load_actors(items):
    return [ActorState(**a) for a in (items or [])]


def _load_history(obj: dict, spec: BEVSpec) -> HistoryBuffer:
    hist = HistoryBuffer(spec.history_steps)
    if obj.get("history"):
        for h in obj["history"][-spec.history_steps:]:
            hist.push(EgoState(**h["ego_state"]), _load_actors(h.get("actor_states", [])))
    return hist


def load_root(path: Path):
    obj = json.loads(path.read_text())
    ego = EgoState(**obj["ego_state"])
    actors = _load_actors(obj["actor_states"])
    mfobj = obj["map_features"]
    mf = MapFeatures(
        [np.asarray(p, dtype=np.float32) for p in mfobj.get("drivable_polygons", [])],
        [np.asarray(p, dtype=np.float32) for p in mfobj.get("lane_centerlines", [])],
        [np.asarray(p, dtype=np.float32) for p in mfobj.get("lane_boundaries", [])],
        [np.asarray(p, dtype=np.float32) for p in mfobj.get("static_obstacles", [])],
        float(mfobj.get("speed_limit_mps", 13.9)),
    )
    robj = obj["route_info"]
    route = RouteInfo(np.asarray(robj["waypoints"], dtype=np.float32), np.asarray(robj.get("command_ids", np.zeros(len(robj["waypoints"]))), dtype=np.int64), float(robj.get("speed_limit_mps", 13.9)))
    return obj, ego, actors, mf, route


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root-dir", required=True)
    ap.add_argument("--split", default="all")
    ap.add_argument("--bev-config", default="configs/bev_256.yaml")
    ap.add_argument("--channels", default="compact")
    ap.add_argument("--history-steps", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--output", required=True)
    ap.add_argument("--save-debug", default="0")
    ap.add_argument("--write-channel-png", action="store_true")
    ap.add_argument("--debug-dir", default="outputs/debug_bev")
    ap.add_argument("--root-id", default=None)
    args = ap.parse_args()
    cfg = load_config(args.bev_config)
    bcfg = cfg.get("bev", {})
    spec = BEVSpec(H=int(bcfg.get("H", 256)), W=int(bcfg.get("W", 256)), range_x=tuple(bcfg.get("range_x", [-40.0, 40.0])), range_y=tuple(bcfg.get("range_y", [-40.0, 40.0])), history_steps=int(args.history_steps or bcfg.get("history_steps", 10)), dt=float(bcfg.get("dt", 0.2)), mode=bcfg.get("mode", args.channels))
    builder = BEVBuilder(spec)
    root_dir = Path(args.root_dir)
    split_file = root_dir / "splits.json"
    if args.root_id:
        ids = [args.root_id]
    elif split_file.exists() and args.split != "all":
        ids = json.loads(split_file.read_text()).get(args.split, [])
    else:
        ids = sorted(p.stem for p in root_dir.glob("root_*.json"))
    bevs=[]; ego_infos=[]; routes=[]; root_ids=[]
    for n, rid in enumerate(ids):
        obj, ego, actors, mf, route = load_root(root_dir / f"{rid}.json")
        hist = _load_history(obj, spec)
        if not hist.ego_history:
            # Synthetic/debug fallback: move actors/ego backward by constant velocity.
            for h in range(spec.history_steps - 1, -1, -1):
                e = EgoState(x=ego.x - ego.v * spec.dt * h, y=ego.y, heading=ego.heading, v=ego.v)
                aa = [ActorState(a.actor_id, a.x - a.vx * spec.dt * h, a.y - a.vy * spec.dt * h, a.heading, a.vx, a.vy, a.length, a.width, a.actor_type, a.dynamic) for a in actors]
                hist.push(e, aa)
        out = builder.build_from_state(ego, actors, mf, route, hist, AffordanceProvider())
        bevs.append(out["bev"]); ego_infos.append(out["ego_info"]); routes.append(out["route_command"]); root_ids.append(rid)
        if args.write_channel_png and (args.save_debug == "all" or n < int(args.save_debug or 0)):
            write_channel_pngs(out["bev"], out["debug"]["channel_names"], Path(args.debug_dir) / rid)
    arrays = {"bev": np.stack(bevs).astype(np.float16), "ego_info": np.stack(ego_infos).astype(np.float32), "route_command": np.stack(routes).astype(np.float32), "root_ids": np.asarray(root_ids)}
    metadata = {"bev_spec": spec.__dict__, "channel_names": builder.channel_names, "split": args.split, "root_dir": str(root_dir)}
    write_dataset(args.output, arrays, metadata)
    print(f"rasterized {len(root_ids)} roots to {args.output}")

if __name__ == "__main__":
    main()
