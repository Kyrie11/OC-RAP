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
import numpy as np

from ocrap.envs.scenario_regimes import sample_regime
from ocrap.utils.datatypes import EgoState, ActorState, MapFeatures, RouteInfo, RootScene
from ocrap.utils.serialization import write_json
from scripts._common import load_config


def synthetic_root(root_id: str, seed: int, regime: str) -> dict:
    rng = np.random.default_rng(seed)
    ego = EgoState(x=0.0, y=0.0, heading=0.0, v=float(rng.uniform(5.0, 12.0)))
    actors = []
    if regime in ("low_headroom", "near_contact", "contact_post_contact"):
        actors.append(ActorState("lead", float(rng.uniform(12, 25)), float(rng.normal(0, 0.6)), 0.0, vx=float(rng.uniform(2, 8)), vy=0.0))
    if regime in ("near_contact", "contact_post_contact"):
        actors.append(ActorState("cutin", float(rng.uniform(8, 18)), float(rng.choice([-1, 1]) * rng.uniform(3.0, 5.0)), 0.0, vx=float(rng.uniform(4, 10)), vy=float(rng.choice([-1, 1]) * rng.uniform(0.5, 1.5))))
    drivable = np.array([[-80, -8], [120, -8], [120, 8], [-80, 8]], dtype=np.float32)
    center = np.stack([np.linspace(-80, 120, 100), np.zeros(100)], axis=-1).astype(np.float32)
    mf = MapFeatures([drivable], [center], [center + [0, 1.8], center + [0, -1.8]], [], 13.9)
    route = RouteInfo.straight(80, 40, 13.9)
    return {
        "root_id": root_id,
        "seed": seed,
        "regime": regime,
        "root_tick": 0,
        "ego_state": ego.__dict__,
        "actor_states": [a.__dict__ for a in actors],
        "map_features": {
            "drivable_polygons": [p.tolist() for p in mf.drivable_polygons],
            "lane_centerlines": [p.tolist() for p in mf.lane_centerlines],
            "lane_boundaries": [p.tolist() for p in mf.lane_boundaries],
            "static_obstacles": [],
            "speed_limit_mps": mf.speed_limit_mps,
        },
        "route_info": {"waypoints": route.waypoints.tolist(), "command_ids": route.command_ids.tolist(), "speed_limit_mps": route.speed_limit_mps},
        "map_config": {"synthetic": True},
        "traffic_config": {"synthetic": True},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--output", default="data/ocrap/roots_raw")
    ap.add_argument("--max-roots", type=int, default=None)
    ap.add_argument("--synthetic", type=str, default="true", help="Backward-compatible flag. Only true is implemented by this script.")
    ap.add_argument("--backend", choices=["synthetic", "metadrive"], default=None, help="Root collection backend. Real MetaDrive collection is not implemented in this script yet.")
    args = ap.parse_args()
    cfg = load_config(args.config)
    synthetic_flag = args.synthetic.lower() in ("1", "true", "yes")
    backend = args.backend or ("synthetic" if synthetic_flag else "metadrive")
    if backend != "synthetic":
        raise NotImplementedError(
            "Real MetaDrive root collection is not implemented in scripts/collect_roots.py. "
            "This script only creates synthetic schema/debug roots. Implement a ScenarioEnv/MetaDriveEnv "
            "collector with true simulator snapshots before claiming paper-final MetaDrive-Recovery data."
        )
    n = int(args.max_roots or cfg.get("dataset", {}).get("num_roots", 32))
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    root_ids = []
    split_map = {"train": [], "calib": [], "test": [], "debug": []}
    for i in range(n):
        regime = sample_regime(rng)
        rid = f"root_{i:06d}"
        root_ids.append(rid)
        split = "train" if i < int(0.70*n) else ("calib" if i < int(0.85*n) else "test")
        if n <= 16:
            split = "debug"
        split_map[split].append(rid)
        write_json(out / f"{rid}.json", synthetic_root(rid, int(rng.integers(0, 10_000_000)), regime))
    write_json(out / "splits.json", split_map)
    write_json(out / "metadata.json", {
        "backend": "metadrive_synthetic",
        "num_roots": n,
        "split_by": "root_scene_id",
        "implementation_level": cfg.get("implementation_level", "mvp"),
        "is_synthetic": True,
        "paper_final_ready": False,
    })
    print(f"wrote {n} roots to {out}")

if __name__ == "__main__":
    main()
