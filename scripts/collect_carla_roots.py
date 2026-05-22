#!/usr/bin/env python
from __future__ import annotations

# Allow running scripts directly from a source checkout without pip install -e .
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import argparse, json
from pathlib import Path
from recap.backends.carla_adapter import CarlaRecorderAdapter

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--recorder-file", required=True)
    ap.add_argument("--root-frame", type=int, required=True)
    ap.add_argument("--map-name", default="unknown")
    ap.add_argument("--carla-version", default="unknown")
    ap.add_argument("--traffic-manager-seed", type=int, default=None)
    ap.add_argument("--fork-support", action="store_true")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    adapter = CarlaRecorderAdapter()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    path = adapter.export_root_stub(args.recorder_file, args.root_frame, out / f"carla_root_{args.root_frame:08d}.json", args.map_name, args.carla_version, args.traffic_manager_seed, args.fork_support)
    meta = {"backend":"carla", "root_frame":args.root_frame, "fork_support":args.fork_support, "allowed_for_mero_teacher_labels":bool(args.fork_support)}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(path)
