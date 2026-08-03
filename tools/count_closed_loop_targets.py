#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ocrap.simulation.closed_loop_runner import _load_closed_loop_targets


def main() -> int:
    ap = argparse.ArgumentParser(description="Count unique scene/time closed-loop targets in an OC-RAP dataset root.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-targets-per-scene", type=int, default=1)
    ap.add_argument("--target-keys-file", type=Path)
    ap.add_argument("--count-only", action="store_true")
    args = ap.parse_args()
    cfg = {"closed_loop": {
        "bucket_split": args.split,
        "max_bucket_targets": 0,
        "max_targets_per_scene": args.max_targets_per_scene,
        "target_keys_file": str(args.target_keys_file) if args.target_keys_file else "",
    }}
    targets = _load_closed_loop_targets(args.dataset, cfg)
    if args.count_only:
        print(len(targets))
    else:
        print(json.dumps({
            "dataset": args.dataset,
            "split": args.split,
            "count": len(targets),
            "unique_scenes": len({x["scene_id"] for x in targets}),
            "buckets": sorted({x["bucket_name"] for x in targets}),
            "examples": targets[:5],
        }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
