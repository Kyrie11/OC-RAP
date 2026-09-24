#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from ocrap.config.yaml_io import load_config
from ocrap.simulation.closed_loop_runner import _load_closed_loop_targets


def _score(seed: int, key: str) -> str:
    return hashlib.sha256(f"{seed}|{key}".encode("utf-8")).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build a deterministic, broad Contact validation diagnostic target subset without running Waymax."
    )
    ap.add_argument("--bucket-dataset", required=True)
    ap.add_argument("--bucket-split", default="val")
    ap.add_argument("--num-targets", type=int, default=100)
    ap.add_argument("--max-targets-per-scene", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.num_targets <= 0 or args.max_targets_per_scene <= 0:
        raise SystemExit("num-targets and max-targets-per-scene must be positive")

    cfg = load_config(None)
    cfg.setdefault("closed_loop", {})
    cfg["closed_loop"].update({
        "bucket_split": str(args.bucket_split),
        "max_targets_per_scene": 1000000,
        "max_bucket_targets": 0,
        "target_keys_file": "",
        "require_target_keys": False,
    })
    rows = _load_closed_loop_targets(str(args.bucket_dataset), cfg)
    if not rows:
        raise SystemExit("no validation targets found")

    ordered = sorted(rows, key=lambda r: (_score(args.seed, str(r["target_key"])), str(r["target_key"])))
    per_scene: dict[str, int] = defaultdict(int)
    chosen = []
    for row in ordered:
        sid = str(row.get("scene_id") or "")
        if per_scene[sid] >= args.max_targets_per_scene:
            continue
        chosen.append(row)
        per_scene[sid] += 1
        if len(chosen) >= args.num_targets:
            break
    if not chosen:
        raise SystemExit("deterministic subset is empty")

    out = {
        "schema": "ocrap-contact-validation-diagnostic-target-subset-v1",
        "diagnostic_only": True,
        "bucket_dataset": str(Path(args.bucket_dataset).resolve()),
        "bucket_split": str(args.bucket_split),
        "seed": int(args.seed),
        "requested_num_targets": int(args.num_targets),
        "available_num_targets": len(rows),
        "selected_num_targets": len(chosen),
        "selected_num_scenes": len({str(r.get("scene_id") or "") for r in chosen}),
        "max_targets_per_scene": int(args.max_targets_per_scene),
        "sampling_policy": "sha256(seed|target_key)_ascending_with_per_scene_cap",
        "target_keys": [str(r["target_key"]) for r in chosen],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in (
        "schema", "selected_num_targets", "selected_num_scenes", "available_num_targets", "seed"
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
