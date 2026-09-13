#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ocrap.evaluation.contact_anchor import select_one_anchor_per_scene


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a scene-disjoint pre-treatment Contact anchor manifest.")
    ap.add_argument("--mining-result", type=Path, required=True)
    ap.add_argument("--min-post-steps", type=int, default=20)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--target-keys-output", type=Path, required=True)
    a = ap.parse_args()
    data = json.loads(a.mining_result.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit("mining result must be a JSON object")
    manifest = select_one_anchor_per_scene(data, min_post_steps=int(a.min_post_steps))
    manifest["source_mining_result"] = str(a.mining_result.resolve())
    manifest["source_method"] = str(data.get("method") or "")
    manifest["source_bucket_dataset"] = data.get("bucket_dataset")
    manifest["source_womd_pattern"] = data.get("dataset") or data.get("womd_pattern")
    manifest["source_bucket_target_count"] = int(data.get("bucket_target_count") or 0)
    if str(data.get("method") or "").lower() != "nominal":
        manifest["valid"] = False
        manifest["error"] = "anchor mining must use exact nominal method"
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    keys = manifest.get("target_keys") or []
    a.target_keys_output.parent.mkdir(parents=True, exist_ok=True)
    a.target_keys_output.write_text(json.dumps(keys, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "valid": bool(manifest.get("valid")),
        "num_selected_anchors": manifest.get("num_selected_anchors"),
        "num_selected_scenes": manifest.get("num_selected_scenes"),
        "min_post_steps": manifest.get("min_post_steps"),
    }))
    return 0 if manifest.get("valid") else 30


if __name__ == "__main__":
    raise SystemExit(main())
