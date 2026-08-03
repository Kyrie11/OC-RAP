#!/usr/bin/env python3
"""Validate a WOMD/Waymax path or TensorFlow-style sharded path specification."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ocrap.data.womd.sharded_path import ensure_sharded_spec, resolve_womd_spec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--default-shards", type=int, default=0, help="Append @N to a bare prefix before validating; 0 disables.")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--print-normalized", action="store_true")
    args = ap.parse_args()
    spec = ensure_sharded_spec(args.spec, args.default_shards) if args.default_shards > 0 else args.spec
    resolution = resolve_womd_spec(spec)
    doc = {"event": "womd_sharded_spec_validation", "input_spec": args.spec, "normalized_spec": spec, **resolution.as_dict()}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.print_normalized:
        print(spec)
    else:
        print(json.dumps(doc, ensure_ascii=False))
    return 0 if resolution.valid else 3


if __name__ == "__main__":
    raise SystemExit(main())
