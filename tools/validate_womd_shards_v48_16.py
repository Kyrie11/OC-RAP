#!/usr/bin/env python3
"""Fail-fast validation for TensorFlow-style `prefix@N` WOMD shard specs."""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--expected-shards", type=int, default=None)
    args = ap.parse_args()
    match = re.fullmatch(r"(.+)@(\d+)", args.source)
    if not match:
        raise SystemExit(f"WOMD source must use TensorFlow prefix@N syntax: {args.source}")
    prefix, count_text = match.groups()
    declared = int(count_text)
    if args.expected_shards is not None and declared != args.expected_shards:
        raise SystemExit(f"declared shard count {declared} != expected {args.expected_shards}: {args.source}")
    base = Path(prefix)
    pattern = base.name + f"-*-of-{declared:05d}"
    files = sorted(base.parent.glob(pattern))
    if len(files) != declared:
        raise SystemExit(
            f"WOMD shard preflight failed: found {len(files)}/{declared} files for {base.parent / pattern}"
        )
    print(json.dumps({"source": args.source, "declared_shards": declared, "found_shards": len(files), "valid": True}))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
