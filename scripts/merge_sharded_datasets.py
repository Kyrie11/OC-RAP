#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from recap.teacher.dataset_writer import ShardedDatasetWriter, read_dataset


def _sample_at(arrays: dict[str, Any], idx: int) -> dict[str, Any]:
    out = {}
    for k, v in arrays.items():
        item = v[idx]
        if isinstance(item, np.ndarray) and item.shape == ():
            item = item.item()
        out[k] = item
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge multiple ReCAP sharded datasets produced by parallel build_teacher_labels parts.")
    ap.add_argument("--inputs", nargs="+", required=True, help="Input .zarr directories, e.g. train_parts/part_*.zarr")
    ap.add_argument("--output", required=True)
    ap.add_argument("--shard-size", type=int, default=4)
    ap.add_argument("--compress-shards", action="store_true")
    args = ap.parse_args()

    inputs = [Path(p) for p in args.inputs]
    if not inputs:
        raise ValueError("no input datasets")
    first_arrays, first_meta = read_dataset(inputs[0])
    expected_keys = set(first_arrays.keys())
    metadata = dict(first_meta)
    metadata.update({
        "merged_from": [str(p) for p in inputs],
        "merge_num_parts": len(inputs),
    })
    total = 0
    with ShardedDatasetWriter(args.output, metadata, shard_size=args.shard_size, compressed=args.compress_shards) as writer:
        for p in inputs:
            arrays, meta = read_dataset(p)
            keys = set(arrays.keys())
            if keys != expected_keys:
                raise ValueError(f"array key mismatch for {p}: missing={expected_keys-keys}, extra={keys-expected_keys}")
            n = len(next(iter(arrays.values()))) if arrays else 0
            for i in range(n):
                writer.append(_sample_at(arrays, i))
            total += n
    meta_path = Path(args.output) / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        meta["num_roots"] = total
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({"merged": total, "output": str(args.output), "parts": len(inputs)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
