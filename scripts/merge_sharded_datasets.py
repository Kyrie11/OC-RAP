#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from ocrap.teacher.dataset_writer import ShardedDatasetWriter, read_dataset
from ocrap.utils.progress import tqdm


def _natural_key(path: Path) -> list[Any]:
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", path.name)]


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

    inputs = sorted((Path(p) for p in args.inputs), key=_natural_key)
    if not inputs:
        raise ValueError("no input datasets")
    missing = [str(p) for p in inputs if not p.exists()]
    if missing:
        raise FileNotFoundError(f"input datasets do not exist: {missing}")
    first_arrays, first_meta = read_dataset(inputs[0])
    expected_keys = set(first_arrays.keys())
    metadata = dict(first_meta)
    metadata.update({
        "merged_from": [str(p) for p in inputs],
        "merge_num_parts": len(inputs),
        "root_start": 0,
        "root_stride": 1,
    })
    total = 0
    seen_root_ids: set[str] = set()
    with ShardedDatasetWriter(args.output, metadata, shard_size=args.shard_size, compressed=args.compress_shards) as writer:
        for p in inputs:
            arrays, meta = read_dataset(p)
            keys = set(arrays.keys())
            if keys != expected_keys:
                raise ValueError(f"array key mismatch for {p}: missing={expected_keys-keys}, extra={keys-expected_keys}")
            n = len(next(iter(arrays.values()))) if arrays else 0
            if "root_ids" in arrays:
                root_ids = [str(x) for x in np.asarray(arrays["root_ids"]).astype(str).tolist()]
                dup = sorted(r for r in root_ids if r in seen_root_ids)
                if dup:
                    raise ValueError(f"duplicate root_ids while merging {p}: {dup[:10]}")
                seen_root_ids.update(root_ids)
            for i in tqdm(range(n), desc=f"merge {p.name}", unit="root", leave=False):
                writer.append(_sample_at(arrays, i))
            total += n
    out = Path(args.output)
    meta_path = out / "metadata.json"
    manifest_path = out / "shards.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        meta["num_roots"] = total
        meta["root_end"] = total
        meta["selected_root_count"] = total
        meta["num_roots_full_selected_split"] = total
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest.setdefault("metadata", {})
        manifest["metadata"]["num_roots"] = total
        manifest["metadata"]["root_end"] = total
        manifest["metadata"]["selected_root_count"] = total
        manifest["metadata"]["num_roots_full_selected_split"] = total
        manifest["metadata"]["merged_from"] = [str(p) for p in inputs]
        manifest["metadata"]["merge_num_parts"] = len(inputs)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"merged": total, "output": str(args.output), "parts": len(inputs)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
