from __future__ import annotations

import bisect
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional

import numpy as np

from ocrap.utils.serialization import write_json


SHARD_MANIFEST = "shards.json"


def _jsonable_meta(arrays_meta: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out = {}
    for k, v in arrays_meta.items():
        out[k] = {
            "shape": [int(x) for x in v["shape"]],
            "dtype": str(v["dtype"]),
        }
    return out


class ShardedArray:
    """Lazy, read-only array backed by per-shard npz files.

    It implements the small ndarray-like surface used by the training dataset:
    `.shape`, `.dtype`, integer indexing, slicing/list indexing by materializing
    the selected rows, and `np.asarray(obj)` for scripts that explicitly need the
    full array.  Avoid `np.asarray()` on the BEV tensor for full-scale datasets.
    """

    def __init__(self, root: str | Path, name: str, shards: list[dict[str, Any]], shape: tuple[int, ...], dtype: str, cache_size: int = 2):
        self.root = Path(root)
        self.name = name
        self.shards = shards
        self.shape = tuple(int(x) for x in shape)
        self.dtype = np.dtype(dtype)
        self._starts = [int(s["start"]) for s in shards]
        self._ends = [int(s["end"]) for s in shards]
        self._cache_size = int(cache_size)
        self._cache: OrderedDict[int, Dict[str, np.ndarray]] = OrderedDict()

    def __len__(self) -> int:
        return self.shape[0]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def tolist(self):
        return [self[i].item() if np.asarray(self[i]).shape == () else self[i] for i in range(len(self))]

    def iter_shard_arrays(self) -> Iterator[np.ndarray]:
        for shard_idx in range(len(self.shards)):
            yield self._load_shard(shard_idx)[self.name]

    def _load_shard(self, shard_idx: int) -> Dict[str, np.ndarray]:
        if shard_idx in self._cache:
            self._cache.move_to_end(shard_idx)
            return self._cache[shard_idx]
        path = self.root / self.shards[shard_idx]["file"]
        with np.load(path, allow_pickle=True) as data:
            if self.name not in data.files:
                raise KeyError(f"array {self.name!r} not found in shard {path}")
            # Load only this array.  Earlier versions loaded every array in the
            # shard, so asking for a scalar field such as root_ids also
            # decompressed the full BEV tensor.  That made health reports and
            # lazy training unexpectedly memory- and I/O-heavy.
            loaded = {self.name: data[self.name]}
        self._cache[shard_idx] = loaded
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return loaded

    def _locate(self, idx: int) -> tuple[int, int]:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        shard_idx = bisect.bisect_right(self._starts, idx) - 1
        if shard_idx < 0 or idx >= self._ends[shard_idx]:
            raise IndexError(idx)
        return shard_idx, idx - self._starts[shard_idx]

    def __getitem__(self, index):
        rest = ()
        if isinstance(index, tuple):
            if len(index) == 0:
                return np.asarray(self)
            index, rest = index[0], index[1:]
        if isinstance(index, (int, np.integer)):
            shard_idx, local = self._locate(int(index))
            out = self._load_shard(shard_idx)[self.name][local]
            return out[rest] if rest else out
        if isinstance(index, slice):
            rng = range(*index.indices(len(self)))
            out = np.asarray([self[i] for i in rng], dtype=self.dtype)
            return out[(slice(None),) + rest] if rest else out
        arr = np.asarray(index)
        if arr.dtype == bool:
            arr = np.where(arr)[0]
        out = np.asarray([self[int(i)] for i in arr.reshape(-1)], dtype=self.dtype)
        return out[(slice(None),) + rest] if rest else out

    def __array__(self, dtype=None):
        parts = []
        for shard_idx in range(len(self.shards)):
            parts.append(self._load_shard(shard_idx)[self.name])
        out = np.concatenate(parts, axis=0) if parts else np.empty(self.shape, dtype=self.dtype)
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def astype(self, dtype):
        return np.asarray(self).astype(dtype)

    def max(self, *args, **kwargs):
        return np.asarray(self).max(*args, **kwargs)

    def min(self, *args, **kwargs):
        return np.asarray(self).min(*args, **kwargs)

    def mean(self, *args, **kwargs):
        return np.asarray(self).mean(*args, **kwargs)


class ShardedDatasetWriter:
    """Append samples and flush them to bounded-size npz shards."""

    def __init__(self, path: str | Path, metadata: Optional[Dict[str, Any]] = None, shard_size: int = 8, compressed: bool = False):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.shard_dir = self.path / "shards"
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = metadata or {}
        self.shard_size = max(1, int(shard_size))
        self.compressed = bool(compressed)
        self.buffer: Dict[str, list[Any]] = {}
        self.shards: list[dict[str, Any]] = []
        self.arrays_meta: Dict[str, Dict[str, Any]] = {}
        self.count = 0
        self._closed = False

    def append(self, sample: Mapping[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("cannot append after close")
        for k, v in sample.items():
            self.buffer.setdefault(k, []).append(v)
        if self._buffer_len() >= self.shard_size:
            self.flush()

    def _buffer_len(self) -> int:
        if not self.buffer:
            return 0
        return len(next(iter(self.buffer.values())))

    def flush(self) -> None:
        n = self._buffer_len()
        if n == 0:
            return
        arrays = {k: np.stack(v, axis=0) if not isinstance(v[0], (str, bytes, np.str_)) else np.asarray(v) for k, v in self.buffer.items()}
        start = self.count
        end = self.count + n
        fname = f"shard_{len(self.shards):06d}.npz"
        tmp = self.shard_dir / (fname + ".tmp")
        final = self.shard_dir / fname
        saver = np.savez_compressed if self.compressed else np.savez
        saver(tmp, **arrays)
        # np.savez appends .npz if needed.
        real_tmp = tmp if tmp.exists() else Path(str(tmp) + ".npz")
        real_tmp.replace(final)
        for k, arr in arrays.items():
            shape = (end,) + tuple(arr.shape[1:])
            if k not in self.arrays_meta:
                self.arrays_meta[k] = {"shape": shape, "dtype": arr.dtype}
            else:
                old = self.arrays_meta[k]
                if tuple(old["shape"][1:]) != tuple(arr.shape[1:]):
                    raise ValueError(f"array {k} shard shape changed from {old['shape']} to {shape}")
                old["shape"] = shape
        self.shards.append({"file": str(Path("shards") / fname), "start": start, "end": end, "n": n})
        self.count = end
        self.buffer = {}
        self._write_manifest()

    def _write_manifest(self) -> None:
        manifest = {
            "format": "sharded_npz_v1",
            "num_samples": int(self.count),
            "shard_size": int(self.shard_size),
            "compressed": bool(self.compressed),
            "arrays": _jsonable_meta(self.arrays_meta),
            "shards": self.shards,
            "metadata": self.metadata,
        }
        write_json(self.path / SHARD_MANIFEST, manifest)
        write_json(self.path / "metadata.json", self.metadata)

    def close(self) -> None:
        if self._closed:
            return
        self.flush()
        self._write_manifest()
        self._closed = True

    def __enter__(self) -> "ShardedDatasetWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def write_dataset(path: str | Path, arrays: Dict[str, Any], metadata: Dict[str, Any] | None = None) -> None:
    """Write a portable dataset.

    Small datasets are written as HDF5 or a single compressed npz directory for
    backwards compatibility.  Full-scale BEV/teacher datasets should use
    `ShardedDatasetWriter`, because a single npz materializes the full tensor in
    RAM and is not viable for 50k+ roots.
    """
    p = Path(path)
    metadata = metadata or {}
    if p.suffix in (".h5", ".hdf5"):
        import h5py
        p.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(p, "w") as f:
            for k, v in arrays.items():
                f.create_dataset(k, data=np.asarray(v), compression="gzip")
            f.attrs["metadata_json"] = json.dumps(metadata)
        return
    p.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p / "arrays.npz", **{k: np.asarray(v) for k, v in arrays.items()})
    write_json(p / "metadata.json", metadata)


def read_dataset(path: str | Path) -> tuple[Dict[str, Any], Dict[str, Any]]:
    p = Path(path)
    manifest_path = p / SHARD_MANIFEST
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        shards = manifest.get("shards", [])
        arrays_meta = manifest.get("arrays", {})
        arrays = {
            k: ShardedArray(p, k, shards, tuple(v["shape"]), v["dtype"])
            for k, v in arrays_meta.items()
        }
        metadata = manifest.get("metadata", {}) or {}
        return arrays, metadata
    if p.is_file() and p.suffix in (".h5", ".hdf5"):
        import h5py
        with h5py.File(p, "r") as f:
            arrays = {k: f[k][()] for k in f.keys()}
            metadata = json.loads(f.attrs.get("metadata_json", "{}"))
        return arrays, metadata
    npz = p / "arrays.npz"
    if not npz.exists():
        raise FileNotFoundError(f"dataset arrays not found at {npz} or sharded manifest {manifest_path}")
    with np.load(npz, allow_pickle=True) as data:
        arrays = {k: data[k] for k in data.files}
    meta_path = p / "metadata.json"
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return arrays, metadata
