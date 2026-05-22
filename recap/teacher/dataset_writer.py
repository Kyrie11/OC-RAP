from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
import numpy as np

from recap.utils.serialization import write_json


def write_dataset(path: str | Path, arrays: Dict[str, Any], metadata: Dict[str, Any] | None = None) -> None:
    """Write a portable dataset.

    If `path` ends with .h5/.hdf5, write HDF5. Otherwise create a directory that
    mimics a zarr artifact using compressed NumPy arrays, so the code works even
    in minimal environments without zarr installed.  Installing zarr is still
    recommended for full-scale data.
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


def read_dataset(path: str | Path) -> tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    p = Path(path)
    if p.is_file() and p.suffix in (".h5", ".hdf5"):
        import h5py
        with h5py.File(p, "r") as f:
            arrays = {k: f[k][()] for k in f.keys()}
            metadata = json.loads(f.attrs.get("metadata_json", "{}"))
        return arrays, metadata
    npz = p / "arrays.npz"
    if not npz.exists():
        raise FileNotFoundError(f"dataset arrays not found at {npz}")
    with np.load(npz, allow_pickle=True) as data:
        arrays = {k: data[k] for k in data.files}
    meta_path = p / "metadata.json"
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return arrays, metadata
