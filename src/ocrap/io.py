from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=True)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def np_savez(path: str | Path, **arrays: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    serializable: dict[str, Any] = {}
    for k, v in arrays.items():
        if isinstance(v, (dict, list, tuple, str)):
            serializable[k] = np.array(v, dtype=object)
        else:
            serializable[k] = v
    np.savez_compressed(path, **serializable)


def load_npz(path: str | Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        out = {k: data[k] for k in data.files}
    for k, v in list(out.items()):
        if isinstance(v, np.ndarray) and v.dtype == object and v.shape == ():
            out[k] = v.item()
    return out
