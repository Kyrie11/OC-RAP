from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
import numpy as np
from .datatypes import dataclass_to_jsonable


def write_json(path: str | Path, obj: Any, indent: int = 2) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dataclass_to_jsonable(obj), indent=indent, sort_keys=True), encoding="utf-8")


def read_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_npz(path: str | Path, **arrays) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, **arrays)


def read_npz(path: str | Path) -> dict:
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def config_hash(obj: Any) -> str:
    import hashlib
    blob = json.dumps(dataclass_to_jsonable(obj), sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]
