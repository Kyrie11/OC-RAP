from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .io import load_npz


TENSOR_KEYS = [
    "agent_history",
    "agent_valid",
    "map_polylines",
    "map_valid",
    "dynamic_map",
    "route",
    "bev_occ",
    "ego_state",
    "prefix_states",
    "prefix_controls",
    "time_index",
    "candidate_index",
    "prefix_macro_id",
    "prefix_param",
    "utility",
    "hard_violation",
    "harm_proxy",
    "feasible",
    "future_probs",
    "root_assignments",
    "root_probs",
    "root_signature",
    "root_future_signature",
    "root_valid",
    "future_to_root_weight",
    "y_obs",
    "c_star",
    "m_star",
    "option_valid",
    "r_orc_star",
    "r_dep_star",
    "oracle_gap_star",
    "i_art_star",
    "is_nominal",
]


class OCRAPDataset(Dataset):
    def __init__(self, dataset_dir: str | Path, split: str | None = None):
        self.dataset_dir = Path(dataset_dir)
        manifest = self.dataset_dir / "manifest.csv"
        if not manifest.exists():
            raise FileNotFoundError(f"Missing manifest: {manifest}")
        rows = []
        with manifest.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if split is None or row.get("split_id") == split:
                    rows.append(row)
        self.rows = rows
        self.paths = [self.dataset_dir / r["path"] for r in rows]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        data = load_npz(self.paths[idx])
        out: dict[str, Any] = {"_path": str(self.paths[idx])}
        for k, v in data.items():
            if k in TENSOR_KEYS:
                arr = np.asarray(v)
                if arr.dtype.kind in "USO":
                    out[k] = v
                else:
                    out[k] = torch.as_tensor(arr)
            else:
                out[k] = v
        return out


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    keys = set().union(*(b.keys() for b in batch))
    for k in keys:
        vals = [b[k] for b in batch if k in b]
        if vals and all(torch.is_tensor(v) for v in vals):
            try:
                out[k] = torch.stack(vals, dim=0)
            except Exception:
                out[k] = vals
        else:
            out[k] = vals
    return out


def group_by_scene_time(samples: list[dict[str, Any]]) -> dict[tuple[str, int], list[dict[str, Any]]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for s in samples:
        scene = s["scene_id"] if isinstance(s.get("scene_id"), str) else str(s.get("scene_id"))
        t = int(torch.as_tensor(s["time_index"]).item()) if torch.is_tensor(s.get("time_index")) else int(s["time_index"])
        groups.setdefault((scene, t), []).append(s)
    for vals in groups.values():
        vals.sort(key=lambda x: int(torch.as_tensor(x["candidate_index"]).item()) if torch.is_tensor(x.get("candidate_index")) else int(x["candidate_index"]))
    return groups
