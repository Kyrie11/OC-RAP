from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ocrap.data.build.diagnose import iter_sample_paths
from ocrap.data.serialization import ensure_dir, load_npz, write_json


def train(dataset: str, output: str, cfg: dict) -> dict:
    out = ensure_dir(output)
    paths = iter_sample_paths(dataset)
    vals = []
    for p in paths:
        d = load_npz(p)
        if str(np.asarray(d.get("split_id", "train")).item()) in {"train", "val", "calibration", "test"}:
            vals.append(float(np.asarray(d["r_dep_star"]).item()))
    stats = {"mean_r_dep": float(np.mean(vals)) if vals else 0.0, "num_samples": len(vals)}
    ckpt = {"cfg": cfg, "stats": stats, "note": "teacher-label baseline checkpoint for OC-RAP pipeline validation"}
    torch.save(ckpt, out / "best.pt")
    write_json({"checkpoint": str(out / "best.pt"), **stats}, out / "train_summary.json")
    return {"checkpoint": str(out / "best.pt"), **stats}
