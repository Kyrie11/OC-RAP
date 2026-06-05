from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .calibrate import predict_recoverability
from .dataset import OCRAPDataset, collate_batch
from .io import read_json, write_json
from .selector import crisp_select
from .train import load_checkpoint


def _to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def deploy(dataset_dir: str | Path, checkpoint: str | Path, scene_id: str, time_index: int, output_path: str | Path, calibration_json: str | Path | None = None, delta: float | None = None) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, flags = load_checkpoint(checkpoint, map_location=device)
    model.to(device).eval()
    ds = OCRAPDataset(dataset_dir, split=None)
    idxs = [i for i, row in enumerate(ds.rows) if row["scene_id"] == scene_id and int(row["time_index"]) == int(time_index)]
    if not idxs:
        raise ValueError(f"No candidates found for scene_id={scene_id} time_index={time_index}")
    subset = Subset(ds, idxs)
    loader = DataLoader(subset, batch_size=len(idxs), shuffle=False, collate_fn=collate_batch)
    batch = next(iter(loader))
    batch = _to_device(batch, device)
    if calibration_json:
        cal = read_json(calibration_json)
        d = str(delta if delta is not None else cfg.get("evaluation", {}).get("delta", 0.05))
        gamma = float(cal["thresholds"].get(d, next(iter(cal["thresholds"].values()))))
    elif flags.get("use_calibration", True):
        gamma = float(cfg.get("selection", {}).get("gamma_rec", 0.0))
    else:
        gamma = float(cfg.get("selection", {}).get("fixed_gamma_rec", 0.0))
    with torch.no_grad():
        r_dep, r_orc, gap, pred, q = predict_recoverability(model, batch, cfg, flags)
    utility = batch["utility"].detach().cpu().numpy().reshape(-1)
    hard = batch["hard_violation"].detach().cpu().numpy().reshape(-1)
    harm = batch["harm_proxy"].detach().cpu().numpy().reshape(-1)
    feasible = batch["feasible"].detach().cpu().numpy().reshape(-1).astype(bool)
    dep = r_dep.detach().cpu().numpy().reshape(-1)
    orc = r_orc.detach().cpu().numpy().reshape(-1)
    sel = crisp_select(utility, dep, hard, harm, feasible, gamma, gamma_H=float(cfg.get("selection", {}).get("gamma_H", 0.0)), gamma_D=float(cfg.get("selection", {}).get("gamma_D", 5.0)))
    result = {
        "scene_id": scene_id,
        "time_index": int(time_index),
        "gamma_rec": gamma,
        "selected_candidate_index": int(sel.selected_index),
        "admitted_indices": sel.admitted_indices,
        "reason": sel.reason,
        "candidates": [
            {
                "candidate_index": int(torch.as_tensor(batch["candidate_index"])[i].detach().cpu()),
                "macro_name": batch["macro_name"][i] if isinstance(batch.get("macro_name"), list) else "unknown",
                "utility": float(utility[i]),
                "r_dep_pred": float(dep[i]),
                "r_orc_pred": float(orc[i]),
                "hard_violation": float(hard[i]),
                "harm_proxy": float(harm[i]),
                "feasible": bool(feasible[i]),
            }
            for i in range(len(dep))
        ],
    }
    write_json(result, output_path)
    return result
