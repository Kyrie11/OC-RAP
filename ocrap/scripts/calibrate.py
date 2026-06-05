from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ocrap.data.dataset import OCRAPDataset, collate_batch
from ocrap.ocrap.io import write_json
from ocrap.ocrap.lcv import finite_sample_upper_quantile
from ocrap.ocrap.ocmero import torch_oc_mero
from ocrap.scripts.train import load_checkpoint, make_flags


def _to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def predict_recoverability(model, batch, cfg, flags):
    pred = model(batch)
    r_dep, r_orc, gap, q = torch_oc_mero(
        pred["margin"], pred["root_prob"], pred["C"],
        alpha=float(cfg.get("ocmero", {}).get("alpha", 0.2)),
        beta=float(cfg.get("ocmero", {}).get("beta", 0.2)),
        option_valid=batch["option_valid"].bool(),
        use_lcvar=bool(flags.get("use_lcvar", True)),
        use_obs_kernel=bool(flags.get("use_obs_kernel", True)),
    )
    return r_dep, r_orc, gap, pred, q


def calibrate(dataset_dir: str | Path, checkpoint: str | Path, output_path: str | Path, cfg_override: dict[str, Any] | None = None) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, flags = load_checkpoint(checkpoint, map_location=device)
    if cfg_override:
        cfg.update(cfg_override)
    model.to(device).eval()
    ds = OCRAPDataset(dataset_dir, split="calibration")
    if len(ds) == 0:
        raise ValueError("Calibration split is empty.")
    loader = DataLoader(ds, batch_size=int(cfg.get("evaluation", {}).get("batch_size", 64)), shuffle=False, collate_fn=collate_batch)
    pred_scores = []
    teacher_dep = []
    paths = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="calibrate"):
            batch = _to_device(batch, device)
            r_dep, _, _, _, _ = predict_recoverability(model, batch, cfg, flags)
            pred_scores.extend(r_dep.cpu().numpy().tolist())
            teacher_dep.extend(batch["r_dep_star"].float().cpu().numpy().reshape(-1).tolist())
            paths.extend(batch.get("_path", []))
    pred_scores_np = np.asarray(pred_scores, dtype=np.float64)
    teacher_dep_np = np.asarray(teacher_dep, dtype=np.float64)
    neg_scores = pred_scores_np[teacher_dep_np < 0.0]
    deltas = cfg.get("calibration", {}).get("deltas", [0.01, 0.05, 0.10])
    result = {"num_calibration": int(len(pred_scores_np)), "num_negative": int(len(neg_scores)), "thresholds": {}, "strict_finite_sample": bool(cfg.get("calibration", {}).get("strict", True))}
    for d in deltas:
        gamma = finite_sample_upper_quantile(neg_scores, float(d), numerical_margin=float(cfg.get("calibration", {}).get("numerical_margin", 0.0)), strict=bool(cfg.get("calibration", {}).get("strict", True)))
        result["thresholds"][str(d)] = gamma
    write_json(result, output_path)
    return result
