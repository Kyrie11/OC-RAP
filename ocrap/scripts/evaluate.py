from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ocrap.scripts.calibrate import predict_recoverability
from ocrap.data.dataset import OCRAPDataset, collate_batch, group_by_scene_time
from ocrap.ocrap.io import load_npz, read_json, write_json
from ocrap.ocrap.metrics import aggregate_records, evaluate_group_teacher_outcome
from ocrap.scripts.train import load_checkpoint, make_flags


def _to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _tensor_to_numpy_dict(batch, pred, r_dep, r_orc):
    B = r_dep.shape[0]
    rows = []
    for i in range(B):
        row = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                row[k] = v[i].detach().cpu().numpy()
            elif isinstance(v, list):
                row[k] = v[i]
            else:
                row[k] = v
        row["pred_r_dep"] = float(r_dep[i].detach().cpu())
        row["pred_r_orc"] = float(r_orc[i].detach().cpu())
        row["pred_margin"] = pred["margin"][i].detach().cpu().numpy()
        row["pred_prob"] = pred["root_prob"][i].detach().cpu().numpy()
        row["pred_C"] = pred["C"][i].detach().cpu().numpy()
        rows.append(row)
    return rows


def evaluate(dataset_dir: str | Path, checkpoint: str | Path, output_path: str | Path, split: str = "test", calibration_json: str | Path | None = None, cfg_override: dict[str, Any] | None = None) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, flags = load_checkpoint(checkpoint, map_location=device)
    if cfg_override:
        cfg.update(cfg_override)
    model.to(device).eval()
    if calibration_json:
        cal = read_json(calibration_json)
        delta = str(cfg.get("evaluation", {}).get("delta", cfg.get("calibration", {}).get("deltas", [0.05])[0]))
        gamma = float(cal["thresholds"].get(delta, next(iter(cal["thresholds"].values()))))
    elif flags.get("use_calibration", True):
        gamma = float(cfg.get("selection", {}).get("gamma_rec", 0.0))
    else:
        gamma = float(cfg.get("selection", {}).get("fixed_gamma_rec", 0.0))
    ds = OCRAPDataset(dataset_dir, split=split)
    loader = DataLoader(ds, batch_size=int(cfg.get("evaluation", {}).get("batch_size", 64)), shuffle=False, collate_fn=collate_batch)
    rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"eval_{split}"):
            batch = _to_device(batch, device)
            r_dep, r_orc, _, pred, _ = predict_recoverability(model, batch, cfg, flags)
            rows.extend(_tensor_to_numpy_dict(batch, pred, r_dep, r_orc))
    groups = group_by_scene_time(rows)
    methods = cfg.get("evaluation", {}).get("methods", ["nominal", "risk_aware", "backup_filter", "contingency", "oracle_filter", "ocrap"])
    result: dict[str, Any] = {"split": split, "gamma_rec": gamma, "methods": {}, "regime": {}}
    for method in methods:
        recs = []
        for _, group in groups.items():
            pred_r_dep = np.asarray([g["pred_r_dep"] for g in group])
            pred_r_orc = np.asarray([g["pred_r_orc"] for g in group])
            pred_margin = np.stack([g["pred_margin"] for g in group])
            pred_prob = np.stack([g["pred_prob"] for g in group])
            pred_C = np.stack([g["pred_C"] for g in group])
            recs.append(evaluate_group_teacher_outcome(group, pred_r_dep, pred_r_orc, pred_margin, pred_prob, pred_C, gamma, cfg, method=method))
        result["methods"][method] = aggregate_records(recs)
        # Multi-label regime-wise reports.
        regime_records: dict[str, list[dict[str, Any]]] = {}
        for r in recs:
            labels = r.get("regime_label", {})
            if isinstance(labels, np.ndarray):
                labels = labels.item() if labels.shape == () else {}
            if isinstance(labels, dict):
                for k, v in labels.items():
                    if v:
                        regime_records.setdefault(k, []).append(r)
        result["regime"][method] = {k: aggregate_records(v) for k, v in regime_records.items()}
    write_json(result, output_path)
    return result
