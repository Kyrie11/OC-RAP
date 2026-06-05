from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from ocrap.ocrap.config import save_config
from ocrap.data.dataset import OCRAPDataset, collate_batch
from ocrap.ocrap.io import ensure_dir, write_json
from ocrap.ocrap.losses import compute_losses
from ocrap.ocrap.model import OCRAPModel


def make_flags(cfg: dict[str, Any], **overrides) -> dict[str, Any]:
    ab = cfg.get("ablation", {})
    flags = {
        "use_obs_kernel": not bool(ab.get("without_observation_kernel", False)),
        "use_lcvar": not bool(ab.get("without_lower_tail", False)),
        "use_calibration": not bool(ab.get("without_calibration", False)),
        "use_anti_oracle": not bool(ab.get("without_anti_oracle", False)),
        "root_target_mode": "full_future" if bool(ab.get("full_future_roots", False)) else cfg.get("training", {}).get("root_target_mode", "recovery_signature"),
    }
    flags.update(overrides)
    return flags


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def _artifact_sampler(ds: OCRAPDataset, artifact_fraction: float) -> WeightedRandomSampler | None:
    if artifact_fraction <= 0 or len(ds) == 0:
        return None
    weights = []
    for row in ds.rows:
        w = 1.0
        if str(row.get("i_art_star", "0")) in {"1", "1.0", "True", "true"}:
            w = max(w, artifact_fraction / max(1e-6, 1.0 - artifact_fraction))
        weights.append(w)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def train(dataset_dir: str | Path, output_dir: str | Path, cfg: dict[str, Any]) -> dict[str, Any]:
    out = ensure_dir(output_dir)
    save_config(cfg, out / "config.yaml")
    dev_cfg = cfg.get("training", {}).get("device", "auto")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if dev_cfg == "auto" else dev_cfg)
    train_ds = OCRAPDataset(dataset_dir, split="train")
    val_ds = OCRAPDataset(dataset_dir, split="val")
    if len(train_ds) == 0:
        raise ValueError("Training split is empty. Rebuild dataset with enough scenarios or adjust split ratios.")
    batch_size = int(cfg.get("training", {}).get("batch_size", 16))
    sampler = _artifact_sampler(train_ds, float(cfg.get("training", {}).get("artifact_sampler_weight", 0.25)))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler, collate_fn=collate_batch, num_workers=int(cfg.get("training", {}).get("num_workers", 0)))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch, num_workers=int(cfg.get("training", {}).get("num_workers", 0))) if len(val_ds) else None
    model = OCRAPModel(cfg).to(device)
    flags = make_flags(cfg)
    # Initialize lazy modules.
    first = _to_device(next(iter(train_loader)), device)
    with torch.no_grad():
        _ = model(first)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("training", {}).get("lr", 1e-3)), weight_decay=float(cfg.get("training", {}).get("weight_decay", 1e-4)))
    best_val = float("inf")
    history = []
    epochs = int(cfg.get("training", {}).get("epochs", 10))
    grad_clip = float(cfg.get("training", {}).get("grad_clip", 5.0))
    for epoch in range(1, epochs + 1):
        model.train()
        logs_accum: dict[str, list[float]] = {}
        for batch in tqdm(train_loader, desc=f"train_epoch_{epoch}"):
            batch = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch)
            loss, logs, _ = compute_losses(pred, batch, cfg, flags)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            for k, v in logs.items():
                logs_accum.setdefault(k, []).append(float(v))
        train_logs = {f"train_{k}": float(np.mean(v)) for k, v in logs_accum.items()}
        val_logs = {}
        if val_loader is not None:
            model.eval()
            vals: dict[str, list[float]] = {}
            with torch.no_grad():
                for batch in val_loader:
                    batch = _to_device(batch, device)
                    pred = model(batch)
                    loss, logs, _ = compute_losses(pred, batch, cfg, flags)
                    for k, v in logs.items():
                        vals.setdefault(k, []).append(float(v))
            val_logs = {f"val_{k}": float(np.mean(v)) for k, v in vals.items()}
        record = {"epoch": epoch, **train_logs, **val_logs}
        history.append(record)
        write_json(history, out / "train_history.json")
        val_score = val_logs.get("val_loss_total", train_logs.get("train_loss_total", float("inf")))
        ckpt = {"model": model.state_dict(), "cfg": cfg, "epoch": epoch, "flags": flags, "val_score": val_score}
        torch.save(ckpt, out / "last.pt")
        if val_score <= best_val:
            best_val = val_score
            torch.save(ckpt, out / "best.pt")
    return {"output_dir": str(out), "best_val": best_val, "epochs": epochs}


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> tuple[OCRAPModel, dict[str, Any], dict[str, Any]]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    cfg = ckpt["cfg"]
    model = OCRAPModel(cfg)
    # Initialize lazy layers with shapes from saved weights by loading directly; PyTorch Lazy modules materialize on load_state_dict.
    model.load_state_dict(ckpt["model"], strict=True)
    flags = ckpt.get("flags", make_flags(cfg))
    model.eval()
    return model, cfg, flags
