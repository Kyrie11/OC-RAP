from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ocrap.data.serialization import ensure_dir, write_json
from ocrap.external_baselines.data import ExternalGroupDataset
from ocrap.external_baselines.models import build_model_from_cfg
from ocrap.utils.seed import seed_everything


def _device(cfg: dict[str, Any]) -> torch.device:
    bcfg = cfg.get("external_baselines", {}) if isinstance(cfg.get("external_baselines", {}), dict) else {}
    tcfg = bcfg.get("training", {}) if isinstance(bcfg.get("training", {}), dict) else {}
    requested = str(tcfg.get("device", (cfg.get("training", {}) or {}).get("device", "auto")))
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in batch[0]}


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.bool() & torch.isfinite(target)
    if not bool(mask.any()):
        return pred.sum() * 0.0
    return F.smooth_l1_loss(pred[mask], target[mask])


def _epoch(model, loader, opt, device, cfg: dict[str, Any], train: bool) -> dict[str, float]:
    model.train(train)
    bcfg = cfg.get("external_baselines", {}) if isinstance(cfg.get("external_baselines", {}), dict) else {}
    lw = bcfg.get("loss_weights", {}) if isinstance(bcfg.get("loss_weights", {}), dict) else {}
    totals: dict[str, float] = {}
    n = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch["x"].float(), batch["mask"].bool())
        loss_policy = F.cross_entropy(out["logits"], batch["target_index"].long())
        mask = batch["mask"].bool()
        loss_utility = _masked_mse(out["utility"], batch["utility"].float(), mask)
        loss_hard = _masked_mse(out["hard"], batch["hard"].float(), mask)
        loss_harm = _masked_mse(out["harm"], batch["harm"].float(), mask)
        loss_orc = _masked_mse(out["r_orc"], batch["r_orc"].float(), mask)
        loss_dep = _masked_mse(out["r_dep"], batch["r_dep"].float(), mask)
        loss = (
            float(lw.get("policy", 1.0)) * loss_policy
            + float(lw.get("utility", 0.10)) * loss_utility
            + float(lw.get("hard", 0.50)) * loss_hard
            + float(lw.get("harm", 0.25)) * loss_harm
            + float(lw.get("oracle_rec", 0.50)) * loss_orc
            + float(lw.get("deploy_rec", 0.25)) * loss_dep
        )
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float((bcfg.get("training", {}) or {}).get("grad_clip", 5.0)))
            opt.step()
        bs = int(batch["x"].shape[0])
        n += bs
        for name, val in {
            "loss": loss,
            "loss_policy": loss_policy,
            "loss_utility": loss_utility,
            "loss_hard": loss_hard,
            "loss_harm": loss_harm,
            "loss_oracle_rec": loss_orc,
            "loss_deploy_rec": loss_dep,
        }.items():
            totals[name] = totals.get(name, 0.0) + float(val.detach().cpu()) * bs
        with torch.no_grad():
            pred = torch.argmax(out["logits"], dim=-1)
            acc = (pred == batch["target_index"]).float().mean()
            totals["target_acc"] = totals.get("target_acc", 0.0) + float(acc.detach().cpu()) * bs
    return {k: v / max(n, 1) for k, v in totals.items()}


def train_external_baseline(dataset: str, output: str, cfg: dict[str, Any], *, val_dataset: str | None = None, baseline: str | None = None) -> dict[str, Any]:
    seed_everything(int(cfg.get("seed", 7)))
    bcfg = cfg.setdefault("external_baselines", {})
    if baseline:
        bcfg["baseline"] = baseline
    baseline_name = str(bcfg.get("baseline", "route_bc_lite"))
    out_dir = ensure_dir(output)
    train_ds = ExternalGroupDataset(dataset, cfg, split="train", baseline=baseline_name)
    if val_dataset:
        val_ds = ExternalGroupDataset(val_dataset, cfg, split="val", baseline=baseline_name)
    else:
        try:
            val_ds = ExternalGroupDataset(dataset, cfg, split="val", baseline=baseline_name)
        except Exception:
            val_ds = train_ds
    tcfg = bcfg.get("training", {}) if isinstance(bcfg.get("training", {}), dict) else {}
    device = _device(cfg)
    model = build_model_from_cfg(train_ds.feature_dim, cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(tcfg.get("lr", 2.0e-4)), weight_decay=float(tcfg.get("weight_decay", 1.0e-4)))
    batch_size = int(tcfg.get("batch_size", 32))
    num_workers = int(tcfg.get("num_workers", 0))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=_collate, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=_collate, num_workers=num_workers)
    epochs = int(tcfg.get("epochs", 10))
    best_val = float("inf")
    best_epoch = 0
    history = []
    t0 = perf_counter()
    for ep in range(1, epochs + 1):
        tr = _epoch(model, train_loader, opt, device, cfg, train=True)
        with torch.no_grad():
            va = _epoch(model, val_loader, None, device, cfg, train=False)
        row = {"epoch": ep, "train": tr, "val": va}
        history.append(row)
        ckpt = {
            "baseline": baseline_name,
            "cfg": cfg,
            "input_dim": int(train_ds.feature_dim),
            "max_candidates": int(train_ds.max_candidates),
            "model_state": model.state_dict(),
            "epoch": int(ep),
            "val_loss": float(va.get("loss", 0.0)),
        }
        torch.save(ckpt, out_dir / "latest.pt")
        if va.get("loss", float("inf")) <= best_val:
            best_val = float(va.get("loss", float("inf")))
            best_epoch = ep
            torch.save(ckpt, out_dir / "best.pt")
        print({"event": "external_baseline_epoch", "baseline": baseline_name, "epoch": ep, "train_loss": tr.get("loss"), "val_loss": va.get("loss"), "target_acc": va.get("target_acc")}, flush=True)
    summary = {
        "baseline": baseline_name,
        "num_train_groups": len(train_ds),
        "num_val_groups": len(val_ds),
        "feature_dim": int(train_ds.feature_dim),
        "max_candidates": int(train_ds.max_candidates),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "seconds": float(perf_counter() - t0),
        "history": history,
    }
    write_json(summary, out_dir / "train_summary.json")
    return summary
