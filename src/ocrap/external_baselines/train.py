from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from ocrap.data.serialization import ensure_dir, write_json
from ocrap.external_baselines.data import ExternalGroupDataset
from ocrap.external_baselines.models import build_model_from_cfg
from ocrap.utils.seed import seed_everything

try:  # tqdm is optional but strongly preferred on training machines.
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def _distributed_available() -> bool:
    return dist.is_available() and dist.is_initialized()


def _setup_distributed(cfg: dict[str, Any]) -> tuple[bool, int, int, int]:
    bcfg = cfg.get("external_baselines", {}) if isinstance(cfg.get("external_baselines", {}), dict) else {}
    tcfg = bcfg.get("training", {}) if isinstance(bcfg.get("training", {}), dict) else {}
    requested = str(tcfg.get("distributed", "auto")).lower()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = (requested in {"true", "1", "yes"}) or (requested == "auto" and world_size > 1)
    if not use_ddp:
        return False, 0, 0, 1
    if not torch.cuda.is_available():
        raise RuntimeError("DDP training was requested but CUDA is not available")
    if not dist.is_initialized():
        dist.init_process_group(backend=str(tcfg.get("dist_backend", "nccl")), init_method="env://")
    rank = int(dist.get_rank())
    local_rank = int(os.environ.get("LOCAL_RANK", rank % max(torch.cuda.device_count(), 1)))
    world_size = int(dist.get_world_size())
    torch.cuda.set_device(local_rank)
    return True, rank, local_rank, world_size


def _cleanup_distributed() -> None:
    if _distributed_available():
        dist.barrier()
        dist.destroy_process_group()


def _device(cfg: dict[str, Any], *, use_ddp: bool = False, local_rank: int = 0) -> torch.device:
    bcfg = cfg.get("external_baselines", {}) if isinstance(cfg.get("external_baselines", {}), dict) else {}
    tcfg = bcfg.get("training", {}) if isinstance(bcfg.get("training", {}), dict) else {}
    requested = str(tcfg.get("device", (cfg.get("training", {}) or {}).get("device", "auto")))
    if use_ddp:
        return torch.device(f"cuda:{int(local_rank)}")
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


def _reduce_totals(totals: dict[str, float], n: int, device: torch.device) -> tuple[dict[str, float], int]:
    if not _distributed_available():
        return totals, n
    keys = sorted(totals)
    vals = torch.tensor([totals[k] for k in keys] + [float(n)], dtype=torch.float64, device=device)
    dist.all_reduce(vals, op=dist.ReduceOp.SUM)
    return {k: float(vals[i].item()) for i, k in enumerate(keys)}, int(vals[-1].item())


def _batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def _forward_model(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return model(
        batch["x"].float(),
        batch["mask"].bool(),
        branch_margins=batch.get("branch_margins", None),
        root_features=batch.get("root_features", None),
        root_probs=batch.get("root_probs", None),
        root_valid=batch.get("root_valid", None),
        option_valid=batch.get("option_valid", None),
    )


def _loss_dict(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], cfg: dict[str, Any]) -> dict[str, torch.Tensor]:
    bcfg = cfg.get("external_baselines", {}) if isinstance(cfg.get("external_baselines", {}), dict) else {}
    lw = bcfg.get("loss_weights", {}) if isinstance(bcfg.get("loss_weights", {}), dict) else {}
    mask = batch["mask"].bool()
    losses: dict[str, torch.Tensor] = {}
    losses["loss_policy"] = F.cross_entropy(out["logits"], batch["target_index"].long())
    # Deep supervision for GameFormer level-k decoders.  The original method
    # supervises every reasoning level; this keeps the gradient signal active at
    # level-0 and intermediate response decoders.
    level_logits = out.get("level_logits")
    if isinstance(level_logits, list) and len(level_logits) > 1:
        lev = [F.cross_entropy(x, batch["target_index"].long()) for x in level_logits]
        losses["loss_levelk"] = torch.stack(lev).mean()
        # Response consistency: later levels should not collapse to a noisier
        # distribution than the previous level.  KL is computed with stop-grad on
        # level k-1, mirroring the hierarchical response interpretation.
        kls = []
        for prev, cur in zip(level_logits[:-1], level_logits[1:]):
            p_prev = F.softmax(prev.detach(), dim=-1)
            log_cur = F.log_softmax(cur, dim=-1)
            kls.append(F.kl_div(log_cur, p_prev, reduction="batchmean"))
        losses["loss_level_response"] = torch.stack(kls).mean() if kls else out["logits"].sum() * 0.0
    else:
        losses["loss_levelk"] = out["logits"].sum() * 0.0
        losses["loss_level_response"] = out["logits"].sum() * 0.0
    losses["loss_utility"] = _masked_mse(out["utility"], batch["utility"].float(), mask)
    losses["loss_hard"] = _masked_mse(out["hard"], batch["hard"].float(), mask)
    losses["loss_harm"] = _masked_mse(out["harm"], batch["harm"].float(), mask)
    losses["loss_oracle_rec"] = _masked_mse(out["r_orc"], batch["r_orc"].float(), mask)
    losses["loss_deploy_rec"] = _masked_mse(out["r_dep"], batch["r_dep"].float(), mask)
    losses["loss"] = (
        float(lw.get("policy", 1.0)) * losses["loss_policy"]
        + float(lw.get("levelk", 0.35)) * losses["loss_levelk"]
        + float(lw.get("level_response", 0.10)) * losses["loss_level_response"]
        + float(lw.get("utility", 0.10)) * losses["loss_utility"]
        + float(lw.get("hard", 0.50)) * losses["loss_hard"]
        + float(lw.get("harm", 0.25)) * losses["loss_harm"]
        + float(lw.get("oracle_rec", 0.50)) * losses["loss_oracle_rec"]
        + float(lw.get("deploy_rec", 0.25)) * losses["loss_deploy_rec"]
    )
    return losses


def _epoch(model, loader, opt, device, cfg: dict[str, Any], train: bool, *, rank: int = 0, epoch: int = 0) -> dict[str, float]:
    model.train(train)
    bcfg = cfg.get("external_baselines", {}) if isinstance(cfg.get("external_baselines", {}), dict) else {}
    tcfg = bcfg.get("training", {}) if isinstance(bcfg.get("training", {}), dict) else {}
    totals: dict[str, float] = {}
    n = 0
    iterator = loader
    show = bool(tcfg.get("tqdm", True)) and rank == 0 and tqdm is not None
    if show:
        iterator = tqdm(loader, desc=("train" if train else "val") + f" ep{epoch:03d}", leave=False, dynamic_ncols=True)
    for batch in iterator:
        batch = _batch_to_device(batch, device)
        with torch.set_grad_enabled(train):
            out = _forward_model(model, batch)
            losses = _loss_dict(out, batch, cfg)
            loss = losses["loss"]
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(tcfg.get("grad_clip", 5.0)))
                opt.step()
        bs = int(batch["x"].shape[0])
        n += bs
        for name, val in losses.items():
            totals[name] = totals.get(name, 0.0) + float(val.detach().cpu()) * bs
        with torch.no_grad():
            pred = torch.argmax(out["logits"], dim=-1)
            acc = (pred == batch["target_index"]).float().mean()
            totals["target_acc"] = totals.get("target_acc", 0.0) + float(acc.detach().cpu()) * bs
        if show:
            iterator.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}")
    totals, n = _reduce_totals(totals, n, device)
    return {k: v / max(n, 1) for k, v in totals.items()}


def _model_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    module = model.module if isinstance(model, DDP) else model
    return module.state_dict()


def train_external_baseline(dataset: str, output: str, cfg: dict[str, Any], *, val_dataset: str | None = None, baseline: str | None = None) -> dict[str, Any]:
    seed_everything(int(cfg.get("seed", 7)))
    use_ddp, rank, local_rank, world_size = _setup_distributed(cfg)
    try:
        bcfg = cfg.setdefault("external_baselines", {})
        if baseline:
            bcfg["baseline"] = baseline
        baseline_name = str(bcfg.get("baseline", "route_bc_lite"))
        out_dir = ensure_dir(output) if rank == 0 else Path(output)
        train_ds = ExternalGroupDataset(dataset, cfg, split="train", baseline=baseline_name)
        if val_dataset:
            val_ds = ExternalGroupDataset(val_dataset, cfg, split="val", baseline=baseline_name)
        else:
            try:
                val_ds = ExternalGroupDataset(dataset, cfg, split="val", baseline=baseline_name)
            except Exception:
                val_ds = train_ds
        tcfg = bcfg.get("training", {}) if isinstance(bcfg.get("training", {}), dict) else {}
        device = _device(cfg, use_ddp=use_ddp, local_rank=local_rank)
        model = build_model_from_cfg(train_ds.feature_dim, cfg).to(device)
        if use_ddp:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=bool(tcfg.get("find_unused_parameters", False)))
        opt = torch.optim.AdamW(model.parameters(), lr=float(tcfg.get("lr", 2.0e-4)), weight_decay=float(tcfg.get("weight_decay", 1.0e-4)))
        batch_size = int(tcfg.get("batch_size", 32))
        num_workers = int(tcfg.get("num_workers", 0))
        pin_memory = bool(tcfg.get("pin_memory", True)) and torch.cuda.is_available()
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False) if use_ddp else None
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False) if use_ddp else None
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=(train_sampler is None), sampler=train_sampler, collate_fn=_collate, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=(num_workers > 0))
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, sampler=val_sampler, collate_fn=_collate, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=(num_workers > 0))
        epochs = int(tcfg.get("epochs", 10))
        best_val = float("inf")
        best_epoch = 0
        history = []
        t0 = perf_counter()
        for ep in range(1, epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(ep)
            tr = _epoch(model, train_loader, opt, device, cfg, train=True, rank=rank, epoch=ep)
            with torch.no_grad():
                va = _epoch(model, val_loader, None, device, cfg, train=False, rank=rank, epoch=ep)
            row = {"epoch": ep, "train": tr, "val": va}
            if rank == 0:
                history.append(row)
                ckpt = {
                    "baseline": baseline_name,
                    "cfg": cfg,
                    "input_dim": int(train_ds.feature_dim),
                    "max_candidates": int(train_ds.max_candidates),
                    "num_roots": int(train_ds.num_roots),
                    "num_options": int(train_ds.num_options),
                    "root_feature_dim": int(train_ds.root_feature_dim),
                    "model_state": _model_state(model),
                    "epoch": int(ep),
                    "val_loss": float(va.get("loss", 0.0)),
                    "world_size": int(world_size),
                }
                torch.save(ckpt, out_dir / "latest.pt")
                if va.get("loss", float("inf")) <= best_val:
                    best_val = float(va.get("loss", float("inf")))
                    best_epoch = ep
                    torch.save(ckpt, out_dir / "best.pt")
                print({"event": "external_baseline_epoch", "baseline": baseline_name, "epoch": ep, "world_size": world_size, "train_loss": tr.get("loss"), "val_loss": va.get("loss"), "target_acc": va.get("target_acc")}, flush=True)
        summary = {
            "baseline": baseline_name,
            "num_train_groups": len(train_ds),
            "num_val_groups": len(val_ds),
            "feature_dim": int(train_ds.feature_dim),
            "max_candidates": int(train_ds.max_candidates),
            "num_roots": int(train_ds.num_roots),
            "num_options": int(train_ds.num_options),
            "root_feature_dim": int(train_ds.root_feature_dim),
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val),
            "seconds": float(perf_counter() - t0),
            "world_size": int(world_size),
            "history": history,
        }
        if rank == 0:
            write_json(summary, out_dir / "train_summary.json")
        return summary
    finally:
        _cleanup_distributed()
