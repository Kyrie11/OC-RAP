from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from ocrap.algorithms.ocmero import torch_oc_mero
from ocrap.data.serialization import ensure_dir, write_json
from ocrap.models.data import OCRAPSampleDataset, OPTION_FEATURE_DIM, iter_sample_paths_many, split_paths_by_npz_split
from ocrap.models.losses import anti_oracle_loss
from ocrap.models.ocrap import OCRAPModel
from ocrap.utils.seed import seed_everything


def _device(cfg: dict) -> torch.device:
    tcfg = cfg.get("training", {}) if isinstance(cfg.get("training", {}), dict) else {}
    requested = str(tcfg.get("device", "auto"))
    if requested == "auto":
        if bool(tcfg.get("require_cuda", False)) and not torch.cuda.is_available():
            raise RuntimeError("training.require_cuda=true, but torch.cuda.is_available() is false")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if bool(tcfg.get("require_cuda", False)) and device.type != "cuda":
        raise RuntimeError("training.require_cuda=true, but selected device is not CUDA")
    if bool(tcfg.get("require_cuda", False)) and not torch.cuda.is_available():
        raise RuntimeError("training.require_cuda=true, but torch.cuda.is_available() is false")
    return device


def _device_summary(device: torch.device) -> dict[str, object]:
    summary: dict[str, object] = {
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_version": str(torch.__version__),
    }
    if torch.cuda.is_available():
        idx = device.index if device.type == "cuda" and device.index is not None else torch.cuda.current_device()
        try:
            summary.update({
                "cuda_device_index": int(idx),
                "cuda_device_name": torch.cuda.get_device_name(idx),
                "cuda_device_count": int(torch.cuda.device_count()),
            })
        except Exception:
            pass
    return summary


def _collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in batch[0].keys()}


def _masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
    mask = mask.bool() & torch.isfinite(target) & (target > -1e8)
    if not bool(mask.any()):
        return pred.sum() * 0.0
    return F.smooth_l1_loss(pred[mask], target[mask])


def _root_signature_loss(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], key: str, valid_key: str = "root_valid") -> torch.Tensor:
    if key not in out or key not in batch or batch[key].shape[-1] == 0:
        ref = next(iter(out.values()))
        return ref.sum() * 0.0
    pred = out[key]
    target = batch[key].float()
    mask = batch[valid_key].bool().unsqueeze(-1).expand_as(target)
    return _masked_smooth_l1(pred, target, mask)


def _obs_bce(pred_c: torch.Tensor, target_y: torch.Tensor, root_valid: torch.Tensor) -> torch.Tensor:
    B, K, _ = pred_c.shape
    eye = torch.eye(K, dtype=torch.bool, device=pred_c.device).unsqueeze(0)
    pair_mask = root_valid.unsqueeze(1) & root_valid.unsqueeze(2) & (~eye)
    if not bool(pair_mask.any()):
        return pred_c.sum() * 0.0
    pred = pred_c.clamp(1e-5, 1.0 - 1e-5)
    return F.binary_cross_entropy(pred[pair_mask], target_y[pair_mask].float())


def _progress_iter(loader: DataLoader, *, enabled: bool, desc: str):
    if not enabled:
        return loader
    try:
        from tqdm.auto import tqdm  # type: ignore
        return tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    except Exception:
        return loader


def _epoch(
    model: OCRAPModel,
    loader: DataLoader,
    cfg: dict,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    stage: str = "train",
    epoch: int | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    lw = cfg.get("loss_weights", {}) if isinstance(cfg.get("loss_weights", {}), dict) else {}
    ocfg = cfg.get("ocmero", {}) if isinstance(cfg.get("ocmero", {}), dict) else {}
    art_cfg = cfg.get("artifact", {}) if isinstance(cfg.get("artifact", {}), dict) else {}
    tcfg = cfg.get("training", {}) if isinstance(cfg.get("training", {}), dict) else {}
    progress = bool(tcfg.get("progress", cfg.get("progress", True)))
    totals: dict[str, float] = {}
    n = 0
    desc = f"{stage} ep{epoch}" if epoch is not None else stage
    for batch in _progress_iter(loader, enabled=progress, desc=desc):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        out = model(batch["x"].float(), batch.get("option_features"))
        root_valid = batch["root_valid"].bool()
        masked_logits = out["root_logits"].masked_fill(~root_valid, -1.0e4)
        root_p = torch.softmax(masked_logits, dim=-1)
        root_target = batch["root_probs"].float() * root_valid.float()
        root_target = root_target / root_target.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        margin_target = torch.clamp(batch["m_star"].float(), min=-float(cfg.get("margin_clip", 5.0)), max=float(cfg.get("margin_clip", 5.0)))
        margin_mask = batch["root_valid"].unsqueeze(-1) & batch["option_valid"].unsqueeze(1)

        loss_root = -(root_target * F.log_softmax(masked_logits, dim=-1)).sum(dim=-1).mean()
        loss_margin = _masked_smooth_l1(out["margins"], margin_target, margin_mask)
        loss_sig = _root_signature_loss(out, batch, "root_signature")
        loss_future_sig = _root_signature_loss(out, batch, "root_future_signature")
        loss_obs = _obs_bce(out["c_star"], batch["y_obs"], batch["root_valid"])
        r_dep, r_orc, gap, _q = torch_oc_mero(
            out["margins"],
            root_p,
            out["c_star"],
            alpha=float(ocfg.get("alpha", 0.2)),
            beta=float(ocfg.get("beta", 0.2)),
            option_valid=batch["option_valid"],
            root_valid=root_valid,
            use_lcvar=not bool((cfg.get("ablation", {}) or {}).get("without_lower_tail", False)),
            use_obs_kernel=not bool((cfg.get("ablation", {}) or {}).get("without_observation_kernel", False)),
            top_m=int(ocfg.get("top_m", 8)),
        )
        loss_dep = F.smooth_l1_loss(r_dep, batch["r_dep_star"].float())
        loss_orc = F.smooth_l1_loss(r_orc, batch["r_orc_star"].float())
        loss_art = anti_oracle_loss(r_orc, r_dep, batch["i_art_star"].float(), delta_neg=float(art_cfg.get("delta_neg", 0.0)))
        loss_util = F.smooth_l1_loss(out["utility"], batch["utility"].float())
        total = (
            float(lw.get("assign", 1.0)) * loss_root
            + float(lw.get("margin", 2.0)) * loss_margin
            + float(lw.get("sig", 0.5)) * (loss_sig + loss_future_sig)
            + float(lw.get("obs", 1.0)) * loss_obs
            + 0.5 * (loss_dep + loss_orc)
            + float(lw.get("anti_oracle", 1.0)) * loss_art
            + float(lw.get("utility", 0.2)) * loss_util
        )
        if training:
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            grad_clip = float(tcfg.get("grad_clip", 5.0))
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        bsz = int(batch["x"].shape[0])
        n += bsz
        vals = {
            "loss": total.item(),
            "loss_root": loss_root.item(),
            "loss_margin": loss_margin.item(),
            "loss_sig": loss_sig.item(),
            "loss_future_sig": loss_future_sig.item(),
            "loss_obs": loss_obs.item(),
            "loss_dep": loss_dep.item(),
            "loss_orc": loss_orc.item(),
            "loss_art": loss_art.item(),
            "loss_utility": loss_util.item(),
            "pred_r_dep_mean": r_dep.mean().item(),
            "teacher_r_dep_mean": batch["r_dep_star"].float().mean().item(),
        }
        for k, v in vals.items():
            totals[k] = totals.get(k, 0.0) + float(v) * bsz
    return {k: float(v / max(n, 1)) for k, v in totals.items()} | {"num_samples": int(n), "num_batches": int(len(loader))}


def _make_sampler(ds: OCRAPSampleDataset, cfg: dict) -> WeightedRandomSampler | None:
    weight_art = float((cfg.get("training", {}) or {}).get("artifact_sampler_weight", 0.25))
    if weight_art <= 0:
        return None
    weights = []
    for p in ds.paths:
        try:
            from ocrap.data.serialization import load_npz

            d = load_npz(p)
            is_art = float(np.asarray(d.get("i_art_star", 0)).item()) > 0.5
            weights.append(1.0 + (weight_art if is_art else 0.0))
        except Exception:
            weights.append(1.0)
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=len(weights), replacement=True)


def train(dataset: str, output: str, cfg: dict) -> dict:
    seed_everything(int(cfg.get("seed", 7)))
    out = ensure_dir(output)
    paths = iter_sample_paths_many(dataset)
    if not paths:
        raise ValueError(f"No OC-RAP sample .npz files found under {dataset}")
    train_paths = split_paths_by_npz_split(paths, "train")
    val_paths = split_paths_by_npz_split(paths, "val")
    if not train_paths:
        train_paths = paths
    if not val_paths:
        val_paths = train_paths[: max(1, min(len(train_paths), len(train_paths) // 10 or 1))]

    train_ds = OCRAPSampleDataset(train_paths, cfg)
    val_ds = OCRAPSampleDataset(val_paths, cfg)
    first = train_ds[0]
    num_roots = int(train_ds.num_roots)
    num_options = int(train_ds.num_options)
    d_signature = int(train_ds.d_signature)
    d_future_signature = int(train_ds.d_future_signature)
    device = _device(cfg)
    device_info = _device_summary(device)
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
    d_model = int(model_cfg.get("d_model", 128))
    d_obs = int(model_cfg.get("d_obs", 64))
    tau_obs = float(model_cfg.get("tau_obs", (cfg.get("ocmero", {}) or {}).get("tau_obs", 1.0)))
    encoder_type = str(model_cfg.get("encoder_type", "mlp"))
    feature_layout = {
        "prefix_param_dim": int(cfg.get("prefix_param_dim", 5)),
        "num_macros": int(model_cfg.get("num_macros", 16)),
        "prefix_flat_dim": int(model_cfg.get("feature_prefix_flat_dim", 80)),
        "control_flat_dim": int(model_cfg.get("feature_control_flat_dim", 40)),
        "feature_max_agents": int(model_cfg.get("feature_max_agents", 32)),
        "bev_channels": int(cfg.get("bev_channels", 7)),
        "route_flat_dim": int(model_cfg.get("feature_route_flat_dim", 64)),
        "map_flat_dim": int(model_cfg.get("feature_map_flat_dim", 64)),
        "dyn_flat_dim": int(model_cfg.get("feature_dynamic_map_flat_dim", 32)),
    }
    model = OCRAPModel(
        train_ds.feature_dim,
        num_roots=num_roots,
        num_options=num_options,
        d_model=d_model,
        d_obs=d_obs,
        tau_obs=tau_obs,
        encoder_type=encoder_type,
        feature_layout=feature_layout,
        num_layers=int(model_cfg.get("transformer_layers", 2)),
        num_heads=int(model_cfg.get("transformer_heads", 4)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        d_signature=d_signature,
        d_future_signature=d_future_signature,
        option_feature_dim=OPTION_FEATURE_DIM,
    ).to(device)
    tcfg = cfg.get("training", {}) if isinstance(cfg.get("training", {}), dict) else {}
    opt = torch.optim.AdamW(model.parameters(), lr=float(tcfg.get("lr", 1e-3)), weight_decay=float(tcfg.get("weight_decay", 1e-4)))
    batch_size = int(tcfg.get("batch_size", 32))
    num_workers = int(tcfg.get("num_workers", 0))
    sampler = _make_sampler(train_ds, cfg)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=sampler is None, sampler=sampler, num_workers=num_workers, collate_fn=_collate, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=_collate, pin_memory=(device.type == "cuda"))
    epochs = int(tcfg.get("epochs", 10))
    best_val = float("inf")
    history = []
    best_path = out / "best.pt"
    latest_path = out / "latest.pt"
    ckpt_dir = ensure_dir(out / "checkpoints")

    def _checkpoint_payload(ep: int, val_loss: float) -> dict:
        return {
            "cfg": cfg,
            "input_dim": train_ds.feature_dim,
            "num_roots": num_roots,
            "num_options": num_options,
            "d_model": d_model,
            "d_obs": d_obs,
            "tau_obs": tau_obs,
            "encoder_type": encoder_type,
            "feature_layout": feature_layout,
            "d_signature": d_signature,
            "d_future_signature": d_future_signature,
            "option_feature_dim": OPTION_FEATURE_DIM,
            "model_state": model.state_dict(),
            "optimizer_state": opt.state_dict(),
            "epoch": int(ep),
            "val_loss": float(val_loss),
            "device_info_at_train": device_info,
            "note": "OC-RAP neural checkpoint: predicts root probabilities, recovery margins, utility, and observation compatibility from scene-prefix features.",
        }
    print({
        "event": "train_start",
        **device_info,
        "num_train_samples": len(train_paths),
        "num_val_samples": len(val_paths),
        "train_batches_per_epoch": len(train_loader),
        "val_batches_per_epoch": len(val_loader),
        "epochs": epochs,
        "batch_size": batch_size,
        "d_model": d_model,
        "d_obs": d_obs,
        "encoder_type": encoder_type,
    }, flush=True)
    t0 = perf_counter()
    for ep in range(1, epochs + 1):
        ep_t0 = perf_counter()
        tr = _epoch(model, train_loader, cfg, device, opt, stage="train", epoch=ep)
        with torch.no_grad():
            va = _epoch(model, val_loader, cfg, device, None, stage="val", epoch=ep)
        row = {"epoch": ep, "train": tr, "val": va, "seconds": float(perf_counter() - ep_t0)}
        history.append(row)
        improved = va.get("loss", float("inf")) <= best_val
        payload = _checkpoint_payload(ep, va.get("loss", float("inf")))
        save_every = bool(tcfg.get("save_every_epoch", True))
        if save_every:
            torch.save(payload, ckpt_dir / f"epoch_{ep:04d}.pt")
        if bool(tcfg.get("save_latest", True)):
            torch.save(payload, latest_path)
        if improved:
            best_val = va["loss"]
            payload["val_loss"] = float(best_val)
            payload["is_best"] = True
            torch.save(payload, best_path)
        print({
            "event": "epoch_end",
            "epoch": ep,
            "train_loss": round(float(tr.get("loss", 0.0)), 6),
            "val_loss": round(float(va.get("loss", 0.0)), 6),
            "best_val_loss": round(float(best_val), 6),
            "improved": bool(improved),
            "seconds": round(float(row["seconds"]), 2),
        }, flush=True)
    result = {
        "checkpoint": str(best_path),
        "latest_checkpoint": str(latest_path),
        "checkpoint_dir": str(ckpt_dir),
        "num_train_samples": len(train_paths),
        "num_val_samples": len(val_paths),
        "input_dim": train_ds.feature_dim,
        "num_roots": num_roots,
        "num_options": num_options,
        "d_model": d_model,
        "d_obs": d_obs,
        "encoder_type": encoder_type,
        "best_val_loss": float(best_val),
        "device_info": device_info,
        "train_batches_per_epoch": len(train_loader),
        "val_batches_per_epoch": len(val_loader),
        "total_train_steps": int(len(train_loader) * epochs),
        "elapsed_seconds": float(perf_counter() - t0),
        "history": history,
    }
    write_json(result, out / "train_summary.json")
    return result
