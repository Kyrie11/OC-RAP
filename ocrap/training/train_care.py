from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import numpy as np
import torch
import yaml

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

from ocrap.models.care import CARE
from ocrap.training.dataset import RecoveryDataset
from ocrap.training.losses import ocrap_loss


def _load_config(config: str | None) -> dict[str, Any]:
    if not config:
        return {}
    p = Path(config)
    if not p.exists():
        raise FileNotFoundError(f"config not found: {config}")
    if p.suffix.lower() in (".yaml", ".yml"):
        return yaml.safe_load(p.read_text()) or {}
    if p.suffix.lower() == ".json":
        return json.loads(p.read_text())
    return yaml.safe_load(p.read_text()) or {}


def _stack_batch(ds: RecoveryDataset, indices: np.ndarray) -> dict:
    items = [ds[int(i)] for i in indices]
    return {k: torch.stack([it[k] for it in items]) for k in items[0].keys()}


def _make_optimizer(model: torch.nn.Module, cfg: dict, lr: float) -> torch.optim.Optimizer:
    train_cfg = cfg.get("training", {}) if isinstance(cfg.get("training", {}), dict) else {}
    opt_name = str(train_cfg.get("optimizer", cfg.get("optimizer", "adamw"))).lower()
    weight_decay = float(train_cfg.get("weight_decay", cfg.get("weight_decay", 1e-4)))
    if opt_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=float(train_cfg.get("momentum", 0.9)), weight_decay=weight_decay)
    if opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def train_care(dataset_path: str, output: str, epochs: int | None = None, batch_size: int | None = None, lr: float | None = None, config: str | None = None, proposal_checkpoint: str | None = None) -> str:
    cfg = _load_config(config)
    train_cfg = cfg.get("training", {}) if isinstance(cfg.get("training", {}), dict) else {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
    planner_cfg = cfg.get("planner", {}) if isinstance(cfg.get("planner", {}), dict) else {}
    epochs = int(epochs if epochs is not None else train_cfg.get("epochs", cfg.get("epochs", 10)))
    batch_size = int(batch_size if batch_size is not None else train_cfg.get("batch_size", cfg.get("batch_size", 2)))
    lr = float(lr if lr is not None else train_cfg.get("lr", cfg.get("lr", 1e-3)))

    print("[train_care] loading dataset", flush=True)
    ds = RecoveryDataset(dataset_path)
    print(f"[train_care] N={len(ds)}", flush=True)
    if len(ds) == 0:
        raise ValueError(f"empty dataset: {dataset_path}")
    sample = ds[0]
    print("[train_care] building ReCoT model", flush=True)
    token_key = "token_states_ref" if "token_states_ref" in sample else "options_states_ref"
    D_token = int(sample.get("token_params", sample.get("options_params", torch.zeros(1, 6))).shape[-1]) if ("token_params" in sample or "options_params" in sample) else 6
    A_anchor = int(sample.get("token_anchor", torch.zeros(1, 3)).shape[-1]) if "token_anchor" in sample else 3
    D_shell = int(sample.get("token_hard_shell", torch.zeros(1, 4)).shape[-1]) if "token_hard_shell" in sample else 4
    model = CARE(
        C_bev=int(sample["bev"].shape[1]),
        H_h=int(sample["bev"].shape[0]),
        M=int(sample["mode_probs"].shape[0]),
        H_p1=int(sample["actions_states"].shape[1]),
        H_r1=int(sample[token_key].shape[2]),
        hidden=int(model_cfg.get("hidden", 128)),
        g_dim=int(model_cfg.get("g_dim", 9)),
        D_token=D_token,
        A_anchor=A_anchor,
        D_shell=D_shell,
    )
    if proposal_checkpoint:
        print(f"[train_care] proposal checkpoint is not consumed by ReCoT training path: {proposal_checkpoint}", flush=True)
    opt = _make_optimizer(model, cfg, lr)
    model.train()
    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    ocmero_params = {
        "tau_R": float(planner_cfg.get("tau_R", 0.20)),
        "c_R": float(planner_cfg.get("c_R", 0.0)),
        "alpha_R": float(planner_cfg.get("alpha_R", 0.20)),
        "alpha_H": float(planner_cfg.get("alpha_H", 0.20)),
        "alpha_K": float(planner_cfg.get("alpha_K", 0.20)),
        "use_observation_consistency": bool(model_cfg.get("use_observation_consistency", True)),
        **cfg.get("loss", {}),
    }
    for epoch in range(epochs):
        order = np.arange(len(ds))
        rng.shuffle(order)
        print(f"[train_care] epoch {epoch + 1}/{epochs}", flush=True)
        for step, start in enumerate(range(0, len(order), batch_size)):
            idx = order[start:start + batch_size]
            b = _stack_batch(ds, idx)
            token_key = "token_states_ref" if "token_states_ref" in b else "options_states_ref"
            control_key = "token_controls_ref" if "token_controls_ref" in b else "options_controls_ref"
            out = model(
                b["bev"].float(),
                b["ego_info"].float(),
                b["route_command"].float(),
                actions_states=b["actions_states"].float(),
                actions_controls=b.get("actions_controls", torch.empty(0)).float() if "actions_controls" in b else None,
                token_states_ref=b[token_key].float(),
                token_controls_ref=b[control_key].float() if control_key in b else None,
                token_params=b.get("token_params", b.get("options_params", None)).float() if ("token_params" in b or "options_params" in b) else None,
                token_anchor=b.get("token_anchor", None).float() if "token_anchor" in b else None,
                token_hard_shell=b.get("token_hard_shell", None).float() if "token_hard_shell" in b else None,
                action_mask=b["action_mask"].bool(),
                option_mask=b["option_mask"].bool(),
            )
            loss = ocrap_loss(out, b, ocmero_params)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get("grad_clip", 5.0)))
            opt.step()
            if step % int(train_cfg.get("log_every", 1)) == 0:
                print(f"[train_care] step {step} batch={len(idx)} loss={float(loss.detach()):.4f}", flush=True)
    p = Path(output); p.mkdir(parents=True, exist_ok=True)
    ckpt = p / "best.pt"
    print("[train_care] saving", flush=True)
    torch.save({"model": model.state_dict(), "metadata": ds.metadata, "config": cfg}, ckpt)
    return str(ckpt)
