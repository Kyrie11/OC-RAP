from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
from recap.models.care import CARE
from recap.training.dataset import RecoveryDataset
from recap.training.losses import care_supervised_loss


def _stack_batch(ds: RecoveryDataset, indices: np.ndarray) -> dict:
    items = [ds[int(i)] for i in indices]
    return {k: torch.stack([it[k] for it in items]) for k in items[0].keys()}


def train_care(dataset_path: str, output: str, epochs: int = 1, batch_size: int = 2, lr: float = 1e-3) -> str:
    print("[train_care] loading dataset", flush=True)
    ds = RecoveryDataset(dataset_path)
    print(f"[train_care] N={len(ds)}", flush=True)
    sample = ds[0]
    print("[train_care] building model", flush=True)
    model = CARE(C_bev=sample["bev"].shape[1], H_h=sample["bev"].shape[0], M=sample["mode_probs"].shape[0], H_p1=sample["actions_states"].shape[1], H_r1=sample["options_states_ref"].shape[2], hidden=32)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model.train()
    rng = np.random.default_rng(0)
    for epoch in range(epochs):
        order = np.arange(len(ds))
        rng.shuffle(order)
        print(f"[train_care] epoch {epoch}", flush=True)
        for step, start in enumerate(range(0, len(order), batch_size)):
            idx = order[start:start + batch_size]
            b = _stack_batch(ds, idx)
            print(f"[train_care] step {step} batch={len(idx)}", flush=True)
            out = model(b["bev"].float(), b["ego_info"].float(), b["route_command"].float(), b["actions_states"].float(), b["options_states_ref"].float(), b["action_mask"].bool(), b["option_mask"].bool())
            loss = care_supervised_loss(out, b)
            opt.zero_grad(); loss.backward(); opt.step()
            print(f"[train_care] loss={float(loss.detach()):.4f}", flush=True)
    p = Path(output); p.mkdir(parents=True, exist_ok=True)
    ckpt = p / "best.pt"
    print("[train_care] saving", flush=True)
    torch.save({"model": model.state_dict(), "metadata": ds.metadata}, ckpt)
    return str(ckpt)
