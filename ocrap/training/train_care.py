from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
from ocrap.models.care import CARE
from ocrap.training.dataset import RecoveryDataset
from ocrap.training.losses import care_supervised_loss


def _stack_batch(ds: RecoveryDataset, indices: np.ndarray) -> dict:
    items = [ds[int(i)] for i in indices]
    return {k: torch.stack([it[k] for it in items]) for k in items[0].keys()}


def train_care(dataset_path: str, output: str, epochs: int = 1, batch_size: int = 2, lr: float = 1e-3) -> str:
    print("[train_care] loading dataset", flush=True)
    ds = RecoveryDataset(dataset_path)
    print(f"[train_care] N={len(ds)}", flush=True)
    sample = ds[0]
    print("[train_care] building model", flush=True)
    token_key = "token_states_ref" if "token_states_ref" in sample else "options_states_ref"
    control_key = "token_controls_ref" if "token_controls_ref" in sample else "options_controls_ref"
    model = CARE(C_bev=sample["bev"].shape[1], H_h=sample["bev"].shape[0], M=sample["mode_probs"].shape[0], H_p1=sample["actions_states"].shape[1], H_r1=sample[token_key].shape[2], hidden=32)
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
            loss = care_supervised_loss(out, b)
            opt.zero_grad(); loss.backward(); opt.step()
            print(f"[train_care] loss={float(loss.detach()):.4f}", flush=True)
    p = Path(output); p.mkdir(parents=True, exist_ok=True)
    ckpt = p / "best.pt"
    print("[train_care] saving", flush=True)
    torch.save({"model": model.state_dict(), "metadata": ds.metadata}, ckpt)
    return str(ckpt)
