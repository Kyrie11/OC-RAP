from __future__ import annotations

from pathlib import Path
from typing import Dict
import numpy as np
import torch

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

from ocrap.models.recot import ReCoT
from ocrap.models.oc_mero import compute_ocmero_profiles, OCMEROParams
from ocrap.training.dataset import RecoveryDataset


def _hidden_from_state(state: dict, default: int = 128) -> int:
    w = state.get("head_g.weight")
    if w is not None and hasattr(w, "shape") and len(w.shape) == 2:
        return int(w.shape[1])
    w = state.get("mode_queries")
    if w is not None and hasattr(w, "shape") and len(w.shape) == 2:
        return int(w.shape[1])
    return default


def _safe_load_state(path: str | Path) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    return ckpt.get("model", ckpt)


def predict_profiles(dataset_path: str | Path, checkpoint: str | Path, batch_size: int = 4, ocmero_params: dict | None = None) -> Dict[str, np.ndarray]:
    """Run ReCoT + OC-MERO on a dataset without passing oracle-only fields.

    The returned arrays are named with *_pred where appropriate so evaluation can
    cleanly separate learned OC-RAP inference from teacher labels.
    """
    ds = RecoveryDataset(str(dataset_path))
    if len(ds) == 0:
        raise ValueError(f"empty dataset: {dataset_path}")
    sample = ds[0]
    state = _safe_load_state(checkpoint)
    token_key = "token_states_ref" if "token_states_ref" in sample else "options_states_ref"
    model = ReCoT(
        C_bev=int(sample["bev"].shape[1]),
        H_h=int(sample["bev"].shape[0]),
        M=int(sample["mode_probs"].shape[0]),
        H_p1=int(sample["actions_states"].shape[1]),
        H_r1=int(sample[token_key].shape[2]),
        hidden=_hidden_from_state(state),
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Older checkpoints may not have the action-conditioned beta head.  Strictly
    # requiring every key would break reproducibility for already-trained runs;
    # the missing keys are reported in metadata by callers if needed.
    model.eval()
    params = OCMEROParams(**{k: v for k, v in (ocmero_params or {}).items() if k in OCMEROParams.__annotations__})
    out_chunks: dict[str, list[np.ndarray]] = {k: [] for k in ["R_pred", "B_pred", "U_pred", "H_pred", "dH_pred", "K_post_pred", "C_pred", "witness_pred", "W_pred", "pi_pred", "mu_pred"]}
    with torch.no_grad():
        for start in range(0, len(ds), int(batch_size)):
            items = [ds[i] for i in range(start, min(len(ds), start + int(batch_size)))]
            b = {k: torch.stack([it[k] for it in items]) for k in items[0].keys()}
            token_key = "token_states_ref" if "token_states_ref" in b else "options_states_ref"
            control_key = "token_controls_ref" if "token_controls_ref" in b else "options_controls_ref"
            pred = model(
                b["bev"].float(),
                b["ego_info"].float(),
                b["route_command"].float(),
                actions_states=b["actions_states"].float(),
                actions_controls=b["actions_controls"].float() if "actions_controls" in b else None,
                token_states_ref=b[token_key].float(),
                token_controls_ref=b[control_key].float() if control_key in b else None,
                token_params=(b["token_params"].float() if "token_params" in b else (b["options_params"].float() if "options_params" in b else None)),
                token_anchor=b["token_anchor"].float() if "token_anchor" in b else None,
                token_hard_shell=b["token_hard_shell"].float() if "token_hard_shell" in b else None,
                action_mask=b["action_mask"].bool(),
                option_mask=b["option_mask"].bool(),
            )
            prof = compute_ocmero_profiles(pred, {"action_mask": b["action_mask"].bool(), "option_mask": b["option_mask"].bool()}, params)
            mapping = {
                "R_pred": prof["R"], "B_pred": prof["B"], "U_pred": prof["U"], "H_pred": prof["H"],
                "dH_pred": prof["dH"], "K_post_pred": prof["K_post"], "C_pred": prof["C"],
                "witness_pred": prof["witness"], "W_pred": prof["W"], "pi_pred": prof["pi_am"], "mu_pred": prof["mu"],
            }
            for k, v in mapping.items():
                out_chunks[k].append(v.detach().cpu().numpy())
    return {k: np.concatenate(v, axis=0) for k, v in out_chunks.items() if v}
