from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ocrap.algorithms.ocmero import oc_mero, torch_oc_mero
from ocrap.models.data import sample_to_feature
from ocrap.models.ocrap import OCRAPModel


@dataclass
class Prediction:
    r_dep: float
    r_orc: float
    gap: float
    q: np.ndarray
    root_probs: np.ndarray
    c_star: np.ndarray
    margins: np.ndarray


@dataclass
class ModelBundle:
    model: OCRAPModel
    cfg: dict[str, Any]
    device: torch.device


def _device_from_cfg(cfg: dict) -> torch.device:
    requested = str((cfg.get("training", {}) or {}).get("device", "auto"))
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def load_model_bundle(checkpoint: str | Path | None, runtime_cfg: dict | None = None) -> ModelBundle | None:
    if not checkpoint:
        return None
    path = Path(checkpoint)
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location="cpu")
    if "model_state" not in ckpt:
        return None
    cfg = dict(ckpt.get("cfg", {}) or {})
    if runtime_cfg:
        # Runtime cfg may override selection/calibration options, but checkpoint
        # geometry must remain authoritative.
        for k, v in runtime_cfg.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                tmp = dict(cfg[k])
                tmp.update(v)
                cfg[k] = tmp
            else:
                cfg[k] = v
    device = _device_from_cfg(cfg)
    model = OCRAPModel(
        int(ckpt["input_dim"]),
        num_roots=int(ckpt["num_roots"]),
        num_options=int(ckpt["num_options"]),
        d_model=int((cfg.get("model", {}) or {}).get("d_model", 128)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return ModelBundle(model=model, cfg=cfg, device=device)


def teacher_prediction_from_sample(d: dict[str, Any], cfg: dict | None = None) -> Prediction:
    cfg = cfg or {}
    res = oc_mero(
        np.asarray(d["m_star"], dtype=np.float64),
        np.asarray(d["root_probs"], dtype=np.float64),
        np.asarray(d["c_star"], dtype=np.float64),
        alpha=float((cfg.get("ocmero", {}) or {}).get("alpha", 0.2)),
        beta=float((cfg.get("ocmero", {}) or {}).get("beta", 0.2)),
        option_valid=np.asarray(d["option_valid"], dtype=bool),
        root_valid=np.asarray(d["root_valid"], dtype=bool) if "root_valid" in d else None,
        use_lcvar=not bool((cfg.get("ablation", {}) or {}).get("without_lower_tail", False)),
        use_obs_kernel=not bool((cfg.get("ablation", {}) or {}).get("without_observation_kernel", False)),
        top_m=int((cfg.get("ocmero", {}) or {}).get("top_m", 8)),
    )
    return Prediction(
        r_dep=float(res.r_dep),
        r_orc=float(res.r_orc),
        gap=float(res.gap),
        q=np.asarray(res.q, dtype=np.float32),
        root_probs=np.asarray(d["root_probs"], dtype=np.float32),
        c_star=np.asarray(d["c_star"], dtype=np.float32),
        margins=np.asarray(d["m_star"], dtype=np.float32),
    )


@torch.no_grad()
def predict_sample(d: dict[str, Any], bundle: ModelBundle | None, cfg: dict | None = None) -> Prediction:
    if bundle is None:
        return teacher_prediction_from_sample(d, cfg)
    x = torch.from_numpy(sample_to_feature(d, bundle.cfg)).float().unsqueeze(0).to(bundle.device)
    out = bundle.model(x)
    p = torch.softmax(out["root_logits"], dim=-1)
    option_valid = torch.from_numpy(np.asarray(d["option_valid"], dtype=np.float32) > 0.5).unsqueeze(0).to(bundle.device)
    r_dep, r_orc, gap, q = torch_oc_mero(
        out["margins"],
        p,
        out["c_star"],
        alpha=float((bundle.cfg.get("ocmero", {}) or {}).get("alpha", 0.2)),
        beta=float((bundle.cfg.get("ocmero", {}) or {}).get("beta", 0.2)),
        option_valid=option_valid,
        use_lcvar=not bool((bundle.cfg.get("ablation", {}) or {}).get("without_lower_tail", False)),
        use_obs_kernel=not bool((bundle.cfg.get("ablation", {}) or {}).get("without_observation_kernel", False)),
    )
    return Prediction(
        r_dep=float(r_dep.squeeze(0).detach().cpu().item()),
        r_orc=float(r_orc.squeeze(0).detach().cpu().item()),
        gap=float(gap.squeeze(0).detach().cpu().item()),
        q=q.squeeze(0).detach().cpu().numpy().astype(np.float32),
        root_probs=p.squeeze(0).detach().cpu().numpy().astype(np.float32),
        c_star=out["c_star"].squeeze(0).detach().cpu().numpy().astype(np.float32),
        margins=out["margins"].squeeze(0).detach().cpu().numpy().astype(np.float32),
    )
