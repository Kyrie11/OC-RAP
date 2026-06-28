from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ocrap.algorithms.ocmero import oc_mero, torch_oc_mero
from ocrap.models.data import OPTION_FEATURE_DIM, fix_sample_geometry, sample_to_feature
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


def _merge_runtime_cfg_for_inference(ckpt_cfg: dict[str, Any], runtime_cfg: dict | None) -> dict[str, Any]:
    """Merge only inference-safe runtime settings into checkpoint config.

    The checkpoint owns model geometry and feature geometry.  A default runtime
    config must not silently overwrite model.d_model, feature dimensions, or
    other fields that affect the module shapes.  This was the cause of the
    256-vs-128 load_state_dict mismatch seen after training with
    ``--set model.d_model=256`` and calibrating with default config.
    """
    cfg = dict(ckpt_cfg or {})
    if not runtime_cfg:
        return cfg
    allow_sections = {"selection", "calibration", "evaluation", "ablation", "ocmero", "baselines", "metrics", "artifact"}
    for k, v in runtime_cfg.items():
        if k in allow_sections:
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                tmp = dict(cfg[k])
                tmp.update(v)
                cfg[k] = tmp
            else:
                cfg[k] = v
    # Only device-related training overrides are inference-safe.
    rt_train = runtime_cfg.get("training", {}) if isinstance(runtime_cfg.get("training", {}), dict) else {}
    if rt_train:
        train_cfg = dict(cfg.get("training", {}) or {})
        for key in ("device", "require_cuda"):
            if key in rt_train:
                train_cfg[key] = rt_train[key]
        cfg["training"] = train_cfg
    return cfg


def _infer_d_model(ckpt: dict[str, Any], cfg: dict[str, Any]) -> int:
    if "d_model" in ckpt:
        return int(ckpt["d_model"])
    state = ckpt.get("model_state", {}) or {}
    for key in ("encoder.net.0.weight", "root_logits.weight", "margin_head.weight"):
        w = state.get(key)
        if w is not None:
            shape = tuple(w.shape)
            if key == "encoder.net.0.weight" and len(shape) == 2:
                return int(shape[0])
            if len(shape) == 2:
                return int(shape[1])
    return int((cfg.get("model", {}) or {}).get("d_model", 128))


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # older torch
        return torch.load(path, map_location="cpu")


def load_model_bundle(checkpoint: str | Path | None, runtime_cfg: dict | None = None) -> ModelBundle | None:
    if not checkpoint:
        return None
    path = Path(checkpoint)
    if not path.exists():
        return None
    ckpt = _load_checkpoint(path)
    if "model_state" not in ckpt:
        return None
    cfg = _merge_runtime_cfg_for_inference(dict(ckpt.get("cfg", {}) or {}), runtime_cfg)
    device = _device_from_cfg(cfg)
    d_model = _infer_d_model(ckpt, cfg)
    d_obs = int(ckpt.get("d_obs", (cfg.get("model", {}) or {}).get("d_obs", 64)))
    tau_obs = float(ckpt.get("tau_obs", (cfg.get("model", {}) or {}).get("tau_obs", (cfg.get("ocmero", {}) or {}).get("tau_obs", 1.0))))
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
    encoder_type = str(ckpt.get("encoder_type", model_cfg.get("encoder_type", "mlp")))
    feature_layout = ckpt.get("feature_layout", None)
    model = OCRAPModel(
        int(ckpt["input_dim"]),
        num_roots=int(ckpt["num_roots"]),
        num_options=int(ckpt["num_options"]),
        d_model=d_model,
        d_obs=d_obs,
        tau_obs=tau_obs,
        encoder_type=encoder_type,
        feature_layout=feature_layout,
        num_layers=int(model_cfg.get("transformer_layers", 2)),
        num_heads=int(model_cfg.get("transformer_heads", 4)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        d_signature=int(ckpt.get("d_signature", 0)),
        d_future_signature=int(ckpt.get("d_future_signature", 0)),
        option_feature_dim=int(ckpt.get("option_feature_dim", OPTION_FEATURE_DIM)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    cfg.setdefault("model", {})
    cfg["model"]["d_model"] = d_model
    cfg["model"]["d_obs"] = d_obs
    cfg["model"]["tau_obs"] = tau_obs
    cfg["model"]["encoder_type"] = encoder_type
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
    fixed = fix_sample_geometry(
        d,
        num_roots=bundle.model.num_roots,
        num_options=bundle.model.num_options,
        d_signature=int(getattr(bundle.model, "d_signature", 0)),
        d_future_signature=int(getattr(bundle.model, "d_future_signature", 0)),
    )
    option_features = torch.from_numpy(fixed["option_features"]).float().unsqueeze(0).to(bundle.device)
    out = bundle.model(x, option_features)
    root_valid = torch.from_numpy(fixed["root_valid"]).bool().unsqueeze(0).to(bundle.device)
    p = torch.softmax(out["root_logits"].masked_fill(~root_valid, -1.0e4), dim=-1)
    option_valid = torch.from_numpy(fixed["option_valid"]).bool().unsqueeze(0).to(bundle.device)
    r_dep, r_orc, gap, q = torch_oc_mero(
        out["margins"],
        p,
        out["c_star"],
        alpha=float((bundle.cfg.get("ocmero", {}) or {}).get("alpha", 0.2)),
        beta=float((bundle.cfg.get("ocmero", {}) or {}).get("beta", 0.2)),
        option_valid=option_valid,
        root_valid=root_valid,
        use_lcvar=not bool((bundle.cfg.get("ablation", {}) or {}).get("without_lower_tail", False)),
        use_obs_kernel=not bool((bundle.cfg.get("ablation", {}) or {}).get("without_observation_kernel", False)),
        top_m=int((bundle.cfg.get("ocmero", {}) or {}).get("top_m", 8)),
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
