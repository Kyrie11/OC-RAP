from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from ocrap.data.build.diagnose import iter_sample_paths
from ocrap.data.serialization import load_npz


def iter_sample_paths_many(dataset: str | Path, max_samples: int | None = None) -> list[Path]:
    """Return sample paths from one or more OC-RAP dataset directories.

    The CLI still exposes a single --dataset argument for backwards
    compatibility.  To train on multiple sharded builds, pass a comma-separated
    list, e.g. /path/w0,/path/w1,/path/w2,/path/w3.
    """
    if isinstance(dataset, Path):
        specs = [dataset]
    else:
        raw = str(dataset)
        sep_specs: list[str] = []
        for chunk in raw.split(','):
            chunk = chunk.strip()
            if chunk:
                sep_specs.append(chunk)
        # Also accept the platform path separator when it is unambiguous.
        if len(sep_specs) == 1 and os.pathsep in sep_specs[0] and not Path(sep_specs[0]).exists():
            sep_specs = [x.strip() for x in sep_specs[0].split(os.pathsep) if x.strip()]
        specs = [Path(x) for x in sep_specs]
    paths: list[Path] = []
    for spec in specs:
        paths.extend(iter_sample_paths(spec))
    paths = sorted(dict.fromkeys(paths))
    return paths[:max_samples] if max_samples else paths


def _arr(d: dict[str, Any], key: str, default_shape: tuple[int, ...] = (0,)) -> np.ndarray:
    if key not in d:
        return np.zeros(default_shape, dtype=np.float32)
    try:
        return np.asarray(d[key], dtype=np.float32)
    except Exception:
        return np.zeros(default_shape, dtype=np.float32)


def _scalar(d: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(np.asarray(d.get(key, default)).item())
    except Exception:
        return default


def _int_scalar(d: dict[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(round(float(np.asarray(d.get(key, default)).item())))
    except Exception:
        return default


def _pad_flat(x: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros(int(n), dtype=np.float32)
    if n <= 0:
        return out
    v = np.asarray(x, dtype=np.float32).reshape(-1)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    out[: min(n, v.size)] = v[:n]
    return out


def _finite_stats(x: np.ndarray) -> np.ndarray:
    v = np.asarray(x, dtype=np.float32).reshape(-1)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.zeros(6, dtype=np.float32)
    return np.array([v.mean(), v.std(), v.min(), np.quantile(v, 0.25), np.quantile(v, 0.75), v.max()], dtype=np.float32)


def _one_hot(index: int, width: int) -> np.ndarray:
    out = np.zeros(int(width), dtype=np.float32)
    if 0 <= int(index) < int(width):
        out[int(index)] = 1.0
    return out


def _agent_features(d: dict[str, Any], max_agents: int) -> np.ndarray:
    hist = _arr(d, "agent_history")
    valid = _arr(d, "agent_valid")
    ego = _arr(d, "ego_state", (9,)).reshape(-1)
    if hist.ndim != 3 or valid.ndim < 2 or hist.shape[1] == 0:
        return np.zeros(8 + max_agents * 10, dtype=np.float32)
    last = hist[-1]
    vmask = valid[-1].astype(bool) if valid.shape[0] else np.zeros((last.shape[0],), dtype=bool)
    ego_xy = ego[:2] if ego.size >= 2 else np.zeros(2, dtype=np.float32)
    rows = []
    dists = []
    speeds = []
    # Agent feature layout follows ocrap.data.schema.AGENT_FEATURES.
    for a in range(1, min(last.shape[0], len(vmask))):
        if not vmask[a]:
            continue
        s = last[a]
        if s.size < 16:
            continue
        dx = float(s[0] - ego_xy[0])
        dy = float(s[1] - ego_xy[1])
        dist = float(np.hypot(dx, dy))
        speed = float(np.hypot(s[3], s[4]))
        dists.append(dist)
        speeds.append(speed)
        rows.append((dist, np.array([
            dx / 80.0,
            dy / 80.0,
            float(s[3]) / 20.0,
            float(s[4]) / 20.0,
            speed / 20.0,
            float(s[8]) if s.size > 8 else 0.0,
            float(s[9]) if s.size > 9 else 1.0,
            float(s[10]) / 10.0 if s.size > 10 else 0.0,
            float(s[11]) / 5.0 if s.size > 11 else 0.0,
            float(s[13]) / 10.0 if s.size > 13 else 0.0,
        ], dtype=np.float32)))
    rows.sort(key=lambda x: x[0])
    packed = np.zeros((max_agents, 10), dtype=np.float32)
    for i, (_, feat) in enumerate(rows[:max_agents]):
        packed[i] = feat
    d_arr = np.asarray(dists, dtype=np.float32)
    s_arr = np.asarray(speeds, dtype=np.float32)
    summary = np.array([
        float(len(rows)) / max(float(max_agents), 1.0),
        float(d_arr.min() / 80.0) if d_arr.size else 0.0,
        float(d_arr.mean() / 80.0) if d_arr.size else 0.0,
        float(d_arr.std() / 80.0) if d_arr.size else 0.0,
        float((d_arr < 8.0).mean()) if d_arr.size else 0.0,
        float((d_arr < 20.0).mean()) if d_arr.size else 0.0,
        float(s_arr.mean() / 20.0) if s_arr.size else 0.0,
        float(s_arr.max() / 20.0) if s_arr.size else 0.0,
    ], dtype=np.float32)
    return np.concatenate([summary, packed.reshape(-1)], axis=0)


def sample_to_feature(d: dict[str, Any], cfg: dict | None = None) -> np.ndarray:
    cfg = cfg or {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
    max_agents = int(model_cfg.get("feature_max_agents", 32))
    num_macros = int(model_cfg.get("num_macros", 16))
    prefix_flat = int(model_cfg.get("feature_prefix_flat_dim", 80))
    control_flat = int(model_cfg.get("feature_control_flat_dim", 40))
    route_flat = int(model_cfg.get("feature_route_flat_dim", 64))
    map_flat = int(model_cfg.get("feature_map_flat_dim", 64))
    dyn_flat = int(model_cfg.get("feature_dynamic_map_flat_dim", 32))

    ego = _pad_flat(_arr(d, "ego_state", (9,)), 9)
    prefix_param = _pad_flat(_arr(d, "prefix_param", (5,)), int(cfg.get("prefix_param_dim", 5)))
    # ``prefix_macro_id`` is kept as the deterministic candidate index for
    # filenames/seeding.  The semantic macro class is stored separately so the
    # model does not lose every candidate whose index exceeds ``num_macros``.
    macro_type_id = _int_scalar(d, "prefix_macro_type_id", _int_scalar(d, "prefix_macro_id", 0))
    macro = _one_hot(macro_type_id, num_macros)
    prefix_states = _arr(d, "prefix_states")
    prefix_controls = _arr(d, "prefix_controls")
    bev = _arr(d, "bev_occ")
    route = _arr(d, "route")
    maps = _arr(d, "map_polylines")
    dyn = _arr(d, "dynamic_map")

    bev_stats = []
    if bev.ndim >= 3:
        ch = bev.reshape(bev.shape[0], -1)
        bev_stats.append(ch.mean(axis=1))
        bev_stats.append(ch.std(axis=1))
    else:
        bev_stats.append(np.zeros(int(cfg.get("bev_channels", 7)), dtype=np.float32))
        bev_stats.append(np.zeros(int(cfg.get("bev_channels", 7)), dtype=np.float32))
    bev_feat = _pad_flat(np.concatenate(bev_stats, axis=0), 2 * int(cfg.get("bev_channels", 7)))

    scalar_feat = np.array([
        _scalar(d, "utility") / 20.0,
        _scalar(d, "hard_violation") / 5.0,
        _scalar(d, "harm_proxy") / 5.0,
        _scalar(d, "feasible"),
        _scalar(d, "is_nominal"),
        _scalar(d, "time_index") / 1000.0,
    ], dtype=np.float32)

    parts = [
        ego,
        prefix_param,
        macro,
        scalar_feat,
        _pad_flat(prefix_states, prefix_flat),
        _pad_flat(prefix_controls, control_flat),
        _agent_features(d, max_agents),
        bev_feat,
        _finite_stats(route),
        _pad_flat(route[..., :2] if route.ndim >= 2 else route, route_flat),
        _finite_stats(maps),
        _pad_flat(maps[..., :2] if maps.ndim >= 3 else maps, map_flat),
        _finite_stats(dyn),
        _pad_flat(dyn, dyn_flat),
    ]
    x = np.concatenate(parts, axis=0).astype(np.float32)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


class OCRAPSampleDataset(Dataset):
    def __init__(self, paths: list[Path], cfg: dict | None = None):
        self.paths = list(paths)
        self.cfg = cfg or {}
        if not self.paths:
            raise ValueError("OCRAPSampleDataset requires at least one sample path")
        first = load_npz(self.paths[0])
        self.feature_dim = int(sample_to_feature(first, self.cfg).shape[0])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        d = load_npz(self.paths[idx])
        x = sample_to_feature(d, self.cfg)
        out = {
            "x": torch.from_numpy(x),
            "root_probs": torch.from_numpy(np.asarray(d["root_probs"], dtype=np.float32)),
            "m_star": torch.from_numpy(np.asarray(d["m_star"], dtype=np.float32)),
            "c_star": torch.from_numpy(np.asarray(d["c_star"], dtype=np.float32)),
            "y_obs": torch.from_numpy(np.asarray(d["y_obs"], dtype=np.float32)),
            "option_valid": torch.from_numpy(np.asarray(d["option_valid"], dtype=np.float32) > 0.5),
            "root_valid": torch.from_numpy(np.asarray(d["root_valid"], dtype=np.float32) > 0.5),
            "r_dep_star": torch.tensor(float(np.asarray(d["r_dep_star"]).item()), dtype=torch.float32),
            "r_orc_star": torch.tensor(float(np.asarray(d["r_orc_star"]).item()), dtype=torch.float32),
            "i_art_star": torch.tensor(float(np.asarray(d["i_art_star"]).item()), dtype=torch.float32),
            "utility": torch.tensor(float(np.asarray(d.get("utility", 0.0)).item()), dtype=torch.float32),
            "root_signature": torch.from_numpy(np.asarray(d.get("root_signature", np.zeros((np.asarray(d["m_star"]).shape[0], 0))), dtype=np.float32)),
            "root_future_signature": torch.from_numpy(np.asarray(d.get("root_future_signature", np.zeros((np.asarray(d["m_star"]).shape[0], 0))), dtype=np.float32)),
        }
        return out


def split_paths_by_npz_split(paths: list[Path], split: str | set[str]) -> list[Path]:
    splits = {split} if isinstance(split, str) else set(split)
    keep: list[Path] = []
    for p in paths:
        try:
            d = load_npz(p)
            sid = str(np.asarray(d.get("split_id", "")).item())
            if sid in splits or "all" in splits:
                keep.append(p)
        except Exception:
            continue
    return keep
