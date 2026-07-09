from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from ocrap.data.serialization import load_npz
from ocrap.models.data import iter_sample_paths_many, sample_to_feature, scalar_metadata_for_path


def _scalar(d: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(np.asarray(d.get(key, default)).item())
    except Exception:
        return float(default)


def _int(d: dict[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(np.asarray(d.get(key, default)).item())
    except Exception:
        return int(default)


def _str(d: dict[str, Any], key: str, default: str = "") -> str:
    try:
        v = np.asarray(d.get(key, default)).item()
        if isinstance(v, bytes):
            return v.decode("utf-8", errors="ignore")
        return str(v)
    except Exception:
        return str(default)


def _cfg_int(cfg: dict[str, Any], path: tuple[str, ...], default: int) -> int:
    cur: Any = cfg
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return int(default)
        cur = cur[key]
    try:
        return int(cur)
    except Exception:
        return int(default)


def _pad_1d(x: Any, n: int, *, fill: float = 0.0, dtype=np.float32) -> np.ndarray:
    out = np.full((int(n),), fill, dtype=dtype)
    if n <= 0:
        return out
    v = np.asarray(x, dtype=dtype).reshape(-1)
    m = min(int(n), int(v.size))
    if m > 0:
        out[:m] = v[:m]
    return np.nan_to_num(out, nan=fill, posinf=fill, neginf=fill)


def _pad_bool_1d(x: Any, n: int, *, fill: bool = False) -> np.ndarray:
    out = np.full((int(n),), bool(fill), dtype=bool)
    if n <= 0:
        return out
    v = np.asarray(x).reshape(-1).astype(bool)
    m = min(int(n), int(v.size))
    if m > 0:
        out[:m] = v[:m]
    return out


def _pad_2d(x: Any, shape: tuple[int, int], *, fill: float = 0.0, dtype=np.float32) -> np.ndarray:
    r, c = int(shape[0]), int(shape[1])
    out = np.full((r, c), fill, dtype=dtype)
    if r <= 0 or c <= 0:
        return out
    v = np.asarray(x, dtype=dtype)
    if v.ndim == 1:
        v = v.reshape(1, -1)
    if v.ndim != 2:
        return out
    rr, cc = min(r, v.shape[0]), min(c, v.shape[1])
    if rr > 0 and cc > 0:
        out[:rr, :cc] = v[:rr, :cc]
    return np.nan_to_num(out, nan=fill, posinf=fill, neginf=fill)


def _branch_arrays(d: dict[str, Any], cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return fixed branch tensors used by GameFormer-level-k training.

    root_features concatenates root_signature/root_future_signature plus validity
    and probability.  The exact root semantics are produced by OC-RAP's dataset
    builder, so no external data format is introduced here.
    """
    K_default = int(np.asarray(d.get('m_star', np.zeros((1, 0)))).shape[0] or cfg.get('num_roots', 10))
    L_default = int(np.asarray(d.get('m_star', np.zeros((0, 1)))).shape[1] or cfg.get('num_recovery_options', 12))
    K = _cfg_int(cfg, ('external_baselines','model','num_roots'), _cfg_int(cfg, ('num_roots',), K_default))
    L = _cfg_int(cfg, ('external_baselines','model','num_options'), _cfg_int(cfg, ('num_recovery_options',), L_default))
    F = _cfg_int(cfg, ('external_baselines','model','root_feature_dim'), 18)
    m = _pad_2d(d.get('m_star', np.zeros((0, 0))), (K, L), fill=-1.0, dtype=np.float32)
    root_probs = _pad_1d(d.get('root_probs', np.ones((K,), dtype=np.float32) / max(K, 1)), K, fill=0.0, dtype=np.float32)
    root_valid = _pad_bool_1d(d.get('root_valid', root_probs > 0), K, fill=False)
    root_probs = np.where(root_valid, np.clip(root_probs, 0.0, None), 0.0).astype(np.float32)
    den = float(root_probs.sum())
    if den > 1e-8:
        root_probs /= den
    sig = np.asarray(d.get('root_signature', np.zeros((K, 0))), dtype=np.float32)
    fsig = np.asarray(d.get('root_future_signature', np.zeros((K, 0))), dtype=np.float32)
    if sig.ndim != 2:
        sig = np.zeros((K, 0), dtype=np.float32)
    if fsig.ndim != 2:
        fsig = np.zeros((K, 0), dtype=np.float32)
    raw_feat = np.concatenate([sig[:K], fsig[:K]], axis=-1) if sig.shape[0] >= K and fsig.shape[0] >= K else np.concatenate([_pad_2d(sig, (K, sig.shape[-1] if sig.ndim == 2 else 0)), _pad_2d(fsig, (K, fsig.shape[-1] if fsig.ndim == 2 else 0))], axis=-1)
    rv = root_valid.astype(np.float32)[:, None]
    rp = root_probs.astype(np.float32)[:, None]
    raw_feat = np.concatenate([raw_feat, rp, rv], axis=-1)
    root_features = _pad_2d(raw_feat, (K, F), fill=0.0, dtype=np.float32)
    option_valid = _pad_bool_1d(d.get('option_valid', np.ones((L,), dtype=bool)), L, fill=False)
    return m, root_features, root_probs.astype(np.float32), root_valid.astype(bool), option_valid.astype(bool)


def _split_matches_path(path: Path, split: str | set[str]) -> bool:
    splits = {split} if isinstance(split, str) else set(split)
    if "all" in {str(x).lower() for x in splits}:
        return True
    sid = str(scalar_metadata_for_path(path, "split_id", ""))
    return sid in splits


def dataset_label_for_path(path: Path) -> str:
    root = path.parent.parent if path.parent.name == "samples" else path.parent
    return root.name


def group_sample_paths(dataset: str | Path, split: str | set[str] = "train") -> list[list[Path]]:
    paths = iter_sample_paths_many(dataset)
    grouped: dict[tuple[str, str, int], list[Path]] = defaultdict(list)
    for p in paths:
        p = Path(p)
        if not _split_matches_path(p, split):
            continue
        d = load_npz(p)
        scene_id = _str(d, "scene_id", "")
        time_index = _int(d, "time_index", 0)
        grouped[(dataset_label_for_path(p), scene_id, time_index)].append(p)
    out: list[list[Path]] = []
    for _, group in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        group = sorted(group, key=lambda p: _int(load_npz(p), "candidate_index", 0))
        out.append(group)
    return out


def _target_index(samples: list[dict[str, Any]], baseline: str, cfg: dict[str, Any]) -> int:
    bcfg = cfg.get("external_baselines", {}) if isinstance(cfg.get("external_baselines", {}), dict) else {}
    baseline = str(baseline).lower()
    feasible = np.asarray([_scalar(d, "feasible", 1.0) > 0.5 for d in samples], dtype=bool)
    utility = np.asarray([_scalar(d, "utility", 0.0) for d in samples], dtype=np.float32)
    hard = np.asarray([_scalar(d, "hard_violation", 0.0) for d in samples], dtype=np.float32)
    harm = np.asarray([_scalar(d, "harm_proxy", 0.0) for d in samples], dtype=np.float32)
    r_orc = np.asarray([_scalar(d, "r_orc_star", 0.0) for d in samples], dtype=np.float32)
    r_dep = np.asarray([_scalar(d, "r_dep_star", 0.0) for d in samples], dtype=np.float32)

    nominal = [i for i, d in enumerate(samples) if _scalar(d, "is_nominal", 0.0) > 0.5]
    if baseline in {"route_bc", "route_bc_lite", "waymax_bc", "waymax_bc_lite", "route_bc_wayformer"}:
        return int(nominal[0] if nominal else 0)

    if baseline in {"gameformer", "gameformer_lite", "gameformer_levelk"}:
        # Interaction-aware planning target: high utility + branch-wise/oracle
        # recoverability, not observation-consistent deployability.
        score = utility - float(bcfg.get("hard_weight", 12.0)) * hard - float(bcfg.get("harm_weight", 2.0)) * harm + float(bcfg.get("oracle_weight", 1.0)) * r_orc
    elif baseline in {"risk_net", "risk_aware_net"}:
        score = utility - float(bcfg.get("hard_weight", 12.0)) * hard - float(bcfg.get("harm_weight", 2.0)) * harm + float(bcfg.get("deploy_weight", 0.5)) * r_dep
    else:
        score = utility
    score = np.where(feasible, score, -1.0e9)
    return int(np.argmax(score)) if score.size else 0


class ExternalGroupDataset(Dataset):
    """Dataset of candidate-prefix sets grouped by scene/time."""

    def __init__(self, dataset: str | Path, cfg: dict[str, Any], *, split: str = "train", baseline: str = "route_bc_lite"):
        self.cfg = cfg
        self.baseline = str(baseline)
        self.groups = group_sample_paths(dataset, split=split)
        if not self.groups:
            raise ValueError(f"No grouped samples found in {dataset!s} for split={split!r}")
        first = load_npz(self.groups[0][0])
        self.feature_dim = int(sample_to_feature(first, cfg).shape[0])
        bcfg = cfg.get("external_baselines", {}) if isinstance(cfg.get("external_baselines", {}), dict) else {}
        self.max_candidates = int(bcfg.get("max_candidates", max(len(g) for g in self.groups)))
        m0, rf0, rp0, rv0, ov0 = _branch_arrays(first, cfg)
        self.num_roots = int(m0.shape[0])
        self.num_options = int(m0.shape[1])
        self.root_feature_dim = int(rf0.shape[-1])

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        paths = self.groups[idx][: self.max_candidates]
        samples = [load_npz(p) for p in paths]
        n = len(samples)
        x = np.zeros((self.max_candidates, self.feature_dim), dtype=np.float32)
        mask = np.zeros((self.max_candidates,), dtype=bool)
        utility = np.zeros((self.max_candidates,), dtype=np.float32)
        hard = np.zeros((self.max_candidates,), dtype=np.float32)
        harm = np.zeros((self.max_candidates,), dtype=np.float32)
        r_orc = np.zeros((self.max_candidates,), dtype=np.float32)
        r_dep = np.zeros((self.max_candidates,), dtype=np.float32)
        feasible = np.zeros((self.max_candidates,), dtype=bool)
        branch_margins = np.zeros((self.max_candidates, self.num_roots, self.num_options), dtype=np.float32)
        root_features = np.zeros((self.max_candidates, self.num_roots, self.root_feature_dim), dtype=np.float32)
        root_probs = np.zeros((self.max_candidates, self.num_roots), dtype=np.float32)
        root_valid = np.zeros((self.max_candidates, self.num_roots), dtype=bool)
        option_valid = np.zeros((self.max_candidates, self.num_options), dtype=bool)
        for i, d in enumerate(samples):
            x[i] = sample_to_feature(d, self.cfg)
            mask[i] = True
            utility[i] = _scalar(d, "utility", 0.0)
            hard[i] = _scalar(d, "hard_violation", 0.0)
            harm[i] = _scalar(d, "harm_proxy", 0.0)
            r_orc[i] = _scalar(d, "r_orc_star", 0.0)
            r_dep[i] = _scalar(d, "r_dep_star", 0.0)
            feasible[i] = _scalar(d, "feasible", 1.0) > 0.5
            bm, rf, rp, rv, ov = _branch_arrays(d, self.cfg)
            branch_margins[i] = bm
            root_features[i] = rf
            root_probs[i] = rp
            root_valid[i] = rv
            option_valid[i] = ov
        target = min(_target_index(samples, self.baseline, self.cfg), max(n - 1, 0))
        return {
            "x": torch.from_numpy(x),
            "mask": torch.from_numpy(mask),
            "target_index": torch.tensor(target, dtype=torch.long),
            "utility": torch.from_numpy(utility),
            "hard": torch.from_numpy(hard),
            "harm": torch.from_numpy(harm),
            "r_orc": torch.from_numpy(r_orc),
            "r_dep": torch.from_numpy(r_dep),
            "feasible": torch.from_numpy(feasible),
            "branch_margins": torch.from_numpy(branch_margins),
            "root_features": torch.from_numpy(root_features),
            "root_probs": torch.from_numpy(root_probs),
            "root_valid": torch.from_numpy(root_valid),
            "option_valid": torch.from_numpy(option_valid),
        }
