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
    if baseline in {"route_bc", "route_bc_lite", "waymax_bc", "waymax_bc_lite"}:
        return int(nominal[0] if nominal else 0)

    if baseline in {"gameformer", "gameformer_lite"}:
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
        for i, d in enumerate(samples):
            x[i] = sample_to_feature(d, self.cfg)
            mask[i] = True
            utility[i] = _scalar(d, "utility", 0.0)
            hard[i] = _scalar(d, "hard_violation", 0.0)
            harm[i] = _scalar(d, "harm_proxy", 0.0)
            r_orc[i] = _scalar(d, "r_orc_star", 0.0)
            r_dep[i] = _scalar(d, "r_dep_star", 0.0)
            feasible[i] = _scalar(d, "feasible", 1.0) > 0.5
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
        }
