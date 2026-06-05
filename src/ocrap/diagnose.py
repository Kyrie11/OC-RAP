from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from .io import load_npz, write_json


REQUIRED_FIELDS = [
    "m_star",
    "y_obs",
    "c_star",
    "r_orc_star",
    "r_dep_star",
    "i_art_star",
    "root_probs",
    "regime_label",
    "split_id",
    "future_sources",
]


def diagnose_dataset(dataset_dir: str | Path, output_path: str | Path | None = None, max_samples: int | None = None) -> dict[str, Any]:
    dataset_dir = Path(dataset_dir)
    manifest = dataset_dir / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing manifest {manifest}")
    rows = list(csv.DictReader(manifest.open("r", encoding="utf-8")))
    if max_samples is not None:
        rows = rows[:max_samples]
    issues = []
    split_scene: dict[str, set[str]] = {}
    counts = {"num_samples": 0, "num_artifacts": 0, "future_source_coverage": {}, "split_counts": {}, "regime_counts": {}}
    for row in rows:
        path = dataset_dir / row["path"]
        data = load_npz(path)
        counts["num_samples"] += 1
        split = str(data.get("split_id", row.get("split_id", "unknown")))
        scene = str(data.get("scene_id", row.get("scene_id", "unknown")))
        split_scene.setdefault(split, set()).add(scene)
        counts["split_counts"][split] = counts["split_counts"].get(split, 0) + 1
        for f in REQUIRED_FIELDS:
            if f not in data:
                issues.append({"path": str(path), "issue": f"missing_field:{f}"})
        if "root_probs" in data:
            s = float(np.asarray(data["root_probs"]).sum())
            if not np.isfinite(s) or abs(s - 1.0) > 1e-3:
                issues.append({"path": str(path), "issue": f"root_probs_sum:{s}"})
        if "c_star" in data:
            C = np.asarray(data["c_star"], dtype=np.float64)
            if C.ndim != 2 or C.shape[0] != C.shape[1]:
                issues.append({"path": str(path), "issue": "c_star_not_square"})
            elif np.max(np.abs(np.diag(C) - 1.0)) > 1e-4:
                issues.append({"path": str(path), "issue": "c_star_diag_not_one"})
        if "y_obs" in data:
            Y = np.asarray(data["y_obs"], dtype=np.float64)
            if Y.ndim != 2 or np.max(np.abs(Y - Y.T)) > 1e-4:
                issues.append({"path": str(path), "issue": "y_obs_not_symmetric"})
        if bool(int(np.asarray(data.get("i_art_star", 0)).reshape(-1)[0])):
            counts["num_artifacts"] += 1
        for src in data.get("future_sources", []):
            counts["future_source_coverage"][str(src)] = counts["future_source_coverage"].get(str(src), 0) + 1
        regimes = data.get("regime_label", {})
        if isinstance(regimes, np.ndarray) and regimes.dtype == object:
            regimes = regimes.item()
        if isinstance(regimes, dict):
            for k, v in regimes.items():
                if v:
                    counts["regime_counts"][k] = counts["regime_counts"].get(k, 0) + 1
        # Required counterfactual source coverage per sample.
        srcs = set(map(str, data.get("future_sources", [])))
        for needed in ["replay", "reactive", "targeted"]:
            if needed not in srcs:
                issues.append({"path": str(path), "issue": f"missing_future_source:{needed}"})
        # Hidden emergence should be marked from unknown mask in metadata.
        metas = data.get("future_metadata", [])
        for m in metas:
            if isinstance(m, dict) and m.get("hidden_emergence", False) and not m.get("from_unknown_mask", False):
                issues.append({"path": str(path), "issue": "hidden_emergence_not_from_unknown_mask"})
    # Scenario-level leakage check.
    splits = list(split_scene.keys())
    for i, a in enumerate(splits):
        for b in splits[i + 1 :]:
            overlap = split_scene[a] & split_scene[b]
            if overlap:
                issues.append({"issue": "scenario_split_leakage", "splits": [a, b], "num_overlap": len(overlap), "examples": list(sorted(overlap))[:5]})
    counts["artifact_fraction"] = float(counts["num_artifacts"] / max(counts["num_samples"], 1))
    result = {"counts": counts, "issues": issues, "passed": len(issues) == 0}
    if output_path is not None:
        write_json(result, output_path)
    return result
