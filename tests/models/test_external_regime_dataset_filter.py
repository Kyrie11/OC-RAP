from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from ocrap.external_baselines.data import group_sample_paths


def _write_group(root: Path, scene: str, t: int, tags_by_candidate: list[str], nominal_index: int = 0) -> None:
    samples = root / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    manifest = root / "manifest.csv"
    rows = []
    if manifest.exists():
        with manifest.open(newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    for ci, tags in enumerate(tags_by_candidate):
        path = samples / f"{scene}_t{t:04d}_a{ci:02d}.npz"
        np.savez(
            path,
            scene_id=np.asarray(scene), time_index=np.asarray(t), candidate_index=np.asarray(ci),
            split_id=np.asarray("train"), is_nominal=np.asarray(int(ci == nominal_index)),
            regime_label=np.asarray("{}"),
        )
        rows.append({
            "path": str(path.relative_to(root)), "scene_id": scene, "time_index": str(t),
            "candidate_index": str(ci), "split_id": "train", "is_nominal": str(int(ci == nominal_index)),
            "regime_label": tags,
        })
    fields = ["path", "scene_id", "time_index", "candidate_index", "split_id", "is_nominal", "regime_label"]
    with manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def test_named_regime_datasets_preserve_all_groups_without_purity_filtering(tmp_path: Path) -> None:
    # Regime directories are exposure strata.  Cross-regime hard examples that
    # were intentionally included by dataset construction must remain visible to
    # external baselines for fair comparison with OC-RAP.
    cases = {
        "train_near_contact": [
            ("near", ["near_contact;occluded", "near_contact"]),
            ("non_near_nominal", ["occluded", "near_contact"]),
            ("post_candidate", ["near_contact", "near_contact;post_contact"]),
        ],
        "train_contact": [
            ("contact", ["post_contact", "post_contact"]),
            ("near_only", ["near_contact", "near_contact"]),
        ],
        "train_safe": [
            ("safe", ["normal", "normal;occluded"]),
            ("contact_candidate", ["normal", "normal;post_contact"]),
        ],
    }
    for dirname, groups in cases.items():
        root = tmp_path / dirname
        for t, (scene, tags) in enumerate(groups, start=1):
            _write_group(root, scene, t, tags)
        observed = group_sample_paths(root, split="train")
        assert len(observed) == len(groups)
        names = {g[0].name.split("_t", 1)[0] for g in observed}
        assert names == {scene for scene, _ in groups}
