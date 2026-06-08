from __future__ import annotations

from ocrap.utils.seed import stable_seed


def scenario_split(scene_id: str, ratios: dict | None = None) -> str:
    ratios = ratios or {"train": 0.7, "val": 0.1, "calibration": 0.1, "test": 0.1}
    keys = ["train", "val", "calibration", "test"]
    total = sum(float(ratios.get(k, 0.0)) for k in keys)
    if total <= 0:
        return "train"
    u = stable_seed("split", scene_id) / float(2**32 - 1)
    acc = 0.0
    for k in keys:
        acc += float(ratios.get(k, 0.0)) / total
        if u <= acc:
            return k
    return "test"
