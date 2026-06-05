from __future__ import annotations

import hashlib


def scenario_split(scenario_id: str, ratios: dict[str, float] | None = None) -> str:
    ratios = ratios or {"train": 0.7, "val": 0.1, "calibration": 0.1, "test": 0.1}
    h = hashlib.sha1(scenario_id.encode("utf-8")).hexdigest()
    u = int(h[:12], 16) / float(16**12)
    cum = 0.0
    for name in ["train", "val", "calibration", "test"]:
        cum += float(ratios.get(name, 0.0))
        if u < cum:
            return name
    return "test"
