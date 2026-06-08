from __future__ import annotations

from pathlib import Path

import numpy as np

from ocrap.data.build.diagnose import iter_sample_paths
from ocrap.data.serialization import load_npz, write_json
from ocrap.planning.selector import crisp_select


def deploy(dataset: str, checkpoint: str | None, scene_id: str, time_index: int, output: str | None = None, calibration_json: str | None = None, delta: float | None = None, cfg: dict | None = None) -> dict:
    items = []
    for p in iter_sample_paths(dataset):
        d = load_npz(p)
        if str(np.asarray(d["scene_id"]).item()) == scene_id and int(np.asarray(d["time_index"]).item()) == int(time_index):
            items.append((p, d))
    if not items:
        raise ValueError(f"No candidates found for scene_id={scene_id} time_index={time_index}")
    items.sort(key=lambda x: int(np.asarray(x[1]["candidate_index"]).item()))
    gamma = float((cfg or {}).get("selection", {}).get("gamma_rec", 0.0))
    if calibration_json:
        import json
        with Path(calibration_json).open("r", encoding="utf-8") as f:
            cal = json.load(f)
        if delta is not None and "thresholds" in cal:
            gamma = float(cal["thresholds"].get(str(delta), gamma))
        else:
            gamma = float(cal.get("gamma_rec", gamma))
    utility = np.array([float(np.asarray(d["utility"]).item()) for _, d in items])
    r_dep = np.array([float(np.asarray(d["r_dep_star"]).item()) for _, d in items])
    hard = np.array([float(np.asarray(d["hard_violation"]).item()) for _, d in items])
    harm = np.array([float(np.asarray(d["harm_proxy"]).item()) for _, d in items])
    feasible = np.array([bool(int(np.asarray(d["feasible"]).item())) for _, d in items])
    sel = crisp_select(utility, r_dep, hard, harm, feasible, gamma_rec=gamma, gamma_H=0.0, gamma_D=5.0)
    selected = items[sel.selected_index][1]
    result = {"scene_id": scene_id, "time_index": int(time_index), "selected_candidate_index": int(np.asarray(selected["candidate_index"]).item()), "reason": sel.reason, "gamma_rec": gamma, "admitted_indices": [int(np.asarray(items[i][1]["candidate_index"]).item()) for i, ok in enumerate(sel.admitted) if ok], "r_dep_star": float(np.asarray(selected["r_dep_star"]).item()), "utility": float(np.asarray(selected["utility"]).item())}
    if output:
        write_json(result, output)
    return result
