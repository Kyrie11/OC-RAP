from __future__ import annotations

from pathlib import Path

import numpy as np

from ocrap.data.build.diagnose import iter_sample_paths
from ocrap.data.serialization import load_npz, write_json
from ocrap.planning.selector import crisp_select

from .metrics import deployable_recovery_success, false_recoverability_admission, nominal_utility_preservation, summarize_selection_metrics


def evaluate(dataset: str | Path, checkpoint: str | Path | None = None, output: str | Path | None = None, split: str = "test", calibration_json: str | Path | None = None, cfg: dict | None = None) -> dict:
    paths = iter_sample_paths(dataset)
    grouped: dict[tuple[str, int], list[dict]] = {}
    for p in paths:
        d = load_npz(p)
        if split and str(np.asarray(d.get("split_id", "")).item()) != split and split != "all":
            continue
        key = (str(np.asarray(d["scene_id"]).item()), int(np.asarray(d["time_index"]).item()))
        grouped.setdefault(key, []).append({"path": p, "data": d})
    records = []
    gamma = 0.0
    if calibration_json:
        import json
        with Path(calibration_json).open("r", encoding="utf-8") as f:
            cal = json.load(f)
        gamma = float(cal.get("gamma_rec", cal.get("gamma", 0.0)))
    for key, items in grouped.items():
        items.sort(key=lambda x: int(np.asarray(x["data"]["candidate_index"]).item()))
        utility = np.array([float(np.asarray(x["data"]["utility"]).item()) for x in items])
        r_dep = np.array([float(np.asarray(x["data"]["r_dep_star"]).item()) for x in items])
        hard = np.array([float(np.asarray(x["data"]["hard_violation"]).item()) for x in items])
        harm = np.array([float(np.asarray(x["data"]["harm_proxy"]).item()) for x in items])
        feasible = np.array([bool(int(np.asarray(x["data"]["feasible"]).item())) for x in items])
        sel = crisp_select(utility, r_dep, hard, harm, feasible, gamma_rec=gamma, gamma_H=0.0, gamma_D=5.0)
        sd = items[sel.selected_index]["data"]
        opt = np.argmax(np.asarray(sd["m_star"], dtype=float), axis=1)
        drs = deployable_recovery_success(sd["m_star"], sd["root_probs"], opt)
        nup = nominal_utility_preservation(utility[0], utility[sel.selected_index], sigma_u=float((cfg or {}).get("metrics", {}).get("sigma_u", 1.0)))
        records.append({"fra_cand": false_recoverability_admission(sel.admitted, r_dep), "fra_exec": float(r_dep[sel.selected_index] < 0.0), "drs": drs, "odg": float(np.asarray(sd["oracle_gap_star"]).item()), "nup": nup["bounded_NUP"], "artifact": bool(int(np.asarray(sd["i_art_star"]).item())), "selected_artifact": bool(int(np.asarray(sd["i_art_star"]).item())), "selection_reason": sel.reason})
    result = summarize_selection_metrics(records, sigma_u=float((cfg or {}).get("metrics", {}).get("sigma_u", 1.0)))
    result.update({"num_scene_time_groups": len(grouped), "split": split})
    if output:
        write_json(result, output)
    return result
