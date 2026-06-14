from __future__ import annotations

from pathlib import Path

import numpy as np

from ocrap.data.serialization import load_npz, write_json
from ocrap.models.data import iter_sample_paths_many
from ocrap.models.inference import load_model_bundle, predict_sample
from ocrap.planning.selector import crisp_select

from .metrics import deployable_recovery_success, false_recoverability_admission, nominal_utility_preservation, summarize_selection_metrics


def _load_gamma(calibration_json: str | Path | None, cfg: dict | None = None) -> float:
    gamma = float(((cfg or {}).get("selection", {}) or {}).get("gamma_rec", 0.0))
    if calibration_json:
        import json

        with Path(calibration_json).open("r", encoding="utf-8") as f:
            cal = json.load(f)
        delta = str(((cfg or {}).get("evaluation", {}) or {}).get("delta", ""))
        if delta and "thresholds" in cal and delta in cal["thresholds"]:
            gamma = float(cal["thresholds"][delta])
        else:
            gamma = float(cal.get("gamma_rec", cal.get("gamma", gamma)))
    return gamma


def evaluate(dataset: str | Path, checkpoint: str | Path | None = None, output: str | Path | None = None, split: str = "test", calibration_json: str | Path | None = None, cfg: dict | None = None) -> dict:
    cfg = cfg or {}
    paths = iter_sample_paths_many(dataset)
    bundle = load_model_bundle(checkpoint, cfg)
    grouped: dict[tuple[str, int], list[dict]] = {}
    for p in paths:
        d = load_npz(p)
        if split and str(np.asarray(d.get("split_id", "")).item()) != split and split != "all":
            continue
        key = (str(np.asarray(d["scene_id"]).item()), int(np.asarray(d["time_index"]).item()))
        pred = predict_sample(d, bundle, cfg)
        grouped.setdefault(key, []).append({"path": p, "data": d, "pred": pred})
    records = []
    gamma = _load_gamma(calibration_json, cfg)
    sel_cfg = cfg.get("selection", {}) if isinstance(cfg.get("selection", {}), dict) else {}
    for key, items in grouped.items():
        items.sort(key=lambda x: int(np.asarray(x["data"]["candidate_index"]).item()))
        utility = np.array([float(np.asarray(x["data"]["utility"]).item()) for x in items])
        pred_r_dep = np.array([float(x["pred"].r_dep) for x in items])
        pred_r_orc = np.array([float(x["pred"].r_orc) for x in items])
        teacher_r_dep = np.array([float(np.asarray(x["data"]["r_dep_star"]).item()) for x in items])
        hard = np.array([float(np.asarray(x["data"]["hard_violation"]).item()) for x in items])
        harm = np.array([float(np.asarray(x["data"]["harm_proxy"]).item()) for x in items])
        feasible = np.array([bool(int(np.asarray(x["data"]["feasible"]).item())) for x in items])
        sel = crisp_select(
            utility,
            pred_r_dep,
            hard,
            harm,
            feasible,
            gamma_rec=gamma,
            gamma_H=float(sel_cfg.get("gamma_H", 0.0)),
            gamma_D=float(sel_cfg.get("gamma_D", 5.0)),
        )
        chosen = items[sel.selected_index]
        sd = chosen["data"]
        pred_q = chosen["pred"].q
        selected_options = np.argmax(pred_q, axis=1) if pred_q.ndim == 2 else 0
        drs = deployable_recovery_success(sd["m_star"], sd["root_probs"], selected_options)
        nup = nominal_utility_preservation(utility[0], utility[sel.selected_index], sigma_u=float((cfg or {}).get("metrics", {}).get("sigma_u", 1.0)))
        records.append({
            "fra_cand": false_recoverability_admission(sel.admitted, teacher_r_dep),
            "fra_exec": float(teacher_r_dep[sel.selected_index] < 0.0),
            "drs": drs,
            "odg": float(np.asarray(sd["oracle_gap_star"]).item()),
            "pred_odg": float(pred_r_orc[sel.selected_index] - pred_r_dep[sel.selected_index]),
            "nup": nup["bounded_NUP"],
            "artifact": bool(int(np.asarray(sd["i_art_star"]).item())),
            "selected_artifact": bool(int(np.asarray(sd["i_art_star"]).item())),
            "selection_reason": sel.reason,
        })
    result = summarize_selection_metrics(records, sigma_u=float((cfg or {}).get("metrics", {}).get("sigma_u", 1.0)))
    if records:
        result["pred_ODG"] = float(np.mean([r["pred_odg"] for r in records]))
    result.update({"num_scene_time_groups": len(grouped), "split": split, "gamma_rec": gamma, "source": "model" if bundle is not None else "teacher_fallback"})
    if output:
        write_json(result, output)
    return result
