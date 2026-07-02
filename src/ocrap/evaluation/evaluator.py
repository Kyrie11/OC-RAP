from __future__ import annotations

from pathlib import Path

import numpy as np

from ocrap.data.serialization import load_npz, write_json
from ocrap.evaluation.baselines import BASELINES, select_baseline
from ocrap.evaluation.metrics import deployable_recovery_success, false_recoverability_admission, nominal_utility_preservation, summarize_selection_metrics
from ocrap.models.data import iter_sample_paths_many
from ocrap.models.inference import load_model_bundle, predict_sample, teacher_prediction_from_sample


def _threshold_lookup(thresholds: dict, delta: float | str | None) -> tuple[float | None, str | None]:
    if not thresholds or delta is None or str(delta) == "":
        return None, None
    candidates = [str(delta)]
    try:
        fd = float(delta)
        candidates.extend([str(fd), f"{fd:g}", f"{fd:.12g}"])
    except Exception:
        fd = None
    for key in candidates:
        if key in thresholds:
            return float(thresholds[key]), key
    if fd is not None:
        for key, val in thresholds.items():
            try:
                if abs(float(key) - fd) <= 1e-12:
                    return float(val), str(key)
            except Exception:
                continue
    return None, None


def _load_gamma(calibration_json: str | Path | None, cfg: dict | None = None) -> float:
    gamma = float(((cfg or {}).get("selection", {}) or {}).get("gamma_rec", 0.0))
    if calibration_json:
        import json

        with Path(calibration_json).open("r", encoding="utf-8") as f:
            cal = json.load(f)
        delta = ((cfg or {}).get("evaluation", {}) or {}).get("delta", "")
        found, _key = _threshold_lookup(cal.get("thresholds", {}) or {}, delta)
        if found is not None:
            gamma = float(found)
        else:
            gamma = float(cal.get("gamma_rec", cal.get("gamma", gamma)))
    if not np.isfinite(gamma) and not bool(((cfg or {}).get("evaluation", {}) or {}).get("allow_infinite_gamma", False)):
        raise ValueError(
            "Loaded gamma_rec is not finite. Check calibration.thresholds for the requested evaluation.delta, "
            "increase calibration set size / delta, or pass --set evaluation.allow_infinite_gamma=true for debugging only."
        )
    return gamma


def _method_list(cfg: dict | None) -> list[str]:
    cfg = cfg or {}
    methods = ((cfg.get("evaluation", {}) or {}).get("methods", None) if isinstance(cfg.get("evaluation", {}), dict) else None)
    if not methods:
        return ["ocrap"]
    out = [str(m).lower() for m in methods]
    return [m for m in out if m in BASELINES]


def _records_summary(records: list[dict], split: str, gamma: float, source: str, num_groups: int) -> dict:
    result = summarize_selection_metrics(records)
    if records:
        result["pred_ODG"] = float(np.mean([r["pred_odg"] for r in records]))
        result["mean_selected_teacher_R_dep"] = float(np.mean([r["selected_teacher_r_dep"] for r in records]))
        result["mean_selected_teacher_R_orc"] = float(np.mean([r["selected_teacher_r_orc"] for r in records]))
        result["mean_selected_utility"] = float(np.mean([r["selected_utility"] for r in records]))
    result.update({"num_scene_time_groups": int(num_groups), "num_records": int(len(records)), "split": split, "gamma_rec": gamma, "source": source})
    return result




def _write_method_tables(result: dict, output: str | Path) -> None:
    """Write slide-friendly method comparison tables next to the JSON output."""
    try:
        out = Path(output)
        methods = result.get("methods", {}) or {}
        order = result.get("method_order", list(methods.keys()))
        cols = [
            "FRA_exec", "FRA_cand", "DRS", "bounded_NUP", "ODG",
            "artifact_selection_rate", "mean_selected_teacher_R_dep", "mean_selected_utility",
        ]
        pretty = {
            "FRA_exec": "Executed false recovery admission ↓",
            "FRA_cand": "Admitted false-recoverable candidates ↓",
            "DRS": "Deployable recovery success ↑",
            "bounded_NUP": "Nominal utility preservation ↑",
            "ODG": "Selected oracle-to-deployable gap ↓",
            "artifact_selection_rate": "Oracle-artifact selection rate ↓",
            "mean_selected_teacher_R_dep": "Selected deployable recovery score ↑",
            "mean_selected_utility": "Selected utility ↑",
        }
        csv_lines = ["method," + ",".join(cols)]
        md = ["| Method | " + " | ".join(pretty[c] for c in cols) + " |", "|---|" + "---|" * len(cols)]
        for m in order:
            row = methods.get(m, {}) or {}
            vals = []
            for c in cols:
                v = row.get(c, None)
                vals.append("" if v is None else f"{float(v):.4f}")
            csv_lines.append(m + "," + ",".join(vals))
            md.append("| " + m + " | " + " | ".join(vals) + " |")
        out.with_suffix(".methods.csv").write_text("\n".join(csv_lines), encoding="utf-8")
        out.with_suffix(".methods.md").write_text("\n".join(md), encoding="utf-8")
    except Exception:
        return

def _dataset_label_for_path(path: Path) -> str:
    try:
        # samples/foo.npz -> dataset root name; fallback to parent.
        if path.parent.name == "samples":
            return path.parent.parent.name
        return path.parent.name
    except Exception:
        return "dataset"


def _evaluate_grouped_items(grouped: dict[tuple, list[dict]], methods: list[str], gamma: float, gamma_H: float, gamma_D: float, cfg: dict, split: str, source: str) -> tuple[dict[str, dict], dict[str, list[dict]]]:
    method_records: dict[str, list[dict]] = {m: [] for m in methods}

    for key, items in grouped.items():
        items.sort(key=lambda x: int(np.asarray(x["data"]["candidate_index"]).item()))
        utility = np.array([float(np.asarray(x["data"].get("utility", 0.0)).item()) for x in items])
        pred_r_dep = np.array([float(x["pred"].r_dep) for x in items])
        pred_r_orc = np.array([float(x["pred"].r_orc) for x in items])
        teacher_r_dep = np.array([float(np.asarray(x["data"]["r_dep_star"]).item()) for x in items])
        teacher_r_orc = np.array([float(np.asarray(x["data"]["r_orc_star"]).item()) for x in items])
        hard = np.array([float(np.asarray(x["data"].get("hard_violation", 0.0)).item()) for x in items])
        harm = np.array([float(np.asarray(x["data"].get("harm_proxy", 0.0)).item()) for x in items])
        feasible = np.array([bool(int(np.asarray(x["data"].get("feasible", 1)).item())) for x in items])

        for method in methods:
            sel = select_baseline(method, utility, pred_r_dep, teacher_r_dep, teacher_r_orc, hard, harm, feasible, gamma, gamma_H, gamma_D, cfg)
            selected_index = int(sel.selected_index)
            chosen = items[selected_index]
            sd = chosen["data"]
            # For deployable execution success, evaluate the best shared recovery
            # option under the teacher OC-MERO kernel.  This keeps DRS comparable
            # across rules that do not explicitly output a recovery option.
            q_eval = chosen["pred"].q if method == "ocrap" else chosen["teacher"].q
            selected_options = np.argmax(q_eval, axis=1) if getattr(q_eval, "ndim", 0) == 2 else 0
            drs = deployable_recovery_success(sd["m_star"], sd["root_probs"], selected_options, sd.get("root_valid", None))
            nup = nominal_utility_preservation(utility[0] if len(utility) else 0.0, utility[selected_index], sigma_u=float((cfg or {}).get("metrics", {}).get("sigma_u", 1.0)))
            method_records[method].append({
                "fra_cand": false_recoverability_admission(sel.admitted, teacher_r_dep),
                "fra_exec": float(teacher_r_dep[selected_index] < 0.0),
                "drs": drs,
                "odg": float(np.asarray(sd.get("oracle_gap_star", teacher_r_orc[selected_index] - teacher_r_dep[selected_index])).item()),
                "pred_odg": float(pred_r_orc[selected_index] - pred_r_dep[selected_index]),
                "nup": nup["bounded_NUP"],
                "artifact": bool(int(np.asarray(sd.get("i_art_star", 0)).item())),
                "selected_artifact": bool(int(np.asarray(sd.get("i_art_star", 0)).item())),
                "selection_reason": sel.reason,
                "selected_index": selected_index,
                "selected_utility": float(utility[selected_index]),
                "selected_teacher_r_dep": float(teacher_r_dep[selected_index]),
                "selected_teacher_r_orc": float(teacher_r_orc[selected_index]),
            })

    summaries = {m: _records_summary(rs, split, gamma, source if m == "ocrap" else "dataset_label_baseline", len(grouped)) for m, rs in method_records.items()}
    return summaries, method_records


def evaluate(dataset: str | Path, checkpoint: str | Path | None = None, output: str | Path | None = None, split: str = "test", calibration_json: str | Path | None = None, cfg: dict | None = None) -> dict:
    cfg = cfg or {}
    paths = iter_sample_paths_many(dataset)
    print({"event": "evaluate_start", "num_npz_paths": len(paths), "split": split, "dataset": str(dataset)}, flush=True)
    bundle = load_model_bundle(checkpoint, cfg)
    eval_cfg = cfg.get("evaluation", {}) if isinstance(cfg.get("evaluation", {}), dict) else {}
    group_by_dataset = bool(eval_cfg.get("group_by_dataset", True))
    grouped: dict[tuple, list[dict]] = {}
    dataset_grouped: dict[str, dict[tuple, list[dict]]] = {}
    for idx, p in enumerate(paths, 1):
        if idx == 1 or idx % 1000 == 0:
            print({"event": "evaluate_progress", "seen": idx, "groups": len(grouped)}, flush=True)
        d = load_npz(p)
        if split and str(np.asarray(d.get("split_id", "")).item()) != split and split != "all":
            continue
        dataset_label = _dataset_label_for_path(Path(p))
        key_base = (str(np.asarray(d["scene_id"]).item()), int(np.asarray(d["time_index"]).item()))
        key = (dataset_label, *key_base) if group_by_dataset else key_base
        pred = predict_sample(d, bundle, cfg)
        teacher = teacher_prediction_from_sample(d, cfg)
        record = {"path": p, "dataset_label": dataset_label, "data": d, "pred": pred, "teacher": teacher}
        grouped.setdefault(key, []).append(record)
        dataset_grouped.setdefault(dataset_label, {}).setdefault(key_base, []).append(record)

    print({"event": "evaluate_grouping_done", "num_scene_time_groups": len(grouped), "group_by_dataset": group_by_dataset, "dataset_labels": sorted(dataset_grouped)}, flush=True)
    gamma = _load_gamma(calibration_json, cfg)
    sel_cfg = cfg.get("selection", {}) if isinstance(cfg.get("selection", {}), dict) else {}
    gamma_H = float(sel_cfg.get("gamma_H", 0.0))
    gamma_D = float(sel_cfg.get("gamma_D", 5.0))
    methods = _method_list(cfg)
    source = "model" if bundle is not None else "teacher_fallback"
    summaries, _records = _evaluate_grouped_items(grouped, methods, gamma, gamma_H, gamma_D, cfg, split, source)
    result = summaries.get("ocrap", next(iter(summaries.values()), {}))
    result = dict(result)
    result["methods"] = summaries
    result["method_order"] = methods
    result["group_by_dataset"] = group_by_dataset
    result["dataset_group_count"] = {k: len(v) for k, v in sorted(dataset_grouped.items())}
    result["per_dataset"] = {}
    for label, sub_grouped in sorted(dataset_grouped.items()):
        sub_summaries, _ = _evaluate_grouped_items(sub_grouped, methods, gamma, gamma_H, gamma_D, cfg, split, source)
        sub_result = dict(sub_summaries.get("ocrap", next(iter(sub_summaries.values()), {})))
        sub_result["methods"] = sub_summaries
        sub_result["method_order"] = methods
        result["per_dataset"][label] = sub_result
    if output:
        write_json(result, output)
        _write_method_tables(result, output)
    return result
