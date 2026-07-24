#!/usr/bin/env python3
"""Scene-disjoint joint calibration for v48 OC-TRAC-SR.

The calibrated policy is exactly the deployed policy:
  physical eligibility -> opportunity LCB -> harm UCB -> robust top-1 score
  -> challenge nominal only above a score-advantage threshold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ocrap.algorithms.ocmero import oc_mero
from ocrap.data.serialization import load_npz
from ocrap.evaluation.metrics import best_shared_option_index, deployable_recovery_success, post_contact_deployability_score
from ocrap.models.data import iter_sample_paths_many, scalar_metadata_for_path
from ocrap.models.inference import load_model_bundle, predict_sample


def _scalar(d: dict[str, Any], key: str, default: Any) -> Any:
    a = np.asarray(d.get(key, default))
    return a.item() if a.shape == () else a


def _sigmoid(x: float) -> float:
    x = float(np.clip(x, -30.0, 30.0))
    return float(1.0 / (1.0 + math.exp(-x)))


def _teacher_pcd(d: dict[str, Any], alpha: float, beta: float, top_m: int) -> float:
    m = np.asarray(d["m_star"], dtype=np.float64)
    p = np.asarray(d["root_probs"], dtype=np.float64)
    c = np.asarray(d.get("c_star", np.eye(m.shape[0])), dtype=np.float64)
    rv = np.asarray(d.get("root_valid", np.ones(m.shape[0])), dtype=bool)
    ov = np.asarray(d.get("option_valid", np.ones(m.shape[1])), dtype=bool)
    res = oc_mero(m, p, c, alpha=alpha, beta=beta, option_valid=ov, root_valid=rv,
                  use_lcvar=True, use_obs_kernel=True, top_m=top_m)
    opt = best_shared_option_index(res.q, p, gamma=0.0, root_valid=rv, option_valid=ov)
    drs = deployable_recovery_success(m, p, opt, root_valid=rv)
    rd = float(_scalar(d, "r_dep_star", res.r_dep))
    ro = float(_scalar(d, "r_orc_star", res.r_orc))
    return float(post_contact_deployability_score(drs, rd, max(0.0, ro - rd)))


def _group_deviation(items: list[dict[str, Any]]) -> list[float]:
    nominal = next((x for x in items if x["nominal"]), items[0])
    try:
        ref = np.asarray(nominal["data"]["prefix_states"], dtype=float)[:, :2]
    except Exception:
        return [0.0] * len(items)
    out: list[float] = []
    for x in items:
        try:
            xy = np.asarray(x["data"]["prefix_states"], dtype=float)[:, :2]
            t = min(len(ref), len(xy))
            out.append(0.0 if t <= 0 else float(np.sqrt(np.mean(np.sum((xy[:t] - ref[:t]) ** 2, axis=-1))) / 5.0))
        except Exception:
            out.append(0.0)
    return out


def _fold(scene: str, folds: int) -> int:
    return int.from_bytes(hashlib.sha1(scene.encode("utf-8", errors="replace")).digest()[:8], "big") % folds


def _wilson(k: int, n: int, *, upper: bool, z: float = 1.6448536269514722) -> float:
    if n <= 0:
        return 1.0 if upper else 0.0
    p = float(k) / float(n)
    z2 = z * z
    center = (p + z2 / (2.0 * n)) / (1.0 + z2 / n)
    radius = z * math.sqrt((p * (1.0 - p) / n) + z2 / (4.0 * n * n)) / (1.0 + z2 / n)
    return float(min(1.0, center + radius) if upper else max(0.0, center - radius))


def _auc(labels: list[bool], scores: list[float]) -> float | None:
    y = np.asarray(labels, dtype=bool)
    s = np.asarray(scores, dtype=float)
    pos = s[y]
    neg = s[~y]
    if pos.size == 0 or neg.size == 0:
        return None
    # Mann-Whitney form, tie-aware and dependency free.
    wins = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((wins + 0.5 * ties) / (pos.size * neg.size))


def _grid(values: list[float], *, minimum: float | None = None, maximum: float | None = None, n: int = 15, reverse: bool = False) -> list[float]:
    a = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if a.size == 0:
        return []
    vals = {float(np.quantile(a, q)) for q in np.linspace(0.0, 1.0, n)}
    if minimum is not None:
        vals.add(float(minimum))
    if maximum is not None:
        vals.add(float(maximum))
    out = [x for x in vals if (minimum is None or x >= minimum) and (maximum is None or x <= maximum)]
    return sorted(out, reverse=reverse)


def _top1(groups: list[dict[str, Any]], opp_thr: float, harm_thr: float, supported_macros: set[int]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for g in groups:
        eligible = [r for r in g["pairs"] if r["macro"] in supported_macros and r["opportunity"] >= opp_thr and r["harm"] <= harm_thr]
        if not eligible:
            continue
        best = sorted(eligible, key=lambda r: (-r["pred_adv"], r["candidate"]))[0]
        row = dict(best)
        row.update(scene=g["scene"], time=g["time"], fold=g["fold"], oracle_best_teacher_adv=g["oracle_best_teacher_adv"])
        out.append(row)
    return out


def _metrics(groups: list[dict[str, Any]], top1: list[dict[str, Any]], score_thr: float, pos_gain: float, neg_gain: float) -> dict[str, Any]:
    selected = [r for r in top1 if r["pred_adv"] >= score_thr]
    positive = [r for r in selected if r["teacher_adv"] >= pos_gain]
    harmful = [r for r in selected if r["teacher_adv"] <= -neg_gain]
    opportunities = [g for g in groups if g["oracle_best_teacher_adv"] >= pos_gain]
    macro_counts = Counter(int(r["macro"]) for r in selected)
    max_macro_share = max(macro_counts.values(), default=0) / max(1, len(selected))
    return {
        "num_groups": len(groups),
        "num_top1_after_joint_gate": len(top1),
        "num_selected": len(selected),
        "selection_rate": len(selected) / max(1, len(groups)),
        "num_positive_selected": len(positive),
        "precision": len(positive) / len(selected) if selected else None,
        "precision_wilson_lcb90": _wilson(len(positive), len(selected), upper=False) if selected else None,
        "num_harmful_selected": len(harmful),
        "harmful_selected_rate": len(harmful) / len(selected) if selected else None,
        "harmful_selected_ucb90": _wilson(len(harmful), len(selected), upper=True) if selected else 1.0,
        "harmful_group_exposure": len(harmful) / max(1, len(groups)),
        "harmful_group_exposure_ucb90": _wilson(len(harmful), len(groups), upper=True),
        "num_opportunities": len(opportunities),
        "positive_recall": len(positive) / len(opportunities) if opportunities else None,
        "teacher_advantage_mean": float(np.mean([r["teacher_adv"] for r in selected])) if selected else None,
        "teacher_advantage_min": float(min(r["teacher_adv"] for r in selected)) if selected else None,
        "selected_macro_counts": dict(sorted(macro_counts.items())),
        "max_selected_macro_share": float(max_macro_share),
    }


def _fit(groups: list[dict[str, Any]], args: argparse.Namespace, supported_macros: set[int]) -> tuple[dict[str, float] | None, dict[str, Any], list[dict[str, Any]]]:
    pairs = [r for g in groups for r in g["pairs"] if r["macro"] in supported_macros]
    opp_grid = _grid([r["opportunity"] for r in pairs], minimum=args.min_opportunity, n=args.grid_size, reverse=True)
    harm_grid = _grid([r["harm"] for r in pairs], maximum=args.max_harm_probability, n=args.grid_size, reverse=False)
    candidates: list[dict[str, Any]] = []
    for opp_thr in opp_grid:
        for harm_thr in harm_grid:
            top1 = _top1(groups, opp_thr, harm_thr, supported_macros)
            score_grid = _grid([r["pred_adv"] for r in top1], minimum=args.min_score_advantage, n=args.grid_size, reverse=True)
            for score_thr in score_grid:
                m = _metrics(groups, top1, score_thr, args.positive_gain, args.negative_gain)
                if (
                    m["num_selected"] >= args.min_fit_selected
                    and m["precision_wilson_lcb90"] is not None
                    and m["precision_wilson_lcb90"] >= args.min_fit_precision_lcb
                    and m["harmful_group_exposure_ucb90"] <= args.max_fit_harmful_group_ucb
                    and m["teacher_advantage_mean"] is not None
                    and m["teacher_advantage_mean"] > 0.0
                ):
                    row = dict(m)
                    row.update(opportunity_threshold=opp_thr, harm_threshold=harm_thr, score_threshold=score_thr)
                    candidates.append(row)
    if not candidates:
        return None, _metrics(groups, [], float("inf"), args.positive_gain, args.negative_gain), []
    candidates.sort(key=lambda x: (
        x["num_positive_selected"], x["precision_wilson_lcb90"], -x["harmful_group_exposure_ucb90"],
        x["num_selected"], -x["max_selected_macro_share"],
    ), reverse=True)
    best = candidates[0]
    rule = {"opportunity_threshold": best["opportunity_threshold"], "harm_threshold": best["harm_threshold"], "score_threshold": best["score_threshold"]}
    return rule, best, candidates[:40]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--bucket", choices=["near", "contact"], required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--rows-output", type=Path)
    ap.add_argument("--macro-ids", default="2,3,5,6,7")
    ap.add_argument("--folds", type=int, default=2)
    ap.add_argument("--fit-fold", type=int, default=0)
    ap.add_argument("--required-min-groups", type=int, default=120)
    ap.add_argument("--required-min-scenes", type=int, default=60)
    ap.add_argument("--min-nominal-deviation", type=float, default=0.002)
    ap.add_argument("--max-hard", type=float, default=1.0)
    ap.add_argument("--positive-gain", type=float, default=0.015)
    ap.add_argument("--negative-gain", type=float, default=0.010)
    ap.add_argument("--min-opportunity", type=float, default=0.05)
    ap.add_argument("--max-harm-probability", type=float, default=0.80)
    ap.add_argument("--min-score-advantage", type=float, default=0.0)
    ap.add_argument("--grid-size", type=int, default=15)
    ap.add_argument("--min-fit-selected", type=int, default=12)
    ap.add_argument("--min-fit-precision-lcb", type=float, default=0.50)
    ap.add_argument("--max-fit-harmful-group-ucb", type=float, default=0.08)
    ap.add_argument("--min-verify-selected", type=int, default=8)
    ap.add_argument("--min-verify-precision-lcb", type=float, default=0.40)
    ap.add_argument("--max-verify-harmful-group-ucb", type=float, default=0.10)
    ap.add_argument("--max-selected-macro-share", type=float, default=0.85)
    args = ap.parse_args()

    supported_macros = {int(x) for x in args.macro_ids.split(",") if x.strip()}
    runtime_cfg = {"selection": {"active_bucket_name": "near_contact" if args.bucket == "near" else "contact"}}
    bundle = load_model_bundle(args.checkpoint, runtime_cfg)
    if bundle is None:
        raise FileNotFoundError(args.checkpoint)
    if not bool(getattr(bundle.model, "direct_recovery_harm_head", False)):
        raise ValueError("v48 calibration requires model.direct_recovery_harm_head=true")

    alpha = float((bundle.cfg.get("ocmero", {}) or {}).get("alpha", 0.2))
    beta = float((bundle.cfg.get("ocmero", {}) or {}).get("beta", 0.2))
    top_m = int((bundle.cfg.get("ocmero", {}) or {}).get("top_m", 8))
    raw: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    paths = iter_sample_paths_many(args.dataset)
    for i, path in enumerate(paths, 1):
        split = str(scalar_metadata_for_path(path, "split_id", ""))
        if split and split not in {"calibration", "val"}:
            continue
        d = load_npz(path)
        pred = predict_sample(d, bundle, runtime_cfg)
        if pred.direct_recovery_value is None or pred.direct_recovery_opportunity_logit is None or pred.direct_recovery_harm_logit is None:
            raise ValueError("checkpoint does not expose v48 score/opportunity/harm outputs")
        row = {
            "data": d,
            "scene": str(_scalar(d, "scene_id", path.stem)),
            "time": int(_scalar(d, "time_index", 0)),
            "candidate": int(_scalar(d, "candidate_index", 0)),
            "macro": int(_scalar(d, "prefix_macro_type_id", _scalar(d, "prefix_macro_id", -1))),
            "nominal": bool(float(_scalar(d, "is_nominal", 0)) > 0.5),
            "pred": float(pred.direct_recovery_value),
            "opp_logit": float(pred.direct_recovery_opportunity_logit),
            "harm_logit": float(pred.direct_recovery_harm_logit),
            "teacher": _teacher_pcd(d, alpha, beta, top_m),
            "hard": float(_scalar(d, "hard_violation", 0.0)),
            "feasible": bool(int(_scalar(d, "feasible", 1))),
        }
        raw[(row["scene"], row["time"])].append(row)
        if i == 1 or i % 1000 == 0 or i == len(paths):
            print({"event": "v48_calibration_progress", "bucket": args.bucket, "seen": i, "total": len(paths)}, flush=True)

    groups: list[dict[str, Any]] = []
    skipped = Counter()
    for (scene, time_index), items in raw.items():
        items.sort(key=lambda x: x["candidate"])
        for x, dev in zip(items, _group_deviation(items)):
            x["deviation"] = dev
        noms = [x for x in items if x["nominal"]]
        if not noms:
            skipped["no_nominal"] += 1
            continue
        nom = noms[0]
        recs = [x for x in items if not x["nominal"] and x["macro"] in supported_macros and x["feasible"] and x["hard"] <= args.max_hard and x["deviation"] >= args.min_nominal_deviation]
        if not recs:
            skipped["no_eligible"] += 1
            continue
        pairs = []
        for r in recs:
            pairs.append({
                "candidate": r["candidate"], "macro": r["macro"], "deviation": r["deviation"],
                "pred_adv": r["pred"] - nom["pred"],
                "opportunity": _sigmoid(r["opp_logit"] - nom["opp_logit"]),
                "harm": _sigmoid(r["harm_logit"] - nom["harm_logit"]),
                "teacher_adv": r["teacher"] - nom["teacher"],
            })
        groups.append({
            "scene": scene, "time": time_index, "fold": _fold(scene, max(2, args.folds)),
            "pairs": pairs, "oracle_best_teacher_adv": max(r["teacher_adv"] for r in pairs),
        })

    fit = [g for g in groups if g["fold"] == args.fit_fold]
    verify = [g for g in groups if g["fold"] != args.fit_fold]
    rule, fit_metrics, candidates = _fit(fit, args, supported_macros)
    if rule is None:
        verify_top1: list[dict[str, Any]] = []
        all_top1: list[dict[str, Any]] = []
        score_thr = float("inf")
    else:
        verify_top1 = _top1(verify, rule["opportunity_threshold"], rule["harm_threshold"], supported_macros)
        all_top1 = _top1(groups, rule["opportunity_threshold"], rule["harm_threshold"], supported_macros)
        score_thr = rule["score_threshold"]
    verify_metrics = _metrics(verify, verify_top1, score_thr, args.positive_gain, args.negative_gain)
    all_metrics = _metrics(groups, all_top1, score_thr, args.positive_gain, args.negative_gain)

    pairs = [r for g in groups for r in g["pairs"]]
    pred = [r["pred_adv"] for r in pairs]
    teacher = [r["teacher_adv"] for r in pairs]
    corr = float(np.corrcoef(pred, teacher)[0, 1]) if len(pred) > 1 and np.std(pred) > 1e-12 and np.std(teacher) > 1e-12 else None
    unconstrained_top1 = [max(g["pairs"], key=lambda r: r["pred_adv"]) for g in groups]
    top1_pred = [r["pred_adv"] for r in unconstrained_top1]
    top1_teacher = [r["teacher_adv"] for r in unconstrained_top1]
    top1_corr = float(np.corrcoef(top1_pred, top1_teacher)[0, 1]) if len(top1_pred) > 1 and np.std(top1_pred) > 1e-12 and np.std(top1_teacher) > 1e-12 else None

    scenes = {g["scene"] for g in groups}
    fit_scenes = {g["scene"] for g in fit}
    verify_scenes = {g["scene"] for g in verify}
    warnings: list[str] = []
    if len(groups) < args.required_min_groups:
        warnings.append("insufficient calibration groups")
    if len(scenes) < args.required_min_scenes:
        warnings.append("insufficient independent calibration scenes")
    if fit_scenes & verify_scenes:
        warnings.append("fit/verify scene leakage")
    if rule is None:
        warnings.append("no joint opportunity-harm-score rule satisfied fit constraints")
    if verify_metrics["num_selected"] < args.min_verify_selected:
        warnings.append("held-out selections below requirement")
    if verify_metrics["precision_wilson_lcb90"] is None or verify_metrics["precision_wilson_lcb90"] < args.min_verify_precision_lcb:
        warnings.append("held-out precision LCB below requirement")
    if verify_metrics["harmful_group_exposure_ucb90"] > args.max_verify_harmful_group_ucb:
        warnings.append("held-out harmful exposure UCB above budget")
    if verify_metrics["max_selected_macro_share"] > args.max_selected_macro_share:
        warnings.append("held-out selections are dominated by one macro")

    valid = not warnings
    result = {
        "method": "v48_scene_disjoint_joint_policy_risk_certificate",
        "bucket": args.bucket,
        "dataset": args.dataset,
        "checkpoint": args.checkpoint,
        "valid_for_deployment": valid,
        "selection_rule": "physical -> opportunity lower bound -> harm upper bound -> robust top1 -> score challenge",
        "rule": rule,
        "selector_overrides": ({} if rule is None else {
            "direct_value_certificate": True,
            "direct_value_score_mode": True,
            "direct_value_uncertainty_mode": "risk_controlled",
            "direct_value_top1_only": True,
            "direct_value_risk_controlled_admission": True,
            "direct_value_opportunity_threshold": rule["opportunity_threshold"],
            "direct_value_harm_threshold": rule["harm_threshold"],
            "direct_value_min_advantage_lcb": rule["score_threshold"],
        }),
        "num_groups": len(groups), "num_scenes": len(scenes), "fit_groups": len(fit), "verify_groups": len(verify),
        "fit_scenes": len(fit_scenes), "verify_scenes": len(verify_scenes), "scene_overlap": len(fit_scenes & verify_scenes),
        "fit": fit_metrics, "verify": verify_metrics, "all": all_metrics,
        "candidate_positive_auc": _auc([r["teacher_adv"] >= args.positive_gain for r in pairs], [r["pred_adv"] for r in pairs]),
        "candidate_harm_auc": _auc([r["teacher_adv"] <= -args.negative_gain for r in pairs], [r["harm"] for r in pairs]),
        "candidate_pred_teacher_correlation": corr,
        "unconstrained_group_top1_correlation": top1_corr,
        "top_fit_candidates": candidates,
        "skipped": dict(skipped),
        "warnings": warnings,
        "constraints": vars(args) | {"output": str(args.output), "rows_output": str(args.rows_output) if args.rows_output else None},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.rows_output:
        args.rows_output.parent.mkdir(parents=True, exist_ok=True)
        with args.rows_output.open("w", encoding="utf-8") as f:
            for row in all_top1:
                row = dict(row)
                row["selected"] = bool(row["pred_adv"] >= score_thr)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if valid else 3


if __name__ == "__main__":
    raise SystemExit(main())
