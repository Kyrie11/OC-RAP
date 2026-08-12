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
from ocrap.algorithms.evidence_targets import (
    ComponentVetoTolerances,
    component_veto_margin_numpy,
    component_veto_terms_numpy,
)
from ocrap.data.serialization import load_npz_selected
from ocrap.evaluation.metrics import best_shared_option_index, deployable_recovery_success, post_contact_deployability_score
from ocrap.evaluation.certificate_stats import certificate_support_feasibility, wilson_interval, wilson_z
from ocrap.models.data import MODEL_SAMPLE_NPZ_KEYS, expand_split_roles, iter_sample_paths_many, scalar_metadata_for_path
from ocrap.models.inference import load_model_bundle, predict_samples


def _json_safe(value: Any) -> Any:
    """Recursively convert CLI/config/results objects to strict JSON values.

    Certificate workers receive ``pathlib.Path`` values from argparse and may
    also accumulate NumPy scalars/arrays.  A valid Natural-gate rejection must
    never be reclassified as a pipeline failure merely because one of these
    diagnostic values cannot be encoded by the stdlib JSON encoder.
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return value


def _scalar(d: dict[str, Any], key: str, default: Any) -> Any:
    a = np.asarray(d.get(key, default))
    return a.item() if a.shape == () else a


def _sigmoid(x: float) -> float:
    x = float(np.clip(x, -30.0, 30.0))
    return float(1.0 / (1.0 + math.exp(-x)))


def _normal_cdf(x: float) -> float:
    return float(0.5 * (1.0 + math.erf(float(np.clip(x, -12.0, 12.0)) / math.sqrt(2.0))))


def _finite_sample_upper_quantile(values: list[float], alpha: float) -> float:
    """One-sided split-conformal finite-sample quantile."""
    a = np.asarray([float(x) for x in values if np.isfinite(x)], dtype=float)
    if a.size == 0:
        return float("inf")
    a.sort()
    rank = int(math.ceil((a.size + 1) * (1.0 - float(alpha))))
    rank = min(max(rank, 1), int(a.size))
    return float(a[rank - 1])


def _teacher_components(d: dict[str, Any], alpha: float, beta: float, top_m: int) -> dict[str, float]:
    m = np.asarray(d["m_star"], dtype=np.float64)
    p = np.asarray(d["root_probs"], dtype=np.float64)
    c = np.asarray(d.get("c_star", np.eye(m.shape[0])), dtype=np.float64)
    rv = np.asarray(d.get("root_valid", np.ones(m.shape[0])), dtype=bool)
    ov = np.asarray(d.get("option_valid", np.ones(m.shape[1])), dtype=bool)
    res = oc_mero(m, p, c, alpha=alpha, beta=beta, option_valid=ov, root_valid=rv,
                  use_lcvar=True, use_obs_kernel=True, top_m=top_m)
    opt = best_shared_option_index(res.q, p, gamma=0.0, root_valid=rv, option_valid=ov)
    drs = float(deployable_recovery_success(m, p, opt, root_valid=rv))
    rd = float(_scalar(d, "r_dep_star", res.r_dep))
    ro = float(_scalar(d, "r_orc_star", res.r_orc))
    gap = max(0.0, ro - rd)
    return {
        "teacher": float(post_contact_deployability_score(drs, rd, gap)),
        "teacher_drs": drs,
        "teacher_r_dep": rd,
        "teacher_gap": gap,
        "teacher_hard": float(_scalar(d, "hard_violation", 0.0)),
        "teacher_harm_proxy": float(_scalar(d, "harm_proxy", 0.0)),
    }


def _is_harmful(row: dict[str, Any], negative_gain: float) -> bool:
    if "teacher_harmful" in row:
        return bool(row["teacher_harmful"])
    return float(row["teacher_adv"]) <= -float(negative_gain)


def _is_positive(
    row: dict[str, Any], positive_gain: float, negative_gain: float, *, safe_only: bool
) -> bool:
    raw = float(row["teacher_adv"]) >= float(positive_gain)
    return bool(raw and (not safe_only or not _is_harmful(row, negative_gain)))


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


def _wilson(
    k: int,
    n: int,
    *,
    upper: bool,
    confidence_level: float = 0.90,
    bound_type: str = "one_sided",
) -> float:
    lower, upper_bound = wilson_interval(
        k, n, confidence_level=confidence_level, bound_type=bound_type,
    )
    return float(upper_bound if upper else lower)


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


def _top1(
    groups: list[dict[str, Any]],
    opp_thr: float,
    harm_thr: float,
    supported_macros: set[int],
    *,
    conditional_rank_margin: bool = False,
    policy_first_no_fallback: bool = False,
    proposal_top_k: int = 1,
    evidence_rerank_top_k: bool = False,
) -> list[dict[str, Any]]:
    """Select one recovery candidate per group.

    With ``policy_first_no_fallback`` the frozen preference policy chooses its
    top recovery before evidence is checked.  If that candidate is uncertified,
    the policy abstains instead of silently falling through to a lower-ranked
    candidate that Stage-E never trained on.
    """
    out: list[dict[str, Any]] = []
    for g in groups:
        physical = [r for r in g["pairs"] if r["macro"] in supported_macros]
        if not physical:
            continue
        if evidence_rerank_top_k:
            # v48.13 TERRA: freeze a rank-based proposal, then permit evidence
            # reranking only inside that proposal.  Stage E is trained on this
            # exact distribution, so selecting a certified runner-up is no longer
            # an out-of-distribution fallback.
            ordered = sorted(physical, key=lambda r: (-r.get("rank_adv", r["pred_adv"]), r["candidate"]))
            proposal = ordered[: min(max(1, int(proposal_top_k)), len(ordered))]
            eligible = [r for r in proposal if r["opportunity"] >= opp_thr and r["harm"] <= harm_thr]
            if not eligible:
                continue
            best = sorted(eligible, key=lambda r: (-float(r["pred_adv"]), int(r["candidate"])))[0]
            alternatives = [float(r["pred_adv"]) for r in eligible if r is not best]
            second = max(alternatives) if alternatives else float(best["pred_adv"] - 1.0)
            rank_margin = float(best["pred_adv"] - second)
            best = dict(best)
            best["proposal_rank"] = 1 + proposal.index(next(r for r in proposal if int(r["candidate"]) == int(best["candidate"])))
            best["proposal_size"] = len(proposal)
        elif policy_first_no_fallback:
            best = sorted(physical, key=lambda r: (-r.get("rank_adv", r["pred_adv"]), r["candidate"]))[0]
            alternatives = [float(r.get("rank_adv", r["pred_adv"])) for r in physical if r is not best]
            if not conditional_rank_margin:
                alternatives.append(0.0)
            second = max(alternatives) if alternatives else float(best.get("rank_adv", best["pred_adv"]) - 1.0)
            rank_margin = float(best.get("rank_adv", best["pred_adv"]) - second)
            if best["opportunity"] < opp_thr or best["harm"] > harm_thr:
                continue
        else:
            eligible = [r for r in physical if r["opportunity"] >= opp_thr and r["harm"] <= harm_thr]
            if not eligible:
                continue
            best = sorted(eligible, key=lambda r: (-r.get("rank_adv", r["pred_adv"]), r["candidate"]))[0]
            alternatives = [float(r.get("rank_adv", r["pred_adv"])) for r in eligible if r is not best]
            if not conditional_rank_margin:
                alternatives.append(0.0)
            second = max(alternatives) if alternatives else float(best.get("rank_adv", best["pred_adv"]) - 1.0)
            rank_margin = float(best.get("rank_adv", best["pred_adv"]) - second)
        row = dict(best)
        row["rank_margin"] = rank_margin
        row.update(scene=g["scene"], time=g["time"], fold=g["fold"], oracle_best_teacher_adv=g["oracle_best_teacher_adv"])
        out.append(row)
    return out


def _policy_top1_pairs(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the frozen preference policy's recovery candidate per group.

    Certificate calibration must match the distribution on which the policy is
    actually evaluated.  Fitting residuals over every unused recovery candidate
    made v48.8's global conformal radius exceed the full prediction range.
    """
    out: list[dict[str, Any]] = []
    for g in groups:
        if not g.get("pairs"):
            continue
        best = sorted(
            g["pairs"],
            key=lambda r: (-float(r.get("rank_adv", r["pred_adv"])), int(r["candidate"])),
        )[0]
        row = dict(best)
        row.update(scene=g["scene"], time=g["time"], fold=g["fold"])
        out.append(row)
    return out


def _metrics(
    groups: list[dict[str, Any]], top1: list[dict[str, Any]],
    score_thr: float, rank_margin_thr: float, pos_gain: float, neg_gain: float,
    *, confidence_level: float = 0.90, bound_type: str = "one_sided",
    safe_positive_only: bool = False,
) -> dict[str, Any]:
    selected = [r for r in top1 if r["pred_adv"] >= score_thr and r.get("rank_margin", 0.0) >= rank_margin_thr]
    positive = [
        r for r in selected
        if _is_positive(r, pos_gain, neg_gain, safe_only=safe_positive_only)
    ]
    harmful = [r for r in selected if _is_harmful(r, neg_gain)]
    opportunities = [
        g for g in groups
        if any(
            _is_positive(r, pos_gain, neg_gain, safe_only=safe_positive_only)
            for r in g.get("pairs", [])
        )
    ]
    macro_counts = Counter(int(r["macro"]) for r in selected)
    max_macro_share = max(macro_counts.values(), default=0) / max(1, len(selected))
    # v48.12 TRIDENT: an absolute diversity cap is invalid when the teacher
    # opportunity distribution itself is concentrated.  Measure whether the
    # learned selector is *more* concentrated than the oracle-positive policy,
    # while retaining the raw share for reporting.
    oracle_macro_counts: Counter[int] = Counter()
    for g in opportunities:
        positive_pairs = [
            r for r in g.get("pairs", [])
            if _is_positive(r, pos_gain, neg_gain, safe_only=safe_positive_only)
        ]
        if not positive_pairs:
            continue
        oracle_best = sorted(
            positive_pairs,
            key=lambda r: (-float(r["teacher_adv"]), int(r["candidate"])),
        )[0]
        oracle_macro_counts[int(oracle_best["macro"])] += 1
    oracle_max_macro_share = (
        max(oracle_macro_counts.values(), default=0) / max(1, sum(oracle_macro_counts.values()))
    )
    macro_excess_share = max(0.0, float(max_macro_share) - float(oracle_max_macro_share))
    precision_lcb = (
        _wilson(
            len(positive), len(selected), upper=False,
            confidence_level=confidence_level, bound_type=bound_type,
        ) if selected else None
    )
    harmful_selected_ucb = (
        _wilson(
            len(harmful), len(selected), upper=True,
            confidence_level=confidence_level, bound_type=bound_type,
        ) if selected else 1.0
    )
    harmful_group_ucb = _wilson(
        len(harmful), len(groups), upper=True,
        confidence_level=confidence_level, bound_type=bound_type,
    )
    return {
        "num_groups": len(groups),
        "num_top1_after_joint_gate": len(top1),
        "num_selected": len(selected),
        "selection_rate": len(selected) / max(1, len(groups)),
        "num_positive_selected": len(positive),
        "precision": len(positive) / len(selected) if selected else None,
        "precision_wilson_lcb": precision_lcb,
        "precision_wilson_lcb90": precision_lcb,  # backward-compatible field name
        "num_harmful_selected": len(harmful),
        "harmful_selected_rate": len(harmful) / len(selected) if selected else None,
        "harmful_selected_ucb": harmful_selected_ucb,
        "harmful_selected_ucb90": harmful_selected_ucb,  # backward-compatible field name
        "harmful_group_exposure": len(harmful) / max(1, len(groups)),
        "harmful_group_exposure_ucb": harmful_group_ucb,
        "harmful_group_exposure_ucb90": harmful_group_ucb,  # backward-compatible field name
        "num_opportunities": len(opportunities),
        "opportunity_label_mode": "safe_benefit" if safe_positive_only else "raw_benefit",
        "positive_recall": len(positive) / len(opportunities) if opportunities else None,
        "teacher_advantage_mean": float(np.mean([r["teacher_adv"] for r in selected])) if selected else None,
        "teacher_advantage_min": float(min(r["teacher_adv"] for r in selected)) if selected else None,
        "selected_macro_counts": dict(sorted(macro_counts.items())),
        "max_selected_macro_share": float(max_macro_share),
        "oracle_positive_macro_counts": dict(sorted(oracle_macro_counts.items())),
        "oracle_positive_max_macro_share": float(oracle_max_macro_share),
        "selected_macro_excess_share": float(macro_excess_share),
    }


def _fit(
    groups: list[dict[str, Any]],
    args: argparse.Namespace,
    supported_macros: set[int],
) -> tuple[dict[str, float] | None, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    pairs = [r for g in groups for r in g["pairs"] if r["macro"] in supported_macros]
    opp_grid = _grid(
        [r["opportunity"] for r in pairs],
        minimum=args.min_opportunity,
        n=args.grid_size,
        reverse=True,
    )
    harm_grid = _grid(
        [r["harm"] for r in pairs],
        maximum=args.max_harm_probability,
        n=args.grid_size,
        reverse=False,
    )
    # Preserve a diagnostic frontier even when every score lies outside the
    # hard probability bounds.  Such rows can never become deployable rules,
    # but they reveal whether failure comes from opportunity saturation, harm
    # saturation, precision, support, or macro concentration.
    diagnostic_probability_violation = False
    if not opp_grid:
        opp_grid = _grid([r["opportunity"] for r in pairs], n=args.grid_size, reverse=True)
        diagnostic_probability_violation = True
    if not harm_grid:
        harm_grid = _grid([r["harm"] for r in pairs], n=args.grid_size, reverse=False)
        diagnostic_probability_violation = True
    candidates: list[dict[str, Any]] = []
    frontier: list[dict[str, Any]] = []
    for opp_thr in opp_grid:
        for harm_thr in harm_grid:
            top1 = _top1(groups, opp_thr, harm_thr, supported_macros, conditional_rank_margin=args.conditional_recovery_ranking, policy_first_no_fallback=args.policy_first_no_fallback, proposal_top_k=args.proposal_top_k, evidence_rerank_top_k=args.evidence_rerank_top_k)
            score_grid = _grid(
                [r["pred_adv"] for r in top1],
                minimum=args.min_score_advantage,
                n=args.grid_size,
                reverse=True,
            )
            rank_grid = _grid(
                [r.get("rank_margin", 0.0) for r in top1],
                minimum=args.min_rank_margin,
                n=max(7, args.grid_size // 2),
                reverse=True,
            )
            for score_thr in score_grid:
                for rank_margin_thr in rank_grid:
                    m = _metrics(
                        groups, top1, score_thr, rank_margin_thr,
                        args.positive_gain, args.negative_gain,
                        confidence_level=args.certificate_confidence_level,
                        bound_type=args.certificate_bound_type,
                        safe_positive_only=args.gate_positive_mode == "safe_benefit",
                    )
                    lcb = float(m["precision_wilson_lcb90"] or 0.0)
                    harm_ucb = float(m["harmful_group_exposure_ucb90"])
                    conditional_harm_ucb = float(m["harmful_selected_ucb90"])
                    mean_adv = float(m["teacher_advantage_mean"] or -1.0)
                    macro_violation = (
                        float(m.get("selected_macro_excess_share", 0.0)) - args.max_macro_excess_share
                        if args.macro_constraint_mode == "opportunity_normalized"
                        else float(m["max_selected_macro_share"]) - args.max_selected_macro_share
                    )
                    probability_deficit = (
                        max(0.0, float(args.min_opportunity) - float(opp_thr))
                        + max(0.0, float(harm_thr) - float(args.max_harm_probability))
                    )
                    deficit = (
                        probability_deficit
                        + max(0, args.min_fit_selected - int(m["num_selected"])) / max(args.min_fit_selected, 1)
                        + max(0.0, args.min_fit_precision_lcb - lcb)
                        + max(0.0, harm_ucb - args.max_fit_harmful_group_ucb)
                        + max(0.0, conditional_harm_ucb - args.max_fit_harmful_selected_ucb)
                        + max(0.0, macro_violation)
                        + max(0.0, -mean_adv)
                    )
                    row = dict(m)
                    row.update(
                        opportunity_threshold=opp_thr,
                        harm_threshold=harm_thr,
                        score_threshold=score_thr,
                        rank_margin_threshold=rank_margin_thr,
                        constraint_deficit=float(deficit),
                    )
                    frontier.append(row)
                    if (
                        opp_thr >= args.min_opportunity
                        and harm_thr <= args.max_harm_probability
                        and m["num_selected"] >= args.min_fit_selected
                        and m["precision_wilson_lcb90"] is not None
                        and m["precision_wilson_lcb90"] >= args.min_fit_precision_lcb
                        and m["harmful_group_exposure_ucb90"] <= args.max_fit_harmful_group_ucb
                        and m["harmful_selected_ucb90"] <= args.max_fit_harmful_selected_ucb
                        and (
                            float(m.get("selected_macro_excess_share", 0.0)) <= args.max_macro_excess_share
                            if args.macro_constraint_mode == "opportunity_normalized"
                            else m["max_selected_macro_share"] <= args.max_selected_macro_share
                        )
                        and m["teacher_advantage_mean"] is not None
                        and m["teacher_advantage_mean"] > 0.0
                    ):
                        candidates.append(row)
    frontier.sort(key=lambda x: (x["constraint_deficit"], -x["num_positive_selected"], -x["num_selected"]))
    if not candidates:
        empty = _metrics(
            groups, [], float("inf"), float("inf"), args.positive_gain, args.negative_gain,
            confidence_level=args.certificate_confidence_level,
            bound_type=args.certificate_bound_type,
            safe_positive_only=args.gate_positive_mode == "safe_benefit",
        )
        return None, empty, [], frontier[:40]
    candidates.sort(
        key=lambda x: (
            x["num_positive_selected"],
            x["precision_wilson_lcb90"],
            -x["harmful_group_exposure_ucb90"],
            -x["harmful_selected_ucb90"],
            x["num_selected"],
            -(
                x.get("selected_macro_excess_share", 0.0)
                if args.macro_constraint_mode == "opportunity_normalized"
                else x["max_selected_macro_share"]
            ),
        ),
        reverse=True,
    )
    best = candidates[0]
    rule = {
        "opportunity_threshold": best["opportunity_threshold"],
        "harm_threshold": best["harm_threshold"],
        "score_threshold": best["score_threshold"],
        "rank_margin_threshold": best["rank_margin_threshold"],
    }
    return rule, best, candidates[:40], frontier[:40]

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--method-version", default="v48_19_support_aware_factorized_policy_risk_certificate")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--bucket", choices=["near", "contact"], required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--rows-output", type=Path)
    ap.add_argument("--proposal-rows-output", type=Path)
    ap.add_argument("--diagnostic-opportunity-threshold", type=float, default=0.65)
    ap.add_argument("--diagnostic-harm-threshold", type=float, default=0.30)
    ap.add_argument("--macro-ids", default="2,3,5,6,7")
    ap.add_argument("--folds", type=int, default=2)
    ap.add_argument("--fit-fold", type=int, default=0)
    ap.add_argument(
        "--development-fit-only", action="store_true",
        help=("Fit a diagnostic threshold rule on all adaptation-dev groups. "
              "The output is never deployment-authorized."),
    )
    ap.add_argument(
        "--verification-only", action="store_true",
        help=("Evaluate one rule frozen outside the certificate on the complete certificate "
              "population. No threshold is fitted from certificate labels."),
    )
    ap.add_argument(
        "--frozen-rule-json", type=Path,
        help="JSON produced on adaptation-dev containing rule or diagnostic_fit_rule.",
    )
    ap.add_argument("--required-min-groups", type=int, default=120)
    ap.add_argument("--required-min-scenes", type=int, default=60)
    ap.add_argument("--min-nominal-deviation", type=float, default=0.002)
    ap.add_argument("--max-hard", type=float, default=1.0)
    ap.add_argument("--positive-gain", type=float, default=0.015)
    ap.add_argument("--negative-gain", type=float, default=0.010)
    ap.add_argument("--harm-label-mode", choices=["signed_advantage", "component_veto"], default="signed_advantage")
    ap.add_argument(
        "--opportunity-label-mode", choices=["raw_benefit", "safe_benefit"],
        default="raw_benefit",
        help="Semantic target of the learned opportunity head.",
    )
    ap.add_argument(
        "--gate-positive-mode", choices=["raw_benefit", "safe_benefit"],
        default="safe_benefit",
        help=("Ground-truth positive used by support, precision and recall. "
              "The Natural gate should remain safe_benefit even when the model "
              "factorizes raw benefit and component harm into separate heads."),
    )
    ap.add_argument("--component-harm-drs-tolerance", type=float, default=0.05)
    ap.add_argument("--component-harm-dep-tolerance", type=float, default=0.05)
    ap.add_argument("--component-harm-gap-tolerance", type=float, default=0.05)
    ap.add_argument("--component-harm-hard-tolerance", type=float, default=0.05)
    ap.add_argument("--component-harm-proxy-tolerance", type=float, default=0.05)
    ap.add_argument("--min-opportunity", type=float, default=0.05)
    ap.add_argument("--max-harm-probability", type=float, default=0.80)
    ap.add_argument("--min-score-advantage", type=float, default=0.0)
    ap.add_argument("--grid-size", type=int, default=15)
    ap.add_argument(
        "--certificate-confidence-level", type=float, default=0.90,
        help="Directional confidence level for Wilson precision/harm bounds.",
    )
    ap.add_argument(
        "--certificate-bound-type", choices=["one_sided", "two_sided"], default="one_sided",
        help="One-sided is the declared Natural-gate protocol; two-sided is retained for historical audits.",
    )
    ap.add_argument("--min-fit-selected", type=int, default=12)
    ap.add_argument("--min-fit-precision-lcb", type=float, default=0.50)
    ap.add_argument("--max-fit-harmful-group-ucb", type=float, default=0.08,
                    help="Maximum population harmful-exposure UCB")
    ap.add_argument("--max-fit-harmful-selected-ucb", type=float, default=1.0,
                    help="Maximum conditional harmful-switch UCB among selected actions")
    ap.add_argument("--min-verify-selected", type=int, default=8)
    ap.add_argument("--min-verify-precision-lcb", type=float, default=0.40)
    ap.add_argument("--max-verify-harmful-group-ucb", type=float, default=0.10,
                    help="Maximum population harmful-exposure UCB")
    ap.add_argument("--max-verify-harmful-selected-ucb", type=float, default=1.0,
                    help="Maximum conditional harmful-switch UCB among selected actions")
    ap.add_argument("--max-selected-macro-share", type=float, default=0.85)
    ap.add_argument(
        "--macro-constraint-mode", choices=["absolute", "opportunity_normalized"],
        default="absolute",
        help="Use an absolute selected-macro cap or limit excess concentration over the oracle-positive macro distribution.",
    )
    ap.add_argument("--max-macro-excess-share", type=float, default=0.10)
    ap.add_argument("--risk-source", choices=["direct_delta", "conformal_delta", "delta_distribution", "heads", "ordinal_evidence"], default="direct_delta")
    ap.add_argument(
        "--allowed-splits", default="calibration",
        help="Comma-separated semantic roles or concrete split_id values. Dedicated certificates should pass certificate_pool exactly.",
    )
    ap.add_argument("--conformal-alpha", type=float, default=0.10)
    ap.add_argument("--conformal-temperature", type=float, default=0.02)
    ap.add_argument(
        "--conformal-scope", choices=["all_pairs", "policy_top1"],
        default="policy_top1",
        help="Fit residuals on all candidates or on the frozen Stage-P policy top-1 distribution.",
    )
    ap.add_argument("--delta-std-floor", type=float, default=0.03)
    ap.add_argument("--min-rank-margin", type=float, default=0.0)
    ap.add_argument("--conditional-recovery-ranking", action="store_true", help="Compute rank margins only against the second recovery candidate; nominal admission is handled by the evidence gate.")
    ap.add_argument("--policy-first-no-fallback", action="store_true", help="Choose preference top-1 before evidence gating; abstain if it is uncertified instead of falling through to a runner-up.")
    ap.add_argument("--proposal-top-k", type=int, default=1, help="Number of frozen preference candidates exposed to the evidence stage.")
    ap.add_argument("--evidence-rerank-top-k", action="store_true", help="Rerank certified candidates only inside the frozen preference proposal.")
    ap.add_argument("--preference-tie-epsilon-near", type=float, default=0.025)
    ap.add_argument("--preference-tie-epsilon-contact", type=float, default=0.010)
    args = ap.parse_args()
    if args.development_fit_only and args.verification_only:
        ap.error("--development-fit-only and --verification-only are mutually exclusive")
    if args.development_fit_only and args.frozen_rule_json is not None:
        ap.error("--development-fit-only cannot use --frozen-rule-json")
    if args.verification_only and args.frozen_rule_json is None:
        ap.error("--verification-only requires --frozen-rule-json")
    if args.frozen_rule_json is not None and not args.frozen_rule_json.is_file():
        ap.error(f"frozen rule JSON not found: {args.frozen_rule_json}")

    supported_macros = {int(x) for x in args.macro_ids.split(",") if x.strip()}
    component_tolerances = ComponentVetoTolerances(
        drs=args.component_harm_drs_tolerance,
        deployability_gate=args.component_harm_dep_tolerance,
        gap_discount=args.component_harm_gap_tolerance,
        hard_violation=args.component_harm_hard_tolerance,
        harm_proxy=args.component_harm_proxy_tolerance,
    )
    runtime_cfg = {"selection": {"active_bucket_name": "near_contact" if args.bucket == "near" else "contact"}}
    bundle = load_model_bundle(args.checkpoint, runtime_cfg)
    if bundle is None:
        raise FileNotFoundError(args.checkpoint)
    if args.risk_source in {"heads", "ordinal_evidence"} and not bool(getattr(bundle.model, "direct_recovery_harm_head", False)):
        raise ValueError("risk-source=heads/ordinal_evidence requires model.direct_recovery_harm_head=true")

    alpha = float((bundle.cfg.get("ocmero", {}) or {}).get("alpha", 0.2))
    beta = float((bundle.cfg.get("ocmero", {}) or {}).get("beta", 0.2))
    top_m = int((bundle.cfg.get("ocmero", {}) or {}).get("top_m", 8))
    raw: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    paths = iter_sample_paths_many(args.dataset)
    requested_splits = {x.strip() for x in str(args.allowed_splits).split(",") if x.strip()}
    allowed_splits = expand_split_roles(requested_splits or {"calibration"})
    kept_split_counts: Counter[str] = Counter()
    for i, path in enumerate(paths, 1):
        split = str(scalar_metadata_for_path(path, "split_id", ""))
        if split not in allowed_splits:
            continue
        kept_split_counts[split] += 1
        d = load_npz_selected(path, MODEL_SAMPLE_NPZ_KEYS)
        row = {
            "data": d,
            "scene": str(_scalar(d, "scene_id", path.stem)),
            "time": int(_scalar(d, "time_index", 0)),
            "candidate": int(_scalar(d, "candidate_index", 0)),
            "macro": int(_scalar(d, "prefix_macro_type_id", _scalar(d, "prefix_macro_id", -1))),
            "nominal": bool(float(_scalar(d, "is_nominal", 0)) > 0.5),
            **_teacher_components(d, alpha, beta, top_m),
            "hard": float(_scalar(d, "hard_violation", 0.0)),
            "feasible": bool(int(_scalar(d, "feasible", 1))),
        }
        raw[(row["scene"], row["time"])].append(row)
        if i == 1 or i % 1000 == 0 or i == len(paths):
            print({"event": "v48_calibration_load_progress", "bucket": args.bucket, "seen": i, "total": len(paths)}, flush=True)

    # v48.3: score complete candidate sets jointly so the nominal-anchored set
    # context used during training is also present during calibration/deployment.
    for gi, items in enumerate(raw.values(), 1):
        items.sort(key=lambda x: x["candidate"])
        preds = predict_samples([x["data"] for x in items], bundle, runtime_cfg, shared_scene_features=True)
        for row, pred in zip(items, preds):
            if pred.direct_recovery_value is None or pred.direct_recovery_std is None:
                raise ValueError("checkpoint does not expose value mean/std outputs")
            row["pred"] = float(pred.direct_recovery_value)
            row["pred_std"] = max(float(args.delta_std_floor), float(pred.direct_recovery_std))
            row["rank"] = float(pred.direct_recovery_rank if pred.direct_recovery_rank is not None else pred.direct_recovery_value)
            row["delta"] = None if pred.direct_recovery_delta is None else float(pred.direct_recovery_delta)
            row["delta_std"] = None if pred.direct_recovery_delta_std is None else max(float(args.delta_std_floor), float(pred.direct_recovery_delta_std))
            row["opp_logit"] = None if pred.direct_recovery_opportunity_logit is None else float(pred.direct_recovery_opportunity_logit)
            row["harm_logit"] = None if pred.direct_recovery_harm_logit is None else float(pred.direct_recovery_harm_logit)
            row["component_harm"] = (
                None
                if pred.direct_recovery_component_harm is None
                else [float(x) for x in np.asarray(pred.direct_recovery_component_harm).reshape(-1)]
            )
            row["component_margins"] = (
                None
                if pred.direct_recovery_component_margins is None
                else [float(x) for x in np.asarray(pred.direct_recovery_component_margins).reshape(-1)]
            )
            if args.risk_source in {"heads", "ordinal_evidence"} and (row["opp_logit"] is None or row["harm_logit"] is None):
                raise ValueError("risk-source=heads/ordinal_evidence requires opportunity/harm outputs")
        if gi == 1 or gi % 200 == 0 or gi == len(raw):
            print({"event": "v48_calibration_group_score_progress", "bucket": args.bucket, "groups": gi, "total_groups": len(raw)}, flush=True)

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
            if args.risk_source in {"direct_delta", "conformal_delta"}:
                if r["delta"] is None or r["delta_std"] is None:
                    raise ValueError("risk-source=direct_delta requires direct delta mean/std outputs")
                pred_adv = float(r["delta"])
            elif args.risk_source == "ordinal_evidence":
                opportunity = _sigmoid(float(r["opp_logit"]) - float(nom["opp_logit"]))
                harm = _sigmoid(float(r["harm_logit"]) - float(nom["harm_logit"]))
                # v48.22 checkpoints expose a separately supervised safe-admission
                # score through direct_recovery_delta.  Legacy ordinal checkpoints
                # expose opportunity-harm through the same field, so preferring the
                # explicit delta is backward compatible and keeps calibration equal
                # to runtime reranking.
                pred_adv = float(r["delta"]) if r["delta"] is not None else opportunity - harm
            else:
                pred_adv = r["pred"] - nom["pred"]
            rank_adv = r["rank"] - nom["rank"]
            delta_std = (
                float(r["delta_std"])
                if args.risk_source in {"direct_delta", "conformal_delta"}
                else max(float(args.delta_std_floor), math.sqrt(r["pred_std"] ** 2 + nom["pred_std"] ** 2))
            )
            if args.risk_source in {"direct_delta", "conformal_delta", "delta_distribution"}:
                opportunity = _normal_cdf((pred_adv - args.positive_gain) / delta_std)
                harm = _normal_cdf((-args.negative_gain - pred_adv) / delta_std)
            elif args.risk_source != "ordinal_evidence":
                opportunity = _sigmoid(float(r["opp_logit"]) - float(nom["opp_logit"]))
                harm = _sigmoid(float(r["harm_logit"]) - float(nom["harm_logit"]))
            head_harm = None
            if r["harm_logit"] is not None and nom["harm_logit"] is not None:
                head_harm = _sigmoid(float(r["harm_logit"]) - float(nom["harm_logit"]))
            teacher_adv = r["teacher"] - nom["teacher"]
            component_terms = component_veto_terms_numpy(
                candidate_drs=r["teacher_drs"], nominal_drs=nom["teacher_drs"],
                candidate_r_dep=r["teacher_r_dep"], nominal_r_dep=nom["teacher_r_dep"],
                candidate_gap=r["teacher_gap"], nominal_gap=nom["teacher_gap"],
                candidate_hard=r["teacher_hard"], nominal_hard=nom["teacher_hard"],
                candidate_harm_proxy=r["teacher_harm_proxy"], nominal_harm_proxy=nom["teacher_harm_proxy"],
                tolerances=component_tolerances,
            )
            component_margin = float(np.max(component_terms))
            pairs.append({
                "candidate": r["candidate"], "macro": r["macro"], "deviation": r["deviation"],
                "pred_adv": pred_adv, "rank_adv": rank_adv, "delta_std": delta_std,
                "opportunity": opportunity, "harm": harm, "head_harm": head_harm,
                "predicted_component_harm": r.get("component_harm"),
                "predicted_component_margins": r.get("component_margins"),
                "teacher_adv": teacher_adv,
                "teacher_component_veto_margin": component_margin,
                "teacher_component_veto_terms": [float(x) for x in component_terms],
                "teacher_harmful": bool(component_margin > 0.0) if args.harm_label_mode == "component_veto" else bool(teacher_adv <= -args.negative_gain),
            })
        safe_pairs = [
            r for r in pairs
            if _is_positive(r, args.positive_gain, args.negative_gain, safe_only=True)
        ]
        groups.append({
            "scene": scene, "time": time_index, "fold": _fold(scene, max(2, args.folds)),
            "pairs": pairs,
            "oracle_best_teacher_adv": max(r["teacher_adv"] for r in pairs),
            "oracle_best_safe_teacher_adv": (
                max(r["teacher_adv"] for r in safe_pairs) if safe_pairs else None
            ),
            "has_safe_opportunity": bool(safe_pairs),
        })

    if args.development_fit_only:
        # All adaptation-dev groups may be used to fit the external threshold
        # rule because this mode never authorizes deployment.
        fit = list(groups)
        verify: list[dict[str, Any]] = []
        fit_support = certificate_support_feasibility(
            num_groups=len(fit),
            num_opportunities=sum(
                g["has_safe_opportunity"]
                if args.gate_positive_mode == "safe_benefit"
                else g["oracle_best_teacher_adv"] >= args.positive_gain
                for g in fit
            ),
            min_selected=args.min_fit_selected,
            min_precision_lcb=args.min_fit_precision_lcb,
            max_harmful_selected_ucb=args.max_fit_harmful_selected_ucb,
            max_harmful_group_ucb=args.max_fit_harmful_group_ucb,
            confidence_level=args.certificate_confidence_level,
            bound_type=args.certificate_bound_type,
        )
        verify_support = certificate_support_feasibility(
            num_groups=0, num_opportunities=0, min_selected=0,
            min_precision_lcb=0.0, max_harmful_selected_ucb=1.0,
            max_harmful_group_ucb=1.0,
            confidence_level=args.certificate_confidence_level,
            bound_type=args.certificate_bound_type,
        )
        verify_support["not_applicable"] = True
        verify_support["population_role"] = "adaptation_dev_threshold_fit"
        support_feasibility = {"fit": fit_support, "verify": verify_support}
        support_feasibility["overall"] = bool(fit_support["feasible"])
    elif args.verification_only:
        # Thresholds are frozen on adaptation-dev.  Every certificate scene is
        # therefore an independent verification example; no certificate label
        # is consumed for threshold fitting.  This avoids discarding half of a
        # sparse, expensive safety certificate while preserving a clean
        # threshold-source / verification-population separation.
        fit: list[dict[str, Any]] = []
        verify = list(groups)
        fit_support = certificate_support_feasibility(
            num_groups=0, num_opportunities=0, min_selected=0,
            min_precision_lcb=0.0, max_harmful_selected_ucb=1.0,
            max_harmful_group_ucb=1.0,
            confidence_level=args.certificate_confidence_level,
            bound_type=args.certificate_bound_type,
        )
        fit_support["not_applicable"] = True
        fit_support["threshold_source"] = "external_frozen_rule"
    else:
        fit = [g for g in groups if g["fold"] == args.fit_fold]
        verify = [g for g in groups if g["fold"] != args.fit_fold]
        fit_support = certificate_support_feasibility(
            num_groups=len(fit),
            num_opportunities=sum(
                g["has_safe_opportunity"]
                if args.gate_positive_mode == "safe_benefit"
                else g["oracle_best_teacher_adv"] >= args.positive_gain
                for g in fit
            ),
            min_selected=args.min_fit_selected,
            min_precision_lcb=args.min_fit_precision_lcb,
            max_harmful_selected_ucb=args.max_fit_harmful_selected_ucb,
            max_harmful_group_ucb=args.max_fit_harmful_group_ucb,
            confidence_level=args.certificate_confidence_level,
            bound_type=args.certificate_bound_type,
        )
    if not args.development_fit_only:
        verify_support = certificate_support_feasibility(
            num_groups=len(verify),
            num_opportunities=sum(
                g["has_safe_opportunity"]
                if args.gate_positive_mode == "safe_benefit"
                else g["oracle_best_teacher_adv"] >= args.positive_gain
                for g in verify
            ),
            min_selected=args.min_verify_selected,
            min_precision_lcb=args.min_verify_precision_lcb,
            max_harmful_selected_ucb=args.max_verify_harmful_selected_ucb,
            max_harmful_group_ucb=args.max_verify_harmful_group_ucb,
            confidence_level=args.certificate_confidence_level,
            bound_type=args.certificate_bound_type,
        )
        support_feasibility = {"fit": fit_support, "verify": verify_support}
        support_feasibility["overall"] = bool(
            verify_support["feasible"]
            if args.verification_only
            else fit_support["feasible"] and verify_support["feasible"]
        )
    conformal = None
    if args.risk_source == "conformal_delta":
        if args.conformal_scope == "policy_top1":
            fit_pairs = _policy_top1_pairs(fit)
        else:
            fit_pairs = [r for g in fit for r in g["pairs"]]
        q_over = _finite_sample_upper_quantile(
            [r["pred_adv"] - r["teacher_adv"] for r in fit_pairs], args.conformal_alpha
        )
        q_under = _finite_sample_upper_quantile(
            [r["teacher_adv"] - r["pred_adv"] for r in fit_pairs], args.conformal_alpha
        )
        temp = max(float(args.conformal_temperature), 1.0e-4)
        for g in groups:
            for r in g["pairs"]:
                r["gain_lcb"] = float(r["pred_adv"] - q_over)
                r["gain_ucb"] = float(r["pred_adv"] + q_under)
                # Admission is based on a held-out lower confidence bound.  The
                # harm score uses the same bound conservatively, so a candidate
                # cannot look simultaneously high-opportunity and low-risk only
                # because its learned variance collapsed.
                r["opportunity"] = _sigmoid((r["gain_lcb"] - args.positive_gain) / temp)
                r["harm"] = _sigmoid((-args.negative_gain - r["gain_lcb"]) / temp)
        conformal = {
            "alpha": float(args.conformal_alpha),
            "temperature": float(temp),
            "fit_pair_count": len(fit_pairs),
            "scope": str(args.conformal_scope),
            "overprediction_quantile": float(q_over),
            "underprediction_quantile": float(q_under),
        }
    frozen_rule_source: dict[str, Any] | None = None
    if args.frozen_rule_json is not None:
        frozen_doc = json.loads(args.frozen_rule_json.read_text(encoding="utf-8"))
        frozen_rule = frozen_doc.get("rule") or frozen_doc.get("diagnostic_fit_rule")
        required_rule_keys = {
            "opportunity_threshold", "harm_threshold",
            "score_threshold", "rank_margin_threshold",
        }
        if not isinstance(frozen_rule, dict) or not required_rule_keys.issubset(frozen_rule):
            raise ValueError(
                f"frozen rule JSON lacks rule/diagnostic_fit_rule keys: {args.frozen_rule_json}"
            )
        rule = {key: float(frozen_rule[key]) for key in sorted(required_rule_keys)}
        fit_metrics = {
            "external_frozen_rule": True,
            "source_valid_for_deployment": bool(frozen_doc.get("valid_for_deployment", False)),
            "source_rejection_kind": frozen_doc.get("rejection_kind"),
        }
        candidates = []
        near_miss = []
        frozen_rule_source = {
            "path": str(args.frozen_rule_json),
            "sha256": hashlib.sha256(args.frozen_rule_json.read_bytes()).hexdigest(),
            "selected_field": "rule" if frozen_doc.get("rule") else "diagnostic_fit_rule",
            "source_rule_satisfied_dev_constraints": bool(frozen_doc.get("rule") is not None),
            "source_valid": bool(frozen_doc.get("valid", False)),
            "source_valid_for_deployment": bool(frozen_doc.get("valid_for_deployment", False)),
            "source_rejection_kind": frozen_doc.get("rejection_kind"),
            "dataset": frozen_doc.get("dataset"),
            "requested_split_roles": frozen_doc.get("requested_split_roles"),
            "allowed_split_ids": frozen_doc.get("allowed_split_ids"),
        }
    else:
        rule, fit_metrics, candidates, near_miss = _fit(fit, args, supported_macros)
    if rule is None:
        verify_top1: list[dict[str, Any]] = []
        all_top1: list[dict[str, Any]] = []
        score_thr = float("inf")
    else:
        verify_top1 = _top1(verify, rule["opportunity_threshold"], rule["harm_threshold"], supported_macros, conditional_rank_margin=args.conditional_recovery_ranking, policy_first_no_fallback=args.policy_first_no_fallback, proposal_top_k=args.proposal_top_k, evidence_rerank_top_k=args.evidence_rerank_top_k)
        all_top1 = _top1(groups, rule["opportunity_threshold"], rule["harm_threshold"], supported_macros, conditional_rank_margin=args.conditional_recovery_ranking, policy_first_no_fallback=args.policy_first_no_fallback, proposal_top_k=args.proposal_top_k, evidence_rerank_top_k=args.evidence_rerank_top_k)
        score_thr = rule["score_threshold"]
    rank_margin_thr = float("inf") if rule is None else rule["rank_margin_threshold"]
    verify_metrics = _metrics(
        verify, verify_top1, score_thr, rank_margin_thr, args.positive_gain, args.negative_gain,
        confidence_level=args.certificate_confidence_level,
        bound_type=args.certificate_bound_type,
        safe_positive_only=args.gate_positive_mode == "safe_benefit",
    )
    all_metrics = _metrics(
        groups, all_top1, score_thr, rank_margin_thr, args.positive_gain, args.negative_gain,
        confidence_level=args.certificate_confidence_level,
        bound_type=args.certificate_bound_type,
        safe_positive_only=args.gate_positive_mode == "safe_benefit",
    )
    near_miss_verify_frontier: list[dict[str, Any]] = []
    for fit_row in near_miss[:20]:
        vtop = _top1(
            verify, float(fit_row["opportunity_threshold"]),
            float(fit_row["harm_threshold"]), supported_macros,
            conditional_rank_margin=args.conditional_recovery_ranking,
            policy_first_no_fallback=args.policy_first_no_fallback,
            proposal_top_k=args.proposal_top_k,
            evidence_rerank_top_k=args.evidence_rerank_top_k,
        )
        vm = _metrics(
            verify, vtop, float(fit_row["score_threshold"]),
            float(fit_row["rank_margin_threshold"]),
            args.positive_gain, args.negative_gain,
            confidence_level=args.certificate_confidence_level,
            bound_type=args.certificate_bound_type,
            safe_positive_only=args.gate_positive_mode == "safe_benefit",
        )
        near_miss_verify_frontier.append({
            "fit_constraint_deficit": float(fit_row.get("constraint_deficit", 0.0)),
            "rule": {k: float(fit_row[k]) for k in (
                "opportunity_threshold", "harm_threshold",
                "score_threshold", "rank_margin_threshold",
            )},
            "fit": {k: fit_row.get(k) for k in (
                "num_selected", "precision", "precision_wilson_lcb90",
                "harmful_selected_rate", "harmful_selected_ucb90",
                "positive_recall", "teacher_advantage_mean",
                "max_selected_macro_share",
            )},
            "verify": vm,
        })

    pairs = [r for g in groups for r in g["pairs"]]
    pred = [r["pred_adv"] for r in pairs]
    rank = [r["rank_adv"] for r in pairs]
    teacher = [r["teacher_adv"] for r in pairs]
    corr = float(np.corrcoef(pred, teacher)[0, 1]) if len(pred) > 1 and np.std(pred) > 1e-12 and np.std(teacher) > 1e-12 else None
    rank_corr = float(np.corrcoef(rank, teacher)[0, 1]) if len(rank) > 1 and np.std(rank) > 1e-12 and np.std(teacher) > 1e-12 else None
    unconstrained_top1: list[dict[str, Any]] = []
    top1_correct_labels: list[bool] = []
    top1_strict_labels: list[bool] = []
    top1_rank_margins: list[float] = []
    tie_epsilon = args.preference_tie_epsilon_near if args.bucket == "near" else args.preference_tie_epsilon_contact
    for g in groups:
        ordered = sorted(g["pairs"], key=lambda r: (-r["rank_adv"], r["candidate"]))
        chosen = ordered[0]
        alternative_scores = [float(r["rank_adv"]) for r in ordered[1:]]
        if not args.conditional_recovery_ranking:
            alternative_scores.append(0.0)
        second_score = max(alternative_scores) if alternative_scores else float(chosen["rank_adv"] - 1.0)
        chosen = dict(chosen)
        chosen["rank_margin"] = float(chosen["rank_adv"] - second_score)
        chosen.update(
            scene=g["scene"], time=g["time"], fold=g["fold"],
            oracle_best_teacher_adv=g["oracle_best_teacher_adv"],
        )
        unconstrained_top1.append(chosen)
        if g["oracle_best_teacher_adv"] >= args.positive_gain:
            oracle = max(g["pairs"], key=lambda r: r["teacher_adv"])
            top1_strict_labels.append(chosen["candidate"] == oracle["candidate"])
            top1_correct_labels.append(chosen["teacher_adv"] >= oracle["teacher_adv"] - tie_epsilon)
            top1_rank_margins.append(chosen["rank_margin"])
    # Proposal quality is reported separately from exact top-1.  TERRA can
    # improve closed-loop decisions even when the teacher-best recovery is only
    # identifiable as a small set.
    proposal_best_hits = 0
    proposal_positive_hits = 0
    proposal_groups = 0
    proposal_positive_groups = 0
    proposal_best_hits_positive = 0
    proposal_positive_hits_positive = 0
    proposal_evidence_top1: list[dict[str, Any]] = []
    # v48.35: deployment-exact diagnostics must use the actual frozen/fitted
    # rule, including score and rank-margin boundaries.  v48.34 incorrectly
    # used fixed 0.65/0.30 opportunity/harm thresholds and called the result
    # "exact eligible", which could disagree completely with deployment.
    proposal_deployed_rule_top1: list[dict[str, Any]] = []
    proposal_candidate_audit_rows: list[dict[str, Any]] = []
    deployed_rule_selected = {
        (str(r["scene"]), int(r["time"]), int(r["fold"])): r
        for r in all_top1
        if rule is not None
        and float(r["pred_adv"]) >= float(score_thr)
        and float(r.get("rank_margin", 0.0)) >= float(rank_margin_thr)
    }
    proposal_deployed_rule_abstentions = 0
    proposal_k = max(1, int(args.proposal_top_k))
    for g in groups:
        deployable_pairs = [r for r in g.get("pairs", []) if int(r.get("macro", -1)) in supported_macros]
        ordered_pairs = sorted(deployable_pairs, key=lambda r: (-float(r["rank_adv"]), int(r["candidate"])))
        if not ordered_pairs:
            continue
        proposal = ordered_pairs[: min(proposal_k, len(ordered_pairs))]
        oracle = max(ordered_pairs, key=lambda r: float(r["teacher_adv"]))
        proposal_groups += 1
        proposal_best_hits += int(any(int(r["candidate"]) == int(oracle["candidate"]) for r in proposal))
        proposal_positive_hits += int(any(float(r["teacher_adv"]) >= args.positive_gain for r in proposal))
        if float(oracle["teacher_adv"]) >= args.positive_gain:
            proposal_positive_groups += 1
            proposal_best_hits_positive += int(any(int(r["candidate"]) == int(oracle["candidate"]) for r in proposal))
            proposal_positive_hits_positive += int(any(float(r["teacher_adv"]) >= args.positive_gain for r in proposal))
        evidence_best = sorted(
            proposal,
            key=lambda r: (-float(r["pred_adv"]), int(r["candidate"])),
        )[0]
        deployed_key = (str(g["scene"]), int(g["time"]), int(g["fold"]))
        deployed_best = deployed_rule_selected.get(deployed_key)
        common = {
            "scene": g["scene"], "time": g["time"], "fold": g["fold"],
            "oracle_best_teacher_adv": g["oracle_best_teacher_adv"],
            "oracle_best_safe_teacher_adv": g.get("oracle_best_safe_teacher_adv"),
            "has_safe_opportunity": bool(g.get("has_safe_opportunity", False)),
        }
        evidence_row = dict(evidence_best)
        evidence_row.update(common)
        proposal_evidence_top1.append(evidence_row)
        if deployed_best is None:
            proposal_deployed_rule_abstentions += 1
        else:
            deployed_row = dict(deployed_best)
            deployed_row.update(common)
            proposal_deployed_rule_top1.append(deployed_row)
        for proposal_rank, candidate_row in enumerate(proposal, start=1):
            audit_row = dict(candidate_row)
            audit_row.update(common)
            audit_row.update({
                "proposal_rank": proposal_rank,
                "deployed_rule_opportunity_threshold": (
                    float(rule["opportunity_threshold"]) if rule is not None else None
                ),
                "deployed_rule_harm_threshold": (
                    float(rule["harm_threshold"]) if rule is not None else None
                ),
                "deployed_rule_score_threshold": (
                    float(rule["score_threshold"]) if rule is not None else None
                ),
                "deployed_rule_rank_margin_threshold": (
                    float(rule["rank_margin_threshold"]) if rule is not None else None
                ),
                "deployed_rule_eligible": bool(
                    rule is not None
                    and float(candidate_row["opportunity"]) >= float(rule["opportunity_threshold"])
                    and float(candidate_row["harm"]) <= float(rule["harm_threshold"])
                ),
                "legacy_evidence_only_chosen": int(candidate_row["candidate"]) == int(evidence_best["candidate"]),
                "deployed_rule_chosen": bool(
                    deployed_best is not None
                    and int(candidate_row["candidate"]) == int(deployed_best["candidate"])
                ),
            })
            proposal_candidate_audit_rows.append(audit_row)
    proposal_best_hit_rate = proposal_best_hits / proposal_groups if proposal_groups else None
    proposal_positive_hit_rate = proposal_positive_hits / proposal_groups if proposal_groups else None
    proposal_best_hit_rate_positive = (
        proposal_best_hits_positive / proposal_positive_groups if proposal_positive_groups else None
    )
    proposal_positive_hit_rate_positive = (
        proposal_positive_hits_positive / proposal_positive_groups if proposal_positive_groups else None
    )

    def _proposal_oracle_partition(partition, *, min_selected, min_precision_lcb,
                                   max_harmful_selected_ucb, max_harmful_group_ucb,
                                   proposal_k_override: int | None = None):
        safe_positive = 0
        nonharm_groups = 0
        for group in partition:
            deployable = [r for r in group.get("pairs", []) if int(r.get("macro", -1)) in supported_macros]
            active_proposal_k = max(1, int(
                proposal_k if proposal_k_override is None else proposal_k_override
            ))
            proposal = sorted(
                deployable, key=lambda r: (-float(r["rank_adv"]), int(r["candidate"]))
            )[: min(active_proposal_k, len(deployable))]
            safe_positive += int(any(
                float(r["teacher_adv"]) >= args.positive_gain
                and not _is_harmful(r, args.negative_gain) for r in proposal
            ))
            nonharm_groups += int(any(not _is_harmful(r, args.negative_gain) for r in proposal))
        harmful_group_ucb = _wilson(
            0, len(partition), upper=True,
            confidence_level=args.certificate_confidence_level,
            bound_type=args.certificate_bound_type,
        ) if partition else 1.0
        # Optimistically enumerate every admissible coverage.  The oracle is
        # allowed to pick safe positives first and then non-harmful dead groups;
        # therefore a negative result proves that neither model tuning nor a
        # denser threshold grid can pass the declared support constraints inside
        # the frozen proposal.  Macro concentration is intentionally ignored,
        # making this a necessary (not sufficient) feasibility certificate.
        candidates = []
        for selected in range(int(min_selected), int(nonharm_groups) + 1):
            true_positive = min(int(safe_positive), selected)
            precision_lcb = _wilson(
                true_positive, selected, upper=False,
                confidence_level=args.certificate_confidence_level,
                bound_type=args.certificate_bound_type,
            )
            harmful_selected_ucb = _wilson(
                0, selected, upper=True,
                confidence_level=args.certificate_confidence_level,
                bound_type=args.certificate_bound_type,
            )
            feasible = bool(
                precision_lcb >= float(min_precision_lcb)
                and harmful_selected_ucb <= float(max_harmful_selected_ucb)
                and harmful_group_ucb <= float(max_harmful_group_ucb)
            )
            candidates.append({
                "selected": selected,
                "true_positive": true_positive,
                "precision_lcb": precision_lcb,
                "harmful_selected_ucb": harmful_selected_ucb,
                "feasible": feasible,
            })
        feasible_candidates = [row for row in candidates if row["feasible"]]
        best = max(
            feasible_candidates or candidates or [{
                "selected": 0, "true_positive": 0, "precision_lcb": 0.0,
                "harmful_selected_ucb": 1.0, "feasible": False,
            }],
            key=lambda row: (
                bool(row["feasible"]), float(row["precision_lcb"]),
                int(row["true_positive"]), int(row["selected"]),
            ),
        )
        return {
            "num_groups": len(partition),
            "proposal_top_k": int(proposal_k if proposal_k_override is None else proposal_k_override),
            "proposal_safe_positive_groups": safe_positive,
            "proposal_nonharm_groups": nonharm_groups,
            "oracle_selected": int(best["selected"]),
            "oracle_true_positive": int(best["true_positive"]),
            "oracle_precision_lcb": float(best["precision_lcb"]),
            "oracle_harmful_selected_ucb": float(best["harmful_selected_ucb"]),
            "oracle_harmful_group_ucb": harmful_group_ucb,
            "optimistic_ignores_macro_constraint": True,
            "feasible": bool(best["feasible"]),
        }

    if args.verification_only:
        fit_oracle = {
            "num_groups": 0, "proposal_top_k": int(proposal_k),
            "proposal_safe_positive_groups": 0, "proposal_nonharm_groups": 0,
            "oracle_selected": 0, "oracle_true_positive": 0,
            "oracle_precision_lcb": None, "oracle_harmful_selected_ucb": None,
            "oracle_harmful_group_ucb": None,
            "optimistic_ignores_macro_constraint": True, "feasible": True,
            "not_applicable": True, "threshold_source": "external_frozen_rule",
        }
    else:
        fit_oracle = _proposal_oracle_partition(
            fit, min_selected=args.min_fit_selected, min_precision_lcb=args.min_fit_precision_lcb,
            max_harmful_selected_ucb=args.max_fit_harmful_selected_ucb,
            max_harmful_group_ucb=args.max_fit_harmful_group_ucb,
        )
    if args.development_fit_only:
        verify_oracle = {
            "num_groups": 0, "proposal_top_k": int(proposal_k),
            "proposal_safe_positive_groups": 0, "proposal_nonharm_groups": 0,
            "oracle_selected": 0, "oracle_true_positive": 0,
            "oracle_precision_lcb": None, "oracle_harmful_selected_ucb": None,
            "oracle_harmful_group_ucb": None,
            "optimistic_ignores_macro_constraint": True, "feasible": True,
            "not_applicable": True, "population_role": "adaptation_dev_threshold_fit",
        }
    else:
        verify_oracle = _proposal_oracle_partition(
            verify, min_selected=args.min_verify_selected, min_precision_lcb=args.min_verify_precision_lcb,
            max_harmful_selected_ucb=args.max_verify_harmful_selected_ucb,
            max_harmful_group_ucb=args.max_verify_harmful_group_ucb,
        )
    proposal_constrained_oracle_gate = {"fit": fit_oracle, "verify": verify_oracle}
    proposal_constrained_oracle_gate["overall"] = bool(
        fit_oracle["feasible"] if args.development_fit_only else
        verify_oracle["feasible"] if args.verification_only else
        fit_oracle["feasible"] and verify_oracle["feasible"]
    )
    # v48.24 SUPPORT-BRIDGE: report whether the structural support failure is
    # caused by the fixed proposal width.  The curve is diagnostic only; the
    # executed width remains the preregistered ``--proposal-top-k``.
    support_k_values = sorted({
        1, 3, 5, 8, int(proposal_k),
    })
    proposal_support_curve = {}
    for support_k in support_k_values:
        fit_support = (
            {
                "num_groups": 0, "proposal_top_k": int(support_k),
                "proposal_safe_positive_groups": 0, "proposal_nonharm_groups": 0,
                "oracle_selected": 0, "oracle_true_positive": 0,
                "oracle_precision_lcb": None, "oracle_harmful_selected_ucb": None,
                "oracle_harmful_group_ucb": None,
                "optimistic_ignores_macro_constraint": True, "feasible": True,
                "not_applicable": True, "threshold_source": "external_frozen_rule",
            }
            if args.verification_only else
            _proposal_oracle_partition(
                fit, min_selected=args.min_fit_selected,
                min_precision_lcb=args.min_fit_precision_lcb,
                max_harmful_selected_ucb=args.max_fit_harmful_selected_ucb,
                max_harmful_group_ucb=args.max_fit_harmful_group_ucb,
                proposal_k_override=support_k,
            )
        )
        verify_support = (
            {
                "num_groups": 0, "proposal_top_k": int(support_k),
                "proposal_safe_positive_groups": 0, "proposal_nonharm_groups": 0,
                "oracle_selected": 0, "oracle_true_positive": 0,
                "oracle_precision_lcb": None, "oracle_harmful_selected_ucb": None,
                "oracle_harmful_group_ucb": None,
                "optimistic_ignores_macro_constraint": True, "feasible": True,
                "not_applicable": True, "population_role": "adaptation_dev_threshold_fit",
            }
            if args.development_fit_only else
            _proposal_oracle_partition(
                verify, min_selected=args.min_verify_selected,
                min_precision_lcb=args.min_verify_precision_lcb,
                max_harmful_selected_ucb=args.max_verify_harmful_selected_ucb,
                max_harmful_group_ucb=args.max_verify_harmful_group_ucb,
                proposal_k_override=support_k,
            )
        )
        proposal_support_curve[str(support_k)] = {
            "fit": fit_support,
            "verify": verify_support,
            "overall": bool(fit_support["feasible"] if args.development_fit_only else verify_support["feasible"] if args.verification_only else fit_support["feasible"] and verify_support["feasible"]),
        }
    proposal_evidence_top1_corr = (
        float(np.corrcoef(
            [float(r["pred_adv"]) for r in proposal_evidence_top1],
            [float(r["teacher_adv"]) for r in proposal_evidence_top1],
        )[0, 1])
        if len(proposal_evidence_top1) > 1
        and np.std([float(r["pred_adv"]) for r in proposal_evidence_top1]) > 1.0e-12
        and np.std([float(r["teacher_adv"]) for r in proposal_evidence_top1]) > 1.0e-12
        else None
    )
    proposal_evidence_top1_benefit_auc = _auc(
        [float(r["teacher_adv"]) >= args.positive_gain for r in proposal_evidence_top1],
        [float(r["pred_adv"]) for r in proposal_evidence_top1],
    )
    proposal_evidence_top1_safe_benefit_auc = _auc(
        [
            float(r["teacher_adv"]) >= args.positive_gain
            and not _is_harmful(r, args.negative_gain)
            for r in proposal_evidence_top1
        ],
        [float(r["pred_adv"]) for r in proposal_evidence_top1],
    )
    proposal_evidence_top1_harm_auc = _auc(
        [_is_harmful(r, args.negative_gain) for r in proposal_evidence_top1],
        [float(r["harm"]) for r in proposal_evidence_top1],
    )
    high_opportunity_top1 = [r for r in proposal_evidence_top1 if float(r["opportunity"]) >= 0.5]
    proposal_evidence_top1_conditional_harm_auc = _auc(
        [_is_harmful(r, args.negative_gain) for r in high_opportunity_top1],
        [float(r["harm"]) for r in high_opportunity_top1],
    )

    evidence_switches = [r for r in proposal_evidence_top1 if float(r["pred_adv"]) > 0.0]
    if args.gate_positive_mode == "safe_benefit":
        evidence_nonpositive = [r for r in proposal_evidence_top1 if not bool(r.get("has_safe_opportunity", False))]
        evidence_positive = [r for r in proposal_evidence_top1 if bool(r.get("has_safe_opportunity", False))]
    else:
        evidence_nonpositive = [r for r in proposal_evidence_top1 if float(r["oracle_best_teacher_adv"]) < args.positive_gain]
        evidence_positive = [r for r in proposal_evidence_top1 if float(r["oracle_best_teacher_adv"]) >= args.positive_gain]
    evidence_false_switches = [r for r in evidence_nonpositive if float(r["pred_adv"]) > 0.0]
    evidence_harmful_switches = [r for r in evidence_switches if _is_harmful(r, args.negative_gain)]
    evidence_positive_regrets = []
    for r in evidence_positive:
        oracle_adv = (
            r.get("oracle_best_safe_teacher_adv")
            if args.gate_positive_mode == "safe_benefit"
            else r.get("oracle_best_teacher_adv")
        )
        if oracle_adv is not None:
            evidence_positive_regrets.append(max(0.0, float(oracle_adv) - float(r["teacher_adv"])))
    proposal_evidence_false_switch_rate = (
        len(evidence_false_switches) / len(evidence_nonpositive) if evidence_nonpositive else None
    )
    proposal_evidence_harmful_switch_rate = (
        len(evidence_harmful_switches) / len(evidence_switches) if evidence_switches else None
    )
    proposal_evidence_positive_regret_mean = (
        float(np.mean(evidence_positive_regrets)) if evidence_positive_regrets else None
    )

    def _proposal_policy_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
        corr_value = (
            float(np.corrcoef(
                [float(r["pred_adv"]) for r in rows],
                [float(r["teacher_adv"]) for r in rows],
            )[0, 1])
            if len(rows) > 1
            and np.std([float(r["pred_adv"]) for r in rows]) > 1.0e-12
            and np.std([float(r["teacher_adv"]) for r in rows]) > 1.0e-12
            else None
        )
        switches = [r for r in rows if float(r["pred_adv"]) > 0.0]
        if args.gate_positive_mode == "safe_benefit":
            nonpositive = [r for r in rows if not bool(r.get("has_safe_opportunity", False))]
            positive = [r for r in rows if bool(r.get("has_safe_opportunity", False))]
        else:
            nonpositive = [r for r in rows if float(r["oracle_best_teacher_adv"]) < args.positive_gain]
            positive = [r for r in rows if float(r["oracle_best_teacher_adv"]) >= args.positive_gain]
        regrets = []
        for r in positive:
            oracle_adv = (
                r.get("oracle_best_safe_teacher_adv")
                if args.gate_positive_mode == "safe_benefit"
                else r.get("oracle_best_teacher_adv")
            )
            if oracle_adv is not None:
                regrets.append(max(0.0, float(oracle_adv) - float(r["teacher_adv"])))
        return {
            "correlation": corr_value,
            "positive_auc": _auc(
                [float(r["teacher_adv"]) >= args.positive_gain for r in rows],
                [float(r["pred_adv"]) for r in rows],
            ),
            "safe_positive_auc": _auc(
                [
                    float(r["teacher_adv"]) >= args.positive_gain
                    and not _is_harmful(r, args.negative_gain) for r in rows
                ],
                [float(r["pred_adv"]) for r in rows],
            ),
            "harm_auc": _auc(
                [_is_harmful(r, args.negative_gain) for r in rows],
                [float(r["harm"]) for r in rows],
            ),
            "nonpositive_false_switch_rate": (
                sum(float(r["pred_adv"]) > 0.0 for r in nonpositive) / len(nonpositive)
                if nonpositive else None
            ),
            "harmful_switch_rate": (
                sum(_is_harmful(r, args.negative_gain) for r in switches) / len(switches)
                if switches else None
            ),
            "positive_top1_regret_mean": float(np.mean(regrets)) if regrets else None,
            "selected_count": len(rows),
        }

    deployed_rule_diag = _proposal_policy_diagnostics(proposal_deployed_rule_top1)

    top1_pred = [r["rank_adv"] for r in unconstrained_top1]
    top1_teacher = [r["teacher_adv"] for r in unconstrained_top1]
    top1_corr = float(np.corrcoef(top1_pred, top1_teacher)[0, 1]) if len(top1_pred) > 1 and np.std(top1_pred) > 1e-12 and np.std(top1_teacher) > 1e-12 else None
    policy_top1_positive_auc = _auc(
        [r["teacher_adv"] >= args.positive_gain for r in unconstrained_top1],
        [r["pred_adv"] for r in unconstrained_top1],
    )
    policy_top1_harm_auc = _auc(
        [_is_harmful(r, args.negative_gain) for r in unconstrained_top1],
        [r["harm"] for r in unconstrained_top1],
    )
    policy_top1_gain_mae = (
        float(np.mean([abs(float(r["pred_adv"]) - float(r["teacher_adv"])) for r in unconstrained_top1]))
        if unconstrained_top1 else None
    )
    recovery_switches = [r for r in unconstrained_top1 if float(r["rank_adv"]) > 0.0]
    nonpositive_groups = [r for r in unconstrained_top1 if float(r["oracle_best_teacher_adv"]) < args.positive_gain]
    positive_policy_groups = [r for r in unconstrained_top1 if float(r["oracle_best_teacher_adv"]) >= args.positive_gain]
    nonpositive_false_switches = [r for r in nonpositive_groups if float(r["rank_adv"]) > 0.0]
    harmful_ranked_switches = [
        r for r in recovery_switches if _is_harmful(r, args.negative_gain)
    ]
    positive_group_activations = [r for r in positive_policy_groups if float(r["rank_adv"]) > 0.0]
    positive_groups = [g for g in groups if g["oracle_best_teacher_adv"] >= args.positive_gain]
    top1_hits = []
    top1_strict_hits = []
    top1_regrets = []
    for g in positive_groups:
        chosen = max(g["pairs"], key=lambda r: r["rank_adv"])
        oracle = max(g["pairs"], key=lambda r: r["teacher_adv"])
        top1_strict_hits.append(chosen["candidate"] == oracle["candidate"])
        top1_hits.append(chosen["teacher_adv"] >= oracle["teacher_adv"] - tie_epsilon)
        top1_regrets.append(max(0.0, oracle["teacher_adv"] - chosen["teacher_adv"] - tie_epsilon))

    scenes = {g["scene"] for g in groups}
    fit_scenes = {g["scene"] for g in fit}
    verify_scenes = {g["scene"] for g in verify}
    warnings: list[str] = []
    if not args.verification_only and not bool(support_feasibility["fit"]["feasible"]):
        warnings.append("fit certificate specification infeasible for observed positive support")
    if not args.development_fit_only and not bool(support_feasibility["verify"]["feasible"]):
        warnings.append("verify certificate specification infeasible for observed positive support")
    if not bool(proposal_constrained_oracle_gate["overall"]):
        warnings.append("proposal-constrained safe-positive oracle cannot satisfy the declared gate")
    if len(groups) < args.required_min_groups:
        warnings.append("insufficient calibration groups")
    if len(scenes) < args.required_min_scenes:
        warnings.append("insufficient independent calibration scenes")
    if not args.verification_only and fit_scenes & verify_scenes:
        warnings.append("fit/verify scene leakage")
    if rule is None:
        warnings.append("no joint opportunity-harm-score rule satisfied fit constraints")
    if not args.development_fit_only and verify_metrics["num_selected"] < args.min_verify_selected:
        warnings.append("held-out selections below requirement")
    if not args.development_fit_only and (verify_metrics["precision_wilson_lcb90"] is None or verify_metrics["precision_wilson_lcb90"] < args.min_verify_precision_lcb):
        warnings.append("held-out precision LCB below requirement")
    if not args.development_fit_only and verify_metrics["harmful_group_exposure_ucb90"] > args.max_verify_harmful_group_ucb:
        warnings.append("held-out harmful exposure UCB above budget")
    if not args.development_fit_only and verify_metrics["harmful_selected_ucb90"] > args.max_verify_harmful_selected_ucb:
        warnings.append("held-out conditional harmful-switch UCB above budget")
    verify_macro_bad = (
        float(verify_metrics.get("selected_macro_excess_share", 0.0)) > args.max_macro_excess_share
        if args.macro_constraint_mode == "opportunity_normalized"
        else verify_metrics["max_selected_macro_share"] > args.max_selected_macro_share
    )
    if not args.development_fit_only and verify_macro_bad:
        warnings.append("held-out selections exceed the macro concentration budget")

    if (
        args.verification_only and frozen_rule_source is not None
        and not bool(frozen_rule_source.get("source_rule_satisfied_dev_constraints", False))
    ):
        warnings.append("adaptation-dev failed to produce a rule satisfying development constraints")
    valid = bool(not warnings and not args.development_fit_only)

    # A failed deployment certificate must still be usable for an explicitly
    # adaptation-dev-only shadow diagnostic.  Persist the closest *fit-derived*
    # rule separately so the deployment loader can never confuse it with an
    # authorized held-out rule.  No verify/test/stress statistic is used to
    # choose this diagnostic rule.
    diagnostic_fit_rule = None
    diagnostic_selector_overrides: dict[str, Any] = {}
    if near_miss:
        diagnostic_fit_rule = {
            key: float(near_miss[0][key]) for key in (
                "opportunity_threshold", "harm_threshold",
                "score_threshold", "rank_margin_threshold",
            )
        }
        diagnostic_selector_overrides = {
            "diagnostic_only": True,
            "selected_from": "fit_nearest_frontier",
            "direct_value_certificate": True,
            "direct_value_score_mode": True,
            "direct_value_uncertainty_mode": "risk_controlled",
            "direct_value_additive_q": 0.0,
            "direct_value_top1_only": True,
            "direct_value_policy_first_no_fallback": bool(args.policy_first_no_fallback),
            "direct_value_proposal_top_k": int(args.proposal_top_k),
            "direct_value_evidence_rerank_top_k": bool(args.evidence_rerank_top_k),
            "direct_value_risk_controlled_admission": True,
            "direct_value_opportunity_threshold": diagnostic_fit_rule["opportunity_threshold"],
            "direct_value_harm_threshold": diagnostic_fit_rule["harm_threshold"],
            "direct_value_min_advantage_lcb": diagnostic_fit_rule["score_threshold"],
            "direct_value_min_rank_margin": diagnostic_fit_rule["rank_margin_threshold"],
            "direct_value_conditional_rank_margin": bool(args.conditional_recovery_ranking),
        }

    result = {
        "method": str(args.method_version),
        "bucket": args.bucket,
        "dataset": args.dataset,
        "checkpoint": args.checkpoint,
        "certificate_mode": ("development_fit_only" if args.development_fit_only else "external_rule_full_verification" if args.verification_only else "internal_fit_verify_split"),
        "development_fit_only": bool(args.development_fit_only),
        "verification_only": bool(args.verification_only),
        "frozen_rule_source": frozen_rule_source,
        "valid_for_deployment": valid,
        "certificate_data_valid": bool(len(groups) > 0 and len(scenes) > 0),
        "certificate_support_feasible": bool(support_feasibility["overall"]),
        "gate_evaluated": bool(not args.development_fit_only and len(groups) > 0 and len(scenes) > 0),
        "rejection_kind": (
            "development_fit_only" if args.development_fit_only else
            None if valid else
            "structural_support_infeasible" if not bool(support_feasibility["overall"]) else
            "development_rule_fit_rejection"
            if args.verification_only and frozen_rule_source is not None
            and not bool(frozen_rule_source.get("source_rule_satisfied_dev_constraints", False)) else
            "certificate_verification_rejection" if args.verification_only else
            "learned_gate_rejection"
        ),
        "selection_rule": (
            "physical -> preference top-k proposal -> evidence rerank -> value challenge"
            if args.evidence_rerank_top_k else
            ("physical -> preference top1 -> evidence -> value challenge" if args.policy_first_no_fallback else "physical -> gain-distribution opportunity/harm -> preference top1 -> value challenge")
        ),
        "risk_source": args.risk_source,
        "harm_label_mode": args.harm_label_mode,
        "opportunity_label_mode": args.opportunity_label_mode,
        "gate_positive_mode": args.gate_positive_mode,
        "component_harm_tolerances": component_tolerances.__dict__,
        "certificate_confidence": {
            "level": float(args.certificate_confidence_level),
            "bound_type": str(args.certificate_bound_type),
            "wilson_z": wilson_z(
                confidence_level=args.certificate_confidence_level,
                bound_type=args.certificate_bound_type,
            ),
            "legacy_metric_suffix": "lcb90/ucb90 retained for reader compatibility",
        },
        "certificate_support_feasibility": support_feasibility,
        "proposal_constrained_oracle_gate": proposal_constrained_oracle_gate,
        "proposal_support_curve": proposal_support_curve,
        "diagnostic_fit_rule": diagnostic_fit_rule,
        "diagnostic_selector_overrides": diagnostic_selector_overrides,
        "conformal": conformal,
        "rule": rule,
        "selector_overrides": ({} if rule is None else {
            "direct_value_certificate": True,
            "direct_value_score_mode": True,
            "direct_value_uncertainty_mode": ("conformal_additive" if conformal is not None else "risk_controlled"),
            "direct_value_additive_q": (0.0 if conformal is None else conformal["overprediction_quantile"]),
            "direct_value_top1_only": True,
            "direct_value_policy_first_no_fallback": bool(args.policy_first_no_fallback),
            "direct_value_proposal_top_k": int(args.proposal_top_k),
            "direct_value_evidence_rerank_top_k": bool(args.evidence_rerank_top_k),
            "direct_value_risk_controlled_admission": True,
            "direct_value_risk_source": args.risk_source,
            "direct_value_positive_gain": args.positive_gain,
            "direct_value_negative_gain": args.negative_gain,
            "direct_value_opportunity_threshold": rule["opportunity_threshold"],
            "direct_value_harm_threshold": rule["harm_threshold"],
            "direct_value_min_advantage_lcb": rule["score_threshold"],
            "direct_value_min_rank_margin": rule["rank_margin_threshold"],
            "direct_value_conditional_rank_margin": bool(args.conditional_recovery_ranking),
            "proposal_top_k": int(args.proposal_top_k),
            "evidence_rerank_top_k": bool(args.evidence_rerank_top_k),
            "direct_value_conformal_overprediction_quantile": (None if conformal is None else conformal["overprediction_quantile"]),
            "direct_value_conformal_underprediction_quantile": (None if conformal is None else conformal["underprediction_quantile"]),
            "direct_value_conformal_temperature": (None if conformal is None else conformal["temperature"]),
        }),
        "num_input_paths": len(paths),
        "requested_split_roles": sorted(requested_splits),
        "allowed_split_ids": sorted(allowed_splits),
        "kept_split_counts": dict(kept_split_counts),
        "num_groups": len(groups), "num_scenes": len(scenes), "fit_groups": len(fit), "verify_groups": len(verify),
        "fit_scenes": len(fit_scenes), "verify_scenes": len(verify_scenes), "scene_overlap": len(fit_scenes & verify_scenes),
        "fit": fit_metrics, "verify": verify_metrics, "all": all_metrics,
        "candidate_positive_auc": _auc([r["teacher_adv"] >= args.positive_gain for r in pairs], [r["pred_adv"] for r in pairs]),
        "candidate_safe_positive_auc": _auc(
            [_is_positive(r, args.positive_gain, args.negative_gain, safe_only=True) for r in pairs],
            [r["pred_adv"] for r in pairs],
        ),
        "candidate_harm_auc": _auc([_is_harmful(r, args.negative_gain) for r in pairs], [r["harm"] for r in pairs]),
        "candidate_risk_harm_auc": _auc([_is_harmful(r, args.negative_gain) for r in pairs], [r["harm"] for r in pairs]),
        "candidate_head_harm_auc": _auc(
            [_is_harmful(r, args.negative_gain) for r in pairs if r.get("head_harm") is not None],
            [float(r["head_harm"]) for r in pairs if r.get("head_harm") is not None],
        ),
        "candidate_benefit_and_harm_overlap_count": sum(
            (r["teacher_adv"] >= args.positive_gain) and _is_harmful(r, args.negative_gain) for r in pairs
        ),
        "candidate_component_harmful_count": sum(_is_harmful(r, args.negative_gain) for r in pairs),
        "candidate_pred_teacher_correlation": corr,
        "candidate_rank_teacher_correlation": rank_corr,
        "unconstrained_group_top1_correlation": top1_corr,
        "policy_top1_positive_auc": policy_top1_positive_auc,
        "policy_top1_harm_auc": policy_top1_harm_auc,
        "proposal_top_k": int(args.proposal_top_k),
        "proposal_oracle_best_hit_rate": proposal_best_hit_rate,
        "proposal_positive_hit_rate": proposal_positive_hit_rate,
        "proposal_positive_group_count": int(proposal_positive_groups),
        "proposal_oracle_best_hit_rate_positive_groups": proposal_best_hit_rate_positive,
        "proposal_any_positive_hit_rate_positive_groups": proposal_positive_hit_rate_positive,
        "proposal_evidence_top1_correlation": proposal_evidence_top1_corr,
        "proposal_evidence_top1_positive_auc": proposal_evidence_top1_benefit_auc,
        "proposal_evidence_top1_safe_positive_auc": proposal_evidence_top1_safe_benefit_auc,
        "proposal_evidence_top1_harm_auc": proposal_evidence_top1_harm_auc,
        "proposal_evidence_top1_conditional_harm_auc": proposal_evidence_top1_conditional_harm_auc,
        "proposal_evidence_top1_high_opportunity_count": len(high_opportunity_top1),
        "proposal_evidence_nonpositive_false_switch_rate": proposal_evidence_false_switch_rate,
        "proposal_evidence_harmful_switch_rate": proposal_evidence_harmful_switch_rate,
        "proposal_evidence_positive_top1_regret_mean": proposal_evidence_positive_regret_mean,
        "legacy_evidence_only_top1_correlation": proposal_evidence_top1_corr,
        "legacy_evidence_only_top1_safe_positive_auc": proposal_evidence_top1_safe_benefit_auc,
        "legacy_evidence_only_harmful_switch_rate": proposal_evidence_harmful_switch_rate,
        "proposal_deployed_rule_top1_correlation": deployed_rule_diag["correlation"],
        "proposal_deployed_rule_top1_positive_auc": deployed_rule_diag["positive_auc"],
        "proposal_deployed_rule_top1_safe_positive_auc": deployed_rule_diag["safe_positive_auc"],
        "proposal_deployed_rule_top1_harm_auc": deployed_rule_diag["harm_auc"],
        "proposal_deployed_rule_nonpositive_false_switch_rate": deployed_rule_diag["nonpositive_false_switch_rate"],
        "proposal_deployed_rule_harmful_switch_rate": deployed_rule_diag["harmful_switch_rate"],
        "proposal_deployed_rule_positive_top1_regret_mean": deployed_rule_diag["positive_top1_regret_mean"],
        "proposal_deployed_rule_selected_count": deployed_rule_diag["selected_count"],
        "proposal_deployed_rule_abstention_count": proposal_deployed_rule_abstentions,
        "proposal_deployed_rule_abstention_rate": (
            proposal_deployed_rule_abstentions / proposal_groups if proposal_groups else None
        ),
        # Backward-compatible aliases.  Their semantics are corrected in v48.35.
        "proposal_exact_eligible_semantics": "deprecated_alias_of_deployed_rule",
        "proposal_exact_eligible_top1_correlation": deployed_rule_diag["correlation"],
        "proposal_exact_eligible_top1_positive_auc": deployed_rule_diag["positive_auc"],
        "proposal_exact_eligible_top1_safe_positive_auc": deployed_rule_diag["safe_positive_auc"],
        "proposal_exact_eligible_top1_harm_auc": deployed_rule_diag["harm_auc"],
        "proposal_exact_eligible_nonpositive_false_switch_rate": deployed_rule_diag["nonpositive_false_switch_rate"],
        "proposal_exact_eligible_harmful_switch_rate": deployed_rule_diag["harmful_switch_rate"],
        "proposal_exact_eligible_positive_top1_regret_mean": deployed_rule_diag["positive_top1_regret_mean"],
        "proposal_exact_eligible_selected_count": deployed_rule_diag["selected_count"],
        "proposal_exact_eligible_abstention_count": proposal_deployed_rule_abstentions,
        "proposal_exact_eligible_abstention_rate": (
            proposal_deployed_rule_abstentions / proposal_groups if proposal_groups else None
        ),
        "policy_top1_gain_mae": policy_top1_gain_mae,
        "unconstrained_recovery_switch_rate": (len(recovery_switches) / len(unconstrained_top1) if unconstrained_top1 else None),
        "nonpositive_group_false_switch_rate": (len(nonpositive_false_switches) / len(nonpositive_groups) if nonpositive_groups else None),
        "harmful_ranked_switch_rate": (len(harmful_ranked_switches) / len(unconstrained_top1) if unconstrained_top1 else None),
        "positive_group_recovery_activation_rate": (len(positive_group_activations) / len(positive_policy_groups) if positive_policy_groups else None),
        "positive_group_top1_accuracy": (float(np.mean(top1_hits)) if top1_hits else None),
        "positive_group_strict_top1_accuracy": (float(np.mean(top1_strict_hits)) if top1_strict_hits else None),
        "positive_group_top1_regret_mean": (float(np.mean(top1_regrets)) if top1_regrets else None),
        "top1_correctness_rank_margin_auc": _auc(top1_correct_labels, top1_rank_margins),
        "strict_top1_correctness_rank_margin_auc": _auc(top1_strict_labels, top1_rank_margins),
        "preference_tie_epsilon": tie_epsilon,
        "top_fit_candidates": candidates,
        "near_miss_frontier": near_miss,
        "near_miss_verify_frontier": near_miss_verify_frontier,
        "skipped": dict(skipped),
        "warnings": warnings,
        "constraints": _json_safe(vars(args) | {
            "output": str(args.output),
            "rows_output": str(args.rows_output) if args.rows_output else None,
            "proposal_rows_output": str(args.proposal_rows_output) if args.proposal_rows_output else None,
        }),
    }
    result = _json_safe(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.rows_output:
        args.rows_output.parent.mkdir(parents=True, exist_ok=True)
        rows_to_write = all_top1 if rule is not None else unconstrained_top1
        with args.rows_output.open("w", encoding="utf-8") as f:
            for row in rows_to_write:
                row = dict(row)
                row["selected"] = bool(
                    rule is not None
                    and row["pred_adv"] >= score_thr
                    and row.get("rank_margin", 0.0) >= rank_margin_thr
                )
                f.write(json.dumps(_json_safe(row), ensure_ascii=False) + "\n")
    if args.proposal_rows_output:
        args.proposal_rows_output.parent.mkdir(parents=True, exist_ok=True)
        with args.proposal_rows_output.open("w", encoding="utf-8") as f:
            for row in proposal_candidate_audit_rows:
                f.write(json.dumps(_json_safe(row), ensure_ascii=False) + "\n")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if len(groups) == 0 or len(scenes) == 0:
        return 4  # protocol/artifact failure: no valid certificate population
    if args.development_fit_only:
        return 0 if (rule is not None or diagnostic_fit_rule is not None) else 3
    if not bool(support_feasibility["overall"]):
        # The certificate was validly evaluated and proved that the declared
        # opportunity/gate contract lacks enough support. This is a structural
        # Natural-gate rejection (controller RC=20), not an engineering failure.
        return 3
    return 0 if valid else 3


if __name__ == "__main__":
    raise SystemExit(main())
