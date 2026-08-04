#!/usr/bin/env python3
"""Fit one deployment rule over multiple audit strata.

The strata (for example Near and Contact) are used only for worst-stratum
certificate constraints.  They are never exposed to the model or used to choose
a different threshold at inference.  One four-scalar rule is emitted and must be
reused unchanged by every verification worker.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ocrap.evaluation.certificate_stats import wilson_interval

RULE_KEYS = (
    "opportunity_threshold",
    "harm_threshold",
    "score_threshold",
    "rank_margin_threshold",
)


def _json_safe(x: Any) -> Any:
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    return x


def _read_jsonl(path: Path, stratum: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = {
                "scene", "time", "fold", "candidate", "macro", "proposal_rank",
                "opportunity", "harm", "pred_adv", "teacher_adv",
            }
            missing = required.difference(row)
            if missing:
                raise ValueError(f"{path}:{line_no}: missing {sorted(missing)}")
            row = dict(row)
            row["audit_stratum"] = stratum
            rows.append(row)
    if not rows:
        raise ValueError(f"empty proposal rows: {path}")
    return rows


def _parse_mapping(text: str, cast=float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in (x.strip() for x in text.split(",") if x.strip()):
        if "=" not in item:
            raise ValueError(f"expected NAME=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        out[key.strip()] = cast(value.strip())
    return out


def _bound(k: int, n: int, *, upper: bool, confidence: float) -> float:
    lo, hi = wilson_interval(k, n, confidence_level=confidence, bound_type="one_sided")
    return float(hi if upper else lo)


def _grid(values: list[float], n: int, *, include: tuple[float, ...] = ()) -> list[float]:
    a = np.asarray([float(v) for v in values if np.isfinite(v)], dtype=float)
    if a.size == 0:
        return list(include)
    values_out = {float(np.quantile(a, q)) for q in np.linspace(0.0, 1.0, max(2, n))}
    values_out.update(float(v) for v in include)
    return sorted(values_out)


def _group_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["scene"]), int(row["time"]), int(row["fold"]),
            str(row["audit_stratum"]),
        )
        grouped[key].append(row)
    out: list[dict[str, Any]] = []
    for (scene, time, fold, stratum), items in grouped.items():
        items.sort(key=lambda r: (int(r["proposal_rank"]), int(r["candidate"])))
        out.append({
            "scene": scene, "time": time, "fold": fold,
            "audit_stratum": stratum, "proposal": items,
            "has_safe_opportunity": bool(any(bool(r.get("has_safe_opportunity", False)) for r in items)),
        })
    return out


def _select(
    groups: list[dict[str, Any]], rule: dict[str, float], positive_gain: float,
    negative_gain: float,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for group in groups:
        eligible = [
            r for r in group["proposal"]
            if float(r["opportunity"]) >= rule["opportunity_threshold"]
            and float(r["harm"]) <= rule["harm_threshold"]
        ]
        if not eligible:
            continue
        chosen = min(eligible, key=lambda r: (-float(r["pred_adv"]), int(r["candidate"])))
        alternatives = [float(r["pred_adv"]) for r in eligible if r is not chosen]
        second = max(alternatives) if alternatives else float(chosen["pred_adv"] - 1.0)
        rank_margin = float(chosen["pred_adv"] - second)
        if float(chosen["pred_adv"]) < rule["score_threshold"]:
            continue
        if rank_margin < rule["rank_margin_threshold"]:
            continue
        row = dict(chosen)
        row["rank_margin"] = rank_margin
        row["safe_positive"] = bool(
            float(row["teacher_adv"]) >= positive_gain
            and not bool(row.get("teacher_harmful", float(row["teacher_adv"]) <= -negative_gain))
        )
        row["harmful"] = bool(
            row.get("teacher_harmful", float(row["teacher_adv"]) <= -negative_gain)
        )
        selected.append(row)
    return selected


def _metrics(
    groups: list[dict[str, Any]], selected: list[dict[str, Any]], confidence: float,
) -> dict[str, Any]:
    n_groups = len(groups)
    n_selected = len(selected)
    safe = sum(bool(r["safe_positive"]) for r in selected)
    harmful = sum(bool(r["harmful"]) for r in selected)
    opportunities = sum(bool(g["has_safe_opportunity"]) for g in groups)
    selected_by_macro: dict[str, int] = defaultdict(int)
    for row in selected:
        selected_by_macro[str(int(row["macro"]))] += 1
    max_macro_share = (
        max(selected_by_macro.values()) / n_selected if n_selected and selected_by_macro else 0.0
    )
    return {
        "num_groups": n_groups,
        "num_selected": n_selected,
        "num_safe_positive_selected": safe,
        "num_harmful_selected": harmful,
        "precision": safe / n_selected if n_selected else 0.0,
        "precision_wilson_lcb90": _bound(safe, n_selected, upper=False, confidence=confidence),
        "harmful_group_exposure": harmful / n_groups if n_groups else 0.0,
        "harmful_group_exposure_ucb90": _bound(harmful, n_groups, upper=True, confidence=confidence),
        "harmful_selected_rate": harmful / n_selected if n_selected else 0.0,
        "harmful_selected_ucb90": _bound(harmful, n_selected, upper=True, confidence=confidence),
        "positive_recall": safe / opportunities if opportunities else 0.0,
        "safe_opportunity_groups": opportunities,
        "teacher_advantage_mean": (
            float(np.mean([float(r["teacher_adv"]) for r in selected])) if selected else 0.0
        ),
        "teacher_advantage_min": (
            float(min(float(r["teacher_adv"]) for r in selected)) if selected else 0.0
        ),
        "max_selected_macro_share": float(max_macro_share),
        "selected_by_macro": dict(selected_by_macro),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stratum", action="append", required=True, metavar="NAME=PROPOSAL_ROWS_JSONL",
        help="Repeat for every audit stratum. The name is never a policy input.",
    )
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--grid-size", type=int, default=11)
    ap.add_argument("--positive-gain", type=float, default=0.015)
    ap.add_argument("--negative-gain", type=float, default=0.010)
    ap.add_argument("--confidence-level", type=float, default=0.90)
    ap.add_argument("--min-selected", default="near=10,contact=16")
    ap.add_argument("--min-precision-lcb", default="near=0.50,contact=0.50")
    ap.add_argument("--max-harmful-group-ucb", default="near=0.12,contact=0.14")
    ap.add_argument("--max-harmful-selected-ucb", default="near=0.22,contact=0.22")
    ap.add_argument("--max-macro-share", type=float, default=0.75)
    ap.add_argument("--min-opportunity-threshold", type=float, default=0.50)
    ap.add_argument("--max-harm-threshold", type=float, default=0.50)
    ap.add_argument("--min-score-threshold", type=float, default=0.0)
    args = ap.parse_args()

    sources: dict[str, Path] = {}
    all_rows: list[dict[str, Any]] = []
    for spec in args.stratum:
        if "=" not in spec:
            ap.error(f"--stratum expects NAME=PATH, got {spec!r}")
        name, raw_path = spec.split("=", 1)
        name = name.strip().lower()
        path = Path(raw_path).resolve()
        if name in sources:
            ap.error(f"duplicate stratum {name!r}")
        if not path.is_file():
            ap.error(f"proposal rows not found: {path}")
        sources[name] = path
        all_rows.extend(_read_jsonl(path, name))

    groups = _group_rows(all_rows)
    strata = sorted(sources)
    constraints = {
        "min_selected": _parse_mapping(args.min_selected, int),
        "min_precision_lcb": _parse_mapping(args.min_precision_lcb, float),
        "max_harmful_group_ucb": _parse_mapping(args.max_harmful_group_ucb, float),
        "max_harmful_selected_ucb": _parse_mapping(args.max_harmful_selected_ucb, float),
    }
    for field, mapping in constraints.items():
        missing = set(strata).difference(mapping)
        if missing:
            ap.error(f"{field} lacks values for {sorted(missing)}")

    opp_grid = _grid([r["opportunity"] for r in all_rows], args.grid_size, include=(0.5,))
    harm_grid = _grid([r["harm"] for r in all_rows], args.grid_size, include=(0.5,))
    score_grid = _grid([r["pred_adv"] for r in all_rows], args.grid_size, include=(0.0,))
    # Rank margins are rule-dependent. Candidate pairwise differences form a
    # conservative reusable grid without consulting a stratum-specific rule.
    rank_values = [0.0, 1.0]
    for group in groups:
        vals = sorted((float(r["pred_adv"]) for r in group["proposal"]), reverse=True)
        rank_values.extend(max(0.0, vals[i] - vals[i + 1]) for i in range(len(vals) - 1))
    rank_grid = _grid(rank_values, args.grid_size, include=(0.0,))

    best: tuple[tuple[Any, ...], dict[str, Any]] | None = None
    valid_best: tuple[tuple[Any, ...], dict[str, Any]] | None = None
    evaluated = 0
    for opp in opp_grid:
        for harm in harm_grid:
            if harm < 0.0 or harm > min(1.0, args.max_harm_threshold):
                continue
            if opp < max(0.0, args.min_opportunity_threshold) or opp > 1.0:
                continue
            for score in score_grid:
                if score < args.min_score_threshold:
                    continue
                for rank_margin in rank_grid:
                    rule = {
                        "opportunity_threshold": float(opp),
                        "harm_threshold": float(harm),
                        "score_threshold": float(score),
                        "rank_margin_threshold": float(rank_margin),
                    }
                    chosen = _select(groups, rule, args.positive_gain, args.negative_gain)
                    by_stratum: dict[str, dict[str, Any]] = {}
                    failures = 0
                    deficits = 0.0
                    for stratum in strata:
                        sg = [g for g in groups if g["audit_stratum"] == stratum]
                        ss = [r for r in chosen if r["audit_stratum"] == stratum]
                        m = _metrics(sg, ss, args.confidence_level)
                        checks = {
                            "min_selected": m["num_selected"] >= constraints["min_selected"][stratum],
                            "min_precision_lcb": m["precision_wilson_lcb90"] >= constraints["min_precision_lcb"][stratum],
                            "max_harmful_group_ucb": m["harmful_group_exposure_ucb90"] <= constraints["max_harmful_group_ucb"][stratum],
                            "max_harmful_selected_ucb": m["harmful_selected_ucb90"] <= constraints["max_harmful_selected_ucb"][stratum],
                            "max_macro_share": m["max_selected_macro_share"] <= args.max_macro_share,
                        }
                        failures += sum(not ok for ok in checks.values())
                        deficits += max(0.0, constraints["min_selected"][stratum] - m["num_selected"])
                        deficits += 100.0 * max(0.0, constraints["min_precision_lcb"][stratum] - m["precision_wilson_lcb90"])
                        deficits += 100.0 * max(0.0, m["harmful_group_exposure_ucb90"] - constraints["max_harmful_group_ucb"][stratum])
                        deficits += 100.0 * max(0.0, m["harmful_selected_ucb90"] - constraints["max_harmful_selected_ucb"][stratum])
                        deficits += 100.0 * max(0.0, m["max_selected_macro_share"] - args.max_macro_share)
                        m["checks"] = checks
                        by_stratum[stratum] = m
                    pooled = _metrics(groups, chosen, args.confidence_level)
                    min_lcb = min(by_stratum[s]["precision_wilson_lcb90"] for s in strata)
                    min_recall = min(by_stratum[s]["positive_recall"] for s in strata)
                    total_safe = sum(by_stratum[s]["num_safe_positive_selected"] for s in strata)
                    total_harmful = sum(by_stratum[s]["num_harmful_selected"] for s in strata)
                    score_key = (
                        -failures, -deficits, min_lcb, total_safe, min_recall,
                        -total_harmful, pooled["teacher_advantage_mean"], pooled["num_selected"],
                    )
                    record = {
                        "rule": rule,
                        "valid": failures == 0,
                        "constraint_failures": failures,
                        "constraint_deficit": deficits,
                        "by_stratum": by_stratum,
                        "pooled": pooled,
                    }
                    evaluated += 1
                    if best is None or score_key > best[0]:
                        best = (score_key, record)
                    if failures == 0 and (valid_best is None or score_key > valid_best[0]):
                        valid_best = (score_key, record)

    assert best is not None
    chosen_record = (valid_best or best)[1]
    valid = valid_best is not None
    source_meta = {
        name: {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "row_count": sum(r["audit_stratum"] == name for r in all_rows),
            "group_count": sum(g["audit_stratum"] == name for g in groups),
        }
        for name, path in sources.items()
    }
    result: dict[str, Any] = {
        "method_version": "v48.35_continuous_frontier_shared_rule",
        "valid": valid,
        "valid_for_deployment": valid,
        "rejection_kind": None if valid else "shared_development_rule_fit_rejection",
        "strategy_regime_conditioning": False,
        "audit_strata_only": strata,
        "shared_rule_count": 1,
        "rule": chosen_record["rule"] if valid else None,
        "diagnostic_fit_rule": None if valid else chosen_record["rule"],
        "fit": chosen_record,
        "constraints": constraints | {
            "max_macro_share": args.max_macro_share,
            "semantic_rule_domain": {
                "min_opportunity_threshold": args.min_opportunity_threshold,
                "max_harm_threshold": args.max_harm_threshold,
                "min_score_threshold": args.min_score_threshold,
            },
        },
        "sources": source_meta,
        "grid": {
            "grid_size": args.grid_size,
            "opportunity_count": len(opp_grid),
            "harm_count": len(harm_grid),
            "score_count": len(score_grid),
            "rank_margin_count": len(rank_grid),
            "evaluated_rules": evaluated,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_safe(result), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(_json_safe(result), ensure_ascii=False))
    return 0 if valid else 3


if __name__ == "__main__":
    raise SystemExit(main())
