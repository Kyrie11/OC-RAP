#!/usr/bin/env python3
"""Summarize v48.42 physical frontier diagnostics without reading test roots."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from statistics import median

COMPONENTS = ("drs", "deployability", "gap", "hard_rule", "harm_proxy")


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _f(x, default=float("nan")) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _median(vals: list[float]) -> float | None:
    vals = [x for x in vals if math.isfinite(x)]
    return None if not vals else float(median(vals))


def _frac(flags: list[bool]) -> float | None:
    return None if not flags else float(sum(flags) / len(flags))


def _auc(pos: list[float], neg: list[float]) -> float | None:
    pos = [x for x in pos if math.isfinite(x)]
    neg = [x for x in neg if math.isfinite(x)]
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return wins / (len(pos) * len(neg))


def _summarize(path: Path, positive_gain: float) -> dict:
    rows = _rows(path)
    safe_positive = [r for r in rows if _f(r.get("teacher_adv")) > positive_gain and not bool(r.get("teacher_harmful", False))]
    harmful = [r for r in rows if bool(r.get("teacher_harmful", False))]
    out = {
        "path": str(path),
        "rows": len(rows),
        "safe_positive_rows": len(safe_positive),
        "harmful_rows": len(harmful),
        "teacher_component_terms_available": any(isinstance(r.get("teacher_component_veto_terms"), list) for r in rows),
        "components": {},
    }
    harmful_teacher_max = Counter()
    for r in harmful:
        terms = r.get("teacher_component_veto_terms")
        if isinstance(terms, list) and terms:
            valid = [_f(x) for x in terms[: len(COMPONENTS)]]
            if valid and any(math.isfinite(x) for x in valid):
                idx = max(range(len(valid)), key=lambda i: valid[i] if math.isfinite(valid[i]) else -math.inf)
                harmful_teacher_max[COMPONENTS[idx]] += 1
    out["harmful_teacher_dominant_component_counts"] = dict(harmful_teacher_max)
    for idx, name in enumerate(COMPONENTS):
        sp_prob: list[float] = []
        h_prob: list[float] = []
        sp_pred_margin: list[float] = []
        h_pred_margin: list[float] = []
        sp_teacher: list[float] = []
        h_teacher: list[float] = []
        for r, dst_prob, dst_margin, dst_teacher in [
            *[(r, sp_prob, sp_pred_margin, sp_teacher) for r in safe_positive],
            *[(r, h_prob, h_pred_margin, h_teacher) for r in harmful],
        ]:
            probs = r.get("predicted_component_harm")
            margins = r.get("predicted_component_margins")
            terms = r.get("teacher_component_veto_terms")
            if isinstance(probs, list) and idx < len(probs): dst_prob.append(_f(probs[idx]))
            if isinstance(margins, list) and idx < len(margins): dst_margin.append(_f(margins[idx]))
            if isinstance(terms, list) and idx < len(terms): dst_teacher.append(_f(terms[idx]))
        out["components"][name] = {
            "safe_positive_predicted_harm_median": _median(sp_prob),
            "safe_positive_false_veto_fraction_harm_gt_0_5": _frac([x > 0.5 for x in sp_prob if math.isfinite(x)]),
            "safe_positive_predicted_margin_median": _median(sp_pred_margin),
            "safe_positive_teacher_margin_median": _median(sp_teacher),
            "harmful_predicted_harm_median": _median(h_prob),
            "harmful_false_safe_fraction_harm_le_0_5": _frac([x <= 0.5 for x in h_prob if math.isfinite(x)]),
            "harmful_predicted_margin_median": _median(h_pred_margin),
            "harmful_teacher_margin_median": _median(h_teacher),
            "harmful_vs_safe_positive_predicted_harm_auc": _auc(h_prob, sp_prob),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputdir", type=Path, required=True)
    ap.add_argument("--variant", choices=("precision", "balanced"), default="precision")
    ap.add_argument("--positive-gain", type=float, default=0.015)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    cal = args.outputdir / "candidates" / args.variant / "calibration"
    files = {
        "near_development": cal / "dev_diagnostic_near_v48.proposal_rows.jsonl",
        "contact_development": cal / "dev_diagnostic_contact_v48.proposal_rows.jsonl",
        "near_certificate": cal / "direct_value_risk_near_v48.proposal_rows.jsonl",
        "contact_certificate": cal / "direct_value_risk_contact_v48.proposal_rows.jsonl",
    }
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise SystemExit("missing proposal rows: " + ", ".join(missing))
    report = {
        "schema": "v48.42-hpfr-frontier-diagnostics-v1",
        "variant": args.variant,
        "positive_gain": args.positive_gain,
        "diagnostic_only": True,
        "test_roots_read": False,
        "splits": {k: _summarize(v, args.positive_gain) for k, v in files.items()},
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
