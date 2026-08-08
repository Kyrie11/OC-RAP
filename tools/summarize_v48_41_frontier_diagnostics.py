#!/usr/bin/env python3
"""Summarize v48.41 component-frontier errors without reading test roots.

Consumes only already-materialized calibration proposal rows.  It reports which
physical harm components false-veto safe-positive candidates and which fail to
veto teacher-harmful candidates.  This is diagnostic only: it never refits a
threshold or changes a gate decision.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median

COMPONENTS = ("drs", "deployability", "gap", "hard_rule", "harm_proxy")


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _safe_float(x, default=float("nan")) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _fraction(flags: list[bool]) -> float | None:
    return None if not flags else sum(bool(x) for x in flags) / len(flags)


def _median(vals: list[float]) -> float | None:
    vals = [v for v in vals if math.isfinite(v)]
    return None if not vals else float(median(vals))


def _summarize(path: Path, positive_gain: float) -> dict:
    rows = _rows(path)
    safe_positive = [
        r for r in rows
        if _safe_float(r.get("teacher_adv")) > positive_gain
        and not bool(r.get("teacher_harmful", False))
    ]
    harmful = [r for r in rows if bool(r.get("teacher_harmful", False))]
    report = {
        "path": str(path),
        "rows": len(rows),
        "safe_positive_rows": len(safe_positive),
        "harmful_rows": len(harmful),
        "component_diagnostics_available": any(
            isinstance(r.get("predicted_component_margins"), list) for r in rows
        ),
        "components": {},
    }
    for idx, name in enumerate(COMPONENTS):
        sp_margin = []
        h_margin = []
        sp_prob = []
        h_prob = []
        for r in safe_positive:
            margins = r.get("predicted_component_margins")
            probs = r.get("predicted_component_harm")
            if isinstance(margins, list) and idx < len(margins):
                sp_margin.append(_safe_float(margins[idx]))
            if isinstance(probs, list) and idx < len(probs):
                sp_prob.append(_safe_float(probs[idx]))
        for r in harmful:
            margins = r.get("predicted_component_margins")
            probs = r.get("predicted_component_harm")
            if isinstance(margins, list) and idx < len(margins):
                h_margin.append(_safe_float(margins[idx]))
            if isinstance(probs, list) and idx < len(probs):
                h_prob.append(_safe_float(probs[idx]))
        report["components"][name] = {
            "safe_positive_predicted_margin_median": _median(sp_margin),
            "safe_positive_false_veto_fraction_margin_gt_0": _fraction([v > 0 for v in sp_margin if math.isfinite(v)]),
            "safe_positive_predicted_harm_median": _median(sp_prob),
            "harmful_predicted_margin_median": _median(h_margin),
            "harmful_false_safe_fraction_margin_le_0": _fraction([v <= 0 for v in h_margin if math.isfinite(v)]),
            "harmful_predicted_harm_median": _median(h_prob),
        }
    return report


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
        "schema": "v48.41-fcfr-frontier-diagnostics-v1",
        "variant": args.variant,
        "positive_gain": args.positive_gain,
        "test_roots_read": False,
        "diagnostic_only": True,
        "splits": {k: _summarize(v, args.positive_gain) for k, v in files.items()},
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
