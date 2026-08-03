#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


KEYS = {
    "safe": [
        "closed_loop_bounded_NUP", "intervention_rate", "min_clearance_m_min",
        "ttc_s_min", "overlap_duration_s", "offroad_any", "acceleration_abs_p95_mps2",
        "jerk_p95", "yaw_rate_p95", "route_progression_m",
    ],
    "near_contact": [
        "min_clearance_m_min", "min_clearance_m_p05", "ttc_s_min", "ttc_s_p05",
        "near_contact_exposure_rate", "critical_ttc_exposure_rate",
        "clearance_deficit_auc_m_s", "ttc_deficit_auc_s2", "overlap_duration_s",
        "secondary_overlap_event", "intervention_rate",
    ],
    "contact": [
        "overlap_duration_s", "longest_overlap_run_s", "recontact_event",
        "secondary_overlap_event", "post_contact_clearance_m_mean",
        "post_contact_terminal_clearance_m", "post_contact_free_space_auc_m_s",
        "post_contact_escape_event", "new_stable_stop_quality_event",
        "yaw_rate_p95", "jerk_p95", "offroad_any", "intervention_rate",
    ],
}


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    ap = argparse.ArgumentParser(description="Create a compact three-regime ungated closed-loop report.")
    ap.add_argument("--safe", type=Path, required=True)
    ap.add_argument("--near", type=Path, required=True)
    ap.add_argument("--contact", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--variant", required=True)
    args = ap.parse_args()

    docs = {"safe": _load(args.safe), "near_contact": _load(args.near), "contact": _load(args.contact)}
    report: dict[str, Any] = {
        "version": 1,
        "variant": args.variant,
        "exploratory_only": True,
        "deployment_gate_passed": False,
        "warning": "v48.33 Natural gate did not pass. These paired test results are diagnostic and must not be presented as a deployment certificate.",
        "regimes": {},
    }
    for regime, doc in docs.items():
        metrics = doc.get("metrics") or {}
        report["regimes"][regime] = {
            "num_paired_scenes": doc.get("num_paired_scenes"),
            "metrics": {name: metrics[name] for name in KEYS[regime] if name in metrics},
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        f"# OC-RAP v48.33 ungated full closed-loop summary — {args.variant}",
        "",
        "> **Exploratory only.** The v48.33 Natural deployment gate did not pass. The results below are paired diagnostic evidence, not a deployment certificate.",
        "",
    ]
    for regime in ("safe", "near_contact", "contact"):
        block = report["regimes"][regime]
        lines += [f"## {regime}", "", f"Paired scenes: **{block['num_paired_scenes']}**", "", "| Metric | Control | OC-RAP | Delta | 95% CI | Direction |", "|---|---:|---:|---:|---:|---|"]
        for name, row in block["metrics"].items():
            lo, hi = row.get("bootstrap_95ci", [None, None])
            ci = "n/a" if lo is None else f"[{lo:+.5g}, {hi:+.5g}]"
            lines.append(
                f"| {name} | {row.get('control_mean', float('nan')):.6g} | {row.get('method_mean', float('nan')):.6g} | "
                f"{row.get('paired_delta', float('nan')):+.6g} | {ci} | {row.get('direction', '')} |"
            )
        lines.append("")
    lines += [
        "## Interpretation rule",
        "",
        "Only claim a regime-level gain when the paired effect is physically meaningful, the confidence interval is compatible with that claim, and the improvement is not purchased by new overlap, re-contact, off-road, or severe comfort regressions. Scene-level visualizations must include both best improvements and worst regressions.",
        "",
    ]
    args.output.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "markdown": str(args.output.with_suffix('.md'))}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
