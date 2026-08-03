#!/usr/bin/env python3
"""Combine paired reports into display-ready per-regime progress tables."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

PRIMARY = {
    "safe": ["closed_loop_bounded_NUP", "intervention_rate", "overlap_any", "offroad_any", "jerk_p95", "yaw_rate_p95", "route_progression_m"],
    "near": ["closed_loop_bounded_NUP", "intervention_rate", "ttc_s_p05", "min_clearance_m_p05", "critical_ttc_exposure_duration_s", "near_zero_clearance_exposure_rate", "overlap_any", "offroad_any"],
    "contact": ["closed_loop_bounded_NUP", "intervention_rate", "post_contact_terminal_clearance_m", "post_contact_free_space_auc_normalized_m", "post_contact_escape_event", "recontact_event", "stable_stop_quality_event", "offroad_any"],
}


def _fmt(value: Any) -> str:
    if value is None or value == "":
        return "—"
    try:
        v = float(value)
    except Exception:
        return str(value)
    return f"{v:.4f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--variants", default="balanced,precision")
    ap.add_argument("--enabled-regimes", default="safe,near,contact")
    args = ap.parse_args()
    variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    regimes = [x.strip() for x in args.enabled_regimes.split(",") if x.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index: dict[str, Any] = {
        "event": "v48_34_1_progress_table_index",
        "exploratory_only": True,
        "paper_claim_allowed": False,
        "regimes": {},
    }
    overall_valid = True

    for regime in regimes:
        method_metrics: dict[str, dict[str, Any]] = {}
        reference_metrics: dict[str, dict[str, Any]] = {}
        sources: list[str] = []
        scene_counts: set[int] = set()
        missing_reports: list[str] = []
        for variant in variants:
            p = args.reports_dir / f"{variant}_{regime}_paired.json"
            if not p.is_file():
                missing_reports.append(str(p))
                continue
            doc = json.loads(p.read_text(encoding="utf-8"))
            sources.append(str(p))
            scene_counts.add(int(doc.get("num_scenes", 0)))
            for method in doc.get("methods", []):
                name = str(method.get("method"))
                if name.startswith("ocrap_") and name != f"ocrap_{variant}":
                    continue
                metrics = method.get("metrics", {}) or {}
                for metric, rep in metrics.items():
                    ref_record = {
                        "n": rep.get("n"),
                        "method_mean": rep.get("reference_mean"),
                        "raw_delta_mean": 0.0,
                        "oriented_delta_mean": 0.0,
                    }
                    if metric in reference_metrics:
                        previous_ref = reference_metrics[metric]
                        for field in ("n", "method_mean"):
                            av, bv = previous_ref.get(field), ref_record.get(field)
                            if av is None and bv is None:
                                continue
                            try:
                                equal = abs(float(av) - float(bv)) <= 1.0e-10
                            except Exception:
                                equal = av == bv
                            if not equal:
                                raise SystemExit(f"inconsistent scalar reference {metric}/{field} for {regime}: {av} vs {bv}")
                    else:
                        reference_metrics[metric] = ref_record
                if name in method_metrics:
                    # External baselines are repeated in each variant report.
                    # Bootstrap CIs may differ if earlier method metric coverage
                    # consumes a different RNG stream, so compare only the
                    # deterministic absolute and paired means before deduplication.
                    previous = method_metrics[name]
                    for metric in set(previous) | set(metrics):
                        a = previous.get(metric) or {}; b = metrics.get(metric) or {}
                        for field in ("n", "reference_mean", "method_mean", "raw_delta_mean", "oriented_delta_mean"):
                            av, bv = a.get(field), b.get(field)
                            if av is None and bv is None:
                                continue
                            try:
                                equal = abs(float(av) - float(bv)) <= 1.0e-10
                            except Exception:
                                equal = av == bv
                            if not equal:
                                raise SystemExit(f"inconsistent duplicate method {name}/{metric}/{field} for {regime}: {av} vs {bv}")
                else:
                    method_metrics[name] = metrics
        required_ocrap = {f"ocrap_{v}" for v in variants}
        missing_methods = sorted(required_ocrap - set(method_metrics))
        if reference_metrics:
            method_metrics = {"scalar_control": reference_metrics, **method_metrics}
        regime_valid = not missing_reports and not missing_methods and len(scene_counts) == 1 and bool(method_metrics) and bool(reference_metrics)
        overall_valid &= regime_valid

        def method_order(name: str) -> tuple[int, str]:
            if name == "scalar_control":
                return (0, name)
            if name.startswith("ocrap_"):
                return (1, name)
            return (2, name)

        rows: list[dict[str, Any]] = []
        for method in sorted(method_metrics, key=method_order):
            row: dict[str, Any] = {"regime": regime, "method": method, "num_scenes": next(iter(scene_counts), 0)}
            for metric in PRIMARY[regime]:
                rep = method_metrics[method].get(metric) or {}
                row[f"{metric}__mean"] = rep.get("method_mean")
                row[f"{metric}__delta_vs_scalar"] = rep.get("raw_delta_mean")
                row[f"{metric}__oriented_delta"] = rep.get("oriented_delta_mean")
                row[f"{metric}__n"] = rep.get("n")
            rows.append(row)

        fields = ["regime", "method", "num_scenes"]
        for metric in PRIMARY[regime]:
            fields.extend([f"{metric}__mean", f"{metric}__delta_vs_scalar", f"{metric}__oriented_delta", f"{metric}__n"])
        csv_path = args.output_dir / f"{regime}_progress_comparison.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(rows)

        all_metrics = list(dict.fromkeys(PRIMARY[regime] + sorted({metric for metrics in method_metrics.values() for metric in metrics})))
        full_fields = ["regime", "method", "num_scenes"]
        for metric in all_metrics:
            full_fields.extend([f"{metric}__mean", f"{metric}__delta_vs_scalar", f"{metric}__oriented_delta", f"{metric}__n"])
        full_csv_path = args.output_dir / f"{regime}_progress_comparison_full.csv"
        with full_csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=full_fields)
            w.writeheader()
            for method in sorted(method_metrics, key=method_order):
                row: dict[str, Any] = {"regime": regime, "method": method, "num_scenes": next(iter(scene_counts), 0)}
                for metric in all_metrics:
                    rep = method_metrics[method].get(metric) or {}
                    row[f"{metric}__mean"] = rep.get("method_mean")
                    row[f"{metric}__delta_vs_scalar"] = rep.get("raw_delta_mean")
                    row[f"{metric}__oriented_delta"] = rep.get("oriented_delta_mean")
                    row[f"{metric}__n"] = rep.get("n")
                w.writerow(row)

        md_path = args.output_dir / f"{regime}_progress_comparison.md"
        headers = ["Method"] + [m for m in PRIMARY[regime]]
        lines = [f"# {regime} exploratory paired comparison", "", "All values are paired on the exact same target set. Parentheses show raw delta versus scalar control.", "", "| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
        for row in rows:
            cells = [str(row["method"])]
            for metric in PRIMARY[regime]:
                mean = _fmt(row.get(f"{metric}__mean")); delta = _fmt(row.get(f"{metric}__delta_vs_scalar"))
                cells.append(f"{mean} ({delta})")
            lines.append("| " + " | ".join(cells) + " |")
        lines.extend(["", "> Exploratory progress display only; not deployment- or paper-authorized when the Natural gate has not passed."])
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        index["regimes"][regime] = {
            "valid": regime_valid,
            "num_scenes": next(iter(scene_counts), 0),
            "methods": sorted(method_metrics),
            "missing_reports": missing_reports,
            "missing_methods": missing_methods,
            "source_reports": sources,
            "csv": str(csv_path),
            "full_csv": str(full_csv_path),
            "markdown": str(md_path),
            "primary_metrics": PRIMARY[regime],
            "all_metrics": all_metrics,
        }

    index["valid"] = overall_valid
    out = args.output_dir / "ALL_REGIMES_REPORT_INDEX.json"
    out.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": index["event"], "valid": overall_valid, "output": str(out)}))
    return 0 if overall_valid else 4


if __name__ == "__main__":
    raise SystemExit(main())
