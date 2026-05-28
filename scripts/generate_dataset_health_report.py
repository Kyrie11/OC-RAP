#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ocrap.teacher.dataset_writer import read_dataset, ShardedArray


def _iter_rows(arr):
    if isinstance(arr, ShardedArray):
        for i in range(len(arr)):
            yield arr[i]
    else:
        a = np.asarray(arr)
        for i in range(a.shape[0]):
            yield a[i]


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _percentiles(vals):
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"mean": None, "p05": None, "p50": None, "p95": None}
    return {
        "mean": float(vals.mean()),
        "p05": float(np.percentile(vals, 5)),
        "p50": float(np.percentile(vals, 50)),
        "p95": float(np.percentile(vals, 95)),
    }


def build_report(dataset: str | Path) -> dict:
    arrays, meta = read_dataset(dataset)
    n = int(next(iter(arrays.values())).shape[0]) if arrays else 0
    report: dict[str, Any] = {
        "dataset": str(dataset),
        "num_samples": n,
        "metadata": {k: meta.get(k) for k in ["dataset_version", "split", "root_backend", "rollout_backend", "is_synthetic", "paper_final_ready", "K", "L", "M", "H_p", "H_r", "dt"] if k in meta},
        "arrays": {k: {"shape": list(v.shape), "dtype": str(getattr(v, "dtype", "unknown"))} for k, v in arrays.items()},
        "warnings": [],
    }
    if n == 0:
        report["warnings"].append("empty dataset")
        return report

    if "regime" in arrays:
        report["regime_counts"] = dict(Counter(str(x) for x in np.asarray(arrays["regime"]).astype(str)))
    if "root_ids" in arrays:
        roots = [str(x) for x in np.asarray(arrays["root_ids"]).astype(str)]
        dup = len(roots) - len(set(roots))
        report["root_id_duplicates"] = int(dup)
        if dup:
            report["warnings"].append(f"{dup} duplicated root_ids")

    def mean_valid(name: str) -> float | None:
        if name not in arrays:
            return None
        vals = []
        for row in _iter_rows(arrays[name]):
            vals.append(float(np.asarray(row).astype(bool).mean()))
        return float(np.mean(vals)) if vals else None

    report["validity"] = {
        "action_valid_fraction": mean_valid("action_mask"),
        "option_valid_fraction": mean_valid("option_mask"),
    }

    for k in ["bev", "ego_info", "route_command", "actions_states", "token_states_ref", "token_anchor", "token_hard_shell"]:
        if k not in arrays:
            report["warnings"].append(f"missing deployable input field: {k}")

    finite_checks = {}
    for k in ["actions_states", "token_states_ref", "token_anchor", "token_hard_shell", "g_star", "h_star", "k_star", "u_star", "c_rule_star", "R_star"]:
        if k in arrays:
            bad = 0
            total = 0
            for row in _iter_rows(arrays[k]):
                a = np.asarray(row)
                total += int(a.size)
                bad += int((~np.isfinite(a)).sum()) if np.issubdtype(a.dtype, np.number) else 0
            finite_checks[k] = {"nonfinite": int(bad), "total": int(total)}
            if bad:
                report["warnings"].append(f"{k} contains {bad} non-finite values")
    report["finite_checks"] = finite_checks

    if "R_star" in arrays:
        R_vals = []
        for row in _iter_rows(arrays["R_star"]):
            R_vals.extend(np.asarray(row, dtype=np.float32).reshape(-1).tolist())
        report["R_star"] = _percentiles(R_vals)

    if "Y_oc" in arrays:
        y = []
        for row in _iter_rows(arrays["Y_oc"]):
            y.extend(np.asarray(row, dtype=np.float32).reshape(-1).tolist())
        report["Y_oc_success_rate"] = _safe_float(np.mean(y)) if y else None
    if "Y_action" in arrays and "Y_oc" in arrays:
        gaps = []
        for ya, yo in zip(_iter_rows(arrays["Y_action"]), _iter_rows(arrays["Y_oc"])):
            gaps.extend((np.asarray(ya, dtype=np.float32) - np.asarray(yo, dtype=np.float32)).reshape(-1).tolist())
        report["oracle_minus_oc_success_gap"] = _percentiles(gaps)
        if gaps and float(np.mean(gaps)) > 0.10:
            report["warnings"].append("large oracle-vs-OC success gap; observation consistency is materially changing labels")

    if "obs_equiv" in arrays:
        class_sizes = []
        for eq_row in _iter_rows(arrays["obs_equiv"]):
            eq_arr = np.asarray(eq_row).astype(bool)
            matrices = eq_arr.reshape((-1,) + eq_arr.shape[-2:]) if eq_arr.ndim >= 2 else []
            for eq in matrices:
                seen = set()
                for i in range(eq.shape[0]):
                    cls = tuple(np.where(eq[i])[0].tolist())
                    if cls not in seen:
                        seen.add(cls)
                        class_sizes.append(len(cls))
        report["obs_equiv_class_size"] = _percentiles(class_sizes)
        if class_sizes and max(class_sizes) == min(class_sizes) == int(meta.get("M", max(class_sizes))):
            report["warnings"].append("all modes appear observation-equivalent; check post-prefix observation signature richness")

    if "c_rule_star" in arrays:
        vals = []
        for row in _iter_rows(arrays["c_rule_star"]):
            vals.extend(np.asarray(row, dtype=np.float32).reshape(-1).tolist())
        report["c_rule_star"] = _percentiles(vals)

    if report["validity"].get("action_valid_fraction") is not None and report["validity"]["action_valid_fraction"] < 0.25:
        report["warnings"].append("low valid action fraction; proposal/projector may be over-pruning")
    if report["validity"].get("option_valid_fraction") is not None and report["validity"]["option_valid_fraction"] < 0.25:
        report["warnings"].append("low valid recovery-token fraction; affordance generator may be over-pruning")
    return report


def write_markdown(report: dict, path: Path) -> None:
    lines = ["# OC-RAP dataset health report", "", f"Dataset: `{report['dataset']}`", f"Samples: **{report['num_samples']}**", ""]
    if report.get("metadata"):
        lines += ["## Metadata", ""]
        for k, v in report["metadata"].items():
            lines.append(f"- `{k}`: `{v}`")
        lines.append("")
    if report.get("regime_counts"):
        lines += ["## Regime counts", ""]
        for k, v in report["regime_counts"].items():
            lines.append(f"- {k}: {v}")
        lines.append("")
    lines += ["## Validity", ""]
    for k, v in report.get("validity", {}).items():
        lines.append(f"- {k}: {v}")
    for section in ["R_star", "oracle_minus_oc_success_gap", "obs_equiv_class_size", "c_rule_star"]:
        if section in report:
            lines += ["", f"## {section}", ""]
            for k, v in report[section].items():
                lines.append(f"- {k}: {v}")
    if report.get("warnings"):
        lines += ["", "## Warnings", ""]
        for w in report["warnings"]:
            lines.append(f"- {w}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output", required=True, help="Output directory or .json path")
    args = ap.parse_args()
    report = build_report(args.dataset)
    out = Path(args.output)
    if out.suffix.lower() == ".json":
        out.parent.mkdir(parents=True, exist_ok=True)
        json_path = out
        md_path = out.with_suffix(".md")
    else:
        out.mkdir(parents=True, exist_ok=True)
        json_path = out / "health_report.json"
        md_path = out / "health_report.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, md_path)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "warnings": report.get("warnings", [])}, indent=2))


if __name__ == "__main__":
    main()
