#!/usr/bin/env python3
"""Build a regime-agnostic support contract for continuous safety coordinates.

The contract does not route Safe/Near/Contact.  It only estimates whether each
nominal-relative physical margin has enough variation and boundary crossings to
be learned reliably from the registered training population.  Unsupported
coordinates are shrunk toward the semantic non-harm prior in the neural slack
projection; measured hard vetoes remain independent and fail closed.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ocrap.data.serialization import load_npz_selected

COMPONENTS = ("drs", "deployability", "gap", "hard_rule", "harm_proxy")
SUPPORT_SAMPLE_KEYS = frozenset({"prefix_states", "feasible", "hard_violation"})


def _sigmoid(x: float) -> float:
    x = max(-30.0, min(30.0, float(x)))
    return 1.0 / (1.0 + math.exp(-x))


def _prefix_xy(path: Path) -> np.ndarray | None:
    try:
        d = load_npz_selected(path, SUPPORT_SAMPLE_KEYS)
        return np.asarray(d.get("prefix_states"), dtype=np.float64)[:, :2]
    except Exception:
        return None


def _deviation(candidate: np.ndarray | None, nominal: np.ndarray | None) -> float | None:
    if candidate is None or nominal is None:
        return None
    length = min(len(candidate), len(nominal))
    if length <= 0:
        return None
    return float(
        np.sqrt(np.mean(np.sum((candidate[:length] - nominal[:length]) ** 2, axis=-1)))
        / 5.0
    )


def _terms(
    row: dict[str, Any],
    nominal: dict[str, Any],
    tolerances: tuple[float, float, float, float, float],
) -> list[float]:
    drs_tol, dep_tol, gap_tol, hard_tol, proxy_tol = tolerances
    return [
        float(nominal["teacher_drs"]) - float(row["teacher_drs"]) - drs_tol,
        _sigmoid(float(nominal["teacher_r_dep"]))
        - _sigmoid(float(row["teacher_r_dep"]))
        - dep_tol,
        math.exp(-max(0.0, min(float(nominal["teacher_gap"]), 20.0)))
        - math.exp(-max(0.0, min(float(row["teacher_gap"]), 20.0)))
        - gap_tol,
        float(row.get("teacher_hard_violation", 0.0))
        - float(nominal.get("teacher_hard_violation", 0.0))
        - hard_tol,
        float(row.get("teacher_harm_proxy", 0.0))
        - float(nominal.get("teacher_harm_proxy", 0.0))
        - proxy_tol,
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--env-output", type=Path, required=True)
    ap.add_argument("--macro-ids", default="2,3,5,6,7")
    ap.add_argument("--drs-tolerance", type=float, default=0.05)
    ap.add_argument("--dep-tolerance", type=float, default=0.05)
    ap.add_argument("--gap-tolerance", type=float, default=0.05)
    ap.add_argument("--hard-tolerance", type=float, default=0.05)
    ap.add_argument("--proxy-tolerance", type=float, default=0.05)
    ap.add_argument("--max-hard", type=float, default=1.0)
    ap.add_argument("--min-nominal-deviation", type=float, default=0.002)
    ap.add_argument("--min-positive", type=int, default=40)
    ap.add_argument("--min-unique", type=int, default=3)
    ap.add_argument("--min-std", type=float, default=1.0e-6)
    ap.add_argument("--min-supported-reliability", type=float, default=0.15)
    ap.add_argument("--require-readable-samples", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(line) for line in args.index.read_text().splitlines() if line.strip()]
    groups: dict[tuple[int, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(int(row.get("bucket", 0)), str(row["scene"]), int(row["time"]))].append(row)
    macros = {int(x) for x in args.macro_ids.split(",") if x.strip()}
    values = [[] for _ in COMPONENTS]
    eligible = 0
    fallback_deviation = 0
    skipped = defaultdict(int)

    tolerances = (
        float(args.drs_tolerance), float(args.dep_tolerance),
        float(args.gap_tolerance), float(args.hard_tolerance),
        float(args.proxy_tolerance),
    )

    for group in groups.values():
        nominal = next((r for r in group if bool(r.get("nominal", False))), None)
        if nominal is None:
            skipped["no_nominal"] += 1
            continue
        nominal_path = Path(str(nominal.get("path", "")))
        nominal_prefix = _prefix_xy(nominal_path) if nominal_path.is_file() else None
        if args.require_readable_samples and nominal_prefix is None:
            raise FileNotFoundError(f"nominal sample required by exact support contract is unreadable: {nominal_path}")
        for row in group:
            if bool(row.get("nominal", False)) or int(row.get("macro", -1)) not in macros:
                continue
            path = Path(str(row.get("path", "")))
            feasible = True
            candidate_hard = float(row.get("teacher_hard_violation", 0.0))
            candidate_prefix = None
            if path.is_file():
                try:
                    d = load_npz_selected(path, SUPPORT_SAMPLE_KEYS)
                    feasible = bool(float(np.asarray(d.get("feasible", 1.0)).item()) > 0.5)
                    candidate_hard = float(np.asarray(d.get("hard_violation", candidate_hard)).item())
                    candidate_prefix = np.asarray(d.get("prefix_states"), dtype=np.float64)[:, :2]
                except Exception:
                    skipped["sample_read_error"] += 1
                    if args.require_readable_samples:
                        raise
            elif args.require_readable_samples:
                raise FileNotFoundError(f"candidate sample required by exact support contract is unreadable: {path}")
            if not feasible:
                skipped["infeasible"] += 1
                continue
            if candidate_hard > float(args.max_hard):
                skipped["hard_filter"] += 1
                continue
            dev = _deviation(candidate_prefix, nominal_prefix)
            if dev is None:
                fallback_deviation += 1
            elif dev < float(args.min_nominal_deviation):
                skipped["low_deviation"] += 1
                continue
            for index, value in enumerate(_terms(row, nominal, tolerances)):
                values[index].append(float(value))
            eligible += 1

    component_stats: dict[str, Any] = {}
    reliabilities: list[float] = []
    for name, raw in zip(COMPONENTS, values):
        a = np.asarray(raw, dtype=np.float64)
        if a.size == 0:
            unique = 0
            std = 0.0
            positive = 0
            reliability = 0.0
        else:
            unique = int(np.unique(a).size)
            std = float(np.std(a))
            positive = int(np.sum(a > 0.0))
            if unique < int(args.min_unique) or std < float(args.min_std) or positive <= 0:
                reliability = 0.0
            else:
                reliability = min(1.0, positive / max(1, int(args.min_positive)))
                reliability = max(float(args.min_supported_reliability), reliability)
        reliabilities.append(float(reliability))
        component_stats[name] = {
            "count": int(a.size),
            "unique": unique,
            "std": std,
            "positive_count": positive,
            "positive_fraction": float(positive / a.size) if a.size else 0.0,
            "reliability": float(reliability),
            "min": float(np.min(a)) if a.size else None,
            "median": float(np.median(a)) if a.size else None,
            "max": float(np.max(a)) if a.size else None,
        }

    reliability_csv = ",".join(f"{x:.8g}" for x in reliabilities)
    doc = {
        "version": "v48.32-IDENTITY-UTILITY-BRIDGE",
        "regime_routing": False,
        "semantic": "global support reliability for continuous nominal-relative physical margins",
        "index": str(args.index.resolve()),
        "num_rows": len(rows),
        "num_groups": len(groups),
        "num_eligible_candidates": eligible,
        "deviation_fallback_candidates": fallback_deviation,
        "component_tolerances": dict(zip(COMPONENTS, tolerances)),
        "eligibility": {
            "macro_ids": sorted(macros),
            "max_hard": float(args.max_hard),
            "min_nominal_deviation": float(args.min_nominal_deviation),
        },
        "components": component_stats,
        "component_order": list(COMPONENTS),
        "reliability": reliabilities,
        "reliability_csv": reliability_csv,
        "independent_measured_hard_veto_preserved": True,
        "skipped": dict(skipped),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n")
    args.env_output.write_text(
        "# Generated by build_v48_32_factor_support_contract.py\n"
        f"EVIDENCE_COMPONENT_RELIABILITY={reliability_csv}\n"
    )
    print(json.dumps(doc, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
