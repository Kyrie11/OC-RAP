#!/usr/bin/env python3
"""Compute regime-free train-only linear canonical scales for v48.55 TCBC.

The output intentionally uses a single pooled Near+Contact estimate.  Bucket/
regime identity is reported only for diagnostics and never enters the scale
calculation.  The transform is linear and sign preserving; no threshold or
hard-veto semantics are changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _component_margins(row: dict[str, Any], nominal: dict[str, Any], args: argparse.Namespace) -> tuple[float, ...]:
    drs = float(nominal["teacher_drs"]) - float(row["teacher_drs"]) - float(args.drs_tolerance)
    dep = (
        _sigmoid(float(nominal["teacher_r_dep"]))
        - _sigmoid(float(row["teacher_r_dep"]))
        - float(args.dep_tolerance)
    )
    nom_gap = max(0.0, min(float(nominal["teacher_gap"]), 20.0))
    row_gap = max(0.0, min(float(row["teacher_gap"]), 20.0))
    gap = math.exp(-nom_gap) - math.exp(-row_gap) - float(args.gap_tolerance)
    hard = float(row.get("teacher_hard_violation", 0.0)) - float(nominal.get("teacher_hard_violation", 0.0)) - float(args.hard_tolerance)
    proxy = float(row.get("teacher_harm_proxy", 0.0)) - float(nominal.get("teacher_harm_proxy", 0.0)) - float(args.proxy_tolerance)
    values = (drs, dep, gap, hard, proxy)
    if not all(math.isfinite(v) for v in values):
        raise ValueError(f"non-finite component margin for row={row.get('path')!r}: {values!r}")
    return values


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--target-scale", type=float, default=0.10)
    ap.add_argument("--drs-tolerance", type=float, default=0.05)
    ap.add_argument("--dep-tolerance", type=float, default=0.05)
    ap.add_argument("--gap-tolerance", type=float, default=0.05)
    ap.add_argument("--hard-tolerance", type=float, default=0.05)
    ap.add_argument("--proxy-tolerance", type=float, default=0.05)
    args = ap.parse_args()

    if not args.index.is_file():
        raise SystemExit(f"missing teacher index: {args.index}")
    if not math.isfinite(args.target_scale) or args.target_scale <= 0.0:
        raise SystemExit("target-scale must be finite and > 0")

    rows: list[dict[str, Any]] = []
    nominal: dict[tuple[int, str, int], dict[str, Any]] = {}
    bucket_rows: Counter[int] = Counter()
    with args.index.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                raise SystemExit(f"invalid JSON at {args.index}:{lineno}: {exc}")
            key = (int(row["bucket"]), str(row["scene"]), int(row["time"]))
            rows.append(row)
            if bool(row.get("nominal", False)):
                prev = nominal.get(key)
                if prev is not None:
                    raise SystemExit(f"duplicate nominal row for group {key}")
                nominal[key] = row

    sum_sq = [0.0] * 5
    count = 0
    positive = [0] * 5
    negative = [0] * 5
    bucket_candidate_counts: Counter[int] = Counter()
    for row in rows:
        if bool(row.get("nominal", False)):
            continue
        key = (int(row["bucket"]), str(row["scene"]), int(row["time"]))
        nom = nominal.get(key)
        if nom is None:
            raise SystemExit(f"missing nominal row for candidate group {key}")
        margins = _component_margins(row, nom, args)
        count += 1
        bucket_candidate_counts[int(row["bucket"])] += 1
        for i, value in enumerate(margins):
            sum_sq[i] += value * value
            positive[i] += int(value > 0.0)
            negative[i] += int(value < 0.0)

    if count <= 0:
        raise SystemExit("teacher index has no non-nominal candidates")
    rms = [math.sqrt(x / count) for x in sum_sq]
    if any((not math.isfinite(x) or x <= 1.0e-12) for x in rms[:3]):
        raise SystemExit(f"invalid pooled RMS for supported components: {rms[:3]!r}")

    # DRS remains raw under the Y-axis canonicalization so that C isolates only
    # continuous DEP/GAP scaling.  Since the runtime transform is
    # target_scale * margin / scale, scale=target_scale is the identity map.
    canonical_scales = [float(args.target_scale), float(rms[1]), float(rms[2]), float(args.target_scale), float(args.target_scale)]
    names = ["drs", "deployability", "gap_quality", "hard_rule", "harm_proxy"]
    payload = {
        "event": "v48_55_component_boundary_scales",
        "version": "v48.55-DCP-DRFC-BCDE-TCBC",
        "scale_source": "pooled_train_near_contact_rms",
        "linear_transform": "target_scale_times_raw_margin_divided_by_component_scale",
        "zero_crossing_preserved": True,
        "within_component_order_preserved": True,
        "saturating_transform": False,
        "strategy_regime_conditioning": False,
        "test_roots_read": False,
        "teacher_index": str(args.index.resolve()),
        "teacher_index_sha256": _sha256(args.index),
        "non_nominal_candidates": count,
        "bucket_candidate_counts_diagnostic_only": {str(k): int(v) for k, v in sorted(bucket_candidate_counts.items())},
        "target_scale": float(args.target_scale),
        "component_names": names,
        "pooled_rms": {name: float(rms[i]) for i, name in enumerate(names)},
        "positive_margin_fraction": {name: float(positive[i] / count) for i, name in enumerate(names)},
        "negative_margin_fraction": {name: float(negative[i] / count) for i, name in enumerate(names)},
        "canonical_scales": canonical_scales,
        "canonical_scales_csv": ",".join(f"{x:.12g}" for x in canonical_scales),
        "drs_transform_under_canonicalization": "identity_raw_margin",
        "continuous_components_canonicalized": ["deployability", "gap_quality"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(args.output)
    print(json.dumps({"event": payload["event"], "output": str(args.output), "canonical_scales_csv": payload["canonical_scales_csv"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
