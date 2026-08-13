#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


def _pct(xs: list[float], q: float) -> float | None:
    vals = sorted(float(x) for x in xs if math.isfinite(float(x)))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * float(q)
    lo = int(math.floor(pos)); hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    w = pos - lo
    return vals[lo] * (1.0 - w) + vals[hi] * w


def summarize(path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if path.is_file():
        for line in path.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and "gpu" in row:
                rows.append(row)
    by_gpu: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        try:
            by_gpu[int(row["gpu"])].append(row)
        except Exception:
            continue
    out: dict[str, Any] = {"telemetry_file": str(path), "samples": len(rows), "gpus": {}}
    for gpu, grp in sorted(by_gpu.items()):
        util = [float(r.get("gpu_util_pct", 0.0)) for r in grp]
        mem_used = [float(r.get("gpu_mem_used_mb", 0.0)) for r in grp]
        mem_total = [float(r.get("gpu_mem_total_mb", 0.0)) for r in grp]
        power = [float(r["power_w"]) for r in grp if r.get("power_w") is not None]
        total_ref = max(mem_total) if mem_total else 0.0
        out["gpus"][str(gpu)] = {
            "samples": len(grp),
            "gpu_util_mean_pct": mean(util) if util else None,
            "gpu_util_median_pct": median(util) if util else None,
            "gpu_util_p10_pct": _pct(util, 0.10),
            "gpu_util_p90_pct": _pct(util, 0.90),
            "gpu_util_lt30_fraction": (sum(x < 30.0 for x in util) / len(util)) if util else None,
            "gpu_util_ge80_fraction": (sum(x >= 80.0 for x in util) / len(util)) if util else None,
            "gpu_mem_used_max_mb": max(mem_used) if mem_used else None,
            "gpu_mem_total_mb": total_ref or None,
            "gpu_mem_peak_fraction": (max(mem_used) / total_ref) if mem_used and total_ref > 0 else None,
            "power_mean_w": mean(power) if power else None,
        }
    mem_avail = [int(r["mem_available_kb"]) for r in rows if r.get("mem_available_kb") is not None]
    out["host"] = {"mem_available_min_kb": min(mem_avail) if mem_avail else None}
    # This is deliberately descriptive, not an auto-tuner: changing process
    # concurrency based on a partial run would alter execution conditions across arms.
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    out = summarize(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
