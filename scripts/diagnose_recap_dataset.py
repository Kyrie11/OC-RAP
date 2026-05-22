#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

# Allow running scripts directly from a source checkout without pip install -e .
import sys as _sys
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from recap.teacher.dataset_writer import read_dataset


def _stats(x):
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0}
    return {"count": int(arr.size), "mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "p05": float(np.quantile(arr, 0.05)), "p50": float(np.quantile(arr, 0.50)), "p95": float(np.quantile(arr, 0.95)), "max": float(arr.max())}


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnostics for ReCAP/MetaDrive-Recovery label datasets.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--roots", default=None)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    arrays, meta = read_dataset(args.dataset)
    root_ids = [str(x) for x in arrays.get("root_ids", [])]
    regimes = [str(x) for x in arrays.get("regime", [])]
    report = {
        "dataset": args.dataset,
        "metadata": meta,
        "num_roots": len(root_ids),
        "root_id_sample": root_ids[:10],
        "regime_counts": dict(Counter(regimes)),
        "action_valid_ratio": float(np.mean(arrays["action_mask"])) if "action_mask" in arrays else None,
        "option_valid_ratio": float(np.mean(arrays["option_mask"])) if "option_mask" in arrays else None,
        "R_star": _stats(arrays["R_star"]) if "R_star" in arrays else None,
        "Y_action_rate": _stats(arrays["Y_action"].astype(np.float32)) if "Y_action" in arrays else None,
        "H_action_star": _stats(arrays["H_action_star"]) if "H_action_star" in arrays else None,
        "margin_option": _stats(arrays["margin_option"]) if "margin_option" in arrays else None,
        "witness_gap": _stats(arrays["witness_gap"]) if "witness_gap" in arrays else None,
        "synthetic_guard": {
            "is_synthetic": bool(meta.get("is_synthetic", True)),
            "paper_final_ready": bool(meta.get("paper_final_ready", False)),
            "rollout_backend": meta.get("rollout_backend"),
            "root_backend": meta.get("root_backend"),
        },
    }
    if args.roots:
        root_dir = Path(args.roots)
        missing = [rid for rid in root_ids if not (root_dir / f"{rid}.json").exists()]
        scenario_backed = 0
        for rid in root_ids[: min(len(root_ids), 1000)]:
            p = root_dir / f"{rid}.json"
            if p.exists():
                obj = json.loads(p.read_text())
                if obj.get("scenario_data", {}).get("scenario_pkl"):
                    scenario_backed += 1
        report["roots_check"] = {"missing_root_json": missing[:20], "num_missing": len(missing), "scenario_backed_in_sample": scenario_backed}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
