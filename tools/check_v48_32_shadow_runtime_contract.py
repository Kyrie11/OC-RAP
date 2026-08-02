#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from ocrap.utils.regimes import canonical_regime_name


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Fail-closed v48.32 shadow runtime contract audit")
    ap.add_argument("--near", type=Path, required=True)
    ap.add_argument("--contact", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--require-positive-gamma", action="store_true")
    args = ap.parse_args()

    checks: dict[str, dict] = {}
    valid = True
    for expected, path in (("near_contact", args.near), ("contact", args.contact)):
        data = json.loads(path.read_text())
        scenes = list(data.get("scenes", []) or [])
        regimes = [canonical_regime_name(s.get("bucket_name")) for s in scenes]
        gamma_values = [float(s.get("gamma_rec", float("nan"))) for s in scenes]
        gamma_values_json = [v if _finite(v) else None for v in gamma_values]
        regime_ok = bool(scenes) and all(r == expected for r in regimes)
        gamma_ok = bool(gamma_values) and all(_finite(v) for v in gamma_values)
        if args.require_positive_gamma:
            gamma_ok = gamma_ok and all(v > 0.0 for v in gamma_values)
        contact_ok = True
        contact_anchors = []
        if expected == "contact":
            for scene in scenes:
                summary = scene.get("metric_summary", {}) or {}
                anchor = summary.get("contact_anchor_step")
                contact_anchors.append(anchor if _finite(anchor) else None)
                contact_ok = contact_ok and bool(scene.get("post_contact_target", False)) and _finite(anchor)
        else:
            contact_ok = all(not bool(s.get("post_contact_target", False)) for s in scenes)
        item_valid = bool(data.get("metrics_valid", False)) and regime_ok and gamma_ok and contact_ok
        valid = valid and item_valid
        checks[expected] = {
            "path": str(path),
            "num_scenes": len(scenes),
            "regimes": sorted({str(r) for r in regimes}),
            "gamma_values": sorted({v for v in gamma_values_json if v is not None}),
            "num_nonfinite_gamma": sum(v is None for v in gamma_values_json),
            "contact_anchors": contact_anchors,
            "metrics_valid": bool(data.get("metrics_valid", False)),
            "regime_ok": regime_ok,
            "gamma_ok": gamma_ok,
            "contact_semantics_ok": contact_ok,
            "valid": item_valid,
        }

    doc = {"event": "v48_32_shadow_runtime_contract", "valid": valid, "checks": checks}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
    return 0 if valid else 30


if __name__ == "__main__":
    raise SystemExit(main())
