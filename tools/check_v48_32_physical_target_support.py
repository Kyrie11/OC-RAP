#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def _mean(metrics: dict, name: str) -> float | None:
    value = (metrics.get(name, {}) or {}).get("control_mean")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit whether dev-shadow targets can identify the requested physics effects")
    ap.add_argument("--near", type=Path, required=True)
    ap.add_argument("--contact", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    near = _load(args.near)
    contact = _load(args.contact)
    nm = near.get("metrics", {}) or {}
    cm = contact.get("metrics", {}) or {}
    near_exposure = _mean(nm, "near_contact_exposure_rate")
    critical_exposure = _mean(nm, "critical_ttc_exposure_rate")
    secondary = _mean(cm, "secondary_overlap_event")
    escape = _mean(cm, "post_contact_escape_event")
    stable = _mean(cm, "new_stable_stop_quality_event")
    overlap = _mean(cm, "overlap_duration_s")

    checks = {
        "paired_near_scenes": int(near.get("num_paired_scenes", 0)) > 0,
        "paired_contact_scenes": int(contact.get("num_paired_scenes", 0)) > 0,
        "near_has_hazard_exposure": bool(
            near_exposure is not None and critical_exposure is not None
            and max(near_exposure, critical_exposure) > 0.0
        ),
        "contact_recontact_not_floor_saturated": bool(secondary is not None and secondary > 0.0),
        "contact_escape_not_ceiling_saturated": bool(escape is not None and 0.0 < escape < 1.0),
        "contact_stable_stop_not_floor_saturated": bool(stable is not None and stable > 0.0),
        "contact_overlap_not_floor_saturated": bool(overlap is not None and overlap > 0.0),
    }
    contact_challenge_informative = any(
        checks[key]
        for key in (
            "contact_recontact_not_floor_saturated",
            "contact_escape_not_ceiling_saturated",
            "contact_stable_stop_not_floor_saturated",
            "contact_overlap_not_floor_saturated",
        )
    )
    doc = {
        "event": "v48_32_physical_target_support_audit",
        "near_paired_scenes": int(near.get("num_paired_scenes", 0)),
        "contact_paired_scenes": int(contact.get("num_paired_scenes", 0)),
        "checks": checks,
        "near_physics_informative": checks["paired_near_scenes"] and checks["near_has_hazard_exposure"],
        "contact_challenge_informative": checks["paired_contact_scenes"] and contact_challenge_informative,
        "warning": None if contact_challenge_informative else (
            "Contact sample is floor/ceiling saturated for overlap, re-contact, escape and stable-stop; "
            "clearance/free-space deltas remain valid, but absence of event improvement is not evidence of success."
        ),
        "control_means": {
            "near_contact_exposure_rate": near_exposure,
            "critical_ttc_exposure_rate": critical_exposure,
            "secondary_overlap_event": secondary,
            "post_contact_escape_event": escape,
            "new_stable_stop_quality_event": stable,
            "overlap_duration_s": overlap,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(doc, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
