#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ocrap.v48_93_factor_mediation import (
    ENGINEERING_VERSION,
    FACTOR_NAMES,
    POSITIVE_GAIN,
    adjudicate_factor_mediation,
)

ROLES = ("dev_near", "dev_contact", "certificate_near", "certificate_contact")
MODES = ("drs_activation", "deployability_gain", "gap_gain", "multi_factor_necessary", "redundant_or_interaction")


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _mean(vals: list[float]) -> float | None:
    x = [float(v) for v in vals if math.isfinite(float(v))]
    return float(np.mean(x)) if x else None


def _fraction(rows: list[dict[str, Any]], pred) -> float | None:
    if not rows:
        return None
    return float(np.mean([bool(pred(r)) for r in rows]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v48-92-audit", type=Path, required=True)
    ap.add_argument("--v48-92-summary", type=Path, required=True)
    ap.add_argument("--v48-92-comparison", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--positive-gain", type=float, default=POSITIVE_GAIN)
    args = ap.parse_args()

    s92 = json.loads(args.v48_92_summary.read_text())
    c92 = json.loads(args.v48_92_comparison.read_text())
    q92 = c92.get("preregistered_decision") or {}
    errors: list[str] = []
    if not (s92.get("valid") and s92.get("attribution_ready")):
        errors.append("invalid V48.92 summary prerequisite")
    if not (c92.get("valid") and c92.get("attribution_ready")):
        errors.append("invalid V48.92 comparison prerequisite")
    if q92.get("status") != "SHARED_RECOVERY_ADVANTAGE_MEDIATOR_GO":
        errors.append("V48.92 shared-mediator screening GO prerequisite missing")
    winners = list(q92.get("shared_mediator_winners") or [])
    if len(winners) < 2:
        errors.append("V48.93 is only authorized for the multi-winner V48.92 tie case")
    if str(s92.get("output_sha256")) != _sha(args.v48_92_audit):
        errors.append("V48.92 audit SHA mismatch")

    rows: list[dict[str, Any]] = []
    max_adv_error = 0.0
    with args.v48_92_audit.open(encoding="utf-8") as f:
        for line in f:
            base = json.loads(line)
            if not base.get("valid"):
                continue
            try:
                rec = adjudicate_factor_mediation(
                    nominal_drs=float(base["nominal_drs"]),
                    candidate_drs=float(base["candidate_drs"]),
                    nominal_deployability_gate=float(base["nominal_deployability_gate"]),
                    candidate_deployability_gate=float(base["candidate_deployability_gate"]),
                    nominal_gap_discount=float(base["nominal_gap_discount"]),
                    candidate_gap_discount=float(base["candidate_gap_discount"]),
                    positive_gain=float(args.positive_gain),
                ).to_dict()
                adv_err = abs(float(rec["full_advantage"]) - float(base["teacher_adv"]))
                max_adv_error = max(max_adv_error, adv_err)
                if adv_err > 2.0e-6:
                    raise ValueError(f"PCD mediation advantage mismatch {adv_err}")
                out = dict(base)
                out.update(
                    schema="ocrap-v48.93-factor-mediation-row-v1",
                    engineering_version=ENGINEERING_VERSION,
                    planner_parameters_trained=0,
                    dataset_reconstruction=False,
                    dataset_reselection=False,
                    teacher_labels_changed=False,
                    teacher_metadata_input_to_model=False,
                    boundary_transport=False,
                    relative_ranker_modified=False,
                    regime_conditioning=False,
                    womd_replay_performed=False,
                    v48_92_audit_reused=True,
                    positive_gain=float(args.positive_gain),
                    **rec,
                )
                rows.append(out)
            except Exception as exc:
                errors.append(f"row mediation failure scene={base.get('scene_id')} time={base.get('time_index')} cand={base.get('candidate_index')}: {exc}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    role_summary: dict[str, Any] = {}
    for role in ROLES:
        rr = [r for r in rows if str(r.get("dataset_role")) == role]
        safe = [r for r in rr if bool(r.get("safe_positive"))]
        harm = [r for r in rr if bool(r.get("teacher_harmful"))]
        factor_stats: dict[str, Any] = {}
        for name in FACTOR_NAMES:
            factor_stats[name] = {
                "safe_necessity_fraction": _fraction(safe, lambda r, n=name: r[f"necessary_{n}"]),
                "safe_sufficiency_fraction": _fraction(safe, lambda r, n=name: r[f"sufficient_{n}"]),
                "harmful_single_factor_false_rescue_fraction": _fraction(
                    harm, lambda r, n=name: float(r[f"single_{n}_advantage"]) >= float(args.positive_gain)
                ),
                "safe_knockout_advantage_mean": _mean([r[f"knockout_{name}_advantage"] for r in safe]),
                "safe_single_advantage_mean": _mean([r[f"single_{name}_advantage"] for r in safe]),
            }
        mode_counts = {m: sum(str(r.get("mediation_mode")) == m for r in safe) for m in MODES}
        explained = mode_counts["drs_activation"] + mode_counts["deployability_gain"]
        role_summary[role] = {
            "rows": len(rr),
            "safe_positive_rows": len(safe),
            "harmful_rows": len(harm),
            "factor_stats": factor_stats,
            "safe_mode_counts": mode_counts,
            "drs_or_deployability_necessity_coverage": float(explained / len(safe)) if safe else None,
            "gap_necessity_fraction": factor_stats["gap_discount"]["safe_necessity_fraction"],
            "multi_or_unexplained_fraction": float(
                (mode_counts["multi_factor_necessary"] + mode_counts["redundant_or_interaction"]) / len(safe)
            ) if safe else None,
            "nominal_drs_active_safe_fraction": _fraction(safe, lambda r: float(r["nominal_drs"]) >= 0.5),
            "structural_response_safe_mean": _mean([r["structural_response_score"] for r in safe]),
            "structural_response_harmful_mean": _mean([r["structural_response_score"] for r in harm]),
        }

    summary = {
        "schema": "ocrap-v48.93-factor-mediation-summary-v1",
        "engineering_version": ENGINEERING_VERSION,
        "valid": not errors,
        "attribution_ready": not errors,
        "errors": errors[:100],
        "experiment_type": "audit_only_exact_pcd_factor_mediation_complementarity",
        "planner_parameters_trained": 0,
        "dataset_reconstruction": False,
        "dataset_reselection": False,
        "teacher_labels_changed": False,
        "teacher_metadata_input_to_model": False,
        "boundary_transport": False,
        "relative_ranker_modified": False,
        "regime_conditioning": False,
        "womd_replay_performed": False,
        "same_v48_92_cohort": True,
        "positive_gain": float(args.positive_gain),
        "v48_92_multiwinner_screening_set": winners,
        "rows": len(rows),
        "max_advantage_identity_error": max_adv_error,
        "roles": role_summary,
        "output": str(args.output.resolve()),
        "output_sha256": _sha(args.output),
        "test_roots_read": False,
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"valid": summary["valid"], "rows": len(rows), "max_advantage_identity_error": max_adv_error}))
    return 0 if summary["valid"] else 30


if __name__ == "__main__":
    raise SystemExit(main())
