#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ocrap.v48_93_factor_mediation import ENGINEERING_VERSION


def sha(p: Path) -> str:
    h = hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    for n in ("runtime", "audit", "audit_summary", "comparison", "v48_92_pipeline", "v48_92_comparison"):
        ap.add_argument("--" + n.replace("_", "-"), type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(); errors: list[str] = []
    docs = {k: json.loads(getattr(args, k).read_text()) for k in ("runtime", "audit_summary", "comparison", "v48_92_pipeline", "v48_92_comparison")}
    for k in ("runtime", "audit_summary", "comparison"):
        if not (docs[k].get("valid") and docs[k].get("attribution_ready")): errors.append(f"{k} invalid")
        if str(docs[k].get("engineering_version")) != ENGINEERING_VERSION: errors.append(f"{k} version mismatch")
    if not (docs["v48_92_pipeline"].get("valid") and docs["v48_92_pipeline"].get("attribution_ready")): errors.append("V48.92 pipeline invalid")
    q92 = docs["v48_92_comparison"].get("preregistered_decision") or {}
    if q92.get("status") != "SHARED_RECOVERY_ADVANTAGE_MEDIATOR_GO": errors.append("V48.92 multi-mediator screening prerequisite missing")
    if len(q92.get("shared_mediator_winners") or []) < 2: errors.append("V48.92 did not produce a multi-winner tie")
    if str(docs["audit_summary"].get("output_sha256")) != sha(args.audit): errors.append("audit SHA mismatch")
    d = docs["comparison"].get("preregistered_decision") or {}
    allowed = {"SINGLE_PCD_MEDIATOR_GO", "PCD_FACTOR_COMPLEMENTARITY_GO", "PCD_FACTOR_MEDIATION_STOP"}
    if d.get("status") not in allowed: errors.append("unexpected V48.93 scientific status")
    out = {
        "schema": "ocrap-v48.93-pipeline-complete-v1",
        "engineering_version": ENGINEERING_VERSION,
        "valid": not errors,
        "attribution_ready": not errors,
        "errors": errors,
        "experiment_type": "audit_only_exact_pcd_factor_mediation_complementarity",
        "planner_parameters_trained": 0,
        "dataset_reconstruction": False,
        "dataset_reselection": False,
        "teacher_labels_changed": False,
        "teacher_metadata_input_to_model": False,
        "regime_conditioning": False,
        "boundary_transport": False,
        "relative_ranker_modified": False,
        "womd_replay_performed": False,
        "test_roots_read": False,
        "preregistered_status": d.get("status"),
        "artifacts": {k: {"path": str(getattr(args, k).resolve()), "sha256": sha(getattr(args, k))} for k in ("runtime", "audit", "audit_summary", "comparison")},
    }
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"valid": out["valid"], "attribution_ready": out["attribution_ready"], "status": out["preregistered_status"]}))
    return 0 if out["valid"] else 30


if __name__ == "__main__":
    raise SystemExit(main())
