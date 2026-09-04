#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _js(path: Path):
    return json.loads(path.read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", type=Path, required=True)
    ap.add_argument("--audit-index", type=Path, required=True)
    ap.add_argument("--audit-summary", type=Path, required=True)
    ap.add_argument("--comparison", type=Path, required=True)
    ap.add_argument("--v48-88-comparison", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    errors: list[str] = []
    artifacts = {}
    for p in [args.runtime, args.audit_index, args.audit_summary, args.comparison, args.v48_88_comparison]:
        if not p.is_file():
            errors.append(f"missing {p}")
            continue
        artifacts[str(p)] = _sha(p)
    for p, name in [(args.runtime, "runtime"), (args.audit_summary, "audit_summary"), (args.comparison, "comparison")]:
        if p.is_file() and not _js(p).get("valid"):
            errors.append(f"{name} invalid")
    if args.audit_summary.is_file() and args.audit_index.is_file():
        summary = _js(args.audit_summary)
        if summary.get("output_sha256") != _sha(args.audit_index):
            errors.append("audit index sha mismatch")
        if summary.get("planner_parameters_trained") != 0:
            errors.append("V48.89 must be audit-only with zero planner parameters")
    if args.v48_88_comparison.is_file():
        prev = _js(args.v48_88_comparison)
        pd = prev.get("preregistered_decision") or {}
        if pd.get("status") != "QUOTIENT_TAIL_RESPONSE_STOP":
            errors.append("V48.88 STOP prerequisite missing")
    doc = {
        "schema": "ocrap-v48.89-rcpi-pipeline-complete-v1",
        "engineering_version": "v48.89.0-OC-RCPI",
        "valid": not errors,
        "attribution_ready": not errors,
        "errors": errors,
        "artifacts": artifacts,
        "experiment_type": "audit_only_identifiability_adjudication",
        "planner_parameters_trained": 0,
        "teacher_labels_changed": False,
        "teacher_metadata_input_to_model": False,
        "dataset_reconstruction": False,
        "regime_conditioning": False,
        "boundary_transport": False,
        "relative_ranker_modified": False,
        "test_roots_read": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"valid": doc["valid"], "errors": errors}))
    return 0 if doc["valid"] else 30


if __name__ == "__main__":
    raise SystemExit(main())
