#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

REQUIRED = [
    "OC-RAP-v48.124.10.2-observation-legal-near-adjudication.json",
    "OC-RAP-v48.124.10.2-result-bundle-manifest.json",
    "base/balanced/closed_loop_ocrap.json",
    "base/precision/closed_loop_ocrap.json",
    "cohort/balanced_keys.json",
    "cohort/precision_keys.json",
    "provenance/frozen_checkpoint_contract.json",
    "provenance/observation_legal_runtime_contract.json",
    "provenance/runtime_source_snapshot_manifest.json",
    "provenance/all_route_audit.json",
    "support/near_dataset_support.json",
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-zip", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    src = Path(args.results_zip).resolve()
    out = Path(args.output_dir).resolve()
    if not src.is_file():
        raise SystemExit(f"missing V48.124.10.2 result zip: {src}")
    out.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(src) as zf:
        names = set(zf.namelist())
        missing = [name for name in REQUIRED if name not in names]
        if missing:
            raise SystemExit(f"reference result zip missing required files: {missing}")
        manifest = json.loads(zf.read("OC-RAP-v48.124.10.2-result-bundle-manifest.json"))
        if not (manifest.get("complete_exit_zero") is True and int(manifest.get("pipeline_exit_code", -1)) == 0):
            raise SystemExit("reference V48.124.10.2 pipeline did not complete with exit zero")
        listed = manifest.get("files") or {}
        for rel, meta in listed.items():
            if rel not in names:
                raise SystemExit(f"manifest member missing from zip: {rel}")
            raw = zf.read(rel)
            if sha256_bytes(raw) != str((meta or {}).get("sha256", "")):
                raise SystemExit(f"manifest SHA mismatch: {rel}")
            if len(raw) != int((meta or {}).get("size", -1)):
                raise SystemExit(f"manifest size mismatch: {rel}")
        adj = json.loads(zf.read("OC-RAP-v48.124.10.2-observation-legal-near-adjudication.json"))
        if not (adj.get("valid") and adj.get("attribution_ready")):
            raise SystemExit("reference V48.124.10.2 adjudication is not attribution-ready")
        if adj.get("status") != "OBSERVATION_LEGAL_NEAR_SYSTEM_AXIS_STOP":
            raise SystemExit(f"unexpected reference adjudication status: {adj.get('status')}")
        if adj.get("next_branch") != "keep_recovery_mechanism_frozen_audit_absolute_admitted_candidate_action_quality_no_v48_125_threshold_capacity_or_retraining_sweep":
            raise SystemExit("reference adjudication does not authorize candidate-quality audit")
        for rel in REQUIRED:
            target = out / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(rel))

    provenance = {
        "schema": "ocrap-near-candidate-quality-reference-v1",
        "source_zip": str(src),
        "source_zip_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
        "reference_status": adj.get("status"),
        "reference_valid": bool(adj.get("valid")),
        "reference_attribution_ready": bool(adj.get("attribution_ready")),
        "num_manifest_files_verified": len(listed),
        "required_files": REQUIRED,
        "valid": True,
    }
    (out / "REFERENCE_CONTRACT.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(provenance, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
