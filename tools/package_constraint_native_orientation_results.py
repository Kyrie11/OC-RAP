#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from ocrap.audits.viability_rank_persistence import ENGINEERING_VERSION, SCIENTIFIC_VERSION

EXPECTED = {
    "runtime": "OC-RAP-v48.121-runtime-code-contract.json",
    "balanced": "OC-RAP-v48.121-VRPC-balanced.json",
    "precision": "OC-RAP-v48.121-VRPC-precision.json",
    "balanced_state": "OC-RAP-v48.121-VRPC-balanced.pt",
    "precision_state": "OC-RAP-v48.121-VRPC-precision.pt",
    "comparison": "OC-RAP-v48.121-DCP-DRFC-BCDE-RIFA-OC-VRPC-comparison.json",
}
PIPELINE_NAME = "OC-RAP-v48.121-PIPELINE_COMPLETE.json"
MANIFEST_NAME = "OC-RAP-v48.121-OC-VRPC-result-bundle-manifest.json"
RESULT_NAME = "OC-RAP-v48.121-OC-VRPC-results.zip"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", type=Path, required=True)
    ap.add_argument("--base-out", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    errors: list[str] = []
    pipe = json.loads(a.pipeline.read_text()) if a.pipeline.is_file() else {}
    if not (pipe.get("valid") and pipe.get("attribution_ready")):
        errors.append("pipeline_not_valid")
    if pipe.get("engineering_version") != ENGINEERING_VERSION:
        errors.append("pipeline_engineering_version")
    if pipe.get("scientific_version") != SCIENTIFIC_VERSION:
        errors.append("pipeline_scientific_version")
    if pipe.get("run_instance_id") != a.run_id:
        errors.append("pipeline_run_instance_id")

    allowed = set(EXPECTED.values()) | {PIPELINE_NAME, MANIFEST_NAME, RESULT_NAME}
    stale = sorted(str(p) for p in a.base_out.glob("OC-RAP-v48.121-*") if p.name not in allowed)
    if stale:
        errors.append("noncanonical_v48_121_artifacts_present")

    resolved: list[Path] = []
    files: dict[str, dict[str, object]] = {}
    arts = pipe.get("artifacts") or {}
    for key, name in EXPECTED.items():
        rec = arts.get(key) or {}
        p = Path(str(rec.get("path", "")))
        if p.name != name:
            errors.append(f"{key}:noncanonical_name:{p.name}")
        if not p.is_file():
            errors.append(f"{key}:missing")
            continue
        actual = sha(p)
        if actual != rec.get("sha256"):
            errors.append(f"{key}:sha")
        resolved.append(p)
        files[name] = {"sha256": actual, "size": p.stat().st_size}
    if a.pipeline.name != PIPELINE_NAME:
        errors.append("pipeline_noncanonical_name")
    files[PIPELINE_NAME] = {"sha256": sha(a.pipeline), "size": a.pipeline.stat().st_size}

    manifest = {
        "schema": "ocrap-v48.121-vrpc-result-bundle-manifest-v1",
        "engineering_version": ENGINEERING_VERSION,
        "scientific_version": SCIENTIFIC_VERSION,
        "run_instance_id": a.run_id,
        "valid": not errors,
        "errors": errors,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": files,
        "noncanonical_v48_121_artifacts": stale,
    }
    a.manifest.parent.mkdir(parents=True, exist_ok=True)
    a.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if errors:
        print(json.dumps({"valid": False, "errors": errors}))
        return 30

    members = [a.pipeline, a.manifest] + resolved
    with zipfile.ZipFile(a.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for p in members:
            zf.write(p, arcname=p.name)
    print(json.dumps({"valid": True, "output": str(a.output), "members": [p.name for p in members]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
