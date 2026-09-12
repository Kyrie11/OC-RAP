#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from ocrap.audits.fixed_main_stability import ENGINEERING_VERSION, SCIENTIFIC_VERSION

PIPELINE_NAME = "OC-RAP-v48.124-PIPELINE_COMPLETE.json"
RUNTIME_NAME = "OC-RAP-v48.124-runtime-code-contract.json"
ADJUDICATION_NAME = "OC-RAP-v48.124-fixed-main-adjudication.json"
SENTINEL_INDEX_NAME = "OC-RAP-v48.124-sentinel-index.json"
MANIFEST_NAME = "OC-RAP-v48.124-OC-FMSA-result-bundle-manifest.json"
RESULT_NAME = "OC-RAP-v48.124-OC-FMSA-results.zip"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", type=Path, required=True)
    ap.add_argument("--base-out", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    errors: list[str] = []
    pipe = json.loads(a.pipeline.read_text(encoding="utf-8")) if a.pipeline.is_file() else {}
    if not (pipe.get("valid") and pipe.get("attribution_ready")):
        errors.append("pipeline_not_valid")
    if pipe.get("engineering_version") != ENGINEERING_VERSION or pipe.get("scientific_version") != SCIENTIFIC_VERSION:
        errors.append("pipeline_version")
    if pipe.get("run_instance_id") != a.run_id:
        errors.append("pipeline_run_instance_id")

    arts = pipe.get("artifacts") or {}
    # Canonical archive names are independent of nested run directories.
    canonical: dict[str, str] = {
        "runtime": RUNTIME_NAME,
        "adjudication": ADJUDICATION_NAME,
        "sentinel_index": SENTINEL_INDEX_NAME,
    }
    for regime in ("safe", "near", "contact"):
        for variant in ("nominal", "balanced", "precision"):
            canonical[f"{variant}_{regime}"] = f"OC-RAP-v48.124-{variant}-{regime}-closed-loop.json"
        for variant in ("balanced", "precision"):
            canonical[f"{variant}_{regime}_comparison"] = f"OC-RAP-v48.124-{variant}-{regime}-vs-nominal.json"
            canonical[f"{variant}_{regime}_sentinel"] = f"OC-RAP-v48.124-{variant}-{regime}-sentinel.json"

    members: list[tuple[Path, str]] = []
    files: dict[str, dict[str, object]] = {}
    for key, arcname in canonical.items():
        rec = arts.get(key) or {}
        p = Path(str(rec.get("path", "")))
        if not p.is_file():
            errors.append(f"{key}:missing")
            continue
        got = sha(p)
        if rec.get("sha256") and got != rec.get("sha256"):
            errors.append(f"{key}:sha")
        members.append((p, arcname))
        files[arcname] = {"sha256": got, "size": p.stat().st_size}

    if a.pipeline.name != PIPELINE_NAME:
        errors.append("pipeline_noncanonical_name")
    files[PIPELINE_NAME] = {"sha256": sha(a.pipeline), "size": a.pipeline.stat().st_size}
    members.append((a.pipeline, PIPELINE_NAME))

    allowed = set(files) | {MANIFEST_NAME, RESULT_NAME}
    stale = sorted(p.name for p in a.base_out.glob("OC-RAP-v48.124-*") if p.name not in allowed)
    if stale:
        errors.append("noncanonical_v48_124_artifacts_present")

    manifest = {
        "schema": "ocrap-v48.124-fmsa-result-bundle-manifest-v1",
        "engineering_version": ENGINEERING_VERSION,
        "scientific_version": SCIENTIFIC_VERSION,
        "run_instance_id": a.run_id,
        "valid": not errors,
        "errors": errors,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": files,
        "noncanonical_v48_124_artifacts": stale,
    }
    a.manifest.parent.mkdir(parents=True, exist_ok=True)
    a.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if errors:
        print(json.dumps({"valid": False, "errors": errors}))
        return 30

    members.append((a.manifest, MANIFEST_NAME))
    with zipfile.ZipFile(a.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for p, arcname in members:
            zf.write(p, arcname=arcname)
    print(json.dumps({"valid": True, "output": str(a.output), "members": [n for _, n in members]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
