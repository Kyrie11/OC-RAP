#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from ocrap.audits.heterogeneous_constraint_normal_cone import ENGINEERING_VERSION, SCIENTIFIC_VERSION

EXPECTED = {
    "runtime": "OC-RAP-v48.112-runtime-code-contract.json",
    "balanced": "OC-RAP-v48.112-HCNC-balanced.json",
    "precision": "OC-RAP-v48.112-HCNC-precision.json",
    "balanced_state": "OC-RAP-v48.112-HCNC-balanced.pt",
    "precision_state": "OC-RAP-v48.112-HCNC-precision.pt",
    "comparison": "OC-RAP-v48.112-DCP-DRFC-BCDE-RIFA-OC-HCNC-comparison.json",
}
PIPELINE_NAME = "OC-RAP-v48.112-PIPELINE_COMPLETE.json"
MANIFEST_NAME = "OC-RAP-v48.112-OC-HCNC-result-bundle-manifest.json"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
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

    if not a.pipeline.is_file():
        errors.append("missing_pipeline")
        pipe = {}
    else:
        pipe = json.loads(a.pipeline.read_text())
    if not (pipe.get("valid") and pipe.get("attribution_ready")):
        errors.append("pipeline_not_valid")
    if pipe.get("engineering_version") != ENGINEERING_VERSION:
        errors.append("pipeline_engineering_version")
    if pipe.get("scientific_version") != SCIENTIFIC_VERSION:
        errors.append("pipeline_scientific_version")
    if pipe.get("run_instance_id") != a.run_id:
        errors.append("pipeline_run_instance_id")

    # Refuse stale current-version aliases so one upload cannot mix runs.
    stale = sorted(
        str(p) for p in a.base_out.glob("OC-RAP-v48.112-*")
        if p.name not in set(EXPECTED.values()) | {PIPELINE_NAME, MANIFEST_NAME, "OC-RAP-v48.112-OC-HCNC-results.zip"}
    )
    if stale:
        errors.append("noncanonical_v48_112_artifacts_present")

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
            errors.append(f"{key}:sha_mismatch")
        if not key.endswith("state") and p.suffix == ".json":
            try:
                d = json.loads(p.read_text())
                if d.get("engineering_version") != ENGINEERING_VERSION:
                    errors.append(f"{key}:engineering_version")
                if d.get("scientific_version") != SCIENTIFIC_VERSION:
                    errors.append(f"{key}:scientific_version")
                if d.get("run_instance_id") != a.run_id:
                    errors.append(f"{key}:run_instance_id")
            except Exception as exc:
                errors.append(f"{key}:json:{type(exc).__name__}")
        resolved.append(p)
        files[name] = {"sha256": actual, "bytes": p.stat().st_size}

    if a.pipeline.name != PIPELINE_NAME:
        errors.append("pipeline_noncanonical_name")
    if a.pipeline.is_file():
        files[PIPELINE_NAME] = {"sha256": sha(a.pipeline), "bytes": a.pipeline.stat().st_size}

    manifest = {
        "schema": "ocrap-v48.112-hcnc-result-bundle-v1",
        "engineering_version": ENGINEERING_VERSION,
        "scientific_version": SCIENTIFIC_VERSION,
        "run_instance_id": a.run_id,
        "valid": not errors,
        "errors": errors,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_sha256": sha(a.pipeline) if a.pipeline.is_file() else None,
        "files": files,
        "noncanonical_v48_112_artifacts": stale,
        "upload_contract": "upload_this_zip_as_the_single_authoritative_v48_112_result_bundle",
    }
    a.manifest.parent.mkdir(parents=True, exist_ok=True)
    a.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if errors:
        print(json.dumps({"valid": False, "errors": errors}))
        return 30

    if a.output.exists():
        a.output.unlink()
    with zipfile.ZipFile(a.output, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in resolved + [a.pipeline, a.manifest]:
            z.write(p, arcname=p.name)
    with zipfile.ZipFile(a.output) as z:
        names = z.namelist()
        expected = [p.name for p in resolved] + [a.pipeline.name, a.manifest.name]
        if sorted(names) != sorted(expected):
            raise RuntimeError("bundle member mismatch")
        for name, rec in files.items():
            if hashlib.sha256(z.read(name)).hexdigest() != rec["sha256"]:
                raise RuntimeError(f"bundle sha mismatch {name}")
    print(json.dumps({"valid": True, "output": str(a.output.resolve()), "run_instance_id": a.run_id, "files": len(resolved) + 2}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
