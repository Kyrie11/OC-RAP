#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ENGINEERING_VERSION = "v48.124.9-OC-FMSA-RUNTIME-SNAPSHOT-ENGFIX"
REQUIRED = (
    "OC-RAP-v48.124-fixed-main-adjudication.json",
    "OC-RAP-v48.124-nominal-near-closed-loop.json",
    "OC-RAP-v48.124-nominal-near-dataset-support.json",
    "OC-RAP-v48.124-balanced-near-closed-loop.json",
    "OC-RAP-v48.124-balanced-near-dataset-support.json",
    "OC-RAP-v48.124-precision-near-closed-loop.json",
    "OC-RAP-v48.124-precision-near-dataset-support.json",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_json_bytes(data: bytes, name: str) -> dict:
    doc = json.loads(data.decode("utf-8"))
    if not isinstance(doc, dict):
        raise RuntimeError(f"{name}: expected JSON object")
    return doc


def verify_adjudication(doc: dict) -> None:
    decision = doc.get("preregistered_decision") or {}
    status = decision.get("status") or doc.get("status")
    if not (doc.get("valid") is True and doc.get("attribution_ready") is True):
        raise RuntimeError("V48.124.9 adjudication is not attribution-ready")
    if doc.get("engineering_version") != ENGINEERING_VERSION:
        raise RuntimeError(f"unexpected engineering_version={doc.get('engineering_version')!r}")
    if status != "FIXED_MAIN_NEAR_VALIDITY_STOP":
        raise RuntimeError(f"expected FIXED_MAIN_NEAR_VALIDITY_STOP, got {status!r}")


def from_zip(results_zip: Path, out: Path) -> dict:
    with zipfile.ZipFile(results_zip) as zf:
        names = set(zf.namelist())
        manifest_name = "OC-RAP-v48.124-OC-FMSA-result-bundle-manifest.json"
        if manifest_name not in names:
            raise RuntimeError("canonical results zip is missing embedded manifest")
        manifest = load_json_bytes(zf.read(manifest_name), manifest_name)
        if not (manifest.get("valid") is True and manifest.get("engineering_version") == ENGINEERING_VERSION):
            raise RuntimeError("canonical V48.124.9 bundle manifest contract failed")
        files = manifest.get("files") or {}
        for name, rec in files.items():
            if name not in names:
                raise RuntimeError(f"manifested artifact missing from zip: {name}")
            payload = zf.read(name)
            if len(payload) != int((rec or {}).get("size", -1)):
                raise RuntimeError(f"size mismatch: {name}")
            if sha256_bytes(payload) != str((rec or {}).get("sha256") or ""):
                raise RuntimeError(f"sha256 mismatch: {name}")
        for name in REQUIRED:
            if name not in files:
                raise RuntimeError(f"required Near reference not manifested: {name}")
            (out / name).write_bytes(zf.read(name))
        (out / manifest_name).write_bytes(zf.read(manifest_name))
        adj = load_json_bytes(zf.read(REQUIRED[0]), REQUIRED[0])
        verify_adjudication(adj)
        return {
            "source": "canonical_results_zip",
            "results_zip": str(results_zip.resolve()),
            "results_zip_sha256": sha256_file(results_zip),
            "run_instance_id": manifest.get("run_instance_id"),
            "manifest_tree_count": len(files),
        }


def from_workdir(base_out: Path, work: Path, out: Path) -> dict:
    mapping = {
        REQUIRED[0]: base_out / REQUIRED[0],
        REQUIRED[1]: work / "nominal/near/closed_loop_nominal.json",
        REQUIRED[2]: work / "nominal/near/closed_loop_dataset_support.json",
        REQUIRED[3]: work / "balanced/near/closed_loop_ocrap.json",
        REQUIRED[4]: work / "balanced/near/closed_loop_dataset_support.json",
        REQUIRED[5]: work / "precision/near/closed_loop_ocrap.json",
        REQUIRED[6]: work / "precision/near/closed_loop_dataset_support.json",
    }
    for name, src in mapping.items():
        if not src.is_file():
            raise RuntimeError(f"missing historical V48.124.9 prerequisite: {src}")
        shutil.copy2(src, out / name)
    verify_adjudication(json.loads((out / REQUIRED[0]).read_text(encoding="utf-8")))
    return {
        "source": "historical_workdir_fallback",
        "workdir": str(work.resolve()),
        "warning": "canonical results zip unavailable; copied already-adjudicated historical files",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Materialize and verify canonical V48.124.9 Near reference evidence.")
    ap.add_argument("--base-out", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--results-zip", type=Path)
    ap.add_argument("--historical-workdir", type=Path)
    args = ap.parse_args()
    base_out = args.base_out.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    results_zip = (args.results_zip or (base_out / "OC-RAP-v48.124-OC-FMSA-results.zip")).resolve()
    work = (args.historical_workdir or (base_out / "ocrap_v48_124_contact_anchored_fixed_main")).resolve()
    try:
        provenance = from_zip(results_zip, out) if results_zip.is_file() else from_workdir(base_out, work, out)
        checks = {name: {"sha256": sha256_file(out / name), "size": (out / name).stat().st_size} for name in REQUIRED}
        doc = {
            "schema": "ocrap-v48.124.10-near-reference-materialization-v1",
            "valid": True,
            "engineering_version": ENGINEERING_VERSION,
            "reference_provenance": provenance,
            "files": checks,
        }
        (out / "V48.124.9-near-reference-contract.json").write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"valid": True, "output_dir": str(out), **provenance}, indent=2))
        return 0
    except Exception as e:
        print(json.dumps({"valid": False, "error": str(e)}, indent=2))
        return 30


if __name__ == "__main__":
    raise SystemExit(main())
