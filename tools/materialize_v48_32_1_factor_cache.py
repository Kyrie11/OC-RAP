#!/usr/bin/env python3
"""Atomically materialize and verify an exact-contract Stage-1 factor cache."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import time
from pathlib import Path


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read(path: Path) -> dict:
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise TypeError(f"JSON object required: {path}")
    return doc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-stage", type=Path, required=True)
    ap.add_argument("--destination-stage", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    source = args.source_stage.resolve()
    destination = args.destination_stage.resolve()
    report = {
        "event": "v48_32_1_factor_cache_materialization",
        "created_unix": time.time(), "valid": False,
        "source_stage": str(source), "destination_stage": str(destination),
    }
    try:
        if source == destination:
            raise ValueError("factor cache source and destination must be different")
        required = (
            "model_v48_trac_sr", "STAGE_ARCHITECTURE.json", "POLICY_CONTRACT.env",
            "TRAINING_COMPLETE.json", "EVIDENCE_CORRECTION_COMPLETE.json",
            "FACTOR_CACHE_CONTRACT.json",
        )
        missing = [name for name in required if not (source / name).exists()]
        if missing:
            raise FileNotFoundError("missing factor-cache artifacts: " + ",".join(missing))
        source_checkpoint = source / "model_v48_trac_sr" / "best.pt"
        if not source_checkpoint.is_file():
            raise FileNotFoundError(source_checkpoint)
        source_sha = _sha(source_checkpoint)
        training = _read(source / "TRAINING_COMPLETE.json")
        evidence = _read(source / "EVIDENCE_CORRECTION_COMPLETE.json")
        for label, doc in (("training", training), ("evidence", evidence)):
            expected = doc.get("checkpoint_sha256")
            if expected != source_sha:
                raise ValueError(f"{label} checkpoint SHA mismatch: metadata={expected} actual={source_sha}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp = Path(tempfile.mkdtemp(prefix=destination.name + ".tmp.", dir=destination.parent))
        try:
            for name in required:
                src = source / name
                dst = temp / name
                if src.is_dir():
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
            copied_checkpoint = temp / "model_v48_trac_sr" / "best.pt"
            copied_sha = _sha(copied_checkpoint)
            if copied_sha != source_sha:
                raise IOError("copied factor checkpoint SHA mismatch")
            destination_checkpoint = destination / "model_v48_trac_sr" / "best.pt"
            for filename in ("TRAINING_COMPLETE.json", "EVIDENCE_CORRECTION_COMPLETE.json"):
                path = temp / filename
                doc = _read(path)
                doc["checkpoint"] = str(destination_checkpoint)
                doc["checkpoint_sha256"] = copied_sha
                doc["factor_cache_reused"] = True
                doc["factor_cache_source_stage"] = str(source)
                doc["factor_cache_materialized_unix"] = time.time()
                path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            architecture_path = temp / "STAGE_ARCHITECTURE.json"
            architecture = _read(architecture_path)
            architecture["factor_cache_reused"] = True
            architecture["factor_cache_source_stage"] = str(source)
            architecture_path.write_text(json.dumps(architecture, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            if destination.exists():
                shutil.rmtree(destination)
            temp.rename(destination)
        except Exception:
            if temp.exists():
                shutil.rmtree(temp)
            raise
        final_checkpoint = destination / "model_v48_trac_sr" / "best.pt"
        final_sha = _sha(final_checkpoint)
        if final_sha != source_sha:
            raise IOError("materialized factor checkpoint SHA mismatch")
        report.update({
            "valid": True, "source_checkpoint": str(source_checkpoint),
            "destination_checkpoint": str(final_checkpoint),
            "checkpoint_sha256": final_sha,
            "metadata_paths_rewritten": True,
        })
        materialization = destination / "FACTOR_CACHE_MATERIALIZATION.json"
        materialization.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except Exception as exc:
        report.update({"error_type": type(exc).__name__, "error": str(exc)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["valid"] else 30


if __name__ == "__main__":
    raise SystemExit(main())
