#!/usr/bin/env python3
"""Materialize a verified v48.35 Stage-1 factor cache without stale-path ambiguity.

Checkpoint tensors are hard-linked when possible, while mutable metadata is
copied and rewritten to the destination stage. The source artifact is never
modified. Checkpoint SHA256 values are verified before and after materialization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy_tree(source: Path, destination: Path) -> dict[str, int]:
    modes = {"hardlink": 0, "copy": 0, "symlink": 0}
    for src in sorted(source.rglob("*")):
        rel = src.relative_to(source)
        dst = destination / rel
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_symlink():
            dst.symlink_to(os.readlink(src))
            modes["symlink"] += 1
            continue
        # Only immutable tensor artifacts are hard-linked. Metadata/log files
        # are copied because destination paths are rewritten below.
        if src.suffix in {".pt", ".pth", ".ckpt"}:
            try:
                os.link(src, dst)
                modes["hardlink"] += 1
                continue
            except OSError:
                pass
        shutil.copy2(src, dst)
        modes["copy"] += 1
    return modes


def _rewrite_metadata(destination: Path, source: Path) -> int:
    replacements = 0
    source_forms = {str(source), str(source.resolve())}
    destination_text = str(destination)
    for path in destination.rglob("*"):
        if not path.is_file() or path.suffix not in {".json", ".jsonl", ".env", ".txt", ".log"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        updated = text
        for old in source_forms:
            updated = updated.replace(old, destination_text)
        if updated != text:
            path.write_text(updated, encoding="utf-8")
            replacements += 1
    return replacements


def _metadata_checkpoint_sha(stage: Path, name: str) -> str | None:
    path = stage / name
    if not path.is_file():
        return None
    doc = json.loads(path.read_text(encoding="utf-8"))
    value = doc.get("checkpoint_sha256")
    return str(value) if value else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-stage", type=Path, required=True)
    ap.add_argument("--destination-stage", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    source = args.source_stage
    destination = args.destination_stage
    source_best = source / "model_v48_trac_sr" / "best.pt"
    contract = source / "FACTOR_CACHE_CONTRACT.json"
    if not source_best.is_file():
        raise FileNotFoundError(source_best)
    if not contract.is_file():
        raise FileNotFoundError(contract)
    source_sha = _sha(source_best)
    for metadata in ("TRAINING_COMPLETE.json", "EVIDENCE_CORRECTION_COMPLETE.json"):
        recorded = _metadata_checkpoint_sha(source, metadata)
        if recorded is not None and recorded != source_sha:
            raise ValueError(f"{metadata} checkpoint SHA mismatch: {recorded} != {source_sha}")

    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    modes = _copy_tree(source, destination)
    rewritten = _rewrite_metadata(destination, source)
    destination_best = destination / "model_v48_trac_sr" / "best.pt"
    destination_sha = _sha(destination_best)
    if destination_sha != source_sha:
        raise ValueError("materialized factor checkpoint differs from source")
    for metadata in ("TRAINING_COMPLETE.json", "EVIDENCE_CORRECTION_COMPLETE.json"):
        recorded = _metadata_checkpoint_sha(destination, metadata)
        if recorded is not None and recorded != destination_sha:
            raise ValueError(f"materialized {metadata} checkpoint SHA mismatch")

    doc: dict[str, Any] = {
        "event": "v48_35_factor_cache_materialized",
        "created_unix": time.time(),
        "source_stage": str(source.resolve()),
        "destination_stage": str(destination.resolve()),
        "source_checkpoint_sha256": source_sha,
        "destination_checkpoint_sha256": destination_sha,
        "checkpoint_identity_preserved": True,
        "source_stage_modified": False,
        "metadata_files_rewritten": rewritten,
        "copy_modes": modes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (destination / "FACTOR_CACHE_MATERIALIZED.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(doc, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
