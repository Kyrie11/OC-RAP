#!/usr/bin/env python3
"""Create a fresh, self-auditing v48.35 result ZIP.

The archive is always opened in write mode after deleting any previous target.
This prevents `zip -r existing.zip ...` from retaining files that no longer
exist in the run directory, which was the source of the false RC=30 report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import time
import zipfile
from pathlib import Path
from typing import Any

from audit_v48_35_run_state import resolve


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def symlink_info(name: str, target: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name)
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    return info


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-checkpoints", action="store_true")
    parser.add_argument("--include-status-history", action="store_true")
    args = parser.parse_args()

    run = args.run.resolve()
    if not run.is_dir():
        raise SystemExit(f"run directory does not exist: {run}")
    state = resolve(run)
    if not state.get("valid"):
        print(json.dumps(state, ensure_ascii=False, indent=2))
        raise SystemExit("refusing to package an inconsistent run state")

    stale_names = {Path(item["path"]).resolve() for item in state.get("stale_markers") or []}
    excluded: list[dict[str, str]] = []
    entries: list[dict[str, Any]] = []
    root_name = run.name
    output = args.output.resolve()
    hash_path = output.with_suffix(output.suffix + ".sha256").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    for generated in (output, hash_path):
        try:
            generated.unlink()
        except FileNotFoundError:
            pass

    candidates: list[Path] = []
    for path in sorted(run.rglob("*")):
        rel = path.relative_to(run)
        resolved = path.resolve()
        if resolved in {output, hash_path}:
            excluded.append({"path": str(rel), "reason": "packaging_output_excluded"})
            continue
        if rel.as_posix() in {"AUTHORITATIVE_RUN_STATUS.json", "PACKAGING_MANIFEST.json"}:
            excluded.append({"path": str(rel), "reason": "generated_packaging_metadata_replaced"})
            continue
        if rel.parts and rel.parts[0] == "status_history" and not args.include_status_history:
            excluded.append({"path": str(rel), "reason": "historical_status_excluded"})
            continue
        if path.resolve() in stale_names:
            excluded.append({"path": str(rel), "reason": "stale_terminal_marker_excluded"})
            continue
        if path.is_file() and not args.include_checkpoints and path.suffix.lower() in {".pt", ".pth", ".ckpt"}:
            excluded.append({"path": str(rel), "reason": "checkpoint_excluded"})
            continue
        if path.is_dir() and not path.is_symlink():
            continue
        candidates.append(path)

    created_unix = time.time()
    state_for_archive = dict(state)
    state_for_archive.update(
        {
            "packaged_unix": created_unix,
            "packaging_version": "v48.35.2-ENGINEERING-INTEGRITY",
            "stale_markers_excluded_from_archive": [str(p.relative_to(run)) for p in sorted(stale_names) if p.is_relative_to(run)],
        }
    )

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for path in candidates:
            rel = path.relative_to(run).as_posix()
            arcname = f"{root_name}/{rel}"
            if path.is_symlink():
                target = os.readlink(path)
                data = target.encode("utf-8")
                archive.writestr(symlink_info(arcname, target), data)
                entries.append({"path": rel, "type": "symlink", "target": target, "sha256": sha256_bytes(data), "size": len(data)})
            else:
                digest = sha256_file(path)
                size = path.stat().st_size
                archive.write(path, arcname)
                entries.append({"path": rel, "type": "file", "sha256": digest, "size": size})

        state_data = json_bytes(state_for_archive)
        state_arc = f"{root_name}/AUTHORITATIVE_RUN_STATUS.json"
        archive.writestr(state_arc, state_data)
        entries = [entry for entry in entries if entry["path"] != "AUTHORITATIVE_RUN_STATUS.json"]
        entries.append({"path": "AUTHORITATIVE_RUN_STATUS.json", "type": "generated", "sha256": sha256_bytes(state_data), "size": len(state_data)})

        manifest = {
            "event": "v48_35_result_package_manifest",
            "version": "v48.35.2-ENGINEERING-INTEGRITY",
            "created_unix": created_unix,
            "source_run": str(run),
            "archive_root": root_name,
            "authoritative_exit_code": state.get("authoritative_exit_code"),
            "pipeline_valid": state.get("pipeline_valid"),
            "include_checkpoints": args.include_checkpoints,
            "include_status_history": args.include_status_history,
            "entry_count": len(entries),
            "entries": entries,
            "excluded": excluded,
            "test_roots_read": False,
        }
        manifest_data = json_bytes(manifest)
        archive.writestr(f"{root_name}/PACKAGING_MANIFEST.json", manifest_data)

    # Round-trip integrity and duplicate-name audit.
    with zipfile.ZipFile(output, "r") as archive:
        names = archive.namelist()
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise SystemExit(f"duplicate ZIP entries: {duplicates[:10]}")
        archived_state = json.loads(archive.read(f"{root_name}/AUTHORITATIVE_RUN_STATUS.json"))
        if not archived_state.get("valid") or archived_state.get("authoritative_exit_code") != state.get("authoritative_exit_code"):
            raise SystemExit("authoritative state changed during packaging")
        for stale in state.get("stale_markers") or []:
            stale_arc = f"{root_name}/{Path(stale['path']).name}"
            if stale_arc in names:
                raise SystemExit(f"stale marker leaked into package: {stale_arc}")
        archived_manifest = json.loads(archive.read(f"{root_name}/PACKAGING_MANIFEST.json"))
        for entry in archived_manifest["entries"]:
            arcname = f"{root_name}/{entry['path']}"
            payload = archive.read(arcname)
            if sha256_bytes(payload) != entry["sha256"]:
                raise SystemExit(f"round-trip hash mismatch: {arcname}")

    zip_hash = sha256_file(output)
    hash_path.write_text(f"{zip_hash}  {output.name}\n", encoding="utf-8")
    result = {
        "output": str(output),
        "sha256": zip_hash,
        "sha256_file": str(hash_path),
        "authoritative_exit_code": state.get("authoritative_exit_code"),
        "pipeline_valid": state.get("pipeline_valid"),
        "stale_markers_excluded": [Path(item["path"]).name for item in state.get("stale_markers") or []],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
