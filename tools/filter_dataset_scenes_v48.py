#!/usr/bin/env python3
"""Create a linked OC-RAP root after excluding scenes found in other roots.

WOMD replay provenance is carried through the filtering transform so a derived
calibration/test bucket remains replayable under fail-closed source resolution.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path
from typing import Any

_UNKNOWN = {"", "unknown", "none", "null", "auto"}


def scene_id(row: dict[str, str]) -> str:
    value = (row.get("original_scenario_id") or row.get("scene_id") or "").strip()
    if not value:
        raise ValueError("manifest row missing original_scenario_id/scene_id")
    return value


def read_rows(root: Path) -> tuple[list[dict[str, str]], list[str]]:
    manifest = root / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(manifest)
    with manifest.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _clean_role(value: Any) -> str:
    role = str(value or "").strip().lower()
    return {
        "val": "validation",
        "interactive": "validation_interactive",
        "val_interactive": "validation_interactive",
        "validation-interactive": "validation_interactive",
        "train": "training",
    }.get(role, role)


def _source_replay_contract(root: Path, rows: list[dict[str, str]]) -> dict[str, Any] | None:
    explicit = _read_json(root / "womd_replay_contract.json")
    manifest_roles = {
        _clean_role(r.get("womd_source_role"))
        for r in rows
        if _clean_role(r.get("womd_source_role")) not in _UNKNOWN
    }
    if len(manifest_roles) > 1:
        raise RuntimeError(f"input manifest mixes WOMD source roles: root={root}, roles={sorted(manifest_roles)}")
    manifest_role = next(iter(manifest_roles)) if manifest_roles else "unknown"

    if explicit is not None:
        contract_role = _clean_role(explicit.get("womd_source_role"))
        if contract_role not in _UNKNOWN and manifest_role not in _UNKNOWN and contract_role != manifest_role:
            raise RuntimeError(
                f"input WOMD replay contract conflicts with manifest: root={root}, "
                f"contract_role={contract_role}, manifest_role={manifest_role}"
            )
        doc = dict(explicit)
        doc["womd_source_role"] = contract_role if contract_role not in _UNKNOWN else manifest_role
        return doc

    # New merged roots expose the same conservative evidence in their summary.
    merged = _read_json(root / "merged_dataset_summary.json") or {}
    summary_role = _clean_role(merged.get("womd_source_role"))
    known = {x for x in (manifest_role, summary_role) if x not in _UNKNOWN}
    if len(known) > 1:
        raise RuntimeError(f"input WOMD provenance conflicts: root={root}, roles={sorted(known)}")
    if not known:
        return None
    doc: dict[str, Any] = {
        "schema": "ocrap_womd_replay_contract_v1",
        "womd_source_role": next(iter(known)),
    }
    pattern = merged.get("source_womd_pattern")
    if isinstance(pattern, str) and pattern.strip():
        doc["womd_pattern"] = pattern.strip()
    return doc


def link(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            mode = "symlink"
    if mode == "symlink":
        try:
            dst.symlink_to(src.resolve())
            return
        except OSError:
            mode = "copy"
    shutil.copy2(src, dst)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--exclude-root", action="append", type=Path, default=[])
    ap.add_argument("--link-mode", choices=["hardlink", "symlink", "copy"], default="hardlink")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    rows, fields = read_rows(args.input)
    replay_contract = _source_replay_contract(args.input, rows)
    excluded: set[str] = set()
    for root in args.exclude_root:
        other, _ = read_rows(root)
        excluded.update(scene_id(row) for row in other)

    kept = [row for row in rows if scene_id(row) not in excluded]
    if not kept:
        raise ValueError("all scenes were excluded")
    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(args.output)
        shutil.rmtree(args.output)
    (args.output / "samples").mkdir(parents=True)

    out_rows: list[dict[str, str]] = []
    for row in kept:
        raw = Path(row.get("path", ""))
        src = raw if raw.is_absolute() else args.input / raw
        if not src.exists():
            alt = args.input / "samples" / raw.name
            if not alt.exists():
                raise FileNotFoundError(src)
            src = alt
        dst = args.output / "samples" / src.name
        link(src, dst, args.link_mode)
        new = dict(row)
        new["path"] = str(Path("samples") / dst.name)
        out_rows.append(new)

    with (args.output / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(out_rows)

    if replay_contract is not None:
        out_contract = dict(replay_contract)
        out_contract["schema"] = "ocrap_womd_replay_contract_v1"
        out_contract["derived_from"] = str(args.input.resolve())
        (args.output / "womd_replay_contract.json").write_text(
            json.dumps(out_contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    summary: dict[str, Any] = {
        "source": str(args.input.resolve()),
        "exclude_roots": [str(p.resolve()) for p in args.exclude_root],
        "input_samples": len(rows),
        "output_samples": len(out_rows),
        "input_scenes": len({scene_id(r) for r in rows}),
        "output_scenes": len({scene_id(r) for r in out_rows}),
        "excluded_scene_count": len({scene_id(r) for r in rows} & excluded),
    }
    if replay_contract is not None:
        summary["womd_source_role"] = replay_contract.get("womd_source_role")
        summary["source_womd_pattern"] = replay_contract.get("womd_pattern")
    (args.output / "scene_filter_provenance.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
