#!/usr/bin/env python3
"""Create a linked OC-RAP target root containing selected scene/time targets.

Selection can be a critical-scene JSON produced by select_critical_scenes_v48_34.py
or a JSON/JSONL file containing target_key, scene_id, and target_time_index fields.
The tool is fail-closed: ambiguous scene-only matches are rejected unless explicitly
allowed, and every requested target must be matched exactly.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterable


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _scene_id(row: dict[str, Any]) -> str:
    return _clean(row.get("original_scenario_id") or row.get("scene_id") or row.get("scenario_id"))


def _time_index(row: dict[str, Any]) -> str:
    for key in ("target_time_index", "time_index", "anchor_time_index", "current_time_index"):
        value = _clean(row.get(key))
        if value:
            try:
                return str(int(float(value)))
            except ValueError:
                return value
    return ""


def _target_key(row: dict[str, Any]) -> str:
    return _clean(row.get("target_key") or row.get("bucket_target_key") or row.get("resume_key"))


def _load_manifest(root: Path) -> tuple[list[dict[str, str]], list[str]]:
    path = root / "manifest.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def _selection_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                envelope = json.loads(line)
                rows.append(envelope.get("scene", envelope))
        return rows
    doc = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(doc, list):
        return [dict(x) for x in doc]
    for key in ("selected", "targets", "rows", "all_scene_scores"):
        value = doc.get(key)
        if isinstance(value, list):
            return [dict(x) for x in value]
    return [dict(doc)]


def _request_identity(row: dict[str, Any]) -> tuple[str, str, str]:
    key = _target_key(row)
    scene = _scene_id(row)
    time = _time_index(row)
    if not key and not scene:
        raise ValueError(f"selection row lacks target_key/scene_id: {row}")
    return key, scene, time


def _link(src: Path, dst: Path, mode: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return "hardlink"
        except OSError:
            mode = "symlink"
    if mode == "symlink":
        try:
            dst.symlink_to(src.resolve())
            return "symlink"
        except OSError:
            mode = "copy"
    shutil.copy2(src, dst)
    return "copy"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--link-mode", choices=("hardlink", "symlink", "copy"), default="hardlink")
    ap.add_argument("--allow-scene-only", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    rows, fields = _load_manifest(args.input)
    requests = list(dict.fromkeys(_request_identity(row) for row in _selection_rows(args.selection)))
    if not requests:
        raise ValueError("selection contains no targets")

    by_key: dict[str, list[int]] = {}
    by_scene_time: dict[tuple[str, str], list[int]] = {}
    by_scene: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        key, scene, time = _target_key(row), _scene_id(row), _time_index(row)
        if key:
            by_key.setdefault(key, []).append(i)
        if scene:
            by_scene.setdefault(scene, []).append(i)
            if time:
                by_scene_time.setdefault((scene, time), []).append(i)

    matched: list[int] = []
    audit: list[dict[str, Any]] = []
    for key, scene, time in requests:
        candidates: list[int] = []
        match_mode = ""
        if key and key in by_key:
            candidates = by_key[key]
            match_mode = "target_key"
        elif scene and time and (scene, time) in by_scene_time:
            candidates = by_scene_time[(scene, time)]
            match_mode = "scene_time"
        elif scene and args.allow_scene_only and scene in by_scene:
            candidates = by_scene[scene]
            match_mode = "scene_only"
        if not candidates:
            raise ValueError(f"requested target not found: target_key={key!r}, scene={scene!r}, time={time!r}")
        if len(candidates) != 1:
            raise ValueError(
                f"ambiguous target ({match_mode}) target_key={key!r}, scene={scene!r}, time={time!r}: "
                f"{len(candidates)} manifest rows"
            )
        index = candidates[0]
        matched.append(index)
        audit.append({"requested_target_key": key, "requested_scene_id": scene, "requested_time_index": time,
                      "match_mode": match_mode, "manifest_row_index": index})

    matched = list(dict.fromkeys(matched))
    if len(matched) != len(requests):
        raise ValueError("multiple requested targets resolved to the same manifest row")
    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(args.output)
        shutil.rmtree(args.output)
    (args.output / "samples").mkdir(parents=True)

    out_rows: list[dict[str, str]] = []
    link_modes: dict[str, int] = {}
    for index in matched:
        row = dict(rows[index])
        raw = Path(row.get("path", ""))
        src = raw if raw.is_absolute() else args.input / raw
        if not src.exists():
            alt = args.input / "samples" / raw.name
            if not alt.exists():
                raise FileNotFoundError(src)
            src = alt
        dst = args.output / "samples" / src.name
        mode = _link(src, dst, args.link_mode)
        link_modes[mode] = link_modes.get(mode, 0) + 1
        row["path"] = str(Path("samples") / dst.name)
        out_rows.append(row)

    with (args.output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(out_rows)

    provenance = {
        "event": "v48_34_target_subset",
        "source": str(args.input.resolve()),
        "selection": str(args.selection.resolve()),
        "requested_targets": len(requests),
        "output_targets": len(out_rows),
        "output_scenes": len({_scene_id(row) for row in out_rows}),
        "allow_scene_only": bool(args.allow_scene_only),
        "link_modes": link_modes,
        "matches": audit,
    }
    (args.output / "TARGET_SUBSET_PROVENANCE.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: provenance[k] for k in ("event", "requested_targets", "output_targets", "output_scenes")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
