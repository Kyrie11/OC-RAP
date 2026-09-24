#!/usr/bin/env python3
"""Remove stale main rank media not referenced by the current visualization indexes.

Supplement directories are never touched.  This is intentionally conservative:
only files under ``<root>/{videos,paper_figures}/{regime}/rank_*`` are eligible.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable


def _load(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _normalized_paths(values: Iterable[object], root: Path) -> set[Path]:
    out: set[Path] = set()
    for value in values:
        if not isinstance(value, str) or not value:
            continue
        p = Path(value)
        if not p.is_absolute():
            # Index paths are often repository-relative.  Matching by resolved
            # suffix is brittle; basename/parent matching below handles both.
            p = (Path.cwd() / p).resolve()
        else:
            p = p.resolve()
        out.add(p)
    return out


def _keep_relpaths_from_video_index(index: dict) -> set[tuple[str, str]]:
    keep: set[tuple[str, str]] = set()
    for rec in index.get("records") or []:
        if not isinstance(rec, dict):
            continue
        rank = int(rec.get("rank") or 0)
        if rank <= 0:
            continue
        rank_dir = f"rank_{rank:02d}"
        for row in rec.get("videos") or []:
            if isinstance(row, dict) and row.get("path"):
                keep.add((rank_dir, Path(str(row["path"])).name))
    return keep


def _keep_relpaths_from_figure_index(index: dict) -> set[tuple[str, str]]:
    keep: set[tuple[str, str]] = set()
    for rec in index.get("records") or []:
        if not isinstance(rec, dict):
            continue
        rank = int(rec.get("rank") or 0)
        if rank <= 0:
            continue
        rank_dir = f"rank_{rank:02d}"
        for key in ("pair_files", "all_method_files"):
            vals = rec.get(key) or []
            if isinstance(vals, str):
                vals = [vals]
            for value in vals:
                if value:
                    keep.add((rank_dir, Path(str(value)).name))
    return keep


def _prune_tree(base: Path, keep: set[tuple[str, str]]) -> list[str]:
    removed: list[str] = []
    if not base.is_dir():
        return removed
    for rank_dir in sorted(base.glob("rank_*")):
        if not rank_dir.is_dir():
            continue
        for path in sorted(rank_dir.iterdir()):
            if not path.is_file():
                continue
            if (rank_dir.name, path.name) not in keep:
                removed.append(str(path))
                path.unlink()
        try:
            next(rank_dir.iterdir())
        except StopIteration:
            rank_dir.rmdir()
    return removed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    args = ap.parse_args()
    root = args.root
    removed: list[str] = []
    details = {}
    for regime in ("safe", "near", "contact"):
        vindex = _load(root / "videos" / f"{regime.upper()}_VIDEO_INDEX.json")
        findex = _load(root / "paper_figures" / f"{regime.upper()}_PAPER_FIGURE_INDEX.json")
        vkeep = _keep_relpaths_from_video_index(vindex)
        fkeep = _keep_relpaths_from_figure_index(findex)
        vr = _prune_tree(root / "videos" / regime, vkeep) if vindex else []
        fr = _prune_tree(root / "paper_figures" / regime, fkeep) if findex else []
        removed.extend(vr + fr)
        details[regime] = {"videos_removed": len(vr), "figures_removed": len(fr)}
    doc = {
        "event": "prune_stale_regime_visualization_media_v1",
        "root": str(root),
        "num_removed": len(removed),
        "supplement_directories_preserved": True,
        "details": details,
        "removed": removed,
    }
    print(json.dumps(doc, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
