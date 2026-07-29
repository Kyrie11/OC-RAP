#!/usr/bin/env python3
"""Split dedicated Near/Contact calibration roots into three scene-disjoint roles.

The protocol intentionally separates:
  * evidence_adapt_train: fine-tune only the small evidence adapter;
  * evidence_adapt_dev: early stopping/model selection for that adapter;
  * certificate_pool: threshold fitting and held-out verification.

No test_* root is read.  Samples are hard-linked by default and the original
calibration roots are never modified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def _score(scene: str, seed: int) -> float:
    h = hashlib.sha256(f"v48.14|{seed}|{scene}".encode("utf-8", errors="replace")).digest()
    return int.from_bytes(h[:8], "big") / float(2**64)


def _link(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
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


def _read_manifest(root: Path) -> tuple[list[dict[str, str]], list[str], dict[str, list[dict[str, str]]]]:
    manifest = root / "manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    with manifest.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        fields = list(rows[0].keys()) if rows else []
    if not rows:
        raise ValueError(f"empty manifest: {manifest}")
    by_scene: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        scene = (row.get("original_scenario_id") or row.get("scene_id") or "").strip()
        if not scene:
            raise ValueError(f"manifest row missing scene id: {manifest}")
        by_scene[scene].append(row)
    return rows, fields, by_scene


def _resolve_sample(root: Path, row: dict[str, str]) -> Path:
    raw = Path(str(row.get("path", "")))
    candidates = [raw if raw.is_absolute() else root / raw, root / "samples" / raw.name]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(candidates[0])


def _write_partition(
    source: Path,
    output: Path,
    scenes: Iterable[str],
    by_scene: dict[str, list[dict[str, str]]],
    fields: list[str],
    *,
    role: str,
    seed: int,
    fractions: dict[str, float],
    link_mode: str,
    overwrite: bool,
) -> dict[str, object]:
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"output exists: {output}; pass --overwrite")
        shutil.rmtree(output)
    (output / "samples").mkdir(parents=True)
    out_rows: list[dict[str, str]] = []
    scene_list = sorted(set(scenes))
    for scene in scene_list:
        for row in by_scene[scene]:
            src = _resolve_sample(source, row)
            dst = output / "samples" / src.name
            _link(src, dst, link_mode)
            new = dict(row)
            new["path"] = str(Path("samples") / dst.name)
            new["split_id"] = role
            new["calibration_protocol_role"] = role
            out_rows.append(new)
    fieldnames = list(fields)
    for name in ("split_id", "calibration_protocol_role"):
        if name not in fieldnames:
            fieldnames.append(name)
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(out_rows)
    prov = {
        "event": "v48_14_dedicated_calibration_partition",
        "source": str(source.resolve()),
        "output": str(output.resolve()),
        "role": role,
        "seed": seed,
        "fractions": fractions,
        "num_scenes": len(scene_list),
        "num_samples": len(out_rows),
        "scene_ids_sha256": hashlib.sha256("\n".join(scene_list).encode()).hexdigest(),
        "scene_disjoint_by_construction": True,
        "test_roots_read": False,
    }
    (output / "split_provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=2) + "\n")
    return prov


def _partition_regime(
    source: Path,
    output_root: Path,
    regime: str,
    train_fraction: float,
    dev_fraction: float,
    seed: int,
    link_mode: str,
    overwrite: bool,
) -> dict[str, object]:
    _, fields, by_scene = _read_manifest(source)
    roles: dict[str, set[str]] = {
        "evidence_adapt_train": set(),
        "evidence_adapt_dev": set(),
        "certificate_pool": set(),
    }
    for scene in by_scene:
        x = _score(scene, seed)
        if x < train_fraction:
            roles["evidence_adapt_train"].add(scene)
        elif x < train_fraction + dev_fraction:
            roles["evidence_adapt_dev"].add(scene)
        else:
            roles["certificate_pool"].add(scene)
    if any(not v for v in roles.values()):
        raise ValueError(f"empty partition for {regime}: { {k: len(v) for k,v in roles.items()} }")
    assert not (roles["evidence_adapt_train"] & roles["evidence_adapt_dev"])
    assert not (roles["evidence_adapt_train"] & roles["certificate_pool"])
    assert not (roles["evidence_adapt_dev"] & roles["certificate_pool"])
    fractions = {
        "evidence_adapt_train": train_fraction,
        "evidence_adapt_dev": dev_fraction,
        "certificate_pool": 1.0 - train_fraction - dev_fraction,
    }
    outputs: dict[str, object] = {}
    for role, scenes in roles.items():
        out = output_root / f"{role}_{regime}"
        outputs[role] = _write_partition(
            source, out, scenes, by_scene, fields,
            role=role, seed=seed, fractions=fractions,
            link_mode=link_mode, overwrite=overwrite,
        )
    return {
        "regime": regime,
        "source": str(source.resolve()),
        "source_scenes": len(by_scene),
        "partitions": outputs,
        "scene_union_complete": set().union(*roles.values()) == set(by_scene),
        "scene_overlap": 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--near", type=Path, required=True)
    ap.add_argument("--contact", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--adapt-train-fraction", type=float, default=0.45)
    ap.add_argument("--adapt-dev-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=4814)
    ap.add_argument("--link-mode", choices=["hardlink", "symlink", "copy"], default="hardlink")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if not 0.0 < args.adapt_train_fraction < 1.0:
        raise ValueError("adapt train fraction must be in (0,1)")
    if not 0.0 < args.adapt_dev_fraction < 1.0:
        raise ValueError("adapt dev fraction must be in (0,1)")
    if args.adapt_train_fraction + args.adapt_dev_fraction >= 1.0:
        raise ValueError("adapt train + dev fractions must be < 1")
    args.output_root.mkdir(parents=True, exist_ok=True)
    report = {
        "event": "v48_14_scene_disjoint_dedicated_protocol_complete",
        "version": "v48.14-PRISM",
        "seed": args.seed,
        "test_roots_read": False,
        "regimes": [
            _partition_regime(
                args.near, args.output_root, "near_contact",
                args.adapt_train_fraction, args.adapt_dev_fraction,
                args.seed, args.link_mode, args.overwrite,
            ),
            _partition_regime(
                args.contact, args.output_root, "contact",
                args.adapt_train_fraction, args.adapt_dev_fraction,
                args.seed, args.link_mode, args.overwrite,
            ),
        ],
    }
    path = args.output_root / "CALIBRATION_PROTOCOL_COMPLETE.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
