#!/usr/bin/env python3
"""Package final CSV tables and qualitative visualization artifacts atomically."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path

REQUIRED_ROOT_CSVS = (
    "safe_compare.csv",
    "near_compare.csv",
    "contact_compare.csv",
    "ablation_results.csv",
    "ablation_balanced_safe.csv",
    "ablation_balanced_near.csv",
    "ablation_balanced_contact.csv",
    "ablation_precision_safe.csv",
    "ablation_precision_near.csv",
    "ablation_precision_contact.csv",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def add_file(zf: zipfile.ZipFile, src: Path, arc: str, manifest: list[dict[str, object]]) -> None:
    compression = zipfile.ZIP_STORED if src.suffix.lower() in {".mp4", ".gif", ".png", ".jpg", ".jpeg"} else zipfile.ZIP_DEFLATED
    zf.write(src, arcname=arc, compress_type=compression)
    manifest.append({"path": arc, "size": src.stat().st_size, "sha256": sha256(src)})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=Path, required=True)
    ap.add_argument("--visualization-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    runs = args.runs_root.resolve()
    vis = args.visualization_root.resolve()
    out = args.output.resolve()
    if not vis.is_dir():
        raise SystemExit(f"missing visualization directory: {vis}")
    required = [runs / name for name in REQUIRED_ROOT_CSVS]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise SystemExit("missing final result CSVs: " + ", ".join(missing))
    video_index = vis / "videos" / "REGIME_VIDEO_INDEX.json"
    if not video_index.is_file():
        raise SystemExit(f"missing visualization video index: {video_index}")

    out.parent.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    fd, tmp_name = tempfile.mkstemp(prefix=out.name + ".", suffix=".tmp", dir=out.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with zipfile.ZipFile(tmp, "w", allowZip64=True) as zf:
            for src in required:
                add_file(zf, src, f"tables/{src.name}", manifest)
            for optional in (runs / "ablation_results_index.json",):
                if optional.is_file():
                    add_file(zf, optional, f"metadata/{optional.name}", manifest)
            for src in sorted(p for p in vis.rglob("*") if p.is_file()):
                rel = src.relative_to(vis)
                # Selective trace journals are large intermediate replay artifacts.
                # The downloadable result package carries the rendered videos,
                # paper figures, selections and provenance instead.
                if rel.parts and rel.parts[0] == "selective_traces":
                    continue
                add_file(zf, src, f"visualization/{rel.as_posix()}", manifest)
            payload = {
                "schema_version": 1,
                "contents": "final comparison/ablation CSVs plus regime visualization artifacts",
                "num_files": len(manifest),
                "files": manifest,
            }
            zf.writestr("MANIFEST.json", json.dumps(payload, ensure_ascii=False, indent=2) + "\n", compress_type=zipfile.ZIP_DEFLATED)
        os.replace(tmp, out)
    finally:
        if tmp.exists():
            tmp.unlink()
    print(json.dumps({"event": "final_results_package", "output": str(out), "num_files": len(manifest)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
