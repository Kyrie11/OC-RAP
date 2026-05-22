#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], dry_run: bool = False) -> None:
    print(" ".join(cmd), flush=True)
    if not dry_run:
        subprocess.check_call(cmd)


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert Waymo Open Motion Dataset tf_example files to a MetaDrive/ScenarioNet database.")
    ap.add_argument("--womd-root", required=True, help="Root containing training/validation/testing_interactive tf_example folders.")
    ap.add_argument("--output-root", required=True, help="Where ScenarioNet databases will be written.")
    ap.add_argument("--splits", nargs="+", default=["training", "validation", "testing_interactive"])
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--num-files", type=int, default=None, help="Optional per-split debug cap.")
    ap.add_argument("--start-file-index", type=int, default=0)
    ap.add_argument("--dataset-prefix", default="womd_1_3_1")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    womd_root = Path(args.womd_root)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    commands = []
    for split in args.splits:
        raw = womd_root / split
        if not raw.exists():
            raise FileNotFoundError(f"WOMD split not found: {raw}")
        db = out_root / split
        cmd = [
            sys.executable,
            "-m",
            "scenarionet.convert_waymo",
            "--database_path",
            str(db),
            "--dataset_name",
            f"{args.dataset_prefix}_{split}",
            "--version",
            "v1.3.1",
            "--num_workers",
            str(args.num_workers),
            "--raw_data_path",
            str(raw),
            "--start_file_index",
            str(args.start_file_index),
        ]
        if args.num_files is not None:
            cmd += ["--num_files", str(args.num_files)]
        if args.overwrite:
            cmd.append("--overwrite")
        commands.append(cmd)
        _run(cmd, args.dry_run)
    manifest = {
        "womd_root": str(womd_root),
        "output_root": str(out_root),
        "splits": args.splits,
        "commands": [" ".join(c) for c in commands],
    }
    (out_root / "conversion_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
