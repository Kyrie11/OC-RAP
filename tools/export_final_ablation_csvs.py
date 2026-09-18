#!/usr/bin/env python3
"""Export authoritative split ablation CSVs plus a convenience long-format union.

The paper-facing tables stay split by variant and regime because Safe/Near/Contact
have different metric schemas.  The combined file is a lossless union for
searching/plotting only; it never recomputes a metric.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

VARIANTS = ("balanced", "precision")
REGIMES = ("safe", "near", "contact")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission-table-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    args = ap.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    split: list[dict[str, str]] = []
    combined_rows: list[dict[str, str]] = []
    combined_fields = ["variant", "regime"]

    for variant in VARIANTS:
        for regime in REGIMES:
            src = args.submission_table_root / variant / regime / f"{regime}_comparison.csv"
            if not src.is_file():
                raise SystemExit(f"missing ablation table: {src}")
            dst = args.output_root / f"ablation_{variant}_{regime}.csv"
            shutil.copyfile(src, dst)
            with src.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                fields = list(reader.fieldnames or [])
                if not fields:
                    raise SystemExit(f"empty ablation CSV schema: {src}")
                for field in fields:
                    if field not in combined_fields:
                        combined_fields.append(field)
                for row in reader:
                    combined_rows.append({"variant": variant, "regime": regime, **row})
            split.append({"variant": variant, "regime": regime, "source": str(src), "output": str(dst)})

    combined = args.output_root / "ablation_results.csv"
    with combined.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=combined_fields, extrasaction="ignore")
        writer.writeheader()
        for row in combined_rows:
            writer.writerow({key: row.get(key, "") for key in combined_fields})

    index = {
        "schema_version": 1,
        "authoritative_layout": "split_by_variant_and_regime",
        "reason": "Safe/Near/Contact use different metric schemas; split tables avoid sparse/incomparable cross-regime columns.",
        "combined_csv_role": "lossless convenience union only; no metric recomputation",
        "split_tables": split,
        "combined": str(combined),
    }
    (args.output_root / "ablation_results_index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"event": "final_ablation_csv_export", "split_tables": len(split), "combined": str(combined)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
