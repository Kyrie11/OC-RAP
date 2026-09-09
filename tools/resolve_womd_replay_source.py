#!/usr/bin/env python3
"""Resolve the raw WOMD replay spec from the OC-RAP bucket provenance.

The OC-RAP dataset stores ``womd_source_role`` per sample.  For qualitative or
closed-loop replay we should follow that provenance instead of hard-coding a
collection in each launcher.  This tool maps the recorded role to the user's
WOMD Motion v1.3.1 ``tf_example`` root and validates the complete shard set.

Example root layout::

    <womd_root>/validation/validation_tfexample.tfrecord-00000-of-00150
    <womd_root>/validation_interactive/validation_interactive_tfexample.tfrecord-00000-of-00150

The printed stdout is the normalized TensorFlow sharded spec so shell launchers
can use command substitution safely.  A JSON audit can be written separately.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ocrap.data.womd.sharded_path import resolve_womd_spec  # noqa: E402
from ocrap.models.data import iter_sample_paths_many, scalar_metadata_for_path  # noqa: E402


ROLE_TO_PREFIX = {
    "validation": ("validation", "validation_tfexample.tfrecord"),
    "validation_interactive": ("validation_interactive", "validation_interactive_tfexample.tfrecord"),
    "training": ("training", "training_tfexample.tfrecord"),
    "testing": ("testing", "testing_tfexample.tfrecord"),
    "test": ("testing", "testing_tfexample.tfrecord"),
}


def _clean_role(value: Any) -> str:
    role = str(value or "").strip().lower()
    aliases = {
        "val": "validation",
        "interactive": "validation_interactive",
        "val_interactive": "validation_interactive",
        "validation-interactive": "validation_interactive",
        "train": "training",
    }
    return aliases.get(role, role)


def infer_dataset_source_role(dataset: str | Path, split: str = "") -> tuple[str, Counter[str], int]:
    """Return one homogeneous known source role from the requested bucket split.

    Manifest metadata is used first by ``scalar_metadata_for_path`` so this scan
    normally does not decompress the large sample tensors.
    """
    counts: Counter[str] = Counter()
    matched = 0
    for path in iter_sample_paths_many(dataset):
        row_split = str(scalar_metadata_for_path(path, "split_id", "") or "").strip()
        if split and row_split != split:
            continue
        matched += 1
        role = _clean_role(scalar_metadata_for_path(path, "womd_source_role", "unknown"))
        counts[role or "unknown"] += 1
    if matched <= 0:
        raise RuntimeError(f"no dataset rows found for dataset={dataset!s}, split={split!r}")
    known = {r for r, n in counts.items() if n > 0 and r not in {"", "unknown", "none"}}
    if not known:
        raise RuntimeError(
            f"dataset has no usable womd_source_role metadata: dataset={dataset!s}, split={split!r}, counts={dict(counts)}"
        )
    if len(known) != 1:
        raise RuntimeError(
            f"dataset mixes multiple raw WOMD source roles: dataset={dataset!s}, split={split!r}, counts={dict(counts)}"
        )
    return next(iter(known)), counts, matched


def spec_for_role(womd_root: str | Path, role: str, shards: int = 150) -> str:
    role = _clean_role(role)
    if role not in ROLE_TO_PREFIX:
        raise ValueError(f"unsupported WOMD source role {role!r}; supported={sorted(ROLE_TO_PREFIX)}")
    directory, prefix = ROLE_TO_PREFIX[role]
    return str(Path(womd_root).expanduser().resolve() / directory / prefix) + f"@{int(shards)}"


def resolve_for_dataset(
    dataset: str | Path,
    *,
    split: str,
    womd_root: str | Path,
    shards: int = 150,
    role: str = "auto",
) -> dict[str, Any]:
    inferred, counts, matched = infer_dataset_source_role(dataset, split)
    requested = _clean_role(role)
    chosen = inferred if requested in {"", "auto"} else requested
    if requested not in {"", "auto"} and chosen != inferred:
        raise RuntimeError(
            f"explicit replay role {chosen!r} disagrees with dataset provenance {inferred!r}; "
            "use --role auto or rebuild/rerun the correct bucket"
        )
    spec = spec_for_role(womd_root, chosen, shards)
    resolved = resolve_womd_spec(spec)
    if not resolved.valid:
        raise RuntimeError(
            "resolved WOMD replay source is incomplete/invalid: "
            f"role={chosen}, spec={spec}, missing={list(resolved.missing_files[:5])}, "
            f"errors={list(resolved.errors[:5])}. Expected the collection below --womd-root."
        )
    return {
        "event": "ocrap_womd_replay_source_v53",
        "dataset": str(Path(dataset).resolve()),
        "split": split,
        "womd_root": str(Path(womd_root).expanduser().resolve()),
        "dataset_source_role": inferred,
        "requested_role": role,
        "resolved_role": chosen,
        "role_counts": dict(sorted(counts.items())),
        "num_dataset_rows": matched,
        "womd_spec": spec,
        "num_resolved_womd_files": len(resolved.files),
        "womd_shard_spec": resolved.as_dict(),
        "valid": True,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--womd-root", required=True,
                    help="Directory containing validation/ and validation_interactive/ (the tf_example root).")
    ap.add_argument("--shards", type=int, default=150)
    ap.add_argument("--role", default="auto",
                    help="auto (recommended), validation, or validation_interactive. Explicit roles must match dataset metadata.")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--json", action="store_true", help="Print the full JSON record instead of only the WOMD spec.")
    args = ap.parse_args()
    try:
        doc = resolve_for_dataset(
            args.dataset, split=args.split, womd_root=args.womd_root, shards=args.shards, role=args.role
        )
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 30
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(doc, ensure_ascii=False))
    else:
        print(doc["womd_spec"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
