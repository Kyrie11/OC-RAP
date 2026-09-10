#!/usr/bin/env python3
"""Resolve the raw WOMD replay spec from OC-RAP dataset provenance.

Newer OC-RAP datasets persist ``womd_source_role`` per sample/manifest row.
Historical val/test builds may predate that field even though their dataset-level
``resume_contract.json`` still records the original ``semantic_config`` and its
``womd_patterns``.  Replay resolution therefore uses a fail-closed evidence
hierarchy instead of assuming every NPZ contains the newest metadata:

1. homogeneous per-sample/manifest ``womd_source_role`` (strongest);
2. dataset ``resume_contract.semantic_config.womd_patterns`` / explicit role;
3. a user-supplied explicit ``--role`` only when no stored evidence conflicts.

Example root layout::

    <womd_root>/validation/validation_tfexample.tfrecord-00000-of-00150
    <womd_root>/validation_interactive/validation_interactive_tfexample.tfrecord-00000-of-00150

The printed stdout is the normalized TensorFlow sharded spec so shell launchers
can use command substitution safely.  ``--json`` exposes the evidence used.
"""
from __future__ import annotations

import argparse
import json
import re
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
_UNKNOWN_ROLES = {"", "unknown", "none", "null", "auto"}


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


def _role_from_source_text(value: Any) -> str:
    """Infer a WOMD collection role from a stored path/spec without opening it."""
    if value is None:
        return "unknown"
    if isinstance(value, (list, tuple, set)):
        text = ",".join(str(x) for x in value)
    else:
        text = str(value)
    low = text.strip().lower().replace("\\", "/")
    if not low:
        return "unknown"
    # Check the more specific collection before standard validation.
    if "validation_interactive" in low or "validation-interactive" in low:
        return "validation_interactive"
    if re.search(r"(^|[/_])validation([/_]|$)", low):
        return "validation"
    if re.search(r"(^|[/_])training([/_]|$)", low):
        return "training"
    if re.search(r"(^|[/_])testing([/_]|$)", low):
        return "testing"
    return "unknown"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _dataset_level_role_evidence(dataset: str | Path) -> list[dict[str, Any]]:
    """Collect conservative dataset-level replay-role evidence.

    ``resume_contract.semantic_config.womd_patterns`` is authoritative for a
    normal (non-adopted) resumable build because that semantic config is part of
    the dataset fingerprint.  For an adopted legacy contract it is still useful
    evidence, but the audit record marks it as adopted so callers/reviewers can
    see the weaker historical provenance.
    """
    root = Path(dataset).expanduser().resolve()
    out: list[dict[str, Any]] = []

    resume_path = root / "resume_contract.json"
    resume = _read_json(resume_path)
    if resume is not None:
        semantic = resume.get("semantic_config") if isinstance(resume.get("semantic_config"), dict) else {}
        candidates = [
            ("semantic_config.womd_source_role", semantic.get("womd_source_role"), "role"),
            ("semantic_config.womd_patterns", semantic.get("womd_patterns"), "pattern"),
            ("semantic_config.womd_pattern", semantic.get("womd_pattern"), "pattern"),
            ("womd_source_role", resume.get("womd_source_role"), "role"),
            ("womd_patterns", resume.get("womd_patterns"), "pattern"),
            ("womd_pattern", resume.get("womd_pattern"), "pattern"),
        ]
        for key, raw, kind in candidates:
            role = _clean_role(raw) if kind == "role" else _role_from_source_text(raw)
            if role in _UNKNOWN_ROLES:
                continue
            out.append({
                "source": f"resume_contract.json:{key}",
                "role": role,
                "raw": raw,
                "adopted_legacy": bool(resume.get("adopted_legacy", False)),
                "generator_version": resume.get("generator_version"),
                "fingerprint": resume.get("fingerprint"),
            })

    # Some migrated datasets may have a dataset-level source field even if the
    # sample manifest was created before womd_source_role was added.  Only exact
    # source keys are considered; unrelated output paths are intentionally not
    # searched recursively because names such as ``test_near_contact`` do not
    # prove a raw WOMD collection.
    for filename in ("dataset_summary.json", "dataset_status.json"):
        path = root / filename
        doc = _read_json(path)
        if doc is None:
            continue
        for key, kind in (
            ("womd_source_role", "role"),
            ("womd_patterns", "pattern"),
            ("womd_pattern", "pattern"),
            ("raw_womd_pattern", "pattern"),
            ("source_womd_pattern", "pattern"),
        ):
            raw = doc.get(key)
            role = _clean_role(raw) if kind == "role" else _role_from_source_text(raw)
            if role in _UNKNOWN_ROLES:
                continue
            out.append({"source": f"{filename}:{key}", "role": role, "raw": raw})
    return out


def infer_dataset_source_provenance(dataset: str | Path, split: str = "") -> dict[str, Any]:
    """Infer one replay role and expose all evidence used.

    Unknown per-row metadata is acceptable for historical datasets.  Conflicting
    *known* evidence is never silently resolved.
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

    row_roles = {r for r, n in counts.items() if n > 0 and r not in _UNKNOWN_ROLES}
    if len(row_roles) > 1:
        raise RuntimeError(
            f"dataset mixes multiple per-sample raw WOMD source roles: dataset={dataset!s}, "
            f"split={split!r}, counts={dict(counts)}"
        )

    dataset_evidence = _dataset_level_role_evidence(dataset)
    dataset_roles = {str(e["role"]) for e in dataset_evidence if str(e.get("role", "")) not in _UNKNOWN_ROLES}
    if len(dataset_roles) > 1:
        raise RuntimeError(
            "dataset-level provenance files disagree on the raw WOMD collection: "
            f"dataset={dataset!s}, evidence={dataset_evidence}"
        )

    row_role = next(iter(row_roles)) if row_roles else None
    dataset_role = next(iter(dataset_roles)) if dataset_roles else None
    if row_role and dataset_role and row_role != dataset_role:
        raise RuntimeError(
            "per-sample WOMD source role disagrees with dataset-level provenance: "
            f"dataset={dataset!s}, sample_role={row_role!r}, dataset_role={dataset_role!r}, "
            f"counts={dict(counts)}, evidence={dataset_evidence}"
        )

    resolved = row_role or dataset_role or "unknown"
    if row_role:
        source = "sample_or_manifest_womd_source_role"
    elif dataset_role:
        source = str(dataset_evidence[0].get("source", "dataset_level_provenance"))
    else:
        source = "unresolved"
    return {
        "role": resolved,
        "source": source,
        "role_counts": dict(sorted(counts.items())),
        "num_dataset_rows": matched,
        "dataset_level_evidence": dataset_evidence,
    }


def infer_dataset_source_role(dataset: str | Path, split: str = "") -> tuple[str, Counter[str], int]:
    """Backward-compatible role API with legacy dataset-level fallback."""
    prov = infer_dataset_source_provenance(dataset, split)
    role = str(prov["role"])
    counts = Counter({str(k): int(v) for k, v in prov["role_counts"].items()})
    matched = int(prov["num_dataset_rows"])
    if role in _UNKNOWN_ROLES:
        raise RuntimeError(
            "dataset has no usable WOMD replay-role provenance in sample/manifest metadata or dataset-level "
            f"contracts: dataset={dataset!s}, split={split!r}, counts={dict(counts)}. "
            "Expected resume_contract.json:semantic_config.womd_patterns for legacy datasets. "
            "If the original collection is known independently, pass --role validation or "
            "--role validation_interactive; the explicit role is accepted only when no stored evidence conflicts."
        )
    return role, counts, matched


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
    provenance = infer_dataset_source_provenance(dataset, split)
    inferred = str(provenance["role"])
    requested = _clean_role(role)

    if requested in {"", "auto"}:
        if inferred in _UNKNOWN_ROLES:
            counts = provenance["role_counts"]
            raise RuntimeError(
                "dataset has no usable WOMD replay-role provenance in sample/manifest metadata or dataset-level "
                f"contracts: dataset={dataset!s}, split={split!r}, counts={counts}, "
                f"dataset_level_evidence={provenance['dataset_level_evidence']}. "
                "For this legacy dataset inspect resume_contract.json:semantic_config.womd_patterns. "
                "If that field is absent but you independently know the original collection, rerun with "
                "--role validation or --role validation_interactive."
            )
        chosen = inferred
        explicit_legacy_role = False
    else:
        if requested not in ROLE_TO_PREFIX:
            raise ValueError(f"unsupported explicit replay role {requested!r}; supported={sorted(ROLE_TO_PREFIX)}")
        if inferred not in _UNKNOWN_ROLES and requested != inferred:
            raise RuntimeError(
                f"explicit replay role {requested!r} disagrees with dataset provenance {inferred!r} "
                f"from {provenance['source']}; use --role auto or replay the correct collection"
            )
        chosen = requested
        explicit_legacy_role = inferred in _UNKNOWN_ROLES

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
        "dataset_source_role_source": provenance["source"],
        "dataset_level_evidence": provenance["dataset_level_evidence"],
        "requested_role": role,
        "resolved_role": chosen,
        "explicit_role_for_unprovenanced_legacy_dataset": bool(explicit_legacy_role),
        "role_counts": provenance["role_counts"],
        "num_dataset_rows": int(provenance["num_dataset_rows"]),
        "womd_spec": spec,
        "num_resolved_womd_files": len(resolved.files),
        "womd_shard_spec": resolved.as_dict(),
        "valid": True,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument(
        "--womd-root",
        required=True,
        help="Directory containing validation/ and validation_interactive/ (the tf_example root).",
    )
    ap.add_argument("--shards", type=int, default=150)
    ap.add_argument(
        "--role",
        default="auto",
        help=(
            "auto (recommended), validation, or validation_interactive. Explicit roles must not conflict "
            "with stored provenance; they can be used to declare the source of a legacy dataset whose old "
            "samples and contracts contain no collection metadata."
        ),
    )
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
