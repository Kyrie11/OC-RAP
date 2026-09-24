#!/usr/bin/env python3
"""Merge OC-RAP dataset roots produced by scenario sharding.

Besides samples/manifest rows, this tool preserves WOMD replay provenance. Older
versions projected every manifest onto a short legacy field list, which silently
dropped ``womd_source_role`` and source-scenario identity fields. That made a
merged/filtered calibration bucket impossible to replay fail-closed later.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path
from typing import Any

# Keep the current builder schema first, while preserving any additional fields
# introduced by future builders after these canonical columns.
PREFERRED_MANIFEST_FIELDS = [
    "path",
    "scene_id",
    "original_scenario_id",
    "official_scenario_id",
    "legacy_scenario_id",
    "source_scenario_index",
    "scenario_id_source",
    "womd_source_role",
    "waymax_max_num_objects",
    "time_index",
    "candidate_index",
    "split_id",
    "is_nominal",
    "r_orc_star",
    "r_dep_star",
    "oracle_gap_star",
    "i_art_star",
    "regime_label",
]
_UNKNOWN = {"", "unknown", "none", "null", "auto"}


def _unique_target(sample_dir: Path, name: str) -> Path:
    target = sample_dir / name
    if not target.exists():
        return target
    stem, suffix = Path(name).stem, Path(name).suffix
    i = 1
    while True:
        candidate = sample_dir / f"{stem}__dup{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


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


def _role_from_source_text(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, (list, tuple, set)):
        text = ",".join(str(x) for x in value)
    else:
        text = str(value)
    low = text.strip().lower().replace("\\", "/")
    if not low:
        return "unknown"
    if "validation_interactive" in low or "validation-interactive" in low:
        return "validation_interactive"
    if re.search(r"(^|[/_])validation([/_]|$)", low):
        return "validation"
    if re.search(r"(^|[/_])training([/_]|$)", low):
        return "training"
    if re.search(r"(^|[/_])testing([/_]|$)", low):
        return "testing"
    return "unknown"


def _root_replay_provenance(root: Path, rows: list[dict[str, str]]) -> tuple[str, str | None, list[dict[str, Any]]]:
    """Return one conservative source role/pattern for a shard root.

    Per-row roles and the builder resume contract must agree whenever both are
    available. A resume-contract pattern may backfill legacy rows whose manifest
    has no role column, but conflicting evidence is never guessed through.
    """
    evidence: list[dict[str, Any]] = []
    row_roles = {
        _clean_role(row.get("womd_source_role"))
        for row in rows
        if _clean_role(row.get("womd_source_role")) not in _UNKNOWN
    }
    if len(row_roles) > 1:
        raise RuntimeError(f"input root mixes WOMD source roles: root={root}, roles={sorted(row_roles)}")
    if row_roles:
        evidence.append({"source": "manifest.csv:womd_source_role", "role": next(iter(row_roles))})

    pattern: str | None = None
    resume = _read_json(root / "resume_contract.json")
    if resume is not None:
        semantic = resume.get("semantic_config") if isinstance(resume.get("semantic_config"), dict) else {}
        raw_pattern = semantic.get("womd_patterns") or semantic.get("womd_pattern") or resume.get("womd_patterns") or resume.get("womd_pattern")
        role_raw = semantic.get("womd_source_role") or resume.get("womd_source_role")
        role = _clean_role(role_raw)
        if role in _UNKNOWN:
            role = _role_from_source_text(raw_pattern)
        if role not in _UNKNOWN:
            evidence.append({"source": "resume_contract.json", "role": role, "raw_pattern": raw_pattern})
        if isinstance(raw_pattern, str) and raw_pattern.strip():
            pattern = raw_pattern.strip()

    known = {str(x["role"]) for x in evidence if str(x.get("role", "")) not in _UNKNOWN}
    if len(known) > 1:
        raise RuntimeError(f"input root WOMD provenance conflicts: root={root}, evidence={evidence}")
    role = next(iter(known)) if known else "unknown"
    return role, pattern, evidence


def _write_replay_contract(output: Path, *, roles: set[str], patterns: set[str], inputs: list[Path], all_patterns_known: bool) -> None:
    # A mixed {validation, unknown} merge is *not* enough evidence to label the
    # unknown root as validation.  Emit a dataset-level contract only when every
    # input root independently resolves to the same known collection.
    if any(r in _UNKNOWN for r in roles) or len(roles) != 1:
        return
    known_roles = set(roles)
    doc: dict[str, Any] = {
        "schema": "ocrap_womd_replay_contract_v1",
        "womd_source_role": next(iter(known_roles)),
        "derived_from": [str(p.resolve()) for p in inputs],
    }
    if all_patterns_known and len(patterns) == 1:
        doc["womd_pattern"] = next(iter(patterns))
    (output / "womd_replay_contract.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def merge(inputs: list[Path], output: Path, *, copy: bool = True) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    sample_dir = output / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    all_fields: list[str] = []
    split_counts: dict[str, int] = {}
    scene_ids: set[str] = set()
    copied = 0
    input_roles: set[str] = set()
    input_patterns: set[str] = set()
    all_input_patterns_known = True
    provenance_evidence: list[dict[str, Any]] = []

    for root in inputs:
        manifest = root / "manifest.csv"
        if not manifest.exists():
            raise FileNotFoundError(f"missing manifest: {manifest}")
        with manifest.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            root_fields = list(reader.fieldnames or [])
            root_rows = list(reader)
        for field in root_fields:
            if field not in all_fields:
                all_fields.append(field)

        root_role, root_pattern, root_evidence = _root_replay_provenance(root, root_rows)
        input_roles.add(root_role)
        if root_pattern:
            input_patterns.add(root_pattern)
        else:
            all_input_patterns_known = False
        provenance_evidence.append({
            "root": str(root.resolve()), "resolved_role": root_role,
            "womd_pattern": root_pattern, "evidence": root_evidence,
        })

        for row in root_rows:
            rel = row.get("path", "")
            src = root / rel
            if not src.exists():
                raise FileNotFoundError(f"manifest sample missing: {src}")
            dst = _unique_target(sample_dir, src.name)
            if copy:
                shutil.copy2(src, dst)
            else:
                try:
                    dst.hardlink_to(src)
                except Exception:
                    shutil.copy2(src, dst)
            out_row = dict(row)
            # Backfill provenance from the authoritative shard resume contract
            # only when the legacy row itself carries no known role.
            if _clean_role(out_row.get("womd_source_role")) in _UNKNOWN and root_role not in _UNKNOWN:
                out_row["womd_source_role"] = root_role
            out_row["path"] = f"samples/{dst.name}"
            rows.append(out_row)
            copied += 1
            split = str(out_row.get("split_id", ""))
            split_counts[split] = split_counts.get(split, 0) + 1
            if out_row.get("scene_id"):
                scene_ids.add(str(out_row["scene_id"]))

    output_fields = list(PREFERRED_MANIFEST_FIELDS)
    for field in all_fields:
        if field not in output_fields:
            output_fields.append(field)
    manifest_out = output / "manifest.csv"
    with manifest_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    homogeneous_known_role = len(input_roles) == 1 and not any(r in _UNKNOWN for r in input_roles)
    known_roles = set(input_roles) if homogeneous_known_role else set()
    summary: dict[str, Any] = {
        "num_samples": len(rows),
        "num_input_roots": len(inputs),
        "input_roots": [str(p) for p in inputs],
        "sample_dir": str(sample_dir),
        "manifest": str(manifest_out),
        "split_counts": split_counts,
        "unique_scene_ids": len(scene_ids),
        "womd_source_role": next(iter(known_roles)) if len(known_roles) == 1 else "unknown",
        "source_womd_pattern": next(iter(input_patterns)) if all_input_patterns_known and len(input_patterns) == 1 else None,
        "womd_provenance_evidence": provenance_evidence,
    }
    (output / "merged_dataset_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_replay_contract(
        output, roles=input_roles, patterns=input_patterns, inputs=inputs,
        all_patterns_known=all_input_patterns_known,
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--hardlink", action="store_true", help="Try hardlinks before falling back to copy2.")
    args = ap.parse_args()
    print(json.dumps(merge([Path(x).expanduser() for x in args.inputs], Path(args.output).expanduser(), copy=not args.hardlink), indent=2))


if __name__ == "__main__":
    main()
