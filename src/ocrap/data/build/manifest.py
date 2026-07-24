from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from ocrap.data.serialization import write_json

from .builder import (
    MANIFEST_FIELDS,
    _dataset_output_lock,
    _is_complete_npz,
    _manifest_row_from_npz,
    _quarantine_invalid_existing_sample,
    _scene_time_key_from_sample_name,
    _write_manifest_atomic,
)


def _load_valid_completion_markers(
    dataset_root: Path,
    existing_names: set[str],
) -> tuple[dict[str, set[str]], bool, str]:
    """Load completion markers that match the dataset resume contract.

    Returns ``(markers_by_scene_time, audit_available, fingerprint)``.  The
    audit is available only when both a readable resume contract and marker file
    exist.  A marker is accepted only when its fingerprint matches and all of
    its declared sample files are present in the validated NPZ set.
    """
    contract_path = dataset_root / "resume_contract.json"
    marker_path = dataset_root / "resume_scene_time_done.jsonl"
    if not contract_path.exists() or not marker_path.exists():
        return {}, False, ""
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        fingerprint = str(contract.get("fingerprint", ""))
    except Exception:
        return {}, False, ""
    if not fingerprint:
        return {}, False, ""

    markers: dict[str, set[str]] = {}
    try:
        with marker_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if str(row.get("fingerprint", "")) != fingerprint:
                    continue
                key = str(row.get("key", ""))
                if not key:
                    continue
                names = {str(x) for x in row.get("sample_names", []) if str(x)}
                if names and not names.issubset(existing_names):
                    continue
                # Keep the latest valid marker for a key.  Resume may append the
                # same deterministic marker more than once after an interruption.
                markers[key] = names
    except Exception:
        return {}, False, fingerprint
    return markers, True, fingerprint


def _move_to_subdir(path: Path, subdir: Path) -> Path:
    subdir.mkdir(parents=True, exist_ok=True)
    target = subdir / path.name
    if target.exists():
        i = 1
        while True:
            candidate = subdir / f"{path.stem}.quarantine{i}{path.suffix}"
            if not candidate.exists():
                target = candidate
                break
            i += 1
    os.replace(path, target)
    return target


def rebuild_manifest(
    dataset: str | Path,
    *,
    require_complete: bool = False,
    quarantine_uncommitted: bool = False,
    quarantine_invalid: bool = False,
) -> dict[str, Any]:
    """Reconstruct ``manifest.csv`` from materialized OC-RAP NPZ samples.

    The generated columns and value extraction are exactly the same as the
    normal dataset builder.  When resume completion markers are available, the
    function also identifies scene-time groups that were not committed before a
    killed build.  Such files remain visible to current dataset readers unless
    they are resumed or explicitly quarantined.
    """
    root = Path(dataset).expanduser().resolve()
    sample_dir = root / "samples"
    if not sample_dir.is_dir():
        raise FileNotFoundError(f"Missing OC-RAP sample directory: {sample_dir}")

    with _dataset_output_lock(root):
        paths = sorted(sample_dir.glob("*.npz"))
        valid_paths: dict[str, Path] = {}
        invalid_paths: list[Path] = []
        for path in paths:
            if _is_complete_npz(path):
                valid_paths[path.name] = path
            else:
                invalid_paths.append(path)

        invalid_quarantined: list[str] = []
        if quarantine_invalid:
            for path in invalid_paths:
                if _quarantine_invalid_existing_sample(path, sample_dir):
                    invalid_quarantined.append(path.name)

        existing_names = set(valid_paths)
        markers, marker_audit_available, fingerprint = _load_valid_completion_markers(root, existing_names)
        groups: dict[str, list[Path]] = defaultdict(list)
        unparseable_names: list[str] = []
        for name, path in valid_paths.items():
            key = _scene_time_key_from_sample_name(name)
            if key is None:
                unparseable_names.append(name)
                continue
            groups[key].append(path)

        uncommitted_keys: list[str] = []
        if marker_audit_available:
            completed_keys = set(markers)
            uncommitted_keys = sorted(key for key in groups if key not in completed_keys)

        if quarantine_uncommitted and not marker_audit_available:
            raise RuntimeError(
                "Cannot quarantine uncommitted scene-times because no valid "
                "resume_contract.json + resume_scene_time_done.jsonl audit is available."
            )

        if require_complete and (invalid_paths or unparseable_names or uncommitted_keys):
            raise RuntimeError(
                "Dataset is not complete enough for a strict manifest rebuild: "
                f"invalid_npz={len(invalid_paths)}, "
                f"unparseable_names={len(unparseable_names)}, "
                f"uncommitted_scene_time_groups={len(uncommitted_keys)}"
            )

        quarantined_uncommitted: list[str] = []
        if quarantine_uncommitted:
            quarantine_dir = sample_dir / "incomplete_scene_times"
            for key in uncommitted_keys:
                for path in groups[key]:
                    _move_to_subdir(path, quarantine_dir)
                    quarantined_uncommitted.append(path.name)
                    valid_paths.pop(path.name, None)

        rows = [_manifest_row_from_npz(path, root) for path in sorted(valid_paths.values())]
        manifest_path = root / "manifest.csv"
        _write_manifest_atomic(manifest_path, rows)

        stale_tmp = sorted(p.name for p in sample_dir.glob(".*.tmp"))
        summary = {
            "dataset": str(root),
            "manifest": str(manifest_path),
            "manifest_fields": list(MANIFEST_FIELDS),
            "num_manifest_rows": int(len(rows)),
            "num_valid_npz_seen": int(len(existing_names)),
            "num_invalid_npz_seen": int(len(invalid_paths)),
            "invalid_npz_names": [p.name for p in invalid_paths],
            "invalid_npz_quarantined": invalid_quarantined,
            "completion_marker_audit_available": bool(marker_audit_available),
            "resume_fingerprint": fingerprint,
            "num_completed_scene_time_groups": int(len(markers)),
            "num_uncommitted_scene_time_groups": int(len(uncommitted_keys)),
            "uncommitted_scene_time_keys": uncommitted_keys,
            "uncommitted_npz_quarantined": quarantined_uncommitted,
            "unparseable_sample_names": unparseable_names,
            "stale_tmp_files": stale_tmp,
            "safe_for_direct_reader_scan": bool(
                (not invalid_paths or len(invalid_quarantined) == len(invalid_paths))
                and not unparseable_names
                and (not marker_audit_available or not uncommitted_keys or quarantine_uncommitted)
            ),
        }
        write_json(summary, root / "manifest_rebuild_summary.json", fsync=True)
        return summary
