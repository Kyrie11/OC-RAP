#!/usr/bin/env python3
"""Resolve historical qualitative-selection keys against a rebuilt OC-RAP bucket.

Metric journals may contain legacy ``waymax_<hash>`` target keys while rebuilt
bucket samples use a different canonical scene id.  The ``__wx########``
source-scenario suffix and target time are stable provenance fields and can be
used as an explicit migration key, provided the mapping is unique.

The tool never guesses: every selected item must resolve uniquely by one of
these methods, in priority order:
  1. exact current target key;
  2. canonical scene alias + target time;
  3. source_scenario_index + target time.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from ocrap.models.data import iter_sample_paths_many, scalar_metadata_for_path

_SOURCE_INDEX_RE = re.compile(r"__wx(?P<index>[0-9]+)(?:$|[^0-9])")
_TARGET_KEY_RE = re.compile(r"^(?P<bucket>[^:]+):(?P<scene>.+):t(?P<time>-?[0-9]+)$")


def _canonical_scene_id(value: Any) -> str:
    return re.sub(r"__wx\d{8}$", "", str(value or "").strip())


def _source_index(*values: Any) -> int:
    for value in values:
        match = _SOURCE_INDEX_RE.search(str(value or ""))
        if match:
            return int(match.group("index"))
    return -1


def _dataset_label(path: Path) -> str:
    return path.parent.parent.name if path.parent.name == "samples" else path.parent.name


def _int_or_none(value: Any) -> int | None:
    try:
        return int(float(value))
    except Exception:
        return None


def _selection_items(doc: Any) -> list[dict[str, Any]]:
    if isinstance(doc, dict) and isinstance(doc.get("selected"), list):
        return [dict(x) for x in doc["selected"] if isinstance(x, dict)]
    raise ValueError("selection must be a JSON document with a 'selected' list")


def _target_key_parts(value: Any) -> tuple[str, str, int | None]:
    text = str(value or "").strip()
    if text.startswith("target:"):
        text = text[len("target:"):]
    match = _TARGET_KEY_RE.match(text)
    if not match:
        return "", "", None
    return match.group("bucket"), match.group("scene"), int(match.group("time"))


def _iter_unique_targets(dataset: str, split_filter: str) -> Iterable[dict[str, Any]]:
    seen: set[str] = set()
    for raw_path in iter_sample_paths_many(dataset):
        path = Path(raw_path)
        split = str(scalar_metadata_for_path(path, "split_id", "") or "")
        if split_filter and split != split_filter:
            continue
        scene = str(scalar_metadata_for_path(path, "scene_id", "") or "").strip()
        original = str(scalar_metadata_for_path(path, "original_scenario_id", "") or "").strip()
        official = str(scalar_metadata_for_path(path, "official_scenario_id", "") or "").strip()
        legacy = str(scalar_metadata_for_path(path, "legacy_scenario_id", "") or "").strip()
        canonical = _canonical_scene_id(official or original or scene)
        time_index = _int_or_none(
            scalar_metadata_for_path(
                path, "time_index", scalar_metadata_for_path(path, "target_time_index", None)
            )
        )
        if not canonical or time_index is None:
            continue
        source_raw = _int_or_none(scalar_metadata_for_path(path, "source_scenario_index", -1))
        source = int(source_raw) if source_raw is not None else -1
        if source < 0:
            source = _source_index(original, scene, legacy)
        bucket = _dataset_label(path)
        key = f"{bucket}:{canonical}:t{time_index}"
        if key in seen:
            continue
        seen.add(key)
        aliases = {
            _canonical_scene_id(v)
            for v in (scene, original, official, legacy)
            if str(v or "").strip()
        }
        aliases.discard("")
        yield {
            "target_key": key,
            "bucket": bucket,
            "scene_id": canonical,
            "saved_scene_id": scene,
            "original_scenario_id": original or None,
            "official_scenario_id": official or None,
            "legacy_scenario_id": legacy or None,
            "scene_aliases": sorted(aliases),
            "source_scenario_index": source,
            "target_time_index": time_index,
            "sample": str(path),
        }


def _unique(candidates: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {str(row["target_key"]): row for row in candidates}
    return [by_key[key] for key in sorted(by_key)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--category", action="append", default=[], help="Resolve only selected rows in these categories; repeatable")
    ap.add_argument("--max-items", type=int, default=0, help="Optional deterministic prefix limit after category filtering")
    ap.add_argument("--target-keys-output", type=Path, required=True)
    ap.add_argument("--selection-output", type=Path, required=True)
    ap.add_argument("--report-output", type=Path, required=True)
    args = ap.parse_args()

    selection_doc = json.loads(args.selection.read_text(encoding="utf-8"))
    source_items = _selection_items(selection_doc)
    categories = {str(x) for x in args.category if str(x)}
    items = [item for item in source_items if not categories or str(item.get("category")) in categories]
    if args.max_items > 0:
        items = items[:args.max_items]
    if not items:
        raise SystemExit("selection contains no selected items after category/limit filtering")

    targets = list(_iter_unique_targets(args.dataset, args.split))
    by_key = {str(x["target_key"]): x for x in targets}
    by_alias_time: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    by_source_time: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for target in targets:
        time_index = int(target["target_time_index"])
        for alias in target["scene_aliases"]:
            by_alias_time[(str(alias), time_index)].append(target)
        source = int(target["source_scenario_index"])
        if source >= 0:
            by_source_time[(source, time_index)].append(target)

    resolved_items: list[dict[str, Any]] = []
    resolutions: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for item in items:
        old_key = str(item.get("target_key") or "").strip()
        old_bucket, key_scene, key_time = _target_key_parts(old_key)
        time_index = _int_or_none(item.get("target_time_index"))
        if time_index is None:
            time_index = key_time
        selected_scene = str(item.get("scene_id") or "").strip()
        source = _int_or_none(item.get("source_scenario_index"))
        if source is None or source < 0:
            source = _source_index(selected_scene, key_scene, old_key)

        method = ""
        candidates: list[dict[str, Any]] = []
        if old_key in by_key:
            method = "exact_target_key"
            candidates = [by_key[old_key]]
        if not candidates and time_index is not None:
            aliases = {
                _canonical_scene_id(v)
                for v in (selected_scene, key_scene)
                if str(v or "").strip()
            }
            aliases.discard("")
            alias_candidates: list[dict[str, Any]] = []
            for alias in sorted(aliases):
                alias_candidates.extend(by_alias_time.get((alias, int(time_index)), []))
            candidates = _unique(alias_candidates)
            if candidates:
                method = "scene_alias_and_time"
        if not candidates and source is not None and source >= 0 and time_index is not None:
            candidates = _unique(by_source_time.get((int(source), int(time_index)), []))
            if candidates:
                method = "source_scenario_index_and_time"

        resolution = {
            "source_target_key": old_key,
            "source_scene_id": selected_scene or None,
            "source_bucket": old_bucket or None,
            "source_scenario_index": source if source is not None and source >= 0 else None,
            "target_time_index": time_index,
            "resolution_method": method or None,
            "num_candidates": len(candidates),
            "candidate_target_keys": [x["target_key"] for x in candidates[:20]],
        }
        if len(candidates) != 1:
            resolution["error"] = "no_unique_mapping" if candidates else "no_mapping"
            failures.append(resolution)
            continue

        target = candidates[0]
        new_item = dict(item)
        new_item["source_target_key"] = old_key
        new_item["source_scene_id"] = selected_scene or None
        new_item["target_key"] = target["target_key"]
        new_item["scene_id"] = target["scene_id"]
        new_item["saved_scene_id"] = target["saved_scene_id"]
        new_item["source_scenario_index"] = target["source_scenario_index"]
        new_item["target_time_index"] = target["target_time_index"]
        new_item["target_resolution_method"] = method
        resolved_items.append(new_item)
        resolutions.append({**resolution, "resolved_target_key": target["target_key"], "resolved_scene_id": target["scene_id"]})

    duplicate_keys = sorted({x["target_key"] for x in resolved_items if sum(y["target_key"] == x["target_key"] for y in resolved_items) > 1})
    report = {
        "event": "v50_selected_target_resolution",
        "dataset": args.dataset,
        "split": args.split,
        "selection": str(args.selection),
        "num_dataset_targets": len(targets),
        "num_source_selected": len(source_items),
        "category_filter": sorted(categories),
        "max_items": args.max_items,
        "num_requested": len(items),
        "num_resolved": len(resolved_items),
        "num_failed": len(failures),
        "duplicate_resolved_target_keys": duplicate_keys,
        "resolutions": resolutions,
        "failures": failures,
        "valid": len(resolved_items) == len(items) and not failures and not duplicate_keys,
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not report["valid"]:
        print(json.dumps(report, ensure_ascii=False))
        return 3

    resolved_doc = dict(selection_doc)
    resolved_doc["source_selection"] = str(args.selection)
    resolved_doc["target_resolution"] = {
        "dataset": args.dataset,
        "split": args.split,
        "report": str(args.report_output),
        "method_priority": ["exact_target_key", "scene_alias_and_time", "source_scenario_index_and_time"],
    }
    resolved_doc["selection_filter"] = {"categories": sorted(categories), "max_items": args.max_items}
    resolved_doc["selected"] = resolved_items
    resolved_doc["target_keys"] = [x["target_key"] for x in resolved_items]
    args.selection_output.parent.mkdir(parents=True, exist_ok=True)
    args.selection_output.write_text(json.dumps(resolved_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.target_keys_output.parent.mkdir(parents=True, exist_ok=True)
    args.target_keys_output.write_text(
        json.dumps({"regime": selection_doc.get("regime"), "target_keys": resolved_doc["target_keys"], "source_selection": str(args.selection)}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"event": report["event"], "num_resolved": len(resolved_items), "target_keys_output": str(args.target_keys_output), "selection_output": str(args.selection_output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
