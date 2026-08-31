#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def _same_path(a: str | None, b: str | None) -> bool:
    if a in {None, ""} or b in {None, ""}:
        return a in {None, ""} and b in {None, ""}
    try:
        return Path(str(a)).expanduser().resolve() == Path(str(b)).expanduser().resolve()
    except Exception:
        return str(a) == str(b)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--method", default=None)
    ap.add_argument("--bucket-dataset", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--dependency", action="append", default=[])
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    p = args.output
    prog = read(p.with_suffix(p.suffix + ".progress.json"))
    result = read(p)
    journal = p.with_suffix(p.suffix + ".scenes.jsonl")
    errors: list[str] = []

    complete = bool(result and prog and prog.get("status") == "complete" and journal.is_file())
    if not complete:
        errors.append("missing_result_progress_or_journal")
    if complete and result.get("bucket_target_count") not in (None, 0):
        complete = int(result.get("num_scenes") or 0) == int(result.get("bucket_target_count") or 0)
        if not complete:
            errors.append("scene_count_does_not_match_bucket_target_count")
    if result and prog:
        result_fp = str(result.get("run_fingerprint", "") or "")
        progress_fp = str(prog.get("run_fingerprint", "") or "")
        if result_fp and progress_fp and result_fp != progress_fp:
            complete = False
            errors.append("result_progress_fingerprint_mismatch")
    if result and args.method is not None and str(result.get("method", "")).lower() != str(args.method).lower():
        complete = False
        errors.append("method_mismatch")
    if result and args.bucket_dataset is not None and not _same_path(result.get("bucket_dataset"), args.bucket_dataset):
        complete = False
        errors.append("bucket_dataset_mismatch")

    dependencies = [Path(x) for x in args.dependency]
    if args.checkpoint:
        dependencies.append(Path(args.checkpoint))
    if p.is_file():
        try:
            out_ns = p.stat().st_mtime_ns
            for dep in dependencies:
                if not dep.exists():
                    complete = False
                    errors.append(f"missing_dependency:{dep}")
                elif dep.stat().st_mtime_ns > out_ns:
                    complete = False
                    errors.append(f"dependency_newer_than_output:{dep}")
        except OSError as exc:
            complete = False
            errors.append(f"freshness_check_failed:{exc}")

    doc = {
        "event": "closed_loop_artifact_check",
        "output": str(p),
        "complete": complete,
        "result_exists": p.is_file(),
        "journal_exists": journal.is_file(),
        "progress_status": prog.get("status") if prog else None,
        "num_scenes": result.get("num_scenes") if result else None,
        "bucket_target_count": result.get("bucket_target_count") if result else None,
        "run_fingerprint": result.get("run_fingerprint") if result else None,
        "errors": errors,
    }
    if not args.quiet:
        print(json.dumps(doc, ensure_ascii=False))
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
