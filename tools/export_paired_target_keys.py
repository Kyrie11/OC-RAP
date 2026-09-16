#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _journal(path: Path) -> Path | None:
    p = Path(str(path) + ".scenes.jsonl")
    return p if p.is_file() else None


def _keys(path: Path) -> set[str]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    out: set[str] = set()
    for scene in doc.get("scenes") or []:
        key = str(scene.get("target_key") or "")
        if key:
            out.add(key)
    if out:
        return out
    journal = _journal(path)
    if journal is None:
        raise SystemExit(f"no embedded scenes or scene journal: {path}")
    for line in journal.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        scene: dict[str, Any] = row.get("scene", row)
        key = str(scene.get("target_key") or row.get("resume_key") or "")
        if key:
            out.add(key)
    if not out:
        raise SystemExit(f"no target keys: {path}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Freeze the exact common target set from completed paired closed-loop results.")
    ap.add_argument("--input", action="append", required=True, metavar="LABEL=RESULT.json")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    rows: dict[str, set[str]] = {}
    for spec in args.input:
        if "=" not in spec:
            raise SystemExit(f"invalid --input {spec!r}")
        label, raw = spec.split("=", 1)
        p = Path(raw)
        if not p.is_file():
            raise SystemExit(f"missing result: {p}")
        rows[label] = _keys(p)
    labels = list(rows)
    ref = rows[labels[0]]
    mismatch = {
        label: {"only_reference": sorted(ref - keys)[:20], "only_method": sorted(keys - ref)[:20]}
        for label, keys in rows.items() if keys != ref
    }
    if mismatch:
        raise SystemExit("unpaired target sets: " + json.dumps(mismatch, ensure_ascii=False))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out = {"schema": "ocrap-final-paired-target-keys-v1", "target_keys": sorted(ref), "num_target_keys": len(ref), "sources": labels}
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "num_target_keys": len(ref), "sources": labels}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
