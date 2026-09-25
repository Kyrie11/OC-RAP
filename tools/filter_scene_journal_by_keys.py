#!/usr/bin/env python3
"""Filter a scene JSONL journal to an exact target-key set, preserving key order.

Used by Contact target-display reference mode so the synthesized OC-RAP journal
and all paired baseline journals contain exactly the same target set. The tool
fails closed on missing or duplicate keys.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _key(scene: dict[str, Any], env: dict[str, Any]) -> str:
    key = str(scene.get("target_key") or env.get("resume_key") or "")
    if key.startswith("target:"):
        key = key[len("target:"):]
    if key:
        return key
    sid = str(scene.get("scene_id") or "")
    ti = scene.get("target_time_index")
    return f"{sid}:t{ti}" if sid and ti is not None else sid


def _load_keys(path: Path) -> list[str]:
    d = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(d, list):
        rows = d
    elif isinstance(d, dict):
        rows = d.get("target_keys") or d.get("selected") or d.get("anchors") or []
    else:
        rows = []
    out: list[str] = []
    for row in rows:
        if isinstance(row, str):
            k = row
        elif isinstance(row, dict):
            k = row.get("target_key") or row.get("key")
        else:
            k = None
        if k:
            out.append(str(k))
    if not out:
        raise SystemExit(f"no target keys in {path}")
    if len(out) != len(set(out)):
        raise SystemExit(f"duplicate target keys in {path}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--target-keys", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--audit-output", type=Path, default=None)
    args = ap.parse_args()

    wanted = _load_keys(args.target_keys)
    wanted_set = set(wanted)
    envs: dict[str, dict[str, Any]] = {}
    with args.input.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            env = json.loads(line)
            scene = env.get("scene", env)
            if not isinstance(scene, dict):
                continue
            k = _key(scene, env)
            if k not in wanted_set:
                continue
            if k in envs:
                raise SystemExit(f"duplicate target {k} in {args.input}")
            envs[k] = env

    missing = [k for k in wanted if k not in envs]
    if missing:
        raise SystemExit(f"missing {len(missing)} requested targets in {args.input}: {missing[:8]}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for k in wanted:
            f.write(json.dumps(envs[k], ensure_ascii=False, separators=(",", ":")) + "\n")

    audit = {
        "event": "scene_journal_exact_target_filter_v1",
        "input": str(args.input),
        "target_keys": str(args.target_keys),
        "output": str(args.output),
        "num_targets": len(wanted),
        "target_set_exact": True,
    }
    if args.audit_output:
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        args.audit_output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
