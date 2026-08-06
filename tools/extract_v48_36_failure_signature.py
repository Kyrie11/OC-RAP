#!/usr/bin/env python3
"""Extract a versioned v48.36 Python/shell failure signature from a stage log."""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--stage", default="unknown")
    ap.add_argument("--exit-code", type=int, required=True)
    ap.add_argument("--implementation-version", default="v48.36.2-STAGE-TRANSFER-HOTFIX")
    args = ap.parse_args()
    text = args.log.read_text(errors="replace") if args.log.is_file() else ""
    lines = text.splitlines()
    exception_type = None
    message = None
    location = None
    pattern = re.compile(r"^([A-Za-z_][\w.]*(?:Error|Exception|Interrupt)):\s*(.*)$")
    for line in reversed(lines):
        match = pattern.match(line.strip())
        if match:
            exception_type, message = match.group(1), match.group(2)
            break
    frame = re.compile(r'^\s*File "([^"]+)", line (\d+), in (.+)$')
    for line in reversed(lines):
        match = frame.match(line)
        if match:
            location = {
                "file": match.group(1),
                "line": int(match.group(2)),
                "function": match.group(3),
            }
            break
    doc = {
        "event": "v48_36_failure_signature",
        "version": "v48.36-OCAF",
        "implementation_version": args.implementation_version,
        "created_unix": time.time(),
        "stage": args.stage,
        "exit_code": args.exit_code,
        "log": str(args.log),
        "exception_type": exception_type,
        "message": message,
        "location": location,
        "tail": "\n".join(lines[-120:]),
        "test_roots_read": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_name(
        f".{args.output.name}.tmp.{os.getpid()}.{time.time_ns()}"
    )
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, args.output)
    print(json.dumps(doc, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
