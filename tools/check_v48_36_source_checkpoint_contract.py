from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="Fail-closed source checkpoint preflight for v48.x adaptation.")
    ap.add_argument("--source-run", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--variants", default="balanced,precision")
    args = ap.parse_args()

    source_input = args.source_run
    source = Path(source_input).expanduser()
    if not source.is_absolute():
        source = (Path.cwd() / source).resolve(strict=False)
    else:
        source = source.resolve(strict=False)

    variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    checks: dict[str, dict[str, object]] = {}
    valid = source.is_dir()
    for variant in variants:
        ckpt = source / "candidates" / variant / "model_v48_trac_sr" / "best.pt"
        exists = ckpt.is_file()
        readable = exists and os.access(ckpt, os.R_OK)
        size = ckpt.stat().st_size if exists else 0
        nonempty = size > 0
        row: dict[str, object] = {
            "checkpoint": str(ckpt),
            "exists": exists,
            "readable": readable,
            "size_bytes": size,
            "nonempty": nonempty,
        }
        if exists and readable and nonempty:
            row["sha256"] = sha256_file(ckpt)
        checks[variant] = row
        valid = valid and exists and readable and nonempty

    doc = {
        "event": "v48_36_source_checkpoint_contract",
        "created_unix": time.time(),
        "source_run_input": source_input,
        "source_run_resolved": str(source),
        "source_run_exists": source.is_dir(),
        "working_directory": str(Path.cwd()),
        "variants": variants,
        "checks": checks,
        "valid": bool(valid),
        "test_roots_read": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f".{output.name}.tmp.{os.getpid()}.{time.time_ns()}")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, output)
    print(json.dumps(doc, ensure_ascii=False))
    return 0 if valid else 3


if __name__ == "__main__":
    raise SystemExit(main())
