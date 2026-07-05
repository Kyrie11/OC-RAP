#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def gamma(path: str) -> float:
    data = json.loads(Path(path).read_text())
    return float(data["gamma_rec"])


def main() -> None:
    ap = argparse.ArgumentParser(description="Create gamma_rec_by_bucket JSON from per-regime calibration files.")
    ap.add_argument("--safe", required=True)
    ap.add_argument("--near", required=True)
    ap.add_argument("--contact", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    mapping = {
        "test_safe": gamma(args.safe),
        "safe": gamma(args.safe),
        "test_near_contact": gamma(args.near),
        "near_contact": gamma(args.near),
        "test_contact": gamma(args.contact),
        "contact": gamma(args.contact),
    }
    out = {"gamma_rec_by_bucket": mapping}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
