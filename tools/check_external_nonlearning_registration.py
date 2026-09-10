#!/usr/bin/env python3
from __future__ import annotations

"""Fast validity check for already-registered non-learning external baselines.

The expensive registrar groups the full regime train/val manifests.  Once its
per-method summaries exist, this checker validates their immutable dataset and
configuration contract without rescanning those datasets.
"""

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from ocrap.config import load_config


def _parse_specs(text: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for raw in str(text).split(","):
        raw = raw.strip()
        if not raw:
            continue
        if "=" not in raw:
            raise ValueError(f"invalid spec {raw!r}; expected method=config.yaml")
        method, config = (x.strip() for x in raw.split("=", 1))
        if not method or not config:
            raise ValueError(f"invalid spec {raw!r}")
        out.append((method, config))
    if not out:
        raise ValueError("no method=config specs")
    return out


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _expected_cfg(config_path: str, method: str) -> dict[str, Any]:
    cfg = copy.deepcopy(load_config(config_path))
    cfg.setdefault("external_baselines", {})["baseline"] = method
    return cfg


def check_registration(*, root: Path, dataset: str, val_dataset: str | None, specs: list[tuple[str, str]]) -> dict[str, Any]:
    errors: list[str] = []
    methods: dict[str, Any] = {}
    for method, config in specs:
        path = root / method / "train_summary.json"
        row: dict[str, Any] = {"summary": str(path), "valid": False}
        if not path.is_file():
            errors.append(f"{method}: missing {path}")
            methods[method] = row
            continue
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            expected_cfg = _expected_cfg(config, method)
            checks = {
                "baseline": doc.get("baseline") == method,
                "mode": doc.get("training_mode") == "non_learning_filter_or_planner",
                "dataset_validated": doc.get("dataset_validated") is True,
                "train_dataset": str(doc.get("train_dataset")) == str(dataset),
                "val_dataset": str(doc.get("val_dataset")) == (str(val_dataset) if val_dataset else "None"),
                "config": _canonical(doc.get("cfg")) == _canonical(expected_cfg),
            }
            row["checks"] = checks
            row["valid"] = all(checks.values())
            if not row["valid"]:
                errors.append(f"{method}: registration contract mismatch: {[k for k,v in checks.items() if not v]}")
        except Exception as exc:
            errors.append(f"{method}: {type(exc).__name__}: {exc}")
        methods[method] = row
    return {"valid": not errors, "errors": errors, "methods": methods}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--val-dataset", default=None)
    ap.add_argument("--specs", required=True, help="comma-separated method=config.yaml")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    doc = check_registration(root=args.root, dataset=args.dataset, val_dataset=args.val_dataset, specs=_parse_specs(args.specs))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not doc["valid"]:
        for e in doc["errors"]:
            print(f"[INVALID] {e}")
        raise SystemExit(1)
    print(json.dumps({"valid": True, "methods": sorted(doc["methods"])}))


if __name__ == "__main__":
    main()
