#!/usr/bin/env python3
from __future__ import annotations

"""Validate one regime dataset once and register multiple non-learning baselines.

Optimization/filter/controller baselines do not fit neural weights.  Historically
we invoked ``train-baseline`` once per method, repeating the same manifest scan
and dataset construction.  This tool preserves each method's ``train_summary``
contract while performing the regime grouping/dataset validation once.
"""

import argparse
import copy
from pathlib import Path

from ocrap.config import load_config
from ocrap.data.serialization import ensure_dir, write_json
from ocrap.external_baselines.data import (
    group_sample_paths,
    load_external_sample,
    sample_to_feature,
)


def _methods(text: str) -> list[str]:
    out = [x.strip() for x in str(text).split(",") if x.strip()]
    if not out:
        raise ValueError("--baselines must contain at least one method")
    return out


def _specs(text: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for raw in str(text).split(","):
        raw = raw.strip()
        if not raw:
            continue
        if "=" not in raw:
            raise ValueError(f"Invalid --specs entry {raw!r}; expected method=config.yaml")
        method, config = raw.split("=", 1)
        method, config = method.strip(), config.strip()
        if not method or not config:
            raise ValueError(f"Invalid --specs entry {raw!r}; expected method=config.yaml")
        out.append((method, config))
    if not out:
        raise ValueError("--specs must contain at least one method=config.yaml entry")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="Shared config for --baselines mode")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--val-dataset", default=None)
    ap.add_argument("--baselines", default=None, help="Comma-separated methods sharing --config")
    ap.add_argument("--specs", default=None, help="Comma-separated method=config.yaml pairs")
    ap.add_argument("--output-root", required=True)
    args = ap.parse_args()

    if args.specs:
        if args.config or args.baselines:
            raise ValueError("Use either --specs or --config + --baselines, not both")
        method_specs = _specs(args.specs)
    else:
        if not args.config or not args.baselines:
            raise ValueError("--config and --baselines are required when --specs is not used")
        method_specs = [(m, str(args.config)) for m in _methods(args.baselines)]

    # This is the expensive part: enumerate manifests, group candidates, and
    # preserve every group present in the requested split.  It is identical for all methods,
    # so perform it exactly once per split.
    train_groups = group_sample_paths(args.dataset, split="train")
    if not train_groups:
        raise ValueError(f"No grouped samples found in {args.dataset!s} for split='train'")
    val_groups = group_sample_paths(args.val_dataset, split="val") if args.val_dataset else []
    if args.val_dataset and not val_groups:
        raise ValueError(f"No grouped samples found in {args.val_dataset!s} for split='val'")

    first = load_external_sample(train_groups[0][0])
    max_group_candidates = max(len(g) for g in train_groups)
    root = ensure_dir(args.output_root)
    methods: list[str] = []
    for method, config_path in method_specs:
        cfg = load_config(config_path)
        method_cfg = copy.deepcopy(cfg)
        eb = method_cfg.setdefault("external_baselines", {})
        eb["baseline"] = method
        methods.append(method)
        out_dir = ensure_dir(root / method)
        summary = {
            "baseline": method,
            "training_mode": "non_learning_filter_or_planner",
            "dataset_validated": True,
            "shared_regime_validation": True,
            "train_dataset": str(args.dataset),
            "val_dataset": str(args.val_dataset) if args.val_dataset else None,
            "num_train_groups": len(train_groups),
            "num_val_groups": len(val_groups) if args.val_dataset else None,
            "feature_dim": int(sample_to_feature(first, method_cfg).shape[0]),
            "max_candidates": int(eb.get("max_candidates", max_group_candidates)),
            "config_path": str(config_path),
            "notes": (
                "This baseline is optimization/rule/filter based and has no neural training step. "
                "The matching regime train/val datasets were grouped and dataset-validated once "
                "for all non-learning methods; calibration-only scalars remain fitted separately "
                "on the matching calibration split, never on test."
            ),
            "cfg": method_cfg,
        }
        write_json(summary, out_dir / "train_summary.json")
    print({
        "event": "registered_nonlearning_baselines",
        "methods": methods,
        "num_train_groups": len(train_groups),
        "num_val_groups": (len(val_groups) if args.val_dataset else None),
        "shared_dataset_scans": 1,
    })


if __name__ == "__main__":
    main()
