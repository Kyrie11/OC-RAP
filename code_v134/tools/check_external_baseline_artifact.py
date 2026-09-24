#!/usr/bin/env python3
"""Lightweight completion checks for external-baseline pipeline artifacts.

This module intentionally does *not* import NumPy or PyTorch.  It is used by
shell launchers before deciding whether expensive training/evaluation work is
necessary, so a metadata/index check must remain usable even when the numeric
runtime itself needs repair.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import math
import os
import pickle
import zipfile
from pathlib import Path
from typing import Any

from ocrap.config import apply_overrides, load_config


class _TorchPlaceholder:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


def _torch_rebuild_placeholder(*args: Any, **kwargs: Any) -> _TorchPlaceholder:
    return _TorchPlaceholder()


class _MetadataUnpickler(pickle.Unpickler):
    """Read torch-save metadata without importing torch/numpy or tensor bytes."""

    _SAFE_GLOBALS = {
        ("collections", "OrderedDict"): collections.OrderedDict,
    }

    def find_class(self, module: str, name: str) -> Any:
        safe = self._SAFE_GLOBALS.get((module, name))
        if safe is not None:
            return safe
        if module.startswith("torch"):
            if name.startswith("_rebuild"):
                return _torch_rebuild_placeholder
            return _TorchPlaceholder
        # Checkpoints produced by this repository only need plain Python
        # containers plus torch tensor/storage rebuild globals.  Refuse all
        # unrelated globals instead of executing arbitrary pickle callables.
        raise pickle.UnpicklingError(f"unsupported checkpoint global: {module}.{name}")

    def persistent_load(self, pid: Any) -> _TorchPlaceholder:
        return _TorchPlaceholder()


def _load_torch_metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path, "r") as zf:
            names = [n for n in zf.namelist() if n.endswith("/data.pkl") or n == "data.pkl"]
            if not names:
                raise ValueError(f"{path} has no data.pkl")
            data = zf.read(names[0])
        obj = _MetadataUnpickler(io.BytesIO(data)).load()
    else:
        # Legacy torch serialization is a pickle stream.  The same restricted
        # unpickler works for metadata-only reading when tensors use persistent ids.
        with path.open("rb") as f:
            obj = _MetadataUnpickler(f).load()
    if not isinstance(obj, dict):
        raise TypeError(f"checkpoint root is {type(obj).__name__}, expected dict")
    return obj


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _config_fingerprint(cfg: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(cfg).encode("utf-8")).hexdigest()


def _same_path(a: str | None, b: str | None) -> bool:
    if a in {None, ""} or b in {None, ""}:
        return a in {None, ""} and b in {None, ""}
    try:
        return Path(str(a)).expanduser().resolve() == Path(str(b)).expanduser().resolve()
    except Exception:
        return str(a) == str(b)


def _fresh_enough(output: Path, dependencies: list[Path]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not output.is_file():
        return False, [f"missing_output:{output}"]
    try:
        out_ns = output.stat().st_mtime_ns
    except OSError as exc:
        return False, [f"output_stat_failed:{exc}"]
    for dep in dependencies:
        if not dep.exists():
            reasons.append(f"missing_dependency:{dep}")
            continue
        try:
            if dep.stat().st_mtime_ns > out_ns:
                reasons.append(f"dependency_newer_than_output:{dep}")
        except OSError as exc:
            reasons.append(f"dependency_stat_failed:{dep}:{exc}")
    return not reasons, reasons


def _effective_cfg(config: str, sets: list[str], baseline: str | None = None) -> dict[str, Any]:
    cfg = apply_overrides(load_config(config), sets)
    if baseline:
        cfg.setdefault("external_baselines", {})["baseline"] = baseline
    return cfg


def check_training(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.output_dir)
    cfg = _effective_cfg(args.config, args.set or [], args.baseline)
    tcfg = ((cfg.get("external_baselines", {}) or {}).get("training", {}) or {})
    target_epoch = int(tcfg.get("epochs", 10))
    latest_path = out / "latest.pt"
    best_path = out / "best.pt"
    summary_path = out / "train_summary.json"
    errors: list[str] = []
    latest: dict[str, Any] = {}
    best: dict[str, Any] = {}
    try:
        latest = _load_torch_metadata(latest_path)
    except Exception as exc:
        errors.append(f"latest_checkpoint_invalid:{exc}")
    try:
        best = _load_torch_metadata(best_path)
    except Exception as exc:
        errors.append(f"best_checkpoint_invalid:{exc}")
    summary = _read_json(summary_path)
    if summary is None:
        errors.append("train_summary_missing_or_invalid")

    latest_epoch = None
    best_epoch = None
    if latest:
        try:
            latest_epoch = int(latest.get("epoch", -1))
        except Exception:
            latest_epoch = -1
        if latest_epoch != target_epoch:
            errors.append(f"latest_epoch_mismatch:{latest_epoch}!={target_epoch}")
        if args.baseline and str(latest.get("baseline", "")).lower() != str(args.baseline).lower():
            errors.append(f"latest_baseline_mismatch:{latest.get('baseline')!r}")
        if not isinstance(latest.get("model_state"), dict) or not latest.get("model_state"):
            errors.append("latest_model_state_missing")
        try:
            if not math.isfinite(float(latest.get("val_loss", float("nan")))):
                errors.append("latest_val_loss_nonfinite")
        except Exception:
            errors.append("latest_val_loss_invalid")
    if best:
        try:
            best_epoch = int(best.get("epoch", -1))
        except Exception:
            best_epoch = -1
        if best_epoch < 1 or best_epoch > target_epoch:
            errors.append(f"best_epoch_out_of_range:{best_epoch}")
        if args.baseline and str(best.get("baseline", "")).lower() != str(args.baseline).lower():
            errors.append(f"best_baseline_mismatch:{best.get('baseline')!r}")
        if not isinstance(best.get("model_state"), dict) or not best.get("model_state"):
            errors.append("best_model_state_missing")

    if summary is not None:
        if args.baseline and str(summary.get("baseline", "")).lower() != str(args.baseline).lower():
            errors.append(f"summary_baseline_mismatch:{summary.get('baseline')!r}")
        history = summary.get("history")
        history_last = None
        if isinstance(history, list) and history:
            try:
                history_last = int(history[-1].get("epoch", -1))
            except Exception:
                history_last = -1
        elif summary.get("epochs_completed") is not None:
            try:
                history_last = int(summary.get("epochs_completed"))
            except Exception:
                history_last = -1
        if history_last != target_epoch:
            errors.append(f"summary_last_epoch_mismatch:{history_last}!={target_epoch}")
        if summary.get("epochs_requested") is not None and int(summary.get("epochs_requested")) != target_epoch:
            errors.append(f"summary_requested_epoch_mismatch:{summary.get('epochs_requested')}!={target_epoch}")
        if best_epoch is not None and summary.get("best_epoch") is not None:
            try:
                if int(summary.get("best_epoch")) != best_epoch:
                    errors.append(f"best_epoch_summary_checkpoint_mismatch:{summary.get('best_epoch')}!={best_epoch}")
            except Exception:
                errors.append("summary_best_epoch_invalid")
        if args.dataset and summary.get("train_dataset") is not None and not _same_path(summary.get("train_dataset"), args.dataset):
            errors.append("train_dataset_mismatch")
        if args.val_dataset and summary.get("val_dataset") is not None and not _same_path(summary.get("val_dataset"), args.val_dataset):
            errors.append("val_dataset_mismatch")

    # The user's reuse contract is epoch-based: if latest.pt reached the
    # requested epoch count, reuse it.  Do not use config-file mtimes here:
    # installing this hotfix would otherwise make every completed run appear
    # stale merely because the YAML file was copied/updated.
    fresh, stale_reasons = _fresh_enough(summary_path, [latest_path, best_path])
    if not fresh:
        errors.extend(stale_reasons)

    return {
        "event": "external_baseline_training_artifact_check",
        "complete": not errors,
        "baseline": args.baseline,
        "output_dir": str(out),
        "target_epoch": target_epoch,
        "latest_epoch": latest_epoch,
        "best_epoch": best_epoch,
        "errors": errors,
    }


def check_nonlearning_training(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.output_root)
    methods = [x.strip() for x in str(args.baselines).split(",") if x.strip()]
    errors: list[str] = []
    for method in methods:
        path = root / method / "train_summary.json"
        doc = _read_json(path)
        if doc is None:
            errors.append(f"{method}:missing_or_invalid_summary")
            continue
        if str(doc.get("baseline", "")).lower() != method.lower():
            errors.append(f"{method}:baseline_mismatch")
        if str(doc.get("training_mode", "")) != "non_learning_filter_or_planner":
            errors.append(f"{method}:training_mode_mismatch")
        if doc.get("dataset_validated") is not True:
            errors.append(f"{method}:dataset_not_validated")
        if not _same_path(doc.get("train_dataset"), args.dataset):
            errors.append(f"{method}:train_dataset_mismatch")
        if args.val_dataset and not _same_path(doc.get("val_dataset"), args.val_dataset):
            errors.append(f"{method}:val_dataset_mismatch")
        # Non-learning methods do not own a neural checkpoint.  Their
        # train-stage completion artifact stores the effective config used by
        # the registration/validation pass, so validate it by content rather
        # than source-file mtime (which changes when a code hotfix is copied).
        config_path = doc.get("config_path")
        stored_cfg = doc.get("cfg")
        if config_path not in {None, ""} and isinstance(stored_cfg, dict):
            try:
                expected_cfg = load_config(str(config_path))
                expected_cfg.setdefault("external_baselines", {})["baseline"] = method
                if _canonical_json(stored_cfg) != _canonical_json(expected_cfg):
                    errors.append(f"{method}:config_mismatch")
            except Exception as exc:
                errors.append(f"{method}:config_validation_failed:{exc}")
    return {
        "event": "external_nonlearning_training_artifact_check",
        "complete": not errors,
        "output_root": str(root),
        "methods": methods,
        "errors": errors,
    }


def check_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output)
    doc = _read_json(output)
    errors: list[str] = []
    expected_methods = [x.strip() for x in str(args.baselines).split(",") if x.strip()]
    if doc is None:
        errors.append("missing_or_invalid_output")
    else:
        if not _same_path(doc.get("dataset"), args.dataset):
            errors.append("dataset_mismatch")
        if str(doc.get("split", "")) != str(args.split):
            errors.append("split_mismatch")
        order = [str(x) for x in doc.get("method_order", [])] if isinstance(doc.get("method_order"), list) else []
        if order != expected_methods:
            errors.append(f"method_order_mismatch:{order!r}!={expected_methods!r}")
        methods_doc = doc.get("methods") if isinstance(doc.get("methods"), dict) else {}
        for method in expected_methods:
            row = methods_doc.get(method) if isinstance(methods_doc, dict) else None
            if not isinstance(row, dict):
                errors.append(f"{method}:missing_summary")
                continue
            if int(row.get("num_scene_time_groups", 0) or 0) <= 0:
                errors.append(f"{method}:no_scene_time_groups")
            if int(row.get("num_records", 0) or 0) <= 0:
                errors.append(f"{method}:no_records")
        if args.checkpoint:
            if not _same_path(doc.get("checkpoint"), args.checkpoint):
                errors.append("checkpoint_path_mismatch")
        elif doc.get("checkpoint") not in {None, ""}:
            errors.append("unexpected_checkpoint")
        if args.config:
            cfg = _effective_cfg(args.config, args.set or [])
            saved_fp = doc.get("requested_config_fingerprint")
            if saved_fp is not None and str(saved_fp) != _config_fingerprint(cfg):
                errors.append("requested_config_fingerprint_mismatch")

    # Config identity is checked by fingerprint for newly produced artifacts.
    # Legacy completed outputs did not contain a fingerprint, so never use the
    # config file's mtime as a proxy: copying patched source code must not force
    # expensive tests to rerun.  Data-derived artifacts (e.g. calibration) and
    # learned checkpoints remain true freshness dependencies.
    deps = [Path(x) for x in (args.dependency or [])]
    if args.checkpoint:
        deps.append(Path(args.checkpoint))
    fresh, stale = _fresh_enough(output, deps)
    if not fresh:
        errors.extend(stale)
    return {
        "event": "external_baseline_evaluation_artifact_check",
        "complete": not errors,
        "output": str(output),
        "methods": expected_methods,
        "errors": errors,
    }


def check_calibration(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output)
    doc = _read_json(output)
    errors: list[str] = []
    if doc is None:
        errors.append("missing_or_invalid_output")
    else:
        if not _same_path(doc.get("dataset"), args.dataset):
            errors.append("dataset_mismatch")
        if str(doc.get("split", "")) != str(args.split):
            errors.append("split_mismatch")
        try:
            if abs(float(doc.get("alpha")) - float(args.alpha)) > 1e-12:
                errors.append("alpha_mismatch")
        except Exception:
            errors.append("alpha_invalid")
        try:
            threshold = float(doc.get("conformal_collision_probability_threshold"))
            if not math.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
                errors.append("threshold_invalid")
        except Exception:
            errors.append("threshold_invalid")
        if doc.get("test_labels_used") is not False:
            errors.append("calibration_contract_invalid")
        if args.config:
            cfg = load_config(args.config)
            saved_fp = doc.get("requested_config_fingerprint")
            if saved_fp is not None and str(saved_fp) != _config_fingerprint(cfg):
                errors.append("requested_config_fingerprint_mismatch")
    # As above, do not invalidate a legacy completed calibration solely because
    # the source YAML's mtime changed when the patched code was installed.
    fresh, stale = _fresh_enough(output, [])
    if not fresh:
        errors.extend(stale)
    return {
        "event": "external_baseline_calibration_artifact_check",
        "complete": not errors,
        "output": str(output),
        "errors": errors,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)

    p = sub.add_parser("training")
    p.add_argument("--config", required=True)
    p.add_argument("--baseline", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--dataset", default=None)
    p.add_argument("--val-dataset", default=None)
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--quiet", action="store_true")

    p = sub.add_parser("nonlearning-training")
    p.add_argument("--output-root", required=True)
    p.add_argument("--baselines", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--val-dataset", default=None)
    p.add_argument("--dependency", action="append", default=[])
    p.add_argument("--quiet", action="store_true")

    p = sub.add_parser("evaluation")
    p.add_argument("--output", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--baselines", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--dependency", action="append", default=[])
    p.add_argument("--quiet", action="store_true")

    p = sub.add_parser("calibration")
    p.add_argument("--output", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", default="calibration")
    p.add_argument("--alpha", type=float, required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--quiet", action="store_true")

    args = ap.parse_args()
    if args.mode == "training":
        doc = check_training(args)
    elif args.mode == "nonlearning-training":
        doc = check_nonlearning_training(args)
    elif args.mode == "evaluation":
        doc = check_evaluation(args)
    elif args.mode == "calibration":
        doc = check_calibration(args)
    else:  # pragma: no cover
        raise AssertionError(args.mode)
    if not getattr(args, "quiet", False):
        print(json.dumps(doc, ensure_ascii=False, sort_keys=True))
    return 0 if doc["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
