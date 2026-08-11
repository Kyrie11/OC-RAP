from __future__ import annotations

import argparse, json, math, os, time
from pathlib import Path


def _finite(x: object) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def _read(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"{path}: expected JSON object")
    return data


def main() -> int:
    ap = argparse.ArgumentParser(description="Structural/dev-only quality preflight for a rebuilt v48.45 source.")
    ap.add_argument("--source-run", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    source = Path(args.source_run).expanduser().resolve(strict=False)
    manifest_path = source / "SOURCE_REBUILD_COMPLETE.json"
    errors: list[str] = []
    rows: dict[str, dict[str, object]] = {}
    if not manifest_path.is_file():
        errors.append("missing SOURCE_REBUILD_COMPLETE.json")
        manifest = {}
    else:
        try:
            manifest = _read(manifest_path)
        except Exception as exc:
            manifest = {}
            errors.append(f"manifest unreadable: {exc!r}")

    specs = {
        "backbone": source / "shared_recovery_backbone" / "model_v48_trac_sr" / "train_summary.json",
        "balanced": source / "candidates" / "balanced" / "model_v48_trac_sr" / "train_summary.json",
        "precision": source / "candidates" / "precision" / "model_v48_trac_sr" / "train_summary.json",
    }
    for name, path in specs.items():
        row: dict[str, object] = {"summary": str(path), "exists": path.is_file()}
        if not path.is_file():
            errors.append(f"{name}: missing train_summary.json")
            rows[name] = row
            continue
        try:
            d = _read(path)
        except Exception as exc:
            errors.append(f"{name}: summary unreadable: {exc!r}")
            rows[name] = row
            continue
        row.update({
            "num_train_samples": d.get("num_train_samples"),
            "num_val_samples": d.get("num_val_samples"),
            "epochs_completed": d.get("epochs_completed"),
            "best_epoch": d.get("best_epoch"),
            "best_metric": d.get("best_metric"),
            "best_metric_value": d.get("best_metric_value"),
            "best_val_loss": d.get("best_val_loss"),
            "init_checkpoint": d.get("init_checkpoint"),
            "trainable_param_prefixes": d.get("trainable_param_prefixes"),
        })
        for k in ("num_train_samples", "num_val_samples", "epochs_completed"):
            try:
                if int(d.get(k, 0)) <= 0: errors.append(f"{name}: {k} <= 0")
            except Exception:
                errors.append(f"{name}: invalid {k}")
        if not _finite(d.get("best_val_loss")): errors.append(f"{name}: non-finite best_val_loss")
        if not _finite(d.get("best_metric_value")): errors.append(f"{name}: non-finite best_metric_value")
        ckpt = Path(str(d.get("checkpoint", ""))).expanduser()
        row["checkpoint_from_summary"] = str(ckpt)
        row["checkpoint_exists"] = ckpt.is_file()
        if not ckpt.is_file(): errors.append(f"{name}: checkpoint from summary missing")
        rows[name] = row

    # Rebuild semantics: backbone must be scratch; S1 must warm-start from the sealed S0 checkpoint.
    if rows.get("backbone", {}).get("init_checkpoint") not in (None, ""):
        errors.append("backbone: expected scratch init_checkpoint")
    expected_backbone = str((source / "shared_recovery_backbone" / "model_v48_trac_sr" / "best.pt").resolve(strict=False))
    for name in ("balanced", "precision"):
        got = str(rows.get(name, {}).get("init_checkpoint") or "")
        if got and str(Path(got).expanduser().resolve(strict=False)) != expected_backbone:
            errors.append(f"{name}: init_checkpoint is not the common S0 checkpoint")

    doc = {
        "event": "v48_45_rebuilt_source_quality_contract",
        "created_unix": time.time(),
        "source_run": str(source),
        "manifest_source_identity": manifest.get("source_identity"),
        "rows": rows,
        "errors": errors,
        "valid": not errors,
        "scope": "train/validation summaries only; no calibration/certificate/test roots",
        "test_roots_read": False,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, out)
    print(json.dumps(doc, ensure_ascii=False))
    return 0 if doc["valid"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
