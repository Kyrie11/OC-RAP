#!/usr/bin/env python3
"""Fail-closed provenance/preflight for submission qualitative visualizations.

The visualization must be built from the same current main-table baselines that
were actually rerun for each regime and from the frozen deployed OC-RAP stack.
This tool intentionally accepts three separate external result roots because the
recommended user workflow runs Safe/Near/Contact launchers independently.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ocrap.external_baselines.provenance import MAIN_TABLE_BY_REGIME  # noqa: E402

SAFE_LEARNED = {
    "gameformer_lite": "source_port_v54",
    "plantf": "source_port_v54",
    "pluto": "source_port_v54",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact_record(path: Path, *, checkpoint: Path | None = None) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    record: dict[str, Any] = {"path": str(path.resolve()), "exists": path.is_file()}
    journal = Path(str(path) + ".scenes.jsonl")
    record["journal"] = str(journal.resolve())
    record["journal_exists"] = journal.is_file()
    if not path.is_file():
        errors.append(f"missing closed-loop result: {path}")
        return record, errors
    try:
        doc = _json(path)
        record["result_event"] = doc.get("event")
        record["num_scenes"] = doc.get("num_scenes") or doc.get("scenes_evaluated")
        record["result_sha256"] = _sha256(path)
    except Exception as exc:
        errors.append(f"invalid result JSON {path}: {exc}")
    if not journal.is_file():
        errors.append(f"missing scene journal: {journal}")
    else:
        record["journal_sha256"] = _sha256(journal)
    checker = REPO / "tools" / "check_closed_loop_artifact.py"
    if checker.is_file():
        proc = subprocess.run([sys.executable, str(checker), "--output", str(path), "--quiet"], cwd=REPO)
        record["complete_artifact_check"] = proc.returncode == 0
        if proc.returncode != 0:
            errors.append(f"closed-loop artifact is incomplete/invalid: {path}")
    if checkpoint is not None and checkpoint.is_file():
        record["result_mtime_ns"] = path.stat().st_mtime_ns
        record["checkpoint_mtime_ns"] = checkpoint.stat().st_mtime_ns
        record["result_not_older_than_checkpoint"] = path.stat().st_mtime_ns >= checkpoint.stat().st_mtime_ns
        if path.stat().st_mtime_ns < checkpoint.stat().st_mtime_ns:
            errors.append(f"result predates checkpoint; rerun required: result={path}, checkpoint={checkpoint}")
    return record, errors


def _checkpoint_record(path: Path, expected_impl: str | None = None) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    record: dict[str, Any] = {"path": str(path.resolve()), "exists": path.is_file()}
    if not path.is_file():
        errors.append(f"missing checkpoint: {path}")
        return record, errors
    record["sha256"] = _sha256(path)
    if expected_impl:
        validator = REPO / "tools" / "validate_external_checkpoint.py"
        proc = subprocess.run(
            [sys.executable, str(validator), "--checkpoint", str(path), "--require-deployable-contract",
             "--require-implementation-version", expected_impl],
            cwd=REPO,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        record["deployable_contract_valid"] = proc.returncode == 0
        if proc.returncode != 0:
            record["validator_output_tail"] = proc.stdout[-1200:]
            errors.append(f"invalid learned external checkpoint contract: {path}")
    return record, errors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ocrap-results-root", type=Path, required=True,
                    help="Root containing safe/near/contact/closed_loop_ocrap.json from the frozen full metric run.")
    ap.add_argument("--ocrap-model-run", type=Path, required=True)
    ap.add_argument("--variant", choices=("balanced", "precision"), default="balanced")
    ap.add_argument("--safe-external-root", type=Path, required=True)
    ap.add_argument("--near-external-root", type=Path, required=True)
    ap.add_argument("--contact-external-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    errors: list[str] = []
    roots = {"safe": args.safe_external_root, "near": args.near_external_root, "contact": args.contact_external_root}
    expected_methods = {k: list(v) for k, v in MAIN_TABLE_BY_REGIME.items()}

    candidate_root = args.ocrap_model_run / "candidates" / args.variant
    if not (candidate_root / "model_v48_trac_sr" / "best.pt").is_file():
        alt = args.ocrap_model_run / "dedicated_candidates" / args.variant
        if (alt / "model_v48_trac_sr" / "best.pt").is_file():
            candidate_root = alt
    ocrap_ckpt = candidate_root / "model_v48_trac_sr" / "best.pt"
    gamma = candidate_root / "calibration" / "gamma_rec_by_bucket_v48.json"
    ocrap_checkpoint, ck_errors = _checkpoint_record(ocrap_ckpt)
    errors.extend(ck_errors)
    if not gamma.is_file():
        errors.append(f"missing OC-RAP bucket calibration: {gamma}")
        gamma_record = {"path": str(gamma.resolve()), "exists": False}
    else:
        gamma_record = {"path": str(gamma.resolve()), "exists": True, "sha256": _sha256(gamma)}
        try:
            gb = _json(gamma).get("gamma_rec_by_bucket") or {}
            required = ("test_safe", "test_near_contact", "test_contact")
            gamma_record["gamma_rec_by_bucket"] = {k: gb.get(k) for k in required}
            if any(gb.get(k) is None for k in required):
                errors.append(f"bucket calibration missing one of {required}: {gamma}")
        except Exception as exc:
            errors.append(f"invalid OC-RAP calibration JSON {gamma}: {exc}")

    deployable_contract_path = args.output.parent / "V48.111-DEPLOYABLE-STACK.json"
    deployable_checker = REPO / "tools" / "check_v48_111_deployable_stack.py"
    deployable_contract: dict[str, Any] = {"path": str(deployable_contract_path.resolve()), "valid": False}
    if deployable_checker.is_file():
        deployable_contract_path.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            [sys.executable, str(deployable_checker), "--model-run", str(args.ocrap_model_run),
             "--variant", args.variant, "--output", str(deployable_contract_path)], cwd=REPO,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        deployable_contract["returncode"] = proc.returncode
        if deployable_contract_path.is_file():
            try:
                dd = _json(deployable_contract_path)
                deployable_contract["valid"] = bool(dd.get("valid"))
                deployable_contract["attribution_ready"] = dd.get("attribution_ready")
            except Exception as exc:
                errors.append(f"invalid deployable stack contract: {exc}")
        if proc.returncode != 0 or not deployable_contract.get("valid"):
            deployable_contract["output_tail"] = proc.stdout[-1600:]
            errors.append("V48.111 deployable stack contract failed")
    else:
        errors.append(f"missing deployable stack checker: {deployable_checker}")

    ocrap_results: dict[str, Any] = {}
    ocrap_supports: dict[str, Any] = {}
    for regime in ("safe", "near", "contact"):
        p = args.ocrap_results_root / regime / "closed_loop_ocrap.json"
        rec, rec_errors = _artifact_record(p, checkpoint=ocrap_ckpt if ocrap_ckpt.is_file() else None)
        ocrap_results[regime] = rec
        errors.extend(rec_errors)
        support = args.ocrap_results_root / regime / "closed_loop_dataset_support.json"
        if not support.is_file():
            errors.append(f"missing OC-RAP closed-loop dataset support contract: {support}")
            ocrap_supports[regime] = {"path": str(support.resolve()), "exists": False}
        else:
            try:
                sd = _json(support)
                ocrap_supports[regime] = {
                    "path": str(support.resolve()), "exists": True, "sha256": _sha256(support),
                    "raw_source_role": sd.get("raw_source_role"), "womd_pattern": sd.get("womd_pattern"),
                    "source_role_valid": sd.get("source_role_valid"), "schema_supports_closed_loop": sd.get("schema_supports_closed_loop"),
                }
                if sd.get("schema_supports_closed_loop") is not True:
                    errors.append(f"OC-RAP closed-loop dataset support is not valid for {regime}: {support}")
            except Exception as exc:
                errors.append(f"invalid OC-RAP dataset support JSON {support}: {exc}")

    external: dict[str, Any] = {}
    for regime in ("safe", "near", "contact"):
        root = roots[regime]
        external[regime] = {"root": str(root.resolve()), "methods": {}}
        support = root / "closed_loop_dataset_support.json"
        if not support.is_file():
            errors.append(f"missing external closed-loop dataset support contract: {support}")
            external[regime]["dataset_support"] = {"path": str(support.resolve()), "exists": False}
        else:
            try:
                sd = _json(support)
                external[regime]["dataset_support"] = {
                    "path": str(support.resolve()), "exists": True, "sha256": _sha256(support),
                    "raw_source_role": sd.get("raw_source_role"), "womd_pattern": sd.get("womd_pattern"),
                    "source_role_valid": sd.get("source_role_valid"), "schema_supports_closed_loop": sd.get("schema_supports_closed_loop"),
                }
                if sd.get("schema_supports_closed_loop") is not True:
                    errors.append(f"external closed-loop dataset support is not valid for {regime}: {support}")
                od = ocrap_supports.get(regime) or {}
                if od.get("exists") and od.get("raw_source_role") != sd.get("raw_source_role"):
                    errors.append(
                        f"raw WOMD source-role mismatch for {regime}: OC-RAP={od.get('raw_source_role')}, "
                        f"external={sd.get('raw_source_role')}; qualitative comparison would not replay the same source collection"
                    )
            except Exception as exc:
                errors.append(f"invalid external dataset support JSON {support}: {exc}")
        for method in MAIN_TABLE_BY_REGIME[regime]:
            ckpt = None
            ckpt_rec = None
            if regime == "safe" and method in SAFE_LEARNED:
                ckpt = root / "checkpoints" / method / "best.pt"
                ckpt_rec, e = _checkpoint_record(ckpt, SAFE_LEARNED[method])
                errors.extend(e)
            result = root / f"closed_loop_{method}.json"
            result_rec, e = _artifact_record(result, checkpoint=ckpt)
            errors.extend(e)
            external[regime]["methods"][method] = {"result": result_rec, "checkpoint": ckpt_rec}
        if regime == "near":
            cal = root / "conformal_calibration.json"
            external[regime]["conformal_calibration"] = {"path": str(cal.resolve()), "exists": cal.is_file()}
            if not cal.is_file():
                errors.append(f"missing Near conformal calibration: {cal}")
            else:
                external[regime]["conformal_calibration"]["sha256"] = _sha256(cal)

    doc = {
        "schema": "ocrap-submission-visualization-input-contract-v52",
        "valid": not errors,
        "errors": errors,
        "variant": args.variant,
        "path_contract": "separate user-run Safe/Near/Contact external roots; no fallback to historical external_baselines_v50",
        "external_main_table_methods": expected_methods,
        "ocrap": {
            "model_run": str(args.ocrap_model_run.resolve()),
            "candidate_root": str(candidate_root.resolve()),
            "checkpoint": ocrap_checkpoint,
            "calibration": gamma_record,
            "deployable_stack_contract": deployable_contract,
            "full_metric_results_root": str(args.ocrap_results_root.resolve()),
            "results": ocrap_results,
            "dataset_support": ocrap_supports,
        },
        "external": external,
        "notes": [
            "mtime freshness is fail-closed but is not treated as cryptographic proof that an old journal was generated by a particular checkpoint; selective trace reruns explicitly load the resolved current checkpoints.",
            "Near/Contact current main-table methods are non-neural controllers/filters; only Safe GameFormer/PlanTF/PLUTO require learned checkpoints.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": doc["schema"], "valid": doc["valid"], "errors": len(errors), "output": str(args.output)}))
    if errors:
        for err in errors:
            print(f"[ERROR] {err}", file=sys.stderr)
        return 30
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
