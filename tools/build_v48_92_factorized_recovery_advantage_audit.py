#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from ocrap.v48_91_common_exogenous_physical_margin import (
    audit_future_physical_response,
    future_nested_tail_influence,
)
from ocrap.v48_92_factorized_recovery_advantage import (
    ENGINEERING_VERSION,
    factorize_recovery_advantage,
)
from tools.build_v48_89_root_correspondence_audit import (
    ROLE_FILES,
    VARIANTS,
    _auc,
    _label_key,
    _load_sample,
    _proposal_rows,
    _quantiles,
)

COMPONENT_FIELDS = (
    "teacher_adv",
    "teacher_candidate_drs",
    "teacher_nominal_drs",
    "teacher_candidate_r_dep",
    "teacher_nominal_r_dep",
    "teacher_candidate_gap",
    "teacher_nominal_gap",
    "teacher_candidate_hard",
    "teacher_nominal_hard",
    "teacher_candidate_harm_proxy",
    "teacher_nominal_harm_proxy",
)
SCORE_FIELDS = (
    "physical_response_score",
    "structural_response_score",
    "shapley_drs",
    "shapley_deployability_gate",
    "shapley_gap_discount",
    "partition_stability",
    "physical_tail_delta",
    "structural_tail_delta",
    "structural_distortion_delta",
)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load_sidecar(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            key = str(Path(row["sample_path"]).resolve())
            if key in out:
                raise ValueError(f"duplicate V48.91 sidecar sample {key}")
            out[key] = row
    return out


def _matrix(row: dict[str, Any], field: str) -> np.ndarray:
    F = int(row["future_count"])
    L = int(row["option_count"])
    out = np.full((F, L), np.nan, dtype=np.float64)
    for k, value in row[field].items():
        out[:, int(k)] = np.asarray(value, dtype=np.float64)
    return out


def _component_labels(l80_run: Path) -> tuple[dict[str, dict[tuple[str, int, int], dict[str, Any]]], dict[str, Any]]:
    merged_by_role: dict[str, dict[tuple[str, int, int], dict[str, Any]]] = {}
    identity: dict[str, Any] = {}
    for role in ROLE_FILES:
        by_variant: dict[str, dict[tuple[str, int, int], dict[str, Any]]] = {}
        for variant in VARIANTS:
            rows = _proposal_rows(l80_run, variant, role)
            table: dict[tuple[str, int, int], dict[str, Any]] = {}
            for row in rows:
                missing = [k for k in COMPONENT_FIELDS if k not in row]
                if "teacher_harmful" not in row or "macro" not in row:
                    missing += [k for k in ("teacher_harmful","macro") if k not in row]
                if missing:
                    raise ValueError(f"L80 proposal row missing fields role={role} variant={variant}: {missing}")
                key = _label_key(row)
                if key in table:
                    raise ValueError(f"duplicate L80 proposal row role={role} variant={variant} key={key}")
                for field in COMPONENT_FIELDS:
                    x = float(row[field])
                    if not math.isfinite(x):
                        raise ValueError(f"non-finite L80 component role={role} variant={variant} field={field}")
                table[key] = row
            by_variant[variant] = table
        a, b = by_variant["balanced"], by_variant["precision"]
        shared = set(a).intersection(b)
        if not shared:
            raise ValueError(f"no balanced/precision overlap role={role}")
        mismatch = 0
        max_component_error = 0.0
        for key in shared:
            for field in COMPONENT_FIELDS:
                err = abs(float(a[key][field]) - float(b[key][field]))
                max_component_error = max(max_component_error, err)
                if err > 1.0e-7:
                    mismatch += 1
                    break
            if bool(a[key]["teacher_harmful"]) != bool(b[key]["teacher_harmful"]) or int(a[key]["macro"]) != int(b[key]["macro"]):
                mismatch += 1
        if mismatch:
            raise ValueError(f"balanced/precision PCD component identity failed role={role} mismatches={mismatch}")
        union = set(a).union(b)
        merged: dict[tuple[str, int, int], dict[str, Any]] = {}
        for key in sorted(union):
            src = a[key] if key in a else b[key]
            merged[key] = dict(src)
        merged_by_role[role] = merged
        identity[role] = {
            "balanced_rows": len(a),
            "precision_rows": len(b),
            "shared_rows": len(shared),
            "union_rows": len(union),
            "shared_component_max_abs_error": max_component_error,
            "component_identity_on_overlap": True,
        }
    return merged_by_role, identity


@lru_cache(maxsize=16384)
def _load_sample_cached(path_text: str) -> dict[str, Any]:
    return _load_sample(Path(path_text))


def _tail_mean(mass: np.ndarray, matrix: np.ndarray) -> float:
    m = np.asarray(mass, dtype=np.float64)
    x = np.asarray(matrix, dtype=np.float64)
    mask = (m > 0.0) & np.isfinite(x)
    denom = float(m[mask].sum())
    if denom <= 1.0e-12:
        return float("nan")
    return float(np.sum(m[mask] * x[mask]) / denom)


def _mean(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [float(r[field]) for r in rows if r.get(field) is not None and math.isfinite(float(r[field]))]
    return float(np.mean(values)) if values else None


def _macro_auc(rows: list[dict[str, Any]], field: str) -> float | None:
    num = 0.0
    den = 0
    by: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by[int(row.get("macro", -1))].append(row)
    for group in by.values():
        p = np.asarray([float(r[field]) for r in group if r.get("safe_positive")], dtype=np.float64)
        n = np.asarray([float(r[field]) for r in group if r.get("teacher_harmful")], dtype=np.float64)
        if not len(p) or not len(n):
            continue
        num += float((p[:, None] > n[None, :]).sum() + 0.5 * (p[:, None] == n[None, :]).sum())
        den += int(len(p) * len(n))
    return float(num / den) if den else None


def _top1(rows: list[dict[str, Any]], field: str) -> tuple[float | None, float | None, float | None, int]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["scene_id"]), int(row["time_index"]))].append(row)
    powered = [g for g in groups.values() if any(bool(r.get("safe_positive")) for r in g)]
    if not powered:
        return None, None, None, 0
    acc, chance = [], []
    for group in powered:
        scores = np.asarray([float(r[field]) for r in group], dtype=np.float64)
        mx = float(np.max(scores))
        idx = np.where(np.abs(scores - mx) <= 1.0e-12)[0]
        acc.append(float(np.mean([bool(group[i].get("safe_positive")) for i in idx])))
        chance.append(float(np.mean([bool(r.get("safe_positive")) for r in group])))
    a = float(np.mean(acc)); c = float(np.mean(chance))
    return a, c, a - c, len(powered)


def _score_summary(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    safe = [r for r in rows if bool(r.get("safe_positive"))]
    harmful = [r for r in rows if bool(r.get("teacher_harmful"))]
    labels = [True] * len(safe) + [False] * len(harmful)
    scores = [float(r[field]) for r in safe] + [float(r[field]) for r in harmful]
    a, c, lift, n = _top1(rows, field)
    return {
        "safe_positive_mean": _mean(safe, field),
        "harmful_mean": _mean(harmful, field),
        "safe_vs_harmful_auc": _auc(labels, scores) if safe and harmful else None,
        "macro_stratified_auc": _macro_auc(rows, field),
        "top1_accuracy": a,
        "top1_chance": c,
        "top1_lift": lift,
        "powered_safe_positive_groups": n,
        "quantiles": _quantiles([float(r[field]) for r in rows]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--l80-run", type=Path, required=True)
    ap.add_argument("--v48-91-audit", type=Path, required=True)
    ap.add_argument("--v48-91-sidecar", type=Path, required=True)
    ap.add_argument("--v48-91-sidecar-summary", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--beta", type=float, default=0.2)
    ap.add_argument("--intra-root-alpha", type=float, default=0.2)
    ap.add_argument("--top-m", type=int, default=8)
    args = ap.parse_args()

    ss = json.loads(args.v48_91_sidecar_summary.read_text())
    if not (ss.get("valid") and ss.get("attribution_ready") and ss.get("dataset_reconstruction") is False):
        raise SystemExit("invalid V48.91 sidecar summary")
    if int(ss.get("valid_samples", 0)) != int(ss.get("requested_samples", -1)):
        raise SystemExit("V48.91 sidecar is incomplete")
    if _sha(args.v48_91_sidecar) != str(ss.get("output_sha256")):
        raise SystemExit("V48.91 sidecar SHA does not match its summary")

    labels, label_identity = _component_labels(args.l80_run.resolve())
    sidecar = _load_sidecar(args.v48_91_sidecar)
    matrix_cache: dict[tuple[str, str], np.ndarray] = {}

    def matrix(path_text: str, field: str) -> np.ndarray:
        key = (path_text, field)
        if key not in matrix_cache:
            matrix_cache[key] = _matrix(sidecar[path_text], field)
        return matrix_cache[key]

    out: list[dict[str, Any]] = []
    errors: list[str] = []
    max_pcd_error = 0.0
    max_v91_response_error = 0.0
    max_shapley_sum_error = 0.0
    with args.v48_91_audit.open(encoding="utf-8") as f:
        for line in f:
            base = json.loads(line)
            if not base.get("valid"):
                continue
            role = str(base["dataset_role"])
            key = (str(base["scene_id"]), int(base["time_index"]), int(base["candidate_index"]))
            lab = labels.get(role, {}).get(key)
            if lab is None:
                errors.append(f"missing L80 factor label role={role} key={key}")
                continue
            cp = str(Path(base["sample_path"]).resolve())
            npth = str(Path(base["nominal_sample_path"]).resolve())
            if cp not in sidecar or npth not in sidecar:
                errors.append(f"missing V48.91 sidecar pair {cp} / {npth}")
                continue
            try:
                fac = factorize_recovery_advantage(
                    candidate_drs=float(lab["teacher_candidate_drs"]),
                    nominal_drs=float(lab["teacher_nominal_drs"]),
                    candidate_r_dep=float(lab["teacher_candidate_r_dep"]),
                    nominal_r_dep=float(lab["teacher_nominal_r_dep"]),
                    candidate_gap=float(lab["teacher_candidate_gap"]),
                    nominal_gap=float(lab["teacher_nominal_gap"]),
                )
                pcd_error = abs(float(fac.teacher_adv_reconstructed) - float(base["teacher_adv"]))
                max_pcd_error = max(max_pcd_error, pcd_error)
                max_shapley_sum_error = max(max_shapley_sum_error, float(fac.shapley_sum_error))
                if pcd_error > 2.0e-6:
                    raise ValueError(
                        f"PCD reconstruction mismatch {pcd_error} for role={role} key={key}; "
                        "registered teacher_adv is not the factorization being audited"
                    )
                if bool(lab.get("teacher_harmful", False)) != bool(base.get("teacher_harmful", False)):
                    raise ValueError(f"teacher_harmful mismatch role={role} key={key}")

                csamp = _load_sample_cached(cp); nsamp = _load_sample_cached(npth)
                cst = matrix(cp, "m_future_structural"); nst = matrix(npth, "m_future_structural")
                cph = matrix(cp, "m_future_physical"); nph = matrix(npth, "m_future_physical")

                phys = audit_future_physical_response(
                    csamp, nsamp, cst, nst, cph, nph,
                    alpha=args.alpha, beta=args.beta, intra_root_alpha=args.intra_root_alpha, top_m=args.top_m,
                )
                if not phys.valid:
                    raise ValueError(f"V48.91 physical response recomputation invalid: {phys.error}")
                v91_error = abs(float(phys.signed_response_score) - float(base["signed_response_score"]))
                max_v91_response_error = max(max_v91_response_error, v91_error)
                if v91_error > 1.0e-12:
                    raise ValueError(f"V48.91 physical response identity failed: {v91_error}")

                structural = audit_future_physical_response(
                    csamp, nsamp, cst, nst, cst, nst,
                    alpha=args.alpha, beta=args.beta, intra_root_alpha=args.intra_root_alpha, top_m=args.top_m,
                )
                if not structural.valid:
                    raise ValueError(f"structural response audit invalid: {structural.error}")

                cmass, _ = future_nested_tail_influence(
                    csamp, cst, alpha=args.alpha, beta=args.beta,
                    intra_root_alpha=args.intra_root_alpha, top_m=args.top_m,
                )
                nmass, _ = future_nested_tail_influence(
                    nsamp, nst, alpha=args.alpha, beta=args.beta,
                    intra_root_alpha=args.intra_root_alpha, top_m=args.top_m,
                )
                c_struct = _tail_mean(cmass, cst); n_struct = _tail_mean(nmass, nst)
                c_phys = _tail_mean(cmass, cph); n_phys = _tail_mean(nmass, nph)

                rec = dict(base)
                rec.update(
                    schema="ocrap-v48.92-factorized-recovery-advantage-row-v1",
                    engineering_version=ENGINEERING_VERSION,
                    planner_parameters_trained=0,
                    dataset_reconstruction=False,
                    dataset_reselection=False,
                    regime_conditioning=False,
                    boundary_transport=False,
                    relative_ranker_modified=False,
                    teacher_labels_changed=False,
                    teacher_metadata_input_to_model=False,
                    v48_91_sidecar_reused=True,
                    womd_replay_performed=False,
                    pcd_reconstruction_error=float(pcd_error),
                    physical_response_score=float(phys.signed_response_score),
                    structural_response_score=float(structural.signed_response_score),
                    structural_response_sign_identifiable_mass=float(structural.response_sign_identifiable_mass),
                    structural_response_informative_mass=float(structural.response_informative_mass),
                    physical_tail_candidate=float(c_phys),
                    physical_tail_nominal=float(n_phys),
                    structural_tail_candidate=float(c_struct),
                    structural_tail_nominal=float(n_struct),
                    physical_tail_delta=float(c_phys - n_phys),
                    structural_tail_delta=float(c_struct - n_struct),
                    structural_distortion_candidate=float(c_struct - c_phys),
                    structural_distortion_nominal=float(n_struct - n_phys),
                    structural_distortion_delta=float((c_struct - c_phys) - (n_struct - n_phys)),
                    **fac.to_dict(),
                )
                out.append(rec)
            except Exception as exc:
                errors.append(f"role={role} key={key}: {exc}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    roles = sorted({str(r["dataset_role"]) for r in out})
    role_summary: dict[str, Any] = {}
    for role in roles:
        rr = [r for r in out if r["dataset_role"] == role]
        role_summary[role] = {
            "rows": len(rr),
            "safe_positive_rows": sum(bool(r.get("safe_positive")) for r in rr),
            "harmful_rows": sum(bool(r.get("teacher_harmful")) for r in rr),
            "scores": {field: _score_summary(rr, field) for field in SCORE_FIELDS},
            "structural_response_sign_identifiable_mass": _quantiles(
                [float(r["structural_response_sign_identifiable_mass"]) for r in rr]
            ),
        }

    summary = {
        "schema": "ocrap-v48.92-factorized-recovery-advantage-summary-v1",
        "engineering_version": ENGINEERING_VERSION,
        "valid": not errors,
        "attribution_ready": not errors,
        "errors": errors[:100],
        "experiment_type": "audit_only_factorized_recovery_advantage_mediation",
        "planner_parameters_trained": 0,
        "dataset_reconstruction": False,
        "dataset_reselection": False,
        "regime_conditioning": False,
        "boundary_transport": False,
        "relative_ranker_modified": False,
        "teacher_labels_changed": False,
        "teacher_metadata_input_to_model": False,
        "womd_replay_performed": False,
        "v48_91_sidecar_reused": True,
        "rows": len(out),
        "roles": role_summary,
        "label_component_identity": label_identity,
        "max_pcd_reconstruction_error": max_pcd_error,
        "max_shapley_sum_error": max_shapley_sum_error,
        "max_v48_91_physical_response_identity_error": max_v91_response_error,
        "v48_91_audit_sha256": _sha(args.v48_91_audit),
        "v48_91_sidecar_sha256": _sha(args.v48_91_sidecar),
        "output": str(args.output.resolve()),
        "output_sha256": _sha(args.output),
        "test_roots_read": False,
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"valid": summary["valid"], "rows": len(out), "errors": len(errors)}))
    return 0 if summary["valid"] else 30


if __name__ == "__main__":
    raise SystemExit(main())
