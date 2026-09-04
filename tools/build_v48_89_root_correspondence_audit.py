#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from ocrap.data.serialization import load_npz_selected
from ocrap.v48_89_root_correspondence import audit_candidate_nominal_pair

ROLE_FILES = {
    "dev_near": "dev_diagnostic_near_v48.proposal_rows.jsonl",
    "dev_contact": "dev_diagnostic_contact_v48.proposal_rows.jsonl",
    "certificate_near": "direct_value_risk_near_v48.proposal_rows.jsonl",
    "certificate_contact": "direct_value_risk_contact_v48.proposal_rows.jsonl",
}
VARIANTS = ("balanced", "precision")
KEYS = frozenset(
    {
        "scene_id",
        "time_index",
        "candidate_index",
        "is_nominal",
        "m_star",
        "root_probs",
        "root_valid",
        "c_star",
        "option_valid",
        "root_assignments",
        "future_probs",
        "future_valid",
        "future_sources",
        "future_metadata",
        "recovery_modes",
        "r_dep_star",
    }
)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _parse_root(value: str) -> tuple[str, Path]:
    role, raw = value.split("=", 1)
    if role not in ROLE_FILES:
        raise argparse.ArgumentTypeError(f"unknown role {role!r}; expected {sorted(ROLE_FILES)}")
    path = Path(raw).expanduser().resolve()
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"dataset root not found: {path}")
    return role, path


def _iter_manifest(root: Path):
    manifest = root / "manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"missing manifest {manifest}")
    with manifest.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw = row.get("path") or row.get("sample_path") or row.get("file")
            if not raw:
                continue
            p = Path(raw)
            yield p.resolve() if p.is_absolute() else (root / p).resolve()


def _scalar(d: dict[str, Any], key: str, default: Any) -> Any:
    try:
        return np.asarray(d.get(key, default)).reshape(-1)[0].item()
    except Exception:
        return default


def _load_sample(path: Path) -> dict[str, Any]:
    d = load_npz_selected(path, KEYS)
    d["__path__"] = str(path.resolve())
    d["__scene__"] = str(_scalar(d, "scene_id", path.stem))
    d["__time__"] = int(_scalar(d, "time_index", -1))
    d["__candidate__"] = int(_scalar(d, "candidate_index", -1))
    d["__nominal__"] = bool(int(_scalar(d, "is_nominal", int(d["__candidate__"] == 0))))
    return d


def _proposal_rows(run: Path, variant: str, role: str) -> list[dict[str, Any]]:
    path = run / "candidates" / variant / "calibration" / ROLE_FILES[role]
    if not path.is_file():
        raise FileNotFoundError(path)
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _label_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return str(row.get("scene", "")), int(row.get("time", -1)), int(row.get("candidate", -1))


def _labels(proposal_run: Path) -> tuple[dict[str, dict[tuple[str, int, int], dict[str, Any]]], dict[str, Any]]:
    by_role: dict[str, dict[tuple[str, int, int], dict[str, Any]]] = {}
    identity: dict[str, Any] = {}
    for role in ROLE_FILES:
        per_variant = {}
        for variant in VARIANTS:
            rows = _proposal_rows(proposal_run, variant, role)
            m = {_label_key(r): r for r in rows}
            if len(m) != len(rows):
                raise ValueError(f"duplicate proposal key role={role} variant={variant}")
            per_variant[variant] = m
        shared = set(per_variant["balanced"]).intersection(per_variant["precision"])
        mismatches = 0
        for key in shared:
            a, b = per_variant["balanced"][key], per_variant["precision"][key]
            fields = ("teacher_adv", "teacher_harmful", "teacher_candidate_r_dep", "macro")
            for field in fields:
                av, bv = a.get(field), b.get(field)
                if isinstance(av, (float, int)) and isinstance(bv, (float, int)):
                    if abs(float(av) - float(bv)) > 1e-7:
                        mismatches += 1
                        break
                elif av != bv:
                    mismatches += 1
                    break
        if set(per_variant["balanced"]) != set(per_variant["precision"]) or mismatches:
            raise ValueError(
                f"balanced/precision teacher-label identity failed role={role}: "
                f"sizes={len(per_variant['balanced'])}/{len(per_variant['precision'])} mismatches={mismatches}"
            )
        by_role[role] = per_variant["balanced"]
        identity[role] = {"rows": len(shared), "mismatches": mismatches, "exact_key_identity": True}
    return by_role, identity


def _auc(labels: list[bool], scores: list[float]) -> float | None:
    y = np.asarray(labels, dtype=bool)
    s = np.asarray(scores, dtype=np.float64)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return None
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def _quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"q10": None, "median": None, "q90": None, "mean": None}
    a = np.asarray(values, dtype=np.float64)
    return {
        "q10": float(np.quantile(a, 0.10)),
        "median": float(np.quantile(a, 0.50)),
        "q90": float(np.quantile(a, 0.90)),
        "mean": float(np.mean(a)),
    }


def _mean(rows: list[dict[str, Any]], field: str) -> float | None:
    vals = [float(r[field]) for r in rows if r.get(field) is not None and math.isfinite(float(r[field]))]
    return float(np.mean(vals)) if vals else None


def _summarize_role(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in rows if r.get("valid")]
    labeled = [r for r in valid if r.get("label_available")]
    safe = [r for r in labeled if r.get("safe_positive")]
    harmful = [r for r in labeled if r.get("teacher_harmful")]
    ti = [r for r in labeled if not r.get("teacher_feasible")]
    tf = [r for r in labeled if r.get("teacher_feasible")]
    sufficiently_identified = [r for r in labeled if float(r["matched_tail_sign_identifiable_mass"]) >= 0.5]

    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for r in labeled:
        groups[(str(r["scene_id"]), int(r["time_index"]))].append(r)
    powered = [g for g in groups.values() if any(r.get("safe_positive") for r in g)]
    top1 = None
    chance = None
    if powered:
        top1 = float(
            np.mean(
                [
                    bool(max(g, key=lambda z: float(z["matched_tail_signed_mass_score"])).get("safe_positive"))
                    for g in powered
                ]
            )
        )
        chance = float(np.mean([sum(bool(r.get("safe_positive")) for r in g) / len(g) for g in powered]))

    safe_vs_harm = None
    if safe and harmful:
        safe_vs_harm = _auc(
            [True] * len(safe) + [False] * len(harmful),
            [float(r["matched_tail_signed_mass_score"]) for r in safe]
            + [float(r["matched_tail_signed_mass_score"]) for r in harmful],
        )
    return {
        "rows": len(rows),
        "valid_rows": len(valid),
        "valid_fraction": len(valid) / max(1, len(rows)),
        "labeled_rows": len(labeled),
        "label_coverage": len(labeled) / max(1, len(valid)),
        "safe_positive_rows": len(safe),
        "harmful_rows": len(harmful),
        "teacher_feasible_rows": len(tf),
        "teacher_infeasible_rows": len(ti),
        "sign_identifiable_ge_0p5_rows": len(sufficiently_identified),
        "sign_identifiable_ge_0p5_fraction": len(sufficiently_identified) / max(1, len(labeled)),
        "shared_future_mass_candidate": _quantiles([float(r["shared_future_mass_candidate"]) for r in valid]),
        "shared_future_mass_nominal": _quantiles([float(r["shared_future_mass_nominal"]) for r in valid]),
        "semantic_identity_fallback_mass": _quantiles(
            [float(r["semantic_identity_fallback_fraction_candidate"]) for r in valid]
        ),
        "exact_root_probability_mass": _quantiles(
            [float(r["exact_candidate_root_probability_mass"]) for r in valid]
        ),
        "soft_root_purity": _quantiles([float(r["mean_soft_root_purity"]) for r in valid]),
        "nested_tail_exact_correspondence_mass": _quantiles(
            [float(r["nested_tail_exact_correspondence_mass"]) for r in valid]
        ),
        "nested_tail_soft_correspondence_mass": _quantiles(
            [float(r["nested_tail_soft_correspondence_mass"]) for r in valid]
        ),
        "matched_tail_informative_response_mass": _quantiles(
            [float(r["matched_tail_informative_response_mass"]) for r in valid]
        ),
        "matched_tail_sign_identifiable_mass": _quantiles(
            [float(r["matched_tail_sign_identifiable_mass"]) for r in valid]
        ),
        "matched_tail_point_identifiable_mass": _quantiles(
            [float(r["matched_tail_point_identifiable_mass"]) for r in valid]
        ),
        "branch_vs_slot_mapping_disagreement_fraction": _quantiles(
            [float(r["branch_vs_slot_mapping_disagreement_fraction"]) for r in valid]
        ),
        "slot_minus_branch_sign_identifiable_mass_mean": (
            float(
                np.mean(
                    [
                        float(r["slot_tail_sign_identifiable_mass"])
                        - float(r["matched_tail_sign_identifiable_mass"])
                        for r in valid
                    ]
                )
            )
            if valid
            else None
        ),
        "safe_positive_signed_score_mean": _mean(safe, "matched_tail_signed_mass_score"),
        "harmful_signed_score_mean": _mean(harmful, "matched_tail_signed_mass_score"),
        "teacher_feasible_signed_score_mean": _mean(tf, "matched_tail_signed_mass_score"),
        "teacher_infeasible_signed_score_mean": _mean(ti, "matched_tail_signed_mass_score"),
        "safe_positive_vs_harmful_auc": safe_vs_harm,
        "powered_safe_positive_groups": len(powered),
        "safe_positive_top1_accuracy": top1,
        "safe_positive_top1_chance": chance,
        "safe_positive_top1_lift": (top1 - chance) if top1 is not None and chance is not None else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", required=True, help="role=/path/to/dataset")
    ap.add_argument("--proposal-run", type=Path, required=True, help="historical L80/reference run used only for labels")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--beta", type=float, default=0.2)
    ap.add_argument("--top-m", type=int, default=8)
    args = ap.parse_args()

    roots = [_parse_root(x) for x in args.root]
    if {r for r, _ in roots} != set(ROLE_FILES):
        raise SystemExit(f"all roles required exactly once: {sorted(ROLE_FILES)}")
    proposal_run = args.proposal_run.expanduser().resolve()
    labels, label_identity = _labels(proposal_run)

    entries: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for role, root in roots:
        for path in _iter_manifest(root):
            if path in seen:
                raise SystemExit(f"duplicate sample path {path}")
            seen.add(path)
            entries.append((role, path))

    t0 = time.perf_counter()
    samples_by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers)), thread_name_prefix="v4889-load") as ex:
        loaded = list(ex.map(lambda z: (z[0], _load_sample(z[1])), entries))
    for role, sample in loaded:
        samples_by_role[role].append(sample)

    out_rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for role in ROLE_FILES:
        groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for sample in samples_by_role[role]:
            groups[(sample["__scene__"], sample["__time__"])].append(sample)
        for key, group in groups.items():
            nom = [s for s in group if s["__nominal__"] or s["__candidate__"] == 0]
            if len(nom) != 1:
                errors.append(f"role={role} group={key} nominal_count={len(nom)}")
                continue
            nominal = nom[0]
            for candidate in group:
                if candidate is nominal:
                    continue
                rec = audit_candidate_nominal_pair(
                    candidate,
                    nominal,
                    alpha=float(args.alpha),
                    beta=float(args.beta),
                    top_m=int(args.top_m),
                ).to_dict()
                lk = (candidate["__scene__"], candidate["__time__"], candidate["__candidate__"])
                lab = labels[role].get(lk)
                rec.update(
                    schema="ocrap-v48.89-root-correspondence-row-v1",
                    engineering_version="v48.89.0-OC-RCPI",
                    dataset_role=role,
                    sample_path=candidate["__path__"],
                    nominal_sample_path=nominal["__path__"],
                    scene_id=candidate["__scene__"],
                    time_index=candidate["__time__"],
                    candidate_index=candidate["__candidate__"],
                    label_available=lab is not None,
                    teacher_adv=float(lab.get("teacher_adv", float("nan"))) if lab else None,
                    teacher_harmful=bool(lab.get("teacher_harmful", False)) if lab else None,
                    teacher_feasible=(float(lab.get("teacher_candidate_r_dep", -1.0)) >= 0.0) if lab else None,
                    safe_positive=(float(lab.get("teacher_adv", -1.0)) >= 0.015 and not bool(lab.get("teacher_harmful", False))) if lab else None,
                    macro=int(lab.get("macro", -1)) if lab else None,
                    teacher_metadata_input_to_model=False,
                    dataset_reconstruction=False,
                )
                if not rec["valid"]:
                    errors.append(f"invalid pair role={role} key={lk}: {rec.get('error')}")
                out_rows.append(rec)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    roles = {role: _summarize_role([r for r in out_rows if r["dataset_role"] == role]) for role in ROLE_FILES}
    summary = {
        "schema": "ocrap-v48.89-root-correspondence-audit-summary-v1",
        "engineering_version": "v48.89.0-OC-RCPI",
        "valid": not errors,
        "attribution_ready": not errors,
        "errors": errors[:100],
        "rows": len(out_rows),
        "roles": roles,
        "label_identity": label_identity,
        "output": str(args.output.resolve()),
        "output_sha256": _sha(args.output),
        "proposal_run": str(proposal_run),
        "proposal_run_role": "teacher/safe-positive labels only; no model feature or training input",
        "alpha": float(args.alpha),
        "beta": float(args.beta),
        "top_m": int(args.top_m),
        "workers": max(1, int(args.workers)),
        "elapsed_seconds": float(time.perf_counter() - t0),
        "teacher_labels_changed": False,
        "teacher_metadata_input_to_model": False,
        "dataset_reconstruction": False,
        "test_roots_read": False,
        "planner_parameters_trained": 0,
        "boundary_transport": False,
        "regime_conditioning": False,
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"valid": summary["valid"], "rows": len(out_rows), "errors": len(errors), "summary": str(args.summary)}))
    return 0 if summary["valid"] else 30


if __name__ == "__main__":
    raise SystemExit(main())
