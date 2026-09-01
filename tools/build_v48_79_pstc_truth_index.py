#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from ocrap.data.build.diagnose import iter_sample_paths
from ocrap.data.serialization import load_npz_selected
from ocrap.v48_79_truth_contract import nested_tail_truth_contract

KEYS = frozenset({
    "scene_id", "time_index", "candidate_index", "split_id",
    "m_star", "root_probs", "root_valid", "c_star", "option_valid",
    "root_assignments", "future_metadata", "recovery_modes", "r_dep_star",
})


def _parse_root(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("--root must be ROLE=/path/to/dataset")
    role, raw = spec.split("=", 1)
    role = role.strip()
    path = Path(raw).expanduser().resolve()
    if not role or not path.is_dir():
        raise argparse.ArgumentTypeError(f"invalid dataset root: {spec}")
    return role, path


def _scalar(d, key, default):
    try:
        return np.asarray(d.get(key, default)).item()
    except Exception:
        return default


def _one(role: str, path: Path, alpha: float, beta: float, top_m: int, tol: float) -> dict:
    d = load_npz_selected(path, KEYS)
    rec = nested_tail_truth_contract(d, alpha=alpha, beta=beta, top_m=top_m, recompute_tolerance=tol)
    out = rec.to_dict()
    out.update({
        "sample_path": str(path.resolve()),
        "dataset_role": role,
        "scene_id": str(_scalar(d, "scene_id", path.stem)),
        "time_index": int(_scalar(d, "time_index", -1)),
        "candidate_index": int(_scalar(d, "candidate_index", -1)),
        "split_id": str(_scalar(d, "split_id", "")),
    })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the V48.79 conservative physical-vs-structural nested-tail truth index")
    ap.add_argument("--root", action="append", required=True, help="ROLE=/dataset/root; repeat for train/dev/certificate Near/Contact")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--beta", type=float, default=0.2)
    ap.add_argument("--top-m", type=int, default=8)
    ap.add_argument("--recompute-tolerance", type=float, default=1e-5)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    roots = [_parse_root(x) for x in args.root]
    entries: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for role, root in roots:
        paths = list(iter_sample_paths(root))
        if not paths:
            raise SystemExit(f"no samples under {root}")
        for p in paths:
            rp = p.resolve()
            if rp in seen:
                raise SystemExit(f"sample appears in multiple indexed roots: {rp}")
            seen.add(rp)
            entries.append((role, rp))

    t0 = time.perf_counter()
    workers = max(1, int(args.workers))
    def work(x):
        role, p = x
        return _one(role, p, float(args.alpha), float(args.beta), int(args.top_m), float(args.recompute_tolerance))
    if workers == 1:
        rows = [work(x) for x in entries]
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="v4879-truth") as pool:
            rows = list(pool.map(work, entries))

    invalid = [r for r in rows if not bool(r.get("valid", False))]
    if invalid:
        worst = sorted(invalid, key=lambda r: float(r.get("r_dep_abs_error", 0.0)), reverse=True)[:5]
        raise SystemExit(f"truth-index OC-MERO recomputation mismatch: invalid={len(invalid)} examples={worst}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    tmp.replace(args.output)

    by_role: dict[str, dict] = {}
    reason_totals = defaultdict(float)
    for role, _root in roots:
        rr = [r for r in rows if r["dataset_role"] == role]
        physical = [r for r in rr if r["physical_identifiable"]]
        ppos = [r for r in physical if float(r.get("r_dep_stored", -1.0)) >= 0.0]
        pneg = [r for r in physical if float(r.get("r_dep_stored", 1.0)) < 0.0]
        p05 = [r for r in physical if abs(float(r.get("r_dep_stored", 0.0)) - 0.5) <= 1.0e-8]
        expos = [r for r in rr if not bool(r.get("physical_identifiable", False))]
        by_role[role] = {
            "rows": len(rr),
            "physical_identifiable_rows": len(physical),
            "structurally_exposed_rows": len(expos),
            "physical_identifiable_fraction": len(physical) / max(len(rr), 1),
            "physical_teacher_feasible_rows": len(ppos),
            "physical_teacher_infeasible_rows": len(pneg),
            "physical_exact_0p5_rows": len(p05),
            "structurally_exposed_teacher_feasible_rows": sum(float(r.get("r_dep_stored", -1.0)) >= 0.0 for r in expos),
            "structurally_exposed_teacher_infeasible_rows": sum(float(r.get("r_dep_stored", 1.0)) < 0.0 for r in expos),
            "mean_structural_exposure_mass": float(np.mean([r["structural_exposure_mass"] for r in rr])) if rr else None,
            "max_r_dep_abs_error": max((float(r["r_dep_abs_error"]) for r in rr), default=0.0),
        }
        for r in rr:
            for k, v in (r.get("structural_reason_mass") or {}).items():
                reason_totals[k] += float(v)

    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    summary = {
        "schema": "ocrap-v48.79-pstc-truth-index-summary-v1",
        "engineering_version": "v48.79.0-OC-PSTC",
        "valid": True,
        "rows": len(rows),
        "alpha": float(args.alpha),
        "beta": float(args.beta),
        "top_m": int(args.top_m),
        "recompute_tolerance": float(args.recompute_tolerance),
        "workers": workers,
        "build_seconds": float(time.perf_counter() - t0),
        "output": str(args.output.resolve()),
        "output_sha256": digest,
        "roles": by_role,
        "structural_reason_tail_mass_sum": dict(sorted(reason_totals.items())),
        "definition": "physical_identifiable iff the exact nested teacher OC-MERO active tail has zero conservative exposure to current teacher structural floor/override/hidden-branch rules",
        "dataset_reconstruction": False,
        "teacher_files_modified": False,
        "teacher_future_input_to_model": False,
        "test_roots_read": False,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "v48_79_pstc_truth_index", "rows": len(rows), "sha256": digest, "seconds": summary["build_seconds"]}))


if __name__ == "__main__":
    main()
