#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _check(name: str, value: Any, op: str, threshold: float, *, required: bool = True) -> bool:
    if value is None:
        ok = not required
        print(f"{name:48s} {'missing':>12s} {op} {threshold:.6f}  {'SKIP' if ok else 'FAIL'}")
        return ok
    v = float(value)
    ok = v <= threshold + 1.0e-9 if op == "<=" else v + 1.0e-9 >= threshold
    print(f"{name:48s} {v:12.6f} {op} {threshold:.6f}  {'PASS' if ok else 'FAIL'}")
    return ok


def _paired(report: dict[str, Any] | None, metric: str) -> tuple[float | None, tuple[float, float] | None]:
    if not report:
        return None, None
    row = (report.get("metrics") or {}).get(metric)
    if not row:
        return None, None
    ci = row.get("bootstrap_95ci")
    return float(row["paired_delta"]), (float(ci[0]), float(ci[1])) if ci else None


def _ci_direction(name: str, ci: tuple[float, float] | None, op: str, *, required: bool) -> bool:
    if not required:
        return True
    bound = None if ci is None else (ci[0] if op == ">=" else ci[1])
    return _check(f"{name} CI directional bound", bound, op, 0.0, required=True)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Check v48.24 Near/Contact physical targets and paired frontier diagnostics."
    )
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--near-paired", type=Path)
    ap.add_argument("--contact-paired", type=Path)
    ap.add_argument("--publication", action="store_true")
    args = ap.parse_args()

    near_path = args.run_dir / "audit_near_contact_selected_topk_v48_v48.json"
    contact_path = args.run_dir / "audit_contact_selected_topk_v48_v48.json"
    if not near_path.is_file() or not contact_path.is_file():
        print("missing Near/Contact closed-loop audit outputs", near_path, contact_path, sep="\n  ")
        return 2
    near, contact = _load(near_path), _load(contact_path)
    near_pair = _load(args.near_paired) if args.near_paired and args.near_paired.is_file() else None
    contact_pair = _load(args.contact_paired) if args.contact_paired and args.contact_paired.is_file() else None
    miss_max = 0.025 if args.publication else 0.034
    require_pair = bool(args.publication)
    ok = True

    print("===== Near-contact: prevent contact and recover continuous safety margin =====")
    ok &= _check("Near paper-PCD selector miss", near.get("closed_loop_audit_paper_pcd_selector_miss_rate"), "<=", miss_max)
    ok &= _check("Near PCD", near.get("closed_loop_post_contact_deployability"), ">=", 0.54)
    ok &= _check("Near FRA", near.get("closed_loop_FRA_exec"), "<=", 0.12)
    ok &= _check("Near DRS", near.get("closed_loop_DRS"), ">=", 0.88)
    ok &= _check("Near bounded NUP", near.get("closed_loop_bounded_NUP"), ">=", 0.995)
    ok &= _check("Near intervention rate", near.get("intervention_rate"), "<=", 0.020)
    ok &= _check("Near intervention episode rate", near.get("intervention_episode_rate"), "<=", 0.012)
    ok &= _check("Near maximum intervention run", near.get("max_intervention_run_length"), "<=", 1.0)
    ok &= _check("Near collision scene rate", near.get("collision_scene_rate"), "<=", 0.0)
    for metric, op, threshold in (
        ("min_clearance_m_min", ">=", 0.10),
        ("ttc_s_min", ">=", 0.20),
        ("near_contact_exposure_rate", "<=", 0.0),
        ("critical_ttc_exposure_rate", "<=", 0.0),
        ("clearance_deficit_auc_m_s", "<=", 0.0),
        ("ttc_deficit_auc_s2", "<=", 0.0),
    ):
        delta, ci = _paired(near_pair, metric)
        ok &= _check(f"Near paired delta: {metric}", delta, op, threshold, required=require_pair)
        ok &= _ci_direction(f"Near {metric}", ci, op, required=require_pair)

    print("===== Contact: reduce re-contact and create post-impact escape space =====")
    ok &= _check("Contact paper-PCD selector miss", contact.get("closed_loop_audit_paper_pcd_selector_miss_rate"), "<=", miss_max)
    ok &= _check("Contact PCD", contact.get("closed_loop_post_contact_deployability"), ">=", 0.52)
    ok &= _check("Contact FRA", contact.get("closed_loop_FRA_exec"), "<=", 0.16)
    ok &= _check("Contact DRS", contact.get("closed_loop_DRS"), ">=", 0.84)
    ok &= _check("Contact bounded NUP", contact.get("closed_loop_bounded_NUP"), ">=", 0.985)
    ok &= _check("Contact intervention rate", contact.get("intervention_rate"), "<=", 0.040)
    ok &= _check("Contact intervention episode rate", contact.get("intervention_episode_rate"), "<=", 0.025)
    ok &= _check("Contact maximum intervention run", contact.get("max_intervention_run_length"), "<=", 2.0)
    for metric, op, threshold in (
        ("secondary_overlap_event", "<=", -0.02),
        ("overlap_duration_s", "<=", 0.0),
        ("longest_overlap_run_s", "<=", 0.0),
        ("post_contact_clearance_m_max", ">=", 0.0),
        ("post_contact_free_space_auc_m_s", ">=", 0.0),
        ("post_contact_escape_event", ">=", 0.02),
        ("time_to_post_contact_escape_s", "<=", 0.0),
        ("new_stable_stop_event", ">=", 0.02),
    ):
        delta, ci = _paired(contact_pair, metric)
        ok &= _check(f"Contact paired delta: {metric}", delta, op, threshold, required=require_pair)
        ok &= _ci_direction(f"Contact {metric}", ci, op, required=require_pair)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
