#!/usr/bin/env python3
"""v48.56 fail-closed audit for teacher/component decision semantics.

The audit is intentionally diagnostic: it does not tune a threshold and never reads
Safe/test roots.  It answers four preregistered questions from v48.55:

1. does the DEP component encode deployable recovery or merely nominal-relative change?
2. is GAP used with the right direction/role?
3. do cached teacher labels agree with a fresh OC-MERO recomputation?
4. do the scalar benefit and non-compensatory veto define contradictory examples?

When dataset roots are supplied, the tool additionally recomputes R_dep/R_orc from
stored m_star/root_probs/c_star and verifies the cached source labels exactly up to a
fixed numerical tolerance.  Index-only mode is sufficient for the role/overlap audit.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ocrap.algorithms.evidence_targets import (
    ComponentVetoTolerances,
    component_veto_terms_numpy,
)
from ocrap.algorithms.ocmero import oc_mero
from ocrap.data.serialization import load_npz_selected
from ocrap.models.data import MODEL_SAMPLE_NPZ_KEYS, iter_sample_paths_many

DEPLOYABLE_MACROS_DEFAULT = {2, 3, 5, 6, 7}
MACRO_NAMES = {
    0: "nominal", 1: "keep", 2: "brake", 3: "yield", 4: "lane_shift",
    5: "merge", 6: "pull_over", 7: "stabilize", 8: "perturb_nominal",
}




def _reject_test_inputs(*paths: str) -> None:
    """Fail closed if an explicitly supplied source points at a test-role root."""
    for raw in paths:
        for token in str(raw).split(","):
            token = token.strip()
            if not token:
                continue
            parts = [x.lower() for x in Path(token).parts]
            for part in parts:
                if part == "test" or part.startswith("test_") or part.endswith("_test"):
                    raise ValueError(f"v48.56 semantic audit refuses test-role input: {token}")

def _sigmoid(x: float) -> float:
    x = max(-30.0, min(30.0, float(x)))
    return 1.0 / (1.0 + math.exp(-x))


def _gap_quality(x: float) -> float:
    return math.exp(-max(0.0, min(20.0, float(x))))


def _regime(row: dict[str, Any]) -> str:
    p = str(row.get("path", ""))
    b = int(row.get("bucket", -1) or -1)
    if "near_contact" in p or b == 1:
        return "near"
    if "contact" in p or b == 2:
        return "contact"
    return f"bucket_{b}"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _group_nominals(rows: list[dict[str, Any]]) -> dict[tuple[int, str, int], dict[str, Any]]:
    out: dict[tuple[int, str, int], dict[str, Any]] = {}
    for r in rows:
        if not bool(r.get("nominal", False)):
            continue
        key = (int(r.get("bucket", -1)), str(r.get("scene", "")), int(r.get("time", 0)))
        out[key] = r
    return out


def _row_terms(r: dict[str, Any], n: dict[str, Any], tol: ComponentVetoTolerances) -> np.ndarray:
    return component_veto_terms_numpy(
        candidate_drs=float(r["teacher_drs"]), nominal_drs=float(n["teacher_drs"]),
        candidate_r_dep=float(r["teacher_r_dep"]), nominal_r_dep=float(n["teacher_r_dep"]),
        candidate_gap=float(r["teacher_gap"]), nominal_gap=float(n["teacher_gap"]),
        candidate_hard=float(r.get("teacher_hard_violation", 0.0)), nominal_hard=float(n.get("teacher_hard_violation", 0.0)),
        candidate_harm_proxy=float(r.get("teacher_harm_proxy", 0.0)), nominal_harm_proxy=float(n.get("teacher_harm_proxy", 0.0)),
        tolerances=tol,
    )


def _audit_index(path: Path, deployable_macros: set[int]) -> dict[str, Any]:
    rows = _read_jsonl(path)
    nominals = _group_nominals(rows)
    tol = ComponentVetoTolerances()
    by_regime: dict[str, dict[str, Any]] = {}

    accum: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "deployable_candidates": 0,
        "legacy_beneficial": 0,
        "legacy_harmful": 0,
        "legacy_overlap": 0,
        "legacy_safe_beneficial": 0,
        "overlap_culprits": Counter(),
        "overlap_max_culprit": Counter(),
        "no_gap_overlap": 0,
        "rdep_rescue": 0,
        "rdep_rescue_harmful": 0,
        "rdep_rescue_safe": 0,
        "rdep_rescue_rdep_half": 0,
        "rdep_rescue_macro": Counter(),
        "rdep_gain_005": 0,
        "rdep_gain_005_harmful": 0,
        "groups_legacy_safe": set(),
        "groups_rdep_rescue_safe": set(),
        "scenes_rdep_rescue_safe": set(),
        "gap_raw_tolerance_equiv": [],
        "benefit_drs_gain": [],
        "benefit_rdep_gain": [],
        "benefit_gap_change": [],
    })

    for r in rows:
        if bool(r.get("nominal", False)) or int(r.get("macro", -1)) not in deployable_macros:
            continue
        key = (int(r.get("bucket", -1)), str(r.get("scene", "")), int(r.get("time", 0)))
        n = nominals.get(key)
        if n is None:
            continue
        reg = _regime(r)
        a = accum[reg]
        a["deployable_candidates"] += 1
        terms = _row_terms(r, n, tol)
        harmful = bool(np.max(terms) > 0.0)
        legacy_beneficial = float(r["teacher_pcd"]) - float(n["teacher_pcd"]) >= 0.015
        if bool(r.get("beneficial", legacy_beneficial)) != legacy_beneficial:
            raise RuntimeError(f"index beneficial field disagrees with legacy PCD definition: {r.get('path')}")
        if bool(r.get("component_harmful", harmful)) != harmful:
            raise RuntimeError(f"index harmful field disagrees with component-veto definition: {r.get('path')}")
        a["legacy_beneficial"] += int(legacy_beneficial)
        a["legacy_harmful"] += int(harmful)
        overlap = legacy_beneficial and harmful
        a["legacy_overlap"] += int(overlap)
        a["legacy_safe_beneficial"] += int(legacy_beneficial and not harmful)
        if legacy_beneficial and not harmful:
            a["groups_legacy_safe"].add(key)
        if overlap:
            names = ["drs", "deployability", "gap", "hard", "harm_proxy"]
            positive = [names[i] for i, x in enumerate(terms) if float(x) > 0.0]
            a["overlap_culprits"].update(positive)
            a["overlap_max_culprit"].update([names[int(np.argmax(terms))]])
            if float(np.max(np.delete(terms, 2))) > 0.0:
                a["no_gap_overlap"] += 1
            a["benefit_drs_gain"].append(float(r["teacher_drs"]) - float(n["teacher_drs"]))
            a["benefit_rdep_gain"].append(float(r["teacher_r_dep"]) - float(n["teacher_r_dep"]))
            a["benefit_gap_change"].append(float(r["teacher_gap"]) - float(n["teacher_gap"]))

        # Paper-native material rescue screen: nominal is below the physical R_dep=0
        # boundary and the candidate crosses it.  This is audit-only in v48.56.
        rescue = float(n["teacher_r_dep"]) < 0.0 <= float(r["teacher_r_dep"])
        if rescue:
            a["rdep_rescue"] += 1
            a["rdep_rescue_harmful"] += int(harmful)
            a["rdep_rescue_safe"] += int(not harmful)
            a["rdep_rescue_rdep_half"] += int(abs(float(r["teacher_r_dep"]) - 0.5) <= 1e-7)
            a["rdep_rescue_macro"].update([MACRO_NAMES.get(int(r.get("macro", -1)), str(r.get("macro", -1)))])
            if not harmful:
                a["groups_rdep_rescue_safe"].add(key)
                a["scenes_rdep_rescue_safe"].add((key[0], key[1]))

        dep_gain = _sigmoid(float(r["teacher_r_dep"])) - _sigmoid(float(n["teacher_r_dep"]))
        if dep_gain >= 0.05:
            a["rdep_gain_005"] += 1
            a["rdep_gain_005_harmful"] += int(harmful)

        # A fixed 0.05 quality drop exp(-g_nom)-exp(-g_cand) is not a fixed raw-gap
        # increment.  Record the raw gap increment which would exactly hit the veto.
        qn = _gap_quality(float(n["teacher_gap"]))
        if qn > 0.05:
            gcross = -math.log(max(1e-12, qn - 0.05))
            a["gap_raw_tolerance_equiv"].append(max(0.0, gcross - max(0.0, float(n["teacher_gap"]))))

    for reg, a in accum.items():
        def med(xs: Iterable[float]) -> float | None:
            vals = list(xs)
            return float(np.median(vals)) if vals else None
        rescue = int(a["rdep_rescue"])
        legacy_ben = int(a["legacy_beneficial"])
        by_regime[reg] = {
            "deployable_candidates": int(a["deployable_candidates"]),
            "legacy_pcd": {
                "beneficial_candidates": legacy_ben,
                "component_harmful_candidates": int(a["legacy_harmful"]),
                "beneficial_and_harmful_candidates": int(a["legacy_overlap"]),
                "beneficial_harm_conflict_fraction": (float(a["legacy_overlap"]) / legacy_ben) if legacy_ben else None,
                "safe_beneficial_candidates": int(a["legacy_safe_beneficial"]),
                "safe_beneficial_groups": len(a["groups_legacy_safe"]),
                "overlap_culprit_counts": dict(a["overlap_culprits"]),
                "overlap_max_culprit_counts": dict(a["overlap_max_culprit"]),
                "overlap_remaining_if_gap_not_hard_veto": int(a["no_gap_overlap"]),
                "overlap_median_drs_gain": med(a["benefit_drs_gain"]),
                "overlap_median_rdep_gain": med(a["benefit_rdep_gain"]),
                "overlap_median_gap_change": med(a["benefit_gap_change"]),
            },
            "paper_native_rdep_screens": {
                "rdep_zero_cross_rescue_candidates": rescue,
                "rdep_zero_cross_harmful_candidates": int(a["rdep_rescue_harmful"]),
                "rdep_zero_cross_safe_candidates": int(a["rdep_rescue_safe"]),
                "rdep_zero_cross_safe_groups": len(a["groups_rdep_rescue_safe"]),
                "rdep_zero_cross_safe_scenes": len(a["scenes_rdep_rescue_safe"]),
                "rdep_zero_cross_exact_0p5_fraction": (float(a["rdep_rescue_rdep_half"]) / rescue) if rescue else None,
                "rdep_zero_cross_macro_counts": dict(a["rdep_rescue_macro"]),
                "sigmoid_rdep_gain_ge_0p05_candidates": int(a["rdep_gain_005"]),
                "sigmoid_rdep_gain_ge_0p05_harmful": int(a["rdep_gain_005_harmful"]),
            },
            "gap_geometry": {
                "quality_direction": "higher exp(-gap) is safer; direction is mathematically monotone/correct",
                "raw_gap_increment_equivalent_to_quality_drop_0p05_median": med(a["gap_raw_tolerance_equiv"]),
                "raw_gap_increment_equivalent_to_quality_drop_0p05_p10": float(np.quantile(a["gap_raw_tolerance_equiv"], 0.1)) if a["gap_raw_tolerance_equiv"] else None,
                "raw_gap_increment_equivalent_to_quality_drop_0p05_p90": float(np.quantile(a["gap_raw_tolerance_equiv"], 0.9)) if a["gap_raw_tolerance_equiv"] else None,
            },
        }

    freshness_keys = {
        "r_dep": "fresh_ocmero_r_dep_abs_error",
        "r_orc": "fresh_ocmero_r_orc_abs_error",
        "gap": "fresh_ocmero_gap_abs_error",
    }
    freshness_available = bool(rows) and all(
        all(key in r for key in freshness_keys.values()) for r in rows
    )
    freshness = None
    if freshness_available:
        atol = 1e-6
        max_err = {
            name: max((float(r.get(key, 0.0)) for r in rows), default=0.0)
            for name, key in freshness_keys.items()
        }
        mismatch = {
            name: sum(float(r.get(key, 0.0)) > atol for r in rows)
            for name, key in freshness_keys.items()
        }
        missing = {
            "r_dep": sum(not bool(r.get("cached_r_dep_present", False)) for r in rows),
            "r_orc": sum(not bool(r.get("cached_r_orc_present", False)) for r in rows),
        }
        freshness = {
            "source": "teacher_index_inline_fresh_ocmero",
            "checked_samples": len(rows),
            "atol": atol,
            "max_abs_error": max_err,
            "mismatch_counts": mismatch,
            "missing_cached_label_counts": missing,
            "source_labels_match_fresh_ocmero": (
                all(int(v) == 0 for v in mismatch.values())
                and all(int(v) == 0 for v in missing.values())
            ),
        }

    return {
        "index": str(path.resolve()),
        "rows": len(rows),
        "groups": len({(int(r.get("bucket", -1)), str(r.get("scene", "")), int(r.get("time", 0))) for r in rows}),
        "by_regime": by_regime,
        "source_recomputation": freshness,
    }


def _combine_index_freshness(parts: list[dict[str, Any]]) -> dict[str, Any] | None:
    recs = [p.get("source_recomputation") for p in parts if p.get("source_recomputation")]
    if not recs:
        return None
    keys = ("r_dep", "r_orc", "gap")
    return {
        "source": "teacher_index_inline_fresh_ocmero",
        "checked_samples": sum(int(r.get("checked_samples", 0)) for r in recs),
        "atol": min(float(r.get("atol", 1e-6)) for r in recs),
        "max_abs_error": {
            k: max(float(r.get("max_abs_error", {}).get(k, 0.0)) for r in recs) for k in keys
        },
        "mismatch_counts": {
            k: sum(int(r.get("mismatch_counts", {}).get(k, 0)) for r in recs) for k in keys
        },
        "missing_cached_label_counts": {
            k: sum(int(r.get("missing_cached_label_counts", {}).get(k, 0)) for r in recs)
            for k in ("r_dep", "r_orc")
        },
        "source_labels_match_fresh_ocmero": all(
            bool(r.get("source_labels_match_fresh_ocmero", False)) for r in recs
        ),
    }


def _scalar(d: dict[str, Any], key: str, default: Any) -> Any:
    a = np.asarray(d.get(key, default))
    return a.item() if a.shape == () else a


def _audit_dataset_sources(dataset: str, *, alpha: float, beta: float, top_m: int, max_samples: int, atol: float) -> dict[str, Any]:
    paths = iter_sample_paths_many(dataset)
    if max_samples > 0:
        paths = paths[:max_samples]
    max_err = {"r_dep": 0.0, "r_orc": 0.0, "gap": 0.0}
    mismatch = Counter()
    missing = Counter()
    checked = 0
    for p in paths:
        d = load_npz_selected(p, MODEL_SAMPLE_NPZ_KEYS)
        m = np.asarray(d["m_star"], dtype=np.float64)
        prob = np.asarray(d["root_probs"], dtype=np.float64)
        c = np.asarray(d.get("c_star", np.eye(m.shape[0])), dtype=np.float64)
        rv = np.asarray(d.get("root_valid", np.ones(m.shape[0])), dtype=bool)
        ov = np.asarray(d.get("option_valid", np.ones(m.shape[1])), dtype=bool)
        res = oc_mero(m, prob, c, alpha=alpha, beta=beta, option_valid=ov, root_valid=rv, use_lcvar=True, use_obs_kernel=True, top_m=top_m)
        has_r_dep = "r_dep_star" in d
        has_r_orc = "r_orc_star" in d
        missing["r_dep"] += int(not has_r_dep)
        missing["r_orc"] += int(not has_r_orc)
        r_dep_stored = float(_scalar(d, "r_dep_star", res.r_dep))
        r_orc_stored = float(_scalar(d, "r_orc_star", res.r_orc))
        stored = {
            "r_dep": r_dep_stored,
            "r_orc": r_orc_stored,
            "gap": max(0.0, r_orc_stored - r_dep_stored),
        }
        fresh = {"r_dep": float(res.r_dep), "r_orc": float(res.r_orc), "gap": max(0.0, float(res.r_orc-res.r_dep))}
        for k in max_err:
            e = abs(stored[k] - fresh[k])
            max_err[k] = max(max_err[k], e)
            mismatch[k] += int(e > atol)
        checked += 1
    return {
        "dataset": dataset,
        "checked_samples": checked,
        "atol": atol,
        "max_abs_error": max_err,
        "mismatch_counts": dict(mismatch),
        "missing_cached_label_counts": dict(missing),
        "source_labels_match_fresh_ocmero": (
            all(int(mismatch[k]) == 0 for k in max_err)
            and int(missing["r_dep"]) == 0
            and int(missing["r_orc"]) == 0
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-index", type=Path, required=True)
    ap.add_argument("--dev-index", type=Path)
    ap.add_argument("--dataset", default="", help="Optional comma-separated source dataset roots for fresh OC-MERO label verification")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--deployable-macro-ids", default="2,3,5,6,7")
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--beta", type=float, default=0.2)
    ap.add_argument("--top-m", type=int, default=8)
    ap.add_argument("--max-source-samples", type=int, default=0)
    ap.add_argument("--source-atol", type=float, default=1e-6)
    args = ap.parse_args()
    _reject_test_inputs(str(args.train_index), str(args.dev_index or ""), str(args.dataset))

    macros = {int(x.strip()) for x in str(args.deployable_macro_ids).split(",") if x.strip()}
    train_audit = _audit_index(args.train_index, macros)
    dev_audit = _audit_index(args.dev_index, macros) if args.dev_index else None
    report: dict[str, Any] = {
        "event": "v48_56_teacher_component_semantic_audit",
        "version": "v48.56-DCP-DRFC-BCDE-TCSA",
        "policy_regime_conditioning": False,
        "test_roots_read": False,
        "questions": {
            "dep_target": "current DEP is nominal-relative sigmoid(R_dep) degradation, not the paper-core absolute deployable-recovery admission event",
            "gap_target": "exp(-gap) has the correct monotone direction; fixed quality-space tolerance is baseline-dependent in raw-gap space and GAP decision role must be audited before treating it as a hard veto",
            "teacher_source": "DRS uses observation-class option semantics; cached R_dep/R_orc must match fresh OC-MERO on the stored m_star contract",
            "component_semantics": "legacy PCD is compensatory whereas component_veto is non-compensatory; overlap is a direct contradictory-supervision diagnostic",
        },
        "train": train_audit,
    }
    if dev_audit is not None:
        report["dev"] = dev_audit
    if args.dataset:
        report["source_recomputation"] = _audit_dataset_sources(
            args.dataset, alpha=args.alpha, beta=args.beta, top_m=args.top_m,
            max_samples=args.max_source_samples, atol=args.source_atol,
        )
    else:
        inline = _combine_index_freshness([x for x in [train_audit, dev_audit] if x is not None])
        if inline is not None:
            report["source_recomputation"] = inline

    # Fail-closed research decision: no evidence centering while the legacy target
    # has material benefit/harm contradictions or source labels are stale.
    conflicts = []
    for split in ["train", "dev"]:
        for reg, rec in report.get(split, {}).get("by_regime", {}).items():
            frac = rec["legacy_pcd"]["beneficial_harm_conflict_fraction"]
            n_overlap = int(rec["legacy_pcd"]["beneficial_and_harmful_candidates"])
            if frac is not None and n_overlap > 0:
                conflicts.append(
                    f"{split}:{reg}:legacy_benefit_harm_conflict={n_overlap};fraction={frac:.6f}"
                )
    src = report.get("source_recomputation")
    if src and not bool(src.get("source_labels_match_fresh_ocmero", False)):
        conflicts.append("cached_rdep_rorc_do_not_match_fresh_ocmero")
    report["decision"] = {
        "boundary_complete_evidence_centering_authorized": False,
        "component_normalization_family_stop": True,
        "teacher_component_correctness_audit_required": True,
        "semantic_conflicts": conflicts,
        "next_research_gate": "audit/rebuild teacher margin source semantics before changing final evidence centering or root-logit calibration",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(args.output)
    print(json.dumps({"event": report["event"], "output": str(args.output), **report["decision"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
