#!/usr/bin/env python3
"""Mine publication-quality OC-RAP motivation/toy cases from existing dataset buckets.

Design goals
------------
1. Make the oracle-to-deployable gap visually *causal* rather than merely showing
   a large scalar gap.  The primary oracle gallery requires an observation-
   compatible root pair for which each root has a positive branch-wise recovery
   but no positive option is shared by the pair.  We additionally compute the
   exact soft-compatibility inner OC-MERO ``max-after-tail`` value and the
   corresponding branch-wise ``max-before-tail`` oracle value for a displayed
   observation anchor.
2. Keep the selector deployable-information clean.  Selection uses only stored
   dataset teacher/audit tensors and observed history; no model checkpoint,
   hidden future trajectory, regime router, or dataset reconstruction is used.
3. Be cheap enough to scan full validation buckets.  Candidate mining loads only
   the handful of NPZ members needed by each test.  Large map/history tensors are
   decompressed only for the final selected cases, once per case.
4. Fail closed for paper figures.  By default, weak high-gap fallbacks are not
   silently mixed into the oracle gallery.  ``--allow-oracle-fallback`` is an
   explicit opt-in for exploratory browsing.

Outputs per selected case
-------------------------
* history_5frame.png        observed history only;
* candidate_cv_5frame.png   selected candidate prefix + observation-only CV
                            context; for reserve/debt examples, nominal prefix is
                            also drawn for direct action comparison;
* toy_ambiguity_matrix.png  compact reviewer-facing proof of the oracle gap;
* signed_recovery_delta.png one-glance reserve/debt delta for Near/Contact;
* metadata.json.

The root/option matrices are teacher-side diagnostics used only to explain the
motivation.  They are not deployed planner inputs.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
from functools import lru_cache
import math
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

_RENDER_SCHEMA = "iclr-toy-v3"
_VISUAL_RERANK_BUDGET = 192

_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np

from ocrap.algorithms.lcv import normalize_weights, weighted_lcvar
from ocrap.algorithms.ocmero import oc_mero
from ocrap.data.serialization import load_npz_selected


# Small NPZ members used while mining. np.savez_compressed stores arrays in
# independent ZIP members, so this avoids decompressing maps/history for every
# artifact candidate.
_ORACLE_SCAN_KEYS = frozenset({
    "root_probs",
    "root_valid",
    "m_star",
    "c_star",
    "option_valid",
    "recovery_modes",
    "prefix_macro_name",
    # Lightweight provenance/semantic fields.  These let the paper figure say
    # what the latent roots mean instead of showing anonymous root ids.
    "future_metadata",
    "future_probs",
    "root_assignments",
})
_PROGRESS_SCAN_KEYS = frozenset({
    "feasible",
    "hard_violation",
    "harm_proxy",
    "prefix_macro_name",
})
_RENDER_KEYS = frozenset({
    "agent_history",
    "agent_valid",
    "prefix_states",
    "map_polylines",
    "map_valid",
    "route",
    "root_probs",
    "root_valid",
    "m_star",
    "c_star",
    "option_valid",
    "recovery_modes",
    "future_metadata",
    "future_probs",
    "root_assignments",
})
_NOMINAL_RENDER_KEYS = frozenset({"prefix_states"})


@dataclass
class Case:
    case_type: str
    dataset_role: str
    path: str
    scene_id: str
    time_index: int
    candidate_index: int
    macro: str
    r_dep: float
    r_orc: float
    gap: float
    score: float
    nominal_path: str | None = None
    nominal_candidate_index: int | None = None
    nominal_r_dep: float | None = None
    delta_r_dep_vs_nominal: float | None = None
    # Reviewer-facing oracle proof.
    proof_grade: str | None = None
    anchor_root: int | None = None
    local_oracle_margin: float | None = None
    local_deployable_margin: float | None = None
    local_gap: float | None = None
    conflict_root_i: int | None = None
    conflict_root_j: int | None = None
    compatibility: float | None = None
    pair_anchor_mass: float | None = None
    option_i: int | None = None
    option_j: int | None = None
    best_shared_option: int | None = None
    pair_oracle_margin: float | None = None
    pair_shared_margin: float | None = None
    strong_pair_conflict: bool | None = None
    recomputed_r_dep: float | None = None
    recomputed_r_orc_legacy: float | None = None
    teacher_recompute_error: float | None = None
    anchor_compat_mass: float | None = None
    margin_override_future_count: int = 0
    artifact_branch_labels: str = ""
    observed_nearest_m: float | None = None
    observed_closing_m: float | None = None
    candidate_nominal_separation_m: float | None = None
    visual_score: float = 0.0
    notes: str = ""


def _f(x: Any, default: float = math.nan) -> float:
    try:
        v = float(np.asarray(x).item())
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _i(x: Any, default: int = -1) -> int:
    try:
        return int(round(float(np.asarray(x).item())))
    except Exception:
        return default


def _s(x: Any, default: str = "") -> str:
    try:
        return str(np.asarray(x).item())
    except Exception:
        return default


def _read_manifest(root: Path, role: str) -> list[dict[str, Any]]:
    """Read only the manifest fields used by this tool.

    The historical implementation retained every CSV string via ``**row``.
    Validation buckets are not enormous, but a compact row representation keeps
    memory bounded and makes the selector's dependency contract explicit.
    """
    p = root / "manifest.csv"
    if not p.is_file():
        raise FileNotFoundError(f"missing manifest: {p}")
    rows: list[dict[str, Any]] = []
    with p.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rel = str(row.get("path") or "")
            sample = root / rel
            rows.append({
                "_role": role,
                "_path": str(sample),
                "_r_dep": _f(row.get("r_dep_star")),
                "_r_orc": _f(row.get("r_orc_star")),
                "_gap": _f(row.get("oracle_gap_star")),
                "_art": bool(_i(row.get("i_art_star"), 0)),
                "_time": _i(row.get("time_index")),
                "_cand": _i(row.get("candidate_index")),
                "_nom": bool(_i(row.get("is_nominal"), 0)),
                "_scene": str(row.get("scene_id") or row.get("original_scenario_id") or ""),
            })
    return rows



def _json_field(x: Any, default: Any) -> Any:
    """Parse JSON saved as a scalar numpy string without trusting its shape."""
    try:
        raw = np.asarray(x).item()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        if isinstance(raw, str):
            return json.loads(raw)
        return raw
    except Exception:
        return default


def _artifact_provenance(d: dict[str, Any]) -> tuple[int, str]:
    """Return override count and human-readable hidden-branch labels.

    The current Near/Contact artifact pass deliberately mines a hidden yield /
    accelerate pair and may apply the configured stress-margin override.  A
    publication figure should expose this provenance rather than letting a
    reviewer infer that +/-6 values are raw physical measurements.
    """
    metas = _json_field(d.get("future_metadata", "[]"), [])
    if not isinstance(metas, list):
        return 0, ""
    override_count = 0
    labels: list[str] = []
    for meta in metas:
        if not isinstance(meta, dict):
            continue
        override_count += int(bool(meta.get("margin_override_applied", False)))
        branch = str(meta.get("artifact_branch") or meta.get("hidden_intent") or "").strip()
        targeted = str(meta.get("targeted_type") or "").strip()
        if branch == "yield":
            label = "hidden vehicle yields"
        elif branch == "accelerate":
            label = "hidden vehicle accelerates"
        elif targeted:
            label = targeted.replace("waymax_", "").replace("_", " ")
        else:
            continue
        if label not in labels:
            labels.append(label)
    return int(override_count), " | ".join(labels[:4])


def _root_semantic_labels(d: dict[str, Any], K: int) -> list[str]:
    """Map latent-root ids to the most useful future semantics stored in NPZ."""
    labels = [f"root {k}" for k in range(K)]
    metas = _json_field(d.get("future_metadata", "[]"), [])
    ra = np.asarray(d.get("root_assignments", []), dtype=int).reshape(-1)
    if not isinstance(metas, list) or ra.size == 0:
        return labels
    bucket: dict[int, list[str]] = {k: [] for k in range(K)}
    for fi, meta in enumerate(metas[: ra.size]):
        if not isinstance(meta, dict):
            continue
        r = int(ra[fi])
        if r < 0 or r >= K:
            continue
        branch = str(meta.get("artifact_branch") or meta.get("hidden_intent") or "").strip()
        targeted = str(meta.get("targeted_type") or "").strip()
        if branch == "yield":
            txt = "hidden vehicle yields"
        elif branch == "accelerate":
            txt = "hidden vehicle accelerates"
        elif targeted:
            txt = targeted.replace("waymax_", "").replace("_", " ")
        else:
            continue
        if txt not in bucket[r]:
            bucket[r].append(txt)
    for r, vals in bucket.items():
        if vals:
            labels[r] = vals[0]
    return labels


def _option_semantic_family(name: str) -> str:
    x = str(name).lower()
    if any(k in x for k in ("stop", "brake")):
        return "stop/brake"
    if any(k in x for k in ("lateral", "pull_over", "yield", "rejoin")):
        return "lateral/yield"
    if "post_contact" in x or "stabil" in x:
        return "stabilize"
    if "avoid_secondary" in x or "mitigate" in x:
        return "contact-mitigation"
    return x or "unknown"


def _semantic_conflict_bonus(modes: Sequence[str], oi: int | None, oj: int | None) -> float:
    if oi is None or oj is None or not (0 <= int(oi) < len(modes) and 0 <= int(oj) < len(modes)):
        return 0.0
    a, b = _option_semantic_family(modes[int(oi)]), _option_semantic_family(modes[int(oj)])
    if a == b:
        return -1.0
    bonus = 2.0
    if {a, b} == {"stop/brake", "lateral/yield"}:
        bonus += 2.5
    # For a first-page Near toy, "post-contact stabilize" is semantically less
    # immediate than a stop-vs-lateral conflict even if its numeric gap is large.
    if "stabilize" in {a, b}:
        bonus -= 1.0
    return bonus


def _observed_visual_metrics(path: Path, nominal_path: Path | None = None) -> tuple[float, float, float | None, float]:
    """Cheap observation-only legibility score for the small top-ranked pool."""
    keys = frozenset({"agent_history", "agent_valid", "prefix_states"})
    try:
        d = load_npz_selected(path, keys)
    except Exception:
        return math.nan, 0.0, None, 0.0
    hist = np.asarray(d.get("agent_history", np.zeros((0, 0, 16))), dtype=float)
    valid = np.asarray(d.get("agent_valid", np.zeros(hist.shape[:2] if hist.ndim == 3 else (0, 0))), dtype=bool)
    nearest = math.nan
    closing = 0.0
    score = 0.0
    if hist.ndim == 3 and hist.shape[0] >= 2 and hist.shape[1] >= 2 and valid.shape == hist.shape[:2]:
        t0, t1 = 0, hist.shape[0] - 1
        ds0: dict[int, float] = {}
        ds1: dict[int, float] = {}
        for a in range(1, hist.shape[1]):
            if valid[t1, 0] and valid[t1, a] and np.isfinite(hist[t1, [0, a], :2]).all():
                ds1[a] = float(np.linalg.norm(hist[t1, a, :2] - hist[t1, 0, :2]))
            if valid[t0, 0] and valid[t0, a] and np.isfinite(hist[t0, [0, a], :2]).all():
                ds0[a] = float(np.linalg.norm(hist[t0, a, :2] - hist[t0, 0, :2]))
        if ds1:
            a_near = min(ds1, key=ds1.get)
            nearest = ds1[a_near]
            score += max(0.0, min(2.0, (24.0 - nearest) / 8.0))
            if a_near in ds0:
                closing = max(0.0, ds0[a_near] - ds1[a_near])
                score += min(1.5, closing / 4.0)
            nearby = sum(dv < 25.0 for dv in ds1.values())
            score += min(1.0, 0.20 * nearby)
        if valid[t0, 0] and valid[t1, 0]:
            ego_move = float(np.linalg.norm(hist[t1, 0, :2] - hist[t0, 0, :2]))
            score += min(0.8, ego_move / 8.0)
    sep: float | None = None
    prefix = np.asarray(d.get("prefix_states", np.zeros((0, 9))), dtype=float)
    if nominal_path is not None and nominal_path.is_file() and prefix.ndim == 2 and len(prefix):
        try:
            nd = load_npz_selected(nominal_path, frozenset({"prefix_states"}))
            nom = np.asarray(nd.get("prefix_states", np.zeros((0, 9))), dtype=float)
            n = min(len(prefix), len(nom))
            if n:
                sep = float(np.nanmax(np.linalg.norm(prefix[:n, :2] - nom[:n, :2], axis=1)))
                score += min(2.5, sep / 1.5)
        except Exception:
            pass
    return nearest, closing, sep, float(score)


def _rerank_for_paper(cases: list[Case], budget: int = _VISUAL_RERANK_BUDGET) -> list[Case]:
    """Re-rank only a bounded top pool by reviewer legibility, not just gap size."""
    if not cases:
        return cases
    ordered = sorted(cases, key=lambda x: x.score, reverse=True)
    for c in ordered[: max(0, int(budget))]:
        nearest, closing, sep, vis = _observed_visual_metrics(
            Path(c.path), Path(c.nominal_path) if c.nominal_path else None
        )
        c.observed_nearest_m = nearest if math.isfinite(nearest) else None
        c.observed_closing_m = float(closing)
        c.candidate_nominal_separation_m = sep
        c.visual_score = float(vis)
        c.score += 2.0 * float(vis)
        # Large candidate-vs-nominal separation is especially important for the
        # reserve/debt panels; otherwise the action difference disappears at paper scale.
        if c.case_type in {"near_reserve", "contact_debt"} and sep is not None:
            c.score += min(4.0, float(sep))
    return sorted(ordered, key=lambda x: x.score, reverse=True)

def _masked_root_best(m: np.ndarray, valid_opts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vals = np.where(valid_opts[None, :], m, -1e9)
    idx = np.argmax(vals, axis=1)
    best = vals[np.arange(vals.shape[0]), idx]
    return best.astype(np.float64), idx.astype(np.int64)


def _lcvar_influence(scores: np.ndarray, weights: np.ndarray, alpha: float) -> np.ndarray:
    """Exact stable-sort fractional lower-tail influence used for explanation."""
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    w = normalize_weights(weights)
    out = np.zeros_like(s, dtype=np.float64)
    order = np.argsort(s, kind="mergesort")
    remaining = float(alpha)
    for idx in order:
        take = min(float(w[idx]), remaining)
        if take > 0:
            out[idx] = take / float(alpha)
            remaining -= take
        if remaining <= 1e-12:
            break
    return out


def _oracle_gap_proof(
    d: dict[str, Any],
    *,
    beta: float,
    min_root_prob: float,
    min_compat: float,
) -> dict[str, Any] | None:
    """Find a reviewer-legible local witness for the oracle/deployable mismatch.

    For an observation anchor ``i`` we compare

        oracle_i = LCVaR_beta( max_l M[j,l] ; w_i[j] )
        deploy_i = max_l LCVaR_beta( M[j,l] ; w_i[j] )

    where ``w_i`` is the same compatibility-weighted root distribution used by
    OC-MERO.  ``oracle_i > 0 > deploy_i`` is a direct local sign-flip caused by
    moving option selection inside versus outside the compatible-root tail.

    We then seek a concrete compatible pair j/k with positive per-root best
    options but no positive shared option.  That pair is the compact hard-limit
    proof shown to the reviewer.
    """
    p = np.asarray(d.get("root_probs", []), dtype=np.float64).reshape(-1)
    m = np.asarray(d.get("m_star", []), dtype=np.float64)
    c = np.asarray(d.get("c_star", []), dtype=np.float64)
    rv = np.asarray(d.get("root_valid", np.ones_like(p)), dtype=bool).reshape(-1)
    ov = np.asarray(
        d.get("option_valid", np.ones(m.shape[1] if m.ndim == 2 else 0)),
        dtype=bool,
    ).reshape(-1)
    if m.ndim != 2 or p.size == 0 or m.shape[0] != p.size or c.shape != (p.size, p.size):
        return None
    K, L = m.shape
    valid_opts = ov[:L]
    if not valid_opts.any():
        return None
    valid_roots = rv[:K] & np.isfinite(p[:K]) & (p[:K] > 0.0)
    if int(valid_roots.sum()) < 2:
        return None
    p_eff = np.where(valid_roots, p[:K], 0.0)
    p_norm = normalize_weights(p_eff)
    root_best, root_best_opt = _masked_root_best(m, valid_opts)
    modes = [str(x) for x in np.asarray(d.get("recovery_modes", np.arange(L))).reshape(-1).tolist()]

    best_rec: dict[str, Any] | None = None
    for anchor in np.flatnonzero(valid_roots):
        compat_row = np.nan_to_num(c[int(anchor), :K], nan=0.0, posinf=1.0, neginf=0.0)
        w = normalize_weights(np.clip(compat_row, 0.0, None) * p_norm)
        q = np.full(L, -1e9, dtype=np.float64)
        for ell in np.flatnonzero(valid_opts):
            q[int(ell)] = weighted_lcvar(m[:, int(ell)], w, beta)
        shared_opt = int(np.argmax(q))
        deploy = float(q[shared_opt])
        oracle = float(weighted_lcvar(root_best, w, beta))
        local_gap = float(oracle - deploy)
        sign_flip = bool(oracle > 0.0 and deploy < 0.0)

        # Tail roots matter more than merely high-probability compatible roots.
        oracle_tail = _lcvar_influence(root_best, w, beta)
        pair_best: dict[str, Any] | None = None
        roots = np.flatnonzero(
            valid_roots
            & (p_norm >= float(min_root_prob))
            & (w >= min(float(min_root_prob), 0.02))
        )
        for ai in range(len(roots)):
            j = int(roots[ai])
            bj = float(root_best[j]); oj = int(root_best_opt[j])
            if not math.isfinite(bj) or bj <= 0.0:
                continue
            for bi in range(ai + 1, len(roots)):
                k = int(roots[bi])
                bk = float(root_best[k]); ok = int(root_best_opt[k])
                if not math.isfinite(bk) or bk <= 0.0 or oj == ok:
                    continue
                pair_compat = float(c[j, k]) if np.isfinite(c[j, k]) else 0.0
                if pair_compat < float(min_compat):
                    continue
                pair_min_by_opt = np.minimum(m[j, :L], m[k, :L])
                pair_min_by_opt = np.where(valid_opts, pair_min_by_opt, -1e9)
                common_opt = int(np.argmax(pair_min_by_opt))
                shared = float(pair_min_by_opt[common_opt])
                pair_oracle = float(min(bj, bk))
                strict = bool(pair_oracle > 0.0 and shared <= 0.0)
                pair_mass = float(w[j] + w[k])
                both_in_oracle_tail = bool(oracle_tail[j] > 0.0 and oracle_tail[k] > 0.0)
                # Lexicographic intent encoded as a scalar only for sorting:
                # strict conflict > both tail roots > compatibility/mass/depth.
                pair_score = (
                    (10.0 if strict else 0.0)
                    + (3.0 if both_in_oracle_tail else 0.0)
                    + 2.0 * pair_compat
                    + 2.0 * pair_mass
                    + max(0.0, -shared)
                    + 0.25 * (bj + bk)
                    + 1.5 * _semantic_conflict_bonus(modes, oj, ok)
                )
                rec = {
                    "root_i": j,
                    "root_j": k,
                    "option_i": oj,
                    "option_j": ok,
                    "best_shared_option": common_opt,
                    "best_margin_i": bj,
                    "best_margin_j": bk,
                    "pair_oracle_margin": pair_oracle,
                    "pair_shared_margin": shared,
                    "compatibility": pair_compat,
                    "pair_anchor_mass": pair_mass,
                    "strong_pair_conflict": strict,
                    "both_in_oracle_tail": both_in_oracle_tail,
                    "pair_score": pair_score,
                }
                if pair_best is None or pair_score > float(pair_best["pair_score"]):
                    pair_best = rec

        # Grade A is the cleanest paper example: exact local soft-OC-MERO sign
        # flip plus a hard-limit pair with incompatible positive recoveries.
        if sign_flip and pair_best is not None and bool(pair_best["strong_pair_conflict"]):
            grade = "A"
        elif sign_flip:
            grade = "B"
        elif pair_best is not None and bool(pair_best["strong_pair_conflict"]):
            grade = "C"
        else:
            grade = "D"
        grade_bonus = {"A": 30.0, "B": 18.0, "C": 10.0, "D": 0.0}[grade]
        anchor_mass = float(np.sum(w[compat_row >= float(min_compat)]))
        score = grade_bonus + 3.0 * local_gap + 1.5 * anchor_mass
        if pair_best is not None:
            score += float(pair_best["pair_score"])
        rec = {
            "proof_grade": grade,
            "anchor_root": int(anchor),
            "local_oracle_margin": oracle,
            "local_deployable_margin": deploy,
            "local_gap": local_gap,
            "anchor_compat_mass": anchor_mass,
            "best_shared_option_anchor": shared_opt,
            "proof_score": score,
            "pair": pair_best,
        }
        if best_rec is None or score > float(best_rec["proof_score"]):
            best_rec = rec
    return best_rec


def _oracle_cases(
    rows: list[dict[str, Any]],
    *,
    alpha: float,
    beta: float,
    min_root_prob: float,
    min_compat: float,
    allow_fallback: bool,
    recompute_tol: float,
) -> tuple[list[Case], dict[str, int]]:
    """Mine theorem-valid sign-flip examples and keep pair conflicts as a bonus.

    The exact OC-MERO phenomenon is the *soft* local sign flip

        LCVaR(max_l M) > 0 > max_l LCVaR(M_l).

    A two-root hard conflict is a very useful visualization but is only a
    sufficient special case.  The old selector accidentally treated that
    visualization convenience as a validity requirement, which removed every
    Contact oracle case at --min-compat 0.95.  Here grades A *and* B are
    publication-valid; C/D remain exploratory and still require
    --allow-oracle-fallback.
    """
    cand = [
        r for r in rows
        if r["_art"]
        and r["_r_orc"] >= 0.0
        and r["_r_dep"] < 0.0
        and math.isfinite(r["_gap"])
    ]
    cand.sort(key=lambda r: r["_gap"], reverse=True)
    out: list[Case] = []
    stats = {
        "manifest_artifact_candidates": len(cand),
        "missing_files": 0,
        "invalid_proof_tensors": 0,
        "teacher_recompute_mismatch": 0,
        "grade_A": 0,
        "grade_B": 0,
        "grade_C": 0,
        "grade_D": 0,
        "stress_margin_override_cases": 0,
        "retained": 0,
    }
    for r in cand:
        path = Path(r["_path"])
        if not path.is_file():
            stats["missing_files"] += 1
            continue
        d = load_npz_selected(path, _ORACLE_SCAN_KEYS)
        proof = _oracle_gap_proof(
            d,
            beta=beta,
            min_root_prob=min_root_prob,
            min_compat=min_compat,
        )
        if proof is None:
            stats["invalid_proof_tensors"] += 1
            continue

        try:
            recalc = oc_mero(
                np.asarray(d["m_star"], dtype=np.float64),
                np.asarray(d["root_probs"], dtype=np.float64),
                np.asarray(d["c_star"], dtype=np.float64),
                alpha=alpha,
                beta=beta,
                option_valid=np.asarray(d.get("option_valid", []), dtype=bool),
                root_valid=np.asarray(d.get("root_valid", []), dtype=bool),
            )
            dep_err = abs(float(recalc.r_dep) - float(r["_r_dep"]))
            if dep_err > float(recompute_tol):
                stats["teacher_recompute_mismatch"] += 1
                continue
        except Exception:
            stats["teacher_recompute_mismatch"] += 1
            continue

        grade = str(proof["proof_grade"])
        stats[f"grade_{grade}"] += 1
        if grade not in {"A", "B"} and not allow_fallback:
            continue
        pair = proof.get("pair") or {}
        macro = _s(d.get("prefix_macro_name", ""))
        modes = [str(x) for x in np.asarray(d.get("recovery_modes", [])).reshape(-1).tolist()]
        override_count, branch_labels = _artifact_provenance(d)
        if override_count > 0:
            stats["stress_margin_override_cases"] += 1

        # Ranking is intentionally not raw-gap-first.  Prefer a robust sign flip,
        # meaningful compatible mass, and human-readable action conflict.  Grade A
        # receives a bonus, but Grade B remains a first-class soft-OC-MERO witness.
        local_orc = float(proof["local_oracle_margin"])
        local_dep = float(proof["local_deployable_margin"])
        anchor_mass = float(proof.get("anchor_compat_mass", 0.0))
        local_gap = float(proof["local_gap"])
        pair_mass = float(pair.get("pair_anchor_mass", 0.0)) if pair else 0.0
        oi = int(pair["option_i"]) if pair else None
        oj = int(pair["option_j"]) if pair else None
        semantic_bonus = _semantic_conflict_bonus(modes, oi, oj)
        grade_bonus = {"A": 16.0, "B": 12.0, "C": 3.0, "D": 0.0}[grade]
        score = (
            grade_bonus
            + 4.0 * min(max(local_orc, 0.0), 1.5)
            + 3.0 * min(max(-local_dep, 0.0), 3.0)
            + 2.0 * min(max(local_gap, 0.0), 3.5)
            + 4.0 * min(max(anchor_mass, 0.0), 1.0)
            + 8.0 * min(max(pair_mass, 0.0), 0.5)
            + semantic_bonus
        )
        # Do not reward the stress override numerically.  It remains valid for a
        # designed toy, but provenance is surfaced and the visual score must carry
        # the scene-level legibility.
        if override_count > 0:
            score -= 0.5

        cse = Case(
            case_type="oracle_gap",
            dataset_role=r["_role"],
            path=str(path),
            scene_id=r["_scene"],
            time_index=r["_time"],
            candidate_index=r["_cand"],
            macro=macro,
            r_dep=float(r["_r_dep"]),
            r_orc=float(r["_r_orc"]),
            gap=float(r["_gap"]),
            score=float(score),
            proof_grade=grade,
            anchor_root=int(proof["anchor_root"]),
            local_oracle_margin=local_orc,
            local_deployable_margin=local_dep,
            local_gap=local_gap,
            anchor_compat_mass=anchor_mass,
            margin_override_future_count=override_count,
            artifact_branch_labels=branch_labels,
            recomputed_r_dep=float(recalc.r_dep),
            recomputed_r_orc_legacy=float(recalc.r_orc),
            teacher_recompute_error=float(dep_err),
            notes=(
                "exact soft OC-MERO sign flip; additionally has a compact two-root hard conflict"
                if grade == "A"
                else "exact soft OC-MERO sign flip across the compatible-root lower tail; no two-root hard witness required"
            ) + ("; stress-mined branches use the configured margin override" if override_count > 0 else ""),
        )
        if pair:
            cse.conflict_root_i = int(pair["root_i"])
            cse.conflict_root_j = int(pair["root_j"])
            cse.compatibility = float(pair["compatibility"])
            cse.pair_anchor_mass = float(pair["pair_anchor_mass"])
            cse.option_i = int(pair["option_i"])
            cse.option_j = int(pair["option_j"])
            cse.best_shared_option = int(pair["best_shared_option"])
            cse.pair_oracle_margin = float(pair["pair_oracle_margin"])
            cse.pair_shared_margin = float(pair["pair_shared_margin"])
            cse.strong_pair_conflict = bool(pair["strong_pair_conflict"])
        else:
            cse.best_shared_option = int(proof.get("best_shared_option_anchor", -1))
        out.append(cse)
    out.sort(key=lambda x: x.score, reverse=True)
    stats["retained"] = len(out)
    return out, stats


def _group_rows(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, int], list[dict[str, Any]]]:
    out: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for r in rows:
        out.setdefault((r["_scene"], r["_time"]), []).append(r)
    return out


def _candidate_prefix_ok(d: dict[str, Any], max_hard: float, max_harm: float) -> bool:
    feasible = bool(_i(d.get("feasible", 1), 1))
    hard = _f(d.get("hard_violation", 0.0), 0.0)
    harm = _f(d.get("harm_proxy", 0.0), 0.0)
    return feasible and hard <= float(max_hard) + 1e-12 and harm <= float(max_harm) + 1e-12


def _progress_cases(
    rows: list[dict[str, Any]],
    kind: str,
    *,
    max_hard: float,
    max_harm: float,
    min_delta: float,
) -> list[Case]:
    out: list[Case] = []
    for (scene, t), group in _group_rows(rows).items():
        nom = next((r for r in group if r["_nom"]), None)
        if nom is None or not math.isfinite(nom["_r_dep"]):
            continue
        ndep = float(nom["_r_dep"])
        if kind == "near_reserve" and not (-0.25 <= ndep <= 0.35):
            continue
        if kind == "contact_debt" and not (ndep < 0.0):
            continue
        alts = [r for r in group if (not r["_nom"]) and math.isfinite(r["_r_dep"])]
        alts.sort(key=lambda r: float(r["_r_dep"] - ndep), reverse=True)
        best: dict[str, Any] | None = None
        best_meta: dict[str, Any] | None = None
        # Historical code tested only the largest-delta alternative and discarded
        # the whole scene-time group if that prefix was unsafe.  Try the next
        # alternative instead; this preserves the intended "best safe recovery"
        # semantics and avoids missing clean paper cases.
        for alt in alts:
            delta = float(alt["_r_dep"] - ndep)
            if delta <= float(min_delta):
                break
            path = Path(alt["_path"])
            if not path.is_file():
                continue
            meta = load_npz_selected(path, _PROGRESS_SCAN_KEYS)
            if not _candidate_prefix_ok(meta, max_hard=max_hard, max_harm=max_harm):
                continue
            best, best_meta = alt, meta
            break
        if best is None or best_meta is None:
            continue
        delta = float(best["_r_dep"] - ndep)
        crossing = bool(float(best["_r_dep"]) >= 0.0 > ndep)
        if kind == "near_reserve":
            # Favor positive headroom, then large improvement, then proximity of
            # nominal to the boundary.  No new hard threshold is introduced.
            positive_bonus = 1.5 if float(best["_r_dep"]) > 0.0 else 0.0
            boundary_bonus = max(0.0, 0.35 - abs(ndep))
            score = 2.0 * delta + positive_bonus + boundary_bonus + 0.10 * max(0.0, float(best["_gap"]))
        else:
            # Contact is recovery debt: crossing zero is especially legible, but
            # strong partial debt repayment remains valid if no crossing exists.
            score = 2.0 * delta + (2.0 if crossing else 0.0) + 0.10 * max(0.0, float(best["_gap"]))
        out.append(Case(
            case_type=kind,
            dataset_role=best["_role"],
            path=str(best["_path"]),
            scene_id=scene,
            time_index=t,
            candidate_index=best["_cand"],
            macro=_s(best_meta.get("prefix_macro_name", "")),
            r_dep=float(best["_r_dep"]),
            r_orc=float(best["_r_orc"]),
            gap=float(best["_gap"]),
            score=float(score),
            nominal_path=str(nom["_path"]),
            nominal_candidate_index=int(nom["_cand"]),
            nominal_r_dep=ndep,
            delta_r_dep_vs_nominal=delta,
            notes=(
                "safe candidate crosses the deployable zero boundary and repays recovery debt"
                if crossing
                else "safe candidate improves signed deployable reserve/debt relative to nominal"
            ),
        ))
    out.sort(key=lambda x: x.score, reverse=True)
    return out


def _diverse_take(cases: list[Case], n: int, seen: set[tuple[str, str]]) -> list[Case]:
    """Take at most one case per physical scene before relaxing anything.

    The old selector only deduplicated (scene,time), so a 24-case gallery often
    spent several slots on adjacent frames from the same WOMD scene.  For a paper
    mining gallery, scene diversity is much more valuable than another 0.1 score
    point from the same interaction.
    """
    if n <= 0:
        return []
    selected: list[Case] = []
    macro_count: dict[str, int] = {}
    ordered = sorted(cases, key=lambda x: x.score, reverse=True)
    for c in ordered:
        key = (c.dataset_role, c.scene_id)
        if key in seen:
            continue
        m = c.macro or "unknown"
        if macro_count.get(m, 0) >= max(2, math.ceil(n / 3)):
            continue
        selected.append(c)
        seen.add(key)
        macro_count[m] = macro_count.get(m, 0) + 1
        if len(selected) >= n:
            return selected
    # Second pass relaxes macro balance, but never scene uniqueness.
    for c in ordered:
        key = (c.dataset_role, c.scene_id)
        if key in seen:
            continue
        selected.append(c)
        seen.add(key)
        if len(selected) >= n:
            break
    return selected


@lru_cache(maxsize=1)
def _mpl():
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt  # type: ignore
    from matplotlib.collections import LineCollection  # type: ignore
    from matplotlib.patches import Rectangle  # type: ignore
    from matplotlib.transforms import Affine2D  # type: ignore
    return plt, Rectangle, Affine2D, LineCollection


def _draw_vehicle(ax, s: np.ndarray, color: str, alpha: float = 0.85, z: int = 5) -> None:
    _, Rectangle, Affine2D, _ = _mpl()
    if s.size < 2 or not np.isfinite(s[:2]).all():
        return
    x, y = float(s[0]), float(s[1])
    heading = float(s[7]) if s.size > 7 and np.isfinite(s[7]) else 0.0
    length = float(s[10]) if s.size > 10 and np.isfinite(s[10]) and s[10] > 0 else 4.8
    width = float(s[11]) if s.size > 11 and np.isfinite(s[11]) and s[11] > 0 else 2.0
    trans = Affine2D().rotate_around(x, y, heading) + ax.transData
    ax.add_patch(Rectangle(
        (x - length / 2, y - width / 2),
        length,
        width,
        transform=trans,
        fc=color,
        ec="k",
        lw=0.45,
        alpha=alpha,
        zorder=z,
    ))


def _prefix_state_to_agent_box(ps: np.ndarray) -> np.ndarray:
    """Map 9-D CandidatePrefix state to the agent-state layout used by renderer.

    CandidatePrefix is [x,y,vx,vy,heading,yaw_rate,speed,length,width], whereas
    agent history uses heading at 7 and length/width at 10/11.  The historical
    toy renderer copied the 9-D vector directly, accidentally interpreting
    ``length`` as ``heading``.  Keep this mapping explicit.
    """
    ps = np.asarray(ps, dtype=float).reshape(-1)
    out = np.zeros(16, dtype=float)
    if ps.size >= 2:
        out[0:2] = ps[0:2]
    if ps.size >= 4:
        out[3:5] = ps[2:4]
    if ps.size >= 5:
        out[7] = ps[4]
    if ps.size >= 9:
        out[10] = ps[7]
        out[11] = ps[8]
    return out


def _limits(d: dict[str, Any], nominal_prefix: np.ndarray | None = None) -> tuple[tuple[float, float], tuple[float, float]]:
    """Paper-scale ego-centric limits; do not let distant traffic shrink the story."""
    hist = np.asarray(d.get("agent_history", np.zeros((0, 0, 16))), float)
    valid = np.asarray(d.get("agent_valid", np.zeros(hist.shape[:2] if hist.ndim == 3 else (0, 0))), bool)
    prefix = np.asarray(d.get("prefix_states", np.zeros((0, 9))), float)
    center = np.array([0.0, 0.0], dtype=float)
    if hist.ndim == 3 and hist.shape[0] and hist.shape[1] and valid.shape == hist.shape[:2] and valid[-1, 0]:
        q = hist[-1, 0, :2]
        if np.isfinite(q).all():
            center = q.astype(float)
    elif prefix.ndim == 2 and len(prefix) and np.isfinite(prefix[0, :2]).all():
        center = prefix[0, :2].astype(float)
    radius = 28.0
    pts: list[np.ndarray] = []
    if prefix.ndim == 2 and prefix.shape[1] >= 2 and len(prefix):
        pts.append(prefix[:, :2])
    if nominal_prefix is not None and nominal_prefix.ndim == 2 and nominal_prefix.shape[1] >= 2 and len(nominal_prefix):
        pts.append(nominal_prefix[:, :2])
    if pts:
        p = np.concatenate(pts, axis=0)
        p = p[np.isfinite(p).all(axis=1)]
        if len(p):
            radius = max(radius, min(42.0, float(np.max(np.linalg.norm(p - center[None, :], axis=1))) + 8.0))
    return ((float(center[0] - radius), float(center[0] + radius)),
            (float(center[1] - radius), float(center[1] + radius)))


def _prepare_map(d: dict[str, Any], xlim, ylim) -> tuple[list[np.ndarray], np.ndarray]:
    maps = np.asarray(d.get("map_polylines", np.zeros((0, 0, 2))), float)
    mv = np.asarray(d.get("map_valid", np.zeros(maps.shape[:2] if maps.ndim == 3 else (0, 0))), bool)
    segments: list[np.ndarray] = []
    if maps.ndim == 3:
        for i in range(min(320, maps.shape[0])):
            v = mv[i] if mv.ndim == 2 and i < mv.shape[0] else np.ones(maps.shape[1], bool)
            q = maps[i, v, :2]
            if len(q) < 2 or not np.isfinite(q).all():
                continue
            inside = (
                (q[:, 0] >= xlim[0])
                & (q[:, 0] <= xlim[1])
                & (q[:, 1] >= ylim[0])
                & (q[:, 1] <= ylim[1])
            )
            if inside.any():
                segments.append(q)
    route = np.asarray(d.get("route", np.zeros((0, 2))), float)
    return segments, route


def _draw_map(ax, prepared: tuple[list[np.ndarray], np.ndarray]) -> None:
    _, _, _, LineCollection = _mpl()
    segments, route = prepared
    if segments:
        ax.add_collection(LineCollection(segments, linewidths=0.45, alpha=0.55, zorder=0))
    if route.ndim == 2 and route.shape[1] >= 2 and len(route) >= 2:
        ax.plot(route[:, 0], route[:, 1], lw=1.2, alpha=0.65, zorder=1)


def _render_five(case: Case, d: dict[str, Any], out: Path, *, dt: float, dpi: int) -> None:
    plt, _, _, _ = _mpl()
    hist = np.asarray(d.get("agent_history", np.zeros((0, 0, 16))), float)
    valid = np.asarray(d.get("agent_valid", np.zeros(hist.shape[:2] if hist.ndim == 3 else (0, 0))), bool)
    prefix = np.asarray(d.get("prefix_states", np.zeros((0, 9))), float)
    nominal_prefix = None
    if case.nominal_path:
        p = Path(case.nominal_path)
        if p.is_file():
            nd = load_npz_selected(p, _NOMINAL_RENDER_KEYS)
            nominal_prefix = np.asarray(nd.get("prefix_states", np.zeros((0, 9))), float)
    xlim, ylim = _limits(d, nominal_prefix)
    prepared_map = _prepare_map(d, xlim, ylim)

    if hist.ndim == 3 and valid.shape == hist.shape[:2] and hist.shape[0]:
        # Spread frames across the whole observed horizon.  The previous last-5
        # rendering covered only ~0.4 s at 10 Hz and looked almost static.
        ids = np.linspace(0, hist.shape[0] - 1, 5).round().astype(int)
        fig, axs = plt.subplots(1, 5, figsize=(17.0, 3.6), sharex=True, sharey=True)
        for q, (ax, ti) in enumerate(zip(axs, ids)):
            _draw_map(ax, prepared_map)
            for a in range(min(hist.shape[1], 48)):
                if not valid[ti, a]:
                    continue
                _draw_vehicle(ax, hist[ti, a], "tab:red" if a == 0 else "tab:blue", 0.95 if a == 0 else 0.45, 6 if a == 0 else 4)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_aspect("equal")
            ax.set_title(f"observed t-{len(ids)-1-q}")
            ax.grid(alpha=0.1)
        if case.case_type == "oracle_gap":
            subtitle = f"stored teacher: R_orc={case.r_orc:.2f}, R_dep={case.r_dep:.2f}, gap={case.gap:.2f}"
        else:
            subtitle = f"nominal R_dep={case.nominal_r_dep:.2f} → candidate R_dep={case.r_dep:.2f} (Δ={case.delta_r_dep_vs_nominal:+.2f})"
        fig.suptitle(f"Observed approach: {case.case_type} | {subtitle}")
        fig.tight_layout()
        fig.savefig(out / "history_5frame.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    if prefix.ndim == 2 and prefix.shape[1] >= 2 and len(prefix):
        ids = np.linspace(0, len(prefix) - 1, 5).round().astype(int)
        fig, axs = plt.subplots(1, 5, figsize=(17.0, 3.6), sharex=True, sharey=True)
        cur = hist[-1] if hist.ndim == 3 and hist.shape[0] else np.zeros((0, 16))
        curv = valid[-1] if valid.ndim == 2 and valid.shape[0] else np.zeros((len(cur),), bool)
        for q, (ax, pi) in enumerate(zip(axs, ids)):
            _draw_map(ax, prepared_map)
            elapsed = float(dt) * float(pi + 1)
            for a in range(min(len(cur), 48)):
                if a == 0 or not curv[a]:
                    continue
                s = cur[a].copy()
                if s.size > 4:
                    s[0] += s[3] * elapsed
                    s[1] += s[4] * elapsed
                _draw_vehicle(ax, s, "tab:blue", 0.38, 3)
            ego = _prefix_state_to_agent_box(prefix[pi])
            _draw_vehicle(ax, ego, "tab:purple", 0.95, 7)
            ax.plot(prefix[: pi + 1, 0], prefix[: pi + 1, 1], lw=1.8, zorder=6, label="selected candidate")
            if nominal_prefix is not None and nominal_prefix.ndim == 2 and len(nominal_prefix):
                ni = min(int(pi), len(nominal_prefix) - 1)
                ax.plot(nominal_prefix[: ni + 1, 0], nominal_prefix[: ni + 1, 1], lw=1.2, ls="--", alpha=0.8, zorder=5, label="nominal")
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_aspect("equal")
            ax.set_title(f"prefix {q + 1}")
            ax.grid(alpha=0.1)
            if q == 0 and nominal_prefix is not None:
                ax.legend(loc="best", fontsize=7)
        title = "Candidate prefix + observation-only constant-velocity context (illustrative; not hidden-future ground truth)"
        if nominal_prefix is not None:
            title += f"\nnominal R_dep={case.nominal_r_dep:.2f} → candidate={case.r_dep:.2f}, Δ={case.delta_r_dep_vs_nominal:+.2f}"
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(out / "candidate_cv_5frame.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)



def _render_signed_delta(case: Case, out: Path, *, dpi: int) -> None:
    """Reviewer-facing one-glance reserve/debt panel for Near/Contact cases."""
    if case.case_type not in {"near_reserve", "contact_debt"} or case.nominal_r_dep is None:
        return
    plt, _, _, _ = _mpl()
    vals = np.array([float(case.nominal_r_dep), float(case.r_dep)], dtype=float)
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    bars = ax.bar([0, 1], vals, width=0.56, alpha=0.82)
    ax.axhline(0.0, lw=1.6, color="black")
    span = max(0.6, float(np.max(np.abs(vals))) * 1.45)
    ax.set_ylim(-span, span)
    ax.set_xticks([0, 1], ["nominal", "selected candidate"])
    ax.set_ylabel(r"signed deployable recoverability $R_{dep}$")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + (0.04*span if v >= 0 else -0.07*span), f"{v:+.2f}", ha="center", va="center", fontweight="bold")
    ax.text(1.02, 0.76, "reserve > 0", transform=ax.transAxes, ha="left", va="center", fontsize=9)
    ax.text(1.02, 0.24, "debt < 0", transform=ax.transAxes, ha="left", va="center", fontsize=9)
    if case.case_type == "near_reserve":
        title = "Near-contact: preserve / enlarge deployable recovery reserve"
    else:
        title = "Contact: repay deployable recovery debt toward the zero boundary"
    ax.set_title(title + f"\nΔR_dep={case.delta_r_dep_vs_nominal:+.2f}")
    ax.grid(axis="y", alpha=0.15)
    fig.tight_layout()
    fig.savefig(out / "signed_recovery_delta.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def _option_label(modes: Sequence[str], idx: int) -> str:
    if 0 <= int(idx) < len(modes):
        s = str(modes[int(idx)])
        return f"{idx}:{s[:16]}"
    return str(idx)


def _render_ambiguity_matrix(case: Case, d: dict[str, Any], out: Path, *, dpi: int, full_matrix: bool, beta: float = 0.20) -> None:
    """Render the exact OC-MERO sign flip, with a hard pair when available.

    Grade A shows the compact two-root sufficient witness.  Grade B shows the
    actual compatible-root lower-tail contributors instead of refusing to render
    a Contact case simply because no single root pair proves the whole tail.
    """
    if case.case_type != "oracle_gap":
        return
    plt, Rectangle, _, _ = _mpl()
    c = np.asarray(d.get("c_star", np.zeros((0, 0))), dtype=float)
    m = np.asarray(d.get("m_star", np.zeros((0, 0))), dtype=float)
    p = np.asarray(d.get("root_probs", []), dtype=float).reshape(-1)
    rv = np.asarray(d.get("root_valid", np.ones(m.shape[0] if m.ndim == 2 else 0)), dtype=bool).reshape(-1)
    ov = np.asarray(d.get("option_valid", np.ones(m.shape[1] if m.ndim == 2 else 0)), dtype=bool).reshape(-1)
    if c.ndim != 2 or m.ndim != 2 or c.size == 0 or m.size == 0:
        return
    K, L = m.shape
    if len(p) != K:
        p = np.ones(K, dtype=float) / max(K, 1)
    valid_opts = ov[:L] if len(ov) >= L else np.ones(L, dtype=bool)
    valid_roots = rv[:K] if len(rv) >= K else np.ones(K, dtype=bool)
    modes_arr = np.asarray(d.get("recovery_modes", np.arange(L))).reshape(-1)
    modes = [str(x) for x in modes_arr.tolist()]
    root_labels = _root_semantic_labels(d, K)

    # Reconstruct the anchor-local OC-MERO objects used by the proof.
    anchor = int(case.anchor_root if case.anchor_root is not None else int(np.flatnonzero(valid_roots)[0]))
    p_eff = np.where(valid_roots, p[:K], 0.0)
    p_norm = normalize_weights(p_eff)
    compat_row = np.nan_to_num(c[anchor, :K], nan=0.0, posinf=1.0, neginf=0.0)
    w = normalize_weights(np.clip(compat_row, 0.0, None) * p_norm)
    root_best, root_best_opt = _masked_root_best(m, valid_opts)
    q = np.full(L, -1e9, dtype=float)
    for ell in np.flatnonzero(valid_opts):
        q[int(ell)] = weighted_lcvar(m[:, int(ell)], w, beta)
    shared_anchor = int(np.argmax(q))
    oracle_tail = _lcvar_influence(root_best, w, beta)

    hard_pair = (
        case.proof_grade == "A"
        and case.conflict_root_i is not None
        and case.conflict_root_j is not None
        and case.option_i is not None
        and case.option_j is not None
    )
    if hard_pair:
        roots = [int(case.conflict_root_i), int(case.conflict_root_j)]
        option_ids: list[int] = []
        for x in (int(case.option_i), int(case.option_j), int(case.best_shared_option if case.best_shared_option is not None else shared_anchor)):
            if 0 <= x < L and x not in option_ids:
                option_ids.append(x)
        if len(option_ids) < 3:
            pair_min = np.where(valid_opts, np.minimum(m[roots[0]], m[roots[1]]), -1e9)
            for idx in np.argsort(pair_min)[::-1]:
                if int(idx) not in option_ids:
                    option_ids.append(int(idx)); break
    else:
        # Choose lower-tail roots that collectively expose different branch-wise
        # best actions.  This is the correct visual object for a soft-tail sign flip.
        candidates = [int(r) for r in np.flatnonzero(valid_roots)]
        candidates.sort(key=lambda r: (oracle_tail[r] > 0.0, oracle_tail[r], w[r], -root_best[r]), reverse=True)
        roots: list[int] = []
        seen_opts: set[int] = set()
        for r in candidates:
            o = int(root_best_opt[r])
            if oracle_tail[r] > 0.0 and (o not in seen_opts or len(roots) < 2):
                roots.append(r); seen_opts.add(o)
            if len(roots) >= 4:
                break
        for r in candidates:
            if r not in roots:
                roots.append(r)
            if len(roots) >= min(4, max(2, len(candidates))):
                break
        option_ids = []
        for r in roots:
            o = int(root_best_opt[r])
            if 0 <= o < L and o not in option_ids:
                option_ids.append(o)
        if shared_anchor not in option_ids:
            option_ids.append(shared_anchor)
        option_ids = option_ids[:4]

    mini = m[np.ix_(roots, option_ids)]
    pair_c = c[np.ix_(roots, roots)]
    nroot = len(roots)
    fig, axs = plt.subplots(1, 3, figsize=(15.6, 4.9), gridspec_kw={"width_ratios": [1.1, 1.9, 1.55]})

    im0 = axs[0].imshow(pair_c, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    rlabels = [root_labels[r] if root_labels[r] != f"root {r}" else f"root {r}" for r in roots]
    axs[0].set_xticks(range(nroot), rlabels, rotation=24, ha="right")
    axs[0].set_yticks(range(nroot), rlabels)
    axs[0].set_title("Same observable history")
    for y in range(nroot):
        for x in range(nroot):
            axs[0].text(x, y, f"{pair_c[y, x]:.2f}", ha="center", va="center", fontsize=8.8,
                        color="white" if pair_c[y, x] < 0.45 else "black")
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04, label="observation compatibility")

    vmax = max(float(np.nanmax(np.abs(mini))), 1e-6)
    im1 = axs[1].imshow(mini, vmin=-vmax, vmax=vmax, aspect="auto", cmap="coolwarm")
    axs[1].set_yticks(range(nroot), rlabels)
    xlabels = [str(modes[o]).replace("_", " ")[:22] if 0 <= o < len(modes) else str(o) for o in option_ids]
    axs[1].set_xticks(range(len(option_ids)), xlabels, rotation=24, ha="right")
    axs[1].set_title("One deployable action must work before the branch is known")
    for y in range(nroot):
        for x in range(len(option_ids)):
            axs[1].text(x, y, f"{mini[y, x]:+.2f}", ha="center", va="center", fontsize=9.2, fontweight="bold")
    for y, r in enumerate(roots):
        o = int(root_best_opt[r])
        if o in option_ids:
            x = option_ids.index(o)
            axs[1].add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, lw=2.6))
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04, label="signed recovery margin")

    axs[2].axis("off")
    local_orc = float(case.local_oracle_margin if case.local_oracle_margin is not None else weighted_lcvar(root_best, w, beta))
    local_dep = float(case.local_deployable_margin if case.local_deployable_margin is not None else np.max(q))
    lines = ["HIDDEN-BRANCH ORACLE"]
    for r in roots[:4]:
        o = int(root_best_opt[r])
        label = root_labels[r]
        action = str(modes[o]).replace("_", " ") if 0 <= o < len(modes) else str(o)
        lines.append(f"{label}: {action}  {root_best[r]:+.2f}")
    lines += [
        "",
        "DEPLOYABLE POLICY",
        f"one shared action: {str(modes[shared_anchor]).replace('_', ' ') if 0 <= shared_anchor < len(modes) else shared_anchor}",
        f"robust shared margin = {local_dep:+.2f}",
        "",
        "THE GAP",
        f"oracle max-before-tail = {local_orc:+.2f}",
        f"deploy max-after-tail  = {local_dep:+.2f}",
        f"gap                    = {float(case.local_gap or (local_orc-local_dep)):+.2f}",
        "",
        "Same observation; different latent futures",
        "need different recoveries.  An oracle can",
        "choose after seeing the branch; a deployable",
        "planner cannot.",
    ]
    if case.margin_override_future_count > 0:
        lines += ["", "stress-mined counterfactual pair;", "configured artifact margin override is active"]
    axs[2].text(0.02, 0.98, "\n".join(lines), transform=axs[2].transAxes, ha="left", va="top",
                family="monospace", fontsize=9.5)
    fig.suptitle(
        f"Oracle-to-deployable recoverability gap — proof {case.proof_grade}\n"
        f"R_orc={case.r_orc:+.2f}, R_dep={case.r_dep:+.2f}, gap={case.gap:+.2f}",
        fontsize=12.5,
    )
    fig.tight_layout()
    fig.savefig(out / "toy_ambiguity_matrix.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    if full_matrix:
        mm = m.copy()
        if len(valid_roots) == m.shape[0] and len(valid_opts) == m.shape[1]:
            mm[~(valid_roots[:, None] & valid_opts[None, :])] = np.nan
        vmax_full = float(np.nanpercentile(np.abs(mm), 95)) if np.isfinite(mm).any() else 1.0
        vmax_full = max(vmax_full, 1e-6)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.2), constrained_layout=True)
        imc = ax1.imshow(c, vmin=0.0, vmax=1.0, aspect="equal", cmap="viridis")
        ax1.set_title("Full observation compatibility C")
        ax1.set_xlabel("latent root j"); ax1.set_ylabel("latent root i")
        fig.colorbar(imc, ax=ax1, fraction=0.046, pad=0.04)
        imm = ax2.imshow(mm, vmin=-vmax_full, vmax=vmax_full, aspect="auto", cmap="coolwarm")
        ax2.set_title("Full root × option signed margin M")
        ax2.set_xlabel("recovery option"); ax2.set_ylabel("latent root")
        fig.colorbar(imm, ax=ax2, fraction=0.046, pad=0.04)
        for r in roots:
            ax1.add_patch(Rectangle((-0.5, r - 0.5), c.shape[1], 1, fill=False, lw=1.5))
            ax2.add_patch(Rectangle((-0.5, r - 0.5), m.shape[1], 1, fill=False, lw=1.5))
        fig.savefig(out / "toy_ambiguity_matrix_full.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def _render_paper_composite(case: Case, d: dict[str, Any], out: Path, *, dpi: int, beta: float = 0.20) -> None:
    """Compact first-page figure: physical observation -> action conflict -> sign flip."""
    if case.case_type != "oracle_gap":
        return
    plt, Rectangle, _, _ = _mpl()
    hist = np.asarray(d.get("agent_history", np.zeros((0, 0, 16))), float)
    valid = np.asarray(d.get("agent_valid", np.zeros(hist.shape[:2] if hist.ndim == 3 else (0, 0))), bool)
    prefix = np.asarray(d.get("prefix_states", np.zeros((0, 9))), float)
    m = np.asarray(d.get("m_star", np.zeros((0, 0))), dtype=float)
    c = np.asarray(d.get("c_star", np.zeros((0, 0))), dtype=float)
    p = np.asarray(d.get("root_probs", []), dtype=float).reshape(-1)
    rv = np.asarray(d.get("root_valid", np.ones(m.shape[0] if m.ndim == 2 else 0)), dtype=bool).reshape(-1)
    ov = np.asarray(d.get("option_valid", np.ones(m.shape[1] if m.ndim == 2 else 0)), dtype=bool).reshape(-1)
    if m.ndim != 2 or c.ndim != 2:
        return
    K, L = m.shape
    if len(p) != K:
        p = np.ones(K, dtype=float) / max(K, 1)
    modes = [str(x) for x in np.asarray(d.get("recovery_modes", np.arange(L))).reshape(-1).tolist()]
    root_labels = _root_semantic_labels(d, K)
    valid_roots = rv[:K] if len(rv) >= K else np.ones(K, dtype=bool)
    valid_opts = ov[:L] if len(ov) >= L else np.ones(L, dtype=bool)
    p_norm = normalize_weights(np.where(valid_roots, p[:K], 0.0))
    anchor = int(case.anchor_root if case.anchor_root is not None else int(np.flatnonzero(valid_roots)[0]))
    w = normalize_weights(np.clip(np.nan_to_num(c[anchor, :K], nan=0.0), 0.0, None) * p_norm)
    root_best, root_best_opt = _masked_root_best(m, valid_opts)
    q = np.full(L, -1e9, dtype=float)
    for ell in np.flatnonzero(valid_opts):
        q[int(ell)] = weighted_lcvar(m[:, int(ell)], w, beta)
    shared = int(np.argmax(q))
    influence = _lcvar_influence(root_best, w, beta)

    hard_pair = case.proof_grade == "A" and case.conflict_root_i is not None and case.conflict_root_j is not None
    if hard_pair:
        roots = [int(case.conflict_root_i), int(case.conflict_root_j)]
    else:
        candidates = [int(r) for r in np.flatnonzero(valid_roots)]
        candidates.sort(key=lambda r: (influence[r] > 0.0, influence[r], w[r]), reverse=True)
        roots = []
        used: set[int] = set()
        for r in candidates:
            o = int(root_best_opt[r])
            if influence[r] > 0.0 and (o not in used or len(roots) < 2):
                roots.append(r); used.add(o)
            if len(roots) >= 3:
                break
        for r in candidates:
            if r not in roots:
                roots.append(r)
            if len(roots) >= min(3, len(candidates)):
                break
    option_ids: list[int] = []
    for r in roots:
        o = int(root_best_opt[r])
        if o not in option_ids:
            option_ids.append(o)
    if shared not in option_ids:
        option_ids.append(shared)
    option_ids = option_ids[:4]
    mini = m[np.ix_(roots, option_ids)]

    fig, axs = plt.subplots(1, 3, figsize=(15.8, 4.7), gridspec_kw={"width_ratios": [1.20, 1.45, 1.05]})

    # (a) Observable physical scene, ego-centric and motion-aware.
    ax = axs[0]
    xlim, ylim = _limits(d, None)
    _draw_map(ax, _prepare_map(d, xlim, ylim))
    nearest_actor = None; nearest_dist = math.inf
    if hist.ndim == 3 and hist.shape[0] and valid.shape == hist.shape[:2] and valid[-1, 0]:
        for a in range(1, min(hist.shape[1], 64)):
            if valid[-1, a]:
                dd = float(np.linalg.norm(hist[-1, a, :2] - hist[-1, 0, :2]))
                if dd < nearest_dist:
                    nearest_actor, nearest_dist = a, dd
        for a in range(min(hist.shape[1], 64)):
            ids = np.flatnonzero(valid[:, a])
            if not len(ids):
                continue
            xy = hist[ids, a, :2]
            if np.isfinite(xy).all():
                ax.plot(xy[:, 0], xy[:, 1], lw=2.0 if a == 0 else 0.9, alpha=0.9 if a == 0 else 0.32, zorder=2)
        for a in range(min(hist.shape[1], 64)):
            if not valid[-1, a]:
                continue
            color = "tab:red" if a == 0 else ("tab:orange" if a == nearest_actor else "tab:blue")
            _draw_vehicle(ax, hist[-1, a], color, 0.98 if a in {0, nearest_actor} else 0.32, 7 if a in {0, nearest_actor} else 4)
        if nearest_actor is not None and nearest_dist < 35.0:
            axy, exy = hist[-1, nearest_actor, :2], hist[-1, 0, :2]
            ax.plot([exy[0], axy[0]], [exy[1], axy[1]], ls="--", lw=1.1, alpha=0.7)
            mid = (exy + axy) / 2
            ax.text(mid[0], mid[1], f"{nearest_dist:.1f} m", fontsize=8.5, ha="center", va="bottom")
    if prefix.ndim == 2 and len(prefix):
        ax.plot(prefix[:, 0], prefix[:, 1], lw=1.8, ls=":", label=f"candidate: {case.macro}", zorder=6)
        ax.legend(loc="best", fontsize=7.5)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal"); ax.grid(alpha=0.10)
    ax.set_title("(a) Same observable history\nfuture branch is still hidden", fontsize=10.5)

    # (b) Small conflict table: branch-specific good cells are outlined.
    ax = axs[1]
    vmax = max(float(np.nanmax(np.abs(mini))), 1e-6)
    im = ax.imshow(mini, vmin=-vmax, vmax=vmax, aspect="auto", cmap="coolwarm")
    ylabels = [root_labels[r] for r in roots]
    xlabels = [str(modes[o]).replace("_", " ")[:20] if 0 <= o < len(modes) else str(o) for o in option_ids]
    ax.set_yticks(range(len(roots)), ylabels)
    ax.set_xticks(range(len(option_ids)), xlabels, rotation=23, ha="right")
    for yy, r in enumerate(roots):
        for xx, o in enumerate(option_ids):
            ax.text(xx, yy, f"{m[r,o]:+.1f}", ha="center", va="center", fontsize=9, fontweight="bold")
        o_best = int(root_best_opt[r])
        if o_best in option_ids:
            xx = option_ids.index(o_best)
            ax.add_patch(Rectangle((xx - .5, yy - .5), 1, 1, fill=False, lw=2.5))
    ax.set_title("(b) Different latent futures\nneed different recoveries", fontsize=10.5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="signed recovery")

    # (c) The reviewer-facing takeaway; no algebra needed to understand the sign flip.
    ax = axs[2]; ax.axis("off")
    local_orc = float(case.local_oracle_margin if case.local_oracle_margin is not None else weighted_lcvar(root_best, w, beta))
    local_dep = float(case.local_deployable_margin if case.local_deployable_margin is not None else np.max(q))
    ax.text(0.50, 0.80, f"ORACLE\n{local_orc:+.2f}", ha="center", va="center", fontsize=17, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="black", lw=1.5))
    ax.text(0.50, 0.61, "sees hidden branch\n↓", ha="center", va="center", fontsize=9)
    ax.text(0.50, 0.40, f"DEPLOYABLE\n{local_dep:+.2f}", ha="center", va="center", fontsize=17, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="black", lw=1.5))
    ax.text(0.50, 0.18, "one action must be chosen\nfrom the same observation", ha="center", va="center", fontsize=9.5)
    ax.text(0.50, 0.045, f"gap = {float(case.local_gap or (local_orc-local_dep)):+.2f}", ha="center", va="center", fontsize=13, fontweight="bold")
    ax.set_title("(c) Oracle ≠ deployable\nrecoverability", fontsize=10.5)

    branches = case.artifact_branch_labels or "observation-compatible latent futures"
    fig.suptitle(f"Oracle-to-deployable recoverability gap: {branches}", fontsize=13.0)
    if case.margin_override_future_count > 0:
        fig.text(0.995, 0.01, "stress-mined counterfactual branches; see caption/metadata for teacher provenance",
                 ha="right", va="bottom", fontsize=7.5, alpha=0.75)
    fig.tight_layout(rect=(0, 0.025, 1, 0.94))
    fig.savefig(out / "paper_toy_figure.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)[:48] or "scene"


def _case_render_complete(folder: Path, case: Case, full_matrix: bool) -> bool:
    meta = folder / "metadata.json"
    if not meta.is_file():
        return False
    try:
        old = json.loads(meta.read_text(encoding="utf-8"))
        if old.get("render_schema") != _RENDER_SCHEMA:
            return False
    except Exception:
        return False
    wanted = [meta, folder / "history_5frame.png", folder / "candidate_cv_5frame.png"]
    if case.case_type in {"near_reserve", "contact_debt"}:
        wanted.append(folder / "signed_recovery_delta.png")
    if case.case_type == "oracle_gap":
        wanted.extend([folder / "toy_ambiguity_matrix.png", folder / "paper_toy_figure.png"])
        if full_matrix:
            wanted.append(folder / "toy_ambiguity_matrix_full.png")
    return all(q.is_file() and q.stat().st_size > 0 for q in wanted)


def _quarantine_stale_case_dirs(output: Path, expected_names: set[str]) -> int:
    """Move stale rank folders left by --resume instead of silently mixing runs."""
    if not output.is_dir():
        return 0
    stale = [q for q in output.iterdir() if q.is_dir() and re.match(r"^\d{3}_", q.name) and q.name not in expected_names]
    if not stale:
        return 0
    dst_root = output / "_stale_resume"
    dst_root.mkdir(exist_ok=True)
    moved = 0
    for q in stale:
        dst = dst_root / q.name
        if dst.exists():
            suffix = 1
            while (dst_root / f"{q.name}__{suffix}").exists():
                suffix += 1
            dst = dst_root / f"{q.name}__{suffix}"
        shutil.move(str(q), str(dst))
        moved += 1
    return moved


def main() -> int:
    ap = argparse.ArgumentParser(description="Mine and render OC-RAP oracle-gap / reserve / debt toy examples.")
    ap.add_argument("--near", type=Path, required=True, help="Near-contact dataset root with manifest.csv")
    ap.add_argument("--contact", type=Path, required=True, help="Contact dataset root with manifest.csv")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--oracle-gap-count", type=int, default=24)
    ap.add_argument("--near-reserve-count", type=int, default=8)
    ap.add_argument("--contact-debt-count", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=0.20, help="Outer OC-MERO lower-tail mass; used for stored-teacher recomputation.")
    ap.add_argument("--beta", type=float, default=0.20, help="Inner compatible-root lower-tail mass used by proof panel.")
    ap.add_argument("--min-root-prob", type=float, default=0.05)
    ap.add_argument("--min-compat", type=float, default=0.50)
    ap.add_argument("--teacher-recompute-tol", type=float, default=1e-5)
    ap.add_argument("--max-hard", type=float, default=0.0, help="Maximum prefix hard_violation for reserve/debt examples.")
    ap.add_argument("--max-harm", type=float, default=0.05, help="Maximum prefix harm_proxy for reserve/debt examples.")
    ap.add_argument("--min-progress-delta", type=float, default=0.05)
    ap.add_argument("--dt", type=float, default=0.1, help="Prefix/CV visualization step in seconds.")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--metadata-only", action="store_true", help="Select cases but skip PNG rendering.")
    ap.add_argument("--allow-oracle-fallback", action="store_true", help="Allow exploratory proof grades C/D; exact soft sign-flip grade B is publication-valid by default.")
    ap.add_argument("--render-full-matrix", action="store_true", help="Also write the legacy full C and M heatmaps.")
    ap.add_argument("--resume", action="store_true", help="Skip already complete case folders instead of re-rendering them.")
    args = ap.parse_args()

    if not (0.0 < args.alpha <= 1.0 and 0.0 < args.beta <= 1.0):
        raise ValueError("--alpha and --beta must be in (0,1]")
    if args.oracle_gap_count < 0 or args.near_reserve_count < 0 or args.contact_debt_count < 0:
        raise ValueError("requested counts must be nonnegative")

    t_total = time.perf_counter()
    t_phase = time.perf_counter()
    print(json.dumps({"event": "toy_selection_phase", "phase": "manifest_read_start"}), flush=True)
    near = _read_manifest(args.near, "near")
    contact = _read_manifest(args.contact, "contact")
    manifest_seconds = time.perf_counter() - t_phase
    print(json.dumps({"event": "toy_selection_phase", "phase": "manifest_read_done", "seconds": manifest_seconds, "near_rows": len(near), "contact_rows": len(contact)}), flush=True)

    t_phase = time.perf_counter()
    print(json.dumps({"event": "toy_selection_phase", "phase": "oracle_scan_start"}), flush=True)
    near_oracle, near_stats = _oracle_cases(
        near,
        alpha=args.alpha,
        beta=args.beta,
        min_root_prob=args.min_root_prob,
        min_compat=args.min_compat,
        allow_fallback=args.allow_oracle_fallback,
        recompute_tol=args.teacher_recompute_tol,
    )
    contact_oracle, contact_stats = _oracle_cases(
        contact,
        alpha=args.alpha,
        beta=args.beta,
        min_root_prob=args.min_root_prob,
        min_compat=args.min_compat,
        allow_fallback=args.allow_oracle_fallback,
        recompute_tol=args.teacher_recompute_tol,
    )
    near_oracle = _rerank_for_paper(near_oracle)
    contact_oracle = _rerank_for_paper(contact_oracle)
    oracle_scan_seconds = time.perf_counter() - t_phase
    print(json.dumps({"event": "toy_selection_phase", "phase": "oracle_scan_done", "seconds": oracle_scan_seconds, "near": near_stats, "contact": contact_stats}, ensure_ascii=False), flush=True)

    # Persist scan diagnostics before any publication fail-closed decision so a
    # shortage of grade-A examples is debuggable without repeating the scan.
    args.output.mkdir(parents=True, exist_ok=True)
    early_scan_stats = {"near": near_stats, "contact": contact_stats}
    (args.output / "toy_selection_scan_stats.json").write_text(
        json.dumps(early_scan_stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    seen: set[tuple[str, str]] = set()
    selected: list[Case] = []
    n_near = args.oracle_gap_count // 2
    n_contact = args.oracle_gap_count - n_near
    selected += _diverse_take(near_oracle, n_near, seen)
    selected += _diverse_take(contact_oracle, n_contact, seen)
    current_oracle = sum(c.case_type == "oracle_gap" for c in selected)
    if current_oracle < args.oracle_gap_count:
        selected += _diverse_take(
            near_oracle + contact_oracle,
            args.oracle_gap_count - current_oracle,
            seen,
        )
    current_oracle = sum(c.case_type == "oracle_gap" for c in selected)
    if current_oracle < args.oracle_gap_count and not args.allow_oracle_fallback:
        raise RuntimeError(
            f"Only {current_oracle}/{args.oracle_gap_count} exact soft sign-flip oracle proofs were found. "
            "For a paper figure this tool fails closed before using grades C/D. Inspect toy_selection_scan_stats.json, "
            "or explicitly pass --allow-oracle-fallback for exploratory browsing."
        )

    t_phase = time.perf_counter()
    print(json.dumps({"event": "toy_selection_phase", "phase": "progress_selection_start"}), flush=True)
    near_progress = _rerank_for_paper(_progress_cases(
        near,
        "near_reserve",
        max_hard=args.max_hard,
        max_harm=args.max_harm,
        min_delta=args.min_progress_delta,
    ))
    contact_progress = _rerank_for_paper(_progress_cases(
        contact,
        "contact_debt",
        max_hard=args.max_hard,
        max_harm=args.max_harm,
        min_delta=args.min_progress_delta,
    ))
    selected += _diverse_take(near_progress, args.near_reserve_count, seen)
    selected += _diverse_take(contact_progress, args.contact_debt_count, seen)
    progress_selection_seconds = time.perf_counter() - t_phase
    print(json.dumps({"event": "toy_selection_phase", "phase": "progress_selection_done", "seconds": progress_selection_seconds}), flush=True)

    t_phase = time.perf_counter()
    records: list[dict[str, Any]] = []
    folders: list[Path] = []
    for rank, case in enumerate(selected):
        folder = args.output / f"{rank:03d}_{case.case_type}_{case.dataset_role}_{_slug(case.scene_id)}_t{case.time_index}_c{case.candidate_index}"
        folders.append(folder)
        rec = asdict(case)
        rec["rank"] = rank
        rec["folder"] = str(folder)
        rec["render_schema"] = _RENDER_SCHEMA
        records.append(rec)
    stale_moved = _quarantine_stale_case_dirs(args.output, {q.name for q in folders})

    for case, folder, rec in zip(selected, folders, records):
        folder.mkdir(parents=True, exist_ok=True)
        complete = bool(args.resume and _case_render_complete(folder, case, args.render_full_matrix))
        (folder / "metadata.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if args.metadata_only or complete:
            continue
        d = load_npz_selected(case.path, _RENDER_KEYS)
        _render_five(case, d, folder, dt=args.dt, dpi=args.dpi)
        _render_signed_delta(case, folder, dpi=args.dpi)
        _render_ambiguity_matrix(case, d, folder, dpi=args.dpi, full_matrix=args.render_full_matrix, beta=args.beta)
        _render_paper_composite(case, d, folder, dpi=args.dpi, beta=args.beta)
        del d
        gc.collect()

    rendering_seconds = time.perf_counter() - t_phase
    total_seconds = time.perf_counter() - t_total
    scan_stats = {"near": near_stats, "contact": contact_stats}
    timings_seconds = {
        "manifest_read": manifest_seconds,
        "oracle_scan": oracle_scan_seconds,
        "progress_selection": progress_selection_seconds,
        "rendering_or_metadata_write": rendering_seconds,
        "total": total_seconds,
    }
    selection_contract = {
        "oracle_gap": (
            "manifest i_art=1, stored R_orc>=0 and R_dep<0; publication validity requires the exact "
            "soft-compatible inner-tail sign flip max-before-tail>0>max-after-tail. Grade A additionally "
            "has a compact two-root hard conflict; Grade B is rendered from the full lower-tail contributors."
        ),
        "near_reserve": (
            "near-boundary nominal R_dep; choose the largest-delta prefix-safe alternative, retrying lower-delta alternatives "
            "if the top manifest candidate violates prefix hard/harm constraints"
        ),
        "contact_debt": (
            "negative nominal R_dep; choose the largest prefix-safe debt-repayment alternative, preferring zero-boundary crossing"
        ),
        "frames": (
            "history panels span the full observed horizon and use ego-centric zoom; candidate panels use observation-only constant-velocity context; "
            "oracle cases additionally emit paper_toy_figure.png combining the physical observation with the exact proof"
        ),
        "teacher_use": (
            "m_star/c_star/root probabilities are used only for offline toy selection/proof; they are not deployed planner inputs"
        ),
        "artifact_provenance": (
            "stress-mined hidden branches and margin_override_applied are surfaced in per-case metadata/figures rather than presented as raw natural margins"
        ),
    }
    summary = {
        "schema_version": 3,
        "render_schema": _RENDER_SCHEMA,
        "stale_case_dirs_quarantined": stale_moved,
        "selection_contract": selection_contract,
        "scan_stats": scan_stats,
        "timings_seconds": timings_seconds,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "cases": records,
    }
    (args.output / "selected_cases.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    fields = list(records[0].keys()) if records else ["rank", "case_type", "dataset_role", "path"]
    with (args.output / "selected_cases.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)

    event = {
        "event": "toy_example_selection",
        "selected": len(selected),
        "oracle_gap": sum(c.case_type == "oracle_gap" for c in selected),
        "oracle_grade_A": sum(c.case_type == "oracle_gap" and c.proof_grade == "A" for c in selected),
        "oracle_grade_B": sum(c.case_type == "oracle_gap" and c.proof_grade == "B" for c in selected),
        "near_reserve": sum(c.case_type == "near_reserve" for c in selected),
        "contact_debt": sum(c.case_type == "contact_debt" for c in selected),
        "output": str(args.output),
        "scan_stats": scan_stats,
        "timings_seconds": timings_seconds,
    }
    print(json.dumps(event, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
