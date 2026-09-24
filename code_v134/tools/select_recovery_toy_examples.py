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
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

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
    # Preserve global ranking correctness: scan every manifest-level artifact,
    # but load only tiny proof tensors from each NPZ.
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

        # Verify stored deployability against the production OC-MERO algebra.
        # Legacy r_orc is recorded separately because the paper's nested oracle
        # definition is intentionally a stronger theoretical object.
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
                # A publication toy should never silently use a sample whose
                # stored teacher cannot be reproduced from its own tensors.
                continue
        except Exception:
            stats["teacher_recompute_mismatch"] += 1
            continue

        grade = str(proof["proof_grade"])
        stats[f"grade_{grade}"] += 1
        if grade != "A" and not allow_fallback:
            continue
        pair = proof.get("pair") or {}
        macro = _s(d.get("prefix_macro_name", ""))
        score = float(100.0 - 8.0 * (ord(grade) - ord("A")) + proof["proof_score"] + r["_gap"])
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
            score=score,
            proof_grade=grade,
            anchor_root=int(proof["anchor_root"]),
            local_oracle_margin=float(proof["local_oracle_margin"]),
            local_deployable_margin=float(proof["local_deployable_margin"]),
            local_gap=float(proof["local_gap"]),
            recomputed_r_dep=float(recalc.r_dep),
            recomputed_r_orc_legacy=float(recalc.r_orc),
            teacher_recompute_error=float(dep_err),
            notes=(
                "grade-A direct proof: compatible-root lower-tail oracle is positive, "
                "shared-option lower-tail is negative, and the displayed root pair "
                "has branch-wise positive recoveries but no positive shared option"
                if grade == "A"
                else f"exploratory oracle-gap fallback grade {grade}"
            ),
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


def _diverse_take(cases: list[Case], n: int, seen: set[tuple[str, int]]) -> list[Case]:
    if n <= 0:
        return []
    selected: list[Case] = []
    macro_count: dict[str, int] = {}
    ordered = sorted(cases, key=lambda x: x.score, reverse=True)
    for c in ordered:
        key = (c.scene_id, c.time_index)
        if key in seen:
            continue
        m = c.macro or "unknown"
        if macro_count.get(m, 0) >= max(2, math.ceil(n / 3)):
            continue
        selected.append(c)
        seen.add(key)
        macro_count[m] = macro_count.get(m, 0) + 1
        if len(selected) >= n:
            break
    if len(selected) < n:
        for c in ordered:
            key = (c.scene_id, c.time_index)
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
    hist = np.asarray(d.get("agent_history", np.zeros((0, 0, 16))), float)
    valid = np.asarray(d.get("agent_valid", np.zeros(hist.shape[:2] if hist.ndim == 3 else (0, 0))), bool)
    prefix = np.asarray(d.get("prefix_states", np.zeros((0, 9))), float)
    pts: list[np.ndarray] = []
    if hist.ndim == 3 and valid.shape == hist.shape[:2] and hist.shape[0]:
        cur = hist[-1, valid[-1], :2]
        if cur.size:
            pts.append(cur)
    if prefix.ndim == 2 and prefix.shape[1] >= 2 and len(prefix):
        pts.append(prefix[:, :2])
    if nominal_prefix is not None and nominal_prefix.ndim == 2 and nominal_prefix.shape[1] >= 2 and len(nominal_prefix):
        pts.append(nominal_prefix[:, :2])
    if not pts:
        return (-25, 25), (-25, 25)
    p = np.concatenate(pts, axis=0)
    p = p[np.isfinite(p).all(axis=1)]
    if not len(p):
        return (-25, 25), (-25, 25)
    cx, cy = np.median(p, axis=0)
    span = max(18.0, min(55.0, 0.6 * max(np.ptp(p[:, 0]), np.ptp(p[:, 1])) + 15.0))
    return (float(cx - span), float(cx + span)), (float(cy - span), float(cy + span))


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
        if hist.shape[0] >= 5:
            ids = np.arange(hist.shape[0] - 5, hist.shape[0], dtype=int)
        else:
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


def _render_ambiguity_matrix(case: Case, d: dict[str, Any], out: Path, *, dpi: int, full_matrix: bool) -> None:
    """Render a compact proof panel that directly explains max-before/after-tail."""
    if case.case_type != "oracle_gap":
        return
    plt, Rectangle, _, _ = _mpl()
    c = np.asarray(d.get("c_star", np.zeros((0, 0))), dtype=float)
    m = np.asarray(d.get("m_star", np.zeros((0, 0))), dtype=float)
    if c.ndim != 2 or m.ndim != 2 or c.size == 0 or m.size == 0:
        return
    modes_arr = np.asarray(d.get("recovery_modes", np.arange(m.shape[1]))).reshape(-1)
    modes = [str(x) for x in modes_arr.tolist()]
    ri, rj = case.conflict_root_i, case.conflict_root_j
    oi, oj, os = case.option_i, case.option_j, case.best_shared_option
    if None in (ri, rj, oi, oj, os):
        return
    ri, rj, oi, oj, os = int(ri), int(rj), int(oi), int(oj), int(os)
    option_ids: list[int] = []
    for x in (oi, oj, os):
        if x not in option_ids:
            option_ids.append(x)
    # Add one extra strongest common-ish option if only two are present, for a
    # slightly richer but still immediately readable matrix.
    if len(option_ids) < 3:
        valid = np.asarray(d.get("option_valid", np.ones(m.shape[1])), dtype=bool).reshape(-1)[: m.shape[1]]
        pair_min = np.where(valid, np.minimum(m[ri], m[rj]), -1e9)
        for idx in np.argsort(pair_min)[::-1]:
            if int(idx) not in option_ids:
                option_ids.append(int(idx))
                break
    mini = m[[ri, rj]][:, option_ids]

    fig, axs = plt.subplots(1, 3, figsize=(14.8, 4.5), gridspec_kw={"width_ratios": [1.0, 1.7, 1.45]})

    # Panel A: pair compatibility.
    pair_c = c[np.ix_([ri, rj], [ri, rj])]
    im0 = axs[0].imshow(pair_c, vmin=0.0, vmax=1.0, cmap="viridis")
    axs[0].set_xticks([0, 1], [f"root {ri}", f"root {rj}"])
    axs[0].set_yticks([0, 1], [f"root {ri}", f"root {rj}"])
    axs[0].set_title("Observation-compatible roots")
    for y in range(2):
        for x in range(2):
            axs[0].text(x, y, f"{pair_c[y, x]:.2f}", ha="center", va="center", color="white" if pair_c[y, x] < 0.45 else "black", fontsize=10)
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    # Panel B: only the conflicting roots/options, with values annotated.
    vmax = max(float(np.nanmax(np.abs(mini))), 1e-6)
    im1 = axs[1].imshow(mini, vmin=-vmax, vmax=vmax, aspect="auto", cmap="coolwarm")
    axs[1].set_yticks([0, 1], [f"root {ri}", f"root {rj}"])
    axs[1].set_xticks(range(len(option_ids)), [_option_label(modes, x) for x in option_ids], rotation=25, ha="right")
    axs[1].set_title("One option must serve both roots")
    for y in range(2):
        for x in range(len(option_ids)):
            axs[1].text(x, y, f"{mini[y, x]:+.2f}", ha="center", va="center", fontsize=10, fontweight="bold")
    for row, opt in ((0, oi), (1, oj)):
        if opt in option_ids:
            x = option_ids.index(opt)
            axs[1].add_patch(Rectangle((x - 0.5, row - 0.5), 1, 1, fill=False, lw=2.8))
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04, label="stored signed recovery margin")

    # Panel C: the theorem-level takeaway in numbers.
    axs[2].axis("off")
    pair_orc = float(case.pair_oracle_margin or math.nan)
    pair_shared = float(case.pair_shared_margin or math.nan)
    local_orc = float(case.local_oracle_margin or math.nan)
    local_dep = float(case.local_deployable_margin or math.nan)
    text = (
        "BRANCH-WISE ORACLE\n"
        f"root {ri} → {_option_label(modes, oi)} : {m[ri, oi]:+.2f}\n"
        f"root {rj} → {_option_label(modes, oj)} : {m[rj, oj]:+.2f}\n"
        f"pair oracle margin = {pair_orc:+.2f}\n\n"
        "DEPLOYABLE SHARED CHOICE\n"
        f"best common option = {_option_label(modes, os)}\n"
        f"pair shared margin = {pair_shared:+.2f}\n\n"
        "SOFT OC-MERO ANCHOR\n"
        f"max-before-tail = {local_orc:+.2f}\n"
        f"max-after-tail  = {local_dep:+.2f}\n"
        f"local gap       = {case.local_gap:+.2f}\n\n"
        "⇒ each hidden branch has a positive recovery,\n"
        "   but the observation-measurable shared\n"
        "   choice remains nonpositive."
    )
    axs[2].text(0.02, 0.98, text, transform=axs[2].transAxes, ha="left", va="top", family="monospace", fontsize=10.2)
    fig.suptitle(
        f"Oracle-to-deployable recoverability gap — grade {case.proof_grade}\n"
        f"stored teacher diagnostic: R_orc={case.r_orc:+.2f}, R_dep={case.r_dep:+.2f}, gap={case.gap:+.2f}",
        fontsize=12.5,
    )
    fig.tight_layout()
    fig.savefig(out / "toy_ambiguity_matrix.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    if full_matrix:
        rv = np.asarray(d.get("root_valid", np.ones(c.shape[0])), dtype=bool).reshape(-1)
        ov = np.asarray(d.get("option_valid", np.ones(m.shape[1])), dtype=bool).reshape(-1)
        mm = m.copy()
        if rv.size == m.shape[0] and ov.size == m.shape[1]:
            mm[~(rv[:, None] & ov[None, :])] = np.nan
        vmax = float(np.nanpercentile(np.abs(mm), 95)) if np.isfinite(mm).any() else 1.0
        vmax = max(vmax, 1e-6)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.2), constrained_layout=True)
        im1 = ax1.imshow(c, vmin=0.0, vmax=1.0, aspect="equal", cmap="viridis")
        ax1.set_title("Full observation compatibility C")
        ax1.set_xlabel("latent root j")
        ax1.set_ylabel("latent root i")
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        im2 = ax2.imshow(mm, vmin=-vmax, vmax=vmax, aspect="auto", cmap="coolwarm")
        ax2.set_title("Full root × option signed margin M")
        ax2.set_xlabel("recovery option")
        ax2.set_ylabel("latent root")
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        for r in (ri, rj):
            ax1.add_patch(Rectangle((-0.5, r - 0.5), c.shape[1], 1, fill=False, lw=1.8))
            ax2.add_patch(Rectangle((-0.5, r - 0.5), m.shape[1], 1, fill=False, lw=1.8))
        fig.savefig(out / "toy_ambiguity_matrix_full.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)[:48] or "scene"


def _case_render_complete(folder: Path, case: Case, full_matrix: bool) -> bool:
    wanted = [folder / "metadata.json", folder / "history_5frame.png", folder / "candidate_cv_5frame.png"]
    if case.case_type in {"near_reserve", "contact_debt"}:
        wanted.append(folder / "signed_recovery_delta.png")
    if case.case_type == "oracle_gap":
        wanted.append(folder / "toy_ambiguity_matrix.png")
        if full_matrix:
            wanted.append(folder / "toy_ambiguity_matrix_full.png")
    return all(p.is_file() and p.stat().st_size > 0 for p in wanted)


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
    ap.add_argument("--allow-oracle-fallback", action="store_true", help="Allow proof grades B/C/D if fewer grade-A paper examples exist.")
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
    oracle_scan_seconds = time.perf_counter() - t_phase
    print(json.dumps({"event": "toy_selection_phase", "phase": "oracle_scan_done", "seconds": oracle_scan_seconds, "near": near_stats, "contact": contact_stats}, ensure_ascii=False), flush=True)

    # Persist scan diagnostics before any publication fail-closed decision so a
    # shortage of grade-A examples is debuggable without repeating the scan.
    args.output.mkdir(parents=True, exist_ok=True)
    early_scan_stats = {"near": near_stats, "contact": contact_stats}
    (args.output / "toy_selection_scan_stats.json").write_text(
        json.dumps(early_scan_stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    seen: set[tuple[str, int]] = set()
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
            f"Only {current_oracle}/{args.oracle_gap_count} grade-A oracle proofs were found. "
            "For a paper figure this tool fails closed by default. Inspect toy_selection_scan_stats.json, "
            "or explicitly pass --allow-oracle-fallback for exploratory browsing."
        )

    t_phase = time.perf_counter()
    print(json.dumps({"event": "toy_selection_phase", "phase": "progress_selection_start"}), flush=True)
    selected += _diverse_take(
        _progress_cases(
            near,
            "near_reserve",
            max_hard=args.max_hard,
            max_harm=args.max_harm,
            min_delta=args.min_progress_delta,
        ),
        args.near_reserve_count,
        seen,
    )
    selected += _diverse_take(
        _progress_cases(
            contact,
            "contact_debt",
            max_hard=args.max_hard,
            max_harm=args.max_harm,
            min_delta=args.min_progress_delta,
        ),
        args.contact_debt_count,
        seen,
    )
    progress_selection_seconds = time.perf_counter() - t_phase
    print(json.dumps({"event": "toy_selection_phase", "phase": "progress_selection_done", "seconds": progress_selection_seconds}), flush=True)

    t_phase = time.perf_counter()
    records: list[dict[str, Any]] = []
    for rank, case in enumerate(selected):
        folder = args.output / f"{rank:03d}_{case.case_type}_{case.dataset_role}_{_slug(case.scene_id)}_t{case.time_index}_c{case.candidate_index}"
        folder.mkdir(parents=True, exist_ok=True)
        rec = asdict(case)
        rec["rank"] = rank
        rec["folder"] = str(folder)
        records.append(rec)
        (folder / "metadata.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if args.metadata_only:
            continue
        if args.resume and _case_render_complete(folder, case, args.render_full_matrix):
            continue
        # Load exactly once per selected case and only the members that are
        # required by the figures.
        d = load_npz_selected(case.path, _RENDER_KEYS)
        _render_five(case, d, folder, dt=args.dt, dpi=args.dpi)
        _render_signed_delta(case, folder, dpi=args.dpi)
        _render_ambiguity_matrix(case, d, folder, dpi=args.dpi, full_matrix=args.render_full_matrix)
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
            "manifest i_art=1, stored R_orc>=0 and R_dep<0; grade-A paper cases additionally require "
            "a soft-compatible inner-tail sign flip (max-before-tail>0>max-after-tail) and a displayed "
            "compatible root pair whose branch-wise best recoveries are positive but whose best shared option is nonpositive"
        ),
        "near_reserve": (
            "near-boundary nominal R_dep; choose the largest-delta prefix-safe alternative, retrying lower-delta alternatives "
            "if the top manifest candidate violates prefix hard/harm constraints"
        ),
        "contact_debt": (
            "negative nominal R_dep; choose the largest prefix-safe debt-repayment alternative, preferring zero-boundary crossing"
        ),
        "frames": (
            "history panels are observed only; candidate panels use observation-only constant-velocity context and are explicitly illustrative; "
            "reserve/debt panels overlay the nominal prefix"
        ),
        "teacher_use": (
            "m_star/c_star/root probabilities are used only for offline toy selection/proof; they are not deployed planner inputs"
        ),
    }
    summary = {
        "schema_version": 2,
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
