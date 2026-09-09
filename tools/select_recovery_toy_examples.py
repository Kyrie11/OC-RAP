#!/usr/bin/env python3
"""Mine publication-quality OC-RAP motivation/toy cases from existing dataset buckets.

The primary target is the oracle-to-deployable recoverability gap.  Selection is
manifest-first (cheap), then verifies a root/option conflict from the stored
OC-MERO tensors before rendering.  No model checkpoint, teacher future, regime
router, or dataset reconstruction is used.

For each selected case the tool writes:
  * history_5frame.png  -- five consecutive *observed* history frames;
  * candidate_cv_5frame.png -- planned ego prefix with observation-only CV
    context for other agents (explicitly marked illustrative, not ground truth);
  * toy_ambiguity_matrix.png -- observation compatibility and root-option margins;
  * metadata.json.
It also writes selected_cases.csv/json for manual filtering.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

# Allow direct `python tools/...py` use from a clean checkout without requiring
# the caller to pre-export PYTHONPATH. This only affects import discovery.
_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np

from ocrap.data.serialization import load_npz


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
    nominal_r_dep: float | None = None
    delta_r_dep_vs_nominal: float | None = None
    conflict_root_i: int | None = None
    conflict_root_j: int | None = None
    compatibility: float | None = None
    option_i: int | None = None
    option_j: int | None = None
    pair_shared_margin: float | None = None
    strong_pair_conflict: bool | None = None
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
    p = root / "manifest.csv"
    if not p.is_file():
        raise FileNotFoundError(f"missing manifest: {p}")
    rows: list[dict[str, Any]] = []
    with p.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rel = str(row.get("path") or "")
            sample = (root / rel).resolve()
            rows.append({
                **row,
                "_root": str(root),
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


def _root_option_conflict(d: dict[str, Any], min_root_prob: float, min_compat: float) -> dict[str, Any] | None:
    p = np.asarray(d.get("root_probs", []), dtype=float).reshape(-1)
    rv = np.asarray(d.get("root_valid", np.ones_like(p)), dtype=bool).reshape(-1)
    m = np.asarray(d.get("m_star", []), dtype=float)
    c = np.asarray(d.get("c_star", []), dtype=float)
    ov = np.asarray(d.get("option_valid", np.ones(m.shape[1] if m.ndim == 2 else 0)), dtype=bool).reshape(-1)
    if m.ndim != 2 or c.ndim != 2 or p.size == 0 or m.shape[0] != p.size or c.shape != (p.size, p.size):
        return None
    valid_opts = np.flatnonzero(ov[: m.shape[1]])
    roots = np.flatnonzero(rv[:p.size] & np.isfinite(p) & (p >= float(min_root_prob)))
    if roots.size < 2 or valid_opts.size < 1:
        return None
    best: dict[str, Any] | None = None
    for ai in range(len(roots)):
        i = int(roots[ai])
        mi = m[i, valid_opts]
        if not np.isfinite(mi).any():
            continue
        oi = int(valid_opts[int(np.nanargmax(mi))]); bi = float(m[i, oi])
        for aj in range(ai + 1, len(roots)):
            j = int(roots[aj])
            cj = float(c[i, j]) if np.isfinite(c[i, j]) else 0.0
            if cj < float(min_compat):
                continue
            mj = m[j, valid_opts]
            if not np.isfinite(mj).any():
                continue
            oj = int(valid_opts[int(np.nanargmax(mj))]); bj = float(m[j, oj])
            if oi == oj or bi <= 0.0 or bj <= 0.0:
                continue
            shared = float(np.nanmax(np.minimum(m[i, valid_opts], m[j, valid_opts])))
            # Strongest examples have high compatible root mass, different
            # branch-wise recovery choices, and no positive shared option.
            pair_mass = float(p[i] * p[j])
            conflict_strength = max(0.0, -shared) + 0.35 * (bi + bj)
            # A pair with no positive shared option is especially legible in a
            # paper figure.  Prefer it strongly, but do not require it because
            # the full OC-MERO artifact can involve more than this displayed pair.
            strict_bonus = 5.0 if shared <= 0.0 else 0.0
            score = 3.0 * cj + 4.0 * pair_mass + conflict_strength + strict_bonus
            rec = {
                "root_i": i, "root_j": j, "compatibility": cj,
                "option_i": oi, "option_j": oj,
                "best_margin_i": bi, "best_margin_j": bj,
                "pair_shared_margin": shared, "pair_mass": pair_mass,
                "conflict_score": score,
            }
            if best is None or score > float(best["conflict_score"]):
                best = rec
    return best


def _oracle_cases(rows: list[dict[str, Any]], *, min_root_prob: float, min_compat: float) -> list[Case]:
    # Fast ordering first; only load the strongest artifact candidates.
    cand = [r for r in rows if r["_art"] and r["_r_orc"] >= 0.0 and r["_r_dep"] < 0.0 and math.isfinite(r["_gap"])]
    cand.sort(key=lambda r: r["_gap"], reverse=True)
    out: list[Case] = []
    for r in cand:
        path = Path(r["_path"])
        if not path.is_file():
            continue
        d = load_npz(path)
        conflict = _root_option_conflict(d, min_root_prob, min_compat)
        # Keep a high-gap fallback even when no pair reaches the strict visual
        # conflict test, but rank verified root/option conflicts first.
        bonus = 0.0 if conflict is None else 2.0 + float(conflict["conflict_score"])
        macro = _s(d.get("prefix_macro_name", ""))
        c = Case(
            case_type="oracle_gap", dataset_role=r["_role"], path=str(path),
            scene_id=r["_scene"], time_index=r["_time"], candidate_index=r["_cand"], macro=macro,
            r_dep=float(r["_r_dep"]), r_orc=float(r["_r_orc"]), gap=float(r["_gap"]),
            score=float(r["_gap"] + bonus),
            notes="verified compatible-root option conflict" if conflict else "high-gap artifact fallback",
        )
        if conflict:
            c.conflict_root_i = int(conflict["root_i"]); c.conflict_root_j = int(conflict["root_j"])
            c.compatibility = float(conflict["compatibility"]); c.option_i = int(conflict["option_i"]); c.option_j = int(conflict["option_j"])
            c.pair_shared_margin = float(conflict["pair_shared_margin"])
            c.strong_pair_conflict = bool(c.pair_shared_margin <= 0.0)
            if c.strong_pair_conflict:
                c.notes = "oracle-only artifact with a displayed compatible-root pair that has different positive branch-wise options and no positive shared option"
            else:
                c.notes = "oracle-only artifact; displayed root pair shows option disagreement, while full OC-MERO supplies the deployability failure"
        out.append(c)
    return out


def _group_rows(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, int], list[dict[str, Any]]]:
    out: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for r in rows:
        out.setdefault((r["_scene"], r["_time"]), []).append(r)
    return out


def _progress_cases(rows: list[dict[str, Any]], kind: str) -> list[Case]:
    out: list[Case] = []
    for (scene, t), group in _group_rows(rows).items():
        nom = next((r for r in group if r["_nom"]), None)
        if nom is None or not math.isfinite(nom["_r_dep"]):
            continue
        ndep = float(nom["_r_dep"])
        if kind == "near_reserve" and not (-0.35 <= ndep <= 0.35):
            continue
        if kind == "contact_debt" and not (ndep < 0.0):
            continue
        alts = [r for r in group if (not r["_nom"]) and math.isfinite(r["_r_dep"])]
        if not alts:
            continue
        best = max(alts, key=lambda r: float(r["_r_dep"] - ndep))
        delta = float(best["_r_dep"] - ndep)
        if delta <= 0.05:
            continue
        path = Path(best["_path"])
        if not path.is_file():
            continue
        d = load_npz(path)
        feasible = bool(_i(d.get("feasible", 1), 1))
        hard = _f(d.get("hard_violation", 0.0), 0.0)
        harm = _f(d.get("harm_proxy", 0.0), 0.0)
        if (not feasible) or hard > 1.0 or harm > 1.0:
            continue
        crossing = float(best["_r_dep"] >= 0.0)
        boundary_bonus = max(0.0, 0.35 - abs(ndep)) if kind == "near_reserve" else crossing
        score = 2.0 * delta + boundary_bonus + 0.15 * max(0.0, float(best["_gap"]))
        out.append(Case(
            case_type=kind, dataset_role=best["_role"], path=str(path), scene_id=scene,
            time_index=t, candidate_index=best["_cand"], macro=_s(d.get("prefix_macro_name", "")),
            r_dep=float(best["_r_dep"]), r_orc=float(best["_r_orc"]), gap=float(best["_gap"]),
            score=float(score), nominal_r_dep=ndep, delta_r_dep_vs_nominal=delta,
            notes=("candidate crosses physical/deployable zero boundary" if best["_r_dep"] >= 0 > ndep else "candidate improves signed reserve/debt"),
        ))
    out.sort(key=lambda x: x.score, reverse=True)
    return out


def _diverse_take(cases: list[Case], n: int, seen: set[tuple[str, int]]) -> list[Case]:
    if n <= 0:
        return []
    selected: list[Case] = []
    # First pass enforces unique scene-time and macro diversity when possible.
    macro_count: dict[str, int] = {}
    for c in sorted(cases, key=lambda x: x.score, reverse=True):
        key = (c.scene_id, c.time_index)
        if key in seen:
            continue
        m = c.macro or "unknown"
        if macro_count.get(m, 0) >= max(2, math.ceil(n / 3)):
            continue
        selected.append(c); seen.add(key); macro_count[m] = macro_count.get(m, 0) + 1
        if len(selected) >= n:
            break
    if len(selected) < n:
        for c in sorted(cases, key=lambda x: x.score, reverse=True):
            key = (c.scene_id, c.time_index)
            if key in seen:
                continue
            selected.append(c); seen.add(key)
            if len(selected) >= n:
                break
    return selected


def _mpl():
    import matplotlib.pyplot as plt  # type: ignore
    from matplotlib.patches import Rectangle  # type: ignore
    from matplotlib.transforms import Affine2D  # type: ignore
    return plt, Rectangle, Affine2D


def _draw_vehicle(ax, s: np.ndarray, color: str, alpha: float = 0.85, z: int = 5) -> None:
    plt, Rectangle, Affine2D = _mpl()
    del plt
    if s.size < 2 or not np.isfinite(s[:2]).all():
        return
    x, y = float(s[0]), float(s[1]); heading = float(s[7]) if s.size > 7 and np.isfinite(s[7]) else 0.0
    length = float(s[10]) if s.size > 10 and np.isfinite(s[10]) and s[10] > 0 else 4.8
    width = float(s[11]) if s.size > 11 and np.isfinite(s[11]) and s[11] > 0 else 2.0
    trans = Affine2D().rotate_around(x, y, heading) + ax.transData
    ax.add_patch(Rectangle((x-length/2, y-width/2), length, width, transform=trans,
                           fc=color, ec="k", lw=0.45, alpha=alpha, zorder=z))


def _limits(d: dict[str, Any]) -> tuple[tuple[float, float], tuple[float, float]]:
    hist = np.asarray(d.get("agent_history", np.zeros((0, 0, 16))), float)
    valid = np.asarray(d.get("agent_valid", np.zeros(hist.shape[:2] if hist.ndim == 3 else (0,0))), bool)
    prefix = np.asarray(d.get("prefix_states", np.zeros((0, 9))), float)
    pts = []
    if hist.ndim == 3 and valid.shape == hist.shape[:2] and hist.shape[0]:
        cur = hist[-1, valid[-1], :2]
        if cur.size: pts.append(cur)
    if prefix.ndim == 2 and prefix.shape[1] >= 2 and len(prefix): pts.append(prefix[:, :2])
    if not pts:
        return (-25, 25), (-25, 25)
    p = np.concatenate(pts, axis=0); p = p[np.isfinite(p).all(axis=1)]
    if not len(p): return (-25,25),(-25,25)
    cx,cy=np.median(p,axis=0); span=max(18.0,min(55.0,0.6*max(np.ptp(p[:,0]),np.ptp(p[:,1]))+15.0))
    return (float(cx-span),float(cx+span)),(float(cy-span),float(cy+span))


def _draw_map(ax, d: dict[str, Any], xlim, ylim) -> None:
    maps=np.asarray(d.get("map_polylines",np.zeros((0,0,2))),float); mv=np.asarray(d.get("map_valid",np.zeros(maps.shape[:2] if maps.ndim==3 else (0,0))),bool)
    if maps.ndim==3:
        for i in range(min(320,maps.shape[0])):
            v=mv[i] if mv.ndim==2 and i<mv.shape[0] else np.ones(maps.shape[1],bool); q=maps[i,v,:2]
            if len(q)>=2 and np.isfinite(q).all() and ((q[:,0]>=xlim[0])&(q[:,0]<=xlim[1])&(q[:,1]>=ylim[0])&(q[:,1]<=ylim[1])).any():
                ax.plot(q[:,0],q[:,1],lw=.45,alpha=.55,zorder=0)
    route=np.asarray(d.get("route",np.zeros((0,2))),float)
    if route.ndim==2 and route.shape[1]>=2 and len(route)>=2:
        ax.plot(route[:,0],route[:,1],lw=1.2,alpha=.65,zorder=1)


def _render_five(case: Case, out: Path) -> None:
    plt,_,_= _mpl(); d=load_npz(case.path)
    hist=np.asarray(d.get("agent_history",np.zeros((0,0,16))),float); valid=np.asarray(d.get("agent_valid",np.zeros(hist.shape[:2] if hist.ndim==3 else (0,0))),bool)
    prefix=np.asarray(d.get("prefix_states",np.zeros((0,9))),float)
    xlim,ylim=_limits(d)
    # Five consecutive observed frames, preserving actual observation history.
    if hist.ndim==3 and valid.shape==hist.shape[:2] and hist.shape[0]:
        ids=np.linspace(max(0,hist.shape[0]-5),hist.shape[0]-1,5).round().astype(int)
        fig,axs=plt.subplots(1,5,figsize=(17.0,3.6),sharex=True,sharey=True)
        for q,(ax,ti) in enumerate(zip(axs,ids)):
            _draw_map(ax,d,xlim,ylim)
            for a in range(min(hist.shape[1],48)):
                if not valid[ti,a]: continue
                _draw_vehicle(ax,hist[ti,a],"tab:red" if a==0 else "tab:blue",.95 if a==0 else .45,6 if a==0 else 4)
            ax.set_xlim(*xlim);ax.set_ylim(*ylim);ax.set_aspect("equal");ax.set_title(f"obs frame {q+1}");ax.grid(alpha=.1)
        fig.suptitle(f"Observed approach: {case.case_type} | R_orc={case.r_orc:.2f}, R_dep={case.r_dep:.2f}, gap={case.gap:.2f}")
        fig.tight_layout();fig.savefig(out/"history_5frame.png",dpi=210,bbox_inches="tight");plt.close(fig)
    # Five ego-prefix frames with current-state constant-velocity context. This is
    # intentionally labelled illustrative; it never pretends to be hidden future GT.
    if prefix.ndim==2 and prefix.shape[1]>=2 and len(prefix):
        ids=np.linspace(0,len(prefix)-1,5).round().astype(int); fig,axs=plt.subplots(1,5,figsize=(17.0,3.6),sharex=True,sharey=True)
        cur=hist[-1] if hist.ndim==3 and hist.shape[0] else np.zeros((0,16)); curv=valid[-1] if valid.ndim==2 and valid.shape[0] else np.zeros((len(cur),),bool)
        for q,(ax,pi) in enumerate(zip(axs,ids)):
            _draw_map(ax,d,xlim,ylim); dt=0.1*float(pi+1)
            for a in range(min(len(cur),48)):
                if a==0 or not curv[a]: continue
                s=cur[a].copy();
                if s.size>4: s[0]+=s[3]*dt; s[1]+=s[4]*dt
                _draw_vehicle(ax,s,"tab:blue",.38,3)
            ps=prefix[pi]; ego=np.zeros(16,float); ego[:min(len(ps),9)]=ps[:min(len(ps),9)]
            if len(ps)>8: ego[10]=ps[7];ego[11]=ps[8]
            _draw_vehicle(ax,ego,"tab:purple",.95,7);ax.plot(prefix[:pi+1,0],prefix[:pi+1,1],lw=1.8,zorder=6)
            ax.set_xlim(*xlim);ax.set_ylim(*ylim);ax.set_aspect("equal");ax.set_title(f"prefix {q+1}");ax.grid(alpha=.1)
        fig.suptitle("Candidate prefix + observation-only constant-velocity context (illustrative; not hidden-future ground truth)")
        fig.tight_layout();fig.savefig(out/"candidate_cv_5frame.png",dpi=210,bbox_inches="tight");plt.close(fig)


def _render_ambiguity_matrix(case: Case, d: dict[str, Any], out: Path) -> None:
    """Render an explicit proof panel for the oracle-to-deployable gap."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    c = np.asarray(d.get("c_star", np.zeros((0, 0))), dtype=float)
    m = np.asarray(d.get("m_star", np.zeros((0, 0))), dtype=float)
    if c.ndim != 2 or m.ndim != 2 or c.size == 0 or m.size == 0:
        return
    rv = np.asarray(d.get("root_valid", np.ones(c.shape[0])), dtype=bool).reshape(-1)
    ov = np.asarray(d.get("option_valid", np.ones(m.shape[1])), dtype=bool).reshape(-1)
    mm = m.copy()
    if rv.size == m.shape[0] and ov.size == m.shape[1]:
        mm[~(rv[:, None] & ov[None, :])] = np.nan
    vmax = float(np.nanpercentile(np.abs(mm), 95)) if np.isfinite(mm).any() else 1.0
    vmax = max(vmax, 1e-6)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.2), constrained_layout=True)
    im1 = ax1.imshow(c, vmin=0.0, vmax=1.0, aspect="equal", cmap="viridis")
    ax1.set_title("Observation compatibility $C_{ij}$")
    ax1.set_xlabel("latent root $j$"); ax1.set_ylabel("latent root $i$")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="1 = observationally indistinguishable")
    im2 = ax2.imshow(mm, vmin=-vmax, vmax=vmax, aspect="auto", cmap="coolwarm")
    ax2.set_title(r"Root × recovery-option signed margin $M_{k\ell}$")
    ax2.set_xlabel(r"recovery option $\ell$"); ax2.set_ylabel(r"latent root $k$")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label=">0 recoverable, <0 unsafe")
    ri, rj, oi, oj = case.conflict_root_i, case.conflict_root_j, case.option_i, case.option_j
    if ri is not None and rj is not None:
        for r in (int(ri), int(rj)):
            ax1.add_patch(Rectangle((-0.5, r-0.5), c.shape[1], 1, fill=False, lw=1.8))
            ax1.add_patch(Rectangle((r-0.5, -0.5), 1, c.shape[0], fill=False, lw=1.8))
        ax1.add_patch(Rectangle((int(rj)-0.5, int(ri)-0.5), 1, 1, fill=False, lw=3.0))
    if None not in (ri, rj, oi, oj):
        for r, o in [(int(ri), int(oi)), (int(rj), int(oj)), (int(ri), int(oj)), (int(rj), int(oi))]:
            ax2.add_patch(Rectangle((o-0.5, r-0.5), 1, 1, fill=False, lw=2.2))
        pair = "n/a" if case.pair_shared_margin is None else f"{case.pair_shared_margin:.2f}"
        ax2.text(0.02, -0.16, f"compatible roots {ri}/{rj}: best options {oi}/{oj}; pair shared margin={pair}",
                 transform=ax2.transAxes, ha="left", va="top", fontsize=9)
    extra = " | highlighted pair has no positive shared option" if case.strong_pair_conflict else ""
    fig.suptitle(f"Oracle-to-deployable gap: R_orc={case.r_orc:.2f}, R_dep={case.r_dep:.2f}, gap={case.gap:.2f}{extra}", fontsize=12.5)
    fig.savefig(out / "toy_ambiguity_matrix.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+","_",s)[:48] or "scene"


def main() -> int:
    ap=argparse.ArgumentParser(description="Mine and render OC-RAP oracle-gap / reserve / debt toy examples.")
    ap.add_argument("--near",type=Path,required=True,help="Near-contact dataset root with manifest.csv")
    ap.add_argument("--contact",type=Path,required=True,help="Contact dataset root with manifest.csv")
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--oracle-gap-count",type=int,default=24)
    ap.add_argument("--near-reserve-count",type=int,default=8)
    ap.add_argument("--contact-debt-count",type=int,default=8)
    ap.add_argument("--min-root-prob",type=float,default=0.05)
    ap.add_argument("--min-compat",type=float,default=0.50)
    ap.add_argument("--metadata-only",action="store_true",help="Select cases but skip PNG rendering.")
    args=ap.parse_args()
    near=_read_manifest(args.near,"near"); contact=_read_manifest(args.contact,"contact")
    seen:set[tuple[str,int]]=set(); selected=[]
    # Keep the primary oracle-gap gallery balanced between Near and Contact so
    # the manual shortlist is not dominated by the larger-gap Near bucket.
    near_oracle = _oracle_cases(near,min_root_prob=args.min_root_prob,min_compat=args.min_compat)
    contact_oracle = _oracle_cases(contact,min_root_prob=args.min_root_prob,min_compat=args.min_compat)
    n_near = args.oracle_gap_count // 2
    n_contact = args.oracle_gap_count - n_near
    selected += _diverse_take(near_oracle,n_near,seen)
    selected += _diverse_take(contact_oracle,n_contact,seen)
    if sum(c.case_type == "oracle_gap" for c in selected) < args.oracle_gap_count:
        # Fill any shortage from the remaining strongest role without duplicating scenes.
        selected += _diverse_take(near_oracle + contact_oracle, args.oracle_gap_count - sum(c.case_type == "oracle_gap" for c in selected), seen)
    selected += _diverse_take(_progress_cases(near,"near_reserve"),args.near_reserve_count,seen)
    selected += _diverse_take(_progress_cases(contact,"contact_debt"),args.contact_debt_count,seen)
    args.output.mkdir(parents=True,exist_ok=True)
    records=[]
    for rank,c in enumerate(selected):
        folder=args.output/f"{rank:03d}_{c.case_type}_{c.dataset_role}_{_slug(c.scene_id)}_t{c.time_index}_c{c.candidate_index}"
        folder.mkdir(parents=True,exist_ok=True)
        rec=asdict(c);rec["rank"]=rank;rec["folder"]=str(folder);records.append(rec)
        (folder/"metadata.json").write_text(json.dumps(rec,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
        if not args.metadata_only:
            _render_five(c,folder)
            _render_ambiguity_matrix(c, load_npz(c.path), folder)
    (args.output/"selected_cases.json").write_text(json.dumps({"schema_version":1,"selection_contract":{
        "oracle_gap":"i_art=1, R_orc>=0, R_dep<0; balanced Near/Contact shortlist; rank by gap and compatible-root option conflict, strongly preferring a displayed pair with no positive shared option",
        "near_reserve":"near-zero nominal R_dep; choose executable alternative with largest positive delta R_dep",
        "contact_debt":"negative nominal R_dep; choose executable alternative with largest debt repayment",
        "frames":"history frames are observed; candidate frames use observation-only CV context and are illustrative"
    },"cases":records},indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    fields=list(records[0].keys()) if records else ["rank","case_type","dataset_role","path"]
    with (args.output/"selected_cases.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(records)
    print(json.dumps({"event":"toy_example_selection","selected":len(selected),"oracle_gap":sum(c.case_type=="oracle_gap" for c in selected),"near_reserve":sum(c.case_type=="near_reserve" for c in selected),"contact_debt":sum(c.case_type=="contact_debt" for c in selected),"output":str(args.output)},ensure_ascii=False))
    return 0


if __name__=="__main__": raise SystemExit(main())
