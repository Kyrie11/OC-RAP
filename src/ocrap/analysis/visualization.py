from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from ocrap.data.serialization import load_npz


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(np.asarray(x).item())
    except Exception:
        return default


def _safe_str(x: Any, default: str = "") -> str:
    try:
        return str(np.asarray(x).item())
    except Exception:
        return default


def _mpl():
    import matplotlib.pyplot as plt  # type: ignore
    from matplotlib.patches import Rectangle  # type: ignore
    from matplotlib.transforms import Affine2D  # type: ignore
    return plt, Rectangle, Affine2D


def plot_recoverability_story(rows: list[dict[str, Any]], out_dir: Path) -> str | None:
    if not rows:
        return None
    plt, _, _ = _mpl()
    r_dep = np.asarray([r["r_dep"] for r in rows], dtype=float)
    r_orc = np.asarray([r["r_orc"] for r in rows], dtype=float)
    gap = np.asarray([r["gap"] for r in rows], dtype=float)
    art = np.asarray([r["artifact"] for r in rows], dtype=bool)
    finite = np.isfinite(r_dep) & np.isfinite(r_orc) & np.isfinite(gap)
    r_dep, r_orc, gap, art = r_dep[finite], r_orc[finite], gap[finite], art[finite]
    if r_dep.size == 0:
        return None
    x_min = min(float(r_dep.min()) - 0.2, -0.5)
    x_max = max(float(r_dep.max()) + 0.2, 0.8)
    y_min = min(float(r_orc.min()) - 0.2, -0.5)
    y_max = max(float(r_orc.max()) + 0.2, 0.8)
    oracle_only = (r_dep < 0.0) & (r_orc >= 0.0)
    deployable = r_dep >= 0.0
    critical = r_orc < 0.0

    fig = plt.figure(figsize=(9.5, 6.6))
    ax = fig.add_subplot(111)
    ax.axvspan(x_min, 0, ymin=max(0, (0 - y_min) / max(y_max - y_min, 1e-8)), ymax=1, alpha=0.10, color="tab:red", label="Oracle-only trap zone")
    ax.axvspan(0, x_max, alpha=0.06, color="tab:green", label="Deployable-safe side")
    ax.axhspan(y_min, 0, alpha=0.06, color="0.2", label="No branch recovery")
    ax.scatter(r_dep[~art], r_orc[~art], s=13, alpha=0.38, c="tab:blue", edgecolors="none", label="non-artifact candidates")
    if art.any():
        ax.scatter(r_dep[art], r_orc[art], s=28, alpha=0.82, c="tab:orange", edgecolors="k", linewidths=0.2, label="oracle artifacts")
    ax.axvline(0, color="k", lw=1.2)
    ax.axhline(0, color="k", lw=1.2)
    lo = min(x_min, y_min)
    hi = max(x_max, y_max)
    ax.plot([lo, hi], [lo, hi], ls="--", color="0.35", lw=1.0, label="R_orc = R_dep")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Deployable recoverability  $R_{dep}^*$")
    ax.set_ylabel("Oracle recoverability  $R_{orc}^*$")
    ax.set_title("The OC-RAP motivation: oracle recovery can be non-deployable")
    n = max(len(r_dep), 1)
    txt = (
        f"Oracle-only traps: {oracle_only.mean()*100:.1f}%\n"
        f"Deployable-safe: {deployable.mean()*100:.1f}%\n"
        f"Unrecoverable/critical: {critical.mean()*100:.1f}%\n"
        f"Median ODG: {np.median(gap):.2f}"
    )
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.75", alpha=0.92))
    ax.annotate("Looks recoverable only if the planner\nknows the hidden root after the fact",
                xy=(min(-0.4, np.percentile(r_dep, 5)), max(0.2, np.percentile(r_orc, 70))),
                xycoords="data", xytext=(0.40, 0.20), textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", lw=1.2), fontsize=10,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", alpha=0.90))
    ax.legend(loc="lower right", fontsize=9, framealpha=0.94)
    ax.grid(True, alpha=0.22)
    p = out_dir / "story_oracle_vs_deployable.png"
    fig.savefig(p, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(p)


def plot_criticality_ladder(rows: list[dict[str, Any]], out_dir: Path) -> str | None:
    if not rows:
        return None
    plt, _, _ = _mpl()
    names = [
        "normal\nhigh-headroom",
        "recoverable\nlow/mixed",
        "oracle-only\ntrap",
        "unrecoverable\ncritical",
    ]
    counts = np.zeros((4, 2), dtype=int)  # non-art, artifact
    for r in rows:
        if r["r_dep"] >= 0.50 and r["gap"] < 0.25 and not r["artifact"]:
            b = 0
        elif r["r_dep"] >= 0.0:
            b = 1
        elif r["r_orc"] >= 0.0:
            b = 2
        else:
            b = 3
        counts[b, int(bool(r["artifact"]))] += 1
    fig = plt.figure(figsize=(9.2, 4.8))
    ax = fig.add_subplot(111)
    x = np.arange(len(names))
    ax.bar(x, counts[:, 0], label="non-artifact", color="tab:blue", alpha=0.75)
    ax.bar(x, counts[:, 1], bottom=counts[:, 0], label="oracle artifact", color="tab:orange", alpha=0.85)
    total = max(int(counts.sum()), 1)
    for i in range(len(names)):
        c = int(counts[i].sum())
        ax.text(i, c + total * 0.01, f"{c}\n{c/total*100:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("samples")
    ax.set_title("Dataset criticality ladder: normal → low-headroom → oracle-only trap → critical")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    p = out_dir / "story_criticality_ladder.png"
    fig.savefig(p, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(p)


def _topdown_limits(points: list[np.ndarray], pad: float = 12.0) -> tuple[tuple[float, float], tuple[float, float]]:
    xs, ys = [], []
    for p in points:
        a = np.asarray(p, dtype=float)
        if a.size == 0:
            continue
        a = a.reshape(-1, a.shape[-1]) if a.ndim > 1 else a.reshape(1, -1)
        if a.shape[-1] >= 2:
            finite = np.isfinite(a[:, 0]) & np.isfinite(a[:, 1])
            xs.extend(a[finite, 0].tolist())
            ys.extend(a[finite, 1].tolist())
    if not xs:
        return (-30, 30), (-30, 30)
    return (min(xs) - pad, max(xs) + pad), (min(ys) - pad, max(ys) + pad)


def _draw_vehicle(ax, x: float, y: float, heading: float, length: float, width: float, *, color: str, alpha: float, label: str | None = None):
    plt, Rectangle, Affine2D = _mpl()
    length = float(length) if np.isfinite(length) and length > 0 else 4.8
    width = float(width) if np.isfinite(width) and width > 0 else 2.0
    trans = Affine2D().rotate_around(x, y, heading) + ax.transData
    patch = Rectangle((x - length / 2, y - width / 2), length, width, transform=trans,
                      fc=color, ec="k", lw=0.6, alpha=alpha, label=label)
    ax.add_patch(patch)
    ax.plot([x, x + 0.55 * length * math.cos(heading)], [y, y + 0.55 * length * math.sin(heading)], color="k", lw=0.7, alpha=alpha)


def plot_sample_scene(sample_path: str | Path, out_dir: Path, *, name: str = "toy_scene") -> str | None:
    plt, _, _ = _mpl()
    d = load_npz(sample_path)
    hist = np.asarray(d.get("agent_history", np.zeros((0, 0, 16))), dtype=float)
    valid = np.asarray(d.get("agent_valid", np.zeros(hist.shape[:2] if hist.ndim >= 2 else (0, 0))), dtype=bool)
    prefix = np.asarray(d.get("prefix_states", np.zeros((0, 9))), dtype=float)
    maps = np.asarray(d.get("map_polylines", np.zeros((0, 0, 2))), dtype=float)
    map_valid = np.asarray(d.get("map_valid", np.zeros(maps.shape[:2] if maps.ndim >= 2 else (0, 0))), dtype=bool)
    route = np.asarray(d.get("route", np.zeros((0, 2))), dtype=float)
    scene = _safe_str(d.get("scene_id", ""))
    t = int(round(_safe_float(d.get("time_index", -1), -1)))
    cand = int(round(_safe_float(d.get("candidate_index", -1), -1)))
    macro = _safe_str(d.get("prefix_macro_name", ""))
    r_dep = _safe_float(d.get("r_dep_star", 0.0))
    r_orc = _safe_float(d.get("r_orc_star", 0.0))
    gap = _safe_float(d.get("oracle_gap_star", r_orc - r_dep))
    art = bool(round(_safe_float(d.get("i_art_star", 0.0))))

    fig = plt.figure(figsize=(8.5, 8.0))
    ax = fig.add_subplot(111)
    if maps.ndim >= 3:
        for i in range(min(maps.shape[0], 180)):
            mv = map_valid[i] if map_valid.ndim >= 2 and i < map_valid.shape[0] else np.ones(maps.shape[1], dtype=bool)
            pts = maps[i, mv]
            if pts.shape[0] >= 2:
                ax.plot(pts[:, 0], pts[:, 1], color="0.72", lw=0.55, alpha=0.65, zorder=0)
    if route.ndim >= 2 and route.shape[0] >= 2:
        ax.plot(route[:, 0], route[:, 1], color="tab:green", lw=2.0, alpha=0.65, label="route", zorder=1)
    if hist.ndim >= 3 and hist.shape[0] > 0 and valid.ndim >= 2:
        last = hist[-1]
        vmask = valid[-1] if valid.shape[0] else np.zeros(last.shape[0], dtype=bool)
        for a in range(min(last.shape[0], len(vmask))):
            if not vmask[a]:
                continue
            s = last[a]
            if s.shape[0] < 12:
                continue
            is_ego = a == 0
            color = "tab:red" if is_ego else "tab:blue"
            label = "ego at decision time" if is_ego else ("other vehicles" if a == 1 else None)
            heading = float(s[7]) if np.isfinite(s[7]) else 0.0
            _draw_vehicle(ax, float(s[0]), float(s[1]), heading, float(s[10]) if s.size > 10 else 4.8, float(s[11]) if s.size > 11 else 2.0, color=color, alpha=0.90 if is_ego else 0.42, label=label)
        # ego history trace
        ev = valid[:, 0] if valid.shape[1] > 0 else np.zeros(hist.shape[0], dtype=bool)
        epts = hist[ev, 0, :2] if hist.shape[1] > 0 else np.zeros((0, 2))
        if epts.shape[0] >= 2:
            ax.plot(epts[:, 0], epts[:, 1], color="tab:red", lw=1.4, ls="--", alpha=0.75, label="ego log history")
    if prefix.ndim >= 2 and prefix.shape[0] >= 2:
        ax.plot(prefix[:, 0], prefix[:, 1], color="k", lw=2.4, marker="o", ms=2.5, label=f"candidate prefix: {macro}", zorder=5)
        if prefix.shape[1] >= 9:
            _draw_vehicle(ax, float(prefix[-1, 0]), float(prefix[-1, 1]), float(prefix[-1, 4]), float(prefix[-1, 7]), float(prefix[-1, 8]), color="tab:purple", alpha=0.55, label="ego after prefix")
    xlim, ylim = _topdown_limits([hist[:, :, :2] if hist.ndim >= 3 else np.zeros((0, 2)), prefix[:, :2] if prefix.ndim >= 2 else np.zeros((0, 2)), route[:, :2] if route.ndim >= 2 else np.zeros((0, 2))])
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.18)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title(f"Toy scene from OC-RAP NPZ: scene={scene}, t={t}, cand={cand}")
    subtitle = f"R_dep*={r_dep:.2f}, R_orc*={r_orc:.2f}, ODG*={gap:.2f}, artifact={art}, macro={macro}"
    ax.text(0.02, 0.02, subtitle, transform=ax.transAxes, ha="left", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.75", alpha=0.92))
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=8, framealpha=0.9)
    p = out_dir / f"{name}_topdown.png"
    fig.savefig(p, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(p)


def plot_sample_ambiguity(sample_path: str | Path, out_dir: Path, *, name: str = "toy_scene") -> str | None:
    plt, _, _ = _mpl()
    d = load_npz(sample_path)
    c = np.asarray(d.get("c_star", np.zeros((0, 0))), dtype=float)
    m = np.asarray(d.get("m_star", np.zeros((0, 0))), dtype=float)
    rv = np.asarray(d.get("root_valid", np.ones(c.shape[0] if c.ndim >= 2 else 0)), dtype=bool)
    ov = np.asarray(d.get("option_valid", np.ones(m.shape[1] if m.ndim >= 2 else 0)), dtype=bool)
    if c.ndim != 2 or c.size == 0:
        return None
    fig = plt.figure(figsize=(11.5, 4.8))
    ax1 = fig.add_subplot(121)
    im = ax1.imshow(c, vmin=0, vmax=1, cmap="viridis")
    ax1.set_title("Post-prefix observation compatibility C*")
    ax1.set_xlabel("root j"); ax1.set_ylabel("root i")
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    ax2 = fig.add_subplot(122)
    mm = np.asarray(m, dtype=float).copy()
    if m.ndim == 2:
        invalid = ~(rv[:, None] & ov[None, :])
        mm[invalid] = np.nan
    im2 = ax2.imshow(mm, cmap="coolwarm", aspect="auto")
    ax2.set_title("Teacher recovery margins m* [root, option]")
    ax2.set_xlabel("recovery option"); ax2.set_ylabel("root")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    fig.suptitle("Why branch-wise recovery can be misleading: ambiguity + shared recovery constraints", y=1.02, fontsize=12)
    p = out_dir / f"{name}_ambiguity_matrix.png"
    fig.savefig(p, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(p)


def write_toy_gallery(rows: list[dict[str, Any]], out_dir: Path, *, max_examples: int = 2) -> list[str]:
    paths: list[str] = []
    if not rows:
        return paths
    # Prefer high-gap oracle-only artifacts because these are the most direct
    # motivation examples for OC-RAP.
    candidates = [r for r in rows if r.get("r_orc", -1) >= 0 and r.get("r_dep", 1) < 0]
    candidates.sort(key=lambda r: (float(r.get("gap", 0)), bool(r.get("artifact", False))), reverse=True)
    if not candidates:
        candidates = sorted(rows, key=lambda r: float(r.get("gap", 0)), reverse=True)
    seen = set()
    k = 0
    for r in candidates:
        key = (r.get("scene_id"), r.get("time_index"))
        if key in seen:
            continue
        seen.add(key)
        path = r.get("path")
        if not path:
            continue
        name = f"toy_{k:02d}_scene_t{r.get('time_index', 'x')}_c{r.get('candidate_index', 'x')}"
        for p in (plot_sample_scene(path, out_dir, name=name), plot_sample_ambiguity(path, out_dir, name=name)):
            if p:
                paths.append(p)
        k += 1
        if k >= max_examples:
            break
    return paths
