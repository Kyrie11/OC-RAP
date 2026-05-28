#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Allow running scripts directly from a source checkout without pip install -e .
import sys as _sys
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from ocrap.teacher.dataset_writer import read_dataset


def _plot_poly(ax, arr, closed=True, **kw):
    a = np.asarray(arr, dtype=np.float32)
    if len(a) < 2:
        return
    if closed and len(a) >= 3:
        a = np.concatenate([a, a[:1]], axis=0)
    ax.plot(a[:, 0], a[:, 1], **kw)


def _find_root_index(root_ids, rid: str) -> int | None:
    target = str(rid)
    for i in range(len(root_ids)):
        if str(root_ids[i]) == target:
            return int(i)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize one ReCAP root and optional action/recovery labels.")
    ap.add_argument("--root", required=True, help="Path to root JSON or root_id when --root-dir is provided.")
    ap.add_argument("--root-dir", default=None)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--action-index", type=int, default=0)
    ap.add_argument("--auto-action", choices=["none", "best_recovery", "worst_recovery", "least_harm", "largest_harm"],
                    default="none",
                    help="Select action index from labels instead of --action-index; useful for diagnostics galleries.")
    ap.add_argument("--summary-output", default=None, help="Optional JSON summary for the visualized root/action/mode.")
    ap.add_argument("--option-index", type=int, default=None, help="If omitted, use witness option for mode 0 when labels exist.")
    ap.add_argument("--mode-index", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    root_path = Path(args.root)
    if args.root_dir:
        root_path = Path(args.root_dir) / f"{Path(args.root).stem}.json"
    root = json.loads(root_path.read_text())
    fig, ax = plt.subplots(figsize=(8, 8))
    mf = root.get("map_features", {})
    for p in mf.get("drivable_polygons", []):
        _plot_poly(ax, p, closed=True, linewidth=0.7, alpha=0.35)
    for p in mf.get("lane_boundaries", []):
        _plot_poly(ax, p, closed=False, linewidth=0.6, alpha=0.7)
    for p in mf.get("lane_centerlines", []):
        _plot_poly(ax, p, closed=False, linewidth=0.8, alpha=0.7)
    route = np.asarray(root.get("route_info", {}).get("waypoints", []), dtype=np.float32)
    if route.ndim == 2 and len(route):
        ax.plot(route[:, 0], route[:, 1], linewidth=1.8, label="route")
    ego = root["ego_state"]
    ax.scatter([ego["x"]], [ego["y"]], marker="*", s=120, label="ego")
    for a in root.get("actor_states", [])[:80]:
        ax.scatter([a["x"]], [a["y"]], s=10, alpha=0.7)
    if args.labels:
        arrays, meta = read_dataset(args.labels)
        rid = root["root_id"]
        i = _find_root_index(arrays["root_ids"], rid) if "root_ids" in arrays else None
        if i is not None:
            ai = args.action_index; mi = args.mode_index
            if args.auto_action != "none":
                mask = np.asarray(arrays.get("action_mask", np.ones_like(arrays["R_star"][i], dtype=bool))[i],
                                  dtype=bool) if "R_star" in arrays else None
                if args.auto_action in ("best_recovery", "worst_recovery") and "R_star" in arrays:
                    score = np.asarray(arrays["R_star"][i], dtype=np.float32).copy()
                    if mask is not None and mask.shape == score.shape:
                        score[~mask] = -np.inf if args.auto_action == "best_recovery" else np.inf
                    ai = int(np.nanargmax(score) if args.auto_action == "best_recovery" else np.nanargmin(score))
                elif args.auto_action in ("least_harm", "largest_harm") and "H_action_star" in arrays:
                    score = np.asarray(arrays["H_action_star"][i], dtype=np.float32).copy()
                    if mask is not None and mask.shape == score.shape:
                        score[~mask] = np.inf if args.auto_action == "least_harm" else -np.inf
                    ai = int(np.nanargmin(score) if args.auto_action == "least_harm" else np.nanargmax(score))
            act = arrays["actions_states"][i, ai]
            x0, y0, hd = float(ego["x"]), float(ego["y"]), float(ego["heading"])
            c, s = np.cos(hd), np.sin(hd)
            aw = act.copy()
            x, y = aw[:, 0].copy(), aw[:, 1].copy()
            aw[:, 0] = x0 + c * x - s * y
            aw[:, 1] = y0 + s * x + c * y
            ax.plot(aw[:, 0], aw[:, 1], linewidth=2.2, label=f"action {ai}")
            oi = args.option_index
            if oi is None and "witness" in arrays:
                oi = int(arrays["witness"][i, ai, mi])
            if oi is not None:
                opt = arrays["options_states_ref"][i, ai, oi]
                ow = opt.copy()
                x, y = ow[:, 0].copy(), ow[:, 1].copy()
                ow[:, 0] = x0 + c * x - s * y
                ow[:, 1] = y0 + s * x + c * y
                ax.plot(ow[:, 0], ow[:, 1], linewidth=1.8, linestyle="--", label=f"option {oi}")
            parts = []
            if "R_star" in arrays:
                parts.append(f"R={float(arrays['R_star'][i, ai]):.3f}")
            if "H_action_star" in arrays:
                parts.append(f"H={float(arrays['H_action_star'][i, ai]):.3f}")
            title_extra = (" " + " ".join(parts)) if parts else ""
            ax.set_title(f"{rid} action={ai} mode={mi}{title_extra}")
            if args.summary_output:
                summary = {"root_id": rid, "action_index": int(ai), "mode_index": int(mi)}
                for key in ["R_star", "H_action_star", "Y_action", "witness", "witness_gap"]:
                    if key in arrays:
                        try:
                            val = np.asarray(arrays[key][i])
                            summary[key] = val.tolist() if val.ndim <= 2 else "omitted_high_dim"
                        except Exception:
                            pass
                sp = Path(args.summary_output)
                sp.parent.mkdir(parents=True, exist_ok=True)
                sp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.2)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(str(out), flush=True)


if __name__ == "__main__":
    main()
