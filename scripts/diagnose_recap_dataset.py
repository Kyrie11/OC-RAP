#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# Allow running scripts directly from a source checkout without pip install -e .
import sys as _sys
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from ocrap.teacher.dataset_writer import read_dataset


def _iter_rows(x):
    if hasattr(x, "iter_shard_arrays"):
        yield from x.iter_shard_arrays()
    else:
        yield np.asarray(x)


def _as_str_list(x, limit=None):
    n = len(x) if hasattr(x, "__len__") else len(np.asarray(x))
    if limit is not None:
        n = min(n, int(limit))
    return [str(x[i]) for i in range(n)]


def _stats(x, max_values: int = 2_000_000):
    count = 0
    total = 0.0
    total2 = 0.0
    mn = float("inf")
    mx = float("-inf")
    samples = []
    remaining_budget = max(1, int(max_values))
    for chunk in _iter_rows(x):
        arr = np.asarray(chunk, dtype=np.float32).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        count += int(arr.size)
        total += float(arr.sum(dtype=np.float64))
        total2 += float(np.square(arr, dtype=np.float64).sum(dtype=np.float64))
        mn = min(mn, float(arr.min()))
        mx = max(mx, float(arr.max()))
        if remaining_budget > 0:
            stride = max(1, int(np.ceil(arr.size / remaining_budget)))
            take = arr[::stride]
            if take.size > remaining_budget:
                take = take[:remaining_budget]
            samples.append(take.astype(np.float32, copy=False))
            remaining_budget -= int(take.size)
    if count == 0:
        return {"count": 0}
    mean = total / count
    var = max(0.0, total2 / count - mean * mean)
    sample = np.concatenate(samples) if samples else np.asarray([], dtype=np.float32)
    if sample.size:
        p05, p50, p95 = [float(x) for x in np.quantile(sample, [0.05, 0.50, 0.95])]
        quantile_note = "approx_from_sample" if sample.size < count else "exact"
    else:
        p05 = p50 = p95 = float("nan")
        quantile_note = "unavailable"
    return {"count": int(count), "mean": float(mean), "std": float(np.sqrt(var)), "min": mn, "p05": p05, "p50": p50, "p95": p95, "max": mx, "quantiles": quantile_note, "quantile_sample_count": int(sample.size)}


def _update_scalar_acc(acc: dict, value: float) -> None:
    if not np.isfinite(value):
        acc["nan"] += 1
        return
    acc["count"] += 1
    acc["sum"] += float(value)
    acc["sum2"] += float(value) * float(value)
    acc["min"] = min(acc["min"], float(value))
    acc["max"] = max(acc["max"], float(value))

def _finish_scalar_acc(acc: dict) -> dict:
    n = int(acc.get("count", 0))
    if n == 0:
        return {"count": 0, "nan": int(acc.get("nan", 0))}
    mean = acc["sum"] / n
    var = max(0.0, acc["sum2"] / n - mean * mean)
    return {"count": n, "mean": float(mean), "std": float(np.sqrt(var)), "min": float(acc["min"]),
            "max": float(acc["max"]), "nan": int(acc.get("nan", 0))}

def _new_acc() -> dict:
    return {"count": 0, "sum": 0.0, "sum2": 0.0, "min": float("inf"), "max": float("-inf"), "nan": 0}

def _label_health(arrays: dict, root_ids: list[str], regimes: list[str], max_roots: int | None = None) -> dict:
    n = len(root_ids) if max_roots is None else min(len(root_ids), int(max_roots))
    by_regime = defaultdict(
        lambda: {"n": 0, "R_mean": _new_acc(), "R_spread": _new_acc(), "recoverable_action_rate": _new_acc(),
                 "harm_action_rate": _new_acc(), "mode_disagreement": _new_acc()})
    global_acc = {"R_mean": _new_acc(), "R_spread": _new_acc(), "recoverable_action_rate": _new_acc(),
                       "harm_action_rate": _new_acc(), "mode_disagreement": _new_acc()}
    all_zero_R = 0
    all_one_R = 0
    nontrivial_roots = 0
    best_low = []
    most_spread = []
    for i in range(n):
        reg = regimes[i] if i < len(regimes) else "unknown"
        by_regime[reg]["n"] += 1
        R = np.asarray(arrays["R_star"][i], dtype=np.float32) if "R_star" in arrays else np.asarray([],
                                                                                                    dtype=np.float32)
        H = np.asarray(arrays["H_action_star"][i], dtype=np.float32) if "H_action_star" in arrays else np.asarray([],
                                                                                                                  dtype=np.float32)
        Ya = np.asarray(arrays["Y_action"][i], dtype=np.float32) if "Y_action" in arrays else np.asarray([],
                                                                                                         dtype=np.float32)
        if R.size:
            r_mean = float(np.nanmean(R))
            r_spread = float(np.nanmax(R) - np.nanmin(R))
            rec_rate = float(np.nanmean(R >= 0.70))
            all_zero_R += int(np.nanmax(R) <= 1e-6)
            all_one_R += int(np.nanmin(R) >= 1.0 - 1e-6)
            nontrivial_roots += int(r_spread > 0.05)
            best_low.append((r_mean, root_ids[i]))
            most_spread.append((r_spread, root_ids[i]))
            for acc_name, val in [("R_mean", r_mean), ("R_spread", r_spread), ("recoverable_action_rate", rec_rate)]:
                _update_scalar_acc(global_acc[acc_name], val)
                _update_scalar_acc(by_regime[reg][acc_name], val)
        if H.size:
            harm_rate = float(np.nanmean(H > 0.05))
            _update_scalar_acc(global_acc["harm_action_rate"], harm_rate)
            _update_scalar_acc(by_regime[reg]["harm_action_rate"], harm_rate)
        if Ya.size and Ya.ndim >= 2:
            # Mean action-level standard deviation over modes: zero means modes are not creating label diversity.
            md = float(np.nanmean(np.nanstd(Ya, axis=-1)))
            _update_scalar_acc(global_acc["mode_disagreement"], md)
            _update_scalar_acc(by_regime[reg]["mode_disagreement"], md)
    best_low = sorted(best_low, key=lambda x: x[0])[:10]
    most_spread = sorted(most_spread, key=lambda x: x[0], reverse=True)[:10]
    return {
        "sampled_roots": int(n),
        "all_zero_R_roots": int(all_zero_R),
        "all_one_R_roots": int(all_one_R),
        "nontrivial_action_ranking_roots": int(nontrivial_roots),
        "nontrivial_action_ranking_rate": float(nontrivial_roots / max(n, 1)),
        "global": {k: _finish_scalar_acc(v) for k, v in global_acc.items()},
        "by_regime": {
            r: {k: (_finish_scalar_acc(v) if isinstance(v, dict) and "count" in v else v) for k, v in d.items()} for
            r, d in by_regime.items()},
        "example_roots": {
            "lowest_mean_recovery": [{"root_id": rid, "R_mean": float(v)} for v, rid in best_low],
            "largest_action_recovery_spread": [{"root_id": rid, "R_spread": float(v)} for v, rid in
                                                            most_spread],
            },
        }

def _mask_health(arrays: dict, max_roots: int | None = None) -> dict:
    out = {}
    if "action_mask" in arrays:
        x = arrays["action_mask"]
        n = len(x) if max_roots is None else min(len(x), int(max_roots))
        vals = [float(np.asarray(x[i]).sum()) for i in range(n)]
        out["valid_actions_per_root"] = _stats(np.asarray(vals, dtype=np.float32))
        out["action_valid_ratio"] = float(np.mean([v / max(np.asarray(x[0]).size, 1) for v in vals])) if vals else None
    if "option_mask" in arrays:
        x = arrays["option_mask"]
        n = len(x) if max_roots is None else min(len(x), int(max_roots))
        vals = [float(np.asarray(x[i]).sum()) for i in range(n)]
        denom = float(np.asarray(x[0]).size) if n else 1.0
        out["valid_options_per_root"] = _stats(np.asarray(vals, dtype=np.float32))
        out["option_valid_ratio"] = float(np.mean([v / max(denom, 1.0) for v in vals])) if vals else None
    return out

def _bev_health(arrays: dict, meta: dict, sample_roots: int) -> dict | None:
    if "bev" not in arrays:
        return None
    n = min(len(arrays["bev"]), int(sample_roots))
    if n <= 0:
        return {"sampled_roots": 0}
    channel_names = meta.get("channel_names") or (meta.get("bev_spec", {}) or {}).get("channel_names") or []
    acc = None
    nonzero = None
    for i in range(n):
        bev = np.asarray(arrays["bev"][i], dtype=np.float32)
        # [T,C,H,W] -> per-channel mean/nonzero, averaged over time.
        cm = bev.mean(axis=(0, 2, 3)) if bev.ndim == 4 else np.asarray([], dtype=np.float32)
        cz = (bev > 0).mean(axis=(0, 2, 3)) if bev.ndim == 4 else np.asarray([], dtype=np.float32)
        acc = cm if acc is None else acc + cm
        nonzero = cz if nonzero is None else nonzero + cz
    acc = acc / max(n, 1)
    nonzero = nonzero / max(n, 1)
    names = [str(c) for c in channel_names]
    if not names or len(names) != len(acc):
        names = [f"ch_{i}" for i in range(len(acc))]
    suspicious = [names[i] for i, z in enumerate(nonzero) if float(z) < 1e-6]
    return {
        "sampled_roots": int(n),
        "channel_nonzero_fraction": {names[i]: float(nonzero[i]) for i in range(len(names))},
        "channel_mean": {names[i]: float(acc[i]) for i in range(len(names))},
        "empty_channels_in_sample": suspicious,
    }


def _mode_health(arrays: dict, max_roots: int | None = None) -> dict:
    out = {}
    if "Y_action" in arrays:
        x = arrays["Y_action"]
        n = len(x) if max_roots is None else min(len(x), int(max_roots))
        vals = []
        for i in range(n):
            arr = np.asarray(x[i], dtype=np.float32)
            if arr.ndim >= 2:
                vals.append(float(np.nanmean(np.nanstd(arr, axis=-1))))
        out["mode_label_disagreement"] = _stats(np.asarray(vals, dtype=np.float32)) if vals else {"count": 0}
    if "margin_option" in arrays:
        x = arrays["margin_option"]
        n = len(x) if max_roots is None else min(len(x), int(max_roots))
        vals = []
        for i in range(n):
            arr = np.asarray(x[i], dtype=np.float32)  # [K,L,M]
            if arr.ndim >= 3:
                best_per_mode = np.nanmax(arr, axis=1)  # [K,M]
                vals.append(float(np.nanmean(np.nanstd(best_per_mode, axis=-1))))
        out["mode_best_margin_disagreement"] = _stats(np.asarray(vals, dtype=np.float32)) if vals else {"count": 0}
    return out


def _paper_quality_gate(report: dict) -> dict:
    """Conservative pre-full-generation checks, not final paper metrics."""
    n = int(report.get("num_roots", 0) or 0)
    lh = report.get("label_health") or {}
    all_zero = int(lh.get("all_zero_R_roots", 0) or 0)
    nontriv = float(lh.get("nontrivial_action_ranking_rate", 0.0) or 0.0)
    pos_root_rate = 1.0 - float(all_zero) / max(n, 1)
    regimes = report.get("regime_counts") or {}
    mode_h = report.get("mode_health") or {}
    mode_label = float(((mode_h.get("mode_label_disagreement") or {}).get("mean", 0.0) or 0.0))
    mode_margin = float(((mode_h.get("mode_best_margin_disagreement") or {}).get("mean", 0.0) or 0.0))
    margin_max = float(((report.get("margin_option") or {}).get("max", -1.0) or -1.0))
    checks = {
        "has_positive_recovery_roots": pos_root_rate >= 0.20,
        "has_same_root_action_ranking": nontriv >= 0.20,
        "has_root_shared_mode_variation": (mode_label > 0.01) or (mode_margin > 0.02),
        "has_all_four_regimes": all(regimes.get(k, 0) > 0 for k in ["normal_high_headroom", "low_headroom", "near_contact", "contact_post_contact"]),
        "has_positive_option_margin": margin_max > 0.0,
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "positive_recovery_root_rate": float(pos_root_rate),
        "nontrivial_action_ranking_rate": float(nontriv),
        "mode_label_disagreement_mean": float(mode_label),
        "mode_best_margin_disagreement_mean": float(mode_margin),
        "note": "Use as a pre-full-generation gate; final paper acceptance still requires held-out closed-loop evaluation.",
    }

def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnostics for ReCAP/MetaDrive-Recovery label datasets.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--roots", default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-stat-values", type=int, default=2_000_000, help="Maximum values kept for approximate quantiles per array; means/min/max are streamed exactly.")
    ap.add_argument("--sample-roots", type=int, default=512,
                    help="Number of roots for per-root diagnostics and BEV channel checks.")
    args = ap.parse_args()
    arrays, meta = read_dataset(args.dataset)
    root_ids = _as_str_list(arrays.get("root_ids", []))
    regimes = _as_str_list(arrays.get("regime", []))
    report = {
        "dataset": args.dataset,
        "metadata": meta,
        "num_roots": len(root_ids),
        "root_id_sample": root_ids[:10],
        "regime_counts": dict(Counter(regimes)),
        "mask_health": _mask_health(arrays, args.sample_roots),
        "R_star": _stats(arrays["R_star"], args.max_stat_values) if "R_star" in arrays else None,
        "Y_action_rate": _stats(arrays["Y_action"], args.max_stat_values) if "Y_action" in arrays else None,
        "H_action_star": _stats(arrays["H_action_star"],
                                         args.max_stat_values) if "H_action_star" in arrays else None,
        "margin_option": _stats(arrays["margin_option"],
                                         args.max_stat_values) if "margin_option" in arrays else None,
        "M_path_raw": _stats(arrays["M_path_raw"], args.max_stat_values) if "M_path_raw" in arrays else None,
        "M_path_rec": _stats(arrays["M_path_rec"], args.max_stat_values) if "M_path_rec" in arrays else None,
        "M_return": _stats(arrays["M_return"], args.max_stat_values) if "M_return" in arrays else None,
        "M_ctrl": _stats(arrays["M_ctrl"], args.max_stat_values) if "M_ctrl" in arrays else None,
        "M_post": _stats(arrays["M_post"], args.max_stat_values) if "M_post" in arrays else None,
        "witness_gap": _stats(arrays["witness_gap"],
                                       args.max_stat_values) if "witness_gap" in arrays else None,
        "label_health": _label_health(arrays, root_ids, regimes,
                                               args.sample_roots) if "R_star" in arrays else None,
        "mode_health": _mode_health(arrays, args.sample_roots),
        "bev_health": _bev_health(arrays, meta, args.sample_roots),
        "synthetic_guard": {
            "is_synthetic": bool(meta.get("is_synthetic", True)),
            "paper_final_ready": bool(meta.get("paper_final_ready", False)),
            "rollout_backend": meta.get("rollout_backend"),
            "root_backend": meta.get("root_backend"),
        },
    }
    report["paper_quality_gate"] = _paper_quality_gate(report)
    if not report["paper_quality_gate"]["passed"]:
        failed = [k for k, ok in report["paper_quality_gate"]["checks"].items() if not ok]
        report.setdefault("warnings", []).append(
            "Paper-quality gate failed: " + ", ".join(failed) + ". Do not start full training/generation until this passes on a paper-check sample."
        )
    # Signature of a common MetaDrive adapter failure: the ego vehicle leaked into
    # the surrounding-actor list.  Its self-clearance is approximately
    # -0.5*(ego_length+ego_length)/8 = -4.7/8 = -0.5875, which exactly caps the
    # best option margin and makes every recoverability label false.
    mo = report.get("margin_option") or {}
    lh = report.get("label_health") or {}
    if float(mo.get("max", 1.0)) <= -0.55 and int(lh.get("all_zero_R_roots", -1)) == int(report.get("num_roots", -2)):
        report.setdefault("warnings", []).append(
            "All roots have zero R_star and max margin_option is near -0.5875. "
            "This is the signature of ego/self being included as a traffic actor in MetaDriveStateAdapter.get_actor_states(). "
            "Regenerate teacher labels after filtering the ego from actor_states."
        )
    if args.roots:
        root_dir = Path(args.roots)
        missing = [rid for rid in root_ids if not (root_dir / f"{rid}.json").exists()]
        scenario_backed = 0
        scenario_ids = []
        root_ticks = []
        for rid in root_ids[: min(len(root_ids), max(args.sample_roots, 1000))]:
            p = root_dir / f"{rid}.json"
            if p.exists():
                obj = json.loads(p.read_text())
                sd = obj.get("scenario_data", {}) or {}
                if sd.get("scenario_pkl"):
                    scenario_backed += 1
                scenario_ids.append(str(sd.get("scenario_id", "")))
                root_ticks.append(int(sd.get("current_time_index", obj.get("root_tick", -1))))
        dup_scenarios = [sid for sid, c in Counter(scenario_ids).items() if sid and c > 1]
        report["roots_check"] = {
            "missing_root_json": missing[:20],
            "num_missing": len(missing),
            "scenario_backed_in_sample": scenario_backed,
            "duplicate_scenario_ids_in_sample": dup_scenarios[:20],
            "num_duplicate_scenario_ids_in_sample": len(dup_scenarios),
            "root_tick_counts_in_sample": dict(Counter(root_ticks)),
        }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
