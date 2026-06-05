from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
from tqdm import tqdm

from .futures import generate_counterfactual_futures
from .geometry import agent_state_to_box, compute_ttc, min_box_clearance, transform_points_to_ego, transform_states_to_ego
from .io import ensure_dir, np_savez, write_json
from .observation import compatibility_labels, render_base_occ_mask, render_observation
from .ocmero import oc_mero
from .prefix_generation import generate_candidate_prefixes
from .recovery_options import default_recovery_options, option_valid_mask
from .root_clustering import aggregate_root_margins, cluster_roots, future_trajectory_signature
from .schema import CounterfactualFuture, DatasetSample, RawScenario, SceneHistory
from .split import scenario_split
from .synth import iter_synthetic_scenarios
from .teacher import compute_future_option_margins
from .womd import iter_womd_tfrecords


def ego_from_agent_state(agent_state: np.ndarray) -> np.ndarray:
    return np.array([agent_state[0], agent_state[1], agent_state[3], agent_state[4], agent_state[5], 0.0, math.hypot(agent_state[3], agent_state[4]), agent_state[6], agent_state[7]], dtype=np.float32)


def _transform_map(raw: RawScenario, ego_state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ego_xy = ego_state[:2]
    ego_h = float(ego_state[5])
    maps = raw.map_polylines.copy().astype(np.float32)
    if maps.size:
        xy = transform_points_to_ego(maps[..., :2], ego_xy, ego_h)
        dxy = maps[..., 3:5] @ np.array([[math.cos(-ego_h), -math.sin(-ego_h)], [math.sin(-ego_h), math.cos(-ego_h)]], dtype=np.float64).T
        maps[..., :2] = xy
        maps[..., 3:5] = dxy
    route = raw.route.copy().astype(np.float32)
    if route.size:
        route_xy = transform_points_to_ego(route[..., :2], ego_xy, ego_h)
        route[..., :2] = route_xy
    return maps, raw.map_valid.copy(), route


def select_planning_times(raw: RawScenario, cfg: dict) -> list[int]:
    sr = float(cfg.get("sample_rate_hz", 10))
    H = int(round(float(cfg.get("history_horizon_s", 1.0)) * sr))
    T_future = int(round((float(cfg.get("prefix_horizon_s", 1.0)) + float(cfg.get("recovery_horizon_s", 4.0))) * sr))
    start = H
    end = raw.agent_states.shape[0] - T_future - 1
    if end <= start:
        return []
    all_times = np.arange(start, end, dtype=np.int64)
    stride = max(1, int(round(float(cfg.get("planning_time_stride_s", 0.5)) * sr)))
    uniform = set(all_times[::stride].tolist())
    scores = []
    for t in all_times:
        ego = raw.agent_states[t, raw.sdc_track_index]
        if not raw.agent_valid[t, raw.sdc_track_index]:
            continue
        agents = np.delete(raw.agent_states[t], raw.sdc_track_index, axis=0)
        valids = np.delete(raw.agent_valid[t], raw.sdc_track_index, axis=0)
        boxes = np.asarray([agent_state_to_box(a) for a in agents], dtype=np.float32) if len(agents) else np.zeros((0, 9), dtype=np.float32)
        ego_box = agent_state_to_box(ego)
        dist = min_box_clearance(ego_box, boxes, valids) if len(boxes) else 99.0
        ttc = compute_ttc(np.array([ego[0], ego[1], 0, ego[3], ego[4], ego[5], 0, ego[6], ego[7]], dtype=np.float32), boxes, valids) if len(boxes) else 99.0
        score = max(0.0, 15.0 - dist) + max(0.0, 5.0 - ttc) * 2.0
        # Future near-contact bias.
        fut_end = min(raw.agent_states.shape[0], t + T_future)
        if fut_end > t + 1:
            ego_f = raw.agent_states[t:fut_end, raw.sdc_track_index, :2]
            for a in range(raw.agent_states.shape[1]):
                if a == raw.sdc_track_index:
                    continue
                mask = raw.agent_valid[t:fut_end, a]
                if mask.any():
                    d = np.linalg.norm(raw.agent_states[t:fut_end, a, :2] - ego_f, axis=-1)
                    score += max(0.0, 8.0 - float(np.min(d[mask])))
        scores.append((score, int(t)))
    scores.sort(reverse=True)
    n_bias = int(cfg.get("max_biased_times_per_scenario", 8))
    chosen = set(uniform)
    for _, t in scores[:n_bias]:
        chosen.add(t)
    max_times = int(cfg.get("max_times_per_scenario", 16))
    return sorted(chosen)[:max_times]


def construct_history(raw: RawScenario, t: int, cfg: dict) -> SceneHistory:
    sr = float(cfg.get("sample_rate_hz", 10))
    H = int(round(float(cfg.get("history_horizon_s", 1.0)) * sr))
    T_future = int(round((float(cfg.get("prefix_horizon_s", 1.0)) + float(cfg.get("recovery_horizon_s", 4.0))) * sr))
    sdc = raw.sdc_track_index
    agent_order = [sdc] + [i for i in range(raw.agent_states.shape[1]) if i != sdc]
    max_agents = min(int(cfg.get("max_agents", 64)), len(agent_order))
    agent_order = agent_order[:max_agents]
    ego_raw = raw.agent_states[t, sdc]
    hist = raw.agent_states[t - H + 1 : t + 1, agent_order]
    hist_valid = raw.agent_valid[t - H + 1 : t + 1, agent_order]
    future = raw.agent_states[t : t + T_future, agent_order]
    future_valid = raw.agent_valid[t : t + T_future, agent_order]
    hist_e = transform_states_to_ego(hist, ego_raw)
    fut_e = transform_states_to_ego(future, ego_raw)
    maps, map_valid, route = _transform_map(raw, ego_raw)
    dyn = raw.dynamic_map[max(0, t - H + 1) : t + 1]
    if dyn.shape[0] < H:
        pad = np.zeros((H - dyn.shape[0],) + dyn.shape[1:], dtype=np.float32)
        dyn = np.concatenate([pad, dyn], axis=0)
    ego = ego_from_agent_state(hist_e[-1, 0])
    h = SceneHistory(
        scene_id=raw.scenario_id,
        time_index=int(t),
        agent_history=hist_e.astype(np.float32),
        agent_valid=hist_valid.astype(bool),
        map_polylines=maps.astype(np.float32),
        map_valid=map_valid.astype(bool),
        dynamic_map=dyn.astype(np.float32),
        route=route.astype(np.float32),
        occ_mask=np.zeros((int(cfg.get("bev_channels", 7)), int(round(2 * float(cfg.get("local_radius_m", 80.0)) / float(cfg.get("bev_resolution_m", 0.5)))), int(round(2 * float(cfg.get("local_radius_m", 80.0)) / float(cfg.get("bev_resolution_m", 0.5))))), dtype=np.float32),
        ego_state=ego,
        future_agent_states=fut_e.astype(np.float32),
        future_agent_valid=future_valid.astype(bool),
        metadata={
            "speed_limit": float(np.nanmax(maps[..., 6]) if maps.size and np.nanmax(maps[..., 6]) > 0 else cfg.get("speed_limit_default", 13.4)),
            "shoulder_available": True,
            "near_contact": False,
            "post_contact": False,
        },
    )
    h.occ_mask = render_base_occ_mask(h, cfg)
    return h


def _assign_regimes(samples: list[DatasetSample], history: SceneHistory, cfg: dict) -> None:
    tau_high = float(cfg.get("regime_thresholds", {}).get("tau_high", 1.0))
    tau_d = float(cfg.get("regime_thresholds", {}).get("tau_d", 2.0))
    tau_ttc = float(cfg.get("regime_thresholds", {}).get("tau_ttc", 3.0))
    tau_occ = float(cfg.get("regime_thresholds", {}).get("tau_occ", 0.15))
    if not samples:
        return
    nominal_dep = float(samples[0].r_dep_star)
    max_dep = max(float(s.r_dep_star) for s in samples)
    boxes = np.asarray([agent_state_to_box(a) for a in history.agent_history[-1, 1:]], dtype=np.float32) if history.agent_history.shape[1] > 1 else np.zeros((0, 9), dtype=np.float32)
    valids = history.agent_valid[-1, 1:] if history.agent_history.shape[1] > 1 else np.zeros((0,), dtype=bool)
    ego_box = agent_state_to_box(history.agent_history[-1, 0])
    dmin = min_box_clearance(ego_box, boxes, valids) if len(boxes) else 99.0
    ttc = compute_ttc(history.ego_state, boxes, valids) if len(boxes) else 99.0
    unknown_ratio = float(np.mean(history.occ_mask[2] > 0.5)) if history.occ_mask.size else 0.0
    for s in samples:
        near = bool(dmin < tau_d or ttc < 1.5 or s.prefix.hard_violation > 0.0)
        post = bool(s.prefix.hard_violation > 0.0 or any(f.metadata.get("contact_surrogate", False) for f in s.futures))
        regimes = {
            "normal": bool(nominal_dep > tau_high and ttc > tau_ttc and dmin > tau_d),
            "low_headroom": bool(nominal_dep <= tau_high and max_dep > 0.0),
            "occluded": bool(unknown_ratio > tau_occ or any(f.metadata.get("hidden_emergence", False) for f in s.futures)),
            "near_contact": near,
            "post_contact": post,
            "oracle_artifact": bool(s.i_art_star),
        }
        s.regime_label.clear()
        s.regime_label.update(regimes)


def build_samples_for_history(history: SceneHistory, split_id: str, cfg: dict) -> list[DatasetSample]:
    prefixes = generate_candidate_prefixes(history, cfg)
    options = default_recovery_options(shoulder_available=bool(history.metadata.get("shoulder_available", True)), adjacent_available=True)
    option_valid = option_valid_mask(options)
    samples: list[DatasetSample] = []
    for a_idx, prefix in enumerate(prefixes):
        futures = generate_counterfactual_futures(history, prefix, cfg)
        future_probs = np.asarray([f.prior for f in futures], dtype=np.float32)
        future_probs = future_probs / max(float(future_probs.sum()), 1e-8)
        M_future, teacher_diags = compute_future_option_margins(history, prefix, futures, options, cfg)
        root = cluster_roots(M_future, future_probs, futures, cfg)
        K = int(cfg.get("num_roots", 8))
        M_star = aggregate_root_margins(M_future, root.assignments, future_probs, K)
        root_future_signature = future_trajectory_signature(futures, root.assignments, future_probs, K, width=int(cfg.get("model", {}).get("d_future_signature", 32)))
        observations = []
        for k in range(K):
            rep = int(root.representative_indices[k]) if root.root_valid[k] and root.representative_indices[k] >= 0 else 0
            observations.append(render_observation(history, prefix, futures[rep], cfg))
        Y, C, Dobs = compatibility_labels(observations, cfg)
        res = oc_mero(M_star, root.root_probs, C, alpha=float(cfg.get("ocmero", {}).get("alpha", 0.2)), beta=float(cfg.get("ocmero", {}).get("beta", 0.2)), option_valid=option_valid, use_lcvar=True, use_obs_kernel=True, top_m=int(cfg.get("ocmero", {}).get("top_m", 8)))
        gamma_orc = float(cfg.get("artifact", {}).get("gamma_orc", 0.0))
        gamma_dep = float(cfg.get("artifact", {}).get("gamma_dep", 0.0))
        sample = DatasetSample(
            scene_id=history.scene_id,
            time_index=history.time_index,
            candidate_index=a_idx,
            split_id=split_id,
            is_nominal=(a_idx == 0),
            h_t=history,
            prefix=prefix,
            futures=futures,
            future_probs=future_probs,
            root_assignments=root.assignments,
            root_probs=root.root_probs,
            root_signature=root.root_signature,
            root_future_signature=root_future_signature,
            root_valid=root.root_valid,
            future_to_root_weight=root.future_to_root_weight,
            observations=observations,
            y_obs=Y,
            c_star=C,
            recovery_options=options,
            m_star=M_star,
            option_valid=option_valid,
            r_orc_star=res.r_orc,
            r_dep_star=res.r_dep,
            oracle_gap_star=res.gap,
            i_art_star=bool(res.r_orc >= gamma_orc and res.r_dep < gamma_dep),
            regime_label={},
            valid_masks={"root_valid": root.root_valid.astype(bool).tolist(), "option_valid": option_valid.astype(bool).tolist()},
            diagnostics={
                "future_sources": [f.source for f in futures],
                "obs_distance": Dobs.tolist(),
                "teacher_component_sample": teacher_diags[0][0].component_margins if teacher_diags and teacher_diags[0] else {},
            },
        )
        samples.append(sample)
    _assign_regimes(samples, history, cfg)
    return samples


def scenario_iterator(cfg: dict) -> Iterator[RawScenario]:
    source = cfg.get("data_source", "synthetic")
    if source == "synthetic":
        yield from iter_synthetic_scenarios(int(cfg.get("num_synthetic_scenarios", 16)), seed=int(cfg.get("seed", 0)))
    elif source == "womd":
        patterns = cfg.get("womd_patterns")
        if not patterns:
            raise ValueError("data_source=womd requires womd_patterns in config or CLI override")
        yield from iter_womd_tfrecords(patterns, max_scenarios=cfg.get("max_scenarios"), max_agents=int(cfg.get("max_agents", 64)), max_polylines=int(cfg.get("max_map_polylines", 256)), max_points=int(cfg.get("max_polyline_points", 32)))
    else:
        raise ValueError(f"Unknown data_source {source}")


def build_dataset(output_dir: str | Path, cfg: dict) -> dict:
    out = ensure_dir(output_dir)
    sample_dir = ensure_dir(out / "samples")
    manifest_rows = []
    split_counts: dict[str, int] = {"train": 0, "val": 0, "calibration": 0, "test": 0}
    total_samples = 0
    for raw in tqdm(scenario_iterator(cfg), desc="build_scenarios"):
        split_id = scenario_split(raw.scenario_id, cfg.get("split_ratios"))
        times = select_planning_times(raw, cfg)
        for t in times:
            history = construct_history(raw, t, cfg)
            samples = build_samples_for_history(history, split_id, cfg)
            for sample in samples:
                fname = f"{sample.scene_id}_t{sample.time_index:04d}_a{sample.candidate_index:02d}.npz".replace("/", "_")
                path = sample_dir / fname
                np_savez(path, **sample.to_npz_dict())
                manifest_rows.append({
                    "path": str(path.relative_to(out)),
                    "scene_id": sample.scene_id,
                    "time_index": sample.time_index,
                    "candidate_index": sample.candidate_index,
                    "split_id": split_id,
                    "is_nominal": int(sample.is_nominal),
                    "r_orc_star": sample.r_orc_star,
                    "r_dep_star": sample.r_dep_star,
                    "i_art_star": int(sample.i_art_star),
                    "regime_label": ";".join(k for k, v in sample.regime_label.items() if v),
                })
                split_counts[split_id] = split_counts.get(split_id, 0) + 1
                total_samples += 1
    manifest_path = out / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()) if manifest_rows else ["path"])
        writer.writeheader()
        writer.writerows(manifest_rows)
    summary = {"num_samples": total_samples, "split_counts": split_counts, "manifest": str(manifest_path)}
    write_json(summary, out / "dataset_summary.json")
    return summary
