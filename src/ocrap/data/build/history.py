from __future__ import annotations

import math

import numpy as np

from ocrap.data.schema import RawScenario, SceneHistory
from ocrap.simulation.observation.bev import render_base_occ_mask
from ocrap.utils.geometry import transform_points_to_ego, transform_states_to_ego


def ego_from_agent_state(agent_state: np.ndarray) -> np.ndarray:
    return np.array([agent_state[0], agent_state[1], agent_state[3], agent_state[4], agent_state[7], 0.0, math.hypot(agent_state[3], agent_state[4]), agent_state[10], agent_state[11]], dtype=np.float32)


def transform_map_and_route(raw: RawScenario, ego_state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ego_xy = ego_state[:2]
    ego_h = float(ego_state[7])
    maps = raw.map_polylines.copy().astype(np.float32)
    if maps.size:
        maps[..., :2] = transform_points_to_ego(maps[..., :2], ego_xy, ego_h)
    route = raw.route.copy().astype(np.float32)
    if route.size:
        route[..., :2] = transform_points_to_ego(route[..., :2], ego_xy, ego_h)
        if route.shape[-1] > 2:
            route[..., 2] = route[..., 2] - ego_h
    return maps, raw.map_valid.copy(), route


def construct_history(raw: RawScenario, t: int, cfg: dict) -> SceneHistory:
    sr = float(cfg.get("sample_rate_hz", 10))
    H = max(1, int(round(float(cfg.get("history_horizon_s", 1.0)) * sr)))
    total_future = max(2, int(round((float(cfg.get("prefix_horizon_s", 1.0)) + float(cfg.get("recovery_horizon_s", 4.0))) * sr)))
    sdc = int(raw.sdc_track_index)
    order = [sdc] + [i for i in range(raw.agent_states.shape[1]) if i != sdc]
    max_agents = min(int(cfg.get("max_agents", len(order))), len(order))
    order = order[:max_agents]
    t0 = max(0, t - H + 1)
    hist = raw.agent_states[t0 : t + 1, order]
    hist_valid = raw.agent_valid[t0 : t + 1, order]
    if hist.shape[0] < H:
        pad = np.zeros((H - hist.shape[0], hist.shape[1], hist.shape[2]), dtype=np.float32)
        pad_valid = np.zeros((H - hist.shape[0], hist.shape[1]), dtype=bool)
        hist = np.concatenate([pad, hist], axis=0)
        hist_valid = np.concatenate([pad_valid, hist_valid], axis=0)
    tend = min(raw.agent_states.shape[0], t + total_future)
    future = raw.agent_states[t:tend, order]
    future_valid = raw.agent_valid[t:tend, order]
    if future.shape[0] < total_future:
        pad = np.zeros((total_future - future.shape[0], len(order), raw.agent_states.shape[-1]), dtype=np.float32)
        pad_valid = np.zeros((total_future - future.shape[0], len(order)), dtype=bool)
        future = np.concatenate([future, pad], axis=0)
        future_valid = np.concatenate([future_valid, pad_valid], axis=0)
    ego_raw = raw.agent_states[t, sdc]
    hist_e = transform_states_to_ego(hist, ego_raw)
    fut_e = transform_states_to_ego(future, ego_raw)
    maps, map_valid, route = transform_map_and_route(raw, ego_raw)
    dyn = raw.dynamic_map[max(0, t - H + 1) : t + 1]
    if dyn.shape[0] < H:
        pad = np.zeros((H - dyn.shape[0],) + dyn.shape[1:], dtype=np.float32)
        dyn = np.concatenate([pad, dyn], axis=0)
    h = SceneHistory(
        scene_id=raw.scenario_id,
        original_scenario_id=str(raw.metadata.get("original_scenario_id", raw.scenario_id)),
        time_index=int(t),
        agent_history=hist_e.astype(np.float32),
        agent_valid=hist_valid.astype(bool),
        map_polylines=maps.astype(np.float32),
        map_valid=map_valid.astype(bool),
        dynamic_map=dyn.astype(np.float32),
        route=route.astype(np.float32),
        occ_mask=np.zeros((int(cfg.get("bev_channels", 7)), 2, 2), dtype=np.float32),
        ego_state=ego_from_agent_state(hist_e[-1, 0]),
        future_agent_states=fut_e.astype(np.float32),
        future_agent_valid=future_valid.astype(bool),
        metadata={
            "speed_limit": float(cfg.get("speed_limit_default", 13.4)),
            "shoulder_available": True,
            "adjacent_available": True,
            "time_sampling_reasons": [],
            "source": raw.metadata.get("source", "unknown"),
        },
    )
    h.occ_mask = render_base_occ_mask(h, cfg)
    return h
