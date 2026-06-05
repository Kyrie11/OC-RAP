from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from .geometry import agent_state_to_box, compute_ttc, ego_state_to_box, wrap_angle
from .schema import CounterfactualFuture, Observation, CandidatePrefix, SceneHistory


def _grid_coords(radius: float, resolution: float) -> tuple[np.ndarray, np.ndarray]:
    n = int(round(2 * radius / resolution))
    xs = (np.arange(n, dtype=np.float32) + 0.5) * resolution - radius
    ys = (np.arange(n, dtype=np.float32) + 0.5) * resolution - radius
    return np.meshgrid(xs, ys, indexing="xy")


def render_base_occ_mask(history: SceneHistory, cfg: dict) -> np.ndarray:
    radius = float(cfg.get("local_radius_m", 80.0))
    res = float(cfg.get("bev_resolution_m", 0.5))
    C = int(cfg.get("bev_channels", 7))
    X, Y = _grid_coords(radius, res)
    H, W = X.shape
    mask = np.zeros((C, H, W), dtype=np.float32)
    r = np.sqrt(X**2 + Y**2)
    in_range = r <= radius
    # channels: visible_free, occupied_visible, unknown, occluder, route, drivable, confidence
    mask[0] = in_range.astype(np.float32)
    mask[2] = (~in_range).astype(np.float32)
    mask[5] = in_range.astype(np.float32)
    if history.route.size > 0:
        for pt in history.route[:, :2]:
            ix = int((pt[0] + radius) / res)
            iy = int((pt[1] + radius) / res)
            if 0 <= ix < W and 0 <= iy < H:
                y0, y1 = max(0, iy - 2), min(H, iy + 3)
                x0, x1 = max(0, ix - 2), min(W, ix + 3)
                mask[4, y0:y1, x0:x1] = 1.0
    mask[6] = mask[0]
    return mask


def angular_interval_for_box(box: np.ndarray, ego_xy: np.ndarray) -> tuple[float, float, float]:
    x, y, vx, vy, heading, length, width = box[:7]
    dx, dy = x - ego_xy[0], y - ego_xy[1]
    d = math.hypot(dx, dy)
    center = math.atan2(dy, dx)
    half = math.atan2(0.5 * max(width, length * 0.5), max(d, 1e-3))
    return center - half, center + half, d


def is_occluded_by_dynamic(box: np.ndarray, occluders: list[np.ndarray], ego_xy: np.ndarray) -> bool:
    if not occluders:
        return False
    a0, a1, d = angular_interval_for_box(box, ego_xy)
    c = 0.5 * (a0 + a1)
    for occ in occluders:
        b0, b1, od = angular_interval_for_box(occ, ego_xy)
        if od >= d - 0.5:
            continue
        # handle without wrap by comparing wrapped angle to interval center
        width = abs(b1 - b0)
        if abs(float(wrap_angle(c - 0.5 * (b0 + b1)))) <= 0.5 * width:
            return True
    return False


def visible_agent_boxes(agent_states: np.ndarray, agent_valid: np.ndarray, ego_state: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray, list[int]]:
    radius = float(cfg.get("local_radius_m", 80.0))
    ego_xy = ego_state[:2]
    boxes = []
    idxs = []
    occluders = []
    for i, (s, ok) in enumerate(zip(agent_states, agent_valid.astype(bool))):
        if i == 0 or not ok:
            continue
        b = agent_state_to_box(s)
        if np.linalg.norm(b[:2] - ego_xy) <= radius:
            # vehicles/trucks/buses act as occluders; type id <= 2 treated as vehicle-like.
            if b[-1] in (1, 2, 3) or b[5] > 4.5:
                occluders.append(b)
    for i, (s, ok) in enumerate(zip(agent_states, agent_valid.astype(bool))):
        if i == 0 or not ok:
            continue
        b = agent_state_to_box(s)
        dist = np.linalg.norm(b[:2] - ego_xy)
        if dist <= radius and not is_occluded_by_dynamic(b, occluders, ego_xy):
            boxes.append(b)
            idxs.append(i)
    if not boxes:
        return np.zeros((0, 9), dtype=np.float32), np.zeros((0,), dtype=bool), []
    return np.asarray(boxes, dtype=np.float32), np.ones((len(boxes),), dtype=bool), idxs


def paint_boxes_on_mask(mask: np.ndarray, boxes: np.ndarray, cfg: dict) -> np.ndarray:
    out = mask.copy()
    radius = float(cfg.get("local_radius_m", 80.0))
    res = float(cfg.get("bev_resolution_m", 0.5))
    H, W = out.shape[-2:]
    for b in boxes:
        ix = int((b[0] + radius) / res)
        iy = int((b[1] + radius) / res)
        radx = max(1, int(max(b[5], b[6]) / res / 2))
        y0, y1 = max(0, iy - radx), min(H, iy + radx + 1)
        x0, x1 = max(0, ix - radx), min(W, ix + radx + 1)
        if 0 <= ix < W and 0 <= iy < H:
            out[1, y0:y1, x0:x1] = 1.0
            out[0, y0:y1, x0:x1] = 0.0
    # Unknown is range/occlusion not visible free or occupied; retain existing unknown.
    out[6] = np.maximum(out[6], out[1])
    return out


def render_observation(history: SceneHistory, prefix: CandidatePrefix, future: CounterfactualFuture, cfg: dict) -> Observation:
    T_p = prefix.prefix_states.shape[0]
    idx = min(T_p - 1, future.agent_states.shape[0] - 1)
    ego = prefix.prefix_states[-1].copy().astype(np.float32)
    if future.metadata.get("contact_surrogate", False):
        ego[5] += float(future.metadata.get("yaw_rate_impulse", 0.0))
        lateral = float(future.metadata.get("lateral_velocity_impulse", 0.0))
        ego[2] += -math.sin(float(ego[4])) * lateral
        ego[3] += math.cos(float(ego[4])) * lateral
    boxes, valid, visible_idxs = visible_agent_boxes(future.agent_states[idx], future.agent_valid[idx], ego, cfg)
    occ = paint_boxes_on_mask(render_base_occ_mask(history, cfg), boxes, cfg)
    contact_flag = bool(prefix.hard_violation > 0.0 or future.metadata.get("contact_surrogate", False))
    stability = np.array([ego[5], np.linalg.norm(ego[2:4]), float(contact_flag)], dtype=np.float32)
    return Observation(ego_state=ego, boxes=boxes, box_valid=valid, occ_mask=occ, contact_flag=contact_flag, stability_proxy=stability)


def box_distance(obs_i: Observation, obs_j: Observation, cfg: dict) -> float:
    params = cfg.get("obs_distance", {})
    s_c = float(params.get("s_c", 2.0))
    s_v = float(params.get("s_v", 2.0))
    lam_psi = float(params.get("lambda_psi", 0.5))
    lam_v = float(params.get("lambda_v", 0.5))
    lam_type = float(params.get("lambda_type", 2.0))
    p_unmatch = float(params.get("p_unmatch", 5.0))
    A = obs_i.boxes[obs_i.box_valid.astype(bool)]
    B = obs_j.boxes[obs_j.box_valid.astype(bool)]
    n, m = len(A), len(B)
    if max(n, m) == 0:
        return 0.0
    if n == 0 or m == 0:
        return p_unmatch
    cost = np.zeros((n, m), dtype=np.float64)
    for a in range(n):
        for b in range(m):
            cost[a, b] = (
                np.linalg.norm(A[a, :2] - B[b, :2]) / s_c
                + lam_psi * (1.0 - math.cos(float(A[a, 4] - B[b, 4])))
                + lam_v * np.linalg.norm(A[a, 2:4] - B[b, 2:4]) / s_v
                + lam_type * float(A[a, 8] != B[b, 8])
            )
    try:
        from scipy.optimize import linear_sum_assignment

        row, col = linear_sum_assignment(cost)
        matched_cost = float(cost[row, col].sum())
        unmatched = n + m - 2 * len(row)
    except Exception:
        matched_cost = 0.0
        unmatched = abs(n - m)
        used_b: set[int] = set()
        for a in range(n):
            b = int(np.argmin(cost[a]))
            if b not in used_b:
                used_b.add(b)
                matched_cost += float(cost[a, b])
            else:
                unmatched += 1
    return float((matched_cost + p_unmatch * unmatched) / max(n, m, 1))


def occ_distance(obs_i: Observation, obs_j: Observation, cfg: dict) -> float:
    ch_unknown = 2
    A = obs_i.occ_mask[ch_unknown] > 0.5
    B = obs_j.occ_mask[ch_unknown] > 0.5
    inter = np.logical_and(A, B).sum()
    union = np.logical_or(A, B).sum()
    if union == 0:
        return 0.0
    return float(1.0 - inter / union)


def ego_dynamic_distance(obs_i: Observation, obs_j: Observation, cfg: dict) -> float:
    params = cfg.get("obs_distance", {})
    s_p = float(params.get("s_p", 2.0))
    s_v = float(params.get("s_v", 2.0))
    s_yaw = float(params.get("s_yaw", 0.2))
    lam_r = float(params.get("lambda_r", 0.5))
    return float(np.linalg.norm(obs_i.ego_state[:2] - obs_j.ego_state[:2]) / s_p + abs(obs_i.ego_state[6] - obs_j.ego_state[6]) / s_v + lam_r * abs(obs_i.ego_state[5] - obs_j.ego_state[5]) / s_yaw)


def observation_distance(obs_i: Observation, obs_j: Observation, cfg: dict) -> float:
    params = cfg.get("obs_distance", {})
    return float(
        box_distance(obs_i, obs_j, cfg)
        + float(params.get("lambda_occ", 1.0)) * occ_distance(obs_i, obs_j, cfg)
        + float(params.get("lambda_ego", 0.5)) * ego_dynamic_distance(obs_i, obs_j, cfg)
    )


def compatibility_labels(observations: list[Observation], cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    K = len(observations)
    eps_o = float(cfg.get("epsilon_obs", 1.0))
    tau_o = float(cfg.get("tau_obs", eps_o**2 / math.log(2.0)))
    D = np.zeros((K, K), dtype=np.float32)
    Y = np.zeros((K, K), dtype=np.float32)
    C = np.zeros((K, K), dtype=np.float32)
    for i in range(K):
        for j in range(K):
            d = observation_distance(observations[i], observations[j], cfg)
            D[i, j] = d
            Y[i, j] = float(d <= eps_o)
            C[i, j] = math.exp(-(d**2) / max(tau_o, 1e-8))
    np.fill_diagonal(Y, 1.0)
    np.fill_diagonal(C, 1.0)
    return Y, C, D
