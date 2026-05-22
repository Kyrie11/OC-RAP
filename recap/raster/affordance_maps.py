from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from recap.utils.datatypes import BEVSpec, EgoState, RouteInfo, MapFeatures
from .geometry import bev_pixel_to_ego

AFFORDANCE_NAMES = ["stop", "lane", "route", "escape", "stabilize"]


@dataclass
class AffordanceProvider:
    version: str = "heuristic_v1"
    eta_A: float = 0.35

    def build_maps(self, spec: BEVSpec, ego: EgoState, route_info: RouteInfo, map_features: MapFeatures) -> dict[str, np.ndarray]:
        rows, cols = np.meshgrid(np.arange(spec.H, dtype=np.float32), np.arange(spec.W, dtype=np.float32), indexing="ij")
        pts = bev_pixel_to_ego(np.stack([rows, cols], axis=-1), spec)
        x, y = pts[..., 0], pts[..., 1]
        route_y = 0.0
        lane_width = 3.6
        # Values are target/energy-like maps normalized to [0,1], higher is more available.
        lane = np.exp(-np.abs(y - route_y) / lane_width).astype(np.float32)
        route = np.exp(-(np.abs(y - route_y) / 4.0 + np.maximum(0.0, -x) / 20.0)).astype(np.float32)
        stop_anchor_x = max(ego.v * ego.v / (2 * 3.0) + 2.0, 5.0)
        stop = np.exp(-np.sqrt((x - stop_anchor_x) ** 2 + y**2) / 15.0).astype(np.float32)
        escape = np.maximum(np.exp(-np.abs(y - 4.0) / 2.0), np.exp(-np.abs(y + 4.0) / 2.0)).astype(np.float32)
        stabilize = np.exp(-(np.abs(y) / 5.0 + np.abs(x - 5.0) / 20.0)).astype(np.float32)
        return {"stop": stop, "lane": lane, "route": route, "escape": escape, "stabilize": stabilize}


def heuristic_affordance_cost(state: np.ndarray, kind: str = "route") -> float:
    x, y, psi, v = float(state[0]), float(state[1]), float(state[2]), float(state[3])
    if kind == "stop":
        return abs(v) / 10.0 + max(0.0, abs(y) - 2.0) / 5.0
    if kind == "lane":
        return abs(y) / 2.0 + abs(psi) / np.pi
    if kind == "route":
        return abs(y) / 4.0 + max(0.0, -x) / 20.0 + abs(psi) / np.pi
    if kind == "escape":
        return max(0.0, 1.5 - abs(y)) / 1.5 + abs(psi) / np.pi
    if kind == "stabilize":
        return abs(v) / 10.0 + abs(psi) / np.pi + abs(y) / 5.0
    raise ValueError(kind)
