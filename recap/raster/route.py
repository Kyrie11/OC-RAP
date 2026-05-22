from __future__ import annotations

import numpy as np
from recap.utils.datatypes import RouteInfo, EgoState
from .geometry import world_to_ego


def route_command_from_route(route: RouteInfo, ego: EgoState, N_q: int = 20, D_q: int = 6) -> np.ndarray:
    wp_world = np.asarray(route.waypoints, dtype=np.float32)
    if wp_world.shape[-1] >= 2:
        wp_ego = world_to_ego(wp_world[..., :2], ego)
    else:
        wp_ego = np.zeros((0, 2), dtype=np.float32)
    out = np.zeros((N_q, D_q), dtype=np.float32)
    n = min(N_q, len(wp_ego))
    for i in range(n):
        if wp_world.shape[-1] >= 3:
            heading = float(wp_world[i, 2]) - ego.heading
        elif i + 1 < len(wp_ego):
            d = wp_ego[i + 1] - wp_ego[i]
            heading = float(np.arctan2(d[1], d[0]))
        else:
            heading = 0.0
        out[i, 0:2] = wp_ego[i]
        out[i, 2] = np.sin(heading)
        out[i, 3] = np.cos(heading)
        out[i, 4] = route.speed_limit_mps / 20.0
        cid = 0 if route.command_ids is None or i >= len(route.command_ids) else int(route.command_ids[i])
        out[i, 5] = cid
    return out
