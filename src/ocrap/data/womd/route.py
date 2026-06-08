from __future__ import annotations

import numpy as np


def parse_route(scenario, map_polylines: np.ndarray, map_valid: np.ndarray, max_points: int = 80) -> np.ndarray:
    # Prefer sdc_paths when present, else use route-flagged lane center, else SDC logged path proxy.
    pts = []
    paths = list(getattr(scenario, "sdc_paths", [])) if hasattr(scenario, "sdc_paths") else []
    if paths:
        for p in getattr(paths[0], "points", []):
            pts.append((float(getattr(p, "x", 0.0)), float(getattr(p, "y", 0.0))))
    if not pts and map_polylines.size:
        lane_mask = (map_polylines[..., 5] == 1) & map_valid
        pts = [(float(x), float(y)) for x, y in map_polylines[..., :2][lane_mask][:max_points]]
    if not pts:
        sdc = int(getattr(scenario, "sdc_track_index", 0))
        tracks = list(getattr(scenario, "tracks", []))
        if 0 <= sdc < len(tracks):
            for st in getattr(tracks[sdc], "states", []):
                if bool(getattr(st, "valid", False)):
                    pts.append((float(getattr(st, "center_x", 0.0)), float(getattr(st, "center_y", 0.0))))
    if not pts:
        pts = [(float(i), 0.0) for i in range(max_points)]
    pts_arr = np.asarray(pts[:max_points], dtype=np.float32)
    if len(pts_arr) < max_points:
        pad = np.repeat(pts_arr[-1:, :], max_points - len(pts_arr), axis=0)
        pts_arr = np.concatenate([pts_arr, pad], axis=0)
    route = np.zeros((max_points, 6), dtype=np.float32)
    route[:, :2] = pts_arr[:, :2]
    if max_points > 1:
        d = np.diff(route[:, :2], axis=0, append=route[-1:, :2])
        route[:, 2] = np.arctan2(d[:, 1], d[:, 0])
    route[:, 3] = 13.4
    route[:, 5] = 1.0
    # Mark route proximity flag in map array if caller uses returned route only this is harmless.
    return route
