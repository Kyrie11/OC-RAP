from __future__ import annotations

import math
from typing import Iterable

import numpy as np


EPS = 1e-8


def wrap_angle(a: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def rotation_matrix(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def transform_points_to_ego(points: np.ndarray, ego_xy: np.ndarray, ego_heading: float) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    R = rotation_matrix(-ego_heading)
    out = (pts[..., :2] - ego_xy[:2]) @ R.T
    if pts.shape[-1] > 2:
        return np.concatenate([out, pts[..., 2:]], axis=-1)
    return out


def transform_states_to_ego(states: np.ndarray, ego_state: np.ndarray) -> np.ndarray:
    s = np.asarray(states, dtype=np.float64).copy()
    ego_xy = ego_state[:2]
    ego_h = float(ego_state[5] if s.shape[-1] >= 10 else ego_state[4])
    xy = transform_points_to_ego(s[..., :2], ego_xy, ego_h)
    vxy = s[..., 3:5] @ rotation_matrix(-ego_h).T if s.shape[-1] >= 5 else np.zeros_like(xy)
    s[..., :2] = xy
    if s.shape[-1] >= 5:
        s[..., 3:5] = vxy
    if s.shape[-1] >= 6:
        s[..., 5] = wrap_angle(s[..., 5] - ego_h)
    return s.astype(np.float32)


def obb_corners(cx: float, cy: float, length: float, width: float, heading: float) -> np.ndarray:
    hl, hw = 0.5 * max(float(length), 0.05), 0.5 * max(float(width), 0.05)
    local = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]], dtype=np.float64)
    return local @ rotation_matrix(float(heading)).T + np.array([cx, cy], dtype=np.float64)


def polygon_axes(poly: np.ndarray) -> list[np.ndarray]:
    axes = []
    for i in range(len(poly)):
        edge = poly[(i + 1) % len(poly)] - poly[i]
        normal = np.array([-edge[1], edge[0]], dtype=np.float64)
        n = np.linalg.norm(normal)
        if n > EPS:
            axes.append(normal / n)
    return axes


def project(poly: np.ndarray, axis: np.ndarray) -> tuple[float, float]:
    vals = poly @ axis
    return float(vals.min()), float(vals.max())


def sat_overlap_depth(poly_a: np.ndarray, poly_b: np.ndarray) -> tuple[bool, float]:
    min_overlap = float("inf")
    for axis in polygon_axes(poly_a) + polygon_axes(poly_b):
        a0, a1 = project(poly_a, axis)
        b0, b1 = project(poly_b, axis)
        overlap = min(a1, b1) - max(a0, b0)
        if overlap <= 0:
            return False, 0.0
        min_overlap = min(min_overlap, overlap)
    return True, min_overlap


def point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(ab @ ab)
    if denom <= EPS:
        return float(np.linalg.norm(p - a))
    t = float(np.clip(((p - a) @ ab) / denom, 0.0, 1.0))
    q = a + t * ab
    return float(np.linalg.norm(p - q))


def polygon_distance(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
    inter, depth = sat_overlap_depth(poly_a, poly_b)
    if inter:
        return -float(depth)
    best = float("inf")
    for p in poly_a:
        for i in range(len(poly_b)):
            best = min(best, point_segment_distance(p, poly_b[i], poly_b[(i + 1) % len(poly_b)]))
    for p in poly_b:
        for i in range(len(poly_a)):
            best = min(best, point_segment_distance(p, poly_a[i], poly_a[(i + 1) % len(poly_a)]))
    return best


def box_signed_clearance(box_a: np.ndarray, box_b: np.ndarray) -> float:
    # box format x,y,vx,vy,heading,l,w,h,type or x,y,heading,l,w
    if box_a.shape[-1] >= 8:
        ca = obb_corners(box_a[0], box_a[1], box_a[5], box_a[6], box_a[4])
    else:
        ca = obb_corners(box_a[0], box_a[1], box_a[3], box_a[4], box_a[2])
    if box_b.shape[-1] >= 8:
        cb = obb_corners(box_b[0], box_b[1], box_b[5], box_b[6], box_b[4])
    else:
        cb = obb_corners(box_b[0], box_b[1], box_b[3], box_b[4], box_b[2])
    return polygon_distance(ca, cb)


def min_box_clearance(ego_box: np.ndarray, boxes: np.ndarray, valid: np.ndarray | None = None) -> float:
    if boxes.size == 0:
        return float("inf")
    if valid is None:
        valid = np.ones((boxes.shape[0],), dtype=bool)
    best = float("inf")
    for b, ok in zip(boxes, valid):
        if ok:
            best = min(best, box_signed_clearance(ego_box, b))
    return best


def polyline_distance(point: np.ndarray, polyline: np.ndarray, valid: np.ndarray | None = None) -> float:
    pts = np.asarray(polyline, dtype=np.float64)
    if pts.ndim != 2 or len(pts) == 0:
        return float("inf")
    if valid is not None:
        pts = pts[np.asarray(valid).astype(bool)]
    if len(pts) == 0:
        return float("inf")
    if len(pts) == 1:
        return float(np.linalg.norm(point[:2] - pts[0, :2]))
    best = float("inf")
    p = np.asarray(point[:2], dtype=np.float64)
    for a, b in zip(pts[:-1, :2], pts[1:, :2]):
        best = min(best, point_segment_distance(p, a, b))
    return best


def nearest_polyline_distance(point: np.ndarray, polylines: np.ndarray, valid: np.ndarray | None = None, route_flag_index: int | None = None) -> float:
    best = float("inf")
    if polylines.size == 0:
        return best
    for i, pl in enumerate(polylines):
        if route_flag_index is not None and pl.shape[-1] > route_flag_index:
            if np.nanmax(pl[:, route_flag_index]) <= 0.5:
                continue
        v = valid[i] if valid is not None and len(valid) > i else None
        best = min(best, polyline_distance(point, pl, v))
    return best


def compute_ttc(ego_state: np.ndarray, boxes: np.ndarray, valid: np.ndarray, max_ttc: float = 99.0) -> float:
    ego_xy = ego_state[:2]
    ego_v = np.array([ego_state[3] if len(ego_state) > 3 else 0.0, ego_state[4] if len(ego_state) > 4 else 0.0])
    best = max_ttc
    for b, ok in zip(boxes, valid):
        if not ok:
            continue
        rel_p = b[:2] - ego_xy
        rel_v = (b[2:4] if len(b) >= 4 else np.zeros(2)) - ego_v
        closing = -float(rel_p @ rel_v) / (np.linalg.norm(rel_p) + EPS)
        if closing <= 0:
            continue
        ttc = float(np.linalg.norm(rel_p) / closing)
        best = min(best, ttc)
    return best


def smooth_step(s: np.ndarray) -> np.ndarray:
    return 3 * s**2 - 2 * s**3


def agent_state_to_box(state: np.ndarray) -> np.ndarray:
    s = np.asarray(state, dtype=np.float64)
    if s.shape[-1] >= 10:
        return np.array([s[0], s[1], s[3], s[4], s[5], s[6], s[7], s[8], s[9]], dtype=np.float64)
    if s.shape[-1] >= 9:
        return np.array([s[0], s[1], s[2], s[3], s[4], s[6], s[7], 1.5, 0.0], dtype=np.float64)
    raise ValueError(f"Unsupported agent state shape {s.shape}")


def ego_state_to_box(state: np.ndarray) -> np.ndarray:
    s = np.asarray(state, dtype=np.float64)
    if s.shape[-1] >= 9:
        return np.array([s[0], s[1], s[2], s[3], s[4], s[7], s[8], 1.5, 1.0], dtype=np.float64)
    raise ValueError(f"Unsupported ego state shape {s.shape}")


def fast_box_signed_clearance(box_a: np.ndarray, box_b: np.ndarray, precise_radius: float = 12.0) -> float:
    a = np.asarray(box_a, dtype=np.float64)
    b = np.asarray(box_b, dtype=np.float64)
    da = 0.5 * math.hypot(float(a[5]), float(a[6])) if a.shape[-1] >= 7 else 2.5
    db = 0.5 * math.hypot(float(b[5]), float(b[6])) if b.shape[-1] >= 7 else 2.5
    center = float(np.linalg.norm(a[:2] - b[:2]))
    lower = center - da - db
    if center > precise_radius and lower > 0.0:
        return lower
    return box_signed_clearance(a, b)
