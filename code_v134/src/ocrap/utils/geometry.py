from __future__ import annotations

import math
from itertools import permutations
from typing import Iterable

import numpy as np


EPS = 1e-8


def wrap_angle(a: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(a) + math.pi) % (2.0 * math.pi) - math.pi


def rotation_matrix(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def transform_points_to_ego(points: np.ndarray, ego_xy: np.ndarray, ego_heading: float) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    R = rotation_matrix(-float(ego_heading))
    return (pts[..., :2] - np.asarray(ego_xy, dtype=np.float32)) @ R.T


def transform_vectors_to_ego(vecs: np.ndarray, ego_heading: float) -> np.ndarray:
    R = rotation_matrix(-float(ego_heading))
    return np.asarray(vecs, dtype=np.float32) @ R.T


def transform_states_to_ego(states: np.ndarray, ego_state: np.ndarray) -> np.ndarray:
    out = np.asarray(states, dtype=np.float32).copy()
    if out.size == 0:
        return out
    ego_xy = np.asarray(ego_state[:2], dtype=np.float32)
    ego_heading = float(ego_state[7] if len(ego_state) > 7 else ego_state[5])
    xy = transform_points_to_ego(out[..., :2], ego_xy, ego_heading)
    vxy = transform_vectors_to_ego(out[..., 3:5], ego_heading)
    out[..., :2] = xy
    out[..., 3:5] = vxy
    if out.shape[-1] > 7:
        out[..., 7] = wrap_angle(out[..., 7] - ego_heading)
        out[..., 8] = np.sin(out[..., 7])
        out[..., 9] = np.cos(out[..., 7])
    elif out.shape[-1] > 5:
        out[..., 5] = wrap_angle(out[..., 5] - ego_heading)
    return out


def speed_from_state(s: np.ndarray) -> float:
    return float(math.hypot(float(s[3]), float(s[4])))


def heading_from_state(s: np.ndarray) -> float:
    # Agent state uses heading at index 7 in the new schema and index 5 in the legacy schema.
    if len(s) >= 10:
        return float(s[7])
    return float(s[5])


def agent_state_to_box(s: np.ndarray) -> np.ndarray:
    """Return [x,y,vx,vy,heading,length,width,height,type]."""
    s = np.asarray(s, dtype=np.float32)
    if s.shape[-1] >= 16:
        return np.array([s[0], s[1], s[3], s[4], s[7], s[10], s[11], s[12], s[13]], dtype=np.float32)
    return np.array([s[0], s[1], s[3], s[4], s[5], s[6], s[7], s[8], s[9]], dtype=np.float32)


def ego_state_to_box(s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, dtype=np.float32)
    return np.array([s[0], s[1], s[2], s[3], s[4], s[7], s[8], 1.5, 1.0], dtype=np.float32)


def oriented_box_corners(box: np.ndarray) -> np.ndarray:
    x, y, _, _, h, length, width = np.asarray(box, dtype=np.float32)[:7]
    local = np.array([[ length/2,  width/2], [ length/2, -width/2], [-length/2, -width/2], [-length/2,  width/2]], dtype=np.float32)
    return local @ rotation_matrix(float(h)).T + np.array([x, y], dtype=np.float32)


def _polygon_axes(poly: np.ndarray) -> list[np.ndarray]:
    """Return unit separating axes for a convex polygon.

    The closed-loop publication metrics use oriented vehicle footprints, so
    center-distance/circumscribed-circle approximations are not sufficiently
    precise near contact.  Rectangles only require their edge normals, but this
    helper is kept generic for small convex polygons.
    """
    p = np.asarray(poly, dtype=np.float64)
    axes: list[np.ndarray] = []
    for a, b in zip(p, np.roll(p, -1, axis=0)):
        edge = b - a
        axis = np.asarray([-edge[1], edge[0]], dtype=np.float64)
        n = float(np.linalg.norm(axis))
        if n > EPS:
            axes.append(axis / n)
    return axes


def _interval(poly: np.ndarray, axis: np.ndarray) -> tuple[float, float]:
    proj = np.asarray(poly, dtype=np.float64) @ np.asarray(axis, dtype=np.float64)
    return float(np.min(proj)), float(np.max(proj))


def _polygons_overlap_sat(poly_a: np.ndarray, poly_b: np.ndarray, *, eps: float = 1.0e-9) -> bool:
    """Exact overlap/touch test for two convex polygons via SAT."""
    for axis in _polygon_axes(poly_a) + _polygon_axes(poly_b):
        amin, amax = _interval(poly_a, axis)
        bmin, bmax = _interval(poly_b, axis)
        if amax < bmin - eps or bmax < amin - eps:
            return False
    return True


def _point_segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    p = np.asarray(point, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= EPS:
        return float(np.linalg.norm(p - a))
    t = float(np.dot(p - a, ab) / denom)
    t = min(1.0, max(0.0, t))
    return float(np.linalg.norm(p - (a + t * ab)))


def convex_polygon_distance(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
    """Euclidean boundary clearance between two convex polygons.

    Returns 0 for overlap or touch.  Unlike the historical circle proxy this
    is the true 2-D footprint clearance, including rotated rectangles and the
    containment case.
    """
    a = np.asarray(poly_a, dtype=np.float64)
    b = np.asarray(poly_b, dtype=np.float64)
    if len(a) < 2 or len(b) < 2:
        return 99.0
    if _polygons_overlap_sat(a, b):
        return 0.0
    edges_a = list(zip(a, np.roll(a, -1, axis=0)))
    edges_b = list(zip(b, np.roll(b, -1, axis=0)))
    best = float("inf")
    for p in a:
        for c, d in edges_b:
            best = min(best, _point_segment_distance(p, c, d))
    for p in b:
        for c, d in edges_a:
            best = min(best, _point_segment_distance(p, c, d))
    return float(best) if np.isfinite(best) else 99.0


def convex_polygon_signed_distance(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
    """Signed separation for two convex polygons.

    Positive values are exact Euclidean boundary clearance.  Zero means touch.
    Negative values are SAT minimum-translation penetration depth.  For the
    rectangle footprints used by the publication metrics this provides a
    deterministic geometric penetration proxy without conflating overlap with
    zero clearance.
    """
    a = np.asarray(poly_a, dtype=np.float64)
    b = np.asarray(poly_b, dtype=np.float64)
    axes = _polygon_axes(a) + _polygon_axes(b)
    if not axes:
        return 99.0
    separation_depths: list[float] = []
    separated = False
    for axis in axes:
        amin, amax = _interval(a, axis)
        bmin, bmax = _interval(b, axis)
        if amax < bmin - 1.0e-9 or bmax < amin - 1.0e-9:
            separated = True
            break
        # Minimum translation along this unit SAT axis.  The common
        # ``intersection length`` formula is wrong for interval containment
        # (e.g. a pedestrian footprint fully inside a vehicle footprint): the
        # smaller interval must travel to the *nearest exterior boundary*.
        depth = min(amax - bmin, bmax - amin)
        separation_depths.append(max(0.0, float(depth)))
    if separated:
        return convex_polygon_distance(a, b)
    return -float(min(separation_depths)) if separation_depths else 0.0


def oriented_box_distance(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Exact non-negative Euclidean clearance between two oriented boxes."""
    return convex_polygon_distance(oriented_box_corners(box_a), oriented_box_corners(box_b))


def oriented_box_signed_distance(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Signed OBB gap: positive clearance, zero touch, negative penetration."""
    return convex_polygon_signed_distance(oriented_box_corners(box_a), oriented_box_corners(box_b))


def _valid_boxes(boxes: np.ndarray, valid: np.ndarray) -> np.ndarray:
    arr = np.asarray(boxes, dtype=np.float64)
    mask = np.asarray(valid).astype(bool).reshape(-1)
    if arr.ndim != 2 or arr.shape[0] != mask.shape[0]:
        return np.empty((0, 9), dtype=np.float64)
    return arr[mask]


def min_oriented_box_signed_clearance(ego_box: np.ndarray, boxes: np.ndarray, valid: np.ndarray) -> float:
    """Worst signed OBB gap with an exact circumscribed-circle broad phase.

    Circle separation is a lower bound on rectangle clearance.  We therefore
    evaluate exact polygon distance only for candidates that can still beat the
    current best value; when penetration is found, only circle-overlapping
    candidates can be more negative.  This preserves exactness while avoiding
    O(N) vertex/edge distance work for far-away agents.
    """
    ego = np.asarray(ego_box, dtype=np.float64)
    other = _valid_boxes(boxes, valid)
    if other.size == 0:
        return 99.0
    ego_r = 0.5 * math.hypot(float(ego[5]), float(ego[6]))
    radii = 0.5 * np.hypot(other[:, 5], other[:, 6])
    centers = np.linalg.norm(other[:, :2] - ego[None, :2], axis=1)
    lower = np.maximum(0.0, centers - ego_r - radii)
    order = np.argsort(lower, kind="stable")
    best = float("inf")
    for idx in order:
        lb = float(lower[idx])
        if best < 0.0 and lb > 0.0:
            break
        if best >= 0.0 and lb >= best:
            break
        value = oriented_box_signed_distance(ego, other[int(idx)])
        if value < best:
            best = float(value)
    return float(best) if np.isfinite(best) else 99.0


def min_oriented_box_clearance(ego_box: np.ndarray, boxes: np.ndarray, valid: np.ndarray) -> float:
    """Minimum exact non-negative OBB clearance to any valid other agent."""
    return max(0.0, min_oriented_box_signed_clearance(ego_box, boxes, valid))


def oriented_box_ttc(box_a: np.ndarray, box_b: np.ndarray, *, max_ttc_s: float = 99.0) -> float:
    """Constant-velocity TTC between two oriented rectangles using swept SAT.

    Headings and footprint dimensions are frozen over the prediction.  Linear
    velocities are taken from ``box[2:4]``.  The result is 0 for current
    overlap/touch and ``max_ttc_s`` when no footprint collision is predicted
    within the horizon.  This is still a proxy for future dynamics, but unlike
    the previous radial center TTC it is footprint-aware and geometrically
    consistent with the clearance metric.
    """
    a = np.asarray(box_a, dtype=np.float64)
    b = np.asarray(box_b, dtype=np.float64)
    pa = np.asarray(oriented_box_corners(a), dtype=np.float64)
    pb = np.asarray(oriented_box_corners(b), dtype=np.float64)
    rel_v = np.asarray(b[2:4] - a[2:4], dtype=np.float64)

    t_enter = 0.0
    t_exit = float("inf")
    for axis in _polygon_axes(pa) + _polygon_axes(pb):
        amin, amax = _interval(pa, axis)
        bmin, bmax = _interval(pb, axis)
        vel = float(np.dot(rel_v, axis))
        if abs(vel) <= EPS:
            if amax < bmin or bmax < amin:
                return float(max_ttc_s)
            continue

        t0 = (amin - bmax) / vel
        t1 = (amax - bmin) / vel
        axis_enter = min(t0, t1)
        axis_exit = max(t0, t1)
        t_enter = max(t_enter, axis_enter)
        t_exit = min(t_exit, axis_exit)
        if t_enter > t_exit:
            return float(max_ttc_s)

    if t_exit < 0.0 or not np.isfinite(t_enter):
        return float(max_ttc_s)
    return float(min(max_ttc_s, max(0.0, t_enter)))


def _circle_ttc_lower_bounds(ego: np.ndarray, other: np.ndarray, max_ttc_s: float) -> np.ndarray:
    """Earliest circumscribed-circle contact; a lower bound on OBB TTC."""
    rel_p = other[:, :2] - ego[None, :2]
    rel_v = other[:, 2:4] - ego[None, 2:4]
    ego_r = 0.5 * math.hypot(float(ego[5]), float(ego[6]))
    radii = ego_r + 0.5 * np.hypot(other[:, 5], other[:, 6])
    a = np.sum(rel_v * rel_v, axis=1)
    b = 2.0 * np.sum(rel_p * rel_v, axis=1)
    c = np.sum(rel_p * rel_p, axis=1) - radii * radii
    out = np.full(len(other), float(max_ttc_s), dtype=np.float64)
    out[c <= 0.0] = 0.0
    moving = (a > EPS) & (c > 0.0)
    disc = b * b - 4.0 * a * c
    hit = moving & (disc >= 0.0)
    if np.any(hit):
        root = np.sqrt(np.maximum(disc[hit], 0.0))
        ah = a[hit]; bh = b[hit]
        enter = (-bh - root) / (2.0 * ah)
        exit_ = (-bh + root) / (2.0 * ah)
        t = np.where(exit_ >= 0.0, np.maximum(0.0, enter), float(max_ttc_s))
        out[np.flatnonzero(hit)] = np.minimum(float(max_ttc_s), t)
    return out


def min_oriented_box_ttc(
    ego_box: np.ndarray,
    boxes: np.ndarray,
    valid: np.ndarray,
    *,
    max_ttc_s: float = 99.0,
) -> float:
    """Minimum exact swept-OBB TTC with a safe circle-TTC broad phase."""
    ego = np.asarray(ego_box, dtype=np.float64)
    other = _valid_boxes(boxes, valid)
    if other.size == 0:
        return float(max_ttc_s)
    lower = _circle_ttc_lower_bounds(ego, other, float(max_ttc_s))
    order = np.argsort(lower, kind="stable")
    best = float(max_ttc_s)
    for idx in order:
        lb = float(lower[idx])
        if lb >= best:
            break
        value = oriented_box_ttc(ego, other[int(idx)], max_ttc_s=max_ttc_s)
        if value < best:
            best = float(value)
        if best <= 0.0:
            break
    return float(best)


def approximate_box_distance(box_a: np.ndarray, box_b: np.ndarray) -> float:
    a = np.asarray(box_a, dtype=np.float32)
    b = np.asarray(box_b, dtype=np.float32)
    center = float(np.linalg.norm(a[:2] - b[:2]))
    radius_a = 0.5 * math.hypot(float(a[5]), float(a[6]))
    radius_b = 0.5 * math.hypot(float(b[5]), float(b[6]))
    return max(0.0, center - radius_a - radius_b)


def min_box_clearance(ego_box: np.ndarray, boxes: np.ndarray, valid: np.ndarray) -> float:
    vals = [approximate_box_distance(ego_box, b) for b, ok in zip(boxes, valid.astype(bool)) if ok]
    return float(min(vals)) if vals else 99.0


def compute_ttc(ego_state: np.ndarray, boxes: np.ndarray, valid: np.ndarray) -> float:
    ego = np.asarray(ego_state, dtype=np.float32)
    ego_xy = ego[:2]
    ego_v = ego[2:4] if ego.shape[0] <= 9 else ego[3:5]
    best = 99.0
    for b, ok in zip(boxes, valid.astype(bool)):
        if not ok:
            continue
        rel = np.asarray(b[:2], dtype=np.float32) - ego_xy
        rv = ego_v - np.asarray(b[2:4], dtype=np.float32)
        closing = float(np.dot(rv, rel) / (np.linalg.norm(rel) + EPS))
        if closing > 0:
            best = min(best, max(0.0, float(np.linalg.norm(rel)) / closing))
    return float(best)


def greedy_assignment_cost(A: np.ndarray, B: np.ndarray, unmatch_penalty: float) -> float:
    """Small dependency-free visible-box matching cost."""
    n, m = len(A), len(B)
    if n == 0 and m == 0:
        return 0.0
    if n == 0 or m == 0:
        return float(max(n, m) * unmatch_penalty)
    if min(n, m) <= 7:
        if n <= m:
            best = float("inf")
            for perm in permutations(range(m), n):
                cost = sum(float(np.linalg.norm(A[i, :2] - B[j, :2])) for i, j in enumerate(perm))
                cost += (m - n) * unmatch_penalty
                best = min(best, cost)
            return float(best)
        best = float("inf")
        for perm in permutations(range(n), m):
            cost = sum(float(np.linalg.norm(A[i, :2] - B[j, :2])) for j, i in enumerate(perm))
            cost += (n - m) * unmatch_penalty
            best = min(best, cost)
        return float(best)
    # Greedy fallback for many boxes.
    remaining = set(range(m))
    cost = 0.0
    for i in range(n):
        if not remaining:
            cost += unmatch_penalty
            continue
        j = min(remaining, key=lambda jj: float(np.linalg.norm(A[i, :2] - B[jj, :2])))
        cost += float(np.linalg.norm(A[i, :2] - B[j, :2]))
        remaining.remove(j)
    cost += len(remaining) * unmatch_penalty
    return float(cost)
