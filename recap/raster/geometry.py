from __future__ import annotations

import math
import numpy as np
from PIL import Image, ImageDraw
from recap.utils.datatypes import BEVSpec, EgoState


def rotmat(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def world_to_ego(points_world: np.ndarray, ego: EgoState | np.ndarray) -> np.ndarray:
    p = np.asarray(points_world, dtype=np.float32)
    if isinstance(ego, EgoState):
        ex, ey, eh = ego.x, ego.y, ego.heading
    else:
        ex, ey, eh = float(ego[0]), float(ego[1]), float(ego[2])
    rel = p[..., :2] - np.array([ex, ey], dtype=np.float32)
    # rotate by -heading: world x/y into ego forward/left coordinates
    c, s = math.cos(eh), math.sin(eh)
    x = c * rel[..., 0] + s * rel[..., 1]
    y = -s * rel[..., 0] + c * rel[..., 1]
    return np.stack([x, y], axis=-1)


def ego_to_world(points_ego: np.ndarray, ego: EgoState | np.ndarray) -> np.ndarray:
    p = np.asarray(points_ego, dtype=np.float32)
    if isinstance(ego, EgoState):
        ex, ey, eh = ego.x, ego.y, ego.heading
    else:
        ex, ey, eh = float(ego[0]), float(ego[1]), float(ego[2])
    c, s = math.cos(eh), math.sin(eh)
    x = c * p[..., 0] - s * p[..., 1] + ex
    y = s * p[..., 0] + c * p[..., 1] + ey
    return np.stack([x, y], axis=-1)


def ego_to_bev_pixel(points_ego: np.ndarray, spec: BEVSpec) -> np.ndarray:
    p = np.asarray(points_ego, dtype=np.float32)
    # row decreases as x-forward increases. range_x[0] is backward.
    row = (spec.range_x[1] - p[..., 0]) / (spec.range_x[1] - spec.range_x[0]) * spec.H
    col = (p[..., 1] - spec.range_y[0]) / (spec.range_y[1] - spec.range_y[0]) * spec.W
    return np.stack([row, col], axis=-1).astype(np.float32)


def bev_pixel_to_ego(pixels: np.ndarray, spec: BEVSpec) -> np.ndarray:
    pix = np.asarray(pixels, dtype=np.float32)
    x = spec.range_x[1] - pix[..., 0] / spec.H * (spec.range_x[1] - spec.range_x[0])
    y = pix[..., 1] / spec.W * (spec.range_y[1] - spec.range_y[0]) + spec.range_y[0]
    return np.stack([x, y], axis=-1)


def rectangle_corners(length: float, width: float) -> np.ndarray:
    l, w = length / 2.0, width / 2.0
    return np.array([[l, w], [l, -w], [-l, -w], [-l, w]], dtype=np.float32)


def oriented_box(center_xy: np.ndarray, heading: float, length: float, width: float) -> np.ndarray:
    return rectangle_corners(length, width) @ rotmat(heading).T + np.asarray(center_xy, dtype=np.float32)


def rasterize_polygon(mask: np.ndarray, polygon_pixels: np.ndarray, value: float = 1.0) -> None:
    h, w = mask.shape
    pts = [(float(c), float(r)) for r, c in polygon_pixels]
    img = Image.fromarray(mask)
    draw = ImageDraw.Draw(img)
    draw.polygon(pts, fill=float(value))
    mask[:] = np.asarray(img, dtype=mask.dtype)


def draw_polyline(mask: np.ndarray, pts_pixels: np.ndarray, value: float = 1.0, width: int = 1) -> None:
    pts = [(float(c), float(r)) for r, c in pts_pixels]
    if len(pts) < 2:
        return
    img = Image.fromarray(mask)
    draw = ImageDraw.Draw(img)
    draw.line(pts, fill=float(value), width=width)
    mask[:] = np.asarray(img, dtype=mask.dtype)


def clip01(x):
    return np.clip(x, 0.0, 1.0)
