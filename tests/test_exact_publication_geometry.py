from __future__ import annotations

import numpy as np

from ocrap.utils.geometry import (
    approximate_box_distance,
    min_oriented_box_clearance,
    min_oriented_box_signed_clearance,
    min_oriented_box_ttc,
    oriented_box_distance,
    oriented_box_signed_distance,
    oriented_box_ttc,
)


def _box(x: float, y: float = 0.0, *, vx: float = 0.0, vy: float = 0.0, yaw: float = 0.0, length: float = 4.0, width: float = 2.0) -> np.ndarray:
    return np.asarray([x, y, vx, vy, yaw, length, width, 1.5, 1.0], dtype=np.float32)


def test_exact_clearance_does_not_saturate_when_only_circumscribed_circles_overlap() -> None:
    ego = _box(0.0)
    other = _box(4.2)
    assert approximate_box_distance(ego, other) == 0.0
    assert abs(oriented_box_distance(ego, other) - 0.2) < 1.0e-5


def test_exact_clearance_is_zero_for_touch_or_overlap() -> None:
    ego = _box(0.0)
    assert oriented_box_distance(ego, _box(4.0)) == 0.0
    assert oriented_box_distance(ego, _box(3.5)) == 0.0


def test_exact_clearance_handles_rotated_boxes_and_valid_mask() -> None:
    ego = _box(0.0)
    boxes = np.stack([_box(20.0), _box(5.0, y=2.5, yaw=np.pi / 4)], axis=0)
    valid = np.asarray([False, True])
    value = min_oriented_box_clearance(ego, boxes, valid)
    assert np.isfinite(value)
    assert value > 0.0


def test_swept_sat_ttc_uses_vehicle_footprints() -> None:
    ego = _box(0.0)
    other = _box(5.0, vx=-1.0)
    # Surface gap is 1 m and relative closing speed is 1 m/s.
    assert abs(oriented_box_ttc(ego, other) - 1.0) < 1.0e-6


def test_swept_sat_ttc_returns_cap_when_separating() -> None:
    ego = _box(0.0)
    other = _box(5.0, vx=1.0)
    assert oriented_box_ttc(ego, other) == 99.0
    assert min_oriented_box_ttc(ego, np.stack([other]), np.asarray([True])) == 99.0


def test_signed_clearance_distinguishes_touch_and_penetration() -> None:
    ego = _box(0.0)
    assert abs(oriented_box_signed_distance(ego, _box(4.2)) - 0.2) < 1.0e-5
    assert abs(oriented_box_signed_distance(ego, _box(4.0))) < 1.0e-7
    # Axis-aligned 4 m boxes centered 3.5 m apart overlap by 0.5 m.
    assert abs(oriented_box_signed_distance(ego, _box(3.5)) + 0.5) < 1.0e-5
    boxes = np.stack([_box(5.0), _box(3.5)], axis=0)
    assert abs(min_oriented_box_signed_clearance(ego, boxes, np.asarray([True, True])) + 0.5) < 1.0e-5


def test_signed_penetration_uses_exit_translation_for_containment() -> None:
    # A small 1x1 footprint centered inside a 4x2 vehicle footprint must move
    # 1.5 m along the short axis to be fully separated.  Plain interval
    # intersection length would incorrectly report only 1.0 m.
    outer = _box(0.0, length=4.0, width=2.0)
    inner = _box(0.0, length=1.0, width=1.0)
    assert abs(oriented_box_signed_distance(outer, inner) + 1.5) < 1.0e-6
