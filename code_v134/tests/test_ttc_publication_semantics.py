from __future__ import annotations

import numpy as np

from ocrap.utils.geometry import oriented_box_ttc


def _box(x: float, y: float, vx: float, vy: float, yaw: float = 0.0, length: float = 4.0, width: float = 2.0) -> np.ndarray:
    return np.asarray([x, y, vx, vy, yaw, length, width, 1.5, 1.0], dtype=np.float64)


def test_swept_obb_ttc_is_zero_only_for_current_contact_in_simple_longitudinal_case() -> None:
    ego = _box(0.0, 0.0, 2.0, 0.0)
    touching = _box(4.0, 0.0, 0.0, 0.0)
    separated = _box(6.0, 0.0, 0.0, 0.0)
    assert oriented_box_ttc(ego, touching) == 0.0
    # Center gap 6 m minus two 2 m half-lengths leaves 2 m footprint clearance;
    # at 2 m/s relative closing speed, contact occurs in 1 second.
    assert np.isclose(oriented_box_ttc(ego, separated), 1.0)


def test_swept_obb_ttc_returns_horizon_for_receding_pair() -> None:
    ego = _box(0.0, 0.0, 0.0, 0.0)
    other = _box(6.0, 0.0, 2.0, 0.0)
    assert oriented_box_ttc(ego, other, max_ttc_s=99.0) == 99.0


def test_linear_scene_p05_floor_at_250_scenes_matches_near_table_pattern() -> None:
    # For N=250, NumPy's linear q=.05 rank is 12.45.  Thirteen zeros occupy
    # indices 0..12 and therefore still interpolate toward the 14th positive
    # value; fourteen zeros occupy 0..13 and force the p05 to zero.
    thirteen = np.asarray([0.0] * 13 + [0.05] + [1.0] * 236)
    fourteen = np.asarray([0.0] * 14 + [1.0] * 236)
    assert np.isclose(np.quantile(thirteen, 0.05, method="linear"), 0.0225)
    assert np.quantile(fourteen, 0.05, method="linear") == 0.0
