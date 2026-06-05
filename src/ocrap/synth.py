from __future__ import annotations

import math
from typing import Iterator

import numpy as np

from .schema import RawScenario


def make_synthetic_scenario(idx: int, T: int = 91, A: int = 16, seed: int = 0) -> RawScenario:
    rng = np.random.default_rng(seed + idx)
    timestamps = np.arange(T, dtype=np.float32) * 0.1
    states = np.zeros((T, A, 10), dtype=np.float32)
    valid = np.zeros((T, A), dtype=bool)
    # Ego along x-axis.
    v0 = rng.uniform(5.0, 12.0)
    for t in range(T):
        x = v0 * timestamps[t]
        states[t, 0] = [x, 0.0, 0.0, v0, 0.0, 0.0, 4.8, 2.0, 1.5, 1.0]
        valid[t, 0] = True
    for a in range(1, A):
        start_x = rng.uniform(10.0, 80.0)
        lane_y = rng.choice([-7.0, -3.5, 3.5, 7.0, 0.0])
        speed = rng.uniform(0.0, 14.0)
        heading = 0.0 if rng.random() > 0.15 else math.pi / 2.0
        appear = rng.integers(0, 20)
        for t in range(appear, T):
            dt = timestamps[t] - timestamps[appear]
            states[t, a] = [start_x + speed * math.cos(heading) * dt, lane_y + speed * math.sin(heading) * dt, 0.0, speed * math.cos(heading), speed * math.sin(heading), heading, 4.8 if a % 5 else 0.8, 2.0 if a % 5 else 0.8, 1.5, 1.0 if a % 5 else 2.0]
            valid[t, a] = True
    P, Q = 8, 64
    polylines = np.zeros((P, Q, 12), dtype=np.float32)
    map_valid = np.zeros((P, Q), dtype=bool)
    xs = np.linspace(-20.0, 120.0, Q, dtype=np.float32)
    lane_ys = [-7.0, -3.5, 0.0, 3.5, 7.0]
    for i, y in enumerate(lane_ys):
        polylines[i, :, 0] = xs
        polylines[i, :, 1] = y
        polylines[i, :, 3] = np.gradient(xs)
        polylines[i, :, 5] = 1.0
        polylines[i, :, 6] = 13.4
        polylines[i, :, 8] = 1.0 if y == 0.0 else 0.0
        map_valid[i] = True
    route = np.stack([xs, np.zeros_like(xs), np.zeros_like(xs), np.ones_like(xs)], axis=1).astype(np.float32)
    dyn = np.zeros((T, 1, 8), dtype=np.float32)
    return RawScenario(f"synthetic_{idx:06d}", timestamps, 0, states, valid, polylines, map_valid, route, dyn, [str(i) for i in range(A)])


def iter_synthetic_scenarios(num_scenarios: int, seed: int = 0) -> Iterator[RawScenario]:
    for i in range(num_scenarios):
        yield make_synthetic_scenario(i, seed=seed)
