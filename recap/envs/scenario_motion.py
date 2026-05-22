from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from recap.utils.datatypes import EgoState


def local_states_to_world(root_ego: EgoState, states: np.ndarray) -> np.ndarray:
    st = np.asarray(states, dtype=np.float32).copy()
    c = math.cos(root_ego.heading)
    s = math.sin(root_ego.heading)
    x = st[:, 0].copy()
    y = st[:, 1].copy()
    st[:, 0] = root_ego.x + c * x - s * y
    st[:, 1] = root_ego.y + s * x + c * y
    st[:, 2] = st[:, 2] + root_ego.heading
    return st


def world_states_to_local(root_ego: EgoState, states: np.ndarray) -> np.ndarray:
    st = np.asarray(states, dtype=np.float32).copy()
    c = math.cos(root_ego.heading)
    s = math.sin(root_ego.heading)
    dx = st[:, 0].copy() - root_ego.x
    dy = st[:, 1].copy() - root_ego.y
    st[:, 0] = c * dx + s * dy
    st[:, 1] = -s * dx + c * dy
    st[:, 2] = (st[:, 2] - root_ego.heading + np.pi) % (2 * np.pi) - np.pi
    return st
