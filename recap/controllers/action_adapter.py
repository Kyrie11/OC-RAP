from __future__ import annotations

import numpy as np


class ActionAdapter:
    """Centralized MetaDrive action convention.

    MetaDrive EnvInputPolicy continuous input is a 2-D Box in [-1, 1].  This
    adapter keeps sign conventions out of models and selectors.
    """

    def __init__(self, steer_left_positive: bool = True):
        self.steer_left_positive = steer_left_positive

    def to_metadrive(self, steering: float, throttle_brake: float) -> np.ndarray:
        steer = float(steering if self.steer_left_positive else -steering)
        return np.array([np.clip(steer, -1.0, 1.0), np.clip(throttle_brake, -1.0, 1.0)], dtype=np.float32)

    def from_metadrive(self, action) -> tuple[float, float]:
        arr = np.asarray(action, dtype=np.float32)
        steer = float(arr[0] if self.steer_left_positive else -arr[0])
        return steer, float(arr[1])
