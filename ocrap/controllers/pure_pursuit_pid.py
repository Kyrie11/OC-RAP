from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from ocrap.utils.datatypes import EgoState
from .action_adapter import ActionAdapter


class TrackingController:
    def reset(self):
        pass

    def track(self, ego_state: EgoState, ref_states: np.ndarray, ref_controls: np.ndarray | None = None) -> np.ndarray:
        raise NotImplementedError


@dataclass
class PurePursuitPID(TrackingController):
    kappa_max: float = 0.25
    a_max: float = 4.0
    brake_max: float = 6.0
    kp: float = 0.8
    kd: float = 0.05
    adapter: ActionAdapter = ActionAdapter()

    def reset(self):
        return None

    def track(self, ego_state: EgoState, ref_states: np.ndarray, ref_controls: np.ndarray | None = None) -> np.ndarray:
        ref = np.asarray(ref_states, dtype=np.float32)
        if ref.ndim == 1:
            ref = ref[None, :]
        v = max(0.0, float(ego_state.v))
        lookahead = float(np.clip(0.5 * v + 2.0, 3.0, 8.0))
        pts = ref[:, :2]
        dists = np.linalg.norm(pts - np.array([ego_state.x, ego_state.y], dtype=np.float32), axis=-1)
        idxs = np.where(dists >= lookahead)[0]
        idx = int(idxs[0]) if len(idxs) else len(ref) - 1
        target = pts[idx]
        dx, dy = target[0] - ego_state.x, target[1] - ego_state.y
        alpha = np.arctan2(dy, dx) - ego_state.heading
        alpha = (alpha + np.pi) % (2 * np.pi) - np.pi
        curvature_cmd = 2.0 * np.sin(alpha) / max(lookahead, 1e-3)
        steer = float(np.clip(curvature_cmd / self.kappa_max, -1.0, 1.0))
        v_ref = float(ref[idx, 3]) if ref.shape[1] > 3 else v
        a_ref = float(ref_controls[min(idx, len(ref_controls) - 1), 0]) if ref_controls is not None and len(ref_controls) else 0.0
        a_cmd = self.kp * (v_ref - v) + self.kd * (a_ref - ego_state.a_long)
        if a_cmd >= 0:
            tb = np.clip(a_cmd / self.a_max, 0.0, 1.0)
        else:
            tb = np.clip(a_cmd / self.brake_max, -1.0, 0.0)
        return self.adapter.to_metadrive(steer, tb)
