from __future__ import annotations

import numpy as np
from recap.utils.datatypes import RolloutTrace
from .margins import compute_margins


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def harm_from_trace(trace: RolloutTrace, H_p: int, dt: float = 0.2, T_h_guard: float = 0.4) -> tuple[float, int]:
    guard_steps = int(T_h_guard / dt)
    fc = trace.first_contact_idx
    if fc >= 0 and fc <= H_p + guard_steps:
        impact_norm = np.clip(trace.relative_speed_at_first_contact / 10.0, 0.0, 1.0)
        w = {"rear": 0.7, "side": 1.0, "front": 1.0, "pedestrian_or_cyclist": 1.5, "static": 0.8, "unknown": 1.0}.get(trace.contact_type, 1.0)
        return float(np.clip(impact_norm * w, 0.0, 1.0)), 1 if fc <= H_p else 2
    return 0.0, 0


def evidence_from_trace(trace: RolloutTrace, params: dict | None = None) -> dict:
    p = params or {}
    m = compute_margins(trace, p)
    tau_P = p.get("tau_P", 0.25)
    tau_G = p.get("tau_G", 0.25)
    tau_K = p.get("tau_K", 0.25)
    H_p = int(p.get("H_p", trace.stage_boundary_idx))
    H, H_source = harm_from_trace(trace, H_p, p.get("dt", 0.2), p.get("T_h_guard", 0.4))
    return {
        **m,
        "P_star": float(sigmoid(-m["M_path_rec"] / tau_P)),
        "P_raw_star": float(sigmoid(-m["M_path_raw"] / tau_P)),
        "G_star": float(sigmoid(m["M_return"] / tau_G)),
        "C_star": float(np.clip((m["M_ctrl"] + 1.0) / 2.0, 0.0, 1.0)),
        "H_star": float(H),
        "H_source": int(H_source),
        "K_star": float(0.0 if trace.first_contact_idx < 0 else sigmoid(-m["M_post"] / tau_K)),
    }


def scene_uncertainty_from_action(action_states: np.ndarray, occlusion_score: float = 0.0, local_traffic_density: float = 0.0) -> float:
    lateral = float(np.max(np.abs(action_states[:, 1]))) / 4.0
    speed = float(np.max(action_states[:, 3])) / 20.0
    conflict = np.clip(0.5 * lateral + 0.5 * speed, 0.0, 1.0)
    return float(np.clip(0.4 * occlusion_score + 0.3 * local_traffic_density + 0.3 * conflict, 0.0, 1.0))


def combine_uncertainty(U_scene: float, U_mode: float, U_interact: float = 0.0) -> float:
    return float(np.clip(0.45 * U_scene + 0.35 * U_mode + 0.20 * U_interact, 0.0, 1.0))
