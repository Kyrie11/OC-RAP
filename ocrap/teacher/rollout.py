from __future__ import annotations

import numpy as np
from ocrap.utils.datatypes import ActionPrefix, RecoveryOption, RootModeSeed, RolloutTrace


def synthetic_rollout(action: ActionPrefix, option: RecoveryOption, mode: RootModeSeed, H_p: int = 10, H_r: int = 25, dt: float = 0.2, regime: str = "normal_high_headroom") -> RolloutTrace:
    """Portable closed-loop teacher approximation for smoke tests.

    Real teacher generation should restore MetaDrive and track the prefix then
    option.  This fallback preserves the same tensor semantics and label logic.
    """
    states = np.concatenate([action.states, option.states_ref[1:]], axis=0).astype(np.float32)
    controls = np.concatenate([action.controls[:, :2], option.controls_ref[:, :2]], axis=0).astype(np.float32)
    T = states.shape[0]
    # Normalized margins: positive is feasible.  Make failures depend on lateral/speed/mode.
    lateral = np.abs(states[:, 1])
    speed = states[:, 3]
    collision_margin = np.ones(T, dtype=np.float32) * 0.7
    if regime in ("near_contact", "contact_post_contact"):
        pressure = 1.0 - np.clip((lateral + (mode.aggressiveness + 1.0) * 0.5) / 3.5, 0.0, 2.0)
        collision_margin[: H_p + 3] = np.minimum(collision_margin[: H_p + 3], pressure[: H_p + 3])
    first_contact_idx = -1
    if regime == "contact_post_contact" and (action.states[-1, 3] > 6.0 or mode.semantic in ("aggressive_traffic", "lead_vehicle_hard_brake")):
        first_contact_idx = min(H_p + (0 if mode.semantic == "aggressive_traffic" else 3), T - 1)
        collision_margin[first_contact_idx] = -1.0
    drivable = 1.0 - np.maximum(0.0, lateral - 4.0) / 2.0
    direction = 1.0 - np.abs(states[:, 2]) / (np.pi / 3)
    route = 1.0 - lateral / 4.0
    speed_margin = 1.0 - np.maximum(0.0, speed - 15.9) / 5.0
    lat_acc = speed**2 * np.abs(states[:, 5])
    stability = np.minimum(1.0 - lat_acc / 4.0, 1.0 - 0.2 * np.abs(states[:, 2]))
    ttc = np.ones(T, dtype=np.float32)
    # Affordance cost arrays; lower is better.
    costs = {
        "stop": np.abs(states[:, 3]) / 10.0 + np.abs(states[:, 1]) / 4.0,
        "lane": np.abs(states[:, 1]) / 2.0 + np.abs(states[:, 2]) / np.pi,
        "route": np.abs(states[:, 1]) / 4.0 + np.maximum(0.0, -states[:, 0]) / 20.0,
        "escape": np.maximum(0.0, 1.5 - np.abs(states[:, 1])) / 1.5,
        "stabilize": np.abs(states[:, 3]) / 10.0 + np.abs(states[:, 2]) / np.pi + np.maximum(0.0, np.abs(states[:, 1]) - 4.0) / 4.0,
    }
    secondary = -1
    if first_contact_idx >= 0 and np.any(np.abs(states[first_contact_idx + 3 :, 1]) > 5.5):
        secondary = first_contact_idx + 3
    stage = 0 if first_contact_idx < 0 else (1 if first_contact_idx <= H_p else 2)
    return RolloutTrace(
        ego_states=states,
        ego_controls=controls,
        stage_boundary_idx=H_p,
        first_contact_idx=int(first_contact_idx),
        first_contact_stage=stage,
        secondary_collision_idx=int(secondary),
        contact_type="front" if first_contact_idx >= 0 else "none",
        relative_speed_at_first_contact=float(speed[first_contact_idx] if first_contact_idx >= 0 else 0.0),
        collision_margin=collision_margin,
        drivable_margin=drivable.astype(np.float32),
        direction_margin=direction.astype(np.float32),
        route_margin=route.astype(np.float32),
        speed_margin=speed_margin.astype(np.float32),
        stability_margin=stability.astype(np.float32),
        ttc_margin=ttc.astype(np.float32),
        affordance_costs=costs,
    )
