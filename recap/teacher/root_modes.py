from __future__ import annotations

from typing import List
import numpy as np
from recap.utils.datatypes import RootModeSeed
from recap.envs.traffic_policies import LatentTrafficContext

MODE_SLOT_SEMANTICS = [
    "nominal",
    "aggressive_traffic",
    "conservative_yielding",
    "delayed_ego_actuation",
    "low_friction",
    "occlusion_release_early",
    "lead_vehicle_hard_brake",
    "lateral_cutin_bias",
]


def generate_root_modes(root_seed: int, M: int = 8) -> List[RootModeSeed]:
    modes: List[RootModeSeed] = []
    for m in range(M):
        sem = MODE_SLOT_SEMANTICS[m % len(MODE_SLOT_SEMANTICS)]
        rng_seed = int(root_seed * 1009 + m * 9176 + 13)
        seed = RootModeSeed(mode_id=m, rng_seed=rng_seed, semantic=sem)
        if sem == "aggressive_traffic":
            seed.reaction_delay = 0.2; seed.aggressiveness = 0.8; seed.desired_speed_scale = 1.2
        elif sem == "conservative_yielding":
            seed.reaction_delay = 0.6; seed.aggressiveness = -0.5; seed.desired_speed_scale = 0.85
        elif sem == "delayed_ego_actuation":
            seed.actuation_delay = 0.3; seed.control_noise_std = 0.02
        elif sem == "low_friction":
            seed.friction_scale = 0.75
        elif sem == "occlusion_release_early":
            seed.occlusion_release_time = 0.8; seed.hidden_actor_spawn = True
        elif sem == "lead_vehicle_hard_brake":
            seed.braking_noise = -3.0
        elif sem == "lateral_cutin_bias":
            seed.lateral_noise = 1.0; seed.aggressiveness = 0.4
        modes.append(seed)
    return modes


def build_latent_context(root_id: str, mode_seed: RootModeSeed) -> LatentTrafficContext:
    return LatentTrafficContext(
        mode_seed.mode_id,
        mode_seed.semantic,
        mode_seed.reaction_delay,
        mode_seed.aggressiveness,
        mode_seed.desired_speed_scale,
        mode_seed.lateral_noise,
        mode_seed.braking_noise,
        mode_seed.rng_seed,
    )


def mode_seed_params_array(modes: List[RootModeSeed]) -> np.ndarray:
    return np.stack([m.to_vector() for m in modes]).astype(np.float32)


def normalized_mode_uncertainty(modes: List[RootModeSeed]) -> np.ndarray:
    vals = []
    for m in modes:
        mag = abs(m.aggressiveness) + abs(m.reaction_delay) / 1.0 + abs(1.0 - m.friction_scale) / 0.5 + abs(m.actuation_delay) / 0.5 + abs(m.lateral_noise) / 2.0 + (0.5 if m.hidden_actor_spawn else 0.0)
        vals.append(np.clip(mag / 4.0, 0.0, 1.0))
    return np.asarray(vals, dtype=np.float32)
