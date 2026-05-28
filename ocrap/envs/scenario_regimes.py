from __future__ import annotations

from typing import Dict
import numpy as np

REGIME_RATIOS = {
    "normal_high_headroom": 0.35,
    "low_headroom": 0.25,
    "near_contact": 0.25,
    "contact_post_contact": 0.15,
}


def sample_regime(rng: np.random.Generator, ratios: Dict[str, float] | None = None) -> str:
    ratios = ratios or REGIME_RATIOS
    keys = list(ratios.keys())
    p = np.array([ratios[k] for k in keys], dtype=float)
    p = p / p.sum()
    return str(rng.choice(keys, p=p))


def regime_to_env_config(regime: str, seed: int) -> dict:
    if regime == "normal_high_headroom":
        return {"traffic_density": 0.08, "map": "S", "start_seed": seed}
    if regime == "low_headroom":
        return {"traffic_density": 0.25, "map": "C", "start_seed": seed}
    if regime == "near_contact":
        return {"traffic_density": 0.30, "map": "X", "start_seed": seed}
    if regime == "contact_post_contact":
        return {"traffic_density": 0.35, "map": "X", "start_seed": seed, "crash_vehicle_done": False, "crash_object_done": False}
    raise ValueError(f"unknown regime {regime}")
