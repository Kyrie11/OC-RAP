from __future__ import annotations

import numpy as np
from ocrap.utils.datatypes import BEVSpec, EgoState, ActorState, MapFeatures


def rasterize_occlusion_simple(spec: BEVSpec, ego: EgoState, actors: list[ActorState], map_features: MapFeatures) -> np.ndarray:
    # MVP approximation: far cells behind dynamic objects become uncertain.
    rows, cols = np.meshgrid(np.arange(spec.H), np.arange(spec.W), indexing="ij")
    mask = np.zeros((spec.H, spec.W), dtype=np.float32)
    # Mark outer ring as occluded/unknown.
    margin = int(0.08 * min(spec.H, spec.W))
    mask[:margin, :] = 1.0
    mask[-margin:, :] = 1.0
    mask[:, :margin] = 1.0
    mask[:, -margin:] = 1.0
    return mask
