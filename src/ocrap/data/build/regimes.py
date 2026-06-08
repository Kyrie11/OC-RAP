from __future__ import annotations

import numpy as np

from ocrap.data.schema import DatasetSample, SceneHistory
from ocrap.simulation.observation.bev import unknown_ratio_in_corridor
from ocrap.utils.geometry import agent_state_to_box, compute_ttc, min_box_clearance


def assign_regimes(samples: list[DatasetSample], history: SceneHistory, cfg: dict) -> None:
    if not samples:
        return
    thresholds = cfg.get("regime_thresholds", {})
    tau_high = float(thresholds.get("tau_high", 1.0))
    tau_d = float(thresholds.get("tau_d", 2.0))
    tau_ttc = float(thresholds.get("tau_ttc", 3.0))
    tau_occ = float(thresholds.get("tau_occ", 0.05))
    nominal_dep = float(samples[0].r_dep_star)
    max_dep = max(float(s.r_dep_star) for s in samples)
    boxes = np.asarray([agent_state_to_box(a) for a in history.agent_history[-1, 1:]], dtype=np.float32) if history.agent_history.shape[1] > 1 else np.zeros((0, 9), dtype=np.float32)
    valids = history.agent_valid[-1, 1:] if history.agent_history.shape[1] > 1 else np.zeros((0,), dtype=bool)
    ego_box = agent_state_to_box(history.agent_history[-1, 0])
    dmin = min_box_clearance(ego_box, boxes, valids) if len(boxes) else 99.0
    ttc = compute_ttc(history.ego_state, boxes, valids) if len(boxes) else 99.0
    occ_ratio = unknown_ratio_in_corridor(history.occ_mask)
    for s in samples:
        near = bool(dmin < tau_d or ttc < tau_ttc or s.prefix.diagnostics.get("prefix_collision", False))
        post = bool(s.prefix.diagnostics.get("prefix_contact", False) or any(f.metadata.get("contact_surrogate", False) for f in s.futures))
        regimes = {
            "normal": bool(nominal_dep > tau_high and ttc > tau_ttc and dmin > tau_d and occ_ratio <= tau_occ),
            "low_headroom": bool(nominal_dep <= tau_high and max_dep > 0.0),
            "occluded": bool(occ_ratio > tau_occ),
            "near_contact": near,
            "post_contact": post,
            "oracle_artifact": bool(s.i_art_star),
        }
        s.regime_label.clear()
        s.regime_label.update(regimes)
