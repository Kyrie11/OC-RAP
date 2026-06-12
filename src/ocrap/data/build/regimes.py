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
    # ``tau_occ`` is intentionally strict for the occluded-regime tag.  In WOMD
    # BEV crops, however, an unknown corridor ratio above 0.05 is common even for
    # otherwise benign, uniformly sampled frames.  Using the same threshold for
    # normal/NUP tagging makes ``normal`` permanently false on real Waymax/WOMD
    # smoke sets.  Treat occlusion as an overlapping attribute and use a separate,
    # much looser threshold for normal samples.
    tau_normal_occ = float(thresholds.get("tau_normal_occ", 0.75))
    tau_normal_dep = float(thresholds.get("tau_normal_dep", 0.50))
    tau_prefix_hard = float(thresholds.get("tau_prefix_hard", 0.0))
    tau_prefix_harm = float(thresholds.get("tau_prefix_harm", 0.05))
    require_uniform_for_normal = bool(thresholds.get("require_uniform_for_normal", False))
    nominal_dep = float(samples[0].r_dep_star)
    max_dep = max(float(s.r_dep_star) for s in samples)
    boxes = np.asarray([agent_state_to_box(a) for a in history.agent_history[-1, 1:]], dtype=np.float32) if history.agent_history.shape[1] > 1 else np.zeros((0, 9), dtype=np.float32)
    valids = history.agent_valid[-1, 1:] if history.agent_history.shape[1] > 1 else np.zeros((0,), dtype=bool)
    ego_box = agent_state_to_box(history.agent_history[-1, 0])
    dmin = min_box_clearance(ego_box, boxes, valids) if len(boxes) else 99.0
    ttc = compute_ttc(history.ego_state, boxes, valids) if len(boxes) else 99.0
    occ_ratio = unknown_ratio_in_corridor(history.occ_mask)
    time_reasons = set(str(x) for x in history.metadata.get("time_sampling_reasons", []) if x is not None)
    for s in samples:
        near = bool(dmin < tau_d or ttc < tau_ttc or s.prefix.diagnostics.get("prefix_collision", False))
        post = bool(s.prefix.diagnostics.get("prefix_contact", False) or any(f.metadata.get("contact_surrogate", False) for f in s.futures))
        prefix_safe = bool(
            s.prefix.feasible
            and float(s.prefix.hard_violation) <= tau_prefix_hard
            and float(s.prefix.harm_proxy) <= tau_prefix_harm
        )
        sample_headroom = bool(float(s.r_dep_star) > tau_normal_dep and float(s.r_orc_star) > tau_normal_dep)
        uniform_ok = bool((not require_uniform_for_normal) or ("uniform" in time_reasons))
        normal = bool(
            (not s.i_art_star)
            and sample_headroom
            and prefix_safe
            and not near
            and not post
            and occ_ratio <= tau_normal_occ
            and uniform_ok
        )
        regimes = {
            "normal": normal,
            "low_headroom": bool(nominal_dep <= tau_high and max_dep > 0.0),
            "occluded": bool(occ_ratio > tau_occ),
            "near_contact": near,
            "post_contact": post,
            "oracle_artifact": bool(s.i_art_star),
        }
        s.regime_label.clear()
        s.regime_label.update(regimes)
