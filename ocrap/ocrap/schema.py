from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass
class RawScenario:
    scenario_id: str
    timestamps: np.ndarray  # [T]
    sdc_track_index: int
    agent_states: np.ndarray  # [T, A, F], x,y,z,vx,vy,heading,l,w,h,type
    agent_valid: np.ndarray  # [T, A]
    map_polylines: np.ndarray  # [P, Q, F_map]
    map_valid: np.ndarray  # [P, Q]
    route: np.ndarray  # [R, F_route]
    dynamic_map: np.ndarray  # [T, B, F_signal]
    object_ids: list[str] = field(default_factory=list)


@dataclass
class SceneHistory:
    scene_id: str
    time_index: int
    agent_history: np.ndarray  # [T_h, A, F_agent]
    agent_valid: np.ndarray  # [T_h, A]
    map_polylines: np.ndarray  # [P, Q, F_map]
    map_valid: np.ndarray  # [P, Q]
    dynamic_map: np.ndarray  # [T_h, B, F_signal]
    route: np.ndarray  # [R, F_route]
    occ_mask: np.ndarray  # [C, H, W]
    ego_state: np.ndarray  # [F_ego]
    future_agent_states: np.ndarray  # [T_future, A, F_agent]
    future_agent_valid: np.ndarray  # [T_future, A]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidatePrefix:
    macro_id: int
    macro_name: str
    params: np.ndarray
    prefix_states: np.ndarray  # [T_p, F_ego]
    prefix_controls: np.ndarray  # [T_p-1, F_ctrl]
    utility: float
    feasible: bool
    hard_violation: float
    harm_proxy: float
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class CounterfactualFuture:
    future_id: int
    source: str
    prior: float
    agent_states: np.ndarray  # [T_total, A, F_agent]
    agent_valid: np.ndarray  # [T_total, A]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryOption:
    option_id: int
    mode: str
    params: np.ndarray
    valid: bool = True



def _pad_recovery_params(options: list[RecoveryOption], width: int = 3) -> np.ndarray:
    out = np.zeros((len(options), width), dtype=np.float32)
    for i, g in enumerate(options):
        p = np.asarray(g.params, dtype=np.float32).reshape(-1)
        out[i, : min(width, p.size)] = p[:width]
    return out

@dataclass
class Observation:
    ego_state: np.ndarray
    boxes: np.ndarray  # [N, F_box], x,y,vx,vy,heading,l,w,h,type
    box_valid: np.ndarray  # [N]
    occ_mask: np.ndarray  # [C, H, W]
    contact_flag: bool
    stability_proxy: np.ndarray


@dataclass
class DatasetSample:
    scene_id: str
    time_index: int
    candidate_index: int
    split_id: str
    is_nominal: bool
    h_t: SceneHistory
    prefix: CandidatePrefix
    futures: list[CounterfactualFuture]
    future_probs: np.ndarray  # [J]
    root_assignments: np.ndarray  # [J]
    root_probs: np.ndarray  # [K]
    root_signature: np.ndarray  # [K, D_sig]
    root_future_signature: np.ndarray  # [K, D_future_sig]
    root_valid: np.ndarray  # [K]
    future_to_root_weight: np.ndarray  # [J, K]
    observations: list[Observation]
    y_obs: np.ndarray  # [K, K]
    c_star: np.ndarray  # [K, K]
    recovery_options: list[RecoveryOption]
    m_star: np.ndarray  # [K, L]
    option_valid: np.ndarray  # [L]
    r_orc_star: float
    r_dep_star: float
    oracle_gap_star: float
    i_art_star: bool
    regime_label: dict[str, bool]
    valid_masks: dict[str, Any]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_npz_dict(self) -> dict[str, Any]:
        h = self.h_t
        p = self.prefix
        return {
            "scene_id": self.scene_id,
            "time_index": np.int64(self.time_index),
            "candidate_index": np.int64(self.candidate_index),
            "split_id": self.split_id,
            "is_nominal": np.int64(self.is_nominal),
            "agent_history": h.agent_history.astype(np.float32),
            "agent_valid": h.agent_valid.astype(np.float32),
            "map_polylines": h.map_polylines.astype(np.float32),
            "map_valid": h.map_valid.astype(np.float32),
            "dynamic_map": h.dynamic_map.astype(np.float32),
            "route": h.route.astype(np.float32),
            "bev_occ": h.occ_mask.astype(np.float32),
            "ego_state": h.ego_state.astype(np.float32),
            "prefix_states": p.prefix_states.astype(np.float32),
            "prefix_controls": p.prefix_controls.astype(np.float32),
            "prefix_macro_id": np.int64(p.macro_id),
            "prefix_param": p.params.astype(np.float32),
            "utility": np.float32(p.utility),
            "hard_violation": np.float32(p.hard_violation),
            "harm_proxy": np.float32(p.harm_proxy),
            "feasible": np.int64(p.feasible),
            "future_probs": self.future_probs.astype(np.float32),
            "root_assignments": self.root_assignments.astype(np.int64),
            "root_probs": self.root_probs.astype(np.float32),
            "root_signature": self.root_signature.astype(np.float32),
            "root_future_signature": self.root_future_signature.astype(np.float32),
            "root_valid": self.root_valid.astype(np.float32),
            "future_to_root_weight": self.future_to_root_weight.astype(np.float32),
            "y_obs": self.y_obs.astype(np.float32),
            "c_star": self.c_star.astype(np.float32),
            "m_star": self.m_star.astype(np.float32),
            "option_valid": self.option_valid.astype(np.float32),
            "r_orc_star": np.float32(self.r_orc_star),
            "r_dep_star": np.float32(self.r_dep_star),
            "oracle_gap_star": np.float32(self.oracle_gap_star),
            "i_art_star": np.int64(self.i_art_star),
            "regime_label": self.regime_label,
            "valid_masks": self.valid_masks,
            "diagnostics": self.diagnostics,
            "macro_name": p.macro_name,
            "prefix_diagnostics": p.diagnostics,
            "future_sources": [f.source for f in self.futures],
            "future_metadata": [f.metadata for f in self.futures],
            "recovery_modes": [g.mode for g in self.recovery_options],
            "recovery_params": _pad_recovery_params(self.recovery_options).astype(np.float32),
        }


def dataclass_to_dict(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    return obj
