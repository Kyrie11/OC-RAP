from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schema import RecoveryOption


RECOVERY_MODES = [
    "stop",
    "brake_lane",
    "lateral_escape",
    "yield_rejoin",
    "pull_over",
    "mitigate_contact",
    "post_contact_stabilize",
    "avoid_secondary",
]


@dataclass(frozen=True)
class ModeSpec:
    mode: str
    param_names: tuple[str, ...]
    proposals: tuple[tuple[float, ...], ...]


MODE_SPECS: list[ModeSpec] = [
    ModeSpec("stop", ("a_dec", "s_stop"), ((-2.0, 8.0), (-3.5, 12.0), (-5.0, 18.0))),
    ModeSpec("brake_lane", ("a_dec", "T_brake"), ((-1.5, 2.0), (-3.0, 2.5), (-5.0, 3.0))),
    ModeSpec("lateral_escape", ("d_y", "v_tar", "T_lat"), ((-1.0, 4.0, 2.0), (-0.5, 5.0, 2.0), (0.5, 5.0, 2.0), (1.0, 4.0, 2.0))),
    ModeSpec("yield_rejoin", ("a_yield", "s_gap", "T_rejoin"), ((-1.5, 6.0, 3.0), (-1.5, 10.0, 4.0), (-3.0, 6.0, 3.0), (-3.0, 10.0, 4.0))),
    ModeSpec("pull_over", ("d_y", "s_shoulder", "v_stop"), ((-2.5, 6.0, 0.0), (2.5, 6.0, 0.0))),
    ModeSpec("mitigate_contact", ("a_dec", "delta_psi", "v_impact"), ((-3.0, -0.2, 2.0), (-3.0, 0.2, 2.0))),
    ModeSpec("post_contact_stabilize", ("k_psi", "k_r", "a_decay"), ((1.0, 1.5, -2.0), (1.5, 2.0, -3.0))),
    ModeSpec("avoid_secondary", ("d_y", "a_dec", "s_clear"), ((-1.0, -2.0, 4.0), (1.0, -2.0, 4.0), (-1.5, -4.0, 6.0), (1.5, -4.0, 6.0))),
]


def default_recovery_options(include_invalid_modes: bool = True, shoulder_available: bool = True, adjacent_available: bool = True) -> list[RecoveryOption]:
    opts: list[RecoveryOption] = []
    idx = 0
    for spec in MODE_SPECS:
        for proposal in spec.proposals:
            valid = True
            if spec.mode == "pull_over" and not shoulder_available:
                valid = include_invalid_modes and False
            if spec.mode in {"lateral_escape", "yield_rejoin", "avoid_secondary"} and not adjacent_available:
                valid = include_invalid_modes and False
            opts.append(RecoveryOption(idx, spec.mode, np.asarray(proposal, dtype=np.float32), bool(valid)))
            idx += 1
    return opts


def option_valid_mask(options: list[RecoveryOption]) -> np.ndarray:
    return np.asarray([o.valid for o in options], dtype=bool)


def option_modes(options: list[RecoveryOption]) -> list[str]:
    return [o.mode for o in options]


def options_to_param_matrix(options: list[RecoveryOption], width: int = 3) -> np.ndarray:
    out = np.zeros((len(options), width), dtype=np.float32)
    for i, opt in enumerate(options):
        p = np.asarray(opt.params, dtype=np.float32).reshape(-1)
        out[i, : min(width, p.size)] = p[:width]
    return out
