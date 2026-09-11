from __future__ import annotations

"""Observation-consistent executable constraint-Jacobian audit.

This module implements the preregistered post-HCNC branch.  V48.112 showed
that a prefix-level heterogeneous constraint cone is not a population-shared
Support/Reserve orientation and, critically, that persistent re-entry is almost
absent from the first-eight-state prefix audit.  The next scientific object is
therefore the *same-option executable recovery response*:

    J_l,c(t; a) ~= h_c(Phi(a, g_l), t) - h_c(Phi(a0, g_l), t)

where Phi is the deterministic actuator-projected recovery controller, g_l is a
fixed recovery option, and h_c is one of the observation-only signed physical
constraints {clearance, stopping, route, persistent re-entry}.  This is a
finite-difference directional Jacobian analogue, not an infinitesimal learned
Jacobian.

The audit keeps one common option identity across candidate and nominal in each
response.  It compares two equal-capacity readouts:

  nominal_option  -- choose the max-min executable option under the nominal
                     prefix and evaluate the candidate-minus-nominal response
                     for that same option;
  candidate_option -- choose the max-min executable option under the candidate
                      prefix and evaluate the same-option response.

Both use all four signed constraints at eight fixed knots spanning the already
configured recovery horizon, with two channels (delta and delta * nominal).
Thus both append 4 x 8 x 2 = 64 dimensions to the unchanged 156-D candidate
response, yielding the same 220-D linear function class used by V48.112.

No teacher future, hidden root, regime id, learned selector, source update,
boundary transport, or new horizon/threshold is used.
"""

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from ocrap.audits.constraint_native_orientation import RAW_CANDIDATE_DIM
from ocrap.audits.heterogeneous_constraint_normal_cone import (
    CONSTRAINT_NAMES,
    NUM_CONSTRAINTS,
    ConstraintConeConfig,
    cone_config_from_mapping,
)
from ocrap.data.schema import CandidatePrefix, RecoveryOption
from ocrap.simulation.teacher.controllers import rollout_recovery_controller

ENGINEERING_VERSION = "v48.113.0-OC-ECJ"
SCIENTIFIC_VERSION = "v48.113-OC-ECJ"
ALGORITHM_NAME = "Observation-Consistent Executable Constraint Jacobian Audit"

RECOVERY_KNOTS = 8
JACOBIAN_CHANNELS = 2
JACOBIAN_GEOMETRY_DIM = RECOVERY_KNOTS * NUM_CONSTRAINTS * JACOBIAN_CHANNELS
MATCHED_DIM = RAW_CANDIDATE_DIM + JACOBIAN_GEOMETRY_DIM


@dataclass(frozen=True)
class ExecutableConstraintField:
    values: np.ndarray          # [L, 8, 4], normalized signed constraints
    masks: np.ndarray           # [L, 8, 4], semantic activity
    option_valid: np.ndarray    # [L]
    option_scores: np.ndarray   # [L], full-horizon max-min input scores
    option_modes: tuple[str, ...]
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class JacobianScaler:
    u_scale: np.ndarray


def _decode_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    arr = np.asarray(value)
    if arr.shape == ():
        v = arr.item()
        if isinstance(v, bytes):
            return v.decode("utf-8", errors="replace")
        return str(v)
    return str(value)


def _scalar(d: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(np.asarray(d.get(key, default)).reshape(-1)[0])
    except Exception:
        return float(default)


def prefix_from_sample(d: Mapping[str, Any]) -> CandidatePrefix:
    states = np.asarray(d.get("prefix_states", np.zeros((0, 9))), dtype=np.float32)
    controls = np.asarray(d.get("prefix_controls", np.zeros((0, 4))), dtype=np.float32)
    if states.ndim != 2 or states.shape[0] < 1 or states.shape[1] < 9:
        raise ValueError(f"ECJ requires prefix_states[T,>=9], got {states.shape}")
    if controls.ndim != 2:
        controls = np.zeros((0, 4), dtype=np.float32)
    if controls.shape[0] and controls.shape[1] < 4:
        raise ValueError(f"ECJ requires prefix_controls[T,>=4], got {controls.shape}")
    macro_id = int(_scalar(d, "prefix_macro_id", 0.0))
    return CandidatePrefix(
        macro_id=macro_id,
        macro_name=_decode_text(d.get("prefix_macro_name", "candidate")),
        params=np.asarray(d.get("prefix_param", np.zeros((0,), dtype=np.float32)), dtype=np.float32).reshape(-1),
        prefix_states=states,
        prefix_controls=controls,
        utility=_scalar(d, "utility", 0.0),
        feasible=_scalar(d, "feasible", 1.0) > 0.5,
        hard_violation=_scalar(d, "hard_violation", 0.0),
        harm_proxy=_scalar(d, "harm_proxy", 0.0),
    )


def recovery_options_from_sample(d: Mapping[str, Any]) -> list[RecoveryOption]:
    modes = np.asarray(d.get("recovery_modes", []), dtype=object).reshape(-1)
    params = np.asarray(d.get("recovery_params", np.zeros((0, 3))), dtype=np.float32)
    valid = np.asarray(d.get("option_valid", np.ones((len(modes),), dtype=bool)), dtype=bool).reshape(-1)
    if params.ndim != 2:
        raise ValueError(f"ECJ requires recovery_params[L,P], got {params.shape}")
    L = max(len(modes), params.shape[0], valid.size)
    if L <= 0:
        raise ValueError("ECJ requires a non-empty recovery option library")
    if len(modes) != L or params.shape[0] != L or valid.size != L:
        raise ValueError(
            f"ECJ recovery option geometry mismatch modes={len(modes)} params={params.shape} valid={valid.size}"
        )
    return [
        RecoveryOption(i, _decode_text(modes[i]), np.asarray(params[i], dtype=np.float32).reshape(-1), bool(valid[i]))
        for i in range(L)
    ]


def _option_library_signature(options: list[RecoveryOption]) -> tuple[Any, ...]:
    return tuple(
        (o.mode, bool(o.valid), tuple(float(x) for x in np.asarray(o.params, dtype=np.float64).reshape(-1)))
        for o in options
    )


def validate_group_contract(nominal: Mapping[str, Any], candidate: Mapping[str, Any]) -> None:
    """Fail closed unless candidate and nominal share the deployable observation/library."""
    for key in ("scene_id", "time_index"):
        a = np.asarray(nominal.get(key, "")).reshape(-1)
        b = np.asarray(candidate.get(key, "")).reshape(-1)
        if a.size != b.size or any(str(x) != str(y) for x, y in zip(a.tolist(), b.tolist())):
            raise ValueError(f"ECJ group mismatch for {key}")

    for key in ("agent_history", "agent_valid", "ego_state"):
        a = np.asarray(nominal.get(key))
        b = np.asarray(candidate.get(key))
        if a.shape != b.shape:
            raise ValueError(f"ECJ observation shape mismatch for {key}: {a.shape} vs {b.shape}")
        if key == "agent_valid":
            same = np.array_equal(a.astype(bool), b.astype(bool))
        else:
            same = np.allclose(a.astype(np.float64), b.astype(np.float64), rtol=0.0, atol=1.0e-6, equal_nan=True)
        if not same:
            raise ValueError(f"ECJ observation changed across candidates for {key}")

    if _option_library_signature(recovery_options_from_sample(nominal)) != _option_library_signature(
        recovery_options_from_sample(candidate)
    ):
        raise ValueError("ECJ recovery option library changed across candidates")


def _observed_agents(d: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    hist = np.asarray(d.get("agent_history", np.zeros((0, 0, 16))), dtype=np.float64)
    valid = np.asarray(d.get("agent_valid", np.zeros((0, 0))), dtype=bool)
    ego = np.asarray(d.get("ego_state", np.zeros((9,))), dtype=np.float64).reshape(-1)
    if ego.size < 9:
        raise ValueError(f"ECJ requires ego_state[>=9], got {ego.shape}")
    ego_xy = ego[:2]
    ego_rad = 0.5 * float(np.hypot(max(abs(float(ego[7])), 1.0e-3), max(abs(float(ego[8])), 1.0e-3)))
    rows: list[np.ndarray] = []
    if hist.ndim == 3 and hist.shape[0] and hist.shape[1] > 1 and valid.ndim >= 2 and valid.shape[0]:
        last = hist[-1]
        vm = valid[-1].reshape(-1)
        for j in range(1, min(last.shape[0], vm.size)):
            if not bool(vm[j]):
                continue
            row = np.asarray(last[j], dtype=np.float64).reshape(-1)
            if row.size < 12 or not np.isfinite(row[:12]).all():
                continue
            rows.append(row)
    if not rows:
        return (
            np.zeros((0, 2), dtype=np.float64),
            np.zeros((0, 2), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
            ego_xy.astype(np.float64),
            ego_rad,
        )
    arr = np.stack(rows, axis=0)
    rel0 = arr[:, :2] - ego_xy[None, :]
    vel = arr[:, 3:5]
    arad = 0.5 * np.hypot(
        np.maximum(np.abs(arr[:, 10]), 1.0e-3),
        np.maximum(np.abs(arr[:, 11]), 1.0e-3),
    )
    return rel0, vel, arad, ego_xy.astype(np.float64), ego_rad


def _raw_clearance(
    states: np.ndarray,
    times: np.ndarray,
    *,
    rel0: np.ndarray,
    vel: np.ndarray,
    arad: np.ndarray,
    ego_xy: np.ndarray,
    ego_rad: float,
) -> np.ndarray:
    if rel0.shape[0] == 0:
        return np.full((len(states),), np.inf, dtype=np.float64)
    ego_rel = np.asarray(states[:, :2], dtype=np.float64) - ego_xy[None, :]
    agents = rel0[None, :, :] + times[:, None, None] * vel[None, :, :]
    delta = ego_rel[:, None, :] - agents
    clear = np.linalg.norm(delta, axis=-1) - float(ego_rad) - arad[None, :]
    return np.min(clear, axis=1)


def _remaining_path_distance_to_conflict(xy: np.ndarray, safety_reserve: np.ndarray, default_available: float) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64)
    reserve = np.asarray(safety_reserve, dtype=np.float64)
    n = len(xy)
    out = np.full((n,), float(default_available), dtype=np.float64)
    if n <= 0:
        return out
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=-1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    for q in range(n):
        hit = np.flatnonzero(reserve[q:] <= 0.0)
        if hit.size:
            u = q + int(hit[0])
            out[q] = max(0.0, float(cum[u] - cum[q]))
    return out


def _prefix_contact(
    prefix: CandidatePrefix,
    *,
    sample_rate_hz: float,
    rel0: np.ndarray,
    vel: np.ndarray,
    arad: np.ndarray,
    ego_xy: np.ndarray,
    ego_rad: float,
) -> bool:
    if rel0.shape[0] == 0:
        return False
    dt = 1.0 / sample_rate_hz
    current_states = np.asarray(prefix.prefix_states, dtype=np.float64)
    times = (np.arange(len(current_states), dtype=np.float64) + 1.0) * dt
    return bool(np.any(_raw_clearance(
        current_states, times, rel0=rel0, vel=vel, arad=arad, ego_xy=ego_xy, ego_rad=ego_rad
    ) <= 0.0))


def executable_constraint_field_from_sample(
    d: Mapping[str, Any],
    cfg: Mapping[str, Any],
    *,
    num_options: int | None = None,
) -> ExecutableConstraintField:
    """Build observation-only signed constraints for actuator-projected recovery options."""
    prefix = prefix_from_sample(d)
    options = recovery_options_from_sample(d)
    if num_options is not None and len(options) != int(num_options):
        raise ValueError(f"ECJ option count mismatch sample={len(options)} expected={num_options}")

    sample_rate = float(cfg.get("sample_rate_hz", 10.0) or 10.0)
    recovery_horizon_s = float(cfg.get("recovery_horizon_s", 4.0) or 4.0)
    if not np.isfinite(sample_rate) or sample_rate <= 0.0:
        raise ValueError("ECJ invalid sample_rate_hz")
    if not np.isfinite(recovery_horizon_s) or recovery_horizon_s <= 0.0:
        raise ValueError("ECJ invalid recovery_horizon_s")
    horizon_steps = max(2, int(round(sample_rate * recovery_horizon_s)))
    if horizon_steps < RECOVERY_KNOTS:
        raise ValueError(
            f"ECJ recovery horizon has {horizon_steps} states, fewer than fixed {RECOVERY_KNOTS} knots"
        )
    knot_idx = np.rint(np.linspace(0, horizon_steps - 1, RECOVERY_KNOTS)).astype(np.int64)
    if len(np.unique(knot_idx)) != RECOVERY_KNOTS:
        raise ValueError("ECJ fixed recovery knots are not unique")

    cone_cfg: ConstraintConeConfig = cone_config_from_mapping(cfg)
    rel0, vel, arad, ego_xy, ego_rad = _observed_agents(d)
    has_agents = rel0.shape[0] > 0
    observed_raw_clear = (
        float(np.min(np.linalg.norm(rel0, axis=1) - ego_rad - arad)) if has_agents else np.inf
    )
    prefix_had_contact = bool(observed_raw_clear <= 0.0) or _prefix_contact(
        prefix,
        sample_rate_hz=sample_rate,
        rel0=rel0,
        vel=vel,
        arad=arad,
        ego_xy=ego_xy,
        ego_rad=ego_rad,
    )

    L = len(options)
    values = np.zeros((L, RECOVERY_KNOTS, NUM_CONSTRAINTS), dtype=np.float64)
    masks = np.zeros_like(values, dtype=bool)
    scores = np.full((L,), -np.inf, dtype=np.float64)
    valid = np.asarray([bool(o.valid) for o in options], dtype=bool)
    prefix_duration = float(len(prefix.prefix_states)) / sample_rate
    dt = 1.0 / sample_rate
    projected_controls = True
    reentry_active_options = 0
    reentry_active_knots = 0
    full_horizon_active_type_counts = {name: 0 for name in CONSTRAINT_NAMES}

    for l, option in enumerate(options):
        if not valid[l]:
            continue
        rec_states, rec_controls, diag = rollout_recovery_controller(
            prefix, option, horizon_steps, dict(cfg), project_control_envelope=True
        )
        if not bool(diag.get("projected_control_envelope", False)):
            raise ValueError("ECJ expected actuator-projected recovery controller")
        rec_states = np.asarray(rec_states, dtype=np.float64)
        rec_controls = np.asarray(rec_controls, dtype=np.float64)
        if rec_states.shape[0] != horizon_steps or rec_states.ndim != 2 or rec_states.shape[1] < 9:
            raise ValueError(f"ECJ invalid recovery rollout for option={l}: {rec_states.shape}")
        if rec_controls.ndim != 2:
            raise ValueError(f"ECJ invalid recovery controls for option={l}: {rec_controls.shape}")

        times = prefix_duration + np.arange(horizon_steps, dtype=np.float64) * dt
        raw_clear = _raw_clearance(
            rec_states, times, rel0=rel0, vel=vel, arad=arad, ego_xy=ego_xy, ego_rad=ego_rad
        )
        speed = np.maximum(np.abs(rec_states[:, 6]), 0.0)
        d_safe = float(cone_cfg.d_safe0_m) + float(cone_cfg.safe_time_headway_s) * speed
        safety_reserve = raw_clear - d_safe if has_agents else np.full((horizon_steps,), np.inf, dtype=np.float64)

        full_v = np.zeros((horizon_steps, NUM_CONSTRAINTS), dtype=np.float64)
        full_m = np.zeros_like(full_v, dtype=bool)

        # clearance
        if has_agents:
            full_v[:, 0] = safety_reserve / cone_cfg.distance_scale
            full_m[:, 0] = True

        # stopping: available executable path before the first safety-boundary
        # conflict minus stopping distance.  Uses the same fixed physical scales
        # as V48.112, no new threshold.
        safety_for_conflict = safety_reserve if has_agents else np.full((horizon_steps,), np.inf)
        available = _remaining_path_distance_to_conflict(
            rec_states[:, :2], safety_for_conflict, cone_cfg.default_available_distance_m
        )
        decel = max(abs(float(cone_cfg.a_min_mps2)), 1.0e-6)
        stop_required = speed * speed / (2.0 * decel)
        full_v[:, 1] = (available - stop_required) / cone_cfg.stop_scale
        full_m[:, 1] = True

        # route: same local route-corridor signed reserve used by V48.112.
        full_v[:, 2] = (float(cone_cfg.route_dev_max_m) - np.abs(rec_states[:, 1])) / cone_cfg.route_scale
        full_m[:, 2] = True

        # persistent re-entry: inactive until physical contact is observed,
        # inherited from the candidate prefix, or caused by the recovery itself.
        # After activation, a positive state is credited only if the entire
        # remaining suffix stays nonnegative.
        reentry = np.zeros((horizon_steps,), dtype=np.float64)
        reentry_mask = np.zeros((horizon_steps,), dtype=bool)
        seen_contact = bool(prefix_had_contact)
        suffix_min = np.minimum.accumulate(safety_reserve[::-1])[::-1] if has_agents else np.zeros((horizon_steps,))
        for q in range(horizon_steps):
            if has_agents and raw_clear[q] <= 0.0:
                seen_contact = True
            if not seen_contact:
                continue
            if has_agents:
                reentry[q] = (
                    safety_reserve[q] if safety_reserve[q] < 0.0 else suffix_min[q]
                ) / cone_cfg.distance_scale
            else:
                reentry[q] = 0.0
            reentry_mask[q] = True
        full_v[:, 3] = reentry
        full_m[:, 3] = reentry_mask
        if np.any(reentry_mask):
            reentry_active_options += 1
            reentry_active_knots += int(np.count_nonzero(reentry_mask[knot_idx]))

        if np.any(~np.any(full_m, axis=1)):
            raise ValueError(f"ECJ option={l} has recovery time without an active constraint")
        z = np.where(full_m, full_v, np.inf)
        active = np.argmin(z, axis=1)
        for ci, name in enumerate(CONSTRAINT_NAMES):
            full_horizon_active_type_counts[name] += int(np.count_nonzero(active == ci))
        scores[l] = float(np.min(z))
        values[l] = full_v[knot_idx]
        masks[l] = full_m[knot_idx]

    if not np.any(valid):
        raise ValueError("ECJ has no valid recovery options")
    if not np.isfinite(scores[valid]).all():
        raise ValueError("ECJ non-finite valid-option max-min score")
    if not np.isfinite(values).all():
        raise ValueError("ECJ non-finite constraint values")

    diag = {
        "valid_option_count": int(np.count_nonzero(valid)),
        "option_count": int(L),
        "recovery_horizon_s": float(recovery_horizon_s),
        "sample_rate_hz": float(sample_rate),
        "horizon_steps": int(horizon_steps),
        "recovery_knots": knot_idx.tolist(),
        "projected_control_envelope": bool(projected_controls),
        "observed_or_prefix_contact": bool(prefix_had_contact),
        "reentry_active_options": int(reentry_active_options),
        "reentry_active_knot_count": int(reentry_active_knots),
        "full_horizon_active_type_counts": full_horizon_active_type_counts,
    }
    return ExecutableConstraintField(
        values=values,
        masks=masks,
        option_valid=valid,
        option_scores=scores,
        option_modes=tuple(o.mode for o in options),
        diagnostics=diag,
    )


def best_option_index(field: ExecutableConstraintField) -> int:
    score = np.where(field.option_valid, field.option_scores, -np.inf)
    if not np.isfinite(score).any():
        raise ValueError("ECJ cannot select an executable recovery option")
    # np.argmax provides deterministic lowest-index tie breaking.
    return int(np.argmax(score))


def jacobian_geometry(
    candidate: ExecutableConstraintField,
    nominal: ExecutableConstraintField,
    option_index: int,
) -> np.ndarray:
    l = int(option_index)
    if l < 0 or l >= len(candidate.option_valid) or not candidate.option_valid[l] or not nominal.option_valid[l]:
        raise ValueError(f"ECJ selected invalid option {l}")
    if candidate.values.shape != nominal.values.shape or candidate.masks.shape != nominal.masks.shape:
        raise ValueError("ECJ candidate/nominal field shape mismatch")
    hc = candidate.values[l]
    h0 = nominal.values[l]
    active = candidate.masks[l] | nominal.masks[l]
    delta = hc - h0
    geom = np.zeros((RECOVERY_KNOTS, NUM_CONSTRAINTS, JACOBIAN_CHANNELS), dtype=np.float64)
    geom[:, :, 0] = np.where(active, delta, 0.0)
    geom[:, :, 1] = np.where(active, delta * h0, 0.0)
    out = geom.reshape(-1)
    if out.size != JACOBIAN_GEOMETRY_DIM:
        raise ValueError(f"ECJ geometry dim mismatch {out.size}")
    return out


def fit_jacobian_scaler(u: np.ndarray) -> JacobianScaler:
    x = np.asarray(u, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != RAW_CANDIDATE_DIM:
        raise ValueError(f"ECJ candidate response dimension mismatch {x.shape}")
    scale = np.sqrt(np.mean(x * x, axis=0, keepdims=True))
    scale = np.where(scale > 1.0e-8, scale, 1.0)
    return JacobianScaler(u_scale=scale)


def base_features(u: np.ndarray, scaler: JacobianScaler) -> np.ndarray:
    return np.asarray(u, dtype=np.float64) / scaler.u_scale


def matched_features(u: np.ndarray, geometry: np.ndarray, scaler: JacobianScaler) -> np.ndarray:
    x = np.asarray(u, dtype=np.float64)
    g = np.asarray(geometry, dtype=np.float64)
    if g.ndim != 2 or g.shape[1] != JACOBIAN_GEOMETRY_DIM or len(g) != len(x):
        raise ValueError(f"ECJ geometry batch mismatch u={x.shape} geometry={g.shape}")
    out = np.concatenate([base_features(x, scaler), g], axis=1)
    if out.shape[1] != MATCHED_DIM:
        raise ValueError(f"ECJ matched dim mismatch {out.shape[1]} != {MATCHED_DIM}")
    return out


def pair_diagnostics(candidate: ExecutableConstraintField, nominal: ExecutableConstraintField) -> dict[str, Any]:
    lc = best_option_index(candidate)
    ln = best_option_index(nominal)
    cvals = candidate.values[lc]
    cmask = candidate.masks[lc]
    z = np.where(cmask, cvals, np.inf)
    active = np.argmin(z, axis=1)
    active_counts = {CONSTRAINT_NAMES[i]: int(np.count_nonzero(active == i)) for i in range(NUM_CONSTRAINTS)}
    return {
        "candidate_option": lc,
        "nominal_option": ln,
        "option_switched": bool(lc != ln),
        "candidate_mode": candidate.option_modes[lc],
        "nominal_mode": nominal.option_modes[ln],
        "candidate_option_score": float(candidate.option_scores[lc]),
        "nominal_option_score": float(nominal.option_scores[ln]),
        "candidate_selected_active_type_counts": active_counts,
        "candidate_selected_reentry_active_knots": int(np.count_nonzero(cmask[:, 3])),
        "candidate_selected_reentry_active": bool(np.any(cmask[:, 3])),
        "candidate_field_reentry_active_options": int(candidate.diagnostics["reentry_active_options"]),
        "nominal_field_reentry_active_options": int(nominal.diagnostics["reentry_active_options"]),
        "valid_option_count": int(np.count_nonzero(candidate.option_valid)),
    }


def contract_checks() -> dict[str, bool]:
    # Small deterministic observation with two non-ego agents.  The first agent
    # overlaps the ego at the current observation so persistent re-entry must be
    # active; permuting the two agents must not change any constraint field.
    sample_rate = 10.0
    T = 8
    states = np.zeros((T, 9), dtype=np.float32)
    states[:, 0] = np.linspace(0.1, 2.0, T)
    states[:, 6] = 3.0
    states[:, 7] = 4.8
    states[:, 8] = 2.0
    controls = np.zeros((T, 4), dtype=np.float32)
    hist = np.zeros((3, 3, 16), dtype=np.float32)
    valid = np.ones((3, 3), dtype=np.float32)
    # ego row 0
    hist[:, 0, 10] = 4.8
    hist[:, 0, 11] = 2.0
    # agent 1 in current contact, agent 2 farther ahead
    hist[:, 1, 0] = 1.0
    hist[:, 1, 3] = 0.0
    hist[:, 1, 10] = 4.5
    hist[:, 1, 11] = 2.0
    hist[:, 2, 0] = 15.0
    hist[:, 2, 3] = -1.0
    hist[:, 2, 10] = 4.0
    hist[:, 2, 11] = 1.8
    modes = np.asarray(["brake_lane", "post_contact_stabilize", "avoid_secondary"], dtype=object)
    params = np.asarray([[-3.0, 1.0, 0.0], [0.8, 1.2, -2.0], [3.5, -3.0, 8.0]], dtype=np.float32)
    d0: dict[str, Any] = {
        "scene_id": "synthetic",
        "time_index": np.int64(3),
        "candidate_index": np.int64(0),
        "is_nominal": np.int64(1),
        "agent_history": hist,
        "agent_valid": valid,
        "ego_state": np.asarray([0, 0, 3, 0, 0, 0, 3, 4.8, 2.0], dtype=np.float32),
        "prefix_states": states,
        "prefix_controls": controls,
        "prefix_macro_id": np.int64(0),
        "prefix_param": np.zeros((3,), dtype=np.float32),
        "recovery_modes": modes,
        "recovery_params": params,
        "option_valid": np.ones((3,), dtype=np.float32),
    }
    dc = dict(d0)
    dc["candidate_index"] = np.int64(1)
    cand_states = states.copy()
    cand_states[:, 1] = np.linspace(0.0, 0.4, T)
    dc["prefix_states"] = cand_states
    validate_group_contract(d0, dc)
    cfg = {
        "sample_rate_hz": sample_rate,
        "recovery_horizon_s": 2.0,
        "d_safe0_m": 1.0,
        "safe_time_headway_s": 0.5,
        "route_dev_max_m": 2.5,
        "margin_scales": {"distance": 2.0, "stop": 5.0, "route": 1.0},
        "control_limits": {"a_min": -6.0, "a_max": 3.0, "delta_max": 0.55, "j_max": 6.0, "steer_rate_max": 0.5},
    }
    f0 = executable_constraint_field_from_sample(d0, cfg)
    fc = executable_constraint_field_from_sample(dc, cfg)
    ln = best_option_index(f0)
    lc = best_option_index(fc)
    gn = jacobian_geometry(fc, f0, ln)
    gc = jacobian_geometry(fc, f0, lc)
    zero = jacobian_geometry(f0, f0, ln)

    dp = dict(dc)
    hp = hist.copy()[:, [0, 2, 1], :]
    vp = valid.copy()[:, [0, 2, 1]]
    dp["agent_history"] = hp
    dp["agent_valid"] = vp
    fp = executable_constraint_field_from_sample(dp, cfg)
    perm_ok = bool(
        np.allclose(fc.values, fp.values, rtol=0.0, atol=1.0e-10)
        and np.array_equal(fc.masks, fp.masks)
        and np.allclose(fc.option_scores, fp.option_scores, rtol=0.0, atol=1.0e-10)
    )

    u = np.arange(5 * RAW_CANDIDATE_DIM, dtype=np.float64).reshape(5, RAW_CANDIDATE_DIM) / 1000.0
    sc = fit_jacobian_scaler(u)
    g = np.tile(gc[None, :], (5, 1))
    feat = matched_features(u, g, sc)
    return {
        "constraint_count_4": NUM_CONSTRAINTS == 4,
        "recovery_knots_8": RECOVERY_KNOTS == 8,
        "jacobian_geometry_dim_64": JACOBIAN_GEOMETRY_DIM == 64,
        "matched_dim_220": MATCHED_DIM == 220,
        "nominal_zero_exact": bool(np.count_nonzero(zero) == 0),
        "same_option_response_finite": bool(np.isfinite(gn).all() and np.isfinite(gc).all()),
        "actuator_projected": bool(f0.diagnostics.get("projected_control_envelope")),
        "reentry_activity_reachable": bool(f0.diagnostics.get("reentry_active_options", 0) > 0),
        "agent_permutation_invariant": perm_ok,
        "option_selector_valid": bool(f0.option_valid[ln] and fc.option_valid[lc]),
        "matched_family_same_dim": feat.shape[1] == MATCHED_DIM,
        "u_scale_finite_positive": bool(np.isfinite(sc.u_scale).all() and np.all(sc.u_scale > 0.0)),
    }
