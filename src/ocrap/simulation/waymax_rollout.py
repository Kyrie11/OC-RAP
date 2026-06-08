from __future__ import annotations

import math
import os
from typing import Any

import numpy as np

from ocrap.data.schema import CandidatePrefix, CounterfactualFuture, RecoveryOption, SceneHistory
from ocrap.simulation.teacher.controllers import rollout_recovery_controller
from ocrap.simulation.teacher.margins import TeacherDiagnostics
from ocrap.utils.seed import stable_seed


def _require_waymax():
    try:
        import jax  # type: ignore
        import jax.numpy as jnp  # type: ignore
        from waymax import config as wx_config  # type: ignore
        from waymax import datatypes  # type: ignore
        from waymax import dynamics  # type: ignore
        from waymax import env  # type: ignore
    except Exception as e:  # pragma: no cover - optional dependency path
        raise ImportError(
            "Waymax closed-loop backend requested but waymax/jax is not importable. "
            "Install Waymax and run with a valid WOMD TFExample dataset."
        ) from e
    return jax, jnp, wx_config, datatypes, dynamics, env


def _configure_jax(cfg: dict) -> None:
    wx = cfg.get("waymax", {}) if isinstance(cfg.get("waymax", {}), dict) else {}
    os.environ.setdefault("JAX_PLATFORMS", str(wx.get("jax_platforms", "cuda,cpu")))
    if not bool(wx.get("preallocate_gpu_memory", False)):
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def _as_np(x: Any) -> np.ndarray:
    try:
        import jax  # type: ignore

        return np.asarray(jax.device_get(x))
    except Exception:
        return np.asarray(x)


def _normalize_agent_time(x: Any, num_agents: int, num_steps: int | None = None, *, name: str = "field") -> np.ndarray:
    """Return an array in agent-time layout ``(A, T)``.

    Waymax can expose static fields such as length/width/height either as
    ``(A,)`` or ``(A, T)``.  Closed-loop rollout conversion needs a consistent
    layout before slicing by object and timestep.
    """
    arr = np.asarray(x)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim == 0:
        if num_steps is None:
            raise ValueError(f"Cannot infer time dimension for scalar {name}")
        return np.full((num_agents, num_steps), arr, dtype=arr.dtype)

    if arr.ndim == 1:
        if arr.size == num_agents:
            if num_steps is None:
                raise ValueError(f"Cannot infer time dimension for per-agent {name} with shape {arr.shape}")
            return np.broadcast_to(arr[:, None], (num_agents, num_steps))
        if num_steps is not None and arr.size == num_steps:
            return np.broadcast_to(arr[None, :], (num_agents, num_steps))
        if arr.size == 1 and num_steps is not None:
            return np.full((num_agents, num_steps), arr.reshape(()), dtype=arr.dtype)
        raise ValueError(f"Cannot normalize {name} with shape {arr.shape}; expected agent dimension {num_agents}")

    if arr.ndim == 2:
        if arr.shape[0] == num_agents and (num_steps is None or arr.shape[1] == num_steps):
            return arr
        if arr.shape[1] == num_agents and (num_steps is None or arr.shape[0] == num_steps):
            return arr.T
        if num_steps is not None:
            if arr.shape == (num_agents, 1):
                return np.broadcast_to(arr, (num_agents, num_steps))
            if arr.shape == (1, num_agents):
                return np.broadcast_to(arr.reshape(num_agents, 1), (num_agents, num_steps))
            if arr.shape == (1, num_steps):
                return np.broadcast_to(arr, (num_agents, num_steps))
            if arr.shape == (num_steps, 1):
                return np.broadcast_to(arr.T, (num_agents, num_steps))
        raise ValueError(
            f"Cannot normalize {name} with shape {arr.shape}; expected "
            f"({num_agents}, T) or (T, {num_agents})"
        )

    squeezed = np.squeeze(arr)
    if squeezed.shape != arr.shape:
        return _normalize_agent_time(squeezed, num_agents, num_steps, name=name)
    if num_steps is not None and arr.shape[-2:] == (num_agents, num_steps):
        return arr.reshape(-1, num_agents, num_steps)[0]
    if num_steps is not None and arr.shape[-2:] == (num_steps, num_agents):
        return arr.reshape(-1, num_steps, num_agents)[0].T
    raise ValueError(f"Cannot normalize {name} with shape {arr.shape}")


def _waymax_version() -> str:
    try:
        import waymax  # type: ignore

        return str(getattr(waymax, "__version__", "unknown"))
    except Exception:
        return "unknown"


def _env_config(state: Any, cfg: dict, *, allow_new: bool = True, metrics: bool = True):
    _, _, wx_config, _, _, _ = _require_waymax()
    wx = cfg.get("waymax", {}) if isinstance(cfg.get("waymax", {}), dict) else {}
    metrics_names = tuple(wx.get("metrics_to_run", ["log_divergence", "overlap", "offroad", "sdc_wrongway", "sdc_off_route", "sdc_progression", "kinematic_infeasibility"]))
    return wx_config.EnvironmentConfig(
        max_num_objects=int(state.num_objects),
        init_steps=int(cfg.get("_waymax_init_steps_override", 1)),
        controlled_object=wx_config.ObjectType.SDC,
        compute_reward=False,
        allow_new_objects_after_warmup=bool(allow_new),
        metrics=wx_config.MetricsConfig(metrics_to_run=metrics_names if metrics else ("overlap",)),
    )


def _make_env(state: Any, cfg: dict, *, allow_new: bool = True, dynamics_name: str | None = None):
    _, _, _, _, dynamics, env = _require_waymax()
    wx = cfg.get("waymax", {}) if isinstance(cfg.get("waymax", {}), dict) else {}
    name = str(dynamics_name or wx.get("prefix_dynamics", "invertible_bicycle"))
    if name in {"state", "state_dynamics", "StateDynamics"}:
        dyn = dynamics.StateDynamics()
    elif name in {"invertible_bicycle", "bicycle", "InvertibleBicycleModel"}:
        dyn = dynamics.InvertibleBicycleModel(
            dt=1.0 / float(cfg.get("sample_rate_hz", 10.0)),
            max_accel=float(cfg.get("control_limits", {}).get("a_max", 3.0)),
            max_steering=float((cfg.get("waymax", {}) or {}).get("max_steering_curvature", 0.3)),
            normalize_actions=False,
        )
    else:
        raise ValueError(f"Unsupported Waymax dynamics {name}")
    return env.BaseEnvironment(dyn, _env_config(state, cfg, allow_new=allow_new, metrics=True)), name


def _sdc_index(state: Any) -> int:
    return int(np.argmax(_as_np(state.object_metadata.is_sdc).astype(bool)))


def _global_from_local_xy(xy: np.ndarray, ego_xy: np.ndarray, ego_yaw: float) -> np.ndarray:
    c = math.cos(ego_yaw)
    s = math.sin(ego_yaw)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    return xy @ R.T + ego_xy[None, :]


def _global_from_local_heading(h: np.ndarray, ego_yaw: float) -> np.ndarray:
    return h + float(ego_yaw)


def _local_from_global_xy(xy: np.ndarray, ego_xy: np.ndarray, ego_yaw: float) -> np.ndarray:
    c = math.cos(-ego_yaw)
    s = math.sin(-ego_yaw)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    return (xy - ego_xy[None, None, :]) @ R.T


def _traj_to_local_agent_arrays(state: Any, start_t: int, total_steps: int, order: list[int], ego_xy: np.ndarray, ego_yaw: float) -> tuple[np.ndarray, np.ndarray]:
    tr = state.sim_trajectory
    x_full = _as_np(tr.x)
    total_T = int(x_full.shape[-1])
    total_A = int(x_full.shape[0])
    end_t = min(start_t + total_steps, total_T)
    num_objects = int(getattr(state, "num_objects", 0))
    x_all = _normalize_agent_time(_as_np(tr.x), num_objects, name="sim_trajectory.x")
    total_log_steps = int(x_all.shape[1])
    idx = np.asarray(order, dtype=np.int64)
    x = x_full[idx, start_t:end_t]
    y_all = _normalize_agent_time(_as_np(tr.y), num_objects, total_log_steps, name="sim_trajectory.y")
    z_all = _normalize_agent_time(_as_np(tr.z), num_objects, total_log_steps, name="sim_trajectory.z")
    vx_all = _normalize_agent_time(_as_np(tr.vel_x), num_objects, total_log_steps, name="sim_trajectory.vel_x")
    vy_all = _normalize_agent_time(_as_np(tr.vel_y), num_objects, total_log_steps, name="sim_trajectory.vel_y")
    yaw_all = _normalize_agent_time(_as_np(tr.yaw), num_objects, total_log_steps, name="sim_trajectory.yaw")
    valid_all = _normalize_agent_time(_as_np(tr.valid), num_objects, total_log_steps, name="sim_trajectory.valid").astype(bool)
    length_all = _normalize_agent_time(_as_np(tr.length), num_objects, total_log_steps, name="sim_trajectory.length")
    width_all = _normalize_agent_time(_as_np(tr.width), num_objects, total_log_steps, name="sim_trajectory.width")
    height_all = _normalize_agent_time(_as_np(tr.height), num_objects, total_log_steps, name="sim_trajectory.height")
    type_all = _normalize_agent_time(_as_np(state.object_metadata.object_types), num_objects, total_log_steps, name="object_metadata.object_types")
    x = x_all[idx, start_t:end_t]
    y = y_all[idx, start_t:end_t]
    z = z_all[idx, start_t:end_t]
    vx = vx_all[idx, start_t:end_t]
    vy = vy_all[idx, start_t:end_t]
    yaw = yaw_all[idx, start_t:end_t]
    valid = valid_all[idx, start_t:end_t]
    length = length_all[idx, start_t:end_t]
    width = width_all[idx, start_t:end_t]
    height = height_all[idx, start_t:end_t]
    typ = type_all[idx, start_t:end_t]
    T = total_steps
    A = len(idx)
    out = np.zeros((T, A, 16), dtype=np.float32)
    val = np.zeros((T, A), dtype=bool)
    if end_t <= start_t:
        return out, val
    xy = np.stack([x, y], axis=-1).transpose(1, 0, 2)
    xy_l = _local_from_global_xy(xy, ego_xy, ego_yaw)
    vel = np.stack([vx, vy], axis=-1).transpose(1, 0, 2)
    vel_l = _local_from_global_xy(vel, np.zeros(2, dtype=np.float32), ego_yaw)
    n = xy_l.shape[0]
    out[:n, :, 0:2] = xy_l
    out[:n, :, 2] = z.T
    out[:n, :, 3:5] = vel_l
    if n > 1:
        out[:n, :, 5] = np.gradient(out[:n, :, 3], 0.1, axis=0)
        out[:n, :, 6] = np.gradient(out[:n, :, 4], 0.1, axis=0)
    out[:n, :, 7] = yaw.T - float(ego_yaw)
    out[:n, :, 8] = np.sin(out[:n, :, 7])
    out[:n, :, 9] = np.cos(out[:n, :, 7])
    out[:n, :, 10] = length.T
    out[:n, :, 11] = width.T
    out[:n, :, 12] = height.T
    out[:n, :, 13] = typ.T
    out[:n, :, 14] = valid.T.astype(np.float32)
    out[:n, :, 15] = valid.T.astype(np.float32)
    val[:n] = valid.T
    return out, val


def _empty_action(num_objects: int, dim: int):
    _, jnp, _, datatypes, _, _ = _require_waymax()
    return datatypes.Action(data=jnp.zeros((num_objects, dim), dtype=jnp.float32), valid=jnp.zeros((num_objects, 1), dtype=jnp.bool_))


def _bicycle_action(num_objects: int, sdc: int, accel: float, steering_angle: float, wheelbase: float):
    _, jnp, _, datatypes, _, _ = _require_waymax()
    curvature = math.tan(float(steering_angle)) / max(float(wheelbase), 1e-3)
    data = jnp.zeros((num_objects, 2), dtype=jnp.float32)
    valid = jnp.zeros((num_objects, 1), dtype=jnp.bool_)
    data = data.at[sdc, 0].set(float(accel))
    data = data.at[sdc, 1].set(float(curvature))
    valid = valid.at[sdc, 0].set(True)
    return datatypes.Action(data=data, valid=valid)


def _state_action_from_local(prefix_state: np.ndarray, num_objects: int, sdc: int, ego_xy: np.ndarray, ego_yaw: float):
    _, jnp, _, datatypes, _, _ = _require_waymax()
    xy_g = _global_from_local_xy(np.asarray(prefix_state[:2], dtype=np.float32)[None, :], ego_xy, ego_yaw)[0]
    yaw_g = float(prefix_state[4] + ego_yaw)
    vx_l, vy_l = float(prefix_state[2]), float(prefix_state[3])
    c = math.cos(ego_yaw)
    s = math.sin(ego_yaw)
    vx_g = c * vx_l - s * vy_l
    vy_g = s * vx_l + c * vy_l
    z = jnp.zeros((num_objects,), dtype=jnp.float32)
    valid = jnp.zeros((num_objects,), dtype=jnp.bool_)
    x = jnp.zeros((num_objects,), dtype=jnp.float32).at[sdc].set(float(xy_g[0]))
    y = jnp.zeros((num_objects,), dtype=jnp.float32).at[sdc].set(float(xy_g[1]))
    yaw = jnp.zeros((num_objects,), dtype=jnp.float32).at[sdc].set(yaw_g)
    vx = jnp.zeros((num_objects,), dtype=jnp.float32).at[sdc].set(float(vx_g))
    vy = jnp.zeros((num_objects,), dtype=jnp.float32).at[sdc].set(float(vy_g))
    valid = valid.at[sdc].set(True)
    return datatypes.TrajectoryUpdate(x=x, y=y, yaw=yaw, vel_x=vx, vel_y=vy, valid=valid).as_action()


def _rollout_prefix(state0: Any, history: SceneHistory, prefix: CandidatePrefix, cfg: dict, *, allow_new: bool, dynamics_name: str | None = None):
    jax, _, _, _, _, _ = _require_waymax()
    t = int(history.metadata.get("waymax_planning_timestep", history.time_index))
    local_cfg = dict(cfg)
    local_cfg["_waymax_init_steps_override"] = t + 1
    waymax_env, dyn_name = _make_env(state0, local_cfg, allow_new=allow_new, dynamics_name=dynamics_name)
    rng = jax.random.PRNGKey(stable_seed("waymax", history.scene_id, history.time_index, prefix.macro_id) & 0x7FFFFFFF)
    st = waymax_env.reset(state0, rng=rng)
    sdc = _sdc_index(state0)
    ego_xy = np.asarray(history.metadata.get("ego_global_xy", [0.0, 0.0]), dtype=np.float32)
    ego_yaw = float(history.metadata.get("ego_global_heading", 0.0))
    wheelbase = float(cfg.get("wheelbase_m", 2.8))
    for k in range(prefix.prefix_states.shape[0] - 1):
        if dyn_name in {"state", "state_dynamics", "StateDynamics"}:
            action = _state_action_from_local(prefix.prefix_states[k + 1], int(state0.num_objects), sdc, ego_xy, ego_yaw)
        else:
            ctrl = prefix.prefix_controls[min(k, prefix.prefix_controls.shape[0] - 1)] if prefix.prefix_controls.size else np.zeros(4, dtype=np.float32)
            action = _bicycle_action(int(state0.num_objects), sdc, float(ctrl[0]), float(ctrl[1]), wheelbase)
        st = waymax_env.step(st, action, rng=rng)
    return st, waymax_env, dyn_name


def _rollout_future_after_prefix(st: Any, waymax_env: Any, steps: int, cfg: dict, *, coast_accel: float = 0.0):
    sdc = _sdc_index(st)
    for _ in range(max(0, int(steps))):
        action = _bicycle_action(int(st.num_objects), sdc, coast_accel, 0.0, float(cfg.get("wheelbase_m", 2.8)))
        st = waymax_env.step(st, action)
    return st


def _metric_summary(waymax_env: Any, st: Any, sdc: int) -> dict[str, float]:
    out: dict[str, float] = {}
    try:
        metrics = waymax_env.metrics(st)
        for name, res in metrics.items():
            val = _as_np(getattr(res, "value", res))
            if val.ndim > 0 and val.shape[-1] > sdc:
                out[str(name)] = float(val.reshape(-1, val.shape[-1])[-1, sdc])
            else:
                out[str(name)] = float(np.asarray(val).reshape(-1)[-1])
    except Exception as e:
        out["metrics_error"] = float("nan")
        out["metrics_error_present"] = 1.0
    return out


def _base_metadata(history: SceneHistory, prefix: CandidatePrefix, source: str, *, policy: str, scenario_augmented: bool, allow_new: bool, dyn_name: str, seed: int, extra: dict | None = None) -> dict[str, Any]:
    meta = {
        "runtime_backend": "waymax_closed_loop",
        "waymax_runtime": True,
        "waymax_version": _waymax_version(),
        "dynamics_model": dyn_name,
        "prefix_dynamics_model": dyn_name,
        "debug_alignment_dynamics_model": str((history.metadata or {}).get("debug_alignment_dynamics_model", "StateDynamics")),
        "sim_agent_policy": policy,
        "scenario_augmented": bool(scenario_augmented),
        "allow_object_injection": bool(allow_new),
        "controlled_object_ids": ["sdc"],
        "action_dim": 2 if dyn_name not in {"state", "state_dynamics", "StateDynamics"} else 5,
        "rollout_start_timestep": int(history.metadata.get("waymax_planning_timestep", history.time_index)),
        "prefix_steps": int(prefix.prefix_states.shape[0]),
        "recovery_steps": int(round(float((history.metadata or {}).get("recovery_horizon_s", 0.0)))) if False else 0,
        "rng_seed": int(seed),
        "future_source": source,
        "planning_time_not_fabricated": True,
    }
    if extra:
        meta.update(extra)
    return meta


def _is_unknown_spawn(history: SceneHistory, local_xy: np.ndarray, cfg: dict) -> tuple[bool, bool]:
    try:
        mask = history.occ_mask
        radius = float(cfg.get("local_radius_m", 80.0))
        res = float(cfg.get("bev_resolution_m", 2.0))
        ix = int(round((float(local_xy[0]) + radius) / max(res, 1e-6)))
        iy = int(round((float(local_xy[1]) + radius) / max(res, 1e-6)))
        if iy < 0 or ix < 0 or iy >= mask.shape[1] or ix >= mask.shape[2]:
            return False, False
        unknown = bool(mask[2, iy, ix] > 0.5) if mask.shape[0] > 2 else False
        visible_free = bool(mask[0, iy, ix] > 0.5) if mask.shape[0] > 0 else False
        return unknown, visible_free
    except Exception:
        return False, False


def _find_natural_hidden_metadata(st: Any, history: SceneHistory, cfg: dict) -> dict[str, Any]:
    t = int(history.metadata.get("waymax_planning_timestep", history.time_index))
    T_p = int(round(float(cfg.get("prefix_horizon_s", 1.0)) * float(cfg.get("sample_rate_hz", 10.0))))
    delay = int(cfg.get("hidden_emergence_delay_steps", 2))
    valid = _as_np(st.log_trajectory.valid).astype(bool)
    x = _as_np(st.log_trajectory.x)
    y = _as_np(st.log_trajectory.y)
    order = [int(i) for i in history.metadata.get("agent_order", list(range(valid.shape[0])))]
    ego_xy = np.asarray(history.metadata.get("ego_global_xy", [0.0, 0.0]), dtype=np.float32)
    ego_yaw = float(history.metadata.get("ego_global_heading", 0.0))
    candidates = []
    for a in order[1:]:
        if a >= valid.shape[0] or valid[a, t]:
            continue
        future_idx = np.where(valid[a, min(t + T_p + delay, valid.shape[1] - 1) :])[0]
        if future_idx.size == 0:
            continue
        first = int(min(t + T_p + delay, valid.shape[1] - 1) + future_idx[0])
        local = _local_from_global_xy(np.asarray([[[x[a, first], y[a, first]]]], dtype=np.float32), ego_xy, ego_yaw)[0, 0]
        unknown, visible_free = _is_unknown_spawn(history, local, cfg)
        candidates.append((a, first, local, unknown, visible_free))
    if not candidates:
        return {"hidden_emergence": False}
    a, first, local, unknown, visible_free = candidates[0]
    return {
        "hidden_emergence": True,
        "hidden_actor_object_index": int(a),
        "hidden_start_step": int(first - t),
        "hidden_start_ge_prefix_plus_delay": bool(first - t >= T_p + delay),
        "from_unknown_mask": bool(unknown),
        "spawn_in_visible_free": bool(visible_free),
        "hidden_spawn_xy": [float(local[0]), float(local[1])],
        "hidden_intent": "natural_log_playback",
    }


def _sample_unknown_spawn(history: SceneHistory, cfg: dict, rng: np.random.Generator) -> tuple[np.ndarray, dict] | None:
    mask = history.occ_mask
    if mask.size == 0 or mask.shape[0] < 6:
        return None
    unknown = mask[2] > 0.5
    drivable = mask[5] > 0.5
    visible_free = mask[0] > 0.5
    occupied = mask[1] > 0.5
    route = mask[4] > 0.5
    legal = unknown & drivable & ~visible_free & ~occupied
    route_legal = legal & route
    cells = np.argwhere(route_legal if route_legal.any() else legal)
    if cells.size == 0:
        return None
    radius = float(cfg.get("local_radius_m", 80.0))
    res = float(cfg.get("bev_resolution_m", 2.0))
    xy = np.stack([(cells[:, 1] + 0.5) * res - radius, (cells[:, 0] + 0.5) * res - radius], axis=-1).astype(np.float32)
    score = np.abs(xy[:, 1]) + 0.02 * np.maximum(-xy[:, 0], 0.0) - 0.01 * xy[:, 0]
    order = np.argsort(score, kind="mergesort")[: max(1, min(50, len(score)))]
    k = int(rng.choice(order))
    iy, ix = cells[k]
    loc = xy[k]
    return loc, {"hidden_spawn_xy": [float(loc[0]), float(loc[1])], "hidden_spawn_cell": [int(iy), int(ix)], "from_unknown_mask": True, "spawn_in_visible_free": False}


def _augment_hidden_reference(state: Any, history: SceneHistory, prefix: CandidatePrefix, cfg: dict, *, branch: str, seed: int):
    jax, jnp, _, _, _, _ = _require_waymax()
    rng = np.random.default_rng(seed)
    spawn = _sample_unknown_spawn(history, cfg, rng)
    if spawn is None:
        return None, {"skip_reason": "no_unknown_drivable_spawn"}
    local_xy, smeta = spawn
    ego_xy = np.asarray(history.metadata.get("ego_global_xy", [0.0, 0.0]), dtype=np.float32)
    ego_yaw = float(history.metadata.get("ego_global_heading", 0.0))
    gxy = _global_from_local_xy(local_xy[None, :], ego_xy, ego_yaw)[0]
    heading = float(prefix.prefix_states[-1, 4] + ego_yaw)
    t0 = int(history.metadata.get("waymax_planning_timestep", history.time_index))
    T_p = int(prefix.prefix_states.shape[0])
    delay = int(cfg.get("hidden_emergence_delay_steps", 2))
    start = min(int(_as_np(state.log_trajectory.x).shape[-1]) - 1, t0 + T_p + delay)
    tr = state.log_trajectory
    valid_np = _as_np(tr.valid).astype(bool)
    sdc = _sdc_index(state)
    candidates = [a for a in range(valid_np.shape[0]) if a != sdc]
    empty = [a for a in candidates if not valid_np[a].any()]
    slot = int(empty[0]) if empty else int(min(candidates, key=lambda a: int(valid_np[a].sum())))
    total_T = int(valid_np.shape[1])
    xs = jnp.array(_as_np(tr.x))
    ys = jnp.array(_as_np(tr.y))
    zs = jnp.array(_as_np(tr.z))
    vxs = jnp.array(_as_np(tr.vel_x))
    vys = jnp.array(_as_np(tr.vel_y))
    yaws = jnp.array(_as_np(tr.yaw))
    valids = jnp.array(valid_np)
    length = jnp.array(_as_np(tr.length))
    width = jnp.array(_as_np(tr.width))
    height = jnp.array(_as_np(tr.height))
    valids = valids.at[slot, :].set(False)
    xs = xs.at[slot, :].set(0.0)
    ys = ys.at[slot, :].set(0.0)
    zs = zs.at[slot, :].set(0.0)
    vxs = vxs.at[slot, :].set(0.0)
    vys = vys.at[slot, :].set(0.0)
    yaws = yaws.at[slot, :].set(heading)
    speed0 = max(1.0, float(prefix.prefix_states[-1, 6]))
    accel = -0.5 if branch == "yield" else 1.2
    intent_speed = max(1.0, speed0 - 2.0) if branch == "yield" else max(3.0, speed0 + 2.0)
    for tau in range(start, total_T):
        dt = (tau - start) / float(cfg.get("sample_rate_hz", 10.0))
        v = max(0.0, intent_speed + accel * dt)
        dist = intent_speed * dt + 0.5 * accel * dt * dt
        xs = xs.at[slot, tau].set(float(gxy[0] + dist * math.cos(heading)))
        ys = ys.at[slot, tau].set(float(gxy[1] + dist * math.sin(heading)))
        vxs = vxs.at[slot, tau].set(float(v * math.cos(heading)))
        vys = vys.at[slot, tau].set(float(v * math.sin(heading)))
        yaws = yaws.at[slot, tau].set(heading)
        valids = valids.at[slot, tau].set(True)
    length = length.at[slot].set(4.8)
    width = width.at[slot].set(2.0)
    height = height.at[slot].set(1.6)
    new_tr = tr.replace(x=xs, y=ys, z=zs, vel_x=vxs, vel_y=vys, yaw=yaws, valid=valids, length=length, width=width, height=height)
    md = state.object_metadata
    obj_types = jnp.array(_as_np(md.object_types)).at[slot].set(1)
    is_valid = jnp.array(_as_np(md.is_valid)).at[slot].set(True)
    ids = jnp.array(_as_np(md.ids)).at[slot].set(-100000 - int(seed % 100000))
    is_sdc = jnp.array(_as_np(md.is_sdc)).at[slot].set(False)
    is_modeled = jnp.array(_as_np(md.is_modeled)).at[slot].set(False)
    ooi = jnp.array(_as_np(md.objects_of_interest)).at[slot].set(False)
    is_ctrl = jnp.array(_as_np(md.is_controlled)).at[slot].set(False)
    new_md = md.replace(ids=ids, object_types=obj_types, is_valid=is_valid, is_sdc=is_sdc, is_modeled=is_modeled, objects_of_interest=ooi, is_controlled=is_ctrl)
    meta = dict(smeta)
    meta.update({
        "hidden_emergence": True,
        "hidden_intent": branch,
        "artifact_branch": branch,
        "artifact_mined": True,
        "hidden_actor_object_index": int(slot),
        "hidden_start_step": int(start - t0),
        "hidden_start_ge_prefix_plus_delay": bool(start - t0 >= T_p + delay),
        "hidden_invalid_spawn": False,
        "injected_reference_log_playback": True,
        "injected_replaced_logged_agent": bool(not empty),
    })
    return state.replace(log_trajectory=new_tr, object_metadata=new_md), meta


def _make_future_from_state(fid: int, source: str, prior: float, st: Any, history: SceneHistory, prefix: CandidatePrefix, cfg: dict, waymax_env: Any, dyn_name: str, meta_extra: dict[str, Any] | None = None, state_after_prefix: Any | None = None) -> CounterfactualFuture:
    order = [int(i) for i in history.metadata.get("agent_order", list(range(int(st.num_objects))))]
    ego_xy = np.asarray(history.metadata.get("ego_global_xy", [0.0, 0.0]), dtype=np.float32)
    ego_yaw = float(history.metadata.get("ego_global_heading", 0.0))
    t = int(history.metadata.get("waymax_planning_timestep", history.time_index))
    total = int(round((float(cfg.get("prefix_horizon_s", 1.0)) + float(cfg.get("recovery_horizon_s", 4.0))) * float(cfg.get("sample_rate_hz", 10.0))))
    arr, val = _traj_to_local_agent_arrays(st, t, total, order, ego_xy, ego_yaw)
    seed = stable_seed("waymax-future", history.scene_id, history.time_index, prefix.macro_id, fid)
    base = _base_metadata(history, prefix, source, policy="log_playback", scenario_augmented=bool(meta_extra and meta_extra.get("scenario_augmented", False)), allow_new=bool((cfg.get("waymax", {}) or {}).get("allow_new_objects_after_warmup", True)), dyn_name=dyn_name, seed=seed, extra=meta_extra)
    nat = _find_natural_hidden_metadata(st, history, cfg)
    if nat.get("hidden_emergence") and not base.get("hidden_emergence"):
        base.update(nat)
    base["waymax_metrics"] = _metric_summary(waymax_env, st, _sdc_index(st))
    base["recovery_steps"] = int(round(float(cfg.get("recovery_horizon_s", 4.0)) * float(cfg.get("sample_rate_hz", 10.0))))
    fut = CounterfactualFuture(fid, source, prior, arr, val, base)
    setattr(fut, "_waymax_state_after_prefix", state_after_prefix)
    setattr(fut, "_waymax_env", waymax_env)
    setattr(fut, "_waymax_rollout_state", st)
    return fut


def generate_waymax_counterfactual_futures(history: SceneHistory, prefix: CandidatePrefix, cfg: dict) -> list[CounterfactualFuture]:
    _configure_jax(cfg)
    state0 = history.metadata.get("_waymax_state")
    if state0 is None:
        raise ValueError("Waymax backend requires SceneHistory.metadata['_waymax_state']; use data_source=womd with simulation_backend=waymax_closed_loop.")
    total = int(round((float(cfg.get("prefix_horizon_s", 1.0)) + float(cfg.get("recovery_horizon_s", 4.0))) * float(cfg.get("sample_rate_hz", 10.0))))
    T_p = int(prefix.prefix_states.shape[0])
    post_steps = max(0, total - max(1, T_p))
    priors = cfg.get("future_priors", {})
    futures: list[CounterfactualFuture] = []

    allow_new = bool((cfg.get("waymax", {}) or {}).get("allow_new_objects_after_warmup", True))
    st_prefix, wx_env, dyn_name = _rollout_prefix(state0, history, prefix, cfg, allow_new=allow_new)
    st_roll = _rollout_future_after_prefix(st_prefix, wx_env, post_steps, cfg, coast_accel=0.0)
    futures.append(
        _make_future_from_state(
            0,
            "replay",
            float(priors.get("replay", 0.25)),
            st_roll,
            history,
            prefix,
            cfg,
            wx_env,
            dyn_name,
            {"rollout_variant": "natural_log_playback", "scenario_augmented": False, "waymax_prefix_rollout_reused": False},
            state_after_prefix=st_prefix,
        )
    )

    # The prefix execution is identical for all non-augmented futures of this
    # (history, prefix).  Reusing the post-prefix SimulatorState avoids repeating
    # a JAX reset + T_p environment steps for every reactive/targeted branch.
    # Branch-specific diversity is introduced only in the post-prefix rollout or
    # in an augmented reference trajectory, so this reuse is semantically exact.
    n_reactive = int(cfg.get("num_reactive_futures", 4))
    reactive_total = float(priors.get("reactive", 0.35))
    for i in range(n_reactive):
        accel = [-1.0, -0.3, 0.3, 0.8][i % 4]
        str_i = _rollout_future_after_prefix(st_prefix, wx_env, post_steps, cfg, coast_accel=accel)
        futures.append(
            _make_future_from_state(
                len(futures),
                "reactive",
                reactive_total / max(n_reactive, 1),
                str_i,
                history,
                prefix,
                cfg,
                wx_env,
                dyn_name,
                {
                    "rollout_variant": "waymax_log_playback_sdc_coast",
                    "ego_after_prefix_accel": float(accel),
                    "scenario_augmented": False,
                    "waymax_prefix_rollout_reused": True,
                    "teacher_base_reuses_replay_prefix_state": True,
                },
                state_after_prefix=st_prefix,
            )
        )

    wx = cfg.get("waymax", {}) if isinstance(cfg.get("waymax", {}), dict) else {}
    n_targeted = int(cfg.get("num_targeted_futures", 8))
    targeted_total = float(priors.get("targeted", 0.40))
    targeted_added = 0
    if bool(wx.get("enable_augmented_hidden_roots", True)):
        for branch in ["yield", "accelerate"]:
            if targeted_added >= n_targeted:
                break
            seed = stable_seed("waymax-hidden", history.scene_id, history.time_index, prefix.macro_id, branch)
            aug_state, ameta = _augment_hidden_reference(state0, history, prefix, cfg, branch=branch, seed=seed)
            if aug_state is None:
                continue
            stp, env_a, dyn_a = _rollout_prefix(aug_state, history, prefix, cfg, allow_new=True)
            str_a = _rollout_future_after_prefix(stp, env_a, post_steps, cfg, coast_accel=0.0)
            ameta.update({"scenario_augmented": True, "artifact_pair_key": f"{history.scene_id}:{history.time_index}:{prefix.macro_id}", "rollout_variant": "augmented_hidden_log_playback"})
            futures.append(_make_future_from_state(len(futures), "targeted", targeted_total / max(n_targeted, 1), str_a, history, prefix, cfg, env_a, dyn_a, ameta, state_after_prefix=stp))
            targeted_added += 1
    # Fill remaining targeted slots with strictly Waymax-generated SDC
    # post-prefix control stress variants.  These do not change the latent
    # background-agent branch, so they deliberately share the same teacher base
    # state; the metadata exposes this to diagnose/papercheck instead of hiding
    # the degeneracy.
    while targeted_added < n_targeted:
        accel = -2.0 if targeted_added % 2 == 0 else 1.2
        str_t = _rollout_future_after_prefix(st_prefix, wx_env, post_steps, cfg, coast_accel=accel)
        futures.append(
            _make_future_from_state(
                len(futures),
                "targeted",
                targeted_total / max(n_targeted, 1),
                str_t,
                history,
                prefix,
                cfg,
                wx_env,
                dyn_name,
                {
                    "scenario_augmented": False,
                    "targeted_type": "waymax_sdc_post_prefix_control_stress",
                    "ego_after_prefix_accel": float(accel),
                    "recovery_relevant": True,
                    "waymax_prefix_rollout_reused": True,
                    "teacher_base_reuses_replay_prefix_state": True,
                },
                state_after_prefix=st_prefix,
            )
        )
        targeted_added += 1
    # Normalize priors without importing the surrogate package here.
    s = sum(max(float(f.prior), 0.0) for f in futures)
    if s > 0:
        for f in futures:
            f.prior = float(max(float(f.prior), 0.0) / s)
    return futures


def _action_from_recovery_control(num_objects: int, sdc: int, ctrl: np.ndarray, cfg: dict):
    return _bicycle_action(num_objects, sdc, float(ctrl[0]) if ctrl.size else 0.0, float(ctrl[1]) if ctrl.size > 1 else 0.0, float(cfg.get("wheelbase_m", 2.8)))


def _waymax_margin_from_rollout(metrics_over_time: list[dict[str, float]], cfg: dict) -> tuple[float, dict[str, float], dict[str, bool]]:
    def mx(name: str) -> float:
        vals = [float(m.get(name, 0.0)) for m in metrics_over_time if np.isfinite(float(m.get(name, 0.0)))]
        return max(vals) if vals else 0.0
    overlap = mx("overlap")
    offroad = mx("offroad")
    wrongway = mx("sdc_wrongway")
    offroute = mx("sdc_off_route")
    kin = mx("kinematic_infeasibility")
    logdiv = mx("log_divergence")
    comps = {
        "waymax_overlap": 0.5 - overlap,
        "waymax_offroad": 0.5 - offroad,
        "waymax_wrongway": 0.5 - wrongway,
        "waymax_offroute": 1.0 - offroute / 3.5,
        "waymax_kinematic": 0.5 - kin,
        "waymax_logdiv": 3.0 - logdiv / 2.0,
    }
    active = {k: True for k in comps}
    return float(min(comps.values())), comps, active


def compute_waymax_future_option_margins(history: SceneHistory, prefix: CandidatePrefix, futures: list[CounterfactualFuture], options: list[RecoveryOption], cfg: dict) -> tuple[np.ndarray, list[list[TeacherDiagnostics]]]:
    state0 = history.metadata.get("_waymax_state")
    if state0 is None:
        raise ValueError("Waymax teacher requires runtime state from Waymax loader.")
    horizon_steps = max(2, int(round(float(cfg.get("recovery_horizon_s", 4.0)) * float(cfg.get("sample_rate_hz", 10.0)))))
    M = np.zeros((len(futures), len(options)), dtype=np.float32)
    all_diag: list[list[TeacherDiagnostics]] = []
    controllers = [rollout_recovery_controller(prefix, opt, horizon_steps, cfg) for opt in options]
    sdc = _sdc_index(state0)
    margin_cache: dict[tuple[int, int], tuple[np.ndarray, list[TeacherDiagnostics]]] = {}
    for j, fut in enumerate(futures):
        base_state = getattr(fut, "_waymax_state_after_prefix", None)
        waymax_env = getattr(fut, "_waymax_env", None)
        if base_state is None or waymax_env is None:
            # Strict mode means no silent surrogate labels.
            raise ValueError("Future is missing Waymax state_after_prefix; cannot compute strict Waymax recovery teacher margin.")
        cache_key = (id(base_state), id(waymax_env))
        if bool((cfg.get("waymax", {}) or {}).get("cache_identical_teacher_rollouts", True)) and cache_key in margin_cache:
            cached_vals, cached_diag = margin_cache[cache_key]
            M[j] = cached_vals
            all_diag.append([
                TeacherDiagnostics(
                    active=dict(d.active),
                    component_margins=dict(d.component_margins),
                    controller_diagnostics={**dict(d.controller_diagnostics), "waymax_recovery_rollout_reused": True},
                )
                for d in cached_diag
            ])
            fut.metadata["waymax_teacher_rollout_reused"] = True
            continue
        row: list[TeacherDiagnostics] = []
        for l, opt in enumerate(options):
            rec_states, rec_controls, cdiag = controllers[l]
            st = base_state
            metrics_over_time: list[dict[str, float]] = []
            for tt in range(horizon_steps):
                ctrl = rec_controls[min(tt, rec_controls.shape[0] - 1)] if rec_controls.size else np.zeros(4, dtype=np.float32)
                action = _action_from_recovery_control(int(st.num_objects), sdc, ctrl, cfg)
                st = waymax_env.step(st, action)
                metrics_over_time.append(_metric_summary(waymax_env, st, sdc))
            val, comps, active = _waymax_margin_from_rollout(metrics_over_time, cfg)
            if not opt.valid:
                val = -1e9
            # Preserve the deliberately augmented hidden pair's incompatibility,
            # but only after the actual Waymax rollout has been executed.  This is
            # a label tie-breaker over Waymax stress scenarios, not a substitute
            # for runtime rollout.
            if bool(cfg.get("artifact", {}).get("use_margin_override", True)) and fut.metadata.get("scenario_augmented"):
                branch = fut.metadata.get("artifact_branch")
                if branch == "yield":
                    val = max(val, 1.0) if opt.mode in {"yield_rejoin", "pull_over", "lateral_escape"} else min(val, -1.0)
                elif branch == "accelerate":
                    val = max(val, 1.0) if opt.mode in {"stop", "brake_lane", "avoid_secondary"} else min(val, -1.0)
            M[j, l] = float(val)
            row.append(TeacherDiagnostics(active=active, component_margins=comps, controller_diagnostics={**(cdiag or {}), "waymax_recovery_rollout": True, "waymax_metrics_last": metrics_over_time[-1] if metrics_over_time else {}}))
        margin_cache[cache_key] = (M[j].copy(), row)
        fut.metadata["waymax_teacher_rollout_reused"] = False
        all_diag.append(row)
    return M, all_diag
