from __future__ import annotations

import math
from typing import Any

import numpy as np

from ocrap.data.schema import RawScenario, SceneHistory
from ocrap.planning.route_lattice import project_to_route
from ocrap.simulation.observation.bev import render_base_occ_mask
from ocrap.utils.geometry import transform_points_to_ego, transform_states_to_ego



def _future_route_proxy(future_ego: np.ndarray, future_valid: np.ndarray, max_points: int) -> np.ndarray:
    if future_ego.size == 0 or future_valid.size == 0:
        pts = np.stack([np.linspace(0, max_points - 1, max_points, dtype=np.float32), np.zeros(max_points, dtype=np.float32)], axis=-1)
    else:
        valid = future_valid[:, 0].astype(bool) if future_valid.ndim == 2 and future_valid.shape[1] else np.zeros((future_ego.shape[0],), dtype=bool)
        pts = future_ego[valid, 0, :2].astype(np.float32)
        if len(pts) < 2:
            pts = np.stack([np.linspace(0, max_points - 1, max_points, dtype=np.float32), np.zeros(max_points, dtype=np.float32)], axis=-1)
    if len(pts) < max_points:
        pad = np.repeat(pts[-1:, :], max_points - len(pts), axis=0)
        pts = np.concatenate([pts, pad], axis=0)
    else:
        idx = np.linspace(0, len(pts) - 1, max_points).round().astype(int)
        pts = pts[idx]
    route = np.zeros((max_points, 6), dtype=np.float32)
    route[:, :2] = pts[:, :2]
    d = np.diff(route[:, :2], axis=0, append=route[-1:, :2])
    route[:, 2] = np.arctan2(d[:, 1], d[:, 0])
    route[:, 3] = 13.4
    route[:, 5] = 1.0
    return route


def _sanitize_route(route: np.ndarray, future_ego: np.ndarray, future_valid: np.ndarray, cfg: dict) -> tuple[np.ndarray, dict]:
    max_points = int(cfg.get("route_points", route.shape[0] if route.size else 80))
    meta = {"route_sanitized": False, "route_projection_distance_m": 0.0}
    if route.size and len(route) >= 2:
        try:
            proj = project_to_route(np.zeros(2, dtype=np.float32), route)
            meta["route_projection_distance_m"] = float(proj.distance)
            length = float(np.sum(np.linalg.norm(np.diff(route[:, :2], axis=0), axis=1)))
            if proj.distance <= float(cfg.get("max_route_projection_distance_m", 8.0)) and length >= float(cfg.get("min_route_length_m", 10.0)):
                return route.astype(np.float32), meta
        except Exception:
            pass
    meta["route_sanitized"] = True
    meta["route_sanitize_reason"] = "route_not_near_ego_or_too_short"
    return _future_route_proxy(future_ego, future_valid, max_points), meta

def ego_from_agent_state(agent_state: np.ndarray) -> np.ndarray:
    return np.array([agent_state[0], agent_state[1], agent_state[3], agent_state[4], agent_state[7], 0.0, math.hypot(agent_state[3], agent_state[4]), agent_state[10], agent_state[11]], dtype=np.float32)


def transform_map_and_route(raw: RawScenario, ego_state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ego_xy = ego_state[:2]
    ego_h = float(ego_state[7])
    maps = raw.map_polylines.copy().astype(np.float32)
    if maps.size:
        maps[..., :2] = transform_points_to_ego(maps[..., :2], ego_xy, ego_h)
    route = raw.route.copy().astype(np.float32)
    if route.size:
        route[..., :2] = transform_points_to_ego(route[..., :2], ego_xy, ego_h)
        if route.shape[-1] > 2:
            route[..., 2] = route[..., 2] - ego_h
    return maps, raw.map_valid.copy(), route


def construct_history(raw: RawScenario, t: int, cfg: dict) -> SceneHistory:
    sr = float(cfg.get("sample_rate_hz", 10))
    H = max(1, int(round(float(cfg.get("history_horizon_s", 1.0)) * sr)))
    total_future = max(2, int(round((float(cfg.get("prefix_horizon_s", 1.0)) + float(cfg.get("recovery_horizon_s", 4.0))) * sr)))
    sdc = int(raw.sdc_track_index)
    order = [sdc] + [i for i in range(raw.agent_states.shape[1]) if i != sdc]
    max_agents = min(int(cfg.get("max_agents", len(order))), len(order))
    order = order[:max_agents]
    t0 = max(0, t - H + 1)
    hist = raw.agent_states[t0 : t + 1, order]
    hist_valid = raw.agent_valid[t0 : t + 1, order]
    if hist.shape[0] < H:
        pad = np.zeros((H - hist.shape[0], hist.shape[1], hist.shape[2]), dtype=np.float32)
        pad_valid = np.zeros((H - hist.shape[0], hist.shape[1]), dtype=bool)
        hist = np.concatenate([pad, hist], axis=0)
        hist_valid = np.concatenate([pad_valid, hist_valid], axis=0)
    tend = min(raw.agent_states.shape[0], t + total_future)
    future = raw.agent_states[t:tend, order]
    future_valid = raw.agent_valid[t:tend, order]
    if future.shape[0] < total_future:
        pad = np.zeros((total_future - future.shape[0], len(order), raw.agent_states.shape[-1]), dtype=np.float32)
        pad_valid = np.zeros((total_future - future.shape[0], len(order)), dtype=bool)
        future = np.concatenate([future, pad], axis=0)
        future_valid = np.concatenate([future_valid, pad_valid], axis=0)
    ego_raw = raw.agent_states[t, sdc]
    hist_e = transform_states_to_ego(hist, ego_raw)
    fut_e = transform_states_to_ego(future, ego_raw)
    maps, map_valid, route = transform_map_and_route(raw, ego_raw)
    route, route_meta = _sanitize_route(route, fut_e, future_valid, cfg)
    dyn = raw.dynamic_map[max(0, t - H + 1) : t + 1]
    if dyn.shape[0] < H:
        pad = np.zeros((H - dyn.shape[0],) + dyn.shape[1:], dtype=np.float32)
        dyn = np.concatenate([pad, dyn], axis=0)
    h = SceneHistory(
        scene_id=raw.scenario_id,
        original_scenario_id=str(raw.metadata.get("original_scenario_id", raw.scenario_id)),
        time_index=int(t),
        agent_history=hist_e.astype(np.float32),
        agent_valid=hist_valid.astype(bool),
        map_polylines=maps.astype(np.float32),
        map_valid=map_valid.astype(bool),
        dynamic_map=dyn.astype(np.float32),
        route=route.astype(np.float32),
        occ_mask=np.zeros((int(cfg.get("bev_channels", 7)), 2, 2), dtype=np.float32),
        ego_state=ego_from_agent_state(hist_e[-1, 0]),
        future_agent_states=fut_e.astype(np.float32),
        future_agent_valid=future_valid.astype(bool),
        metadata={
            "speed_limit": float(cfg.get("speed_limit_default", 13.4)),
            "shoulder_available": True,
            "adjacent_available": True,
            "time_sampling_reasons": [],
            "source": raw.metadata.get("source", "unknown"),
            "agent_order": [int(i) for i in order],
            "ego_global_xy": [float(ego_raw[0]), float(ego_raw[1])],
            "ego_global_heading": float(ego_raw[7]),
            "waymax_planning_timestep": int(t),
            "waymax_sdc_original_index": int(sdc),
            # Private runtime-only handles.  These are intentionally not written
            # to NPZ; they are consumed by the Waymax rollout backend before
            # DatasetSample.to_npz_dict serializes json-safe metadata.
            "_waymax_state": raw.metadata.get("_waymax_state"),
            "_waymax_scenario_index": raw.metadata.get("_waymax_scenario_index"),
            **route_meta,
        },
    )
    h.occ_mask = render_base_occ_mask(h, cfg)
    return h


def _host_get_tree(tree: Any) -> Any:
    """Move a small pytree of Waymax/JAX slices to host in one synchronization."""
    try:
        import jax  # type: ignore

        return jax.device_get(tree)
    except Exception:
        return tree


def _squeeze_waymax_field(x: Any) -> Any:
    """Drop singleton batch axes without materializing the underlying array."""
    out = x
    shape = tuple(getattr(out, "shape", ()))
    while len(shape) > 2 and shape[0] == 1:
        out = out[0]
        shape = tuple(getattr(out, "shape", ()))
    return out


def _waymax_agent_time_shape(x: Any, num_agents: int, *, name: str) -> tuple[Any, int, str]:
    """Return squeezed field, number of timesteps and its 2-D orientation."""
    out = _squeeze_waymax_field(x)
    shape = tuple(getattr(out, "shape", ()))
    if len(shape) != 2:
        raise ValueError(f"Fast Waymax history expects 2-D {name}, got shape={shape}")
    if shape[0] == int(num_agents):
        return out, int(shape[1]), "agent_time"
    if shape[1] == int(num_agents):
        return out, int(shape[0]), "time_agent"
    raise ValueError(f"Cannot identify agent axis for {name} shape={shape}, num_agents={num_agents}")


def _slice_waymax_agent_time(
    x: Any,
    *,
    num_agents: int,
    num_steps: int,
    order: np.ndarray,
    start: int,
    end: int,
    name: str,
) -> Any:
    """Slice a Waymax field on device before host transfer.

    Dynamic trajectory fields are normally ``(A,T)``. Static per-agent fields
    such as object type may be ``(A,)`` and are broadcast after the small agent
    slice reaches host. The less-common time-major layout is also supported.
    """
    out = _squeeze_waymax_field(x)
    shape = tuple(getattr(out, "shape", ()))
    if len(shape) == 0:
        return np.full((len(order), end - start), np.asarray(out).reshape(()), dtype=np.asarray(out).dtype)
    if len(shape) == 1:
        if shape[0] == int(num_agents):
            vals = out[order]
            return ("per_agent", vals, int(end - start))
        if shape[0] == int(num_steps):
            vals = out[start:end]
            return ("per_time", vals, int(len(order)))
        if shape[0] == 1:
            return ("scalar", out[0], (int(len(order)), int(end - start)))
        raise ValueError(f"Cannot slice {name} shape={shape}")
    if len(shape) == 2:
        if shape[0] == int(num_agents) and shape[1] == int(num_steps):
            return out[order, start:end]
        if shape[1] == int(num_agents) and shape[0] == int(num_steps):
            return out[start:end, order].T
        if shape == (int(num_agents), 1):
            return ("per_agent", out[order, 0], int(end - start))
        if shape == (1, int(num_agents)):
            return ("per_agent", out[0, order], int(end - start))
        if shape == (1, int(num_steps)):
            return ("per_time", out[0, start:end], int(len(order)))
        if shape == (int(num_steps), 1):
            return ("per_time", out[start:end, 0], int(len(order)))
    raise ValueError(f"Cannot slice {name} shape={shape}")


def _materialize_sliced_field(value: Any) -> np.ndarray:
    """Convert a sliced field descriptor returned above to ``(A,T)`` NumPy."""
    if isinstance(value, tuple) and value and isinstance(value[0], str):
        kind = value[0]
        if kind == "per_agent":
            vals = np.asarray(value[1])
            return np.broadcast_to(vals[:, None], (vals.size, int(value[2])))
        if kind == "per_time":
            vals = np.asarray(value[1])
            return np.broadcast_to(vals[None, :], (int(value[2]), vals.size))
        if kind == "scalar":
            shape = tuple(value[2])
            return np.full(shape, np.asarray(value[1]).reshape(()), dtype=np.asarray(value[1]).dtype)
    return np.asarray(value)


def _splice_waymax_window(log_value: np.ndarray, sim_value: np.ndarray, *, start: int, cut: int) -> np.ndarray:
    """Match ``closed_loop_splice`` for one already-windowed trajectory field."""
    out = np.array(log_value, copy=True)
    if out.shape != sim_value.shape or out.ndim != 2:
        return out
    n_sim = max(0, min(out.shape[1], int(cut) - int(start) + 1))
    if n_sim > 0:
        out[:, :n_sim] = sim_value[:, :n_sim]
    return out


def _gradient_from_extended_window(
    values: np.ndarray,
    *,
    extended_start: int,
    requested_start: int,
    requested_end: int,
    total_steps: int,
    dt: float = 0.1,
) -> np.ndarray:
    """Reproduce ``np.gradient(full_values, dt, axis=-1)`` on a local window."""
    n = max(0, int(requested_end) - int(requested_start))
    if n <= 0:
        return np.zeros((values.shape[0], 0), dtype=np.asarray(values).dtype)
    if int(total_steps) <= 1:
        return np.zeros((values.shape[0], n), dtype=np.asarray(values).dtype)
    gidx = np.arange(int(requested_start), int(requested_end), dtype=np.int64)
    left_global = np.where(gidx <= 0, 0, gidx - 1)
    right_global = np.where(gidx >= int(total_steps) - 1, int(total_steps) - 1, gidx + 1)
    left = left_global - int(extended_start)
    right = right_global - int(extended_start)
    denom = np.where((gidx == 0) | (gidx == int(total_steps) - 1), float(dt), 2.0 * float(dt)).astype(np.float32)
    return (values[:, right] - values[:, left]) / denom[None, :]


def construct_history_from_waymax_state(
    state: Any,
    static_template: RawScenario,
    t: int,
    cfg: dict,
    *,
    scenario_id: str,
    scenario_index: int,
) -> SceneHistory:
    """Construct the exact closed-loop history from a small Waymax time window.

    The legacy online path first copied every log/sim trajectory field for all
    timesteps from accelerator to CPU, spliced the full arrays, built a complete
    ``RawScenario``, and only then retained the 1 s history plus planning future.
    This function performs the same splice and coordinate transforms after
    slicing the required agents/timesteps on device. Static map/route/dynamic-map
    tensors continue to come from ``static_template``.

    It deliberately raises on unfamiliar Waymax layouts so the caller can fall
    back to the legacy, fully general conversion path without changing results.
    """
    sr = float(cfg.get("sample_rate_hz", 10.0))
    H = max(1, int(round(float(cfg.get("history_horizon_s", 1.0)) * sr)))
    total_future = max(2, int(round((float(cfg.get("prefix_horizon_s", 1.0)) + float(cfg.get("recovery_horizon_s", 4.0))) * sr)))
    t = int(t)

    meta = state.object_metadata
    ids = _squeeze_waymax_field(meta.ids)
    try:
        num_agents = int(state.num_objects)
    except Exception:
        num_agents = int(np.prod(tuple(getattr(ids, "shape", (0,)))))
    sdc = int(static_template.sdc_track_index)
    order_list = [sdc] + [i for i in range(num_agents) if i != sdc]
    max_agents = min(int(cfg.get("max_agents", len(order_list))), len(order_list))
    order = np.asarray(order_list[:max_agents], dtype=np.int64)

    log_tr = state.log_trajectory
    sim_tr = state.sim_trajectory
    x_field, total_steps, _ = _waymax_agent_time_shape(log_tr.x, num_agents, name="log_trajectory.x")
    if t < 0 or t >= total_steps:
        raise ValueError(f"Planning timestep {t} is outside Waymax trajectory length {total_steps}")

    hist_start = max(0, t - H + 1)
    requested_start = hist_start
    requested_end = min(total_steps, t + total_future)
    extended_start = max(0, requested_start - 1)
    extended_end = min(total_steps, requested_end + 1)

    field_names = ["x", "y", "z", "vel_x", "vel_y", "yaw", "valid", "length", "width", "height"]
    sliced: dict[str, Any] = {}
    for name in field_names:
        lv = getattr(log_tr, name)
        sv = getattr(sim_tr, name, lv)
        sliced[f"log_{name}"] = _slice_waymax_agent_time(
            lv,
            num_agents=num_agents,
            num_steps=total_steps,
            order=order,
            start=extended_start,
            end=extended_end,
            name=f"log_trajectory.{name}",
        )
        sliced[f"sim_{name}"] = _slice_waymax_agent_time(
            sv,
            num_agents=num_agents,
            num_steps=total_steps,
            order=order,
            start=extended_start,
            end=extended_end,
            name=f"sim_trajectory.{name}",
        )
    sliced["object_type"] = _slice_waymax_agent_time(
        meta.object_types,
        num_agents=num_agents,
        num_steps=total_steps,
        order=order,
        start=extended_start,
        end=extended_end,
        name="object_metadata.object_types",
    )
    sliced = _host_get_tree(sliced)

    fields: dict[str, np.ndarray] = {}
    for name in field_names:
        lv = _materialize_sliced_field(sliced[f"log_{name}"])
        sv = _materialize_sliced_field(sliced[f"sim_{name}"])
        fields[name] = _splice_waymax_window(lv, sv, start=extended_start, cut=t)
    obj_type = _materialize_sliced_field(sliced["object_type"])

    req0 = requested_start - extended_start
    req1 = requested_end - extended_start
    n_req = requested_end - requested_start
    states = np.zeros((n_req, len(order), 16), dtype=np.float32)
    states[..., 0] = fields["x"][:, req0:req1].T
    states[..., 1] = fields["y"][:, req0:req1].T
    states[..., 2] = fields["z"][:, req0:req1].T
    states[..., 3] = fields["vel_x"][:, req0:req1].T
    states[..., 4] = fields["vel_y"][:, req0:req1].T
    states[..., 5] = _gradient_from_extended_window(
        fields["vel_x"],
        extended_start=extended_start,
        requested_start=requested_start,
        requested_end=requested_end,
        total_steps=total_steps,
        dt=0.1,
    ).T
    states[..., 6] = _gradient_from_extended_window(
        fields["vel_y"],
        extended_start=extended_start,
        requested_start=requested_start,
        requested_end=requested_end,
        total_steps=total_steps,
        dt=0.1,
    ).T
    yaw = fields["yaw"][:, req0:req1]
    valid = fields["valid"][:, req0:req1].astype(bool)
    states[..., 7] = yaw.T
    states[..., 8] = np.sin(yaw).T
    states[..., 9] = np.cos(yaw).T
    states[..., 10] = fields["length"][:, req0:req1].T
    states[..., 11] = fields["width"][:, req0:req1].T
    states[..., 12] = fields["height"][:, req0:req1].T
    states[..., 13] = obj_type[:, req0:req1].T
    states[..., 14] = valid.T.astype(np.float32)
    states[..., 15] = valid.T.astype(np.float32)

    hist_end_local = t - requested_start + 1
    hist = states[:hist_end_local]
    hist_valid = valid[:, :hist_end_local].T
    if hist.shape[0] < H:
        pad = np.zeros((H - hist.shape[0], hist.shape[1], hist.shape[2]), dtype=np.float32)
        pad_valid = np.zeros((H - hist_valid.shape[0], hist_valid.shape[1]), dtype=bool)
        hist = np.concatenate([pad, hist], axis=0)
        hist_valid = np.concatenate([pad_valid, hist_valid], axis=0)

    future_start_local = t - requested_start
    future = states[future_start_local:]
    future_valid = valid[:, future_start_local:].T
    if future.shape[0] < total_future:
        pad = np.zeros((total_future - future.shape[0], len(order), states.shape[-1]), dtype=np.float32)
        pad_valid = np.zeros((total_future - future_valid.shape[0], len(order)), dtype=bool)
        future = np.concatenate([future, pad], axis=0)
        future_valid = np.concatenate([future_valid, pad_valid], axis=0)

    ego_raw = states[t - requested_start, 0]
    hist_e = transform_states_to_ego(hist, ego_raw)
    fut_e = transform_states_to_ego(future, ego_raw)
    maps, map_valid, route = transform_map_and_route(static_template, ego_raw)
    route, route_meta = _sanitize_route(route, fut_e, future_valid, cfg)
    dyn = static_template.dynamic_map[max(0, t - H + 1) : t + 1]
    if dyn.shape[0] < H:
        pad = np.zeros((H - dyn.shape[0],) + dyn.shape[1:], dtype=np.float32)
        dyn = np.concatenate([pad, dyn], axis=0)

    h = SceneHistory(
        scene_id=str(scenario_id),
        original_scenario_id=str(scenario_id),
        time_index=int(t),
        agent_history=hist_e.astype(np.float32),
        agent_valid=hist_valid.astype(bool),
        map_polylines=maps.astype(np.float32),
        map_valid=map_valid.astype(bool),
        dynamic_map=dyn.astype(np.float32),
        route=route.astype(np.float32),
        occ_mask=np.zeros((int(cfg.get("bev_channels", 7)), 2, 2), dtype=np.float32),
        ego_state=ego_from_agent_state(hist_e[-1, 0]),
        future_agent_states=fut_e.astype(np.float32),
        future_agent_valid=future_valid.astype(bool),
        metadata={
            "speed_limit": float(cfg.get("speed_limit_default", 13.4)),
            "shoulder_available": True,
            "adjacent_available": True,
            "time_sampling_reasons": [],
            "source": "womd_waymax",
            "agent_order": [int(i) for i in order],
            "ego_global_xy": [float(ego_raw[0]), float(ego_raw[1])],
            "ego_global_heading": float(ego_raw[7]),
            "waymax_planning_timestep": int(t),
            "waymax_sdc_original_index": int(sdc),
            "_waymax_state": state,
            "_waymax_scenario_index": int(scenario_index),
            "_waymax_branch_from_current": True,
            "_fast_waymax_history": True,
            **route_meta,
        },
    )
    h.occ_mask = render_base_occ_mask(h, cfg)
    return h

