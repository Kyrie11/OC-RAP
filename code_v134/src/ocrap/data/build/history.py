from __future__ import annotations

import math

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


def _sanitize_route(
    route: np.ndarray, future_ego: np.ndarray, future_valid: np.ndarray, cfg: dict, *, route_source: str = "unknown"
) -> tuple[np.ndarray, dict]:
    max_points = int(cfg.get("route_points", route.shape[0] if route.size else 80))
    cl_cfg = cfg.get("closed_loop", {}) if isinstance(cfg.get("closed_loop", {}), dict) else {}
    require_observation_legal = bool(cl_cfg.get("require_observation_legal_route", False))
    allow_future_proxy = bool(cl_cfg.get("allow_future_route_proxy", True)) and not require_observation_legal
    meta = {
        "route_sanitized": False,
        "route_projection_distance_m": 0.0,
        "route_source": str(route_source),
        "route_observation_legal": str(route_source) == "womd_v1_3_1_sdc_paths_connectivity_only",
    }
    if require_observation_legal and not meta["route_observation_legal"]:
        raise ValueError(
            f"closed-loop publication route is not observation-legal: source={route_source!r}; "
            "WOMD validation future/logged-SDC proxies are forbidden"
        )
    if route.size and len(route) >= 2:
        try:
            proj = project_to_route(np.zeros(2, dtype=np.float32), route)
            meta["route_projection_distance_m"] = float(proj.distance)
            length = float(np.sum(np.linalg.norm(np.diff(route[:, :2], axis=0), axis=1)))
            if proj.distance <= float(cfg.get("max_route_projection_distance_m", 8.0)) and length >= float(cfg.get("min_route_length_m", 10.0)):
                return route.astype(np.float32), meta
            if require_observation_legal and length >= float(cfg.get("min_route_length_m", 10.0)):
                # A genuine navigation route remains valid even if the simulated
                # policy deviates > max_route_projection_distance_m.  Do not
                # replace it by the logged future simply because the vehicle is
                # off-route.
                meta["route_sanitize_reason"] = "observation_legal_route_retained_despite_projection_distance"
                return route.astype(np.float32), meta
        except Exception:
            if require_observation_legal:
                raise
    if require_observation_legal or not allow_future_proxy:
        raise ValueError(
            "no usable observation-legal route; refusing future_agent_states route fallback in closed-loop evaluation"
        )
    meta["route_sanitized"] = True
    meta["route_sanitize_reason"] = "route_not_near_ego_or_too_short"
    meta["route_source"] = "logged_future_route_proxy"
    meta["route_observation_legal"] = False
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
    route, route_meta = _sanitize_route(
        route, fut_e, future_valid, cfg, route_source=str((raw.metadata or {}).get("route_source", "unknown"))
    )
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
            "source_scenario_index": raw.metadata.get("_waymax_scenario_index", -1),
            "official_scenario_id": raw.metadata.get("official_scenario_id"),
            "legacy_scenario_id": raw.metadata.get("legacy_scenario_id"),
            "scenario_id_source": raw.metadata.get("scenario_id_source", "unknown"),
            "womd_source_role": raw.metadata.get("womd_source_role", "unknown"),
            "womd_source_pattern": raw.metadata.get("womd_source_pattern", ""),
            "waymax_max_num_objects": raw.metadata.get("waymax_max_num_objects", -1),
            **route_meta,
        },
    )
    h.occ_mask = render_base_occ_mask(h, cfg)
    return h


def construct_history_from_waymax_state(
    state,
    static_template: RawScenario,
    t: int,
    cfg: dict,
    *,
    scenario_id: str | None = None,
    scenario_index: int | None = None,
) -> SceneHistory:
    """Construct closed-loop history from the needed Waymax window only.

    The legacy path first exported a full ``T x A x 16`` RawScenario on every
    receding-horizon decision and then immediately sliced a small history/future
    window from it.  For WOMD this repeats host transfers, trajectory splicing,
    acceleration gradients, and allocation for objects/timesteps that the planner
    never reads.

    This fast path preserves ``closed_loop_splice`` exactly for the standard
    Waymax ``(agents, time)`` trajectory layout.  It transfers only selected
    agents and the requested time window (plus one velocity sample on either side
    so ``np.gradient(..., 0.1)`` matches the legacy acceleration).  Unsupported
    layouts fail closed to the reference conversion below.
    """
    from ocrap.data.waymax_loader import _as_np, raw_scenario_from_waymax_state

    sid = str(scenario_id or static_template.scenario_id)
    idx = int(0 if scenario_index is None else scenario_index)
    t = int(t)

    def _legacy() -> SceneHistory:
        raw = raw_scenario_from_waymax_state(
            state,
            sid,
            idx,
            cfg,
            trajectory_mode="closed_loop_splice",
            splice_until=t,
            static_template=static_template,
        )
        return construct_history(raw, t, cfg)

    try:
        log = state.log_trajectory
        sim = state.sim_trajectory
        x_shape = tuple(int(v) for v in getattr(log.x, "shape", ()))
        sim_x_shape = tuple(int(v) for v in getattr(sim.x, "shape", ()))
        if len(x_shape) != 2 or x_shape != sim_x_shape:
            return _legacy()
        A, T = x_shape
        if A <= 0 or T <= 0 or not (0 <= t < T):
            return _legacy()

        sr = float(cfg.get("sample_rate_hz", 10))
        H = max(1, int(round(float(cfg.get("history_horizon_s", 1.0)) * sr)))
        total_future = max(2, int(round((float(cfg.get("prefix_horizon_s", 1.0)) + float(cfg.get("recovery_horizon_s", 4.0))) * sr)))
        sdc = int(static_template.sdc_track_index)
        if not (0 <= sdc < A):
            return _legacy()
        order = [sdc] + [i for i in range(A) if i != sdc]
        order = order[: min(int(cfg.get("max_agents", len(order))), len(order))]
        order_np = np.asarray(order, dtype=np.int32)

        hist_start = max(0, t - H + 1)
        future_last = min(T - 1, t + total_future - 1)
        lo = max(0, hist_start - 1)
        hi = min(T - 1, max(t, future_last) + 1)
        cut = min(t, hi)

        # Form every device slice first, then transfer the pytree once.  A
        # per-field ``device_get`` would replace one large legacy transfer with
        # ~20 tiny synchronisations and can be slower on GPU despite fewer bytes.
        field_names = ("x", "y", "z", "vel_x", "vel_y", "yaw", "valid", "length", "width", "height")
        right_start = max(lo, cut + 1)
        payload = {}
        for name in field_names:
            lv_obj = getattr(log, name)
            sv_obj = getattr(sim, name, None)
            lshape = tuple(int(v) for v in getattr(lv_obj, "shape", ()))
            if lshape != (A, T) or sv_obj is None or tuple(int(v) for v in getattr(sv_obj, "shape", ())) != (A, T):
                raise ValueError(f"unsupported Waymax fast-history field {name} shape={lshape}")
            payload[f"{name}_left"] = sv_obj[order_np, lo : cut + 1] if cut >= lo else np.empty((len(order), 0))
            payload[f"{name}_right"] = lv_obj[order_np, right_start : hi + 1] if right_start <= hi else np.empty((len(order), 0))

        obj_type_obj = state.object_metadata.object_types
        obj_shape = tuple(int(v) for v in getattr(obj_type_obj, "shape", ()))
        if obj_shape == (A,):
            payload["obj_type"] = obj_type_obj[order_np]
        elif obj_shape == (A, T):
            payload["obj_type"] = obj_type_obj[order_np, lo : hi + 1]
        else:
            return _legacy()
        try:
            import jax  # type: ignore
            payload = jax.device_get(payload)
        except Exception:
            payload = {k: np.asarray(v) for k, v in payload.items()}

        def _read_spliced(name: str, *, dtype=None) -> np.ndarray:
            arr = np.concatenate([np.asarray(payload[f"{name}_left"]), np.asarray(payload[f"{name}_right"])], axis=1)
            if arr.shape != (len(order), hi - lo + 1):
                raise ValueError(f"unexpected Waymax fast-history slice for {name}: {arr.shape}")
            return arr.astype(dtype, copy=False) if dtype is not None else arr

        x = _read_spliced("x", dtype=np.float32)
        y = _read_spliced("y", dtype=np.float32)
        z = _read_spliced("z", dtype=np.float32)
        vx = _read_spliced("vel_x", dtype=np.float32)
        vy = _read_spliced("vel_y", dtype=np.float32)
        yaw = _read_spliced("yaw", dtype=np.float32)
        valid = _read_spliced("valid").astype(bool, copy=False)
        length = _read_spliced("length", dtype=np.float32)
        width = _read_spliced("width", dtype=np.float32)
        height = _read_spliced("height", dtype=np.float32)
        if obj_shape == (A,):
            obj_type_sel = np.asarray(payload["obj_type"], dtype=np.float32)
            obj_type = np.broadcast_to(obj_type_sel[:, None], (len(order), hi - lo + 1))
        else:
            obj_type = np.asarray(payload["obj_type"], dtype=np.float32)

        def _accel_at(field: np.ndarray, global_t: int) -> np.ndarray:
            if T <= 1:
                return np.zeros((len(order),), dtype=np.float32)
            k = global_t - lo
            if global_t <= 0:
                return ((field[:, k + 1] - field[:, k]) / 0.1).astype(np.float32, copy=False)
            if global_t >= T - 1:
                return ((field[:, k] - field[:, k - 1]) / 0.1).astype(np.float32, copy=False)
            return ((field[:, k + 1] - field[:, k - 1]) / 0.2).astype(np.float32, copy=False)

        def _global_states(times: list[int]) -> tuple[np.ndarray, np.ndarray]:
            out = np.zeros((len(times), len(order), 16), dtype=np.float32)
            out_valid = np.zeros((len(times), len(order)), dtype=bool)
            for q, gt in enumerate(times):
                if gt < 0 or gt >= T:
                    continue
                k = gt - lo
                if k < 0 or k >= x.shape[1]:
                    raise ValueError(f"fast-history time {gt} outside transferred window [{lo},{hi}]")
                out[q, :, 0] = x[:, k]
                out[q, :, 1] = y[:, k]
                out[q, :, 2] = z[:, k]
                out[q, :, 3] = vx[:, k]
                out[q, :, 4] = vy[:, k]
                out[q, :, 5] = _accel_at(vx, gt)
                out[q, :, 6] = _accel_at(vy, gt)
                out[q, :, 7] = yaw[:, k]
                out[q, :, 8] = np.sin(yaw[:, k])
                out[q, :, 9] = np.cos(yaw[:, k])
                out[q, :, 10] = length[:, k]
                out[q, :, 11] = width[:, k]
                out[q, :, 12] = height[:, k]
                out[q, :, 13] = obj_type[:, k]
                out[q, :, 14] = valid[:, k].astype(np.float32)
                out[q, :, 15] = valid[:, k].astype(np.float32)
                out_valid[q] = valid[:, k]
            return out, out_valid

        hist_times = list(range(hist_start, t + 1))
        hist, hist_valid = _global_states(hist_times)
        if hist.shape[0] < H:
            pad_n = H - hist.shape[0]
            hist = np.concatenate([np.zeros((pad_n, len(order), 16), dtype=np.float32), hist], axis=0)
            hist_valid = np.concatenate([np.zeros((pad_n, len(order)), dtype=bool), hist_valid], axis=0)
        future_times = list(range(t, t + total_future))
        future, future_valid = _global_states(future_times)
        ego_raw = _global_states([t])[0][0, 0]

        hist_e = transform_states_to_ego(hist, ego_raw)
        fut_e = transform_states_to_ego(future, ego_raw)
        maps = static_template.map_polylines.copy().astype(np.float32)
        if maps.size:
            maps[..., :2] = transform_points_to_ego(maps[..., :2], ego_raw[:2], float(ego_raw[7]))
        route = static_template.route.copy().astype(np.float32)
        if route.size:
            route[..., :2] = transform_points_to_ego(route[..., :2], ego_raw[:2], float(ego_raw[7]))
            if route.shape[-1] > 2:
                route[..., 2] = route[..., 2] - float(ego_raw[7])
        route_source = str((static_template.metadata or {}).get("route_source", "static_template_unknown"))
        route, route_meta = _sanitize_route(route, fut_e, future_valid, cfg, route_source=route_source)

        dyn = static_template.dynamic_map[max(0, t - H + 1) : t + 1]
        if dyn.shape[0] < H:
            dyn = np.concatenate([np.zeros((H - dyn.shape[0],) + dyn.shape[1:], dtype=np.float32), dyn], axis=0)

        h = SceneHistory(
            scene_id=sid,
            original_scenario_id=sid,
            time_index=t,
            agent_history=hist_e.astype(np.float32),
            agent_valid=hist_valid.astype(bool),
            map_polylines=maps.astype(np.float32),
            map_valid=static_template.map_valid.copy().astype(bool),
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
                "waymax_planning_timestep": t,
                "waymax_sdc_original_index": sdc,
                "_waymax_state": state,
                "_waymax_scenario_index": idx,
                "source_scenario_index": idx,
                "official_scenario_id": None,
                "legacy_scenario_id": None,
                "scenario_id_source": "unknown",
                "womd_source_role": "unknown",
                "womd_source_pattern": "",
                "waymax_max_num_objects": -1,
                **route_meta,
            },
        )
        h.occ_mask = render_base_occ_mask(h, cfg)
        return h
    except Exception:
        # Exactness beats speed: unusual Waymax layouts or optional-datatype
        # differences use the long-standing full RawScenario conversion.
        return _legacy()

