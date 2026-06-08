from __future__ import annotations

import functools
import hashlib
import os
from typing import Any, Iterator

import numpy as np

from ocrap.data.schema import RawScenario


def _require_waymax():
    try:
        import jax  # type: ignore
        import jax.numpy as jnp  # type: ignore
        from waymax import config as wx_config  # type: ignore
        from waymax import dataloader as wx_dataloader  # type: ignore
        from waymax.dataloader import womd_factories  # type: ignore
    except Exception as e:  # pragma: no cover - optional dependency path
        raise ImportError(
            "simulation_backend=waymax_closed_loop requires waymax, jax, jaxlib, "
            "tensorflow and WOMD TFExample access. Install the project with the "
            "waymax extra and verify that `python -c 'import waymax, jax'` works."
        ) from e
    return jax, jnp, wx_config, wx_dataloader, womd_factories


def _apply_jax_env(cfg: dict) -> None:
    wx = cfg.get("waymax", {}) if isinstance(cfg.get("waymax", {}), dict) else {}
    platforms = str(wx.get("jax_platforms", "cuda,cpu"))
    os.environ.setdefault("JAX_PLATFORMS", platforms)
    if not bool(wx.get("preallocate_gpu_memory", False)):
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def _as_np(x: Any) -> np.ndarray:
    try:
        import jax  # type: ignore

        return np.asarray(jax.device_get(x))
    except Exception:
        return np.asarray(x)


def _scenario_id_from_payload(payload: dict[str, Any], idx: int, state: Any) -> str:
    sid = payload.get("scenario_id")
    try:
        arr = _as_np(sid)
        if arr.shape == ():
            val = arr.item()
            if isinstance(val, bytes):
                return val.decode("utf-8", errors="ignore") or f"waymax_{idx:08d}"
            return str(val)
    except Exception:
        pass
    ids = _as_np(state.object_metadata.ids).reshape(-1)
    ts = _as_np(state.log_trajectory.timestamp_micros).reshape(-1)
    h = hashlib.sha1(ids.tobytes() + ts[: min(16, ts.size)].tobytes()).hexdigest()[:16]
    return f"waymax_{idx:08d}_{h}"


def _paths_to_waymax_path(patterns: Any) -> str:
    if isinstance(patterns, str):
        return patterns
    if isinstance(patterns, (list, tuple)) and len(patterns) == 1:
        return str(patterns[0])
    if isinstance(patterns, (list, tuple)):
        # Waymax DatasetConfig accepts a string path/pattern.  Keep the common
        # multi-shard glob form by joining only when the caller provided multiple
        # concrete values; TensorFlow's gfile glob can still handle brace/glob
        # syntax in each element on recent TF versions.
        return ",".join(str(x) for x in patterns)
    return str(patterns)


def _make_dataset_config(patterns: Any, cfg: dict):
    _, _, wx_config, _, _ = _require_waymax()
    wx = cfg.get("waymax", {}) if isinstance(cfg.get("waymax", {}), dict) else {}
    path = _paths_to_waymax_path(patterns)
    return wx_config.DatasetConfig(
        path=path,
        data_format=wx_config.DataFormat.TFRECORD,
        repeat=1,
        batch_dims=(),
        shuffle_seed=None,
        deterministic=True,
        include_sdc_paths=bool(wx.get("dataloader_include_sdc_paths", True)),
        aggregate_timesteps=True,
        max_num_rg_points=int(wx.get("max_num_rg_points", 30000)),
        max_num_objects=int(cfg.get("max_agents", 64)),
        num_paths=int(wx.get("num_paths", 45)),
        num_points_per_path=int(wx.get("num_points_per_path", 800)),
        drop_remainder=False,
        batch_by_scenario=True,
    )


def _route_from_sdc_paths(state: Any, max_points: int) -> np.ndarray:
    route = np.zeros((max_points, 6), dtype=np.float32)
    paths = getattr(state, "sdc_paths", None)
    if paths is not None:
        x = _as_np(paths.x)
        y = _as_np(paths.y)
        valid = _as_np(paths.valid).astype(bool)
        on_route = _as_np(paths.on_route).astype(bool)
        if x.ndim >= 2:
            candidates = np.where(on_route.reshape(-1))[0]
            if candidates.size == 0:
                candidates = np.arange(x.shape[-2])
            best = int(candidates[0])
            best_count = -1
            for c in candidates[: min(8, len(candidates))]:
                cnt = int(valid[c].sum())
                if cnt > best_count:
                    best = int(c)
                    best_count = cnt
            pts = np.stack([x[best], y[best]], axis=-1)[valid[best]]
            if len(pts) >= 2:
                idx = np.linspace(0, len(pts) - 1, max_points).round().astype(int)
                pts = pts[idx]
                route[:, :2] = pts[:, :2]
                d = np.diff(route[:, :2], axis=0, append=route[-1:, :2])
                route[:, 2] = np.arctan2(d[:, 1], d[:, 0])
                route[:, 3] = 13.4
                route[:, 5] = 1.0
                return route
    # Fallback to logged SDC path.  This is only a route proxy; diagnose will
    # still expose whether sdc_paths were available for true route metrics.
    meta = state.object_metadata
    sdc_idx = int(np.argmax(_as_np(meta.is_sdc).astype(bool)))
    tr = state.log_trajectory
    valid = _as_np(tr.valid)[sdc_idx].astype(bool)
    xy = np.stack([_as_np(tr.x)[sdc_idx], _as_np(tr.y)[sdc_idx]], axis=-1)[valid]
    if len(xy) < 2:
        xy = np.stack([np.arange(max_points, dtype=np.float32), np.zeros(max_points, dtype=np.float32)], axis=-1)
    idx = np.linspace(0, len(xy) - 1, max_points).round().astype(int)
    route[:, :2] = xy[idx, :2]
    d = np.diff(route[:, :2], axis=0, append=route[-1:, :2])
    route[:, 2] = np.arctan2(d[:, 1], d[:, 0])
    route[:, 3] = 13.4
    route[:, 5] = 1.0
    return route


def _map_from_waymax_roadgraph(state: Any, max_polylines: int, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    F = 10
    arr = np.zeros((max_polylines, max_points, F), dtype=np.float32)
    valid_out = np.zeros((max_polylines, max_points), dtype=bool)
    rg = getattr(state, "roadgraph_points", None)
    if rg is None:
        return arr, valid_out
    x = _as_np(rg.x).reshape(-1)
    y = _as_np(rg.y).reshape(-1)
    z = _as_np(rg.z).reshape(-1)
    dx = _as_np(rg.dir_x).reshape(-1)
    dy = _as_np(rg.dir_y).reshape(-1)
    typ = _as_np(rg.types).reshape(-1)
    val = _as_np(rg.valid).reshape(-1).astype(bool)
    ids = _as_np(rg.ids).reshape(-1)
    keep = np.where(val)[0]
    if keep.size == 0:
        return arr, valid_out
    # Preserve roadgraph feature identity where possible; fall back to chunks.
    groups: list[np.ndarray] = []
    for gid in np.unique(ids[keep])[:max_polylines]:
        idx = keep[ids[keep] == gid]
        if idx.size:
            groups.append(idx[:max_points])
        if len(groups) >= max_polylines:
            break
    if not groups:
        groups = [keep[i : i + max_points] for i in range(0, min(keep.size, max_polylines * max_points), max_points)]
    for p, idx in enumerate(groups[:max_polylines]):
        n = min(max_points, len(idx))
        ii = idx[:n]
        arr[p, :n, 0] = x[ii]
        arr[p, :n, 1] = y[ii]
        arr[p, :n, 2] = z[ii]
        arr[p, :n, 3] = dx[ii]
        arr[p, :n, 4] = dy[ii]
        arr[p, :n, 5] = typ[ii]
        arr[p, :n, 9] = 1.0
        valid_out[p, :n] = True
    return arr, valid_out


def raw_scenario_from_waymax_state(state: Any, scenario_id: str, scenario_index: int, cfg: dict) -> RawScenario:
    tr = state.log_trajectory
    meta = state.object_metadata
    x = _as_np(tr.x)
    y = _as_np(tr.y)
    z = _as_np(tr.z)
    vx = _as_np(tr.vel_x)
    vy = _as_np(tr.vel_y)
    yaw = _as_np(tr.yaw)
    valid = _as_np(tr.valid).astype(bool)
    length = _as_np(tr.length)
    width = _as_np(tr.width)
    height = _as_np(tr.height)
    obj_type = _as_np(meta.object_types)
    T = x.shape[-1]
    A = x.shape[0]
    states = np.zeros((T, A, 16), dtype=np.float32)
    states[..., 0] = x.T
    states[..., 1] = y.T
    states[..., 2] = z.T
    states[..., 3] = vx.T
    states[..., 4] = vy.T
    ax = np.gradient(vx, 0.1, axis=-1) if T > 1 else np.zeros_like(vx)
    ay = np.gradient(vy, 0.1, axis=-1) if T > 1 else np.zeros_like(vy)
    states[..., 5] = ax.T
    states[..., 6] = ay.T
    states[..., 7] = yaw.T
    states[..., 8] = np.sin(yaw).T
    states[..., 9] = np.cos(yaw).T
    states[..., 10] = np.broadcast_to(length[:, None], (A, T)).T
    states[..., 11] = np.broadcast_to(width[:, None], (A, T)).T
    states[..., 12] = np.broadcast_to(height[:, None], (A, T)).T
    states[..., 13] = np.broadcast_to(obj_type[:, None], (A, T)).T
    states[..., 14] = valid.T.astype(np.float32)
    states[..., 15] = valid.T.astype(np.float32)
    timestamps = _as_np(tr.timestamp_micros)
    if timestamps.ndim == 2 and timestamps.shape[0] > 0:
        timestamps = timestamps[0]
    timestamps_s = timestamps.astype(np.float64) * 1e-6 if timestamps.size else np.arange(T, dtype=np.float32) * 0.1
    maps, map_valid = _map_from_waymax_roadgraph(state, int(cfg.get("max_map_polylines", 256)), int(cfg.get("max_polyline_points", 64)))
    route = _route_from_sdc_paths(state, int(cfg.get("route_points", 80)))
    dyn = np.zeros((T, int(cfg.get("max_dynamic_signals", 16)), 8), dtype=np.float32)
    sdc_idx = int(np.argmax(_as_np(meta.is_sdc).astype(bool)))
    object_ids = [str(int(v)) for v in _as_np(meta.ids).reshape(-1)]
    return RawScenario(
        scenario_id=scenario_id,
        timestamps=timestamps_s[:T].astype(np.float32),
        sdc_track_index=sdc_idx,
        agent_states=states,
        agent_valid=valid.T,
        map_polylines=maps,
        map_valid=map_valid,
        route=route,
        dynamic_map=dyn,
        object_ids=object_ids,
        metadata={
            "source": "womd_waymax",
            "original_scenario_id": scenario_id,
            "_waymax_state": state,
            "_waymax_scenario_index": int(scenario_index),
            "waymax_sdc_paths_available": getattr(state, "sdc_paths", None) is not None,
        },
    )


def iter_waymax_womd_scenarios(patterns: Any, max_scenarios: int | None, parser_cfg: dict | None = None) -> Iterator[RawScenario]:
    cfg = parser_cfg or {}
    _apply_jax_env(cfg)
    _, _, _, wx_dataloader, womd_factories = _require_waymax()
    dataset_cfg = _make_dataset_config(patterns, cfg)

    def _postprocess(example):
        state = womd_factories.simulator_state_from_womd_dict(
            example,
            include_sdc_paths=bool((cfg.get("waymax", {}) or {}).get("dataloader_include_sdc_paths", True)),
        )
        return {"state": state, "scenario_id": example.get("scenario/id")}

    parse = functools.partial(wx_dataloader.preprocess_serialized_womd_data, config=dataset_cfg)
    gen = wx_dataloader.get_data_generator(dataset_cfg, parse, _postprocess)
    for i, payload in enumerate(gen):
        if max_scenarios is not None and i >= int(max_scenarios):
            break
        state = payload["state"] if isinstance(payload, dict) else payload
        sid = _scenario_id_from_payload(payload if isinstance(payload, dict) else {}, i, state)
        yield raw_scenario_from_waymax_state(state, sid, i, cfg)
