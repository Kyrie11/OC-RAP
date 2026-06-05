from __future__ import annotations

import glob
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from .schema import RawScenario


def _require_womd_deps():
    try:
        import tensorflow as tf  # type: ignore
        from waymo_open_dataset.protos import scenario_pb2  # type: ignore
    except Exception as exc:
        raise ImportError(
            "WOMD parsing requires optional dependencies. Install with `pip install -e .[womd]` "
            "and ensure Waymo Open Dataset access is configured."
        ) from exc
    return tf, scenario_pb2


def _feature_points(feature) -> tuple[np.ndarray, int]:
    # Return polyline points [Q, 8] with semantic type id in column 5. Robust to multiple WOMD feature kinds.
    points = []
    typ = 0
    speed_limit = 0.0
    traffic_control = 0.0
    route_flag = 0.0
    for name, code in [("lane", 1), ("road_line", 2), ("road_edge", 3), ("crosswalk", 4), ("speed_bump", 5), ("stop_sign", 6), ("driveway", 7)]:
        try:
            has = feature.HasField(name)
        except Exception:
            has = hasattr(feature, name)
        if has:
            obj = getattr(feature, name)
            typ = code
            if hasattr(obj, "speed_limit_mph"):
                speed_limit = float(obj.speed_limit_mph) * 0.44704
            if hasattr(obj, "interpolating"):
                traffic_control = float(bool(obj.interpolating))
            if hasattr(obj, "polyline"):
                points = obj.polyline
            elif hasattr(obj, "polygon"):
                points = obj.polygon
            elif hasattr(obj, "position"):
                points = [obj.position]
            break
    arr = []
    prev = None
    for p in points:
        x = float(getattr(p, "x", 0.0))
        y = float(getattr(p, "y", 0.0))
        z = float(getattr(p, "z", 0.0))
        if prev is None:
            dx, dy = 0.0, 0.0
        else:
            dx, dy = x - prev[0], y - prev[1]
        arr.append([x, y, z, dx, dy, typ, speed_limit, traffic_control, route_flag, 0.0, 0.0, 0.0])
        prev = (x, y)
    return np.asarray(arr, dtype=np.float32), typ


def _parse_sdc_paths(scenario) -> np.ndarray:
    routes = []
    if hasattr(scenario, "sdc_paths") and len(scenario.sdc_paths) > 0:
        for path in scenario.sdc_paths[:1]:
            pts = getattr(path, "points", [])
            for p in pts:
                routes.append([float(getattr(p, "x", 0.0)), float(getattr(p, "y", 0.0)), float(getattr(p, "z", 0.0)), 1.0])
    if not routes:
        sdc = scenario.tracks[scenario.sdc_track_index]
        for st in sdc.states:
            if getattr(st, "valid", False):
                routes.append([float(st.center_x), float(st.center_y), float(st.center_z), 1.0])
    return np.asarray(routes, dtype=np.float32) if routes else np.zeros((1, 4), dtype=np.float32)


def parse_scenario_proto(scenario, max_agents: int = 128, max_polylines: int = 256, max_points: int = 32) -> RawScenario:
    T = len(scenario.timestamps_seconds)
    # Keep the SDC at index 0 even when the proto's sdc_track_index is larger
    # than max_agents.  The previous first-N truncation could silently drop the
    # SDC and later crash or build samples around the wrong ego vehicle.
    sdc = int(scenario.sdc_track_index)
    all_indices = list(range(len(scenario.tracks)))
    if not (0 <= sdc < len(all_indices)):
        raise ValueError(f"Invalid sdc_track_index={sdc} for {len(all_indices)} tracks")
    selected_indices = [sdc] + [i for i in all_indices if i != sdc]
    selected_indices = selected_indices[: max(1, min(max_agents, len(selected_indices)))]
    A = len(selected_indices)
    states = np.zeros((T, A, 10), dtype=np.float32)
    valid = np.zeros((T, A), dtype=bool)
    object_ids: list[str] = []
    for a, track_idx in enumerate(selected_indices):
        tr = scenario.tracks[track_idx]
        object_ids.append(str(getattr(tr, "id", track_idx)))
        obj_type = float(getattr(tr, "object_type", 0))
        for t, st in enumerate(tr.states[:T]):
            states[t, a] = [
                float(getattr(st, "center_x", 0.0)),
                float(getattr(st, "center_y", 0.0)),
                float(getattr(st, "center_z", 0.0)),
                float(getattr(st, "velocity_x", 0.0)),
                float(getattr(st, "velocity_y", 0.0)),
                float(getattr(st, "heading", 0.0)),
                float(getattr(st, "length", 4.8)),
                float(getattr(st, "width", 2.0)),
                float(getattr(st, "height", 1.5)),
                obj_type,
            ]
            valid[t, a] = bool(getattr(st, "valid", False))
    polylines = np.zeros((max_polylines, max_points, 12), dtype=np.float32)
    map_valid = np.zeros((max_polylines, max_points), dtype=bool)
    for i, feat in enumerate(list(scenario.map_features)[:max_polylines]):
        pts, _ = _feature_points(feat)
        n = min(len(pts), max_points)
        if n > 0:
            polylines[i, :n] = pts[:n]
            map_valid[i, :n] = True
    route = _parse_sdc_paths(scenario)
    dyn = np.zeros((T, 1, 8), dtype=np.float32)
    if hasattr(scenario, "dynamic_map_states"):
        for t, dms in enumerate(scenario.dynamic_map_states[:T]):
            if hasattr(dms, "lane_states") and len(dms.lane_states) > 0:
                lane = dms.lane_states[0]
                dyn[t, 0, 0] = float(getattr(lane, "state", 0))
                dyn[t, 0, 1] = float(getattr(lane, "lane", 0))
    return RawScenario(str(scenario.scenario_id), np.asarray(scenario.timestamps_seconds, dtype=np.float32), 0, states, valid, polylines, map_valid, route, dyn, object_ids)


def iter_womd_tfrecords(patterns: str | list[str], max_scenarios: int | None = None, **kwargs) -> Iterator[RawScenario]:
    tf, scenario_pb2 = _require_womd_deps()
    files: list[str] = []
    if isinstance(patterns, str):
        patterns = [patterns]
    for pat in patterns:
        files.extend(glob.glob(pat))
    if not files:
        raise FileNotFoundError(f"No WOMD TFRecord files matched {patterns}")
    count = 0
    for fname in files:
        ds = tf.data.TFRecordDataset(fname, compression_type="")
        for raw in ds:
            scenario = scenario_pb2.Scenario()
            scenario.ParseFromString(bytes(raw.numpy()))
            yield parse_scenario_proto(scenario, **kwargs)
            count += 1
            if max_scenarios is not None and count >= max_scenarios:
                return


def iter_waymax_scenarios(config_name: str = "WOD_1_1_0_TRAINING", max_scenarios: int | None = None, max_num_objects: int = 128):
    try:
        import dataclasses
        from waymax import config as _config  # type: ignore
        from waymax import dataloader  # type: ignore
    except Exception as exc:
        raise ImportError("Waymax loading requires `pip install -e .[waymax]` and configured WOMD access.") from exc
    base = getattr(_config, config_name)
    cfg = dataclasses.replace(base, max_num_objects=max_num_objects)
    iterator = dataloader.simulator_state_generator(config=cfg)
    count = 0
    for state in iterator:
        yield state
        count += 1
        if max_scenarios is not None and count >= max_scenarios:
            return
