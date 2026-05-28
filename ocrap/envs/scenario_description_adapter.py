from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from ocrap.utils.datatypes import ActorState, EgoState, MapFeatures, RouteInfo


def _as_np(x: Any, dtype=np.float32) -> np.ndarray:
    if x is None:
        return np.asarray([], dtype=dtype)
    try:
        return np.asarray(x, dtype=dtype)
    except Exception:
        return np.asarray([], dtype=dtype)


def _safe_scalar(x: Any, default: float = 0.0) -> float:
    try:
        if isinstance(x, np.ndarray):
            if x.size == 0:
                return default
            return float(x.reshape(-1)[0])
        return float(x)
    except Exception:
        return default


def _state_value(state: Dict[str, Any], names: Iterable[str], t: int, default: Any = None) -> Any:
    for name in names:
        if name not in state:
            continue
        arr = _as_np(state[name])
        if arr.size == 0:
            continue
        try:
            if arr.ndim == 0:
                return arr.item()
            idx = int(np.clip(t, 0, arr.shape[0] - 1)) if arr.shape[0] > 1 else 0
            return arr[idx]
        except Exception:
            return default
    return default


def _state_series(state: Dict[str, Any], names: Iterable[str]) -> Optional[np.ndarray]:
    for name in names:
        if name in state:
            arr = _as_np(state[name])
            if arr.size:
                return arr
    return None


def _valid_at(state: Dict[str, Any], t: int) -> bool:
    v = _state_value(state, ["valid", "valid_mask"], t, True)
    try:
        return bool(np.asarray(v).reshape(-1)[0])
    except Exception:
        return True


def _heading_from_state(state: Dict[str, Any], t: int, vx: float, vy: float) -> float:
    h = _state_value(state, ["heading", "heading_theta", "yaw", "theta"], t, None)
    if h is not None:
        return _safe_scalar(h, 0.0)
    if abs(vx) + abs(vy) > 1e-3:
        return float(math.atan2(vy, vx))
    return 0.0


def _velocity_from_state(state: Dict[str, Any], t: int) -> Tuple[float, float]:
    vel = _state_value(state, ["velocity", "vel", "v"], t, None)
    if vel is not None:
        a = np.asarray(vel, dtype=np.float32).reshape(-1)
        if len(a) >= 2:
            return float(a[0]), float(a[1])
        if len(a) == 1:
            heading = _safe_scalar(_state_value(state, ["heading", "heading_theta", "yaw", "theta"], t, 0.0), 0.0)
            return float(a[0] * math.cos(heading)), float(a[0] * math.sin(heading))
    vx = _state_value(state, ["vx"], t, None)
    vy = _state_value(state, ["vy"], t, None)
    if vx is not None and vy is not None:
        return _safe_scalar(vx), _safe_scalar(vy)
    pos = _state_series(state, ["position", "pos", "center"])
    if pos is not None and pos.ndim >= 2 and pos.shape[0] > 1:
        j0 = max(0, min(t, pos.shape[0] - 1))
        j1 = min(pos.shape[0] - 1, j0 + 1)
        j_1 = max(0, j0 - 1)
        dt = 0.1 * max(j1 - j_1, 1)
        dp = pos[j1, :2] - pos[j_1, :2]
        return float(dp[0] / dt), float(dp[1] / dt)
    return 0.0, 0.0


def _size_from_track(track: Dict[str, Any]) -> Tuple[float, float]:
    md = track.get("metadata", {}) or {}
    state = track.get("state", {}) or {}
    length = md.get("length", state.get("length", 4.7))
    width = md.get("width", state.get("width", 1.9))
    try:
        length = float(np.asarray(length).reshape(-1)[0])
    except Exception:
        length = 4.7
    try:
        width = float(np.asarray(width).reshape(-1)[0])
    except Exception:
        width = 1.9
    return length, width


def _object_type(track: Dict[str, Any]) -> str:
    typ = str(track.get("type", track.get("metadata", {}).get("type", "VEHICLE"))).lower()
    if "ped" in typ:
        return "pedestrian"
    if "cycl" in typ or "bike" in typ:
        return "cyclist"
    return "vehicle"


def _track_to_actor(object_id: str, track: Dict[str, Any], t: int) -> Optional[ActorState]:
    state = track.get("state", {}) or {}
    if not _valid_at(state, t):
        return None
    pos = _state_value(state, ["position", "pos", "center"], t, None)
    if pos is None:
        return None
    p = np.asarray(pos, dtype=np.float32).reshape(-1)
    if len(p) < 2 or not np.all(np.isfinite(p[:2])):
        return None
    vx, vy = _velocity_from_state(state, t)
    heading = _heading_from_state(state, t, vx, vy)
    length, width = _size_from_track(track)
    return ActorState(str(object_id), float(p[0]), float(p[1]), heading, vx, vy, length, width, _object_type(track), True)


def _scenario_dict(scenario: Any) -> Dict[str, Any]:
    if isinstance(scenario, dict):
        return scenario
    if hasattr(scenario, "to_dict"):
        return scenario.to_dict()
    try:
        return dict(scenario)
    except Exception:
        return scenario


def _get_by_flexible_key(d: Dict[Any, Any], key: Any) -> Any:
    if key is None:
        return None
    if key in d:
        return d[key]
    sk = str(key)
    if sk in d:
        return d[sk]
    try:
        ik = int(key)
        if ik in d:
            return d[ik]
    except Exception:
        pass
    return None


def scenario_file_path(dataset_dir: str | Path, scenario_id: str, mapping: Optional[Dict[str, str]] = None) -> Path:
    dataset_dir = Path(dataset_dir)
    rel = "" if mapping is None else mapping.get(scenario_id, "")
    candidates = []
    rel_path = Path(rel) if rel else Path()
    if rel_path.suffix == ".pkl":
        candidates.append(dataset_dir / rel_path)
    candidates.append(dataset_dir / rel_path / scenario_id)
    if not str(scenario_id).endswith(".pkl"):
        candidates.append(dataset_dir / rel_path / f"{scenario_id}.pkl")
    candidates.append(dataset_dir / "sd" / f"{scenario_id}.pkl")
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def _centralize_scenario_description(scenario: Any) -> Dict[str, Any]:
    """Return a ScenarioDescription in the same coordinate frame used by ScenarioEnv.

    MetaDrive's ScenarioEnv data manager loads ScenarioNet files with
    ``read_scenario_data(..., centralize=True)``.  If roots are extracted from
    the raw, uncentralized pickle coordinates, the stored root ego/map can be
    kilometers away from the ego state produced by ``ScenarioEnv.reset()`` even
    though both refer to the same WOMD scenario.  This helper mirrors
    ScenarioEnv's loading convention and keeps root JSON, BEV rasterization, and
    teacher rollouts in one coordinate system.
    """
    try:
        from metadrive.scenario.scenario_description import ScenarioDescription as SD
        scenario = SD.centralize_to_ego_car_initial_position(scenario)
    except Exception:
        # Older/non-MetaDrive test paths may only have plain pickles.  In that
        # case return the scenario as-is; downstream alignment checks will still
        # fail loudly if the coordinates do not match ScenarioEnv.
        pass
    return _scenario_dict(scenario)


def read_scenario_description(path: str | Path, centralize: bool = True) -> Dict[str, Any]:
    try:
        from metadrive.scenario import utils as sd_utils
        try:
            scenario = sd_utils.read_scenario_data(str(path), centralize=centralize)
        except TypeError:
            # Compatibility with older MetaDrive versions whose helper did not
            # expose the centralize keyword.
            scenario = sd_utils.read_scenario_data(str(path))
            if centralize:
                return _centralize_scenario_description(scenario)
        return _scenario_dict(scenario)
    except Exception:
        import pickle
        with open(path, "rb") as f:
            scenario = pickle.load(f)
        return _centralize_scenario_description(scenario) if centralize else _scenario_dict(scenario)


def load_scenarionet_summary(dataset_dir: str | Path):
    dataset_dir = Path(dataset_dir)
    try:
        from metadrive.scenario import utils as sd_utils
        out = sd_utils.read_dataset_summary(str(dataset_dir))
        if isinstance(out, tuple) and len(out) == 3:
            return out
        if isinstance(out, dict):
            mapping = _load_dataset_mapping(dataset_dir)
            return out, list(out.keys()), mapping
    except Exception:
        pass
    import pickle
    summary_path = dataset_dir / "dataset_summary.pkl"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing ScenarioNet dataset_summary.pkl under {dataset_dir}")
    with open(summary_path, "rb") as f:
        summary = pickle.load(f)
    mapping = _load_dataset_mapping(dataset_dir)
    return summary, list(summary.keys()), mapping


def _load_dataset_mapping(dataset_dir: Path) -> Dict[str, str]:
    import pickle
    p = dataset_dir / "dataset_mapping.pkl"
    if not p.exists():
        return {}
    with open(p, "rb") as f:
        m = pickle.load(f)
    return {str(k): str(v) for k, v in dict(m).items()}


def scenario_current_time_index(scenario: Dict[str, Any], summary: Optional[Dict[str, Any]] = None) -> int:
    for obj in (summary or {}, scenario.get("metadata", {}) or {}, scenario):
        if "current_time_index" in obj:
            try:
                return int(obj["current_time_index"])
            except Exception:
                pass
    return 0


def scenario_sdc_id(scenario: Dict[str, Any], summary: Optional[Dict[str, Any]] = None) -> Optional[str]:
    for obj in (summary or {}, scenario.get("metadata", {}) or {}, scenario):
        if "sdc_id" in obj and obj["sdc_id"] is not None:
            return str(obj["sdc_id"])
    return None


def extract_root_state_from_scenario(scenario: Dict[str, Any], t: Optional[int] = None, summary: Optional[Dict[str, Any]] = None):
    scenario = _scenario_dict(scenario)
    tracks = scenario.get("tracks", {}) or {}
    if t is None:
        t = scenario_current_time_index(scenario, summary)
    sdc_id = scenario_sdc_id(scenario, summary)
    if sdc_id is None and tracks:
        sdc_id = str(next(iter(tracks.keys())))
    ego_track = _get_by_flexible_key(tracks, sdc_id) if sdc_id is not None else None
    ego_actor = _track_to_actor(sdc_id or "sdc", ego_track, t) if ego_track else None
    if ego_actor is None:
        ego = EgoState()
    else:
        ego = EgoState(
            ego_actor.x,
            ego_actor.y,
            ego_actor.heading,
            float(math.hypot(ego_actor.vx, ego_actor.vy)),
            0.0,
            0.0,
            0.0,
            0.0,
            ego_actor.length,
            ego_actor.width,
        )
    actors: List[ActorState] = []
    for oid, track in tracks.items():
        if sdc_id is not None and str(oid) == str(sdc_id):
            continue
        actor = _track_to_actor(str(oid), track, t)
        if actor is not None:
            actors.append(actor)
    return ego, actors


def _poly_from_feature(feat: Dict[str, Any]) -> Optional[np.ndarray]:
    for key in ("polygon", "polyline", "position"):
        if key in feat:
            arr = _as_np(feat[key])
            if arr.ndim >= 2 and arr.shape[0] >= 2 and arr.shape[1] >= 2:
                return arr[:, :2].astype(np.float32)
    return None


def extract_map_features_from_scenario(scenario: Dict[str, Any], default_speed_limit_mps: float = 13.9) -> MapFeatures:
    scenario = _scenario_dict(scenario)
    mfs = scenario.get("map_features", {}) or {}
    drivable: List[np.ndarray] = []
    centers: List[np.ndarray] = []
    boundaries: List[np.ndarray] = []
    obstacles: List[np.ndarray] = []
    speeds: List[float] = []
    for _, feat in mfs.items():
        if not isinstance(feat, dict):
            continue
        typ = str(feat.get("type", feat.get("metadata", {}).get("type", ""))).lower()
        poly = _poly_from_feature(feat)
        if poly is None:
            continue
        speed = feat.get("speed_limit_mps", feat.get("speed_limit_mph", feat.get("speed_limit_kph", None)))
        if speed is not None:
            try:
                sp = float(speed)
                if "speed_limit_mph" in feat:
                    sp *= 0.44704
                elif "speed_limit_kph" in feat:
                    sp /= 3.6
                speeds.append(sp)
            except Exception:
                pass
        if "lane" in typ and ("surface" in typ or "street" in typ or "freeway" in typ or "center" in typ):
            centers.append(poly)
            if "polygon" in feat:
                pg = _as_np(feat.get("polygon"))
                if pg.ndim >= 2 and pg.shape[0] >= 3:
                    drivable.append(pg[:, :2].astype(np.float32))
        elif "road" in typ or "drive" in typ or "surface" in typ:
            if poly.shape[0] >= 3:
                drivable.append(poly)
            else:
                boundaries.append(poly)
        elif "boundary" in typ or "line" in typ or "edge" in typ or "solid" in typ or "broken" in typ:
            boundaries.append(poly)
        elif "stop" in typ or "crosswalk" in typ or "speed_bump" in typ:
            obstacles.append(poly)
    if not drivable and centers:
        for c in centers:
            if len(c) >= 2:
                # Conservative corridor polygon for rasterization if ScenarioNet lacks lane polygons.
                n = np.gradient(c, axis=0)
                norm = np.linalg.norm(n, axis=1, keepdims=True) + 1e-6
                tangent = n / norm
                normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=-1)
                left = c + 2.0 * normal
                right = c[::-1] - 2.0 * normal[::-1]
                drivable.append(np.concatenate([left, right], axis=0).astype(np.float32))
    speed_limit = float(np.nanmedian(speeds)) if speeds else default_speed_limit_mps
    return MapFeatures(drivable, centers, boundaries, obstacles, speed_limit)


def extract_route_info_from_scenario(scenario: Dict[str, Any], ego: EgoState, summary: Optional[Dict[str, Any]] = None, t: Optional[int] = None) -> RouteInfo:
    scenario = _scenario_dict(scenario)
    if t is None:
        t = scenario_current_time_index(scenario, summary)
    sdc_id = scenario_sdc_id(scenario, summary)
    tracks = scenario.get("tracks", {}) or {}
    sdc_track = _get_by_flexible_key(tracks, sdc_id) if sdc_id is not None else None
    if sdc_track is not None:
        state = sdc_track.get("state", {}) or {}
        pos = _state_series(state, ["position", "pos", "center"])
        heading = _state_series(state, ["heading", "heading_theta", "yaw", "theta"])
        if pos is not None and pos.ndim >= 2 and pos.shape[0] > t + 1:
            end = min(pos.shape[0], t + 80)
            pts = pos[t:end, :2].astype(np.float32)
            if heading is not None and len(heading) >= end:
                hd = np.asarray(heading[t:end], dtype=np.float32).reshape(-1)
            else:
                d = np.gradient(pts, axis=0)
                hd = np.arctan2(d[:, 1], d[:, 0]).astype(np.float32)
            wp = np.concatenate([pts, hd[:, None]], axis=1)
            return RouteInfo(wp, np.zeros(len(wp), dtype=np.int64), extract_map_features_from_scenario(scenario).speed_limit_mps)
    # Fallback to closest lane centerline ahead of ego.
    mf = extract_map_features_from_scenario(scenario)
    if mf.lane_centerlines:
        cands = sorted(mf.lane_centerlines, key=lambda c: float(np.min(np.linalg.norm(c[:, :2] - np.array([ego.x, ego.y]), axis=1))))
        c = cands[0]
        d = np.gradient(c, axis=0)
        hd = np.arctan2(d[:, 1], d[:, 0]).astype(np.float32)
        return RouteInfo(np.concatenate([c[:, :2], hd[:, None]], axis=1).astype(np.float32), np.zeros(len(c), dtype=np.int64), mf.speed_limit_mps)
    return RouteInfo.straight(speed_limit_mps=mf.speed_limit_mps)


def history_from_scenario(scenario: Dict[str, Any], t: int, history_steps: int, summary: Optional[Dict[str, Any]] = None) -> List[Tuple[EgoState, List[ActorState]]]:
    out = []
    for j in range(max(0, t - history_steps + 1), t + 1):
        out.append(extract_root_state_from_scenario(scenario, j, summary=summary))
    while len(out) < history_steps:
        out.insert(0, out[0] if out else (EgoState(), []))
    return out[-history_steps:]
