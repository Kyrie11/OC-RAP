from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
import numpy as np

from ocrap.utils.datatypes import EgoState, ActorState, MapFeatures, RouteInfo, TrafficControlInfo


class AdapterUnavailable(RuntimeError):
    pass


class MetaDriveStateAdapter:
    """Centralized access boundary for MetaDrive internals.

    The adapter never silently fabricates privileged fields.  If a field cannot be
    obtained from the installed MetaDrive version, the method returns a documented
    fallback with an `unavailable` flag in the debug metadata or raises
    AdapterUnavailable for state that is essential for teacher labels.
    """

    def __init__(self, strict: bool = False):
        self.strict = strict
        self.unavailable: Dict[str, str] = {}

    def _mark(self, key: str, reason: str):
        self.unavailable[key] = reason
        if self.strict:
            raise AdapterUnavailable(f"MetaDrive adapter field unavailable: {key}: {reason}")

    def get_ego_state(self, env) -> EgoState:
        agent = getattr(env, "agent", None) or getattr(getattr(env, "engine", None), "agent", None)
        if agent is None:
            self._mark("ego_state", "env.agent not found; returning zero ego for debug only")
            return EgoState()
        try:
            pos = np.asarray(agent.position, dtype=float)
            heading = float(getattr(agent, "heading_theta", getattr(agent, "heading", 0.0)))
            v = float(getattr(agent, "speed", 0.0))
            steering = float(getattr(agent, "steering", 0.0))
            return EgoState(x=float(pos[0]), y=float(pos[1]), heading=heading, v=v, steering=steering)
        except Exception as e:
            self._mark("ego_state", repr(e))
            return EgoState()

    def get_actor_states(self, env) -> List[ActorState]:
        actors: List[ActorState] = []
        engine = getattr(env, "engine", None)
        agent = getattr(env, "agent", None) or getattr(engine, "agent", None)
        ego_ids = set()
        if agent is not None:
            for attr in ("id", "name", "agent_id"):
                val = getattr(agent, attr, None)
                if val is not None:
                    ego_ids.add(str(val))
        manager = getattr(engine, "traffic_manager", None)
        objs = []
        try:
            if manager is not None and hasattr(manager, "vehicles"):
                vehicles = manager.vehicles
                if isinstance(vehicles, dict):
                    objs.extend(vehicles.values())
                else:
                    objs.extend(list(vehicles))
            if hasattr(engine, "get_objects"):
                objs.extend(list(engine.get_objects()))
        except Exception:
            pass
        seen = set()
        for j, obj in enumerate(objs):
            oid = str(getattr(obj, "id", f"actor_{j}"))
            # MetaDrive versions differ in what engine.get_objects() returns.
            # Some include the controllable SDC/ego vehicle.  Treating that
            # object as surrounding traffic makes every root start in collision
            # with itself: _actor_clearance() becomes about -4.7/8 = -0.5875,
            # which collapses all teacher recoverability labels to zero.  Filter
            # by object identity and stable id/name fields, but do not use a
            # loose position-only filter that could hide a genuine overlap with
            # another vehicle.
            if agent is not None and obj is agent:
                continue
            if oid in ego_ids:
                continue
            if oid in seen:
                continue
            seen.add(oid)
            try:
                pos = np.asarray(obj.position, dtype=float)
                heading = float(getattr(obj, "heading_theta", getattr(obj, "heading", 0.0)))
                speed = float(getattr(obj, "speed", 0.0))
                vx, vy = speed * np.cos(heading), speed * np.sin(heading)
                actors.append(ActorState(oid, float(pos[0]), float(pos[1]), heading, vx, vy))
            except Exception:
                continue
        if not actors:
            self._mark("actor_states", "traffic actors not found; returning empty list")
        return actors

    def get_map_features(self, env) -> MapFeatures:
        # MetaDrive exposes map/lane internals differently across versions. Until
        # a version-specific extractor is implemented, return an explicit debug
        # fallback instead of an empty MapFeatures object. Teacher rollouts pass
        # ScenarioNet/WOMD root maps separately and prefer them for margins.
        try:
            current_map = getattr(getattr(env, "engine", None), "current_map", None)
            if current_map is None:
                raise AttributeError("engine.current_map missing")
            raise NotImplementedError("current_map extraction is version-specific")
        except Exception as e:
            self._mark("map_features", f"using synthetic straight corridor fallback: {e!r}")
            drivable = np.array([[-80, -8], [120, -8], [120, 8], [-80, 8]], dtype=np.float32)
            center = np.stack([np.linspace(-80, 120, 100), np.zeros(100)], axis=-1).astype(np.float32)
            left = center + np.array([0.0, 1.8], dtype=np.float32)
            right = center + np.array([0.0, -1.8], dtype=np.float32)
            return MapFeatures([drivable], [center], [left, right], [], 13.9)


    def get_navigation_route(self, env) -> RouteInfo:
        try:
            nav = getattr(getattr(env, "agent", None), "navigation", None)
            if nav is not None and hasattr(nav, "checkpoints"):
                pts = []
                for cp in nav.checkpoints:
                    p = np.asarray(cp, dtype=float)
                    pts.append([p[0], p[1], 0.0])
                if len(pts) >= 2:
                    return RouteInfo(np.asarray(pts, dtype=np.float32))
        except Exception:
            pass
        self._mark("route", "navigation route unavailable; using straight route fallback")
        return RouteInfo.straight()

    def get_traffic_light_or_stop_lines(self, env) -> TrafficControlInfo:
        self._mark("traffic_control", "traffic controls unavailable in adapter fallback")
        return TrafficControlInfo()

    def get_sim_rng_state(self, env) -> dict:
        state = {}
        for name in ("np_random", "random", "rng"):
            obj = getattr(env, name, None)
            if obj is not None and hasattr(obj, "bit_generator"):
                state[name] = obj.bit_generator.state
        return state

    def set_sim_rng_state(self, env, state: dict) -> None:
        for name, value in (state or {}).items():
            obj = getattr(env, name, None)
            if obj is not None and hasattr(obj, "bit_generator"):
                obj.bit_generator.state = value
