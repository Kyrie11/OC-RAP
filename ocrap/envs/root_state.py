from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List
import copy
import random
import numpy as np

from ocrap.utils.seeds import capture_rng_state, restore_rng_state


@dataclass
class RootState:
    ego_state: Any
    actor_states: Any
    traffic_policy_states: Dict[str, Any]
    navigation_state: Dict[str, Any]
    simulator_tick: int
    rng_state: Dict[str, Any]
    sim_rng_state: Dict[str, Any]
    scripted_adversary_states: Dict[str, Any] = field(default_factory=dict)
    map_config: Dict[str, Any] = field(default_factory=dict)
    traffic_config: Dict[str, Any] = field(default_factory=dict)
    history: Any = None
    snapshot_supported: bool = True


class RootStateRestorer:
    """Snapshot restore with deterministic-replay fallback.

    In real MetaDrive runs, users should extend adapter state extraction for their
    exact simulator version.  The generic implementation is deliberately explicit
    about what it can and cannot restore.
    """

    def __init__(self, adapter=None):
        self.adapter = adapter
        self.last_recorded_actions: list[Any] = []

    def save(self, env) -> RootState:
        if hasattr(env, "snapshot"):
            snap = copy.deepcopy(env.snapshot())
            ego = snap.get("ego") if isinstance(snap, dict) else snap
            actors = snap.get("actors", []) if isinstance(snap, dict) else []
            return RootState(ego, actors, snap.get("traffic_policy_states", {}) if isinstance(snap, dict) else {}, snap.get("navigation_state", {}) if isinstance(snap, dict) else {}, getattr(env, "tick", 0), capture_rng_state(), self.adapter.get_sim_rng_state(env) if self.adapter else {}, snapshot_supported=True)
        # Generic Python object fallback for tests and simple env wrappers.
        ego = copy.deepcopy(getattr(env, "ego_state", None))
        actors = copy.deepcopy(getattr(env, "actor_states", []))
        policy = copy.deepcopy(getattr(env, "traffic_policy_states", {}))
        nav = copy.deepcopy(getattr(env, "navigation_state", {}))
        sim_rng = self.adapter.get_sim_rng_state(env) if self.adapter else copy.deepcopy(getattr(env, "sim_rng_state", {}))
        return RootState(ego, actors, policy, nav, int(getattr(env, "tick", 0)), capture_rng_state(), sim_rng, copy.deepcopy(getattr(env, "scripted_adversary_states", {})), copy.deepcopy(getattr(env, "map_config", {})), copy.deepcopy(getattr(env, "traffic_config", {})), copy.deepcopy(getattr(env, "history", None)), snapshot_supported=True)

    def restore(self, env, root_state: RootState) -> None:
        if root_state.snapshot_supported and hasattr(env, "restore"):
            env.restore(copy.deepcopy({
                "ego": root_state.ego_state,
                "actors": root_state.actor_states,
                "traffic_policy_states": root_state.traffic_policy_states,
                "navigation_state": root_state.navigation_state,
            }))
        else:
            setattr(env, "ego_state", copy.deepcopy(root_state.ego_state))
            setattr(env, "actor_states", copy.deepcopy(root_state.actor_states))
            setattr(env, "traffic_policy_states", copy.deepcopy(root_state.traffic_policy_states))
            setattr(env, "navigation_state", copy.deepcopy(root_state.navigation_state))
        setattr(env, "tick", int(root_state.simulator_tick))
        setattr(env, "scripted_adversary_states", copy.deepcopy(root_state.scripted_adversary_states))
        setattr(env, "map_config", copy.deepcopy(root_state.map_config))
        setattr(env, "traffic_config", copy.deepcopy(root_state.traffic_config))
        if root_state.history is not None:
            setattr(env, "history", copy.deepcopy(root_state.history))
        restore_rng_state(root_state.rng_state)
        if self.adapter:
            self.adapter.set_sim_rng_state(env, root_state.sim_rng_state)
        else:
            setattr(env, "sim_rng_state", copy.deepcopy(root_state.sim_rng_state))

    def deterministic_replay(self, env_factory, root_state: RootState, recorded_actions: list[Any]):
        env = env_factory(root_state.map_config, root_state.traffic_config)
        for action in recorded_actions[: root_state.simulator_tick]:
            env.step(action)
        return env
