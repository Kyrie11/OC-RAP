from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from recap.controllers.pure_pursuit_pid import PurePursuitPID
from recap.envs.metadrive_adapter import MetaDriveStateAdapter
from recap.envs.scenario_motion import local_states_to_world, world_states_to_local
from recap.utils.datatypes import ActionPrefix, EgoState, MapFeatures, RecoveryOption, RootModeSeed, RolloutTrace


def _info_bool(info: Dict[str, Any], names: List[str]) -> bool:
    for n in names:
        if bool(info.get(n, False)):
            return True
    return False


def _actor_clearance(ego: EgoState, actors) -> float:
    if not actors:
        return 1.0
    best = 1e9
    for a in actors:
        d = math.hypot(float(a.x - ego.x), float(a.y - ego.y)) - 0.5 * (ego.length + max(a.length, a.width))
        best = min(best, d)
    return float(np.clip(best / 8.0, -1.0, 1.0))


def _point_in_poly(pt: np.ndarray, poly: np.ndarray) -> bool:
    x, y = float(pt[0]), float(pt[1])
    inside = False
    n = len(poly)
    if n < 3:
        return False
    x0, y0 = poly[-1]
    for x1, y1 in poly:
        if ((y1 > y) != (y0 > y)) and (x < (x0 - x1) * (y - y1) / ((y0 - y1) + 1e-9) + x1):
            inside = not inside
        x0, y0 = x1, y1
    return inside


def _drivable_margin(ego: EgoState, map_features) -> float:
    pt = np.array([ego.x, ego.y], dtype=np.float32)
    if not map_features.drivable_polygons:
        return 0.0
    if any(_point_in_poly(pt, np.asarray(p, dtype=np.float32)) for p in map_features.drivable_polygons):
        return 1.0
    best = 1e9
    for p in map_features.drivable_polygons:
        p = np.asarray(p, dtype=np.float32)
        best = min(best, float(np.min(np.linalg.norm(p[:, :2] - pt[None, :], axis=1))))
    return float(np.clip(-best / 4.0, -1.0, 0.0))


def _route_margin(local_state: np.ndarray) -> float:
    return float(np.clip(1.0 - abs(float(local_state[1])) / 4.0, -1.0, 1.0))


def _stability_margin(local_state: np.ndarray) -> float:
    v = max(0.0, float(local_state[3]))
    kappa = abs(float(local_state[5])) if local_state.shape[0] > 5 else 0.0
    lat_acc = v * v * kappa
    return float(np.clip(min(1.0 - lat_acc / 4.0, 1.0 - 0.2 * abs(float(local_state[2]))), -1.0, 1.0))


def _speed_margin(ego: EgoState, speed_limit: float) -> float:
    return float(np.clip(1.0 - max(0.0, ego.v - speed_limit - 2.0) / 5.0, -1.0, 1.0))


def _affordance_costs(local_states: np.ndarray) -> Dict[str, np.ndarray]:
    s = np.asarray(local_states, dtype=np.float32)
    return {
        "stop": np.abs(s[:, 3]) / 10.0 + np.abs(s[:, 1]) / 4.0,
        "lane": np.abs(s[:, 1]) / 2.0 + np.abs(s[:, 2]) / np.pi,
        "route": np.abs(s[:, 1]) / 4.0 + np.maximum(0.0, -s[:, 0]) / 20.0,
        "escape": np.maximum(0.0, 1.5 - np.abs(s[:, 1])) / 1.5,
        "stabilize": np.abs(s[:, 3]) / 10.0 + np.abs(s[:, 2]) / np.pi + np.maximum(0.0, np.abs(s[:, 1]) - 4.0) / 4.0,
    }


def _semantic_context_to_config(mode: RootModeSeed) -> Dict[str, Any]:
    # MetaDrive's public ScenarioEnv config is version dependent and rejects
    # unknown keys.  Do not inject speculative simulator parameters here.
    # Mode-specific perturbations that are version agnostic are applied to the
    # ego reference/control stream inside rollout().
    return {}


def _filter_supported_config(env_cls, cfg: Dict[str, Any]) -> Dict[str, Any]:
    try:
        default = env_cls.default_config()
        supported = set(default.keys())
    except Exception:
        return dict(cfg)
    dropped = sorted(k for k in cfg if k not in supported)
    if dropped:
        print(f"[MetaDriveRolloutRunner] dropping unsupported ScenarioEnv config keys: {dropped}", flush=True)
    return {k: v for k, v in cfg.items() if k in supported}


def _gym_reset(env, seed: int):
    try:
        return env.reset(seed=seed)
    except TypeError:
        try:
            return env.reset()
        except TypeError:
            return None


def _apply_mode_to_reference(ref_local: np.ndarray, mode: RootModeSeed) -> np.ndarray:
    out = np.asarray(ref_local, dtype=np.float32).copy()
    if out.ndim == 2 and out.shape[1] > 3 and np.isfinite(mode.desired_speed_scale):
        out[:, 3] = np.maximum(0.0, out[:, 3] * float(mode.desired_speed_scale))
    if out.ndim == 2 and out.shape[1] > 1 and np.isfinite(mode.lateral_noise) and abs(float(mode.lateral_noise)) > 1e-6:
        # Deterministic root-shared lateral perturbation.  It gives latent modes
        # a physical effect even when installed MetaDrive versions do not expose
        # traffic-policy perturbation knobs through ScenarioEnv config.
        rng = np.random.default_rng(int(mode.rng_seed))
        out[:, 1] += rng.normal(0.0, float(mode.lateral_noise), size=out.shape[0]).astype(np.float32)
    return out


def _apply_mode_to_action(action_md: np.ndarray, mode: RootModeSeed, rng: np.random.Generator) -> np.ndarray:
    out = np.asarray(action_md, dtype=np.float32).copy()
    if np.isfinite(mode.control_noise_std) and float(mode.control_noise_std) > 0.0:
        out += rng.normal(0.0, float(mode.control_noise_std), size=out.shape).astype(np.float32)
    if np.isfinite(mode.braking_noise) and float(mode.braking_noise) > 0.0:
        out[1] = min(float(out[1]), -abs(float(mode.braking_noise)))
    return np.clip(out, -1.0, 1.0).astype(np.float32)


@dataclass
class MetaDriveRolloutRunner:
    scenario_dir: str
    reactive_traffic: bool = True
    use_render: bool = False
    log_level: int = 50
    strict_root_alignment: bool = True
    alignment_tolerance_m: float = 5.0
    controller: PurePursuitPID = field(default_factory=PurePursuitPID)
    adapter: MetaDriveStateAdapter = field(default_factory=lambda: MetaDriveStateAdapter(strict=False))

    def _make_env(self, root_obj: Dict[str, Any], mode: RootModeSeed):
        try:
            from metadrive.envs.scenario_env import ScenarioEnv
        except Exception as exc:
            raise RuntimeError("MetaDrive ScenarioEnv is required for --rollout-backend metadrive") from exc
        try:
            from metadrive.policy.env_input_policy import EnvInputPolicy
            agent_policy = EnvInputPolicy
        except Exception:
            agent_policy = "EnvInputPolicy"
        meta = root_obj.get("scenario_data", {}) or {}
        start_index = int(meta.get("scenario_index", root_obj.get("scenario_index", 0)))
        cfg = {
            "data_directory": str(meta.get("data_directory", self.scenario_dir)),
            "start_scenario_index": start_index,
            "num_scenarios": 1,
            "reactive_traffic": bool(self.reactive_traffic),
            "agent_policy": agent_policy,
            "use_render": bool(self.use_render),
            "crash_vehicle_done": False,
            "crash_object_done": False,
            "crash_human_done": False,
            "out_of_route_done": False,
            "relax_out_of_road_done": True,
            "truncate_as_terminate": False,
            "log_level": int(self.log_level),
        }
        cfg.update(_semantic_context_to_config(mode))
        cfg = _filter_supported_config(ScenarioEnv, cfg)
        env = ScenarioEnv(cfg)
        _gym_reset(env, seed=start_index)
        return env

    def rollout(self, root_obj: Dict[str, Any], root_ego: EgoState, action: ActionPrefix, option: RecoveryOption, mode: RootModeSeed, H_p: int = 10, H_r: int = 25, dt: float = 0.2) -> RolloutTrace:
        ref_local_nominal = np.concatenate([action.states, option.states_ref[1:]], axis=0).astype(np.float32)
        ref_local = _apply_mode_to_reference(ref_local_nominal, mode)
        ref_world = local_states_to_world(root_ego, ref_local)
        ref_controls = np.concatenate([action.controls[:, :3], option.controls_ref[:, :3]], axis=0).astype(np.float32)
        env = self._make_env(root_obj, mode)
        rng = np.random.default_rng(int(mode.rng_seed))
        delayed_actions: List[np.ndarray] = []
        delay_steps = max(0, int(round(float(mode.actuation_delay) / max(float(dt), 1e-6))))
        if self.strict_root_alignment:
            env_ego0 = self.adapter.get_ego_state(env)
            d0 = math.hypot(float(env_ego0.x - root_ego.x), float(env_ego0.y - root_ego.y))
            if d0 > float(self.alignment_tolerance_m):
                rid = root_obj.get("root_id", "unknown")
                tick = (root_obj.get("scenario_data", {}) or {}).get("current_time_index", root_obj.get("root_tick", "unknown"))
                raise RuntimeError(
                    f"MetaDrive reset/root mismatch for {rid}: env ego is {d0:.2f} m from root ego "
                    f"at root tick {tick}. This usually means roots were sampled from arbitrary "
                    "temporal ticks (for example max-samples-per-log > 1) but ScenarioEnv was "
                    "only reset by scenario index. Re-collect paper-final MetaDrive roots with "
                    "--max-samples-per-log 1, or pass --allow-temporal-root-rollout only for "
                    "debug after verifying installed MetaDrive can restore root time."
                )
        states_world: List[np.ndarray] = []
        controls: List[np.ndarray] = []
        collision_margin: List[float] = []
        drivable_margin: List[float] = []
        direction_margin: List[float] = []
        route_margin: List[float] = []
        speed_margin: List[float] = []
        stability_margin: List[float] = []
        ttc_margin: List[float] = []
        first_contact_idx = -1
        secondary_collision_idx = -1
        contact_type = "none"
        rel_speed = 0.0
        try:
            for k in range(ref_world.shape[0]):
                ego = self.adapter.get_ego_state(env)
                actors = self.adapter.get_actor_states(env)
                map_features = self.adapter.get_map_features(env)
                if root_map_features is not None and not map_features.drivable_polygons:
                    # Some MetaDrive versions do not expose current_map internals through the adapter.
                    # Teacher margins should still be evaluated against the ScenarioNet/WOMD map stored in the root,
                    # rather than silently treating drivable area as unknown.
                    map_features = root_map_features
                local = world_states_to_local(root_ego, np.array([[ego.x, ego.y, ego.heading, ego.v, ego.a_long, 0.0]], dtype=np.float32))[0]
                states_world.append(np.array([ego.x, ego.y, ego.heading, ego.v, ego.a_long, local[5]], dtype=np.float32))
                collision_margin.append(_actor_clearance(ego, actors))
                drivable_margin.append(_drivable_margin(ego, map_features))
                direction_margin.append(float(np.clip(1.0 - abs(local[2]) / (np.pi / 3.0), -1.0, 1.0)))
                route_margin.append(_route_margin(local))
                speed_margin.append(_speed_margin(ego, map_features.speed_limit_mps))
                stability_margin.append(_stability_margin(local))
                ttc_margin.append(1.0)
                if k == ref_world.shape[0] - 1:
                    break
                ref_slice = ref_world[k + 1 : min(ref_world.shape[0], k + 10)]
                if len(ref_slice) == 0:
                    ref_slice = ref_world[k : k + 1]
                action_md = self.controller.track(ego, ref_slice, ref_controls[k : k + 10])
                action_md = _apply_mode_to_action(action_md, mode, rng)
                delayed_actions.append(action_md.copy())
                if delay_steps > 0 and len(delayed_actions) > delay_steps:
                    action_to_step = delayed_actions[-delay_steps - 1]
                else:
                    action_to_step = action_md
                controls.append(np.array([float(ref_controls[min(k, len(ref_controls) - 1), 0]), float(ref_controls[min(k, len(ref_controls) - 1), 1]), float(ref_controls[min(k, len(ref_controls) - 1), 2])], dtype=np.float32))
                step_out = env.step(action_to_step)
                if isinstance(step_out, tuple) and len(step_out) == 5:
                    _, _, terminated, truncated, info = step_out
                elif isinstance(step_out, tuple) and len(step_out) == 4:
                    _, _, done, info = step_out
                    terminated, truncated = bool(done), False
                else:
                    terminated, truncated, info = False, False, {}
                info = info or {}
                crashed = _info_bool(info, ["crash_vehicle", "crash_object", "crash_building", "crash", "collision"])
                offroad = _info_bool(info, ["out_of_road", "out_of_route", "out_of_lane"])
                if crashed and first_contact_idx < 0:
                    first_contact_idx = k + 1
                    collision_margin[-1] = -1.0
                    contact_type = "unknown"
                    rel_speed = float(ego.v)
                elif crashed and first_contact_idx >= 0 and secondary_collision_idx < 0 and k + 1 > first_contact_idx + 2:
                    secondary_collision_idx = k + 1
                if offroad:
                    drivable_margin[-1] = min(drivable_margin[-1], -1.0)
        finally:
            try:
                env.close()
            except Exception:
                pass
        states_world_arr = np.stack(states_world).astype(np.float32)
        states_local = world_states_to_local(root_ego, states_world_arr).astype(np.float32)
        if len(controls) < max(0, states_local.shape[0] - 1):
            pad = np.zeros((states_local.shape[0] - 1 - len(controls), 3), dtype=np.float32)
            controls_arr = np.concatenate([np.stack(controls).astype(np.float32) if controls else np.zeros((0, 3), dtype=np.float32), pad], axis=0)
        else:
            controls_arr = np.stack(controls).astype(np.float32)[: max(0, states_local.shape[0] - 1)]
        if states_local.shape[0] > 1:
            states_local[:, 5] = np.gradient(states_local[:, 2], dt) / np.maximum(states_local[:, 3], 1e-3)
            states_local[:, 4] = np.gradient(states_local[:, 3], dt)
        if first_contact_idx < 0:
            stage = 0
        else:
            stage = 1 if first_contact_idx <= H_p else 2
        return RolloutTrace(
            ego_states=states_local,
            ego_controls=controls_arr[:, :2] if controls_arr.shape[1] >= 2 else controls_arr,
            stage_boundary_idx=H_p,
            first_contact_idx=int(first_contact_idx),
            first_contact_stage=stage,
            secondary_collision_idx=int(secondary_collision_idx),
            contact_type=contact_type,
            relative_speed_at_first_contact=float(rel_speed),
            collision_margin=np.asarray(collision_margin[: states_local.shape[0]], dtype=np.float32),
            drivable_margin=np.asarray(drivable_margin[: states_local.shape[0]], dtype=np.float32),
            direction_margin=np.asarray(direction_margin[: states_local.shape[0]], dtype=np.float32),
            route_margin=np.asarray(route_margin[: states_local.shape[0]], dtype=np.float32),
            speed_margin=np.asarray(speed_margin[: states_local.shape[0]], dtype=np.float32),
            stability_margin=np.asarray(stability_margin[: states_local.shape[0]], dtype=np.float32),
            ttc_margin=np.asarray(ttc_margin[: states_local.shape[0]], dtype=np.float32),
            affordance_costs=_affordance_costs(states_local),
        )
