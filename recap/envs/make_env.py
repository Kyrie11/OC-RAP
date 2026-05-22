from __future__ import annotations

from typing import Any, Dict


def default_metadrive_config(seed: int = 0, traffic_density: float = 0.15, map_config: str = "S", num_scenarios: int = 1) -> Dict[str, Any]:
    return {
        "use_render": False,
        "manual_control": False,
        "traffic_density": traffic_density,
        "map": map_config,
        "start_seed": seed,
        "num_scenarios": num_scenarios,
        "random_traffic": True,
        "agent_policy": "EnvInputPolicy",
        "crash_vehicle_done": False,
        "crash_object_done": False,
        "out_of_road_done": False,
        "vehicle_config": {"enable_reverse": False},
    }


def make_metadrive_env(config: Dict[str, Any] | None = None):
    cfg = default_metadrive_config()
    if config:
        cfg.update(config)
    try:
        from metadrive.envs.metadrive_env import MetaDriveEnv
    except Exception as e:
        raise RuntimeError(
            "MetaDrive is not installed. Install it for real closed-loop collection, "
            "or run scripts with --synthetic true for unit/smoke tests."
        ) from e
    return MetaDriveEnv(config=cfg)
