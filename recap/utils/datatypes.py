from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Literal, Optional, Sequence
import numpy as np

Regime = Literal["normal_high_headroom", "low_headroom", "near_contact", "contact_post_contact"]


@dataclass
class EgoState:
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0
    v: float = 0.0
    a_long: float = 0.0
    yaw_rate: float = 0.0
    steering: float = 0.0
    throttle_brake: float = 0.0
    length: float = 4.7
    width: float = 1.9

    def vector6(self) -> np.ndarray:
        return np.array([self.x, self.y, self.heading, self.v, self.a_long, 0.0], dtype=np.float32)


@dataclass
class ActorState:
    actor_id: str
    x: float
    y: float
    heading: float
    vx: float = 0.0
    vy: float = 0.0
    length: float = 4.7
    width: float = 1.9
    actor_type: str = "vehicle"
    dynamic: bool = True


@dataclass
class MapFeatures:
    drivable_polygons: List[np.ndarray] = field(default_factory=list)
    lane_centerlines: List[np.ndarray] = field(default_factory=list)
    lane_boundaries: List[np.ndarray] = field(default_factory=list)
    static_obstacles: List[np.ndarray] = field(default_factory=list)
    speed_limit_mps: float = 13.9


@dataclass
class RouteInfo:
    waypoints: np.ndarray
    command_ids: Optional[np.ndarray] = None
    speed_limit_mps: float = 13.9

    @staticmethod
    def straight(length: float = 80.0, n: int = 40, speed_limit_mps: float = 13.9) -> "RouteInfo":
        xs = np.linspace(0.0, length, n, dtype=np.float32)
        ys = np.zeros_like(xs)
        headings = np.zeros_like(xs)
        wp = np.stack([xs, ys, headings], axis=-1)
        return RouteInfo(wp, np.zeros(n, dtype=np.int64), speed_limit_mps)


@dataclass
class TrafficControlInfo:
    stop_lines: List[np.ndarray] = field(default_factory=list)
    red_light_stop_lines: List[np.ndarray] = field(default_factory=list)


@dataclass
class BEVSpec:
    H: int = 256
    W: int = 256
    range_x: tuple[float, float] = (-40.0, 40.0)
    range_y: tuple[float, float] = (-40.0, 40.0)
    history_steps: int = 10
    dt: float = 0.2
    mode: str = "compact"
    velocity_scale: float = 20.0
    speed_limit_scale: float = 20.0
    ego_origin: str = "vehicle_center"
    image_forward: str = "row_decrease"
    image_left: str = "col_increase"

    @property
    def resolution_x(self) -> float:
        return (self.range_x[1] - self.range_x[0]) / float(self.H)

    @property
    def resolution_y(self) -> float:
        return (self.range_y[1] - self.range_y[0]) / float(self.W)


@dataclass
class ActionPrefix:
    action_id: int
    valid: bool
    type: str
    states: np.ndarray
    controls: np.ndarray
    params: np.ndarray
    swept_polygons: List[np.ndarray] = field(default_factory=list)
    score_prop: float = 0.0
    mask_reason: str = ""


@dataclass
class RecoveryOption:
    option_id: int
    action_id: int
    type: str
    valid: bool
    horizon_steps: int
    target_anchor: np.ndarray
    target_speed: float
    params: np.ndarray
    states_ref: np.ndarray
    controls_ref: np.ndarray
    swept_corridor: Optional[np.ndarray] = None
    mask_reason: str = ""
    conditional: bool = False


@dataclass
class RootModeSeed:
    mode_id: int
    rng_seed: int
    reaction_delay: float = 0.0
    aggressiveness: float = 0.0
    desired_speed_scale: float = 1.0
    braking_noise: float = 0.0
    lateral_noise: float = 0.0
    occlusion_release_time: float = float("inf")
    hidden_actor_spawn: bool = False
    friction_scale: float = 1.0
    actuation_delay: float = 0.0
    control_noise_std: float = 0.0
    semantic: str = "nominal"

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.reaction_delay,
            self.aggressiveness,
            self.desired_speed_scale,
            self.braking_noise,
            self.lateral_noise,
            10.0 if np.isinf(self.occlusion_release_time) else self.occlusion_release_time,
            float(self.hidden_actor_spawn),
            self.friction_scale,
            self.actuation_delay,
            self.control_noise_std,
        ], dtype=np.float32)


@dataclass
class RootScene:
    root_id: str
    seed: int
    map_config: Dict[str, Any]
    traffic_config: Dict[str, Any]
    regime: Regime
    root_tick: int
    scenario_init_config: Dict[str, Any] = field(default_factory=dict)
    rng_state: Dict[str, Any] = field(default_factory=dict)
    ego_state: EgoState = field(default_factory=EgoState)
    actor_states: List[ActorState] = field(default_factory=list)
    traffic_policy_states: Dict[str, Any] = field(default_factory=dict)
    bev_history: Optional[np.ndarray] = None
    ego_info: Optional[np.ndarray] = None
    route_command: Optional[np.ndarray] = None
    actions: Optional[np.ndarray] = None
    options: Optional[np.ndarray] = None
    labels: Optional[Dict[str, Any]] = None


@dataclass
class RolloutTrace:
    ego_states: np.ndarray
    ego_controls: np.ndarray
    actor_states: Optional[np.ndarray] = None
    stage_boundary_idx: int = 10
    first_contact_idx: int = -1
    first_contact_stage: int = 0  # 0 none, 1 prefix, 2 recovery
    offroad_idx: int = -1
    wrongway_idx: int = -1
    route_departure_idx: int = -1
    secondary_collision_idx: int = -1
    contact_type: str = "none"
    relative_speed_at_first_contact: float = 0.0
    # Optional precomputed normalized margins [T]
    collision_margin: Optional[np.ndarray] = None
    drivable_margin: Optional[np.ndarray] = None
    direction_margin: Optional[np.ndarray] = None
    route_margin: Optional[np.ndarray] = None
    speed_margin: Optional[np.ndarray] = None
    stability_margin: Optional[np.ndarray] = None
    ttc_margin: Optional[np.ndarray] = None
    affordance_costs: Optional[Dict[str, np.ndarray]] = None


def dataclass_to_jsonable(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return {k: dataclass_to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [dataclass_to_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): dataclass_to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj
