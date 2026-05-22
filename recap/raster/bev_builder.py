from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

from recap.utils.datatypes import BEVSpec, EgoState, ActorState, MapFeatures, RouteInfo
from recap.envs.metadrive_adapter import MetaDriveStateAdapter
from .geometry import world_to_ego, ego_to_bev_pixel, rasterize_polygon, draw_polyline, oriented_box, rectangle_corners
from .affordance_maps import AffordanceProvider
from .route import route_command_from_route
from .occlusion import rasterize_occlusion_simple

REQUIRED_CHANNELS = [
    "drivable_area", "lane_boundary", "lane_centerline", "route_corridor", "speed_limit_map",
    "traffic_control_stop", "static_obstacle", "occlusion_mask", "free_space_pocket",
    "affordance_stop", "affordance_lane", "affordance_route", "affordance_escape",
    "ego_current", "ego_history_decay", "dyn_occ_t0", "dyn_vx_t0", "dyn_vy_t0",
]


@dataclass
class HistoryBuffer:
    maxlen: int
    ego_history: List[EgoState] = field(default_factory=list)
    actor_history: List[List[ActorState]] = field(default_factory=list)

    def push(self, ego: EgoState, actors: List[ActorState]) -> None:
        self.ego_history.append(ego)
        self.actor_history.append(list(actors))
        self.ego_history = self.ego_history[-self.maxlen :]
        self.actor_history = self.actor_history[-self.maxlen :]


class BEVBuilder:
    def __init__(self, spec: BEVSpec, adapter: Optional[MetaDriveStateAdapter] = None, channel_names: Optional[List[str]] = None):
        self.spec = spec
        self.adapter = adapter or MetaDriveStateAdapter(strict=False)
        self.channel_names = channel_names or self.default_channel_names(spec)
        self.channel_index = {c: i for i, c in enumerate(self.channel_names)}

    @staticmethod
    def default_channel_names(spec: BEVSpec) -> List[str]:
        extra = [f"dyn_occ_hist_{i}" for i in range(max(0, 24 - len(REQUIRED_CHANNELS)))]
        return REQUIRED_CHANNELS + extra

    def build_from_env(self, env, history_buffer: HistoryBuffer, route_info: Optional[RouteInfo] = None, affordance_provider: Optional[AffordanceProvider] = None) -> dict:
        ego = self.adapter.get_ego_state(env)
        actors = self.adapter.get_actor_states(env)
        map_features = self.adapter.get_map_features(env)
        route = route_info or self.adapter.get_navigation_route(env)
        return self.build_from_state(ego, actors, map_features, route, history_buffer, affordance_provider)

    def build_from_state(self, ego: EgoState, actors: List[ActorState], map_features: MapFeatures, route_info: RouteInfo, history_buffer: Optional[HistoryBuffer] = None, affordance_provider: Optional[AffordanceProvider] = None) -> dict:
        if history_buffer is None:
            history_buffer = HistoryBuffer(self.spec.history_steps)
        if len(history_buffer.ego_history) == 0 or history_buffer.ego_history[-1] is not ego:
            history_buffer.push(ego, actors)
        C = len(self.channel_names)
        bev_frame = np.zeros((C, self.spec.H, self.spec.W), dtype=np.float32)
        self.rasterize_static_map(bev_frame, map_features, ego)
        self.rasterize_route(bev_frame, route_info, ego)
        self.rasterize_actors(bev_frame, actors, ego, history_buffer)
        self.rasterize_ego(bev_frame, ego, history_buffer)
        self.rasterize_affordances(bev_frame, affordance_provider or AffordanceProvider(), ego, route_info, map_features)
        # Build history tensor by repeating static/current frame but with true historical dynamic/ego decay already encoded.
        bev = np.stack([bev_frame for _ in range(self.spec.history_steps)], axis=0).astype(np.float16)
        ego_info = self.ego_info(ego, route_info)
        route_command = route_command_from_route(route_info, ego, N_q=20, D_q=6)
        debug = {"channel_names": self.channel_names, "unavailable": getattr(self.adapter, "unavailable", {})}
        return {"bev": bev, "ego_info": ego_info, "route_command": route_command, "debug": debug}

    def _put_channel(self, bev_frame: np.ndarray, name: str, arr: np.ndarray) -> None:
        if name in self.channel_index:
            bev_frame[self.channel_index[name]] = np.maximum(bev_frame[self.channel_index[name]], arr.astype(np.float32))

    def rasterize_static_map(self, bev_frame: np.ndarray, map_features: MapFeatures, ego: EgoState) -> None:
        mask = np.zeros((self.spec.H, self.spec.W), dtype=np.float32)
        for poly in map_features.drivable_polygons:
            poly_ego = world_to_ego(np.asarray(poly, dtype=np.float32), ego)
            pix = ego_to_bev_pixel(poly_ego, self.spec)
            rasterize_polygon(mask, pix, 1.0)
        self._put_channel(bev_frame, "drivable_area", mask)
        boundary = np.zeros_like(mask)
        for line in map_features.lane_boundaries:
            line_ego = world_to_ego(np.asarray(line, dtype=np.float32), ego)
            draw_polyline(boundary, ego_to_bev_pixel(line_ego, self.spec), 1.0, width=2)
        self._put_channel(bev_frame, "lane_boundary", boundary)
        center = np.zeros_like(mask)
        for line in map_features.lane_centerlines:
            line_ego = world_to_ego(np.asarray(line, dtype=np.float32), ego)
            draw_polyline(center, ego_to_bev_pixel(line_ego, self.spec), 1.0, width=2)
        self._put_channel(bev_frame, "lane_centerline", center)
        obs = np.zeros_like(mask)
        for poly in map_features.static_obstacles:
            poly_ego = world_to_ego(np.asarray(poly, dtype=np.float32), ego)
            rasterize_polygon(obs, ego_to_bev_pixel(poly_ego, self.spec), 1.0)
        self._put_channel(bev_frame, "static_obstacle", obs)
        self._put_channel(bev_frame, "speed_limit_map", mask * (map_features.speed_limit_mps / self.spec.speed_limit_scale))
        # Simple free-space pocket is drivable minus route near center.
        self._put_channel(bev_frame, "free_space_pocket", np.clip(mask - center * 0.2, 0, 1))

    def rasterize_route(self, bev_frame: np.ndarray, route_info: RouteInfo, ego: EgoState) -> None:
        route = np.zeros((self.spec.H, self.spec.W), dtype=np.float32)
        wp = np.asarray(route_info.waypoints, dtype=np.float32)
        if len(wp) > 1:
            route_ego = world_to_ego(wp[:, :2], ego)
            draw_polyline(route, ego_to_bev_pixel(route_ego, self.spec), 1.0, width=6)
        self._put_channel(bev_frame, "route_corridor", route)

    def rasterize_actors(self, bev_frame: np.ndarray, actors: List[ActorState], ego: EgoState, history_buffer: HistoryBuffer) -> None:
        occ = np.zeros((self.spec.H, self.spec.W), dtype=np.float32)
        vx = np.zeros_like(occ)
        vy = np.zeros_like(occ)
        for a in actors:
            center_ego = world_to_ego(np.array([[a.x, a.y]], dtype=np.float32), ego)[0]
            heading_ego = a.heading - ego.heading
            box_ego = oriented_box(center_ego, heading_ego, a.length, a.width)
            pix = ego_to_bev_pixel(box_ego, self.spec)
            temp = np.zeros_like(occ)
            rasterize_polygon(temp, pix, 1.0)
            occ = np.maximum(occ, temp)
            # World velocity to ego frame.
            vel_ego = world_to_ego(np.array([[ego.x + a.vx, ego.y + a.vy]], dtype=np.float32), ego)[0]
            vx[temp > 0] = vel_ego[0] / self.spec.velocity_scale
            vy[temp > 0] = vel_ego[1] / self.spec.velocity_scale
        self._put_channel(bev_frame, "dyn_occ_t0", occ)
        self._put_channel(bev_frame, "dyn_vx_t0", vx)
        self._put_channel(bev_frame, "dyn_vy_t0", vy)
        # Dynamic history: older frames have decayed occupancy and are not current copies.
        hist_names = [n for n in self.channel_names if n.startswith("dyn_occ_hist_")]
        for hidx, name in enumerate(hist_names):
            arr = np.zeros_like(occ)
            src_idx = len(history_buffer.actor_history) - 1 - min(hidx + 1, len(history_buffer.actor_history) - 1)
            if src_idx >= 0 and history_buffer.actor_history:
                decay = float(np.exp(-(hidx + 1) / 2.0))
                for a in history_buffer.actor_history[src_idx]:
                    center_ego = world_to_ego(np.array([[a.x, a.y]], dtype=np.float32), ego)[0]
                    box_ego = oriented_box(center_ego, a.heading - ego.heading, a.length, a.width)
                    rasterize_polygon(arr, ego_to_bev_pixel(box_ego, self.spec), decay)
            self._put_channel(bev_frame, name, arr)
        self._put_channel(bev_frame, "occlusion_mask", rasterize_occlusion_simple(self.spec, ego, actors, MapFeatures()))

    def rasterize_ego(self, bev_frame: np.ndarray, ego: EgoState, history_buffer: HistoryBuffer) -> None:
        curr = np.zeros((self.spec.H, self.spec.W), dtype=np.float32)
        box = rectangle_corners(ego.length, ego.width)
        rasterize_polygon(curr, ego_to_bev_pixel(box, self.spec), 1.0)
        self._put_channel(bev_frame, "ego_current", curr)
        hist = np.zeros_like(curr)
        for j, past in enumerate(reversed(history_buffer.ego_history[:-1])):
            decay = float(np.exp(-(j + 1) / 2.0))
            center_ego = world_to_ego(np.array([[past.x, past.y]], dtype=np.float32), ego)[0]
            box_ego = oriented_box(center_ego, past.heading - ego.heading, past.length, past.width)
            rasterize_polygon(hist, ego_to_bev_pixel(box_ego, self.spec), decay)
        self._put_channel(bev_frame, "ego_history_decay", hist)

    def rasterize_affordances(self, bev_frame: np.ndarray, provider: AffordanceProvider, ego: EgoState, route_info: RouteInfo, map_features: MapFeatures) -> None:
        maps = provider.build_maps(self.spec, ego, route_info, map_features)
        self._put_channel(bev_frame, "affordance_stop", maps["stop"])
        self._put_channel(bev_frame, "affordance_lane", maps["lane"])
        self._put_channel(bev_frame, "affordance_route", maps["route"])
        self._put_channel(bev_frame, "affordance_escape", maps["escape"])

    def ego_info(self, ego: EgoState, route_info: RouteInfo) -> np.ndarray:
        speed_limit = route_info.speed_limit_mps
        return np.array([
            ego.v / 20.0,
            ego.a_long / 4.0,
            ego.yaw_rate / 0.7,
            ego.steering,
            ego.throttle_brake,
            0.0,  # lane_heading_error
            0.0,  # lateral_offset
            0.0,  # route_progress
            speed_limit / 20.0,
            ego.steering,
            ego.throttle_brake,
        ], dtype=np.float32)
