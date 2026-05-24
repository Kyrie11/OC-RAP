from __future__ import annotations

import numpy as np
from recap.utils.datatypes import RolloutTrace

DEFAULT_MARGIN_PARAMS = {
    "dt": 0.2,
    "H_p": 10,
    "H_r": 25,
    "T_h_guard": 0.4,
    "grace_contact_steps": 2,
    "eta_A": {"stop": 0.35, "lane": 0.35, "route": 0.35, "escape": 0.35, "stabilize": 0.35},
    "hold_steps": 3,
    "a_accel_max": 4.0,
    "brake_max": 6.0,
    "j_max": 8.0,
}


def _safe_min(arrs, default=1.0):
    vals = []
    for a in arrs:
        if a is not None and len(a):
            vals.append(float(np.nanmin(a)))
    return min(vals) if vals else float(default)


def sustained_min_cost(cost: np.ndarray, indices: range | list[int] | np.ndarray, hold_steps: int) -> float:
    idx = np.asarray(list(indices), dtype=int)
    if len(idx) == 0:
        return float("inf")
    c = np.asarray(cost, dtype=np.float32)[idx]
    if len(c) < hold_steps:
        return float(np.min(c))
    best = float("inf")
    for j in range(0, len(c) - hold_steps + 1):
        best = min(best, float(np.max(c[j : j + hold_steps])))
    return best


def compute_margins(trace: RolloutTrace, params: dict | None = None) -> dict:
    p = {**DEFAULT_MARGIN_PARAMS, **(params or {})}
    H_p = int(trace.stage_boundary_idx)
    H_total = trace.ego_states.shape[0] - 1
    fc = int(trace.first_contact_idx)
    grace = int(p["grace_contact_steps"])

    raw_constraints = [trace.collision_margin, trace.drivable_margin, trace.direction_margin, trace.route_margin, trace.speed_margin, trace.stability_margin, trace.ttc_margin]
    M_path_raw = _safe_min(raw_constraints)

    if fc < 0:
        M_path_pre_no_first_contact = M_path_raw
        M_path_rec = M_path_raw
    else:
        pre = np.arange(0, max(fc, 0), dtype=int)
        post = np.arange(min(H_total + 1, fc + grace), H_total + 1, dtype=int)
        pre_vals = []
        for arr in raw_constraints:
            if arr is not None and len(pre):
                pre_vals.append(arr[pre])
        # After the grace window, do not let the first-contact instant's collision clearance kill R.
        post_rule_vals = []
        for arr in [trace.drivable_margin, trace.direction_margin, trace.route_margin, trace.speed_margin, trace.stability_margin, trace.ttc_margin]:
            if arr is not None and len(post):
                post_rule_vals.append(arr[post])
        M_path_pre_no_first_contact = _safe_min(pre_vals, default=1.0)
        M_path_post_rule = _safe_min(post_rule_vals, default=1.0)
        M_path_rec = min(M_path_pre_no_first_contact, M_path_post_rule)

    controls = trace.ego_controls
    if controls is None or len(controls) == 0:
        M_ctrl = 1.0
    else:
        a = controls[:, 0]
        jerk = np.gradient(a, p["dt"])
        # Keep the control feasibility margin consistent with action projection:
        # acceleration is limited to +4 m/s^2, but braking is allowed up to
        # 6 m/s^2.  The previous symmetric abs(a)/4 rule incorrectly rejected
        # many legitimate recovery-stop options and made labels almost all zero.
        accel_den = max(float(p.get("a_accel_max", 4.0)), 1e-6)
        brake_den = max(float(p.get("brake_max", 6.0)), 1e-6)
        jerk_den = max(float(p.get("j_max", 8.0)), 1e-6)
        reserve_a = np.where(a >= 0.0, 1.0 - a / accel_den, 1.0 - np.abs(a) / brake_den)
        reserve_j = 1.0 - np.abs(jerk) / jerk_den
        M_ctrl = float(np.nanmin([np.nanmin(reserve_a), np.nanmin(reserve_j)]))

    if fc < 0:
        M_secondary = 1.0
    else:
        secondary_collision = trace.secondary_collision_idx >= 0 and trace.secondary_collision_idx > fc + grace
        persistent_offroad = trace.drivable_margin is not None and np.mean(trace.drivable_margin[min(fc + grace, H_total) :] < 0) > 0.5
        uncontrolled_spin = trace.stability_margin is not None and np.mean(trace.stability_margin[min(fc + grace, H_total) :] < -0.2) > 0.5
        M_secondary = min(1.0 - float(secondary_collision), 1.0 - float(persistent_offroad), 1.0 - float(uncontrolled_spin))

    eta_A = p["eta_A"] if isinstance(p["eta_A"], dict) else {k: float(p["eta_A"]) for k in ["stop", "lane", "route", "escape", "stabilize"]}
    hold_steps = int(p["hold_steps"])
    if fc < 0:
        eligible_start = H_p
        eligible_aff = ["stop", "lane", "route", "escape"]
    else:
        eligible_start = min(H_total, fc + grace)
        eligible_aff = ["stop", "route", "stabilize", "escape"]
    eligible_indices = range(eligible_start, H_total + 1)
    affordance_scores = []
    for d in eligible_aff:
        if trace.affordance_costs and d in trace.affordance_costs:
            cost = sustained_min_cost(trace.affordance_costs[d], eligible_indices, hold_steps)
            affordance_scores.append(float(eta_A.get(d, 0.35) - cost))
    M_return = max(affordance_scores) if affordance_scores else -1.0

    if fc < 0:
        M_post = 1.0
        K_star = 0.0
    else:
        final = trace.ego_states[max(0, H_total - int(1.0 / p["dt"])) :, :]
        yaw_rate_proxy = np.abs(np.gradient(final[:, 2], p["dt"])) if len(final) > 1 else np.array([0.0])
        yaw_score = 1.0 - float(np.nanmax(yaw_rate_proxy)) / 0.35
        lat_acc = final[:, 3] ** 2 * np.abs(final[:, 5])
        lat_score = 1.0 - float(np.nanmax(lat_acc)) / 3.0
        safe_stop = float(np.nanmean(final[:, 3]) < 0.5)
        route_rejoin = float(np.nanmean(np.abs(final[:, 1])) < 2.0 and np.nanmean(np.abs(final[:, 2])) < np.deg2rad(20))
        drivable_after = 1.0 if trace.drivable_margin is None else float(np.nanmin(trace.drivable_margin[min(fc + grace, H_total) :]))
        M_post = min(M_secondary, yaw_score, lat_score, max(safe_stop, route_rejoin), drivable_after)
        K_star = 1.0 / (1.0 + np.exp(M_post / 0.25))

    if fc < 0:
        M_option = min(M_path_rec, M_ctrl, M_return)
    else:
        M_option = min(M_path_pre_no_first_contact, M_ctrl, M_secondary, M_return, M_post)

    return {
        "M_path_raw": float(M_path_raw),
        "M_path_rec": float(M_path_rec),
        "M_path_pre_no_first_contact": float(M_path_pre_no_first_contact),
        "M_secondary": float(M_secondary),
        "M_return": float(M_return),
        "M_ctrl": float(M_ctrl),
        "M_post": float(M_post),
        "M_option": float(M_option),
        "Y_option": bool(M_option >= 0.0),
        "K_star": float(K_star),
    }
