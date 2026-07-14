from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ExternalSelection:
    selected_index: int
    reason: str
    admitted: np.ndarray
    score: np.ndarray
    selected_option: int | None = None


def _scalar(d: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(np.asarray(d.get(key, default)).item())
    except Exception:
        return float(default)


def _best(score: np.ndarray, mask: np.ndarray | None = None) -> int:
    score = np.asarray(score, dtype=float)
    if score.size == 0:
        return 0
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.any():
            idxs = np.where(m)[0]
            return int(idxs[np.argmax(score[idxs])])
    return int(np.argmax(score))


def _valid_root_weights(d: dict[str, Any], K: int) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(d.get("root_probs", np.ones((K,), dtype=np.float32) / max(K, 1)), dtype=float).reshape(-1)[:K]
    if p.size < K:
        p = np.pad(p, (0, K - p.size))
    valid = np.asarray(d.get("root_valid", np.ones((K,), dtype=bool)), dtype=bool).reshape(-1)[:K]
    if valid.size < K:
        valid = np.pad(valid, (0, K - valid.size), constant_values=False)
    p = np.where(valid, np.clip(p, 0.0, None), 0.0)
    den = float(p.sum())
    return (p / den if den > 1e-8 else np.zeros(K, dtype=float)), valid


def _option_valid(d: dict[str, Any], L: int) -> np.ndarray:
    v = np.asarray(d.get("option_valid", np.ones((L,), dtype=bool)), dtype=bool).reshape(-1)[:L]
    if v.size < L:
        v = np.pad(v, (0, L - v.size), constant_values=False)
    return v


def _weighted_lower_cvar(values: np.ndarray, weights: np.ndarray, alpha: float) -> float:
    values = np.asarray(values, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not mask.any():
        return 0.0
    values, weights = values[mask], weights[mask]
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    alpha = float(np.clip(alpha, 1e-4, 1.0))
    acc = 0.0
    total = 0.0
    for v, w in zip(values, weights):
        take = min(float(w), alpha - total)
        if take <= 0:
            break
        acc += float(v) * take
        total += take
    return float(acc / max(total, 1e-8))


def _branchwise_values(d: dict[str, Any], alpha: float = 0.2) -> dict[str, Any]:
    M = np.asarray(d.get("m_star", np.zeros((0, 0))), dtype=float)
    if M.ndim != 2 or M.size == 0:
        return {"expected": 0.0, "cvar": 0.0, "worst": 0.0, "fail_prob": 1.0, "best_options": np.zeros((0,), dtype=int), "best_margins": np.zeros((0,), dtype=float)}
    K, L = M.shape
    opt_valid = _option_valid(d, L)
    masked = np.where(opt_valid[None, :], M, -1.0e9)
    best_options = np.argmax(masked, axis=1).astype(int)
    best_margins = masked[np.arange(K), best_options]
    w, valid = _valid_root_weights(d, K)
    best_margins = np.where(valid & np.isfinite(best_margins), best_margins, -1.0e9)
    expected = float(np.sum(w * np.clip(best_margins, -5.0, 5.0)))
    cvar = _weighted_lower_cvar(np.clip(best_margins, -5.0, 5.0), w, alpha=float(alpha))
    worst = float(np.min(np.clip(best_margins[valid], -5.0, 5.0))) if valid.any() else 0.0
    fail_prob = float(np.sum(w * (best_margins < 0.0))) if w.size else 1.0
    return {"expected": expected, "cvar": cvar, "worst": worst, "fail_prob": fail_prob, "best_options": best_options, "best_margins": best_margins}


def _shared_option_success_score(d: dict[str, Any], gamma: float = 0.0) -> tuple[int, float]:
    M = np.asarray(d.get("m_star", np.zeros((0, 0))), dtype=float)
    if M.ndim != 2 or M.size == 0:
        return 0, 0.0
    K, L = M.shape
    w, valid = _valid_root_weights(d, K)
    opt_valid = _option_valid(d, L)
    success = ((M >= float(gamma)) & valid[:, None] & opt_valid[None, :]).astype(float)
    mass = (success * w[:, None]).sum(axis=0)
    score = np.where(opt_valid, mass, -1.0e9)
    idx = int(np.argmax(score)) if score.size else 0
    return idx, float(max(score[idx], 0.0)) if score.size else 0.0


def _control_proxy(d: dict[str, Any]) -> tuple[float, float]:
    ctrl = np.asarray(d.get("prefix_controls", np.zeros((0, 0))), dtype=float)
    if ctrl.ndim != 2 or ctrl.size == 0:
        return 0.0, 0.0
    accel = float(np.nanmax(np.abs(ctrl[:, 0]))) if ctrl.shape[1] >= 1 else 0.0
    steer = float(np.nanmax(np.abs(ctrl[:, 1]))) if ctrl.shape[1] >= 2 else 0.0
    return accel, steer


def _postimpact_mpc_cost(d: dict[str, Any], cfg: dict[str, Any]) -> tuple[float, dict[str, float]]:
    """Planning-integrated post-impact MPC surrogate over OC-RAP candidates.

    The paper controller optimizes stability recovery and secondary-collision
    avoidance with vehicle-dynamics/road-adhesion constraints.  Here the action
    space is the already-built prefix/recovery lattice, so MPC is solved by
    evaluating that finite horizon lattice with the same ingredients: yaw-rate
    damping, acceleration/steer effort, terminal stable-stop, obstacle/hard
    violation, and route rejoin utility.
    """
    pcfg = ((cfg.get("external_baselines", {}) or {}).get("policy", {}) or {})
    states = np.asarray(d.get("prefix_states", np.zeros((0, 0))), dtype=float)
    controls = np.asarray(d.get("prefix_controls", np.zeros((0, 0))), dtype=float)
    hard = _scalar(d, "hard_violation", 0.0)
    harm = _scalar(d, "harm_proxy", 0.0)
    utility = _scalar(d, "utility", 0.0)
    shared_opt, shared = _shared_option_success_score(d, gamma=float(pcfg.get("postimpact_gamma", 0.0)))
    dt = float(pcfg.get("postimpact_dt", 0.1))
    yaw_rate = yaw_acc = speed_terminal = accel_effort = steer_effort = jerk = 0.0
    if states.ndim == 2 and states.shape[0] >= 2:
        if states.shape[1] >= 3:
            yaw = np.unwrap(states[:, 2])
            yr = np.diff(yaw) / max(dt, 1e-3)
            yaw_rate = float(np.nanmax(np.abs(yr))) if yr.size else 0.0
            yaw_acc = float(np.nanmax(np.abs(np.diff(yr) / max(dt, 1e-3)))) if yr.size >= 2 else 0.0
        if states.shape[1] >= 5:
            speed = np.hypot(states[:, 3], states[:, 4])
        elif states.shape[1] >= 4:
            speed = np.abs(states[:, 3])
        else:
            speed = np.zeros((states.shape[0],), dtype=float)
        speed_terminal = float(speed[-1]) if speed.size else 0.0
    if controls.ndim == 2 and controls.size:
        if controls.shape[1] >= 1:
            a = controls[:, 0]
            accel_effort = float(np.nanmean(np.abs(a)))
            jerk = float(np.nanmax(np.abs(np.diff(a) / max(dt, 1e-3)))) if a.size >= 2 else 0.0
        if controls.shape[1] >= 2:
            steer_effort = float(np.nanmean(np.abs(controls[:, 1])))
    stable_stop_cost = (
        float(pcfg.get("postimpact_yaw_rate_weight", 1.4)) * yaw_rate
        + float(pcfg.get("postimpact_yaw_acc_weight", 0.15)) * yaw_acc
        + float(pcfg.get("postimpact_terminal_speed_weight", 0.45)) * speed_terminal
        + float(pcfg.get("postimpact_accel_weight", 0.08)) * accel_effort
        + float(pcfg.get("postimpact_steer_weight", 0.08)) * steer_effort
        + float(pcfg.get("postimpact_jerk_weight", 0.02)) * jerk
    )
    obstacle_cost = float(pcfg.get("postimpact_hard_weight", 8.0)) * hard + float(pcfg.get("postimpact_harm_weight", 2.5)) * harm
    rejoin_reward = float(pcfg.get("postimpact_rejoin_weight", 0.20)) * utility + float(pcfg.get("postimpact_shared_drs_weight", 3.0)) * shared
    total = stable_stop_cost + obstacle_cost - rejoin_reward
    return float(total), {
        "shared_option": float(shared_opt),
        "shared_success": float(shared),
        "yaw_rate": yaw_rate,
        "yaw_acc": yaw_acc,
        "terminal_speed": speed_terminal,
        "stable_stop_cost": float(stable_stop_cost),
        "obstacle_cost": float(obstacle_cost),
        "rejoin_reward": float(rejoin_reward),
    }


def select_external_policy(
    baseline: str,
    samples: list[dict[str, Any]],
    cfg: dict[str, Any] | None = None,
    *,
    model_outputs: dict[str, np.ndarray] | None = None,
) -> ExternalSelection:
    cfg = cfg or {}
    bcfg = cfg.get("external_baselines", {}) if isinstance(cfg.get("external_baselines", {}), dict) else {}
    pcfg = bcfg.get("policy", {}) if isinstance(bcfg.get("policy", {}), dict) else {}
    baseline = str(baseline).lower()
    n = len(samples)
    utility = np.asarray([_scalar(d, "utility", 0.0) for d in samples], dtype=float)
    hard = np.asarray([_scalar(d, "hard_violation", 0.0) for d in samples], dtype=float)
    harm = np.asarray([_scalar(d, "harm_proxy", 0.0) for d in samples], dtype=float)
    feasible = np.asarray([_scalar(d, "feasible", 1.0) > 0.5 for d in samples], dtype=bool)
    r_orc = np.asarray([_scalar(d, "r_orc_star", 0.0) for d in samples], dtype=float)
    r_dep = np.asarray([_scalar(d, "r_dep_star", 0.0) for d in samples], dtype=float)
    safe = feasible & (hard <= float(pcfg.get("gamma_H", 0.0))) & (harm <= float(pcfg.get("gamma_D", 5.0)))

    if baseline in {"nominal", "nominal_replay", "log_replay"}:
        admitted = np.zeros(n, dtype=bool)
        score = utility.copy()
        nominal = [i for i, d in enumerate(samples) if _scalar(d, "is_nominal", 0.0) > 0.5]
        idx = int(nominal[0] if nominal else 0)
        if not feasible[idx]:
            idx = _best(score, feasible)
        admitted[idx] = True
        return ExternalSelection(idx, "logged_nominal_replay", admitted, score)

    if baseline in {"route_bc", "route_bc_lite", "waymax_bc", "waymax_bc_lite", "wayformer_bc", "wayformer_style_bc", "route_bc_wayformer"}:
        admitted = np.zeros(n, dtype=bool)
        if model_outputs and "logits" in model_outputs:
            score = np.asarray(model_outputs["logits"], dtype=float)[:n]
            idx = _best(score, feasible)
            reason = "learned_route_conditioned_wayformer_bc"
        else:
            score = utility.copy()
            nominal = [i for i, d in enumerate(samples) if _scalar(d, "is_nominal", 0.0) > 0.5]
            idx = int(nominal[0] if nominal else 0)
            if not feasible[idx]:
                idx = _best(score, feasible)
            reason = "log_replay_or_nominal_bc"
        admitted[idx] = True
        return ExternalSelection(idx, reason, admitted, score)

    if baseline in {"gameformer", "gameformer_lite", "gameformer_levelk"}:
        if model_outputs:
            u = np.asarray(model_outputs.get("utility", utility), dtype=float)[:n]
            h = np.maximum(0.0, np.asarray(model_outputs.get("hard", hard), dtype=float)[:n])
            hp = np.maximum(0.0, np.asarray(model_outputs.get("harm", harm), dtype=float)[:n])
            ro = np.asarray(model_outputs.get("r_orc", r_orc), dtype=float)[:n]
            score = u - float(pcfg.get("gameformer_hard_weight", 12.0)) * h - float(pcfg.get("gameformer_harm_weight", 2.0)) * hp + float(pcfg.get("gameformer_oracle_weight", 1.0)) * ro
            reason = "learned_interaction_oracle_risk"
        else:
            score = utility - float(pcfg.get("gameformer_hard_weight", 12.0)) * hard - float(pcfg.get("gameformer_harm_weight", 2.0)) * harm + float(pcfg.get("gameformer_oracle_weight", 1.0)) * r_orc
            reason = "teacher_interaction_oracle_risk"
        idx = _best(score, feasible)
        admitted = safe & (r_orc >= float(pcfg.get("gamma_oracle_rec", 0.0)))
        return ExternalSelection(idx, reason, admitted, score)

    if baseline in {"betop", "betop_lite", "betopnet", "betopnet_lite"}:
        if model_outputs:
            logits = np.asarray(model_outputs.get("logits", utility), dtype=float)[:n]
            u = np.asarray(model_outputs.get("utility", utility), dtype=float)[:n]
            h = np.maximum(0.0, np.asarray(model_outputs.get("hard", hard), dtype=float)[:n])
            hp = np.maximum(0.0, np.asarray(model_outputs.get("harm", harm), dtype=float)[:n])
            topo_logits = model_outputs.get("actor_topo_logits", model_outputs.get("topology_logits"))
            if topo_logits is not None:
                topo = np.asarray(topo_logits, dtype=float)[:n]
                # BeTop predicts binary behavioral-topology occupancies.  Use a
                # topology confidence/interaction prior from the strongest actor
                # relation, rather than a deployability certificate.
                if topo.shape[-1] == 1:
                    prob = 1.0 / (1.0 + np.exp(-np.clip(topo[..., 0], -40.0, 40.0)))
                    topo_conf = prob.max(axis=-1)
                else:
                    topo_prob = np.exp(topo - np.max(topo, axis=-1, keepdims=True))
                    topo_prob = topo_prob / np.maximum(topo_prob.sum(axis=-1, keepdims=True), 1e-9)
                    topo_conf = topo_prob.max(axis=-1).mean(axis=-1)
            else:
                topo_conf = np.ones(n, dtype=float)
            score = logits + float(pcfg.get("betop_utility_weight", 0.25)) * u - float(pcfg.get("betop_hard_weight", 10.0)) * h - float(pcfg.get("betop_harm_weight", 1.5)) * hp + float(pcfg.get("betop_topology_conf_weight", 0.25)) * topo_conf
            reason = "learned_behavioral_topology_planner"
        else:
            branch_tmp = [_branchwise_values(d, alpha=float(pcfg.get("cvar_alpha", 0.2))) for d in samples]
            branch_expected_tmp = np.asarray([b["expected"] for b in branch_tmp], dtype=float)
            score = utility + float(pcfg.get("betop_branch_weight", 0.5)) * branch_expected_tmp - float(pcfg.get("betop_hard_weight", 10.0)) * hard - float(pcfg.get("betop_harm_weight", 1.5)) * harm
            reason = "teacher_behavioral_topology_surrogate"
        idx = _best(score, feasible)
        admitted = safe
        return ExternalSelection(idx, reason, admitted, score)

    branch = [_branchwise_values(d, alpha=float(pcfg.get("cvar_alpha", 0.2))) for d in samples]
    branch_expected = np.asarray([b["expected"] for b in branch], dtype=float)
    branch_cvar = np.asarray([b["cvar"] for b in branch], dtype=float)
    branch_worst = np.asarray([b["worst"] for b in branch], dtype=float)
    branch_fail = np.asarray([b["fail_prob"] for b in branch], dtype=float)

    if baseline in {"marc", "marc_lite"}:
        risk_tol = float(pcfg.get("marc_risk_tolerance", 0.35))
        score = utility + float(pcfg.get("marc_branch_rec_weight", 1.0)) * branch_expected - float(pcfg.get("marc_hard_weight", 10.0)) * hard - float(pcfg.get("marc_harm_weight", 1.5)) * harm - risk_tol * branch_fail
        admitted = safe & (branch_expected >= float(pcfg.get("gamma_branch_rec", 0.0)))
        idx = _best(score, feasible)
        return ExternalSelection(idx, "multipolicy_branchwise_contingency", admitted, score)

    if baseline in {"racp", "racp_lite"}:
        rho = float(pcfg.get("racp_risk_tolerance", 0.6))
        # RACP-lite trades expected branch value and lower-tail branch value.
        risk_value = rho * branch_expected + (1.0 - rho) * branch_cvar
        score = utility + float(pcfg.get("racp_branch_rec_weight", 1.0)) * risk_value - float(pcfg.get("racp_hard_weight", 10.0)) * hard - float(pcfg.get("racp_harm_weight", 1.5)) * harm
        admitted = safe & (risk_value >= float(pcfg.get("gamma_branch_rec", 0.0)))
        idx = _best(score, feasible)
        return ExternalSelection(idx, "risk_aware_multimodal_contingency", admitted, score)

    if baseline in {"expected_risk", "expected_risk_filter", "expected_risk_planner"}:
        risk = branch_fail + float(pcfg.get("risk_harm_weight", 0.25)) * harm + float(pcfg.get("risk_hard_weight", 1.0)) * hard
        admitted = feasible & (risk <= float(pcfg.get("expected_risk_threshold", 0.45)))
        score = utility - float(pcfg.get("expected_risk_weight", 3.0)) * risk
        idx = _best(score, admitted if admitted.any() else feasible)
        return ExternalSelection(idx, "expected_branch_risk_filter", admitted, score)

    if baseline in {"cvar_risk", "cvar_risk_filter", "cvar_planner"}:
        risk = -branch_cvar + float(pcfg.get("risk_harm_weight", 0.25)) * harm + float(pcfg.get("risk_hard_weight", 1.0)) * hard
        admitted = feasible & (risk <= float(pcfg.get("cvar_risk_threshold", 0.35)))
        score = utility - float(pcfg.get("cvar_risk_weight", 3.0)) * risk
        idx = _best(score, admitted if admitted.any() else feasible)
        return ExternalSelection(idx, "cvar_branch_risk_filter", admitted, score)

    if baseline in {"dro_cvar", "dro_cvar_filter", "dro_cvar_safety_filter"}:
        ambiguity = float(pcfg.get("dro_ambiguity_radius", 0.10))
        risk = np.maximum(-branch_cvar, -branch_worst) + ambiguity * np.sqrt(np.maximum(branch_fail, 0.0)) + float(pcfg.get("risk_harm_weight", 0.25)) * harm + float(pcfg.get("risk_hard_weight", 1.0)) * hard
        admitted = feasible & (risk <= float(pcfg.get("dro_cvar_threshold", 0.40)))
        score = utility - float(pcfg.get("dro_cvar_risk_weight", 3.5)) * risk
        idx = _best(score, admitted if admitted.any() else feasible)
        return ExternalSelection(idx, "distributionally_robust_cvar_filter", admitted, score)

    if baseline in {"postimpact_mpc", "postimpact_mpc_lite", "post_impact_mpc_lite", "postimpact_mpc_paper"}:
        costs = []
        opts = []
        for d in samples:
            c, details = _postimpact_mpc_cost(d, cfg)
            costs.append(c)
            opts.append(int(details.get("shared_option", 0)))
        cost = np.asarray(costs, dtype=float) if costs else np.zeros(n, dtype=float)
        score = -cost
        yaw_gate = float(pcfg.get("postimpact_yaw_rate_gate", 2.2))
        stable_gate = np.asarray([_postimpact_mpc_cost(d, cfg)[1].get("yaw_rate", 0.0) <= yaw_gate for d in samples], dtype=bool) if samples else np.zeros(n, dtype=bool)
        admitted = feasible & stable_gate & (hard <= float(pcfg.get("postimpact_hard_gate", 0.0)))
        idx = _best(score, admitted if admitted.any() else feasible)
        opt = int(opts[idx]) if opts else 0
        return ExternalSelection(idx, "planning_integrated_postimpact_mpc", admitted, score, selected_option=opt)

    raise ValueError(f"Unknown external baseline {baseline!r}")
