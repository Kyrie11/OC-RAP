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



def _weighted_upper_cvar(values: np.ndarray, weights: np.ndarray, alpha: float) -> float:
    """Weighted CVaR of the upper tail of a nonnegative loss."""
    values = np.asarray(values, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not mask.any():
        return 0.0
    values, weights = values[mask], weights[mask]
    order = np.argsort(values)[::-1]
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


def _effective_root_outcomes(d: dict[str, Any], alpha: float = 0.2, gamma: float = 0.0) -> dict[str, Any]:
    """Branch-wise existential margins and risk-loss samples.

    For a latent root z_k, branch-wise recovery is existential in the option
    dimension: the branch succeeds if any option g_l has margin m_{k,l} >= gamma.
    This is exactly the oracle order in the OC-RAP paper: max over options first,
    then aggregate over latent roots.
    """
    base = _branchwise_values(d, alpha=alpha)
    best = np.asarray(base.get("best_margins", np.zeros((0,), dtype=float)), dtype=float).reshape(-1)
    K = int(best.size)
    w, valid = _valid_root_weights(d, K)
    if K == 0:
        return {**base, "losses": np.zeros((0,), dtype=float), "risk_expected": 1.0, "risk_cvar": 1.0, "risk_worst": 1.0, "oracle_all_roots": False, "oracle_mass": 0.0}
    clipped = np.clip(best, -5.0, 5.0)
    losses = np.where(valid, np.maximum(0.0, float(gamma) - clipped), 5.0)
    risk_expected = float(np.sum(w * losses)) if w.size else float(np.mean(losses))
    risk_cvar = _weighted_upper_cvar(losses, w if w.size else np.ones_like(losses) / max(len(losses), 1), alpha=float(alpha))
    risk_worst = float(np.max(losses[valid])) if valid.any() else float(np.max(losses))
    oracle_ok = valid & (clipped >= float(gamma))
    all_roots = bool(valid.any() and np.all(oracle_ok[valid]))
    mass = float(np.sum(w * oracle_ok.astype(float))) if w.size else 0.0
    return {**base, "losses": losses, "risk_expected": risk_expected, "risk_cvar": risk_cvar, "risk_worst": risk_worst, "oracle_all_roots": all_roots, "oracle_mass": mass}


def _prefix_common_horizon(candidate: dict[str, Any], reference: dict[str, Any] | None, *, threshold: float = 1.0, max_fraction: float = 0.6) -> float:
    """Dynamic branch-point proxy: latest prefix time before scenario divergence."""
    if reference is None:
        return 0.0
    a = np.asarray(candidate.get("prefix_states", np.zeros((0, 0))), dtype=float)
    b = np.asarray(reference.get("prefix_states", np.zeros((0, 0))), dtype=float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] == 0 or b.shape[0] == 0 or a.shape[1] < 2 or b.shape[1] < 2:
        return 0.0
    T = min(a.shape[0], b.shape[0])
    if T <= 1:
        return 0.0
    dist = np.linalg.norm(a[:T, :2] - b[:T, :2], axis=-1)
    ok = np.where(dist <= float(threshold))[0]
    if ok.size == 0:
        return 0.0
    latest = int(ok[-1])
    cap = int(max(1, round(float(max_fraction) * (T - 1))))
    return float(min(latest, cap) / max(T - 1, 1))


def _control_smoothness_cost(d: dict[str, Any], dt: float = 0.2) -> float:
    states = np.asarray(d.get("prefix_states", np.zeros((0, 0))), dtype=float)
    controls = np.asarray(d.get("prefix_controls", np.zeros((0, 0))), dtype=float)
    cost = 0.0
    if controls.ndim == 2 and controls.size:
        if controls.shape[1] >= 1:
            a = controls[:, 0]
            cost += float(np.nanmean(np.abs(a))) / 4.0
            if a.size > 1:
                cost += 0.25 * float(np.nanmax(np.abs(np.diff(a) / max(dt, 1e-3)))) / 8.0
        if controls.shape[1] >= 2:
            steer = controls[:, 1]
            cost += 0.5 * float(np.nanmean(np.abs(steer))) / 0.6
            if steer.size > 1:
                cost += 0.15 * float(np.nanmax(np.abs(np.diff(steer) / max(dt, 1e-3)))) / 1.0
    if states.ndim == 2 and states.shape[0] > 1 and states.shape[1] >= 3:
        yaw = np.unwrap(states[:, 2])
        yr = np.diff(yaw) / max(dt, 1e-3)
        if yr.size:
            cost += 0.3 * float(np.nanmax(np.abs(yr))) / 1.0
    return float(np.nan_to_num(cost, nan=0.0, posinf=10.0, neginf=0.0))


def _nominal_deviation(samples: list[dict[str, Any]]) -> np.ndarray:
    if not samples:
        return np.zeros((0,), dtype=float)
    ref = np.asarray(samples[0].get("prefix_states", np.zeros((0, 0))), dtype=float)
    vals = []
    for d in samples:
        xy = np.asarray(d.get("prefix_states", np.zeros((0, 0))), dtype=float)
        if ref.ndim != 2 or xy.ndim != 2 or ref.shape[0] == 0 or xy.shape[0] == 0 or ref.shape[1] < 2 or xy.shape[1] < 2:
            vals.append(0.0)
            continue
        T = min(ref.shape[0], xy.shape[0])
        vals.append(float(np.sqrt(np.mean(np.sum((xy[:T, :2] - ref[:T, :2]) ** 2, axis=-1))) / 5.0))
    return np.asarray(vals, dtype=float)


def _macro_names(samples: list[dict[str, Any]]) -> list[str]:
    out = []
    for d in samples:
        v = d.get("prefix_macro_name", d.get("macro_name", ""))
        try:
            v = np.asarray(v).item()
            if isinstance(v, bytes):
                v = v.decode("utf-8", errors="ignore")
        except Exception:
            pass
        out.append(str(v))
    return out


def _posterior_root_values(d: dict[str, Any], alpha: float, temperature: float = 0.7) -> dict[str, Any]:
    eff = _effective_root_outcomes(d, alpha=alpha)
    margins = np.asarray(eff.get("best_margins", np.zeros((0,), dtype=float)), dtype=float)
    K = margins.size
    w, valid = _valid_root_weights(d, K)
    if K == 0 or not valid.any():
        return {**eff, "posterior_expected": 0.0, "entropy": 0.0, "posterior": w}
    logits = np.clip(margins / max(float(temperature), 1e-3), -20.0, 20.0)
    likelihood = np.exp(logits - np.nanmax(logits[valid]))
    post = np.where(valid, w * likelihood, 0.0)
    den = float(post.sum())
    post = post / den if den > 1e-8 else w
    entropy = float(-np.sum(post[post > 0] * np.log(post[post > 0])) / max(np.log(max(int(valid.sum()), 2)), 1e-8))
    return {**eff, "posterior_expected": float(np.sum(post * np.clip(margins, -5.0, 5.0))), "entropy": entropy, "posterior": post}

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

    nominal_d = samples[0] if samples else None
    branch_eff = [_effective_root_outcomes(d, alpha=float(pcfg.get("cvar_alpha", 0.2)), gamma=float(pcfg.get("gamma_branch_rec", 0.0))) for d in samples]
    branch_expected = np.asarray([b["expected"] for b in branch_eff], dtype=float)
    branch_cvar = np.asarray([b["cvar"] for b in branch_eff], dtype=float)
    branch_worst = np.asarray([b["worst"] for b in branch_eff], dtype=float)
    branch_fail = np.asarray([b["fail_prob"] for b in branch_eff], dtype=float)
    risk_expected = np.asarray([b["risk_expected"] for b in branch_eff], dtype=float)
    risk_cvar = np.asarray([b["risk_cvar"] for b in branch_eff], dtype=float)
    risk_worst = np.asarray([b["risk_worst"] for b in branch_eff], dtype=float)
    oracle_mass = np.asarray([b["oracle_mass"] for b in branch_eff], dtype=float)
    oracle_all = np.asarray([bool(b["oracle_all_roots"]) for b in branch_eff], dtype=bool)
    common = np.asarray([_prefix_common_horizon(d, nominal_d, threshold=float(pcfg.get("branch_divergence_threshold_m", 1.0)), max_fraction=float(pcfg.get("max_branch_fraction", 0.6))) for d in samples], dtype=float)
    smooth = np.asarray([_control_smoothness_cost(d, dt=float(pcfg.get("dt", 0.2))) for d in samples], dtype=float)
    dev = _nominal_deviation(samples)
    macros = _macro_names(samples)

    if baseline in {"marc", "marc_lite", "marc_contingency"}:
        # MARC core: evaluate semantic ego policies, render policy-conditioned
        # critical scenarios, construct a dynamic branch point, then solve a
        # risk-aware contingency score.  OC-RAP's candidate lattice already
        # materializes semantic policies and counterfactual roots; this block
        # therefore keeps MARC's order of operations over that lattice.
        risk_tol = float(pcfg.get("marc_risk_tolerance", 0.35))
        rec = (1.0 - risk_tol) * branch_expected + risk_tol * branch_cvar
        score = (
            float(pcfg.get("marc_utility_weight", 1.0)) * utility
            + float(pcfg.get("marc_branch_rec_weight", 1.0)) * rec
            + float(pcfg.get("marc_common_prefix_weight", 0.35)) * common
            + float(pcfg.get("marc_oracle_mass_weight", 0.25)) * oracle_mass
            - float(pcfg.get("marc_expected_risk_weight", 2.0)) * risk_expected
            - float(pcfg.get("marc_fail_weight", 1.0)) * branch_fail
            - float(pcfg.get("marc_smoothness_weight", 0.15)) * smooth
            - float(pcfg.get("marc_deviation_weight", 0.10)) * dev
            - float(pcfg.get("marc_hard_weight", 10.0)) * hard
            - float(pcfg.get("marc_harm_weight", 1.5)) * harm
        )
        # Policy-level selection: keep the best candidate under each semantic
        # macro, then select among policies.  This avoids collapsing MARC into a
        # single trajectory scorer while still returning an executable prefix.
        idxs = []
        for m in sorted(set(macros)):
            ids = np.asarray([i for i, mm in enumerate(macros) if mm == m], dtype=int)
            if ids.size:
                ids = ids[feasible[ids]] if feasible[ids].any() else ids
                idxs.append(int(ids[np.argmax(score[ids])]))
        if idxs:
            cand = np.asarray(idxs, dtype=int)
            idx = int(cand[np.argmax(score[cand])])
        else:
            idx = _best(score, feasible)
        admitted = safe & (rec >= float(pcfg.get("gamma_branch_rec", 0.0))) & (risk_expected <= float(pcfg.get("marc_risk_threshold", 1.0)))
        return ExternalSelection(idx, "marc_policy_conditioned_risk_aware_contingency", admitted, score)

    if baseline in {"racp", "racp_lite", "risk_aware_contingency"}:
        # RACP core: maintain Bayesian beliefs over prediction modes, split the
        # plan into shared and contingent parts, and constrain the worst
        # discounted probabilistic risk.  Root probabilities become the prior;
        # branch recovery likelihoods produce a posterior used in the cost.
        post = [_posterior_root_values(d, alpha=float(pcfg.get("cvar_alpha", 0.2)), temperature=float(pcfg.get("racp_belief_temperature", 0.7))) for d in samples]
        posterior_expected = np.asarray([b["posterior_expected"] for b in post], dtype=float)
        entropy = np.asarray([b["entropy"] for b in post], dtype=float)
        rho = float(pcfg.get("racp_risk_tolerance", 0.6))
        risk_value = rho * posterior_expected + (1.0 - rho) * branch_cvar
        eta = risk_expected + float(pcfg.get("racp_tail_weight", 0.5)) * risk_cvar + float(pcfg.get("risk_harm_weight", 0.25)) * harm + float(pcfg.get("risk_hard_weight", 1.0)) * hard
        branch_bonus = common * (1.0 - entropy)
        score = (
            float(pcfg.get("racp_utility_weight", 1.0)) * utility
            + float(pcfg.get("racp_branch_rec_weight", 1.0)) * risk_value
            + float(pcfg.get("racp_belief_branch_weight", 0.45)) * branch_bonus
            - float(pcfg.get("racp_risk_weight", 2.5)) * eta
            - float(pcfg.get("racp_smoothness_weight", 0.10)) * smooth
            - float(pcfg.get("racp_hard_weight", 10.0)) * hard
            - float(pcfg.get("racp_harm_weight", 1.5)) * harm
        )
        admitted = safe & (eta <= float(pcfg.get("racp_risk_threshold", pcfg.get("racp_delta", 0.75)))) & (risk_value >= float(pcfg.get("gamma_branch_rec", 0.0)))
        idx = _best(score, admitted if admitted.any() else feasible)
        return ExternalSelection(idx, "racp_bayesian_belief_contingency", admitted, score)

    if baseline in {"expected_risk", "expected_risk_filter", "expected_risk_planner"}:
        # Expected-risk filter: bound the expected signed collision/recovery loss.
        risk = risk_expected + float(pcfg.get("risk_harm_weight", 0.25)) * harm + float(pcfg.get("risk_hard_weight", 1.0)) * hard
        admitted = feasible & (risk <= float(pcfg.get("expected_risk_threshold", 0.45)))
        score = utility - float(pcfg.get("expected_risk_weight", 3.0)) * risk - float(pcfg.get("risk_deviation_weight", 0.05)) * dev
        idx = _best(score, admitted if admitted.any() else feasible)
        return ExternalSelection(idx, "expected_signed_collision_risk_filter", admitted, score)

    if baseline in {"cvar_risk", "cvar_risk_filter", "cvar_planner"}:
        # CVaR filter: upper-tail risk over root-conditioned losses.
        risk = risk_cvar + float(pcfg.get("risk_harm_weight", 0.25)) * harm + float(pcfg.get("risk_hard_weight", 1.0)) * hard
        admitted = feasible & (risk <= float(pcfg.get("cvar_risk_threshold", 0.55)))
        score = utility - float(pcfg.get("cvar_risk_weight", 3.0)) * risk - float(pcfg.get("risk_deviation_weight", 0.05)) * dev
        idx = _best(score, admitted if admitted.any() else feasible)
        return ExternalSelection(idx, "cvar_tail_signed_collision_risk_filter", admitted, score)

    if baseline in {"dro_cvar", "dro_cvar_filter", "dro_cvar_safety_filter", "dr_cvar_filter"}:
        # DR-CVaR filter: CVaR plus a Wasserstein-ball ambiguity penalty.  This
        # mirrors the paper's safe-halfspace relaxation when only finite samples
        # of future obstacle/root outcomes are available in the OC-RAP dataset.
        ambiguity = float(pcfg.get("dro_ambiguity_radius", 0.10))
        dispersion = np.sqrt(np.maximum(risk_worst - risk_expected, 0.0) ** 2 + np.maximum(branch_fail, 0.0))
        risk = risk_cvar + ambiguity * dispersion / max(float(pcfg.get("cvar_alpha", 0.2)), 1e-3) + float(pcfg.get("risk_harm_weight", 0.25)) * harm + float(pcfg.get("risk_hard_weight", 1.0)) * hard
        admitted = feasible & (risk <= float(pcfg.get("dro_cvar_threshold", 0.65)))
        score = utility - float(pcfg.get("dro_cvar_risk_weight", 3.5)) * risk - float(pcfg.get("risk_deviation_weight", 0.05)) * dev
        idx = _best(score, admitted if admitted.any() else feasible)
        return ExternalSelection(idx, "distributionally_robust_cvar_safe_halfspace_filter", admitted, score)

    if baseline in {"predictive_safety_filter", "psf", "cbf_backup_filter", "predictive_cbf_backup", "backup_cbf_filter"}:
        # Predictive Safety Filter / CBF backup baseline.  Treat candidate 0 as
        # the nominal learning/reference input u_L.  Certify it when there is a
        # finite-horizon branch-wise backup to a safe set; otherwise apply the
        # smallest feasible modification that improves the CBF-like recovery
        # barrier and respects control/comfort bounds.
        gamma_b = float(pcfg.get("psf_gamma_branch_rec", pcfg.get("gamma_branch_rec", 0.0)))
        accel = np.zeros(n, dtype=float)
        steer = np.zeros(n, dtype=float)
        for i, d in enumerate(samples):
            accel[i], steer[i] = _control_proxy(d)
        ctrl_ok = (accel <= float(pcfg.get("psf_accel_gate", 6.0))) & (steer <= float(pcfg.get("psf_steer_gate", 0.75)))
        backup_ok = oracle_all | (branch_worst >= gamma_b)
        cbf_value = branch_worst - float(pcfg.get("psf_hard_barrier_weight", 1.0)) * hard - float(pcfg.get("psf_harm_barrier_weight", 0.25)) * harm
        nominal_cbf = cbf_value[0] if cbf_value.size else 0.0
        cbf_ok = cbf_value >= (1.0 - float(pcfg.get("psf_cbf_kappa", 0.5))) * nominal_cbf - float(pcfg.get("psf_cbf_slack", 0.05))
        admitted = feasible & ctrl_ok & backup_ok & cbf_ok & (hard <= float(pcfg.get("psf_hard_gate", 0.0))) & (harm <= float(pcfg.get("psf_harm_gate", 5.0)))
        # Minimal modification objective: stay close to u_L/nominal, then prefer
        # utility and recovery margin.
        score = (
            -float(pcfg.get("psf_deviation_weight", 2.0)) * dev
            + float(pcfg.get("psf_utility_weight", 0.35)) * utility
            + float(pcfg.get("psf_barrier_weight", 1.0)) * cbf_value
            - float(pcfg.get("psf_smoothness_weight", 0.15)) * smooth
        )
        if admitted.size and admitted[0]:
            idx = 0
        else:
            idx = _best(score, admitted if admitted.any() else feasible)
        return ExternalSelection(idx, "predictive_safety_filter_cbf_backup", admitted, score)

    if baseline in {"oracle_filter", "oracle_recovery_filter", "branchwise_oracle_filter", "oracle_branchwise_recovery"}:
        gamma_o = float(pcfg.get("gamma_oracle_rec", pcfg.get("gamma_branch_rec", 0.0)))
        # Strict existential-by-root certificate: every valid root must have some
        # option above gamma.  The score uses the LCVaR oracle headroom to break
        # ties, exactly matching Eq. oracle_recovery in the paper.
        admitted = safe & oracle_all & (branch_cvar >= gamma_o)
        score = branch_cvar + float(pcfg.get("oracle_utility_tiebreak", 1.0e-3)) * utility
        idx = _best(score, admitted if admitted.any() else feasible)
        opt = None
        if 0 <= idx < len(branch_eff):
            opts = np.asarray(branch_eff[idx].get("best_options", np.zeros((0,), dtype=int)))
            opt = int(opts[0]) if opts.size else None
        return ExternalSelection(idx, "branchwise_existential_oracle_recovery_filter", admitted, score, selected_option=opt)

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
