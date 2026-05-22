from __future__ import annotations

import numpy as np
from recap.evaluation import metrics
from recap.evaluation.baselines import nominal_selector, oracle_selector, risk_aware_selector, backup_filter_selector


def nominal_utility(actions_states: np.ndarray, action_mask: np.ndarray) -> np.ndarray:
    progress = actions_states[:, :, -1, 0]
    lateral = np.abs(actions_states[:, :, -1, 1])
    jerk_proxy = np.mean(np.abs(actions_states[:, :, 1:, 4] - actions_states[:, :, :-1, 4]), axis=-1)
    curv = np.mean(np.abs(actions_states[:, :, :, 5]), axis=-1)
    U = progress - 0.5 * lateral - 0.1 * jerk_proxy - 0.5 * curv
    return np.where(action_mask, U, -1e9)


def evaluate_offline(arrays: dict, method: str = "oracle", eta_R: float = 0.70, epsilon_H: float = 0.05) -> dict:
    action_mask = arrays["action_mask"].astype(bool)
    option_mask = arrays["option_mask"].astype(bool)
    U = nominal_utility(arrays["actions_states"], action_mask)
    R_star = arrays["R_star"]
    H_action = arrays.get("H_action_star", arrays.get("H_star", np.zeros((R_star.shape[0], R_star.shape[1], 1))).max(axis=-1))
    H_gap = H_action - np.min(np.where(action_mask, H_action, np.inf), axis=1, keepdims=True)
    selected = []
    for i in range(action_mask.shape[0]):
        if method == "nominal":
            a = nominal_selector(U[i], action_mask[i])
        elif method == "risk_aware":
            collision = arrays.get("P_star", np.zeros_like(arrays["Y_option"], dtype=float)).max(axis=(1, 3))[i]
            a = risk_aware_selector(U[i], collision, H_action[i], action_mask[i])
        elif method == "backup_filter":
            backup = arrays["Y_option"][i, :, :2, :].max(axis=(1, 2)).astype(bool)
            a = backup_filter_selector(U[i], backup, action_mask[i])
        else:
            a = oracle_selector(R_star[i], H_gap[i], action_mask[i], eta_R, epsilon_H)
        selected.append(a)
    selected = np.asarray(selected, dtype=np.int64)
    return {
        "method": method,
        "N": int(len(selected)),
        "RS": metrics.recovery_success(arrays["Y_option"], selected, option_mask),
        "FR": metrics.false_recoverability(R_star, selected, eta_R),
        "SLR": metrics.selected_lower_tail_recoverability(R_star, selected),
        "SRR": metrics.same_root_recoverability_regret(R_star, selected, action_mask),
        "HNIV": metrics.harm_noninferiority_violation(H_action, selected, epsilon_H),
        "selected_action_idx": selected.tolist(),
    }
