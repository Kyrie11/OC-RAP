from __future__ import annotations

import numpy as np
from recap.utils.masks import masked_argmax


def nominal_selector(U_drv: np.ndarray, action_mask: np.ndarray) -> int:
    return masked_argmax(U_drv, action_mask)


def risk_aware_selector(U_drv: np.ndarray, collision_prob: np.ndarray, H: np.ndarray, action_mask: np.ndarray, lambda_collision: float = 1.0, lambda_harm: float = 1.0) -> int:
    return masked_argmax(U_drv - lambda_collision * collision_prob - lambda_harm * H, action_mask)


def backup_filter_selector(U_drv: np.ndarray, backup_success: np.ndarray, action_mask: np.ndarray) -> int:
    feasible = action_mask & backup_success.astype(bool)
    if feasible.any():
        return masked_argmax(U_drv, feasible)
    return masked_argmax(U_drv, action_mask)


def oracle_selector(R_star: np.ndarray, H_gap: np.ndarray, action_mask: np.ndarray, eta_R: float = 0.70, epsilon_H: float = 0.05) -> int:
    feasible = action_mask & (R_star >= eta_R) & (H_gap <= epsilon_H)
    if feasible.any():
        return masked_argmax(R_star, feasible)
    return masked_argmax(R_star - np.maximum(H_gap - epsilon_H, 0.0), action_mask)
