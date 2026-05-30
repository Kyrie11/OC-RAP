from __future__ import annotations

import json
import numpy as np
from ocrap.evaluation import metrics
from ocrap.evaluation.baselines import nominal_selector, oracle_selector, risk_aware_selector, backup_filter_selector
from ocrap.models.selector import select_action, SelectorParams


def _arr(x):
    return np.asarray(x)


def nominal_utility(actions_states: np.ndarray, action_mask: np.ndarray) -> np.ndarray:
    actions_states = _arr(actions_states)
    action_mask = _arr(action_mask).astype(bool)
    progress = actions_states[:, :, -1, 0]
    lateral = np.abs(actions_states[:, :, -1, 1])
    jerk_proxy = np.mean(np.abs(actions_states[:, :, 1:, 4] - actions_states[:, :, :-1, 4]), axis=-1) if actions_states.shape[-2] > 1 and actions_states.shape[-1] > 4 else 0.0
    curv = np.mean(np.abs(actions_states[:, :, :, 5]), axis=-1) if actions_states.shape[-1] > 5 else 0.0
    U = progress - 0.5 * lateral - 0.1 * jerk_proxy - 0.5 * curv
    return np.where(action_mask, U, -1e9)


def _load_q(calibration: dict | str | None) -> dict:
    if calibration is None:
        return {"q_R": 0.0, "q_H": 0.0, "q_delta": 0.0, "q_C": 0.0}
    if isinstance(calibration, str):
        with open(calibration, "r", encoding="utf-8") as f:
            calibration = json.load(f)
    return {k: float(calibration.get(k, 0.0)) for k in ["q_R", "q_H", "q_delta", "q_C"]}


def _profiles_for_ours(arrays: dict, H_action: np.ndarray, H_gap: np.ndarray) -> tuple[dict, bool]:
    # Prefer learned prediction arrays produced by ocrap.evaluation.inference.
    learned = "R_pred" in arrays
    R = _arr(arrays.get("R_pred", arrays["R_star"]))
    H = _arr(arrays.get("H_pred", H_action))
    dH = _arr(arrays.get("dH_pred", H_gap))
    C_default = np.zeros_like(R)
    if "C_pred" in arrays:
        C = _arr(arrays["C_pred"])
    elif "c_rule_star" in arrays:
        c_modes = _arr(arrays["c_rule_star"]).astype(float)
        mode_probs = _arr(arrays.get("mode_probs", np.ones(c_modes.shape[-1], dtype=np.float32) / c_modes.shape[-1]))
        C = metrics.upper_tail_cvar_np(c_modes, mode_probs, alpha=0.2)
    else:
        C = C_default
    prof = {
        "R": R,
        "B": _arr(arrays.get("B_pred", 1.0 - R)),
        "U": _arr(arrays.get("U_pred", np.zeros_like(R))),
        "H": H,
        "dH": dH,
        "K_post": _arr(arrays.get("K_post_pred", arrays.get("K_action_star", np.zeros_like(R)))),
        "C": C,
    }
    if "witness_pred" in arrays:
        prof["witness"] = _arr(arrays["witness_pred"])
    elif "witness_oc" in arrays:
        prof["witness"] = _arr(arrays["witness_oc"])
    return prof, learned


def _selected_by_ours(arrays: dict, U: np.ndarray, H_action: np.ndarray, H_gap: np.ndarray, q: dict, eta_R: float, eta_H: float, epsilon_H: float, ablation: str | None) -> tuple[np.ndarray, dict, bool]:
    profiles, learned = _profiles_for_ours(arrays, H_action, H_gap)
    if ablation == "oracle_witness" and "Y_option" in arrays:
        Yopt = _arr(arrays["Y_option"]).astype(np.float32)
        mode_probs = _arr(arrays.get("mode_probs", np.ones(Yopt.shape[-1], dtype=np.float32) / Yopt.shape[-1]))
        profiles["R"] = metrics.weighted_lcvar_np(Yopt.max(axis=2), mode_probs, alpha=0.2)
        if "witness_raw_oracle" in arrays:
            profiles["witness"] = _arr(arrays["witness_raw_oracle"])
        learned = learned or True
    elif ablation == "no_observation_consistency" and "R_pred" not in arrays and "Y_option" in arrays:
        # No checkpoint is available to recompute OC-MERO without beta tying.  Use
        # the non-deployable per-mode option max as a clearly marked diagnostic
        # approximation rather than pretending this is learned no-OC inference.
        Yopt = _arr(arrays["Y_option"]).astype(np.float32)
        mode_probs = _arr(arrays.get("mode_probs", np.ones(Yopt.shape[-1], dtype=np.float32) / Yopt.shape[-1]))
        profiles["R"] = metrics.weighted_lcvar_np(Yopt.max(axis=2), mode_probs, alpha=0.2)
        if "witness_raw_oracle" in arrays:
            profiles["witness"] = _arr(arrays["witness_raw_oracle"])
    action_mask = _arr(arrays["action_mask"]).astype(bool)
    params = SelectorParams(eta_R=eta_R, eta_H=eta_H, epsilon_H=epsilon_H)
    if ablation in ("no_harm", "no_harm_constraint"):
        params.no_harm_constraint = True
    if ablation in ("no_rule", "no_rule_constraint"):
        params.no_rule_constraint = True
    if ablation in ("no_controlled_relaxation", "no_relaxation"):
        params.no_controlled_relaxation = True
    if ablation in ("penalize_uncertainty", "with_U_selector"):
        params.lambda_U_selector = 0.2
        params.method = "ablation"
    if ablation in ("no_recovery_constraint", "nominal"):
        return np.asarray([nominal_selector(U[i], action_mask[i]) for i in range(action_mask.shape[0])], dtype=np.int64), profiles, learned
    selected = []
    actions = list(range(action_mask.shape[1]))
    for i in range(action_mask.shape[0]):
        prof_i = {k: v[i] for k, v in profiles.items()}
        sel = select_action(actions, prof_i, U[i], q=q, masks={"action_mask": action_mask[i]}, params=params)
        selected.append(sel["action_index"])
    return np.asarray(selected, dtype=np.int64), profiles, learned


def evaluate_offline(arrays: dict, method: str = "ours", eta_R: float = 0.70, eta_H: float = 0.50, epsilon_H: float = 0.05, calibration: dict | str | None = None, ablation: str | None = None) -> dict:
    action_mask = _arr(arrays["action_mask"]).astype(bool)
    option_mask = _arr(arrays["option_mask"]).astype(bool)
    U = nominal_utility(_arr(arrays["actions_states"]), action_mask)
    R_star = _arr(arrays["R_star"])
    H_action = _arr(arrays.get("H_action_star", _arr(arrays.get("H_star", np.zeros((*R_star.shape, 1)))).max(axis=-1)))
    H_action = np.asarray(H_action, dtype=float)
    H_gap = H_action - np.min(np.where(action_mask, H_action, np.inf), axis=1, keepdims=True)
    q = _load_q(calibration)
    learned_profiles_used = False
    profiles = {}
    method_norm = method.lower().replace("-", "_")
    selected = []
    for i in range(action_mask.shape[0]):
        if method_norm in ("nom", "nominal"):
            a = nominal_selector(U[i], action_mask[i])
        elif method_norm in ("risk_cvar", "risk_aware"):
            collision = _arr(arrays.get("P_star", np.zeros_like(_arr(arrays["Y_option"]), dtype=float))).max(axis=(1, 3))[i]
            a = risk_aware_selector(U[i], collision, H_action[i], action_mask[i])
        elif method_norm in ("psf_backup", "backup_filter"):
            backup = _arr(arrays["Y_option"])[i, :, : min(2, _arr(arrays["Y_option"]).shape[2]), :].max(axis=(1, 2)).astype(bool)
            a = backup_filter_selector(U[i], backup, action_mask[i])
        elif method_norm in ("oracle", "oracle_oc"):
            a = oracle_selector(R_star[i], H_gap[i], action_mask[i], eta_R, epsilon_H)
        else:
            break
        selected.append(a)
    if len(selected) != action_mask.shape[0]:
        selected, profiles, learned_profiles_used = _selected_by_ours(arrays, U, H_action, H_gap, q, eta_R, eta_H, epsilon_H, ablation)
    else:
        selected = np.asarray(selected, dtype=np.int64)
    Y_option = _arr(arrays["Y_option"])
    Y_oc = _arr(arrays["Y_oc"]) if "Y_oc" in arrays else None
    witness_oc = _arr(arrays["witness_oc"]) if "witness_oc" in arrays else None
    oracle_R = metrics.weighted_lcvar_np(Y_option.max(axis=2).astype(np.float32), _arr(arrays.get("mode_probs", np.ones(Y_option.shape[-1]) / Y_option.shape[-1])), alpha=0.2)
    res = {
        "method": method,
        "ablation": ablation or "none",
        "N": int(len(selected)),
        "uses_learned_profiles": bool(learned_profiles_used),
        "uses_teacher_profiles_for_ours": bool(method_norm not in ("nom", "nominal", "risk_cvar", "risk_aware", "psf_backup", "backup_filter", "oracle", "oracle_oc") and not learned_profiles_used),
        "q": q,
        "OCS": metrics.recovery_success(Y_option, selected, option_mask, witness_oc=witness_oc, Y_oc=Y_oc),
        "ORS_oracle_option_success": metrics.oracle_recovery_success(Y_option, selected, option_mask),
        "FAR": metrics.false_recoverability(R_star, selected, eta_R),
        "SLR": metrics.selected_lower_tail_recoverability(R_star, selected),
        "OLG": float(np.mean([(oracle_R[i, a] - R_star[i, a]) if a >= 0 else 0.0 for i, a in enumerate(selected)])),
        "SRR": metrics.same_root_recoverability_regret(R_star, selected, action_mask),
        "HNIV": metrics.harm_noninferiority_violation(H_action, selected, epsilon_H, action_mask),
        "MIR": metrics.minimal_intervention_regret(U, selected, R_star, H_gap, eta_R, epsilon_H),
        "utility_mean": float(np.mean([U[i, a] if a >= 0 else 0.0 for i, a in enumerate(selected)])),
        "selected_action_idx": selected.tolist(),
    }
    if profiles:
        if "R" in profiles:
            res["R_MAE"] = metrics.mero_profile_error(np.asarray(profiles["R"]), R_star, action_mask)
            res["SRA"] = metrics.pairwise_ranking_accuracy(np.asarray(profiles["R"]), R_star, action_mask)
        if "witness" in profiles and "witness_oc" in arrays and "witness_gap" in arrays:
            res["WAcc"] = metrics.witness_accuracy(np.asarray(profiles["witness"]), _arr(arrays["witness_oc"]), _arr(arrays["witness_gap"]))
        if "mu_pred" in arrays and "obs_equiv" in arrays:
            res["OCV_JS"] = metrics.observation_consistency_violation(_arr(arrays["mu_pred"]), _arr(arrays["obs_equiv"]), option_mask)
    return res
