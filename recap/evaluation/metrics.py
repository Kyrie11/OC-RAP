from __future__ import annotations

import numpy as np


def weighted_lcvar_np(values: np.ndarray, weights: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim == 1:
        w = np.broadcast_to(w, vals.shape)
    elif w.ndim == 2 and vals.ndim == 3:
        w = np.broadcast_to(w[:, None, :], vals.shape)
    w = np.maximum(w, 0.0)
    w = w / np.maximum(w.sum(axis=-1, keepdims=True), 1e-12)
    order = np.argsort(vals, axis=-1)
    sv = np.take_along_axis(vals, order, axis=-1)
    sw = np.take_along_axis(w, order, axis=-1)
    out = np.zeros(vals.shape[:-1], dtype=np.float64)
    rem = np.full(vals.shape[:-1], alpha, dtype=np.float64)
    for j in range(vals.shape[-1]):
        take = np.minimum(sw[..., j], rem)
        out += take * sv[..., j]
        rem = np.maximum(rem - take, 0.0)
    return (out / max(alpha, 1e-12)).astype(np.float32)


def recovery_success(Y_option: np.ndarray, selected_action_idx: np.ndarray, option_mask: np.ndarray) -> float:
    vals = []
    for i, a in enumerate(selected_action_idx):
        valid = option_mask[i, a]
        y = Y_option[i, a, valid, :]
        vals.append(float(y.max() == 1) if y.size else 0.0)
    return float(np.mean(vals)) if vals else 0.0


def false_recoverability(R_star: np.ndarray, selected_action_idx: np.ndarray, eta_R: float = 0.70) -> float:
    return float(np.mean([R_star[i, a] < eta_R for i, a in enumerate(selected_action_idx)]))


def selected_lower_tail_recoverability(R_star: np.ndarray, selected_action_idx: np.ndarray) -> float:
    return float(np.mean([R_star[i, a] for i, a in enumerate(selected_action_idx)]))


def pairwise_ranking_accuracy(R_pred: np.ndarray, R_star: np.ndarray, action_mask: np.ndarray, eps: float = 1e-3) -> float:
    correct = 0
    total = 0
    for i in range(R_star.shape[0]):
        valid = np.where(action_mask[i])[0]
        for p, a in enumerate(valid):
            for b in valid[p + 1 :]:
                d_true = R_star[i, a] - R_star[i, b]
                if abs(d_true) <= eps:
                    continue
                d_pred = R_pred[i, a] - R_pred[i, b]
                correct += int(np.sign(d_pred) == np.sign(d_true))
                total += 1
    return float(correct / max(total, 1))


def same_root_recoverability_regret(R_star: np.ndarray, selected_action_idx: np.ndarray, action_mask: np.ndarray) -> float:
    vals = []
    for i, a in enumerate(selected_action_idx):
        vals.append(float(np.max(R_star[i][action_mask[i]]) - R_star[i, a]))
    return float(np.mean(vals)) if vals else 0.0


def care_evidence_error(pred: dict, star: dict, option_mask: np.ndarray, action_mask: np.ndarray) -> float:
    vals = []
    tuple_mask = option_mask[..., None]
    for k in ["P", "G", "C", "Kdef"]:
        sk = "K_star" if k == "Kdef" else f"{k}_star"
        if k in pred and sk in star:
            vals.append(float(np.mean(np.abs(pred[k][tuple_mask] - star[sk][tuple_mask]))))
    am = action_mask[..., None]
    for k in ["U", "H"]:
        sk = f"{k}_star"
        if k in pred and sk in star:
            vals.append(float(np.mean(np.abs(pred[k][am] - star[sk][am]))))
    return float(np.mean(vals)) if vals else 0.0


def mero_profile_error(R_pred: np.ndarray, R_star: np.ndarray, action_mask: np.ndarray) -> float:
    return float(np.mean(np.abs(R_pred[action_mask] - R_star[action_mask])))


def witness_accuracy(witness_pred: np.ndarray, witness_star: np.ndarray, witness_gap: np.ndarray, eps_w: float = 0.10) -> float:
    mask = witness_gap > eps_w
    if not np.any(mask):
        return 0.0
    return float(np.mean(witness_pred[mask] == witness_star[mask]))


def bottleneck_f1(pred_badness: np.ndarray, star_badness: np.ndarray) -> float:
    # pred_badness/star_badness [...,6]
    y_pred = np.argmax(pred_badness, axis=-1).ravel()
    y_true = np.argmax(star_badness, axis=-1).ravel()
    f1s = []
    for c in range(6):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1s.append(2 * prec * rec / max(prec + rec, 1e-12))
    return float(np.mean(f1s))


def harm_noninferiority_violation(H_action_star: np.ndarray, selected_action_idx: np.ndarray, epsilon_H: float = 0.05) -> float:
    vals = []
    for i, a in enumerate(selected_action_idx):
        vals.append(float(H_action_star[i, a] > np.min(H_action_star[i]) + epsilon_H))
    return float(np.mean(vals)) if vals else 0.0


def minimal_intervention_regret(U_drv: np.ndarray, selected_action_idx: np.ndarray, R_star: np.ndarray, H_gap_star: np.ndarray, eta_R: float = 0.70, epsilon_H: float = 0.05) -> float:
    vals = []
    for i in range(U_drv.shape[0]):
        a_nom = int(np.argmax(U_drv[i]))
        if R_star[i, a_nom] >= eta_R and H_gap_star[i, a_nom] <= epsilon_H:
            vals.append(max(0.0, float(U_drv[i, a_nom] - U_drv[i, selected_action_idx[i]])))
    return float(np.mean(vals)) if vals else 0.0
