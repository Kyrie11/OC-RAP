from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .ocmero import torch_oc_mero


def _safe_log(x: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.clamp(x, min=1e-8))


def assignment_loss(pred_root_prob: torch.Tensor, root_assignments: torch.Tensor, future_probs: torch.Tensor) -> torch.Tensor:
    # pred_root_prob: [B,K], root_assignments: [B,J], future_probs: [B,J]
    B, K = pred_root_prob.shape
    assign = root_assignments.long().clamp_min(0).clamp_max(K - 1)
    gathered = torch.gather(pred_root_prob, 1, assign)
    mask = (root_assignments >= 0).float()
    weights = future_probs.float() * mask
    denom = weights.sum().clamp_min(1e-8)
    return -(weights * _safe_log(gathered)).sum() / denom


def signature_loss(pred_sig: torch.Tensor, target_sig: torch.Tensor, root_probs: torch.Tensor, root_valid: torch.Tensor) -> torch.Tensor:
    D = min(pred_sig.shape[-1], target_sig.shape[-1])
    err = F.smooth_l1_loss(pred_sig[..., :D], target_sig[..., :D].float(), reduction="none").mean(dim=-1)
    w = root_probs.float() * root_valid.float()
    return (err * w).sum() / w.sum().clamp_min(1e-8)


def observation_bce_loss(pred_C: torch.Tensor, target_y: torch.Tensor) -> torch.Tensor:
    y = target_y.float()
    p = torch.clamp(pred_C, 1e-6, 1.0 - 1e-6)
    pos = y.sum().clamp_min(1.0)
    neg = (1.0 - y).sum().clamp_min(1.0)
    w_pos = 0.5 * (pos + neg) / pos
    w_neg = 0.5 * (pos + neg) / neg
    w = torch.where(y > 0.5, w_pos, w_neg)
    loss = -(w * (y * torch.log(p) + (1.0 - y) * torch.log(1.0 - p)))
    return loss.mean()


def margin_loss(pred_M: torch.Tensor, target_M: torch.Tensor, root_probs: torch.Tensor, option_valid: torch.Tensor, root_valid: torch.Tensor) -> torch.Tensor:
    err = F.smooth_l1_loss(pred_M, target_M.float(), reduction="none")
    w = root_probs.float().unsqueeze(-1) * option_valid.float().unsqueeze(1) * root_valid.float().unsqueeze(-1)
    return (err * w).sum() / w.sum().clamp_min(1e-8)


def ib_loss(root_prob: torch.Tensor, prior_prob: torch.Tensor) -> torch.Tensor:
    return (root_prob * (_safe_log(root_prob) - _safe_log(prior_prob))).sum(dim=-1).mean()


def utility_loss(pred_u: torch.Tensor, target_u: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(pred_u, target_u.float().view_as(pred_u))


def anti_oracle_loss(r_dep_pred: torch.Tensor, artifact: torch.Tensor, delta_neg: float) -> torch.Tensor:
    art = artifact.float().view_as(r_dep_pred)
    if art.sum() <= 0:
        return r_dep_pred.sum() * 0.0
    return (art * F.relu(r_dep_pred - float(delta_neg))).sum() / art.sum().clamp_min(1.0)


def compute_losses(pred: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], cfg: dict[str, Any], flags: dict[str, Any] | None = None) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
    flags = flags or {}
    weights = cfg.get("loss_weights", {})
    use_lcvar = bool(flags.get("use_lcvar", True))
    use_obs_kernel = bool(flags.get("use_obs_kernel", True))
    root_target_mode = flags.get("root_target_mode", cfg.get("training", {}).get("root_target_mode", "recovery_signature"))
    target_sig_key = "root_future_signature" if root_target_mode == "full_future" and "root_future_signature" in batch else "root_signature"
    opt_valid = batch["option_valid"].bool()
    r_dep, r_orc, gap, q = torch_oc_mero(
        pred["margin"],
        pred["root_prob"],
        pred["C"],
        alpha=float(cfg.get("ocmero", {}).get("alpha", 0.2)),
        beta=float(cfg.get("ocmero", {}).get("beta", 0.2)),
        option_valid=opt_valid,
        use_lcvar=use_lcvar,
        use_obs_kernel=use_obs_kernel,
    )
    components = {
        "assign": assignment_loss(pred["root_prob"], batch["root_assignments"], batch["future_probs"]),
        "sig": signature_loss(pred["root_signature"], batch[target_sig_key], batch["root_probs"], batch["root_valid"]),
        "obs": observation_bce_loss(pred["C"], batch["y_obs"]),
        "marg": margin_loss(pred["margin"], batch["m_star"], batch["root_probs"], opt_valid, batch["root_valid"]),
        "ib": ib_loss(pred["root_prob"], pred["prior_prob"]),
        "util": utility_loss(pred["utility_hat"], batch["utility"]),
    }
    if bool(flags.get("use_anti_oracle", True)):
        components["art"] = anti_oracle_loss(r_dep, batch["i_art_star"], float(cfg.get("artifact", {}).get("delta_neg", 0.0)))
    else:
        components["art"] = r_dep.sum() * 0.0
    total = torch.zeros((), device=r_dep.device)
    name_map = {"assign": "assign", "sig": "sig", "obs": "obs", "marg": "margin", "ib": "ib", "art": "anti_oracle", "util": "utility"}
    for name, loss in components.items():
        total = total + float(weights.get(name_map[name], 1.0)) * loss
    logs = {f"loss_{k}": float(v.detach().cpu()) for k, v in components.items()}
    logs.update({"loss_total": float(total.detach().cpu()), "r_dep_pred": float(r_dep.detach().mean().cpu()), "r_orc_pred": float(r_orc.detach().mean().cpu())})
    aux = {"r_dep": r_dep, "r_orc": r_orc, "gap": gap, "q": q}
    return total, logs, aux
