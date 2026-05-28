from __future__ import annotations

import torch
import torch.nn.functional as F
from ocrap.models.oc_mero import compute_ocmero_profiles, js_divergence


def masked_l1(pred, target, mask):
    m = mask.float()
    return (torch.abs(pred - target) * m).sum() / m.sum().clamp_min(1.0)


def masked_huber(pred, target, mask, beta: float = 1.0):
    m = mask.float()
    return (F.smooth_l1_loss(pred, target, reduction="none", beta=beta) * m).sum() / m.sum().clamp_min(1.0)


def observation_consistency_loss(mu, obs_equiv, option_mask):
    # mu [B,K,L,M], compare distributions over L for equivalent mode pairs.
    B,K,L,M=mu.shape
    eq=obs_equiv.bool()
    losses=[]
    mask=option_mask.bool()
    for m in range(M):
        for n in range(m+1,M):
            pair=eq[...,m,n]
            if pair.any():
                pm=mu[...,m].transpose(-1,-2) if False else mu[:,:,:,m]
                pn=mu[:,:,:,n]
                # renormalize over valid options.
                pm=pm*mask.float(); pn=pn*mask.float()
                div=js_divergence(pm, pn)
                losses.append(div[pair])
    if not losses:
        return mu.sum()*0.0
    return torch.cat([x.reshape(-1) for x in losses]).mean()


def _loss_weight(params: dict | None, name: str, default: float) -> float:
    if not params:
        return float(default)
    if name in params:
        return float(params[name])
    loss_cfg = params.get("loss", {}) if isinstance(params, dict) else {}
    return float(loss_cfg.get(name, default))


def ocrap_loss(outputs: dict, batch: dict, params: dict | None = None) -> torch.Tensor:
    option_mask = batch["option_mask"].bool().unsqueeze(-1)
    tuple_mask = option_mask.expand_as(outputs["k_hat"])
    action_mask = batch["action_mask"].bool().unsqueeze(-1)
    loss = outputs["k_hat"].sum()*0.0
    if "g_star" in batch:
        gs=batch["g_star"].float()
        if gs.dim()==4: gs=gs.unsqueeze(0)
        loss = loss + masked_huber(outputs["g_hat"], gs, tuple_mask.unsqueeze(-1).expand_as(outputs["g_hat"]))
    if "y_star" in batch:
        loss = loss + F.binary_cross_entropy_with_logits(outputs["y_logit"], batch["y_star"].float(), weight=tuple_mask.float(), reduction="sum") / tuple_mask.float().sum().clamp_min(1.0)
    if "h_star" in batch:
        loss = loss + 0.5*masked_huber(outputs["h_hat"], batch["h_star"].float(), action_mask.expand_as(outputs["h_hat"]))
    elif "H_star" in batch:
        loss = loss + 0.5*masked_huber(outputs["h_hat"], batch["H_star"].float(), action_mask.expand_as(outputs["h_hat"]))
    if "k_star" in batch:
        loss = loss + 0.5*masked_huber(outputs["k_hat"], batch["k_star"].float(), tuple_mask)
    elif "K_star" in batch:
        loss = loss + 0.5*masked_huber(outputs["k_hat"], batch["K_star"].float(), tuple_mask)
    if "u_star" in batch:
        tgt=batch["u_star"].float()
        if tgt.shape == outputs["u_hat"].shape:
            loss = loss + _loss_weight(params, "lambda_u", 0.2)*masked_huber(outputs["u_hat"], tgt, action_mask.expand_as(outputs["u_hat"]))
    if "c_rule_star" in batch and "c_rule_hat" in outputs:
        loss = loss + _loss_weight(params, "lambda_c", 1.0)*masked_huber(outputs["c_rule_hat"], batch["c_rule_star"].float(), action_mask.expand_as(outputs["c_rule_hat"]))
    elif "C_star" in batch and "c_rule_hat" in outputs:
        C=batch["C_star"].float()
        if C.dim()==4:
            # Legacy option-level C labels: supervise action-level head through the valid-option max.
            C = C.masked_fill(~batch["option_mask"].bool().unsqueeze(-1), -1e6).max(dim=2).values.clamp_min(0.0)
        loss = loss + _loss_weight(params, "lambda_c", 1.0)*masked_huber(outputs["c_rule_hat"], C, action_mask.expand_as(outputs["c_rule_hat"]))
    if "beta_star" in batch:
        target=batch["beta_star"].float().clamp_min(1e-8)
        pred_log=torch.log_softmax(outputs["beta_logits"], dim=-1)
        loss = loss + 0.5*F.kl_div(pred_log, target, reduction="batchmean")
    if "witness_oc" in batch:
        logits=outputs["mu_logits"].permute(0,1,3,2).reshape(-1, outputs["mu_logits"].shape[2])
        target=batch["witness_oc"].long().reshape(-1)
        valid=target>=0
        if valid.any(): loss = loss + 0.5*F.cross_entropy(logits[valid], target[valid])
    if "obs_equiv" in batch:
        prof=compute_ocmero_profiles(outputs, {"action_mask": batch["action_mask"], "option_mask": batch["option_mask"]}, params or {})
        loss = loss + 0.5*observation_consistency_loss(prof["mu"], batch["obs_equiv"], batch["option_mask"])
        if "R_star" in batch:
            loss = loss + masked_huber(prof["R"], batch["R_star"].float(), batch["action_mask"].bool())
    return loss


def care_supervised_loss(outputs: dict, batch: dict) -> torch.Tensor:
    # Backward-compatible wrapper: new outputs use OC-RAP, old outputs use CARE keys.
    if "g_hat" in outputs:
        return ocrap_loss(outputs, batch)
    option_mask = batch["option_mask"].bool().unsqueeze(-1).expand_as(outputs["P"])
    action_mask = batch["action_mask"].bool().unsqueeze(-1).expand_as(outputs["U"])
    loss = 0.0
    for out_key, star_key in [("P", "P_star"), ("G", "G_star"), ("C", "C_star"), ("Kdef", "K_star")]:
        if star_key in batch:
            loss = loss + masked_l1(outputs[out_key], batch[star_key].float(), option_mask)
    for out_key, star_key in [("U", "U_star"), ("H", "H_star")]:
        if star_key in batch:
            loss = loss + masked_l1(outputs[out_key], batch[star_key].float(), action_mask)
    if "witness" in batch:
        logits = outputs["option_logits"].permute(0, 1, 3, 2).reshape(-1, outputs["option_logits"].shape[2])
        target = batch["witness"].long().reshape(-1)
        valid = target >= 0
        if valid.any(): loss = loss + F.cross_entropy(logits[valid], target[valid])
    return loss
