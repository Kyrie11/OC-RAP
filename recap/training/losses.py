from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_l1(pred, target, mask):
    m = mask.float()
    return (torch.abs(pred - target) * m).sum() / m.sum().clamp_min(1.0)


def care_supervised_loss(outputs: dict, batch: dict) -> torch.Tensor:
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
        if valid.any():
            loss = loss + F.cross_entropy(logits[valid], target[valid])
    return loss
