from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


class MaskedTokenEncoder(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.LazyLinear(out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim), nn.ReLU())

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B = x.shape[0]
        x = x.reshape(B, -1, x.shape[-1]).float()
        h = self.net(x)
        if mask is None:
            return h.mean(dim=1)
        m = mask.reshape(B, -1).float().to(h.device)
        denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (h * m.unsqueeze(-1)).sum(dim=1) / denom


class BEVEncoder(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.LazyConv2d(16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(64, out_dim), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.conv(x.float()))


class OCRAPModel(nn.Module):
    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        model_cfg = cfg.get("model", {})
        self.K = int(cfg.get("num_roots", 8))
        self.L = int(cfg.get("num_recovery_options", 24))
        self.d_model = int(model_cfg.get("d_model", 128))
        self.d_z = int(model_cfg.get("d_z", 128))
        self.d_obs = int(model_cfg.get("d_obs", 64))
        self.d_sig = int(model_cfg.get("d_signature", self.L + 8))
        self.tau_obs = float(cfg.get("tau_obs", cfg.get("epsilon_obs", 1.0) ** 2 / math.log(2.0)))
        self.no_occlusion_bev = bool(model_cfg.get("no_occlusion_bev", False))

        self.agent_encoder = MaskedTokenEncoder(self.d_model)
        self.map_encoder = MaskedTokenEncoder(self.d_model)
        self.route_encoder = MaskedTokenEncoder(self.d_model // 2)
        self.prefix_encoder = nn.Sequential(nn.LazyLinear(self.d_model), nn.ReLU(), nn.Linear(self.d_model, self.d_model), nn.ReLU())
        self.bev_encoder = BEVEncoder(self.d_model)
        self.macro_embed = nn.Embedding(int(model_cfg.get("num_macros", 16)), self.d_model // 4)
        self.fuse = nn.Sequential(nn.LazyLinear(self.d_model * 2), nn.ReLU(), nn.Linear(self.d_model * 2, self.d_model), nn.ReLU())
        self.root_queries = nn.Parameter(torch.randn(self.K, self.d_z) * 0.02)
        self.context_to_root = nn.Linear(self.d_model, self.d_z)
        self.root_mlp = nn.Sequential(nn.Linear(self.d_z, self.d_z), nn.ReLU(), nn.Linear(self.d_z, self.d_z), nn.ReLU())
        self.root_logit = nn.Linear(self.d_z, 1)
        self.prior_logit = nn.Linear(self.d_z, 1)
        self.sig_head = nn.Linear(self.d_z, self.d_sig)
        self.obs_head = nn.Linear(self.d_z, self.d_obs)
        self.margin_head = nn.Sequential(nn.Linear(self.d_z, self.d_z), nn.ReLU(), nn.Linear(self.d_z, self.L))
        self.utility_head = nn.Linear(self.d_model, 1)

    def encode_context(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        agent = batch["agent_history"].float()
        agent_valid = batch.get("agent_valid")
        if agent_valid is not None:
            agent_valid = agent_valid.float()
        agent_h = self.agent_encoder(agent, agent_valid)
        map_h = self.map_encoder(batch["map_polylines"].float(), batch.get("map_valid"))
        route = batch["route"].float()
        route_mask = torch.isfinite(route[..., 0]).float()
        route_h = self.route_encoder(route, route_mask)
        prefix = torch.cat([batch["prefix_states"].flatten(1), batch["prefix_controls"].flatten(1), batch["prefix_param"].float().flatten(1)], dim=1)
        prefix_h = self.prefix_encoder(prefix)
        macro_id = batch["prefix_macro_id"].long().view(-1).clamp_min(0).clamp_max(self.macro_embed.num_embeddings - 1)
        macro_h = self.macro_embed(macro_id)
        if self.no_occlusion_bev:
            bev_h = torch.zeros((agent.shape[0], self.d_model), dtype=agent.dtype, device=agent.device)
        else:
            bev_h = self.bev_encoder(batch["bev_occ"].float())
        return self.fuse(torch.cat([agent_h, map_h, route_h, prefix_h, bev_h, macro_h], dim=1))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        ctx = self.encode_context(batch)
        B = ctx.shape[0]
        root_base = self.root_queries.unsqueeze(0).expand(B, -1, -1) + self.context_to_root(ctx).unsqueeze(1)
        z = self.root_mlp(root_base)
        root_logits = self.root_logit(z).squeeze(-1)
        root_prob = torch.softmax(root_logits, dim=-1)
        prior_logits = self.prior_logit(z).squeeze(-1)
        prior_prob = torch.softmax(prior_logits, dim=-1)
        sig = self.sig_head(z)
        obs = self.obs_head(z)
        diff = obs[:, :, None, :] - obs[:, None, :, :]
        C = torch.exp(-torch.sum(diff * diff, dim=-1) / max(self.tau_obs, 1e-8))
        eye = torch.eye(self.K, dtype=C.dtype, device=C.device).unsqueeze(0)
        C = torch.maximum(C, eye)
        margin = self.margin_head(z)
        utility_hat = self.utility_head(ctx).squeeze(-1)
        return {
            "root_embedding": z,
            "root_logits": root_logits,
            "root_prob": root_prob,
            "prior_prob": prior_prob,
            "root_signature": sig,
            "obs_embedding": obs,
            "C": C,
            "margin": margin,
            "utility_hat": utility_hat,
        }
