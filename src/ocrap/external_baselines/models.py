from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class ExternalBaselineBatch:
    x: torch.Tensor
    mask: torch.Tensor
    target_index: torch.Tensor
    utility: torch.Tensor
    hard: torch.Tensor
    harm: torch.Tensor
    r_orc: torch.Tensor
    r_dep: torch.Tensor
    feasible: torch.Tensor
    branch_margins: torch.Tensor | None = None
    root_features: torch.Tensor | None = None
    root_probs: torch.Tensor | None = None
    root_valid: torch.Tensor | None = None
    option_valid: torch.Tensor | None = None


class ResidualMLP(nn.Module):
    """Four-layer residual MLP used by the Waymax BC baseline.

    Waymax reports a route-conditioned BC planner that reuses a Wayformer-style
    attention encoder and appends a four-layer residual MLP action head.  This
    module keeps that residual action head explicit instead of hiding it in a
    single linear projection.
    """

    def __init__(self, d_model: int, hidden_dim: int = 128, num_layers: int = 4, dropout: float = 0.1, out_dim: int = 1) -> None:
        super().__init__()
        self.in_proj = nn.Linear(d_model, hidden_dim)
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Dropout(dropout),
                )
                for _ in range(int(num_layers))
            ]
        )
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = self.in_proj(h)
        for block in self.layers:
            z = z + block(z)
        return self.out(F.gelu(z))


class ScalarHeads(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.utility = nn.Linear(d_model, 1)
        self.hard = nn.Linear(d_model, 1)
        self.harm = nn.Linear(d_model, 1)
        self.r_orc = nn.Linear(d_model, 1)
        self.r_dep = nn.Linear(d_model, 1)

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "utility": self.utility(h).squeeze(-1),
            "hard": self.hard(h).squeeze(-1),
            "harm": self.harm(h).squeeze(-1),
            "r_orc": self.r_orc(h).squeeze(-1),
            "r_dep": self.r_dep(h).squeeze(-1),
        }


class WayformerRouteBC(nn.Module):
    """Route-conditioned behavior cloning baseline following Waymax.

    OC-RAP samples already contain vectorized ego, agent, map, route and prefix
    features.  We therefore implement the paper baseline at the policy level:
    early-fusion attention over candidate action tokens, optional latent query
    attention for Wayformer-style compression, and a 4-layer residual MLP that
    maximizes likelihood of the logged/nearest expert action.
    """

    def __init__(
        self,
        input_dim: int,
        max_candidates: int = 32,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.15,
        mlp_hidden: int = 128,
        mlp_layers: int = 4,
        num_latents: int = 0,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.max_candidates = int(max_candidates)
        self.d_model = int(d_model)
        self.num_latents = int(num_latents)
        self.token_proj = nn.Sequential(nn.LayerNorm(self.input_dim), nn.Linear(self.input_dim, d_model), nn.GELU(), nn.Dropout(dropout))
        self.pos = nn.Parameter(torch.zeros(1, self.max_candidates, d_model))
        self.type_route = nn.Parameter(torch.zeros(1, 1, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=int(num_heads),
            dim_feedforward=4 * d_model,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(num_layers))
        if self.num_latents > 0:
            self.latents = nn.Parameter(torch.randn(1, self.num_latents, d_model) * 0.02)
            self.latent_xattn = nn.MultiheadAttention(d_model, int(num_heads), dropout=float(dropout), batch_first=True)
            self.token_xattn = nn.MultiheadAttention(d_model, int(num_heads), dropout=float(dropout), batch_first=True)
            self.latent_norm = nn.LayerNorm(d_model)
        else:
            self.latents = None
            self.latent_xattn = None
            self.token_xattn = None
            self.latent_norm = None
        self.norm = nn.LayerNorm(d_model)
        self.policy_head = ResidualMLP(d_model, hidden_dim=int(mlp_hidden), num_layers=int(mlp_layers), dropout=float(dropout), out_dim=1)
        self.scalar_heads = ScalarHeads(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, **_: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.dim() != 3:
            raise ValueError(f"Expected x with shape [B,N,D], got {tuple(x.shape)}")
        B, N, _ = x.shape
        h = self.token_proj(x)
        pos = self._position(N)
        h = h + pos + self.type_route
        key_padding_mask = None if mask is None else ~mask.bool()
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        if self.num_latents > 0 and self.latents is not None:
            lat = self.latents.expand(B, -1, -1)
            lat, _ = self.latent_xattn(lat, h, h, key_padding_mask=key_padding_mask, need_weights=False)
            lat = self.latent_norm(lat)
            h2, _ = self.token_xattn(h, lat, lat, need_weights=False)
            h = h + h2
        h = self.norm(h)
        logits = self.policy_head(h).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(~mask.bool(), -1.0e4)
        out = {"logits": logits}
        out.update(self.scalar_heads(h))
        return out

    def _position(self, N: int) -> torch.Tensor:
        if N > self.pos.shape[1]:
            extra = self.pos[:, -1:, :].expand(1, N - self.pos.shape[1], -1)
            return torch.cat([self.pos, extra], dim=1)[:, :N]
        return self.pos[:, :N]


class LevelKDecoderBlock(nn.Module):
    """One GameFormer level-k interaction decoder block.

    It keeps GameFormer's essential recurrence while avoiding dataset-specific
    continuous trajectory tensors: previous-level future proxies are first
    processed by self-attention, then fused with the scene context and candidate
    query content, and finally refined by another self-attention block.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        fut_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=int(num_heads), dim_feedforward=4 * d_model, dropout=float(dropout), activation="gelu", batch_first=True, norm_first=True)
        resp_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=int(num_heads), dim_feedforward=4 * d_model, dropout=float(dropout), activation="gelu", batch_first=True, norm_first=True)
        self.future_encoder = nn.TransformerEncoder(fut_layer, num_layers=1)
        self.response_encoder = nn.TransformerEncoder(resp_layer, num_layers=1)
        self.fuse = nn.Sequential(nn.LayerNorm(3 * d_model), nn.Linear(3 * d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, d_model))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query: torch.Tensor, scene: torch.Tensor, future: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        key_padding_mask = None if mask is None else ~mask.bool()
        f = self.future_encoder(future, src_key_padding_mask=key_padding_mask)
        fused = self.fuse(torch.cat([query, scene, f], dim=-1))
        z = self.response_encoder(query + fused, src_key_padding_mask=key_padding_mask)
        return self.norm(query + z)


class GameFormerLevelK(nn.Module):
    """GameFormer-style interactive prediction/planning baseline.

    This is not a one-layer candidate scorer.  It retains the GameFormer core:
    Transformer scene encoder, learnable modality/candidate query content,
    level-0 cross-attention decoding, and iterative level-k decoders where each
    level reacts to futures predicted by the previous level.  OC-RAP's roots and
    recovery margins are used as the branch/future proxy that is available in the
    constructed dataset.
    """

    def __init__(
        self,
        input_dim: int,
        max_candidates: int = 32,
        d_model: int = 256,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.15,
        num_levels: int = 4,
        root_feature_dim: int = 18,
        num_roots: int = 10,
        num_options: int = 12,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.max_candidates = int(max_candidates)
        self.d_model = int(d_model)
        self.num_levels = int(num_levels)
        self.num_roots = int(num_roots)
        self.num_options = int(num_options)
        self.root_feature_dim = int(root_feature_dim)
        self.scene_proj = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, d_model), nn.GELU(), nn.Dropout(dropout))
        self.pos = nn.Parameter(torch.zeros(1, self.max_candidates, d_model))
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=int(num_heads), dim_feedforward=4 * d_model, dropout=float(dropout), activation="gelu", batch_first=True, norm_first=True)
        self.scene_encoder = nn.TransformerEncoder(enc_layer, num_layers=int(num_layers))
        self.modality = nn.Parameter(torch.randn(1, self.max_candidates, d_model) * 0.02)
        self.level0_cross = nn.MultiheadAttention(d_model, int(num_heads), dropout=float(dropout), batch_first=True)
        self.level0_norm = nn.LayerNorm(d_model)
        # root feature + m_star over recovery options + root probability/valid flag.
        branch_in = self.root_feature_dim + self.num_options + 2
        self.branch_point = nn.Sequential(nn.LayerNorm(branch_in), nn.Linear(branch_in, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, d_model))
        self.branch_pool = nn.Linear(d_model, d_model)
        self.future_from_level = nn.Sequential(nn.LayerNorm(d_model + 5), nn.Linear(d_model + 5, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, d_model))
        self.level_blocks = nn.ModuleList([LevelKDecoderBlock(d_model, int(num_heads), float(dropout)) for _ in range(max(self.num_levels - 1, 0))])
        self.level_policy_heads = nn.ModuleList([nn.Linear(d_model, 1) for _ in range(self.num_levels)])
        self.scalar_heads = ScalarHeads(d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        branch_margins: torch.Tensor | None = None,
        root_features: torch.Tensor | None = None,
        root_probs: torch.Tensor | None = None,
        root_valid: torch.Tensor | None = None,
        **_: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if x.dim() != 3:
            raise ValueError(f"Expected x with shape [B,N,D], got {tuple(x.shape)}")
        B, N, _ = x.shape
        key_padding_mask = None if mask is None else ~mask.bool()
        scene = self.scene_proj(x) + self._position(N)
        scene = self.scene_encoder(scene, src_key_padding_mask=key_padding_mask)
        q = scene + self._modality(N).expand(B, -1, -1)
        z, _ = self.level0_cross(q, scene, scene, key_padding_mask=key_padding_mask, need_weights=False)
        q = self.level0_norm(q + z)
        branch = self._encode_branch(B, N, x.device, branch_margins, root_features, root_probs, root_valid)
        level_logits: list[torch.Tensor] = []
        # Level 0 independent policy.
        logits0 = self.level_policy_heads[0](q).squeeze(-1)
        if mask is not None:
            logits0 = logits0.masked_fill(~mask.bool(), -1.0e4)
        level_logits.append(logits0)
        for k, block in enumerate(self.level_blocks, start=1):
            prev_prob = torch.softmax(level_logits[-1].masked_fill(~mask.bool(), -1.0e4) if mask is not None else level_logits[-1], dim=-1).unsqueeze(-1)
            # Proxy for previous-level future policy: query content + branch
            # distribution + predicted action probability.
            aux = torch.cat([prev_prob, torch.sigmoid(prev_prob), torch.zeros_like(prev_prob).expand(-1, -1, 3)], dim=-1)
            future = self.future_from_level(torch.cat([q + branch, aux], dim=-1))
            q = block(q, scene, future, mask)
            logits_k = self.level_policy_heads[k](q).squeeze(-1)
            if mask is not None:
                logits_k = logits_k.masked_fill(~mask.bool(), -1.0e4)
            level_logits.append(logits_k)
        h = self.norm(q + branch)
        logits = level_logits[-1]
        out = {"logits": logits, "level_logits": level_logits}
        out.update(self.scalar_heads(h))
        return out

    def _position(self, N: int) -> torch.Tensor:
        if N > self.pos.shape[1]:
            extra = self.pos[:, -1:, :].expand(1, N - self.pos.shape[1], -1)
            return torch.cat([self.pos, extra], dim=1)[:, :N]
        return self.pos[:, :N]

    def _modality(self, N: int) -> torch.Tensor:
        if N > self.modality.shape[1]:
            extra = self.modality[:, -1:, :].expand(1, N - self.modality.shape[1], -1)
            return torch.cat([self.modality, extra], dim=1)[:, :N]
        return self.modality[:, :N]

    def _encode_branch(
        self,
        B: int,
        N: int,
        device: torch.device,
        branch_margins: torch.Tensor | None,
        root_features: torch.Tensor | None,
        root_probs: torch.Tensor | None,
        root_valid: torch.Tensor | None,
    ) -> torch.Tensor:
        K, L, Fdim = self.num_roots, self.num_options, self.root_feature_dim
        if root_features is None:
            root_features = torch.zeros(B, N, K, Fdim, device=device)
        else:
            root_features = root_features.to(device=device, dtype=torch.float32)
            root_features = self._pad_last2(root_features, K, Fdim)
        if branch_margins is None:
            branch_margins = torch.zeros(B, N, K, L, device=device)
        else:
            branch_margins = branch_margins.to(device=device, dtype=torch.float32)
            branch_margins = self._pad_last2(branch_margins, K, L)
            branch_margins = torch.nan_to_num(branch_margins, nan=0.0, posinf=5.0, neginf=-5.0).clamp(-5.0, 5.0)
        if root_probs is None:
            root_probs = torch.full((B, N, K), 1.0 / max(K, 1), device=device)
        else:
            root_probs = self._pad_1d(root_probs.to(device=device, dtype=torch.float32), K).clamp_min(0.0)
        if root_valid is None:
            root_valid = torch.ones(B, N, K, device=device, dtype=torch.float32)
        else:
            root_valid = self._pad_1d(root_valid.to(device=device, dtype=torch.float32), K)
        root_probs = root_probs * (root_valid > 0.5).float()
        root_probs = root_probs / root_probs.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        point = torch.cat([root_features, branch_margins, root_probs.unsqueeze(-1), root_valid.unsqueeze(-1)], dim=-1)
        enc = self.branch_point(point)
        pooled = (enc * root_probs.unsqueeze(-1)).sum(dim=2)
        return self.branch_pool(pooled)

    @staticmethod
    def _pad_1d(x: torch.Tensor, n: int) -> torch.Tensor:
        if x.shape[-1] == n:
            return x
        out = x.new_zeros(*x.shape[:-1], n)
        m = min(n, x.shape[-1])
        if m > 0:
            out[..., :m] = x[..., :m]
        return out

    @staticmethod
    def _pad_last2(x: torch.Tensor, n0: int, n1: int) -> torch.Tensor:
        if x.shape[-2] == n0 and x.shape[-1] == n1:
            return x
        out = x.new_zeros(*x.shape[:-2], n0, n1)
        m0 = min(n0, x.shape[-2])
        m1 = min(n1, x.shape[-1])
        if m0 > 0 and m1 > 0:
            out[..., :m0, :m1] = x[..., :m0, :m1]
        return out


# Backwards-compatible alias for old checkpoints/configs.
CandidateSetTransformer = WayformerRouteBC


def build_model_from_cfg(input_dim: int, cfg: dict[str, Any]) -> nn.Module:
    bcfg = cfg.get("external_baselines", {}) if isinstance(cfg.get("external_baselines", {}), dict) else {}
    mcfg = bcfg.get("model", {}) if isinstance(bcfg.get("model", {}), dict) else {}
    baseline = str(bcfg.get("baseline", "route_bc_lite")).lower()
    arch = str(mcfg.get("arch", "")).lower()
    max_candidates = int(mcfg.get("max_candidates", bcfg.get("max_candidates", 32)))
    common = dict(
        input_dim=int(input_dim),
        max_candidates=max_candidates,
        d_model=int(mcfg.get("d_model", 256 if "gameformer" in baseline else 192)),
        num_layers=int(mcfg.get("num_layers", 3)),
        num_heads=int(mcfg.get("num_heads", 4)),
        dropout=float(mcfg.get("dropout", 0.15)),
    )
    if arch in {"gameformer", "gameformer_levelk", "levelk"} or "gameformer" in baseline:
        return GameFormerLevelK(
            **common,
            num_levels=int(mcfg.get("num_levels", 4)),
            root_feature_dim=int(mcfg.get("root_feature_dim", 18)),
            num_roots=int(mcfg.get("num_roots", cfg.get("num_roots", 10))),
            num_options=int(mcfg.get("num_options", cfg.get("num_recovery_options", 12))),
        )
    return WayformerRouteBC(
        **common,
        mlp_hidden=int(mcfg.get("mlp_hidden", 128)),
        mlp_layers=int(mcfg.get("mlp_layers", 4)),
        num_latents=int(mcfg.get("num_latents", 16)),
    )
