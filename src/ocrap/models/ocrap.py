from __future__ import annotations

import torch
from torch import nn

from .encoders import FlatFeatureLayout, MLPEncoder, StructuredTokenEncoder


class OCRAPModel(nn.Module):
    """Neural OC-RAP model with a learned root-query decoder.

    The previous implementation predicted all roots from one global scene
    embedding.  That was useful as a compact smoke model but did not match the
    paper's pipeline, where learned root queries cross-attend to scene-prefix
    tokens and each root obtains its own observation embedding and recovery
    margins.  This module keeps the MLP fallback, but the default structured
    path now implements the paper-aligned decoder:

    scene/prefix tokens -> K learned root queries -> root probabilities,
    post-prefix observation embeddings, recovery signatures, and per-option
    margins conditioned on recovery-option embeddings.
    """

    def __init__(
        self,
        input_dim: int,
        num_roots: int = 8,
        num_options: int = 24,
        d_model: int = 128,
        d_obs: int = 64,
        tau_obs: float = 1.0,
        encoder_type: str = "mlp",
        feature_layout: dict | None = None,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        d_signature: int = 0,
        d_future_signature: int = 0,
        option_feature_dim: int = 0,
        direct_recovery_value_head: bool = False,
        direct_recovery_value_pooling: str = "scene",
    ):
        super().__init__()
        self.num_roots = int(num_roots)
        self.num_options = int(num_options)
        self.d_obs = int(d_obs)
        self.tau_obs = float(max(tau_obs, 1e-6))
        self.encoder_type = str(encoder_type)
        self.feature_layout = feature_layout or {}
        self.d_model = int(d_model)
        self.d_signature = int(d_signature)
        self.d_future_signature = int(d_future_signature)
        self.option_feature_dim = int(option_feature_dim)
        self.direct_recovery_value_head = bool(direct_recovery_value_head)
        self.direct_recovery_value_pooling = str(direct_recovery_value_pooling or "scene").strip().lower()

        if self.encoder_type == "structured_transformer":
            layout = FlatFeatureLayout(**self.feature_layout)
            self.encoder = StructuredTokenEncoder(
                layout=layout,
                d_model=d_model,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
            )
        else:
            self.encoder = MLPEncoder(input_dim, d_model)

        # Root-query decoder: K learned latent root slots attend to the encoded
        # scene-prefix tokens.  For the MLP fallback, the single global embedding
        # is treated as a one-token memory.
        self.root_queries = nn.Parameter(torch.randn(1, self.num_roots, d_model) * 0.02)
        self.root_cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.root_self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.root_norm1 = nn.LayerNorm(d_model)
        self.root_norm2 = nn.LayerNorm(d_model)
        self.root_ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.root_norm3 = nn.LayerNorm(d_model)

        self.option_embeddings = nn.Parameter(torch.randn(1, 1, self.num_options, d_model) * 0.02)
        self.option_feature_proj = (
            nn.Sequential(nn.Linear(self.option_feature_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
            if self.option_feature_dim > 0
            else None
        )
        self.root_logit_head = nn.Linear(d_model, 1)
        self.margin_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.obs_embed_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, self.d_obs))
        self.utility_head = nn.Linear(d_model, 1)
        # v40 OC-UVRA: decouple the calibrated OC-MERO certificate from the
        # candidate preference signal.  The head predicts a bounded deployable
        # recovery value and aleatoric log-variance for each executable prefix.
        # This avoids forcing R_dep, DRS, and the oracle gap to absorb a separate
        # counterfactual ranking objective.
        # v41 OC-CAVA: the v40 head consumed only the frozen CLS token.  The
        # measured validation loss stayed close to a uniform candidate ranking,
        # which means the frozen v39 CLS representation discarded much of the
        # prefix-specific signal needed for counterfactual action comparison.
        # ``candidate_concat`` exposes the encoded ego/prefix/macro/state/control
        # tokens to a small trainable adapter without changing any OC-MERO head.
        direct_in_dim = d_model
        if self.direct_recovery_value_pooling in {"candidate_concat", "prefix_concat", "action_concat"}:
            direct_in_dim = 6 * d_model
        self.direct_value_head = (
            nn.Sequential(
                nn.LayerNorm(direct_in_dim),
                nn.Linear(direct_in_dim, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 2),
            )
            if self.direct_recovery_value_head
            else None
        )
        self.root_signature_head = nn.Linear(d_model, self.d_signature) if self.d_signature > 0 else None
        self.root_future_signature_head = nn.Linear(d_model, self.d_future_signature) if self.d_future_signature > 0 else None

    def _scene_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if self.encoder_type == "structured_transformer" and hasattr(self.encoder, "forward_tokens"):
            return self.encoder.forward_tokens(x)  # type: ignore[attr-defined]
        h = self.encoder(x)
        return h.unsqueeze(1)

    def _decode_roots(self, memory: torch.Tensor) -> torch.Tensor:
        B = memory.shape[0]
        q0 = self.root_queries.expand(B, -1, -1)
        q, _ = self.root_cross_attn(q0, memory, memory, need_weights=False)
        q = self.root_norm1(q0 + q)
        qs, _ = self.root_self_attn(q, q, q, need_weights=False)
        q = self.root_norm2(q + qs)
        q = self.root_norm3(q + self.root_ffn(q))
        return q

    def _option_tokens(self, x: torch.Tensor, option_features: torch.Tensor | None) -> torch.Tensor:
        learned = self.option_embeddings.expand(x.shape[0], self.num_roots, -1, -1)
        if option_features is None or self.option_feature_proj is None:
            return learned
        opt_feat = option_features.to(dtype=x.dtype, device=x.device)
        if opt_feat.dim() == 2:
            opt_feat = opt_feat.unsqueeze(0).expand(x.shape[0], -1, -1)
        if opt_feat.shape[1] != self.num_options:
            # Keep inference robust when an older checkpoint/sample has a shorter
            # option list: pad or truncate to the checkpoint geometry.
            fixed = torch.zeros((opt_feat.shape[0], self.num_options, opt_feat.shape[-1]), dtype=opt_feat.dtype, device=opt_feat.device)
            n = min(self.num_options, opt_feat.shape[1])
            fixed[:, :n] = opt_feat[:, :n]
            opt_feat = fixed
        semantic = self.option_feature_proj(opt_feat).unsqueeze(1).expand(-1, self.num_roots, -1, -1)
        return learned + semantic

    def forward(self, x: torch.Tensor, option_features: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        memory = self._scene_tokens(x)
        scene_token = memory[:, 0]
        root_tokens = self._decode_roots(memory)

        root_logits = self.root_logit_head(root_tokens).squeeze(-1)
        obs_embeddings = self.obs_embed_head(root_tokens)

        root_expand = root_tokens.unsqueeze(2).expand(-1, -1, self.num_options, -1)
        opt_expand = self._option_tokens(x, option_features)
        margins = self.margin_head(torch.cat([root_expand, opt_expand], dim=-1)).squeeze(-1)

        diff = obs_embeddings.unsqueeze(2) - obs_embeddings.unsqueeze(1)
        dist2 = (diff * diff).mean(dim=-1)
        C = torch.exp(-dist2 / self.tau_obs).clamp(0.0, 1.0)
        eye = torch.eye(self.num_roots, dtype=C.dtype, device=C.device).unsqueeze(0)
        C = C * (1 - eye) + eye

        out: dict[str, torch.Tensor] = {
            "root_logits": root_logits,
            "margins": margins,
            "obs_embeddings": obs_embeddings,
            "c_star": C,
            "utility": self.utility_head(scene_token).squeeze(-1),
        }
        if self.direct_value_head is not None:
            direct_features = scene_token
            if self.direct_recovery_value_pooling in {"candidate_concat", "prefix_concat", "action_concat"}:
                # Token order from StructuredTokenEncoder is
                # [CLS, ego, prefix_param, macro+scalar, prefix_state, control, ...].
                # For the MLP fallback, repeat the only token to retain geometry.
                if memory.shape[1] >= 6:
                    direct_features = torch.cat([memory[:, i] for i in range(6)], dim=-1)
                else:
                    direct_features = scene_token.repeat(1, 6)
            direct = self.direct_value_head(direct_features)
            out["direct_recovery_value_logit"] = direct[:, 0]
            out["direct_recovery_value_logvar"] = direct[:, 1].clamp(-7.0, 2.0)
        if self.root_signature_head is not None:
            out["root_signature"] = self.root_signature_head(root_tokens)
        if self.root_future_signature_head is not None:
            out["root_future_signature"] = self.root_future_signature_head(root_tokens)
        return out
