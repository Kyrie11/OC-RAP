from __future__ import annotations

import torch
from torch import nn

from .encoders import FlatFeatureLayout, MLPEncoder, StructuredTokenEncoder


class RecoverySetTournament(nn.Module):
    """Permutation-equivariant recovery-only set ranker.

    The candidate-level value score is deliberately excluded.  Each recovery
    token is projected from nominal-relative features, contextualised against
    the other recovery candidates in the same scene-time group, and scored by
    a shared head.  Nominal is pinned to zero and is not part of the tournament.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        hidden_dim = max(int(num_heads), int(hidden_dim))
        if hidden_dim % int(num_heads) != 0:
            hidden_dim = int(num_heads) * ((hidden_dim + int(num_heads) - 1) // int(num_heads))
        self.hidden_dim = int(hidden_dim)
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, int(num_heads), dropout=float(dropout), batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.score = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)

    def encode(
        self,
        relative_features: torch.Tensor,
        group_index: torch.Tensor | None,
        is_nominal: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return candidate-set context without changing proposal semantics.

        Nominal rows remain zero.  Recovery rows contain the frozen tournament
        representation used by the proposal head, which is substantially lower
        dimensional and more data-efficient than the raw relative feature vector.
        """
        context = relative_features.new_zeros((relative_features.shape[0], self.hidden_dim))
        if group_index is None or is_nominal is None or relative_features.shape[0] <= 1:
            return context
        groups = group_index.to(device=relative_features.device)
        groups = groups.reshape(-1, 1) if groups.dim() == 1 else groups.reshape(groups.shape[0], -1)
        nominal_mask = is_nominal.to(device=relative_features.device).reshape(-1) > 0.5
        if groups.shape[0] != relative_features.shape[0] or nominal_mask.shape[0] != relative_features.shape[0]:
            return context
        for key in torch.unique(groups, dim=0):
            idx = torch.where((groups == key.unsqueeze(0)).all(dim=1))[0]
            recs = idx[~nominal_mask[idx]]
            if recs.numel() == 0:
                continue
            token = self.input_proj(self.input_norm(relative_features[recs])).unsqueeze(0)
            attended, _ = self.attn(token, token, token, need_weights=False)
            token = self.norm1(token + attended)
            token = self.norm2(token + self.ffn(token)).squeeze(0)
            context[recs] = token.to(dtype=context.dtype)
        return context

    def score_from_context(
        self,
        context: torch.Tensor,
        group_index: torch.Tensor | None,
        is_nominal: torch.Tensor | None,
    ) -> torch.Tensor:
        scores = context.new_zeros((context.shape[0],))
        if group_index is None or is_nominal is None or context.shape[0] <= 1:
            return scores
        groups = group_index.to(device=context.device)
        groups = groups.reshape(-1, 1) if groups.dim() == 1 else groups.reshape(groups.shape[0], -1)
        nominal_mask = is_nominal.to(device=context.device).reshape(-1) > 0.5
        if groups.shape[0] != context.shape[0] or nominal_mask.shape[0] != context.shape[0]:
            return scores
        for key in torch.unique(groups, dim=0):
            idx = torch.where((groups == key.unsqueeze(0)).all(dim=1))[0]
            recs = idx[~nominal_mask[idx]]
            if recs.numel() == 0:
                continue
            group_scores = self.score(context[recs]).squeeze(-1)
            if group_scores.numel() > 1:
                group_scores = group_scores - group_scores.mean()
            scores[recs] = group_scores.to(dtype=scores.dtype)
        return scores

    def forward(
        self,
        relative_features: torch.Tensor,
        group_index: torch.Tensor | None,
        is_nominal: torch.Tensor | None,
    ) -> torch.Tensor:
        context = self.encode(relative_features, group_index, is_nominal)
        return self.score_from_context(context, group_index, is_nominal)


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
        direct_recovery_value_output: str = "probability",
        direct_recovery_value_regime_conditioning: bool = False,
        direct_recovery_value_num_regimes: int = 4,
        direct_recovery_value_regime_dim: int = 16,
        direct_recovery_opportunity_head: bool = False,
        direct_recovery_harm_head: bool = False,
        direct_recovery_value_experts: bool = False,
        direct_recovery_value_num_experts: int = 2,
        direct_recovery_value_expert_routing: str = "bucket",
        direct_recovery_value_router_temperature: float = 1.0,
        direct_recovery_value_router_pooling: str = "candidate",
        direct_recovery_expert_disagreement_penalty: float = 0.5,
        direct_recovery_set_context: bool = False,
        direct_recovery_set_context_hidden: int = 192,
        direct_recovery_set_context_dropout: float = 0.1,
        direct_recovery_preference_head: bool = False,
        direct_recovery_preference_hidden: int = 96,
        direct_recovery_preference_dropout: float = 0.05,
        direct_recovery_preference_context: bool = False,
        direct_recovery_preference_context_hidden: int = 128,
        direct_recovery_relative_features_include_absolute: bool = True,
        direct_recovery_set_tournament: bool = False,
        direct_recovery_set_tournament_hidden: int = 48,
        direct_recovery_set_tournament_heads: int = 4,
        direct_recovery_set_tournament_dropout: float = 0.05,
        direct_recovery_set_tournament_replace_base: bool = True,
        direct_recovery_delta_head: bool = False,
        direct_recovery_delta_regime_experts: bool = False,
        direct_recovery_delta_policy_features: bool = False,
        direct_recovery_delta_hidden: int = 128,
        direct_recovery_delta_dropout: float = 0.05,
        direct_recovery_delta_initial_logvar: float = -4.605170186,
        direct_recovery_delta_mode: str = "gaussian",
        direct_recovery_evidence_calibrator: bool = False,
        direct_recovery_evidence_calibrator_hidden: int = 8,
        direct_recovery_evidence_calibrator_scale: float = 0.25,
        direct_recovery_evidence_calibrator_mode: str = "center_width",
        direct_recovery_evidence_calibrator_context: bool = False,
        direct_recovery_evidence_calibrator_context_detach: bool = True,
        direct_recovery_evidence_calibrator_context_source: str = "relative",
        direct_recovery_evidence_calibrator_shared: bool = False,
        direct_recovery_evidence_calibrator_regime_scale: float = 0.25,
        direct_recovery_evidence_unified_experts: bool = False,
        direct_recovery_evidence_component_heads: bool = False,
        direct_recovery_evidence_component_count: int = 3,
        direct_recovery_evidence_component_scale: float = 6.0,
        direct_recovery_evidence_component_reliability: str | tuple[float, ...] = "",
        direct_recovery_evidence_concord: bool = False,
        direct_recovery_evidence_consensus_disagreement_penalty: float = 0.15,
        direct_recovery_evidence_admission_head: bool = False,
        direct_recovery_evidence_admission_scale: float = 2.0,
        direct_recovery_evidence_admission_bounded: bool = True,
        direct_recovery_evidence_admission_prior_mode: str = "risk_centered",
        direct_recovery_evidence_slack_temperature: float = 0.025,
        direct_recovery_evidence_slack_penalty: float = 1.0,
        direct_recovery_evidence_frontier: bool = False,
        direct_recovery_evidence_component_prior_logit: float = -2.0,
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
        self.direct_recovery_value_output = str(direct_recovery_value_output or "probability").strip().lower()
        self.direct_recovery_value_regime_conditioning = bool(direct_recovery_value_regime_conditioning)
        self.direct_recovery_value_num_regimes = max(1, int(direct_recovery_value_num_regimes))
        self.direct_recovery_value_regime_dim = max(1, int(direct_recovery_value_regime_dim))
        self.direct_recovery_opportunity_head = bool(direct_recovery_opportunity_head)
        self.direct_recovery_harm_head = bool(direct_recovery_harm_head)
        self.direct_recovery_value_experts = bool(direct_recovery_value_experts)
        self.direct_recovery_value_num_experts = max(1, int(direct_recovery_value_num_experts))
        self.direct_recovery_value_expert_routing = str(
            direct_recovery_value_expert_routing or "bucket"
        ).strip().lower()
        self.direct_recovery_value_router_temperature = float(
            max(direct_recovery_value_router_temperature, 1.0e-3)
        )
        self.direct_recovery_value_router_pooling = str(
            direct_recovery_value_router_pooling or "candidate"
        ).strip().lower()
        self.direct_recovery_expert_disagreement_penalty = float(
            max(direct_recovery_expert_disagreement_penalty, 0.0)
        )
        self.direct_recovery_set_context = bool(direct_recovery_set_context)
        self.direct_recovery_set_context_hidden = max(16, int(direct_recovery_set_context_hidden))
        self.direct_recovery_set_context_dropout = float(max(0.0, direct_recovery_set_context_dropout))
        self.direct_recovery_preference_head = bool(direct_recovery_preference_head)
        self.direct_recovery_preference_hidden = max(16, int(direct_recovery_preference_hidden))
        self.direct_recovery_preference_dropout = float(max(0.0, direct_recovery_preference_dropout))
        self.direct_recovery_preference_context = bool(direct_recovery_preference_context)
        self.direct_recovery_preference_context_hidden = max(16, int(direct_recovery_preference_context_hidden))
        self.direct_recovery_relative_features_include_absolute = bool(direct_recovery_relative_features_include_absolute)
        self.direct_recovery_set_tournament = bool(direct_recovery_set_tournament)
        self.direct_recovery_set_tournament_hidden = max(16, int(direct_recovery_set_tournament_hidden))
        self.direct_recovery_set_tournament_heads = max(1, int(direct_recovery_set_tournament_heads))
        self.direct_recovery_set_tournament_dropout = float(max(0.0, direct_recovery_set_tournament_dropout))
        self.direct_recovery_set_tournament_replace_base = bool(direct_recovery_set_tournament_replace_base)
        self.direct_recovery_delta_head = bool(direct_recovery_delta_head)
        self.direct_recovery_delta_regime_experts = bool(direct_recovery_delta_regime_experts)
        self.direct_recovery_delta_policy_features = bool(direct_recovery_delta_policy_features)
        self.direct_recovery_delta_hidden = max(16, int(direct_recovery_delta_hidden))
        self.direct_recovery_delta_dropout = float(max(0.0, direct_recovery_delta_dropout))
        self.direct_recovery_delta_initial_logvar = float(direct_recovery_delta_initial_logvar)
        self.direct_recovery_delta_mode = str(direct_recovery_delta_mode or "gaussian").strip().lower()
        self.direct_recovery_evidence_calibrator = bool(direct_recovery_evidence_calibrator)
        self.direct_recovery_evidence_calibrator_hidden = max(4, int(direct_recovery_evidence_calibrator_hidden))
        self.direct_recovery_evidence_calibrator_scale = float(max(0.0, direct_recovery_evidence_calibrator_scale))
        self.direct_recovery_evidence_calibrator_mode = str(
            direct_recovery_evidence_calibrator_mode or "center_width"
        ).strip().lower()
        self.direct_recovery_evidence_calibrator_context = bool(direct_recovery_evidence_calibrator_context)
        self.direct_recovery_evidence_calibrator_context_detach = bool(
            direct_recovery_evidence_calibrator_context_detach
        )
        self.direct_recovery_evidence_calibrator_context_source = str(
            direct_recovery_evidence_calibrator_context_source or "relative"
        ).strip().lower()
        self.direct_recovery_evidence_calibrator_shared = bool(
            direct_recovery_evidence_calibrator_shared
        )
        self.direct_recovery_evidence_calibrator_regime_scale = float(
            max(0.0, direct_recovery_evidence_calibrator_regime_scale)
        )
        self.direct_recovery_evidence_unified_experts = bool(
            direct_recovery_evidence_unified_experts
        )
        self.direct_recovery_evidence_component_heads = bool(
            direct_recovery_evidence_component_heads
        )
        self.direct_recovery_evidence_component_count = max(3, min(5, int(
            direct_recovery_evidence_component_count
        )))
        self.direct_recovery_evidence_component_scale = float(
            max(0.0, direct_recovery_evidence_component_scale)
        )
        raw_component_reliability = direct_recovery_evidence_component_reliability
        if isinstance(raw_component_reliability, str):
            reliability_values = [
                float(x.strip()) for x in raw_component_reliability.split(",") if x.strip()
            ]
        else:
            reliability_values = [float(x) for x in raw_component_reliability]
        if not reliability_values:
            reliability_values = [1.0] * self.direct_recovery_evidence_component_count
        if len(reliability_values) < self.direct_recovery_evidence_component_count:
            reliability_values.extend(
                [1.0] * (self.direct_recovery_evidence_component_count - len(reliability_values))
            )
        reliability_values = [
            min(1.0, max(0.0, float(x)))
            for x in reliability_values[: self.direct_recovery_evidence_component_count]
        ]
        self.direct_recovery_evidence_component_reliability = tuple(reliability_values)
        self.register_buffer(
            "_direct_recovery_evidence_component_reliability",
            torch.tensor(reliability_values, dtype=torch.float32),
            persistent=False,
        )
        self.direct_recovery_evidence_concord = bool(direct_recovery_evidence_concord)
        self.direct_recovery_evidence_consensus_disagreement_penalty = float(
            max(0.0, direct_recovery_evidence_consensus_disagreement_penalty)
        )
        self.direct_recovery_evidence_admission_head = bool(
            direct_recovery_evidence_admission_head
        )
        self.direct_recovery_evidence_admission_scale = float(
            max(0.0, direct_recovery_evidence_admission_scale)
        )
        self.direct_recovery_evidence_admission_bounded = bool(
            direct_recovery_evidence_admission_bounded
        )
        self.direct_recovery_evidence_admission_prior_mode = str(
            direct_recovery_evidence_admission_prior_mode or "risk_centered"
        ).strip().lower()
        if self.direct_recovery_evidence_admission_prior_mode not in {
            "risk_centered", "benefit_only", "safety_slack"
        }:
            raise ValueError(
                "Unsupported direct_recovery_evidence_admission_prior_mode="
                f"{direct_recovery_evidence_admission_prior_mode!r}"
            )
        self.direct_recovery_evidence_slack_temperature = float(
            max(1.0e-6, direct_recovery_evidence_slack_temperature)
        )
        self.direct_recovery_evidence_slack_penalty = float(
            max(0.0, direct_recovery_evidence_slack_penalty)
        )
        self.direct_recovery_evidence_frontier = bool(direct_recovery_evidence_frontier)
        self.direct_recovery_evidence_component_prior_logit = float(
            direct_recovery_evidence_component_prior_logit
        )
        if self.direct_recovery_evidence_calibrator_mode not in {
            "center_width", "simplex_context", "dual_tail_context"
        }:
            raise ValueError(
                "Unsupported direct_recovery_evidence_calibrator_mode="
                f"{direct_recovery_evidence_calibrator_mode!r}"
            )
        if self.direct_recovery_evidence_calibrator_context_source not in {"relative", "tournament"}:
            raise ValueError(
                "Unsupported direct_recovery_evidence_calibrator_context_source="
                f"{direct_recovery_evidence_calibrator_context_source!r}"
            )
        if (
            self.direct_recovery_evidence_calibrator_context
            and self.direct_recovery_evidence_calibrator_context_source == "tournament"
            and not self.direct_recovery_set_tournament
        ):
            raise ValueError("tournament evidence context requires direct_recovery_set_tournament=true")
        if self.direct_recovery_evidence_unified_experts and not self.direct_recovery_delta_regime_experts:
            raise ValueError(
                "unified expert evidence requires direct_recovery_delta_regime_experts=true"
            )
        if self.direct_recovery_evidence_component_heads and not self.direct_recovery_evidence_unified_experts:
            raise ValueError(
                "component evidence heads currently require unified expert evidence"
            )
        if self.direct_recovery_evidence_concord and not self.direct_recovery_evidence_unified_experts:
            raise ValueError("CONCORD evidence requires unified frozen source experts")
        if self.direct_recovery_delta_mode not in {"gaussian", "ordinal_evidence"}:
            raise ValueError(f"Unsupported direct_recovery_delta_mode={direct_recovery_delta_mode!r}")
        if self.direct_recovery_value_expert_routing not in {
            "bucket",
            "hard_bucket",
            "soft_observation",
            "soft",
            "moe",
            "uniform",
            "uniform_robust",
            "robust_observation",
            "robust_ensemble",
            "risk_ensemble",
        }:
            raise ValueError(
                "Unsupported direct_recovery_value_expert_routing="
                f"{direct_recovery_value_expert_routing!r}"
            )
        if self.direct_recovery_value_output not in {"probability", "score"}:
            raise ValueError(f"Unsupported direct_recovery_value_output={direct_recovery_value_output!r}")
        if self.direct_recovery_value_router_pooling not in {"candidate", "scene", "shared_raw", "ego_shared_raw"}:
            raise ValueError(
                "Unsupported direct_recovery_value_router_pooling="
                f"{direct_recovery_value_router_pooling!r}"
            )

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
        self.direct_candidate_raw_dim = 0
        self.direct_candidate_feature_dim = 0
        self.direct_ego_feature_dim = 0
        if self.encoder_type == "structured_transformer":
            layout = FlatFeatureLayout(**self.feature_layout)
            self.direct_ego_feature_dim = int(layout.ego_dim)
            self.direct_candidate_feature_dim = int(
                layout.ego_dim + layout.prefix_param_dim + layout.num_macros + layout.scalar_dim
                + layout.prefix_flat_dim + layout.control_flat_dim
            )
        if self.direct_recovery_value_pooling in {"candidate_concat", "prefix_concat", "action_concat"}:
            direct_in_dim = 6 * d_model
        elif self.direct_recovery_value_pooling in {"candidate_concat_raw", "action_concat_raw", "certificate_action_adapter"}:
            # Certificate-preserving dual pathway: keep the frozen contextual
            # OC-MERO tokens, but expose raw ego/prefix/macro/state/control
            # blocks to the trainable action-value adapter.  This prevents a
            # frozen certificate encoder from discarding counterfactual action
            # differences while leaving all calibrated certificate heads intact.
            if self.encoder_type != "structured_transformer":
                raise ValueError("candidate_concat_raw requires structured_transformer")
            self.direct_candidate_raw_dim = self.direct_candidate_feature_dim
            direct_in_dim = 6 * d_model + self.direct_candidate_raw_dim
        # Optional regime embedding retained only for controlled ablations. The
        # v44 OC-RAVA default disables it: deployment uses one observation-only
        # value/opportunity model and never receives the evaluation bucket as a
        # neural input. Bucket ids remain available solely to separate training
        # groups and to support an explicit leakage ablation if needed.
        self.direct_regime_embedding = (
            nn.Embedding(self.direct_recovery_value_num_regimes, self.direct_recovery_value_regime_dim)
            if self.direct_recovery_value_head and self.direct_recovery_value_regime_conditioning
            else None
        )
        if self.direct_recovery_value_router_pooling == "scene":
            direct_router_in_dim = d_model
        elif self.direct_recovery_value_router_pooling in {"shared_raw", "ego_shared_raw"}:
            if self.encoder_type != "structured_transformer" or self.direct_candidate_feature_dim <= 0:
                raise ValueError("shared raw router pooling requires structured_transformer")
            direct_router_in_dim = int(input_dim) - self.direct_candidate_feature_dim
            if self.direct_recovery_value_router_pooling == "ego_shared_raw":
                direct_router_in_dim += self.direct_ego_feature_dim
            if direct_router_in_dim <= 0:
                raise ValueError("shared raw router pooling produced an empty observation feature")
        else:
            direct_router_in_dim = direct_in_dim
        # v48.3 OC-TRAC-NASC: candidate value is intrinsically set-relative.
        # The adapter is permutation-equivariant within a scene-time candidate set,
        # explicitly anchored to nominal, and leaves the base pointwise path intact
        # when a complete candidate set is unavailable.
        self.direct_set_context_adapter = (
            nn.Sequential(
                nn.LayerNorm(4 * direct_in_dim),
                nn.Linear(4 * direct_in_dim, self.direct_recovery_set_context_hidden),
                nn.GELU(),
                nn.Dropout(self.direct_recovery_set_context_dropout),
                nn.Linear(self.direct_recovery_set_context_hidden, direct_in_dim),
                nn.LayerNorm(direct_in_dim),
            )
            if self.direct_recovery_value_head and self.direct_recovery_set_context
            else None
        )
        # Start conservatively so v47/v48 pointwise knowledge remains the initial
        # solution and the set residual is learned only when it improves ranking.
        self.direct_set_context_gate = (
            nn.Parameter(torch.tensor(-2.5)) if self.direct_set_context_adapter is not None else None
        )
        # v48.4 ZI-NASC: a warm-started checkpoint must initially reproduce the
        # inherited pointwise policy exactly.  A merely small sigmoid gate still
        # injects a random LayerNorm residual and can destroy candidate AUC before
        # the set adapter learns anything.  Zero-initialize the residual projection;
        # gradients still flow into it on the first update and into earlier adapter
        # layers thereafter.
        if self.direct_set_context_adapter is not None:
            residual_projection = self.direct_set_context_adapter[4]
            if isinstance(residual_projection, nn.Linear):
                nn.init.zeros_(residual_projection.weight)
                nn.init.zeros_(residual_projection.bias)
        if self.direct_regime_embedding is not None:
            direct_in_dim += self.direct_recovery_value_regime_dim
        direct_out_dim = 2 + int(self.direct_recovery_opportunity_head) + int(self.direct_recovery_harm_head)

        # v48.5 ECPR: ranking and admission are separate tasks.  The value head
        # estimates candidate-vs-nominal gain and uncertainty, while this
        # zero-initialized residual preference head learns only the within-set
        # ordering.  At warm start rank == value exactly, preserving v48.1/v48.4
        # checkpoint behaviour until ranking evidence is observed.
        self.direct_preference_adapter = (
            nn.Sequential(
                nn.LayerNorm(direct_in_dim),
                nn.Linear(direct_in_dim, self.direct_recovery_preference_hidden),
                nn.GELU(),
                nn.Dropout(self.direct_recovery_preference_dropout),
                nn.Linear(self.direct_recovery_preference_hidden, 1),
            )
            if self.direct_recovery_value_head and self.direct_recovery_preference_head
            else None
        )
        if self.direct_preference_adapter is not None:
            pref_projection = self.direct_preference_adapter[-1]
            if isinstance(pref_projection, nn.Linear):
                nn.init.zeros_(pref_projection.weight)
                nn.init.zeros_(pref_projection.bias)

        # v48.6 RPGC: do not let a set adapter rewrite the value representation.
        # A separate, zero-initialized relative-context residual augments only the
        # ranking score.  This preserves the v48.5 pointwise preference solution
        # while exposing candidate-minus-nominal and group summaries to ranking.
        relative_in_dim = (4 if self.direct_recovery_relative_features_include_absolute else 3) * direct_in_dim
        self.direct_preference_context_adapter = (
            nn.Sequential(
                nn.LayerNorm(relative_in_dim),
                nn.Linear(relative_in_dim, self.direct_recovery_preference_context_hidden),
                nn.GELU(),
                nn.Dropout(self.direct_recovery_preference_dropout),
                nn.Linear(self.direct_recovery_preference_context_hidden, 1),
            )
            if self.direct_recovery_value_head and self.direct_recovery_preference_context
            else None
        )
        if self.direct_preference_context_adapter is not None:
            ctx_projection = self.direct_preference_context_adapter[-1]
            if isinstance(ctx_projection, nn.Linear):
                nn.init.zeros_(ctx_projection.weight)
                nn.init.zeros_(ctx_projection.bias)

        # v48.11 CASTER: a standalone recovery-only set tournament replaces the
        # inherited candidate-level value ranking.  The latter has repeatedly
        # shown high candidate AUC but near-zero or negative groupwise top-1.
        self.direct_preference_set_ranker = (
            RecoverySetTournament(
                relative_in_dim,
                self.direct_recovery_set_tournament_hidden,
                self.direct_recovery_set_tournament_heads,
                self.direct_recovery_set_tournament_dropout,
            )
            if self.direct_recovery_value_head and self.direct_recovery_set_tournament
            else None
        )

        # Directly estimate candidate-minus-nominal PCD gain and uncertainty.
        # Subtracting two absolute predictions and adding their variances assumes
        # independent errors, even though both candidates share a scene encoder;
        # that assumption made v48.5 admission probabilities excessively diffuse.
        delta_input_dim = relative_in_dim + (2 if self.direct_recovery_delta_policy_features else 0)

        def _make_delta_adapter() -> nn.Sequential:
            adapter = nn.Sequential(
                nn.LayerNorm(delta_input_dim),
                nn.Linear(delta_input_dim, self.direct_recovery_delta_hidden),
                nn.GELU(),
                nn.Dropout(self.direct_recovery_delta_dropout),
                nn.Linear(self.direct_recovery_delta_hidden, 2),
            )
            projection = adapter[-1]
            if isinstance(projection, nn.Linear):
                nn.init.zeros_(projection.weight)
                nn.init.zeros_(projection.bias)
                with torch.no_grad():
                    projection.bias[1] = self.direct_recovery_delta_initial_logvar
            return adapter

        self.direct_delta_adapters = (
            nn.ModuleList([_make_delta_adapter(), _make_delta_adapter()])
            if self.direct_recovery_value_head and self.direct_recovery_delta_head
            and self.direct_recovery_delta_regime_experts
            else None
        )
        self.direct_delta_adapter = (
            _make_delta_adapter()
            if self.direct_recovery_value_head and self.direct_recovery_delta_head
            and not self.direct_recovery_delta_regime_experts
            else None
        )

        # v48.17 BRIDGE: retain the zero-initialised identity correction, but
        # optionally expose the frozen candidate-vs-nominal context representation.
        # v48.16 used only four scalars (source center/width and two rank margins);
        # certificate results showed that this input was insufficient to distinguish
        # beneficial, dead-zone, and harmful candidates with nearly identical source
        # scores.  The low-rank bottleneck keeps target adaptation small while adding
        # the observables needed for conditional correction.
        evidence_context_dim = 0
        if self.direct_recovery_evidence_calibrator_context:
            evidence_context_dim = (
                self.direct_preference_set_ranker.hidden_dim
                if self.direct_recovery_evidence_calibrator_context_source == "tournament"
                and self.direct_preference_set_ranker is not None
                else relative_in_dim
            )
        # v48.20 UNISON-BRIDGE: a single shared evidence model consumes both
        # frozen source experts, their consensus/disagreement, and the frozen
        # proposal context.  No bucket/regime id is exposed to this model.
        evidence_scalar_dim = 10 if self.direct_recovery_evidence_unified_experts else 4
        evidence_calibrator_input_dim = evidence_scalar_dim + evidence_context_dim
        if self.direct_recovery_evidence_component_heads:
            # benefit + all configured non-compensatory harm components.
            # v48.27 uses DRS/deployability/gap/hard-rule/harm-proxy (5).
            evidence_calibrator_output_dim = 1 + self.direct_recovery_evidence_component_count
        else:
            evidence_calibrator_output_dim = (
                3 if self.direct_recovery_evidence_calibrator_mode == "simplex_context" else 2
            )

        def _make_evidence_calibrator(output_dim: int | None = None) -> nn.Sequential:
            final_dim = evidence_calibrator_output_dim if output_dim is None else int(output_dim)
            adapter = nn.Sequential(
                nn.LayerNorm(evidence_calibrator_input_dim),
                nn.Linear(evidence_calibrator_input_dim, self.direct_recovery_evidence_calibrator_hidden),
                nn.GELU(),
                nn.Linear(self.direct_recovery_evidence_calibrator_hidden, final_dim),
            )
            projection = adapter[-1]
            if isinstance(projection, nn.Linear):
                nn.init.zeros_(projection.weight)
                nn.init.zeros_(projection.bias)
            return adapter

        self.direct_evidence_unified_calibrator = (
            _make_evidence_calibrator()
            if self.direct_recovery_value_head
            and self.direct_recovery_delta_head
            and self.direct_recovery_evidence_calibrator
            and self.direct_recovery_evidence_unified_experts
            and not self.direct_recovery_evidence_concord
            else None
        )
        # v48.21 CONCORD-BRIDGE decouples the sparse safe-benefit task from the
        # much denser component-risk task.  A single shared trunk in v48.20 let
        # the 45--54% harmful labels dominate the roughly 3% benefit labels.
        # Both adapters remain bucket-invariant and consume the same symmetric
        # frozen-expert/context representation; only their parameters are
        # decoupled to prevent negative transfer.
        self.direct_evidence_concord_benefit_calibrator = (
            _make_evidence_calibrator(1)
            if self.direct_recovery_value_head
            and self.direct_recovery_delta_head
            and self.direct_recovery_evidence_calibrator
            and self.direct_recovery_evidence_unified_experts
            and self.direct_recovery_evidence_concord
            else None
        )
        self.direct_evidence_concord_harm_calibrator = (
            _make_evidence_calibrator(
                self.direct_recovery_evidence_component_count
                if self.direct_recovery_evidence_component_heads else 1
            )
            if self.direct_recovery_value_head
            and self.direct_recovery_delta_head
            and self.direct_recovery_evidence_calibrator
            and self.direct_recovery_evidence_unified_experts
            and self.direct_recovery_evidence_concord
            else None
        )
        # v48.22 COVENANT-BRIDGE factorises three distinct hypotheses:
        # raw recovery benefit, componentwise harm, and final safe admissibility.
        # The admission adapter never receives a regime id.  It starts from a
        # detached benefit/risk prior and learns only the residual needed to
        # identify candidates that are simultaneously useful and safe.
        self.direct_evidence_concord_admission_calibrator = (
            _make_evidence_calibrator(1)
            if self.direct_recovery_value_head
            and self.direct_recovery_delta_head
            and self.direct_recovery_evidence_calibrator
            and self.direct_recovery_evidence_unified_experts
            and self.direct_recovery_evidence_concord
            and self.direct_recovery_evidence_admission_head
            else None
        )
        self.direct_evidence_calibrators = (
            nn.ModuleList([_make_evidence_calibrator(), _make_evidence_calibrator()])
            if self.direct_recovery_value_head and self.direct_recovery_delta_head
            and self.direct_recovery_evidence_calibrator
            and not self.direct_recovery_evidence_unified_experts
            else None
        )
        # v48.19 legacy partial pooling.  v48.20 disables this branch and uses
        # ``direct_evidence_unified_calibrator`` so inference never first chooses
        # a regime expert and then applies a regime-specific strategy.
        self.direct_evidence_shared_calibrator = (
            _make_evidence_calibrator()
            if self.direct_evidence_calibrators is not None
            and self.direct_recovery_evidence_calibrator_shared
            else None
        )

        def _make_direct_head() -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(direct_in_dim),
                nn.Linear(direct_in_dim, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, direct_out_dim),
            )

        # v45 OC-RAVE: Near-contact and Contact share the observation encoder but
        # use lightweight task experts.  The two datasets may contain the same
        # current observation with different pressure-future teachers, so one
        # unconditional head can receive contradictory labels even after groups
        # are separated.  In soft-observation mode every expert is evaluated and
        # continuously fused from observable scene/prefix features; dataset
        # buckets remain available only for loss stratification/calibration and
        # for the explicit legacy hard-routing ablation.  Safe does not use the
        # direct branch.
        self.direct_value_heads = (
            nn.ModuleList([_make_direct_head() for _ in range(self.direct_recovery_value_num_experts)])
            if self.direct_recovery_value_head and self.direct_recovery_value_experts
            else None
        )
        self.direct_value_head = (
            _make_direct_head()
            if self.direct_recovery_value_head and not self.direct_recovery_value_experts
            else None
        )
        # Observation-conditioned soft routing avoids treating a hand-authored
        # regime id as a policy input.  The router consumes the same observable
        # scene/prefix representation as the experts and produces continuous
        # mixture weights.  Legacy hard bucket routing remains available as an
        # explicit ablation and for loading older checkpoints.
        self.direct_value_router = (
            nn.Sequential(
                nn.LayerNorm(direct_router_in_dim),
                nn.Linear(direct_router_in_dim, max(16, d_model // 2)),
                nn.GELU(),
                nn.Linear(max(16, d_model // 2), self.direct_recovery_value_num_experts),
            )
            if self.direct_value_heads is not None
            and self.direct_recovery_value_expert_routing in {"soft_observation", "soft", "moe", "robust_observation"}
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

    def _candidate_raw_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw candidate-conditioned blocks used only by the value adapter."""
        if self.direct_candidate_raw_dim <= 0:
            return x[:, :0]
        # FlatFeatureLayout places ego, prefix parameters, macro/scalars, prefix
        # states and controls first, so the candidate-conditioned slice is
        # contiguous and does not include future/audit labels.
        return x[:, : self.direct_candidate_raw_dim]

    def _apply_direct_set_context(
        self,
        direct_features: torch.Tensor,
        group_index: torch.Tensor | None,
        is_nominal: torch.Tensor | None,
    ) -> torch.Tensor:
        """Nominal-anchored, permutation-equivariant candidate-set adapter.

        Each candidate is represented relative to the nominal candidate and to
        exchangeable mean/max summaries of recovery alternatives.  Singleton or
        malformed groups deliberately fall back to the original pointwise feature.
        """
        if self.direct_set_context_adapter is None or group_index is None or is_nominal is None:
            return direct_features
        if direct_features.shape[0] <= 1:
            return direct_features
        groups = group_index.to(device=direct_features.device)
        if groups.dim() == 1:
            groups = groups.reshape(-1, 1)
        else:
            groups = groups.reshape(groups.shape[0], -1)
        nominal_mask = is_nominal.to(device=direct_features.device).reshape(-1) > 0.5
        if groups.shape[0] != direct_features.shape[0] or nominal_mask.shape[0] != direct_features.shape[0]:
            return direct_features
        adapted = direct_features.clone()
        unique_groups = torch.unique(groups, dim=0)
        gate = torch.sigmoid(self.direct_set_context_gate) if self.direct_set_context_gate is not None else 1.0
        for key in unique_groups:
            idx = torch.where((groups == key.unsqueeze(0)).all(dim=1))[0]
            if idx.numel() <= 1:
                continue
            noms = idx[nominal_mask[idx]]
            recs = idx[~nominal_mask[idx]]
            if noms.numel() == 0 or recs.numel() == 0:
                continue
            nom_feat = direct_features[noms[0]:noms[0] + 1]
            rel = direct_features[idx] - nom_feat
            rec_rel = direct_features[recs] - nom_feat
            mean_rel = rec_rel.mean(dim=0, keepdim=True).expand(idx.numel(), -1)
            max_rel = rec_rel.max(dim=0, keepdim=True).values.expand(idx.numel(), -1)
            context_input = torch.cat([direct_features[idx], rel, mean_rel, max_rel], dim=-1)
            residual = self.direct_set_context_adapter(context_input)
            adapted[idx] = direct_features[idx] + gate * residual
        return adapted

    def _direct_group_relative_features(
        self,
        direct_features: torch.Tensor,
        group_index: torch.Tensor | None,
        is_nominal: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return permutation-equivariant candidate-relative features.

        The output concatenates the pointwise feature, candidate-minus-nominal,
        mean recovery-relative feature and max recovery-relative feature.  It is
        used only by the v48.6 preference-context and delta heads, so the value
        branch remains pointwise and warm-start compatible.
        """
        zeros = torch.zeros_like(direct_features)
        if self.direct_recovery_relative_features_include_absolute:
            relative = torch.cat([direct_features, zeros, zeros, zeros], dim=-1)
        else:
            relative = torch.cat([zeros, zeros, zeros], dim=-1)
        if group_index is None or is_nominal is None or direct_features.shape[0] <= 1:
            return relative
        groups = group_index.to(device=direct_features.device)
        groups = groups.reshape(-1, 1) if groups.dim() == 1 else groups.reshape(groups.shape[0], -1)
        nominal_mask = is_nominal.to(device=direct_features.device).reshape(-1) > 0.5
        if groups.shape[0] != direct_features.shape[0] or nominal_mask.shape[0] != direct_features.shape[0]:
            return relative
        out = relative.clone()
        for key in torch.unique(groups, dim=0):
            idx = torch.where((groups == key.unsqueeze(0)).all(dim=1))[0]
            if idx.numel() <= 1:
                continue
            noms = idx[nominal_mask[idx]]
            recs = idx[~nominal_mask[idx]]
            if noms.numel() == 0 or recs.numel() == 0:
                continue
            nominal = direct_features[noms[0]:noms[0] + 1]
            rel = direct_features[idx] - nominal
            rec_rel = direct_features[recs] - nominal
            mean_rel = rec_rel.mean(dim=0, keepdim=True).expand(idx.numel(), -1)
            max_rel = rec_rel.max(dim=0, keepdim=True).values.expand(idx.numel(), -1)
            if self.direct_recovery_relative_features_include_absolute:
                out[idx] = torch.cat([direct_features[idx], rel, mean_rel, max_rel], dim=-1)
            else:
                out[idx] = torch.cat([rel, mean_rel, max_rel], dim=-1)
        return out

    def _direct_outputs(
        self,
        memory: torch.Tensor,
        x: torch.Tensor,
        bucket_id: torch.Tensor | None,
        group_index: torch.Tensor | None = None,
        is_nominal: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Evaluate the policy-level recovery branch.

        v48 keeps experts observation-only and uses their disagreement as a
        conservative certificate: recovery gain and opportunity receive a lower
        confidence adjustment, while harmful-switch risk receives an upper
        confidence adjustment.  No hidden regime label is needed.
        """
        if self.direct_value_head is None and self.direct_value_heads is None:
            return {}
        scene_token = memory[:, 0]
        direct_features = scene_token
        if self.direct_recovery_value_pooling in {
            "candidate_concat", "prefix_concat", "action_concat",
            "candidate_concat_raw", "action_concat_raw", "certificate_action_adapter",
        }:
            if memory.shape[1] >= 6:
                direct_features = torch.cat([memory[:, i] for i in range(6)], dim=-1)
            else:
                direct_features = scene_token.repeat(1, 6)
            if self.direct_recovery_value_pooling in {
                "candidate_concat_raw", "action_concat_raw", "certificate_action_adapter"
            }:
                direct_features = torch.cat([direct_features, self._candidate_raw_features(x)], dim=-1)
        # Preserve an untouched pointwise path for the ranking/delta heads.  The
        # legacy NASC adapter may still be enabled as an ablation for the value
        # experts, but cannot silently contaminate the new relative certificate.
        pointwise_direct_features = direct_features
        direct_features = self._apply_direct_set_context(pointwise_direct_features, group_index, is_nominal)
        if self.direct_recovery_value_router_pooling == "shared_raw":
            router_features = x[:, self.direct_candidate_feature_dim:]
        elif self.direct_recovery_value_router_pooling == "ego_shared_raw":
            router_features = torch.cat(
                [x[:, : self.direct_ego_feature_dim], x[:, self.direct_candidate_feature_dim:]],
                dim=-1,
            )
        elif self.direct_recovery_value_router_pooling == "scene":
            router_features = scene_token
        else:
            router_features = direct_features
        if self.direct_regime_embedding is not None:
            if bucket_id is None:
                bucket_id = torch.full(
                    (x.shape[0],), min(3, self.direct_recovery_value_num_regimes - 1),
                    dtype=torch.long, device=x.device,
                )
            regime_id = bucket_id.to(device=x.device, dtype=torch.long).reshape(-1)
            regime_id = regime_id.clamp(0, self.direct_recovery_value_num_regimes - 1)
            regime_features = self.direct_regime_embedding(regime_id)
            direct_features = torch.cat([direct_features, regime_features], dim=-1)
            pointwise_direct_features = torch.cat([pointwise_direct_features, regime_features], dim=-1)

        relative_features = self._direct_group_relative_features(
            pointwise_direct_features, group_index, is_nominal
        )

        out: dict[str, torch.Tensor] = {}
        rank_base: torch.Tensor | None = None
        if self.direct_value_heads is not None:
            all_direct = torch.stack([head(direct_features) for head in self.direct_value_heads], dim=1)
            routing = self.direct_recovery_value_expert_routing
            if routing in {"bucket", "hard_bucket"}:
                if bucket_id is None:
                    bucket_id = torch.ones((x.shape[0],), dtype=torch.long, device=x.device)
                expert_idx = (bucket_id.to(device=x.device, dtype=torch.long).reshape(-1) - 1)
                expert_idx = expert_idx.clamp(0, self.direct_recovery_value_num_experts - 1)
                weights = torch.nn.functional.one_hot(
                    expert_idx, num_classes=self.direct_recovery_value_num_experts
                ).to(dtype=all_direct.dtype)
                direct = (all_direct * weights.unsqueeze(-1)).sum(dim=1)
                rank_base = direct[:, 0]
            elif routing in {"uniform", "uniform_robust", "robust_ensemble", "risk_ensemble"}:
                weights = torch.full(
                    (x.shape[0], self.direct_recovery_value_num_experts),
                    1.0 / float(self.direct_recovery_value_num_experts),
                    dtype=all_direct.dtype, device=x.device,
                )
                mean = all_direct.mean(dim=1)
                std = all_direct.std(dim=1, unbiased=False)
                rank_base = mean[:, 0]
                if routing in {"uniform_robust", "robust_ensemble", "risk_ensemble"}:
                    direct = mean.clone()
                    lam = self.direct_recovery_expert_disagreement_penalty
                    direct[:, 0] = mean[:, 0] - lam * std[:, 0]
                    # Keep aleatoric log-variance conservative as well.
                    direct[:, 1] = mean[:, 1] + lam * std[:, 1]
                    cursor = 2
                    if self.direct_recovery_opportunity_head:
                        direct[:, cursor] = mean[:, cursor] - lam * std[:, cursor]
                        cursor += 1
                    if self.direct_recovery_harm_head:
                        direct[:, cursor] = mean[:, cursor] + lam * std[:, cursor]
                    out["direct_expert_disagreement"] = std
                    out["direct_expert_output_std"] = std
                else:
                    direct = mean
            else:
                if self.direct_value_router is None:
                    raise RuntimeError("soft expert routing configured without a router")
                router_logits = self.direct_value_router(router_features)
                weights = torch.softmax(
                    router_logits / self.direct_recovery_value_router_temperature, dim=-1
                )
                mean = (all_direct * weights.unsqueeze(-1)).sum(dim=1)
                rank_base = mean[:, 0]
                if routing == "robust_observation":
                    centered = all_direct - mean.unsqueeze(1)
                    std = torch.sqrt((weights.unsqueeze(-1) * centered.square()).sum(dim=1).clamp_min(1.0e-8))
                    direct = mean.clone()
                    lam = self.direct_recovery_expert_disagreement_penalty
                    direct[:, 0] = mean[:, 0] - lam * std[:, 0]
                    direct[:, 1] = mean[:, 1] + lam * std[:, 1]
                    cursor = 2
                    if self.direct_recovery_opportunity_head:
                        direct[:, cursor] = mean[:, cursor] - lam * std[:, cursor]
                        cursor += 1
                    if self.direct_recovery_harm_head:
                        direct[:, cursor] = mean[:, cursor] + lam * std[:, cursor]
                    out["direct_expert_disagreement"] = std
                    out["direct_expert_output_std"] = std
                else:
                    direct = mean
                out["direct_expert_logits"] = router_logits
            out["direct_expert_weights"] = weights
            out["direct_expert_outputs"] = all_direct
        elif self.direct_value_head is not None:
            direct = self.direct_value_head(direct_features)
            rank_base = direct[:, 0]
        else:
            raise RuntimeError("direct recovery branch configured without a head")

        out["direct_recovery_value_logit"] = direct[:, 0]
        out["direct_recovery_value_logvar"] = direct[:, 1].clamp(-7.0, 2.0)
        cursor = 2
        if self.direct_recovery_opportunity_head:
            out["direct_recovery_opportunity_logit"] = direct[:, cursor]
            cursor += 1
        if self.direct_recovery_harm_head:
            out["direct_recovery_harm_logit"] = direct[:, cursor]
        if rank_base is None:
            rank_base = direct[:, 0]
        rank_residual = torch.zeros_like(rank_base)
        if self.direct_preference_adapter is not None:
            pointwise_rank_residual = self.direct_preference_adapter(pointwise_direct_features).squeeze(-1)
            out["direct_recovery_rank_pointwise_residual"] = pointwise_rank_residual
            rank_residual = rank_residual + pointwise_rank_residual
        if self.direct_preference_context_adapter is not None:
            context_rank_residual = self.direct_preference_context_adapter(relative_features).squeeze(-1)
            out["direct_recovery_rank_context_residual"] = context_rank_residual
            rank_residual = rank_residual + context_rank_residual
        out["direct_recovery_rank_residual"] = rank_residual
        inherited_rank = rank_base + rank_residual
        if self.direct_preference_set_ranker is not None:
            tournament_context = self.direct_preference_set_ranker.encode(
                relative_features, group_index, is_nominal
            )
            tournament_rank = self.direct_preference_set_ranker.score_from_context(
                tournament_context, group_index, is_nominal
            )
            out["direct_recovery_tournament_context"] = tournament_context
            out["direct_recovery_rank_tournament"] = tournament_rank
            rank_logit = tournament_rank if self.direct_recovery_set_tournament_replace_base else inherited_rank + tournament_rank
        else:
            rank_logit = inherited_rank
        out["direct_recovery_rank_logit"] = rank_logit

        delta_features = relative_features
        if self.direct_recovery_delta_policy_features:
            policy_features = relative_features.new_zeros((relative_features.shape[0], 2))
            if group_index is not None and is_nominal is not None:
                groups = group_index.to(device=relative_features.device)
                groups = groups.reshape(-1, 1) if groups.dim() == 1 else groups.reshape(groups.shape[0], -1)
                nominal_mask = is_nominal.to(device=relative_features.device).reshape(-1) > 0.5
                if groups.shape[0] == rank_logit.shape[0] and nominal_mask.shape[0] == rank_logit.shape[0]:
                    for key in torch.unique(groups, dim=0):
                        idx = torch.where((groups == key.unsqueeze(0)).all(dim=1))[0]
                        noms = idx[nominal_mask[idx]]
                        recs = idx[~nominal_mask[idx]]
                        if noms.numel() == 0 or recs.numel() == 0:
                            continue
                        nom_rank = rank_logit[noms[0]]
                        policy_features[recs, 0] = rank_logit[recs] - nom_rank
                        for rec in recs:
                            others = recs[recs != rec]
                            second = rank_logit[others].max() if others.numel() else nom_rank
                            policy_features[rec, 1] = rank_logit[rec] - second
            delta_features = torch.cat([relative_features, policy_features], dim=-1)
            out["direct_recovery_policy_features"] = policy_features

        delta_expert_idx = None
        if bucket_id is None:
            delta_expert_idx = torch.zeros((delta_features.shape[0],), dtype=torch.long, device=delta_features.device)
        else:
            delta_expert_idx = (bucket_id.to(device=delta_features.device, dtype=torch.long).reshape(-1) - 1).clamp(0, 1)
        if self.direct_delta_adapters is not None:
            all_delta = torch.stack([adapter(delta_features) for adapter in self.direct_delta_adapters], dim=1)
            delta = all_delta[torch.arange(all_delta.shape[0], device=all_delta.device), delta_expert_idx]
            out["direct_recovery_delta_expert_outputs"] = all_delta
        elif self.direct_delta_adapter is not None:
            delta = self.direct_delta_adapter(delta_features)
        else:
            delta = None

        evidence_calibrator_residual = None
        unified_benefit_logit = None
        unified_harm_logit = None
        unified_component_harm_logits = None
        unified_admission_logit = None
        calibrator_enabled = (
            self.direct_evidence_calibrators is not None
            or self.direct_evidence_unified_calibrator is not None
            or self.direct_evidence_concord_benefit_calibrator is not None
        )
        if delta is not None and calibrator_enabled:
            if self.direct_recovery_delta_policy_features and "direct_recovery_policy_features" in out:
                calibrator_policy = out["direct_recovery_policy_features"].to(dtype=delta.dtype)
            else:
                calibrator_policy = delta.new_zeros((delta.shape[0], 2))

            unified_evidence = (
                self.direct_evidence_unified_calibrator is not None
                or self.direct_evidence_concord_benefit_calibrator is not None
            )
            if unified_evidence:
                if self.direct_delta_adapters is None or "direct_recovery_delta_expert_outputs" not in out:
                    raise RuntimeError("unified evidence configured without frozen delta expert outputs")
                expert_delta = out["direct_recovery_delta_expert_outputs"].to(dtype=delta.dtype).detach()
                expert_mean = expert_delta.mean(dim=1)
                expert_disagreement = (expert_delta[:, 0] - expert_delta[:, 1]).abs()
                if self.direct_evidence_concord_benefit_calibrator is not None:
                    # Permutation-invariant expert statistics.  The evidence model
                    # cannot infer a regime from expert ordering and never receives
                    # bucket ids.  Consensus and disagreement are observables, not
                    # a hidden hard router.
                    center_e = expert_delta[:, :, 0]
                    half_e = 0.5 * torch.nn.functional.softplus(expert_delta[:, :, 1])
                    benefit_e = center_e - half_e
                    harm_e = -(center_e + half_e)
                    benefit_stats = torch.stack(
                        [benefit_e.mean(dim=1), benefit_e.amin(dim=1),
                         benefit_e.amax(dim=1), benefit_e.amax(dim=1)-benefit_e.amin(dim=1)],
                        dim=-1,
                    )
                    calibrator_parts = [
                        expert_mean, expert_disagreement, benefit_stats,
                        calibrator_policy.detach(),
                    ]
                else:
                    calibrator_parts = [
                        expert_delta.reshape(expert_delta.shape[0], -1),
                        expert_mean,
                        expert_disagreement,
                        calibrator_policy.detach(),
                    ]
            else:
                calibrator_parts = [delta[:, :2], calibrator_policy]

            if self.direct_recovery_evidence_calibrator_context:
                if (
                    self.direct_recovery_evidence_calibrator_context_source == "tournament"
                    and "direct_recovery_tournament_context" in out
                ):
                    calibrator_context = out["direct_recovery_tournament_context"].to(dtype=delta.dtype)
                else:
                    calibrator_context = relative_features.to(dtype=delta.dtype)
                if self.direct_recovery_evidence_calibrator_context_detach:
                    calibrator_context = calibrator_context.detach()
                calibrator_parts.append(calibrator_context)
            calibrator_input = torch.cat(calibrator_parts, dim=-1)

            if self.direct_evidence_concord_benefit_calibrator is not None:
                benefit_raw = self.direct_evidence_concord_benefit_calibrator(calibrator_input).squeeze(-1)
                harm_raw = self.direct_evidence_concord_harm_calibrator(calibrator_input)
                admission_raw = (
                    self.direct_evidence_concord_admission_calibrator(calibrator_input).squeeze(-1)
                    if self.direct_evidence_concord_admission_calibrator is not None
                    else None
                )
                benefit_residual = (
                    torch.tanh(benefit_raw) * self.direct_recovery_evidence_calibrator_scale
                )
                # Consensus transfer replaces v48.20's exact min envelope.  The
                # exact min let one mismatched frozen expert destroy otherwise
                # useful Near benefit evidence.  Mean consensus preserves shared
                # source information while an explicit disagreement penalty keeps
                # transfer conservative without selecting a regime expert.
                base_benefit = benefit_e.mean(dim=1) - (
                    self.direct_recovery_evidence_consensus_disagreement_penalty
                    * (benefit_e.amax(dim=1) - benefit_e.amin(dim=1))
                )
                unified_benefit_logit = base_benefit + benefit_residual
                if self.direct_recovery_evidence_component_heads:
                    # v48.28: the semantic prior is -2.  A scale of 2 capped
                    # candidate component logits at zero, so the model could
                    # never represent p(harm)>0.5 after a veto tolerance was
                    # exceeded.  The wider bounded range retains stable logits
                    # while allowing strong harmful evidence.
                    component_residual = (
                        torch.tanh(harm_raw[:, : self.direct_recovery_evidence_component_count])
                        * self.direct_recovery_evidence_component_scale
                    )
                    # v48.23 FRONTIER: a zero network output must mean the
                    # nominal-relative component is inside the configured safety
                    # deadband, not p(harm)=0.5.  With tolerance=0.05 and
                    # temperature=0.025 the teacher target at equality is
                    # sigmoid(-2), hence the principled default prior -2.0.
                    if self.direct_recovery_evidence_frontier:
                        unified_component_harm_logits = (
                            component_residual
                            + self.direct_recovery_evidence_component_prior_logit
                        )
                    else:
                        unified_component_harm_logits = component_residual
                    # v48.31 CONTRACT-SLACK-RANK: some physical coordinates can
                    # be degenerate or nearly unsupported in a fixed dataset (for
                    # example a constant harm_proxy).  A max-veto over an
                    # unsupported learned coordinate can dominate every regime even
                    # though no data identify its sign.  Reliability is global and
                    # regime-agnostic: it shrinks only unsupported coordinates toward
                    # the semantic non-harm prior, while the independent measured
                    # hard veto remains unchanged at deployment.
                    reliability = self._direct_recovery_evidence_component_reliability.to(
                        device=unified_component_harm_logits.device,
                        dtype=unified_component_harm_logits.dtype,
                    )
                    neutral_component_logit = (
                        unified_component_harm_logits.new_tensor(
                            self.direct_recovery_evidence_component_prior_logit
                        )
                        if self.direct_recovery_evidence_frontier
                        else unified_component_harm_logits.new_zeros(())
                    )
                    effective_component_harm_logits = neutral_component_logit + reliability * (
                        unified_component_harm_logits - neutral_component_logit
                    )
                    unified_harm_logit = effective_component_harm_logits.amax(dim=-1)
                    out["direct_recovery_evidence_effective_component_harm_logits"] = (
                        effective_component_harm_logits
                    )
                    out["direct_recovery_evidence_component_reliability"] = reliability.expand_as(
                        effective_component_harm_logits
                    )
                    anchor_harm_residual = (
                        component_residual
                        if self.direct_recovery_evidence_frontier
                        else effective_component_harm_logits
                    )
                    evidence_calibrator_residual = torch.cat(
                        [benefit_residual.unsqueeze(-1), anchor_harm_residual], dim=-1
                    )
                else:
                    harm_residual = (
                        torch.tanh(harm_raw.reshape(-1))
                        * self.direct_recovery_evidence_component_scale
                    )
                    unified_harm_logit = harm_residual
                    evidence_calibrator_residual = torch.stack(
                        [benefit_residual, harm_residual], dim=-1
                    )
                if admission_raw is not None:
                    # Admission has a separate semantic target: raw benefit AND
                    # no component veto.  Detaching the prior prevents its sparse
                    # gradient from distorting either the raw-benefit or risk head.
                    # softplus(harm) is a conservative log-risk penalty; the bounded
                    # residual can correct it from context without regime routing.
                    if self.direct_recovery_evidence_admission_prior_mode == "safety_slack":
                        # v48.30 SLACK-RANK: convert the five calibrated component
                        # logits back into signed physical veto margins.  Actions
                        # inside every non-degradation envelope pay no penalty; only
                        # predicted boundary violation reduces recoverability utility.
                        # The independent component veto remains fail-closed at
                        # deployment, while this continuous hinge gives one unified,
                        # regime-agnostic ranking semantic near the safety frontier.
                        predicted_component_margins = (
                            self.direct_recovery_evidence_slack_temperature
                            * effective_component_harm_logits.detach()
                        )
                        max_predicted_veto_margin = predicted_component_margins.amax(dim=-1)
                        slack_barrier = torch.relu(max_predicted_veto_margin)
                        admission_prior = (
                            unified_benefit_logit.detach()
                            - self.direct_recovery_evidence_slack_penalty * slack_barrier
                        )
                        out["direct_recovery_evidence_predicted_component_margins"] = predicted_component_margins
                        out["direct_recovery_evidence_max_predicted_veto_margin"] = max_predicted_veto_margin
                        out["direct_recovery_evidence_slack_barrier"] = slack_barrier
                    elif self.direct_recovery_evidence_admission_prior_mode == "benefit_only":
                        # v48.29: non-compensatory factors are calibrated as a
                        # separate veto. Penalising the same max-risk logit again
                        # inside the admission score double-counted one noisy
                        # factor and suppressed safe-positive actions. The
                        # admission residual is still trained with harmful
                        # candidates mapped to negative safe utility.
                        admission_prior = unified_benefit_logit.detach()
                    elif self.direct_recovery_evidence_frontier:
                        # Center the risk penalty at the semantic non-harm prior.
                        # Zero residual therefore reproduces the transferred
                        # benefit evidence instead of forcing all-abstain.
                        prior_penalty = torch.nn.functional.softplus(
                            unified_harm_logit.new_tensor(
                                self.direct_recovery_evidence_component_prior_logit
                            )
                        )
                        admission_prior = (
                            unified_benefit_logit.detach()
                            - (
                                torch.nn.functional.softplus(unified_harm_logit.detach())
                                - prior_penalty
                            )
                        )
                    else:
                        admission_prior = (
                            unified_benefit_logit.detach()
                            - torch.nn.functional.softplus(unified_harm_logit.detach())
                        )
                    # v48.25 INTEGRITY-BRIDGE optionally removes the tanh ceiling.
                    # The zero-initialised head still preserves the transferred prior
                    # exactly, while an unbounded linear residual can cross the actual
                    # nominal-vs-recovery decision boundary when the source prior is
                    # conservatively negative. Global gradient clipping retains numerical control.
                    admission_basis = (
                        torch.tanh(admission_raw)
                        if self.direct_recovery_evidence_admission_bounded
                        else admission_raw
                    )
                    admission_residual = (
                        admission_basis * self.direct_recovery_evidence_admission_scale
                    )
                    unified_admission_logit = admission_prior + admission_residual
                    evidence_calibrator_residual = torch.cat(
                        [evidence_calibrator_residual, admission_residual.unsqueeze(-1)], dim=-1
                    )
                    out["direct_recovery_evidence_concord_admission_raw"] = admission_raw
                    out["direct_recovery_evidence_admission_prior"] = admission_prior
                out["direct_recovery_evidence_expert_benefit_logits"] = benefit_e
                out["direct_recovery_evidence_expert_harm_logits"] = harm_e
                out["direct_recovery_evidence_expert_base"] = torch.stack(
                    [base_benefit, harm_e.amax(dim=1)], dim=-1
                )
                out["direct_recovery_evidence_concord_benefit_raw"] = benefit_raw
                out["direct_recovery_evidence_concord_harm_raw"] = harm_raw
            elif self.direct_evidence_unified_calibrator is not None:
                combined_residual = self.direct_evidence_unified_calibrator(calibrator_input)
                benefit_residual = (
                    torch.tanh(combined_residual[:, 0])
                    * self.direct_recovery_evidence_calibrator_scale
                )
                # Conservative continuous envelope over the two frozen source
                # experts.  Equal experts reproduce the source exactly; expert
                # disagreement lowers benefit confidence and raises harm.
                center_e = expert_delta[:, :, 0]
                half_e = 0.5 * torch.nn.functional.softplus(expert_delta[:, :, 1])
                benefit_e = center_e - half_e
                harm_e = -(center_e + half_e)
                # Exact lower/upper envelopes are used instead of normalized
                # soft-min/soft-max.  The normalized smooth forms lie *inside*
                # the expert/component range and can therefore overstate benefit
                # or let two low-risk components compensate one high-risk veto.
                # Exact amin/amax are bucket-invariant, preserve the zero-residual
                # source identity when experts/components agree, and implement the
                # intended non-compensatory safety semantics.
                base_benefit = benefit_e.amin(dim=1)
                base_harm = harm_e.amax(dim=1)
                unified_benefit_logit = base_benefit + benefit_residual
                if self.direct_recovery_evidence_component_heads:
                    # FACET changed the harm target semantics from signed total
                    # PCD to componentwise non-compensatory vetoes.  Reusing the
                    # old total-PCD harm logit as an additive base anchors every
                    # component to a source signal that the v48.19 certificate
                    # showed to be near random.  Benefit therefore transfers from
                    # the source experts, while component harm heads are a semantic
                    # reset: zero-initialized, bounded, absolute candidate-vs-
                    # nominal logits learned jointly across all regimes.
                    unified_component_harm_logits = (
                        torch.tanh(combined_residual[:, 1 : 1 + self.direct_recovery_evidence_component_count])
                        * self.direct_recovery_evidence_component_scale
                    )
                    unified_harm_logit = unified_component_harm_logits.amax(dim=-1)
                    evidence_calibrator_residual = torch.cat(
                        [benefit_residual.unsqueeze(-1), unified_component_harm_logits],
                        dim=-1,
                    )
                else:
                    harm_residual = (
                        torch.tanh(combined_residual[:, 1])
                        * self.direct_recovery_evidence_calibrator_scale
                    )
                    unified_harm_logit = base_harm + harm_residual
                    evidence_calibrator_residual = torch.stack(
                        [benefit_residual, harm_residual], dim=-1
                    )
                out["direct_recovery_evidence_expert_benefit_logits"] = benefit_e
                out["direct_recovery_evidence_expert_harm_logits"] = harm_e
                out["direct_recovery_evidence_expert_base"] = torch.stack(
                    [base_benefit, base_harm], dim=-1
                )
                out["direct_recovery_evidence_unified_residual_raw"] = combined_residual
            else:
                all_residuals = torch.stack(
                    [adapter(calibrator_input) for adapter in self.direct_evidence_calibrators], dim=1
                )
                regime_residual = all_residuals[
                    torch.arange(all_residuals.shape[0], device=all_residuals.device), delta_expert_idx
                ]
                if self.direct_evidence_shared_calibrator is not None:
                    shared_residual = self.direct_evidence_shared_calibrator(calibrator_input)
                    combined_residual = (
                        shared_residual
                        + self.direct_recovery_evidence_calibrator_regime_scale * regime_residual
                    )
                    out["direct_recovery_evidence_shared_residual_raw"] = shared_residual
                    out["direct_recovery_evidence_regime_residual_raw"] = regime_residual
                else:
                    combined_residual = regime_residual
                evidence_calibrator_residual = (
                    torch.tanh(combined_residual)
                    * self.direct_recovery_evidence_calibrator_scale
                )
                out["direct_recovery_evidence_calibrator_outputs"] = all_residuals
                if self.direct_recovery_evidence_calibrator_mode == "center_width":
                    delta = delta + evidence_calibrator_residual

            out["direct_recovery_evidence_calibrator_input"] = calibrator_input
            out["direct_recovery_evidence_calibrator_residual"] = evidence_calibrator_residual

        if delta is not None:
            nominal_mask = None
            if is_nominal is not None and is_nominal.numel() == delta.shape[0]:
                nominal_mask = is_nominal.to(device=delta.device).reshape(-1) > 0.5
            if self.direct_recovery_delta_mode == "ordinal_evidence":
                # v48.10 COPE: the exact PCD advantage is strongly tri-modal
                # (harm / dead-zone / benefit).  A continuous regressor collapsed
                # toward zero and made conformal radii span the whole target range.
                # Parameterise two ordered cumulative logits instead:
                #   P(benefit) <= P(non-harm), P(harm)=1-P(non-harm).
                center = delta[:, 0]
                half_width = 0.5 * torch.nn.functional.softplus(delta[:, 1])
                nonharm_logit = center + half_width
                benefit_logit = center - half_width
                if unified_benefit_logit is not None and unified_harm_logit is not None:
                    benefit_logit = unified_benefit_logit
                    harm_logit = unified_harm_logit
                    nonharm_logit = -harm_logit
                else:
                    # Preserve the legacy identity exactly: nominal non-harm is
                    # pinned before deriving the complementary harm logit.
                    if nominal_mask is not None:
                        nonharm_logit = torch.where(
                            nominal_mask, torch.zeros_like(nonharm_logit), nonharm_logit
                        )
                        benefit_logit = torch.where(
                            nominal_mask, torch.zeros_like(benefit_logit), benefit_logit
                        )
                    harm_logit = -nonharm_logit
                if (
                    evidence_calibrator_residual is not None
                    and self.direct_recovery_evidence_calibrator_mode == "dual_tail_context"
                    and unified_benefit_logit is None
                ):
                    # v48.18 DUET-BRIDGE: benefit and harm are independent tails.
                    # A candidate can be simultaneously uncertain/ambiguous in both
                    # tails; deployment then applies the harm veto rather than
                    # forcing one class probability down through a simplex softmax.
                    benefit_logit = benefit_logit + evidence_calibrator_residual[:, 0]
                    harm_logit = harm_logit + evidence_calibrator_residual[:, 1]
                    # The calibrator is a recovery-candidate correction only.
                    # Keep nominal evidence exactly at the source identity even
                    # after the residual output biases have been trained.
                    if nominal_mask is not None:
                        benefit_logit = torch.where(
                            nominal_mask, torch.zeros_like(benefit_logit), benefit_logit
                        )
                        harm_logit = torch.where(
                            nominal_mask, torch.zeros_like(harm_logit), harm_logit
                        )
                    nonharm_logit = -harm_logit
                if nominal_mask is not None and unified_benefit_logit is not None:
                    benefit_logit = torch.where(
                        nominal_mask, torch.zeros_like(benefit_logit), benefit_logit
                    )
                    harm_logit = torch.where(
                        nominal_mask, torch.zeros_like(harm_logit), harm_logit
                    )
                    nonharm_logit = -harm_logit
                    if unified_component_harm_logits is not None:
                        unified_component_harm_logits = torch.where(
                            nominal_mask.unsqueeze(-1),
                            torch.zeros_like(unified_component_harm_logits),
                            unified_component_harm_logits,
                        )
                    if unified_admission_logit is not None:
                        unified_admission_logit = torch.where(
                            nominal_mask, torch.zeros_like(unified_admission_logit), unified_admission_logit
                        )
                benefit_prob = torch.sigmoid(benefit_logit)
                harm_prob = torch.sigmoid(harm_logit)
                if (
                    evidence_calibrator_residual is not None
                    and self.direct_recovery_evidence_calibrator_mode == "simplex_context"
                ):
                    # Identity-preserving tri-class correction.  Adding a bounded
                    # residual to source log-probabilities and renormalising with
                    # softmax preserves a valid harmful/dead/beneficial simplex,
                    # while allowing each tail to move independently.
                    eps = torch.finfo(benefit_prob.dtype).eps
                    dead_prob = (1.0 - benefit_prob - harm_prob).clamp_min(eps)
                    source_probs = torch.stack(
                        [harm_prob.clamp_min(eps), dead_prob, benefit_prob.clamp_min(eps)],
                        dim=-1,
                    )
                    calibrated_probs = torch.softmax(
                        torch.log(source_probs) + evidence_calibrator_residual, dim=-1
                    )
                    harm_prob = calibrated_probs[:, 0]
                    dead_prob = calibrated_probs[:, 1]
                    benefit_prob = calibrated_probs[:, 2]
                    if nominal_mask is not None:
                        harm_prob = torch.where(
                            nominal_mask, torch.full_like(harm_prob, 0.5), harm_prob
                        )
                        benefit_prob = torch.where(
                            nominal_mask, torch.full_like(benefit_prob, 0.5), benefit_prob
                        )
                        dead_prob = torch.where(
                            nominal_mask, torch.zeros_like(dead_prob), dead_prob
                        )
                    harm_logit = torch.logit(harm_prob.clamp(eps, 1.0 - eps))
                    benefit_logit = torch.logit(benefit_prob.clamp(eps, 1.0 - eps))
                    nonharm_logit = torch.logit((1.0 - harm_prob).clamp(eps, 1.0 - eps))
                    out["direct_recovery_evidence_class_probabilities"] = torch.stack(
                        [harm_prob, dead_prob, benefit_prob], dim=-1
                    )
                if unified_admission_logit is not None:
                    admission_prob = torch.sigmoid(unified_admission_logit)
                    # Nominal is pinned to probability 0.5, so subtracting 0.5
                    # yields an exact candidate-vs-nominal admission score.
                    evidence_score = admission_prob - 0.5
                    out["direct_recovery_admission_logit"] = unified_admission_logit
                    out["direct_recovery_admission_probability"] = admission_prob
                else:
                    evidence_score = benefit_prob - harm_prob
                out["direct_recovery_evidence_nonharm_logit"] = nonharm_logit
                out["direct_recovery_evidence_benefit_logit"] = benefit_logit
                out["direct_recovery_evidence_harm_logit"] = harm_logit
                out["direct_recovery_evidence_score"] = evidence_score
                if unified_component_harm_logits is not None:
                    out["direct_recovery_evidence_component_harm_logits"] = unified_component_harm_logits
                    out["direct_recovery_evidence_component_harm_probabilities"] = torch.sigmoid(
                        unified_component_harm_logits
                    )
                # Reuse the established admission plumbing.  These logits are
                # already candidate-vs-nominal evidence; nominal is pinned to 0.
                out["direct_recovery_opportunity_logit"] = benefit_logit
                out["direct_recovery_harm_logit"] = harm_logit
                out["direct_recovery_delta_mean"] = evidence_score
                out["direct_recovery_delta_logvar"] = torch.full_like(
                    evidence_score, self.direct_recovery_delta_initial_logvar
                )
            else:
                delta_mean = torch.tanh(delta[:, 0])
                delta_logvar = delta[:, 1].clamp(-7.0, 2.0)
                if nominal_mask is not None:
                    delta_mean = torch.where(nominal_mask, torch.zeros_like(delta_mean), delta_mean)
                out["direct_recovery_delta_mean"] = delta_mean
                out["direct_recovery_delta_logvar"] = delta_logvar
        return out

    def forward(
        self,
        x: torch.Tensor,
        option_features: torch.Tensor | None = None,
        bucket_id: torch.Tensor | None = None,
        group_index: torch.Tensor | None = None,
        is_nominal: torch.Tensor | None = None,
        direct_only: bool = False,
    ) -> dict[str, torch.Tensor]:
        memory = self._scene_tokens(x)
        if direct_only:
            return self._direct_outputs(memory, x, bucket_id, group_index, is_nominal)
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
        out.update(self._direct_outputs(memory, x, bucket_id, group_index, is_nominal))
        if self.root_signature_head is not None:
            out["root_signature"] = self.root_signature_head(root_tokens)
        if self.root_future_signature_head is not None:
            out["root_future_signature"] = self.root_future_signature_head(root_tokens)
        return out
