from __future__ import annotations

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from .bev_encoder import TemporalBEVEncoder
from .action_encoder import ActionEncoder, OptionEncoder

FORBIDDEN_FORWARD_KEYS = {"mode_seed_params", "future_actor_trajectories", "teacher_margins", "Y_option", "R_star", "witness_star", "contact_labels", "rollout_events", "labels"}


class CARE(nn.Module):
    """Counterfactual Action-Conditioned Recovery Evidence predictor.

    This implementation preserves the required tensor semantics: temporal BEV
    encoder, ego/route encoders, fixed semantic mode queries, action tokens,
    option tokens, tri-factor fusion, and factorized evidence heads.  The swept
    corridor feature is represented by learned option/action embeddings in the MVP;
    users can replace `path_pool` with true feature sampling without changing I/O.
    """

    def __init__(self, C_bev: int = 24, H_h: int = 10, D_ego: int = 11, D_q: int = 6, N_q: int = 20, H_p1: int = 11, H_r1: int = 26, D_state: int = 6, M: int = 8, hidden: int = 128):
        super().__init__()
        self.M = M
        self.hidden = hidden
        self.bev_encoder = TemporalBEVEncoder(C_bev, H_h, hidden)
        self.ego_encoder = nn.Sequential(nn.Linear(D_ego, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, hidden))
        self.route_encoder = nn.Sequential(nn.Linear(N_q * D_q, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, hidden))
        self.action_encoder = ActionEncoder(D_state, H_p1, hidden)
        self.option_encoder = OptionEncoder(D_state, H_r1, hidden)
        self.action_set_attn = nn.MultiheadAttention(hidden, num_heads=4, batch_first=True)
        self.mode_queries = nn.Parameter(torch.randn(M, hidden) * 0.02)
        self.mode_readout = nn.MultiheadAttention(hidden, num_heads=4, batch_first=True)
        self.fuse = nn.Sequential(nn.Linear(hidden * 4, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, hidden), nn.ReLU(inplace=True))
        self.head_P = nn.Linear(hidden, 1)
        self.head_G = nn.Linear(hidden + 2, 1)
        self.head_C = nn.Linear(hidden, 1)
        self.head_K = nn.Linear(hidden, 1)
        self.head_option_logits = nn.Linear(hidden, 1)
        self.head_U = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 1))
        self.head_H = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 1))
        self.mode_prob_head = nn.Linear(hidden, 1)

    def forward(self, bev: torch.Tensor, ego_info: torch.Tensor, route_command: torch.Tensor, actions: torch.Tensor, options: torch.Tensor, action_mask: torch.Tensor, option_mask: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        forbidden = FORBIDDEN_FORWARD_KEYS.intersection(kwargs.keys())
        if forbidden:
            raise ValueError(f"CARE inference/training forward received oracle-only keys: {sorted(forbidden)}")
        B, K = actions.shape[:2]
        L = options.shape[2]
        M = self.M
        bev_f = self.bev_encoder(bev)
        ego_f = self.ego_encoder(ego_info.float())
        route_f = self.route_encoder(route_command.reshape(B, -1).float())
        root_f = bev_f + ego_f + route_f
        a = self.action_encoder(actions)
        a_refined, _ = self.action_set_attn(a, a, a, key_padding_mask=~action_mask.bool())
        o = self.option_encoder(options)
        q = self.mode_queries.unsqueeze(0).expand(B, M, self.hidden)
        mode_f, _ = self.mode_readout(q, root_f.unsqueeze(1), root_f.unsqueeze(1))
        mode_probs = torch.softmax(self.mode_prob_head(mode_f).squeeze(-1), dim=-1)
        h_m = mode_f[:, None, None, :, :].expand(B, K, L, M, self.hidden)
        h_a = a_refined[:, :, None, None, :].expand(B, K, L, M, self.hidden)
        h_o = o[:, :, :, None, :].expand(B, K, L, M, self.hidden)
        h_root = root_f[:, None, None, None, :].expand(B, K, L, M, self.hidden)
        h = self.fuse(torch.cat([h_m, h_a, h_o, h_root], dim=-1))
        P_logit = self.head_P(h)
        C_logit = self.head_C(h)
        P = torch.sigmoid(P_logit).squeeze(-1)
        C = torch.sigmoid(C_logit).squeeze(-1)
        G = torch.sigmoid(self.head_G(torch.cat([h, P.unsqueeze(-1), C.unsqueeze(-1)], dim=-1))).squeeze(-1)
        Kdef = torch.sigmoid(self.head_K(h)).squeeze(-1)
        option_logits = self.head_option_logits(h).squeeze(-1)
        # Action-level U/H, not option-level.
        h_am = torch.cat([
            mode_f[:, None, :, :].expand(B, K, M, self.hidden),
            a_refined[:, :, None, :].expand(B, K, M, self.hidden),
            root_f[:, None, None, :].expand(B, K, M, self.hidden),
        ], dim=-1)
        U = torch.sigmoid(self.head_U(h_am)).squeeze(-1)
        H = torch.sigmoid(self.head_H(h_am)).squeeze(-1)
        mask = option_mask.bool().unsqueeze(-1).expand_as(P)
        P = torch.where(mask, P, torch.zeros_like(P))
        G = torch.where(mask, G, torch.zeros_like(G))
        C = torch.where(mask, C, torch.zeros_like(C))
        Kdef = torch.where(mask, Kdef, torch.zeros_like(Kdef))
        option_logits = torch.where(mask, option_logits, torch.full_like(option_logits, -1e9))
        return {"P": P, "G": G, "C": C, "U": U, "H": H, "Kdef": Kdef, "mode_probs": mode_probs, "option_logits": option_logits}
