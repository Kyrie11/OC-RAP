from __future__ import annotations

from typing import Dict
import torch
import torch.nn as nn
from .bev_encoder import TemporalBEVEncoder
from .action_encoder import ActionEncoder, OptionEncoder

FORBIDDEN_FORWARD_KEYS = {
    "mode_seed_params", "future_actor_trajectories", "teacher_margins", "Y_option", "y_star", "Y_oc",
    "R_star", "witness_star", "witness_oc", "obs_class", "obs_equiv", "beta_star",
    "contact_labels", "rollout_events", "labels", "spec_id_star", "spec_margin_star", "margin_option",
}


class ReCoT(nn.Module):
    def __init__(self, C_bev: int = 24, H_h: int = 10, D_ego: int = 11, D_q: int = 6, N_q: int = 20, H_p1: int = 11, H_r1: int = 26, D_state: int = 6, M: int = 8, hidden: int = 128, g_dim: int = 9, D_token: int = 6, A_anchor: int = 3, D_shell: int = 4):
        super().__init__()
        self.M=M; self.hidden=hidden; self.g_dim=g_dim
        self.bev_encoder=TemporalBEVEncoder(C_bev,H_h,hidden)
        self.ego_encoder=nn.Sequential(nn.Linear(D_ego,hidden),nn.ReLU(inplace=True),nn.Linear(hidden,hidden))
        self.route_encoder=nn.Sequential(nn.Linear(N_q*D_q,hidden),nn.ReLU(inplace=True),nn.Linear(hidden,hidden))
        self.action_encoder=ActionEncoder(D_state,H_p1,hidden)
        self.option_encoder=OptionEncoder(D_state,H_r1,hidden)
        self.token_param_encoder=nn.Sequential(nn.Linear(D_token+A_anchor+D_shell,hidden),nn.ReLU(inplace=True),nn.Linear(hidden,hidden))
        self.action_set_attn=nn.MultiheadAttention(hidden,4,batch_first=True)
        self.mode_queries=nn.Parameter(torch.randn(M,hidden)*0.02)
        self.mode_readout=nn.MultiheadAttention(hidden,4,batch_first=True)
        self.fuse=nn.Sequential(nn.Linear(hidden*4,hidden),nn.ReLU(inplace=True),nn.Linear(hidden,hidden),nn.ReLU(inplace=True))
        self.head_g=nn.Linear(hidden,g_dim)
        self.head_y=nn.Linear(hidden,1)
        self.head_k=nn.Linear(hidden,1)
        self.head_mu=nn.Linear(hidden,1)
        self.head_h=nn.Sequential(nn.Linear(hidden*3,hidden),nn.ReLU(inplace=True),nn.Linear(hidden,1))
        self.head_u=nn.Sequential(nn.Linear(hidden*3,hidden),nn.ReLU(inplace=True),nn.Linear(hidden,1))
        self.head_c=nn.Sequential(nn.Linear(hidden*3,hidden),nn.ReLU(inplace=True),nn.Linear(hidden,1))
        self.head_beta=nn.Linear(hidden*3, M)
        self.mode_prob_head=nn.Linear(hidden,1)

    def forward(self, bev, ego_info, route_command, actions_states=None, actions_controls=None, token_states_ref=None, token_controls_ref=None, token_params=None, token_anchor=None, token_hard_shell=None, action_mask=None, option_mask=None, actions=None, options=None, **kwargs) -> Dict[str, torch.Tensor]:
        forbidden=FORBIDDEN_FORWARD_KEYS.intersection(kwargs.keys())
        if forbidden: raise ValueError(f"ReCoT forward received oracle-only keys: {sorted(forbidden)}")
        # Backward compatible names.
        if actions_states is None: actions_states = actions
        if token_states_ref is None: token_states_ref = options
        B,K=actions_states.shape[:2]; L=token_states_ref.shape[2]; M=self.M
        if action_mask is None: action_mask=torch.ones(B,K,dtype=torch.bool,device=bev.device)
        if option_mask is None: option_mask=torch.ones(B,K,L,dtype=torch.bool,device=bev.device)
        bev_f=self.bev_encoder(bev)
        ego_f=self.ego_encoder(ego_info.float())
        route_f=self.route_encoder(route_command.reshape(B,-1).float())
        root_f=bev_f+ego_f+route_f
        a=self.action_encoder(actions_states)
        a_refined,_=self.action_set_attn(a,a,a,key_padding_mask=~action_mask.bool())
        o=self.option_encoder(token_states_ref)
        if token_params is not None and token_anchor is not None and token_hard_shell is not None:
            tp=torch.cat([token_params.float(), token_anchor.float(), token_hard_shell.float()], dim=-1)
            o=o+self.token_param_encoder(tp)
        q=self.mode_queries.unsqueeze(0).expand(B,M,self.hidden)
        mode_f,_=self.mode_readout(q,root_f.unsqueeze(1),root_f.unsqueeze(1))
        mode_probs=torch.softmax(self.mode_prob_head(mode_f).squeeze(-1), dim=-1)
        h=self.fuse(torch.cat([
            mode_f[:,None,None,:,:].expand(B,K,L,M,self.hidden),
            a_refined[:,:,None,None,:].expand(B,K,L,M,self.hidden),
            o[:,:,:,None,:].expand(B,K,L,M,self.hidden),
            root_f[:,None,None,None,:].expand(B,K,L,M,self.hidden),
        ], dim=-1))
        h_am=torch.cat([mode_f[:,None,:,:].expand(B,K,M,self.hidden), a_refined[:,:,None,:].expand(B,K,M,self.hidden), root_f[:,None,None,:].expand(B,K,M,self.hidden)], dim=-1)
        mask=option_mask.bool().unsqueeze(-1).unsqueeze(-1)
        g_hat=torch.where(mask, self.head_g(h), torch.zeros_like(self.head_g(h)))
        y_logit=torch.where(mask.squeeze(-1), self.head_y(h).squeeze(-1), torch.full((B,K,L,M), -1e9, device=bev.device, dtype=bev.dtype))
        k_hat=torch.where(mask.squeeze(-1), torch.sigmoid(self.head_k(h).squeeze(-1)), torch.zeros(B,K,L,M,device=bev.device,dtype=bev.dtype))
        mu_logits=torch.where(mask.squeeze(-1), self.head_mu(h).squeeze(-1), torch.full((B,K,L,M), -1e9, device=bev.device, dtype=bev.dtype))
        h_hat=torch.sigmoid(self.head_h(h_am)).squeeze(-1)
        u_hat=torch.sigmoid(self.head_u(h_am)).squeeze(-1)
        c_rule_hat=torch.relu(self.head_c(h_am)).squeeze(-1)
        beta_logits=self.head_beta(h_am)
        return {"g_hat":g_hat,"y_logit":y_logit,"h_hat":h_hat,"k_hat":k_hat,"u_hat":u_hat,"c_rule_hat":c_rule_hat,"beta_logits":beta_logits,"mu_logits":mu_logits,"mode_probs":mode_probs}

# Backward compatibility
CARE = ReCoT
