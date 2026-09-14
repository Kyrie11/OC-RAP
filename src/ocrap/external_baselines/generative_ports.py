"""Generative-planner adapters for OC-RAP's WOMD/Waymax action interface.

The adapters in this file deliberately distinguish *algorithmic fidelity* from
*dataset/interface fidelity*.  Diffusion Planner and Flow Planner are trained on
the logged expert trajectory (not on OC-RAP teacher scores), generate one native
continuous trajectory per scene, and only then project that trajectory onto the
common executable candidate lattice.  This preserves the native generative
operator while avoiding an unnecessary solver invocation for every candidate.

Important adaptation boundaries:
* Diffusion Planner remains an ego-only adaptation because the serialized OC-RAP
  training samples do not contain neighboring-agent future ground truth.  The
  VP/x-start/DPM-Solver++ path is preserved, but joint ego-neighbor generation
  and optional classifier guidance require a raw-WOMD future join.
* Flow Planner preserves CondOT x-start training, overlapping action tokens,
  neighbor-only CFG, scale-adaptive scene/trajectory fusion, overlap averaging,
  and four-step midpoint ODE sampling.  Its 8 s / 80-step source horizon is
  proportionally compressed to the benchmark's executable 2 s / 20-step prefix.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from ocrap.external_baselines.third_party.dpm_solver_pytorch import (
    DPM_Solver,
    NoiseScheduleVP,
    model_wrapper as dpm_model_wrapper,
)


def _make_encoder(d_model: int, heads: int, depth: int, dropout: float) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=heads,
        dim_feedforward=4 * d_model,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    try:
        return nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
    except TypeError:
        return nn.TransformerEncoder(layer, num_layers=depth)


def _masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    w = mask.to(dtype=x.dtype).unsqueeze(-1)
    return (x * w).sum(dim=dim) / w.sum(dim=dim).clamp_min(1.0)


def _trajectory_state(prefix_traj: torch.Tensor, prefix_valid: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    """Return [x,y,cos(yaw),sin(yaw)] and valid mask for [B,N,T,2]."""
    xy = prefix_traj.float()
    B, N, T, _ = xy.shape
    if prefix_valid is None:
        valid = torch.ones(B, N, T, dtype=torch.bool, device=xy.device)
    else:
        valid = prefix_valid.bool()
    origin = torch.zeros_like(xy[..., :1, :])
    prev = torch.cat([origin, xy[..., :-1, :]], dim=-2)
    d = xy - prev
    moving = d.square().sum(dim=-1) > 1.0e-8
    dx = torch.where(moving, d[..., 0], torch.ones_like(d[..., 0]))
    dy = torch.where(moving, d[..., 1], torch.zeros_like(d[..., 1]))
    yaw = torch.atan2(dy, dx)
    state = torch.cat([xy, yaw.cos().unsqueeze(-1), yaw.sin().unsqueeze(-1)], dim=-1)
    state = torch.where(valid.unsqueeze(-1), state, torch.zeros_like(state))
    return state, valid


def _candidate_energy(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Masked mean squared state error; pred may be [B,T,4] or [B,N,T,4]."""
    if pred.dim() == 3:
        pred = pred[:, None]
    err = (pred.float() - target.float()).square().sum(dim=-1)
    err = torch.where(valid, err, torch.zeros_like(err))
    return err.sum(dim=-1) / valid.float().sum(dim=-1).clamp_min(1.0)


def _time_embedding(t: torch.Tensor, d_model: int) -> torch.Tensor:
    half = max(1, d_model // 2)
    freq = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / max(half - 1, 1)
    )
    phase = t.unsqueeze(-1) * freq
    emb = torch.cat([phase.sin(), phase.cos()], dim=-1)
    if emb.shape[-1] < d_model:
        emb = F.pad(emb, (0, d_model - emb.shape[-1]))
    return emb[..., :d_model]


def _last_valid_xy(agent_history: torch.Tensor, agent_valid: torch.Tensor) -> torch.Tensor:
    """Observed current xy for each actor, with zeros for empty actors."""
    B, A, H, _ = agent_history.shape
    idx = torch.arange(H, device=agent_history.device).view(1, 1, H)
    last = torch.where(agent_valid, idx, torch.full_like(idx, -1)).amax(dim=-1).clamp_min(0)
    gather = last[..., None, None].expand(B, A, 1, 2)
    xy = agent_history[..., :2].gather(2, gather).squeeze(2)
    return torch.where(agent_valid.any(dim=-1, keepdim=True), xy, torch.zeros_like(xy))


class CompactSceneEncoder(nn.Module):
    """Observation-only actor/vector-map encoder shared by the three ports.

    The bridge tensors are reconstructed from the same WOMD history/map fields used
    by the existing source ports.  Actor and map point work is performed once per
    scene, not repeated for every executable candidate.
    """

    def __init__(self, d_model: int, heads: int, depth: int, dropout: float) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.agent_point = nn.Sequential(nn.Linear(9, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.current = nn.Sequential(nn.Linear(6, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.map_point = nn.Sequential(nn.Linear(6, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.map_meta = nn.Sequential(nn.Linear(7, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.route_point = nn.Sequential(nn.Linear(3, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.type_embed = nn.Parameter(torch.randn(1, 3, d_model) * 0.02)
        self.encoder = _make_encoder(d_model, heads, max(1, depth), dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        *,
        source_agent_history: torch.Tensor | None,
        source_agent_valid: torch.Tensor | None,
        source_current_state: torch.Tensor | None,
        source_map_points: torch.Tensor | None,
        source_map_point_valid: torch.Tensor | None,
        source_map_meta: torch.Tensor | None,
        source_map_center: torch.Tensor | None,
        source_map_valid: torch.Tensor | None,
        source_centerline: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        tokens: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []

        if source_agent_history is not None:
            ah = source_agent_history.float()
            av = source_agent_valid.bool() if source_agent_valid is not None else (ah.abs().sum(-1) > 0)
            enc = self.agent_point(ah)
            agent = _masked_mean(enc, av, dim=2) + self.type_embed[:, 0:1]
            tokens.append(agent)
            masks.append(av.any(dim=2))

        if source_map_points is not None:
            mp = source_map_points.float()
            mpv = source_map_point_valid.bool() if source_map_point_valid is not None else (mp.abs().sum(-1) > 0)
            poly = _masked_mean(self.map_point(mp), mpv, dim=2)
            if source_map_meta is not None and source_map_center is not None:
                meta = torch.cat([source_map_meta.float(), source_map_center.float()], dim=-1)
                poly = poly + self.map_meta(meta)
            poly = poly + self.type_embed[:, 1:2]
            tokens.append(poly)
            masks.append(source_map_valid.bool() if source_map_valid is not None else mpv.any(dim=2))

        if source_centerline is not None:
            route = self.route_point(source_centerline.float()) + self.type_embed[:, 2:3]
            tokens.append(route)
            masks.append(torch.isfinite(source_centerline.float()).all(dim=-1))

        if not tokens:
            return torch.zeros(batch_size, self.d_model, device=device)
        tok = torch.cat(tokens, dim=1)
        valid = torch.cat(masks, dim=1)
        empty = ~valid.any(dim=1)
        if bool(empty.any()):
            valid = valid.clone()
            tok = tok.clone()
            valid[empty, 0] = True
            tok[empty, 0] = 0.0
        tok = self.encoder(tok, src_key_padding_mask=~valid)
        pooled = _masked_mean(tok, valid, dim=1)
        if source_current_state is not None:
            pooled = pooled + self.current(source_current_state[..., :6].float())
        return self.norm(pooled)



class SourceLikeSceneEncoder(nn.Module):
    """Candidate-independent actor/map encoder with source-compatible modality split.

    Flow Planner's defining interaction layer needs modality tokens and their
    spatial positions rather than a single pooled scene vector.  Actor histories
    and map polylines are therefore encoded separately and kept as tokens until
    the generative decoder.  Route/current-state information is pooled into the
    modulation context, matching the role of navigation/timestep conditioning in
    the released planners.
    """

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.agent_point = nn.Sequential(nn.Linear(9, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.map_point = nn.Sequential(nn.Linear(6, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.map_meta = nn.Sequential(nn.Linear(7, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.route_point = nn.Sequential(nn.Linear(3, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.current = nn.Sequential(nn.Linear(6, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.agent_type = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.map_type = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.agent_norm = nn.LayerNorm(d_model)
        self.map_norm = nn.LayerNorm(d_model)
        self.context_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        *,
        source_agent_history: torch.Tensor | None,
        source_agent_valid: torch.Tensor | None,
        source_current_state: torch.Tensor | None,
        source_map_points: torch.Tensor | None,
        source_map_point_valid: torch.Tensor | None,
        source_map_meta: torch.Tensor | None,
        source_map_center: torch.Tensor | None,
        source_map_valid: torch.Tensor | None,
        source_centerline: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        if source_agent_history is None:
            agent = torch.zeros(batch_size, 1, self.d_model, device=device)
            agent_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
            agent_pos = torch.zeros(batch_size, 1, 2, device=device)
        else:
            ah = source_agent_history.float()
            av = source_agent_valid.bool() if source_agent_valid is not None else (ah.abs().sum(-1) > 0)
            agent = _masked_mean(self.agent_point(ah), av, dim=2) + self.agent_type
            agent = self.agent_norm(agent)
            agent_mask = av.any(dim=2)
            # Ego is always retained if serialization produced a fully padded history.
            empty_batch = ~agent_mask.any(dim=1)
            if bool(empty_batch.any()):
                agent_mask = agent_mask.clone()
                agent = agent.clone()
                agent_mask[empty_batch, 0] = True
                agent[empty_batch, 0] = 0.0
            agent_pos = _last_valid_xy(ah, av)

        if source_map_points is None:
            map_tok = torch.zeros(batch_size, 1, self.d_model, device=device)
            map_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
            map_pos = torch.zeros(batch_size, 1, 2, device=device)
        else:
            mp = source_map_points.float()
            mpv = source_map_point_valid.bool() if source_map_point_valid is not None else (mp.abs().sum(-1) > 0)
            map_tok = _masked_mean(self.map_point(mp), mpv, dim=2)
            if source_map_meta is not None and source_map_center is not None:
                meta = torch.cat([source_map_meta.float(), source_map_center.float()], dim=-1)
                map_tok = map_tok + self.map_meta(meta)
            map_tok = self.map_norm(map_tok + self.map_type)
            map_mask = source_map_valid.bool() if source_map_valid is not None else mpv.any(dim=2)
            if source_map_center is not None:
                map_pos = source_map_center[..., :2].float()
            else:
                map_pos = torch.zeros(map_tok.shape[0], map_tok.shape[1], 2, device=map_tok.device, dtype=map_tok.dtype)

        context = torch.zeros(batch_size, self.d_model, device=device, dtype=agent.dtype)
        if source_centerline is not None:
            route = source_centerline.float()
            rv = torch.isfinite(route).all(dim=-1)
            route = torch.nan_to_num(route)
            context = context + _masked_mean(self.route_point(route), rv, dim=1)
        if source_current_state is not None:
            context = context + self.current(source_current_state[..., :6].float())
        context = self.context_norm(context)
        return {
            "agent_tokens": self.dropout(agent),
            "agent_mask": agent_mask,
            "agent_pos": agent_pos,
            "map_tokens": self.dropout(map_tok),
            "map_mask": map_mask,
            "map_pos": map_pos,
            "context": context,
        }


def _drop_nearest_neighbors(scene: dict[str, torch.Tensor], count: int) -> dict[str, torch.Tensor]:
    """Return a shallow scene view with ego kept and first `count` neighbors masked."""
    if count <= 0:
        return scene
    out = dict(scene)
    mask = scene["agent_mask"].clone()
    stop = min(mask.shape[1], 1 + int(count))
    if stop > 1:
        mask[:, 1:stop] = False
    out["agent_mask"] = mask
    return out


def _scene_tokens(scene: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    tok = torch.cat([scene["agent_tokens"], scene["map_tokens"]], dim=1)
    mask = torch.cat([scene["agent_mask"], scene["map_mask"]], dim=1)
    empty = ~mask.any(dim=1)
    if bool(empty.any()):
        tok = tok.clone(); mask = mask.clone()
        mask[empty, 0] = True; tok[empty, 0] = 0.0
    return tok, mask


class _CandidatePrior(nn.Module):
    def __init__(self, input_dim: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)



class _ScalarHeads(nn.Module):
    def __init__(self, input_dim: int, d_model: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, d_model), nn.GELU())
        self.heads = nn.ModuleDict({k: nn.Linear(d_model, 1) for k in ("utility", "hard", "harm", "r_orc", "r_dep")})

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.backbone(x)
        return {k: head(h).squeeze(-1) for k, head in self.heads.items()}


class _AdaLNCrossBlock(nn.Module):
    """Compact DiT-style cross-attention block for the ego trajectory token."""
    def __init__(self, d_model: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm_ff = nn.LayerNorm(d_model, elementwise_affine=False)
        self.cross = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(4*d_model, d_model))
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6*d_model))

    def forward(self, x: torch.Tensor, cond: torch.Tensor, scene: torch.Tensor, scene_mask: torch.Tensor) -> torch.Tensor:
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = self.mod(cond).chunk(6, dim=-1)
        q = self.norm_q(x) * (1 + scale_a[:, None]) + shift_a[:, None]
        attn, _ = self.cross(q, scene, scene, key_padding_mask=~scene_mask, need_weights=False)
        x = x + gate_a[:, None] * attn
        z = self.norm_ff(x) * (1 + scale_f[:, None]) + shift_f[:, None]
        x = x + gate_f[:, None] * self.ff(z)
        return x


class _EgoDiffusionDenoiser(nn.Module):
    """Ego-only DiT-like x-start predictor compatible with DPM-Solver++."""
    model_type = "x_start"

    def __init__(self, state_steps: int, scene_dim: int, d_model: int, depth: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.state_steps = int(state_steps)
        self.pre = nn.Sequential(nn.Linear(state_steps * 4, 512), nn.GELU(), nn.Linear(512, d_model))
        self.scene_proj = nn.Linear(scene_dim, d_model) if scene_dim != d_model else nn.Identity()
        self.cond_proj = nn.Linear(scene_dim, d_model) if scene_dim != d_model else nn.Identity()
        self.time_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.blocks = nn.ModuleList([_AdaLNCrossBlock(d_model, heads, dropout) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.final_mod = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 2*d_model))
        self.out = nn.Sequential(nn.Linear(d_model, 512), nn.GELU(), nn.Linear(512, state_steps * 4))

    def forward(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        *,
        scene_tokens: torch.Tensor,
        scene_mask: torch.Tensor,
        scene_context: torch.Tensor,
    ) -> torch.Tensor:
        B = z.shape[0]
        h = self.pre(z.reshape(B, -1)).unsqueeze(1)
        cond = self.time_mlp(_time_embedding(t.reshape(B), h.shape[-1])) + self.cond_proj(scene_context)
        scene = self.scene_proj(scene_tokens)
        for block in self.blocks:
            h = block(h, cond, scene, scene_mask)
        shift, scale = self.final_mod(cond).chunk(2, dim=-1)
        h = self.final_norm(h) * (1 + scale[:, None]) + shift[:, None]
        return self.out(h.squeeze(1)).reshape(B, self.state_steps, 4)


class DiffusionPlannerPort(nn.Module):
    """VP/x-start/DPM-Solver++ Diffusion Planner adaptation.

    One trajectory is generated per scene with the source 10-step second-order
    multistep DPM-Solver++ path and then projected onto the executable candidate
    lattice.  This is both more source-faithful and faster than solving once per
    candidate.  Joint neighbor-future generation remains intentionally omitted;
    raw-WOMD future targets are required to restore that source component.
    """

    def __init__(self, input_dim: int, max_candidates: int = 24, d_model: int = 192, num_layers: int = 3,
                 num_heads: int = 6, dropout: float = 0.1, future_len: int = 20,
                 scene_layers: int = 2, energy_weight: float = 2.0, eval_noise_scale: float = 0.5,
                 beta_min: float = 0.1, beta_max: float = 20.0, diffusion_steps: int = 10) -> None:
        super().__init__()
        del scene_layers  # retained in config compatibility; modality encoder is source-style point/Mixer-lite.
        self.max_candidates = int(max_candidates)
        self.future_len = int(future_len)
        self.energy_weight = float(energy_weight)
        self.eval_noise_scale = float(eval_noise_scale)
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.diffusion_steps = int(diffusion_steps)
        self.scene = SourceLikeSceneEncoder(d_model, dropout)
        self.denoiser = _EgoDiffusionDenoiser(future_len + 1, d_model, d_model, num_layers, num_heads, dropout)
        self.scalar_heads = _ScalarHeads(input_dim, d_model)

    @staticmethod
    def _prepend_current(future: torch.Tensor) -> torch.Tensor:
        current = future.new_zeros(future.shape[0], 1, 4)
        current[..., 2] = 1.0
        return torch.cat([current, future], dim=1)

    @staticmethod
    def _prepend_valid(valid: torch.Tensor) -> torch.Tensor:
        return torch.cat([torch.ones(valid.shape[0], 1, dtype=torch.bool, device=valid.device), valid], dim=1)

    def _marginal(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        log_a = -0.25 * t.square() * (self.beta_max - self.beta_min) - 0.5 * self.beta_min * t
        a = torch.exp(log_a).view(-1, 1, 1)
        s = torch.sqrt((1.0 - torch.exp(2.0 * log_a)).clamp_min(1.0e-8)).view(-1, 1, 1)
        return a * x0 + s * noise

    def _generate(self, scene: dict[str, torch.Tensor], *, seed: int | None, dtype: torch.dtype) -> torch.Tensor:
        B = scene["context"].shape[0]
        dev = scene["context"].device
        gen = None
        if seed is not None:
            gen = torch.Generator(device=dev)
            gen.manual_seed(int(seed) & 0x7FFFFFFF)
        future_noise = torch.randn((B, self.future_len, 4), device=dev, dtype=dtype, generator=gen) * self.eval_noise_scale
        current = future_noise.new_zeros(B, 1, 4); current[..., 2] = 1.0
        xT = torch.cat([current, future_noise], dim=1)
        scene_tok, scene_mask = _scene_tokens(scene)
        schedule = NoiseScheduleVP(
            schedule="linear",
            continuous_beta_0=self.beta_min,
            continuous_beta_1=self.beta_max,
        )
        wrapped = dpm_model_wrapper(
            self.denoiser,
            schedule,
            model_type="x_start",
            model_kwargs={
                "scene_tokens": scene_tok,
                "scene_mask": scene_mask,
                "scene_context": scene["context"],
            },
            guidance_type="uncond",
        )
        def fix_current(xt: torch.Tensor, _t: torch.Tensor, _step: int) -> torch.Tensor:
            xt = xt.clone()
            xt[:, 0] = current[:, 0]
            return xt
        solver = DPM_Solver(wrapped, schedule, algorithm_type="dpmsolver++", correcting_xt_fn=fix_current)
        sample = solver.sample(
            xT,
            steps=self.diffusion_steps,
            order=2,
            skip_type="logSNR",
            method="multistep",
            denoise_to_zero=True,
        )
        return sample[:, 1:]

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, *, prefix_traj: torch.Tensor | None = None,
                prefix_valid: torch.Tensor | None = None, source_agent_history: torch.Tensor | None = None,
                source_agent_valid: torch.Tensor | None = None, source_current_state: torch.Tensor | None = None,
                source_map_points: torch.Tensor | None = None, source_map_point_valid: torch.Tensor | None = None,
                source_map_meta: torch.Tensor | None = None, source_map_center: torch.Tensor | None = None,
                source_map_valid: torch.Tensor | None = None, source_centerline: torch.Tensor | None = None,
                target_index: torch.Tensor | None = None, sampling_seed: int | None = None,
                **_: torch.Tensor) -> dict[str, torch.Tensor]:
        B, N, _ = x.shape
        if prefix_traj is None:
            raise ValueError("DiffusionPlannerPort requires prefix_traj")
        candidates, candidate_valid = _trajectory_state(prefix_traj, prefix_valid)
        scene = self.scene(
            source_agent_history=source_agent_history, source_agent_valid=source_agent_valid,
            source_current_state=source_current_state, source_map_points=source_map_points,
            source_map_point_valid=source_map_point_valid, source_map_meta=source_map_meta,
            source_map_center=source_map_center, source_map_valid=source_map_valid,
            source_centerline=source_centerline, batch_size=B, device=x.device,
        )
        scene_tok, scene_mask = _scene_tokens(scene)
        if self.training:
            idx = torch.zeros(B, dtype=torch.long, device=x.device) if target_index is None else target_index.long().clamp(0, N - 1)
            b = torch.arange(B, device=x.device)
            target_future = candidates[b, idx]
            target_valid = candidate_valid[b, idx]
            target = self._prepend_current(target_future)
            valid = self._prepend_valid(target_valid)
            t = (torch.rand(B, device=x.device, dtype=target.dtype) * (1.0 - 1.0e-3) + 1.0e-3)
            noise = torch.randn_like(target)
            noise[:, 0] = 0.0
            noisy = target.clone()
            noisy[:, 1:] = self._marginal(target[:, 1:], t, noise[:, 1:])
            pred = self.denoiser(
                noisy, t, scene_tokens=scene_tok, scene_mask=scene_mask, scene_context=scene["context"]
            )
            pred[:, 0] = target[:, 0]
            generated = pred[:, 1:]
            diffusion_pred = pred[:, 1:]
            diffusion_target = target[:, 1:]
            diffusion_valid = valid[:, 1:]
        else:
            generated = self._generate(scene, seed=sampling_seed, dtype=candidates.dtype)
            diffusion_pred = generated
            diffusion_target = generated.detach()
            diffusion_valid = torch.ones(B, self.future_len, dtype=torch.bool, device=x.device)

        energy = _candidate_energy(generated, candidates, candidate_valid)
        logits = -self.energy_weight * energy
        if mask is not None:
            logits = logits.masked_fill(~mask.bool(), -1.0e4)
        out: dict[str, torch.Tensor] = {
            "logits": logits,
            "diffusion_pred_x0": diffusion_pred,
            "diffusion_target_x0": diffusion_target,
            "diffusion_valid": diffusion_valid,
            "diffusion_generated": generated,
            "diffusion_energy": energy,
        }
        out.update(self.scalar_heads(x))
        return out


class _ModalityFFN(nn.Module):
    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(4*d_model, d_model))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ff(self.norm(x))


class ScaleAdaptiveFusionBlock(nn.Module):
    """Eq. (4)-(6)-style joint fusion with learnable distance-scaled attention."""
    def __init__(self, d_model: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.heads = int(heads)
        self.agent_norm = nn.LayerNorm(d_model)
        self.map_norm = nn.LayerNorm(d_model)
        self.traj_norm = nn.LayerNorm(d_model)
        self.agent_mod = nn.Linear(d_model, 2*d_model)
        self.map_mod = nn.Linear(d_model, 2*d_model)
        self.traj_mod = nn.Linear(d_model, 2*d_model)
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.lambda_proj = nn.Linear(d_model, heads)
        self.agent_ff = _ModalityFFN(d_model, dropout)
        self.map_ff = _ModalityFFN(d_model, dropout)
        self.traj_ff = _ModalityFFN(d_model, dropout)

    @staticmethod
    def _adaln(x: torch.Tensor, norm: nn.LayerNorm, mod: nn.Linear, cond: torch.Tensor) -> torch.Tensor:
        shift, scale = mod(cond).chunk(2, dim=-1)
        return norm(x) * (1 + scale[:, None]) + shift[:, None]

    def forward(self, agent: torch.Tensor, amap: torch.Tensor, traj: torch.Tensor,
                agent_mask: torch.Tensor, map_mask: torch.Tensor, traj_mask: torch.Tensor,
                agent_pos: torch.Tensor, map_pos: torch.Tensor, traj_pos: torch.Tensor,
                cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        na, nm, nt = agent.shape[1], amap.shape[1], traj.shape[1]
        a = self._adaln(agent, self.agent_norm, self.agent_mod, cond)
        m = self._adaln(amap, self.map_norm, self.map_mod, cond)
        q = self._adaln(traj, self.traj_norm, self.traj_mod, cond)
        z = torch.cat([a, m, q], dim=1)
        valid = torch.cat([agent_mask, map_mask, traj_mask], dim=1)
        pos = torch.cat([agent_pos, map_pos, traj_pos], dim=1)
        # Distance bias is query-specific: -lambda_q * ||p_q-p_k||.
        dist = torch.cdist(pos.float(), pos.float()).to(z.dtype)
        lam = F.softplus(self.lambda_proj(z)).permute(0, 2, 1)  # B,H,Lq
        bias4 = -(lam.unsqueeze(-1) * dist[:, None])
        # Fold the key padding into the float attention bias.  This preserves
        # exact masking while avoiding PyTorch's deprecated mixed bool/float
        # key_padding_mask + attn_mask path.
        bias4 = bias4.masked_fill((~valid)[:, None, None, :], -1.0e4)
        bias = bias4.reshape(z.shape[0] * self.heads, z.shape[1], z.shape[1])
        out, _ = self.attn(z, z, z, attn_mask=bias, need_weights=False)
        z = z + out
        agent, amap, traj = torch.split(z, [na, nm, nt], dim=1)
        return self.agent_ff(agent), self.map_ff(amap), self.traj_ff(traj)


class FlowPlannerPort(nn.Module):
    """Flow Planner core port: CondOT x-start + neighbor CFG + 4-step midpoint ODE."""

    def __init__(self, input_dim: int, max_candidates: int = 24, d_model: int = 256, num_layers: int = 4,
                 num_heads: int = 8, dropout: float = 0.1, future_len: int = 20, scene_layers: int = 2,
                 token_size: int = 5, token_stride: int = 3, cfg_dropout: float = 0.30,
                 cfg_weight: float = 1.8, consistency_weight: float = 0.5,
                 energy_weight: float = 1.5, cfg_neighbor_num: int = 10,
                 sample_steps: int = 4, sample_temperature: float = 1.0) -> None:
        super().__init__()
        del scene_layers
        self.max_candidates = int(max_candidates)
        self.d_model = int(d_model)
        self.future_len = int(future_len)
        self.token_size = max(2, int(token_size))
        self.token_stride = max(1, int(token_stride))
        self.cfg_dropout = float(cfg_dropout)
        self.cfg_weight = float(cfg_weight)
        self.consistency_weight = float(consistency_weight)
        self.energy_weight = float(energy_weight)
        self.cfg_neighbor_num = int(cfg_neighbor_num)
        self.sample_steps = int(sample_steps)
        self.sample_temperature = float(sample_temperature)
        self.scene = SourceLikeSceneEncoder(d_model, dropout)
        self.segment_in = nn.Linear(self.token_size * 4, d_model)
        self.time_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.blocks = nn.ModuleList([ScaleAdaptiveFusionBlock(d_model, num_heads, dropout) for _ in range(num_layers)])
        self.segment_out = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, self.token_size * 4))
        self.scalar_heads = _ScalarHeads(input_dim, d_model)
        starts = self._make_starts(self.future_len)
        self.register_buffer("segment_starts", torch.as_tensor(starts, dtype=torch.long), persistent=False)

    def _make_starts(self, T: int) -> list[int]:
        starts = list(range(0, max(T - self.token_size + 1, 1), self.token_stride))
        last = max(T - self.token_size, 0)
        if not starts or starts[-1] != last:
            starts.append(last)
        return starts

    def _segments(self, z: torch.Tensor) -> torch.Tensor:
        # T is fixed by the benchmark config; indexed gather avoids Python loops in the hot path.
        T = z.shape[1]
        starts = self.segment_starts.to(z.device)
        offs = torch.arange(self.token_size, device=z.device)
        idx = (starts[:, None] + offs[None, :]).clamp_max(T - 1)
        q = z[:, idx]  # [B,K,S,4]
        return q.reshape(z.shape[0], starts.numel(), -1)

    def _fold(self, seg: torch.Tensor, T: int) -> torch.Tensor:
        B, K, _ = seg.shape
        starts = self.segment_starts.to(seg.device)
        q = seg.reshape(B, K, self.token_size, 4)
        out = seg.new_zeros(B, T, 4)
        cnt = seg.new_zeros(B, T, 1)
        for j, s_t in enumerate(starts.tolist()):
            e = min(s_t + self.token_size, T)
            ln = e - s_t
            out[:, s_t:e] += q[:, j, :ln]
            cnt[:, s_t:e] += 1.0
        return out / cnt.clamp_min(1.0)

    def _traj_token_positions(self, xt: torch.Tensor) -> torch.Tensor:
        seg = self._segments(xt).reshape(xt.shape[0], -1, self.token_size, 4)
        return seg[..., :2].mean(dim=2)

    def _decode_x1(self, xt: torch.Tensor, t: torch.Tensor, scene: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        seg = self._segments(xt)
        traj = self.segment_in(seg)
        cond = self.time_mlp(_time_embedding(t.reshape(-1), self.d_model)) + scene["context"]
        traj_mask = torch.ones(traj.shape[:2], dtype=torch.bool, device=traj.device)
        traj_pos = self._traj_token_positions(xt)
        agent, amap = scene["agent_tokens"], scene["map_tokens"]
        for block in self.blocks:
            agent, amap, traj = block(
                agent, amap, traj,
                scene["agent_mask"], scene["map_mask"], traj_mask,
                scene["agent_pos"], scene["map_pos"], traj_pos,
                cond,
            )
        pred_seg = self.segment_out(traj)
        pred = self._fold(pred_seg, xt.shape[1])
        return pred, pred_seg

    def _guided_velocity(self, xt: torch.Tensor, t: torch.Tensor, cond_scene: dict[str, torch.Tensor]) -> torch.Tensor:
        uncond_scene = _drop_nearest_neighbors(cond_scene, self.cfg_neighbor_num)
        # Source Flow Planner batches the conditioned and unconditioned branches.
        pair_scene = {k: torch.cat([cond_scene[k], uncond_scene[k]], dim=0) for k in cond_scene}
        x_pair = torch.cat([xt, xt], dim=0)
        t_pair = torch.cat([t, t], dim=0)
        pred_pair, _ = self._decode_x1(x_pair, t_pair, pair_scene)
        pred_cond, pred_uncond = pred_pair.chunk(2, dim=0)
        # CondOT x_t=t*x1+(1-t)*x0 => target_to_velocity=(x1-x_t)/(1-t).
        denom = (1.0 - t).view(-1, 1, 1).clamp_min(1.0e-4)
        u_cond = (pred_cond - xt) / denom
        u_uncond = (pred_uncond - xt) / denom
        return (1.0 - self.cfg_weight) * u_uncond + self.cfg_weight * u_cond

    def _generate(self, scene: dict[str, torch.Tensor], *, seed: int | None, dtype: torch.dtype) -> torch.Tensor:
        B = scene["context"].shape[0]
        dev = scene["context"].device
        gen = None
        if seed is not None:
            gen = torch.Generator(device=dev); gen.manual_seed(int(seed) & 0x7FFFFFFF)
        x = torch.randn((B, self.future_len, 4), device=dev, dtype=dtype, generator=gen) * self.sample_temperature
        h = 1.0 / max(self.sample_steps, 1)
        # Explicit midpoint, matching the released four-step Flow ODE configuration.
        for i in range(self.sample_steps):
            t0 = x.new_full((B,), i * h)
            k1 = self._guided_velocity(x, t0, scene)
            x_mid = x + 0.5 * h * k1
            tm = x.new_full((B,), (i + 0.5) * h)
            k2 = self._guided_velocity(x_mid, tm, scene)
            x = x + h * k2
        return x

    def _consistency(self, pred_seg: torch.Tensor) -> torch.Tensor:
        starts = self.segment_starts.tolist()
        q = pred_seg.reshape(pred_seg.shape[0], len(starts), self.token_size, 4)
        terms: list[torch.Tensor] = []
        for j in range(len(starts) - 1):
            overlap = max(0, starts[j] + self.token_size - starts[j + 1])
            if overlap > 0:
                terms.append((q[:, j, -overlap:] - q[:, j + 1, :overlap]).square().sum(dim=-1).mean())
        return torch.stack(terms).mean() if terms else pred_seg.new_zeros(())

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, *, prefix_traj: torch.Tensor | None = None,
                prefix_valid: torch.Tensor | None = None, source_agent_history: torch.Tensor | None = None,
                source_agent_valid: torch.Tensor | None = None, source_current_state: torch.Tensor | None = None,
                source_map_points: torch.Tensor | None = None, source_map_point_valid: torch.Tensor | None = None,
                source_map_meta: torch.Tensor | None = None, source_map_center: torch.Tensor | None = None,
                source_map_valid: torch.Tensor | None = None, source_centerline: torch.Tensor | None = None,
                target_index: torch.Tensor | None = None, sampling_seed: int | None = None,
                **_: torch.Tensor) -> dict[str, torch.Tensor]:
        B, N, _ = x.shape
        if prefix_traj is None:
            raise ValueError("FlowPlannerPort requires prefix_traj")
        candidates, candidate_valid = _trajectory_state(prefix_traj, prefix_valid)
        scene = self.scene(
            source_agent_history=source_agent_history, source_agent_valid=source_agent_valid,
            source_current_state=source_current_state, source_map_points=source_map_points,
            source_map_point_valid=source_map_point_valid, source_map_meta=source_map_meta,
            source_map_center=source_map_center, source_map_valid=source_map_valid,
            source_centerline=source_centerline, batch_size=B, device=x.device,
        )
        consistency = candidates.new_zeros(())
        if self.training:
            idx = torch.zeros(B, dtype=torch.long, device=x.device) if target_index is None else target_index.long().clamp(0, N - 1)
            b = torch.arange(B, device=x.device)
            target = candidates[b, idx]
            valid = candidate_valid[b, idx]
            base = torch.randn_like(target)
            t = torch.rand(B, device=x.device, dtype=target.dtype).clamp_(1.0e-3, 1.0 - 1.0e-3)
            xt = (1.0 - t[:, None, None]) * base + t[:, None, None] * target
            if self.cfg_dropout > 0:
                keep = torch.rand(B, device=x.device) >= self.cfg_dropout
                train_scene = dict(scene)
                # Source semantics: an unconditioned example masks nearest neighbors only.
                dropped = _drop_nearest_neighbors(scene, self.cfg_neighbor_num)
                train_scene["agent_mask"] = torch.where(keep[:, None], scene["agent_mask"], dropped["agent_mask"])
            else:
                train_scene = scene
            pred, pred_seg = self._decode_x1(xt, t, train_scene)
            consistency = self._consistency(pred_seg)
            generated = pred
            flow_target = target
            flow_valid = valid
        else:
            generated = self._generate(scene, seed=sampling_seed, dtype=candidates.dtype)
            flow_target = generated.detach()
            flow_valid = torch.ones(B, self.future_len, dtype=torch.bool, device=x.device)

        energy = _candidate_energy(generated, candidates, candidate_valid)
        logits = -self.energy_weight * energy
        if mask is not None:
            logits = logits.masked_fill(~mask.bool(), -1.0e4)
        out: dict[str, torch.Tensor] = {
            "logits": logits,
            "flow_pred_x1": generated,
            "flow_target_x1": flow_target,
            "flow_valid": flow_valid,
            "flow_endpoint": generated,
            "flow_generated": generated,
            "flow_energy": energy,
            "flow_consistency_loss": self.consistency_weight * consistency,
        }
        out.update(self.scalar_heads(x))
        return out


class PlanR1Port(nn.Module):
    """Plan-R1 trajectory-token LM + variance-decoupled policy-alignment port.

    The supplied Plan-R1 repository discretizes motion into 1024 vehicle tokens,
    pretrains next-token prediction, then aligns a planning model against a frozen
    prediction/reference model with VD-GRPO and a KL penalty.  This port uses the
    supplied 1024-token vehicle codebook when available and exposes both reference
    and planning candidate log-probabilities; the OC-RAP trainer implements the
    fixed-scale, group-centered VD-GRPO objective without per-group variance
    normalization.
    """

    def __init__(self, input_dim: int, max_candidates: int = 24, d_model: int = 128, num_layers: int = 4,
                 num_heads: int = 8, dropout: float = 0.1, future_len: int = 20, scene_layers: int = 2,
                 token_interval: int = 5, num_tokens: int = 1024, token_codebook: str | None = None) -> None:
        super().__init__()
        self.max_candidates = int(max_candidates)
        self.d_model = int(d_model)
        self.future_len = int(future_len)
        self.token_interval = max(1, int(token_interval))
        self.num_tokens = int(num_tokens)
        codebook = self._load_codebook(token_codebook, self.num_tokens)
        self.register_buffer("vehicle_codebook", codebook, persistent=True)
        self.scene = CompactSceneEncoder(d_model, num_heads, scene_layers, dropout)
        self.token_embedding = nn.Embedding(self.num_tokens + 1, d_model)  # + BOS
        self.pos = nn.Parameter(torch.randn(1, max(4, math.ceil(future_len / self.token_interval) + 1), d_model) * 0.02)
        self.reference_transformer = _make_encoder(d_model, num_heads, num_layers, dropout)
        self.plan_transformer = _make_encoder(d_model, num_heads, num_layers, dropout)
        self.reference_head = nn.Linear(d_model, self.num_tokens)
        self.plan_head = nn.Linear(d_model, self.num_tokens)
        self.candidate_residual = _CandidatePrior(input_dim, d_model, dropout)
        self.scalar_heads = _ScalarHeads(input_dim, d_model)

    @staticmethod
    def _load_codebook(path: str | None, num_tokens: int) -> torch.Tensor:
        if path:
            p = Path(path)
            if p.exists():
                try:
                    obj = torch.load(p, map_location="cpu", weights_only=True)
                except TypeError:
                    obj = torch.load(p, map_location="cpu")
                if isinstance(obj, dict) and "Vehicle" in obj:
                    cb = torch.as_tensor(obj["Vehicle"], dtype=torch.float32)
                    if cb.ndim == 2 and cb.shape[1] >= 3 and cb.shape[0] >= num_tokens:
                        return cb[:num_tokens, :3].contiguous()
        # Deterministic fallback for unit tests / source-free packaging.  The
        # shipped config points at the copied author codebook, so publication
        # runs do not use this branch.
        k = torch.arange(num_tokens, dtype=torch.float32)
        angle = (k % 64) / 64.0 * 2.0 * math.pi - math.pi
        radius = 0.2 + 8.0 * torch.floor(k / 64) / max(math.ceil(num_tokens / 64) - 1, 1)
        heading = ((k * 17) % num_tokens) / max(num_tokens - 1, 1) * 2.0 * math.pi - math.pi
        return torch.stack([radius * angle.cos(), radius * angle.sin(), heading], dim=-1)

    def _tokenize(self, prefix_traj: torch.Tensor, prefix_valid: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        xy = prefix_traj.float()
        B, N, T, _ = xy.shape
        valid = prefix_valid.bool() if prefix_valid is not None else torch.ones(B, N, T, dtype=torch.bool, device=xy.device)
        # Heading at every prefix step.
        origin = torch.zeros_like(xy[..., :1, :])
        prev = torch.cat([origin, xy[..., :-1, :]], dim=-2)
        d = xy - prev
        yaw = torch.atan2(d[..., 1], torch.where(d.square().sum(-1) > 1e-8, d[..., 0], torch.ones_like(d[..., 0])))
        indices = list(range(self.token_interval - 1, T, self.token_interval))
        if not indices or indices[-1] != T - 1:
            indices.append(T - 1)
        pos_list = [torch.zeros(B, N, 2, device=xy.device, dtype=xy.dtype)]
        yaw_list = [torch.zeros(B, N, device=xy.device, dtype=xy.dtype)]
        v_list = [torch.ones(B, N, device=xy.device, dtype=torch.bool)]
        for idx in indices:
            pos_list.append(xy[..., idx, :])
            yaw_list.append(yaw[..., idx])
            v_list.append(valid[..., idx])
        pos = torch.stack(pos_list, dim=-2)
        ang = torch.stack(yaw_list, dim=-1)
        vm = torch.stack(v_list, dim=-1)
        p0, p1 = pos[..., :-1, :], pos[..., 1:, :]
        a0, a1 = ang[..., :-1], ang[..., 1:]
        dp = p1 - p0
        ca, sa = a0.cos(), a0.sin()
        relx = ca * dp[..., 0] + sa * dp[..., 1]
        rely = -sa * dp[..., 0] + ca * dp[..., 1]
        dha = torch.atan2(torch.sin(a1 - a0), torch.cos(a1 - a0))
        rel = torch.stack([relx, rely, dha], dim=-1)
        tok_valid = vm[..., :-1] & vm[..., 1:]
        cb = self.vehicle_codebook.to(device=xy.device, dtype=xy.dtype)
        # Heading is wrapped before distance; translational terms dominate like
        # the source average-corner-distance tokenizer.
        diff_xy = rel[..., None, :2] - cb[None, None, None, :, :2]
        dh = torch.atan2(
            torch.sin(rel[..., None, 2] - cb[None, None, None, :, 2]),
            torch.cos(rel[..., None, 2] - cb[None, None, None, :, 2]),
        )
        dist = diff_xy.square().sum(-1) + 0.5 * dh.square()
        tokens = dist.argmin(dim=-1)
        return tokens.long(), tok_valid.bool()

    def _lm(self, tokens: torch.Tensor, scene: torch.Tensor, transformer: nn.TransformerEncoder, head: nn.Linear) -> torch.Tensor:
        B, N, S = tokens.shape
        bos = torch.full((B, N, 1), self.num_tokens, dtype=torch.long, device=tokens.device)
        inp = torch.cat([bos, tokens[..., :-1]], dim=-1)
        h = self.token_embedding(inp).reshape(B * N, S, self.d_model)
        pos = self.pos[:, :S]
        cond = scene[:, None, :].expand(B, N, -1).reshape(B * N, 1, self.d_model)
        h = h + pos + cond
        causal = torch.triu(torch.ones(S, S, dtype=torch.bool, device=tokens.device), diagonal=1)
        h = transformer(h, mask=causal)
        return head(h).reshape(B, N, S, self.num_tokens)

    @staticmethod
    def _sequence_score(logits: torch.Tensor, tokens: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        lp = F.log_softmax(logits.float(), dim=-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
        lp = torch.where(valid, lp, torch.zeros_like(lp))
        return lp.sum(-1) / valid.float().sum(-1).clamp_min(1.0)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, *, prefix_traj: torch.Tensor | None = None,
                prefix_valid: torch.Tensor | None = None, source_agent_history: torch.Tensor | None = None,
                source_agent_valid: torch.Tensor | None = None, source_current_state: torch.Tensor | None = None,
                source_map_points: torch.Tensor | None = None, source_map_point_valid: torch.Tensor | None = None,
                source_map_meta: torch.Tensor | None = None, source_map_center: torch.Tensor | None = None,
                source_map_valid: torch.Tensor | None = None, source_centerline: torch.Tensor | None = None,
                **_: torch.Tensor) -> dict[str, torch.Tensor]:
        B, N, _ = x.shape
        if prefix_traj is None:
            raise ValueError("PlanR1Port requires prefix_traj")
        tokens, token_valid = self._tokenize(prefix_traj, prefix_valid)
        scene = self.scene(
            source_agent_history=source_agent_history, source_agent_valid=source_agent_valid,
            source_current_state=source_current_state, source_map_points=source_map_points,
            source_map_point_valid=source_map_point_valid, source_map_meta=source_map_meta,
            source_map_center=source_map_center, source_map_valid=source_map_valid,
            source_centerline=source_centerline, batch_size=B, device=x.device,
        )
        ref_token_logits = self._lm(tokens, scene, self.reference_transformer, self.reference_head)
        plan_token_logits = self._lm(tokens, scene, self.plan_transformer, self.plan_head)
        ref_score = self._sequence_score(ref_token_logits, tokens, token_valid)
        plan_score = self._sequence_score(plan_token_logits, tokens, token_valid) + self.candidate_residual(x)
        if mask is not None:
            plan_score = plan_score.masked_fill(~mask.bool(), -1.0e4)
            ref_score = ref_score.masked_fill(~mask.bool(), -1.0e4)
        out: dict[str, torch.Tensor] = {
            "logits": plan_score,
            "planr1_reference_logits": ref_score,
            "planr1_reference_token_logits": ref_token_logits,
            "planr1_plan_token_logits": plan_token_logits,
            "planr1_token_target": tokens,
            "planr1_token_valid": token_valid,
        }
        out.update(self.scalar_heads(x))
        return out
