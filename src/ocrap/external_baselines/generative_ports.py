"""Dependency-light ports for the uploaded Diffusion Planner, Flow Planner and Plan-R1.

The public projects use nuPlan-specific feature builders/simulation stacks.  OC-RAP
uses a common WOMD/Waymax executable-candidate interface instead.  These modules
retain the defining learning operators of each method and project their trajectory
likelihood / token likelihood onto the same finite candidate lattice used by the
other external baselines.  They intentionally do not claim checkpoint compatibility
with the authors' repositories.

Runtime choices are deliberately compact because the external-baseline launcher can
run three jobs per GPU: scene encoding is candidate-independent, candidate tensors
are vectorized, and Diffusion/Flow use a one-pass lattice energy at evaluation rather
than launching a Python solver separately for every candidate.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


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
    """Return [x,y,cos(yaw),sin(yaw)] and a valid mask for [B,N,T,2] prefixes."""
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


class DiffusionPlannerPort(nn.Module):
    """Diffusion-Planner conditional denoising-energy projection.

    The uploaded Diffusion-Planner uses a VP diffusion process and DiT-like
    conditional decoder with DPM-Solver inference.  Under OC-RAP's fixed action
    lattice, a vectorized conditional x0-denoising energy is sufficient to rank
    all executable candidates in one network call.  This avoids running a solver
    independently for 24 candidates while retaining diffusion corruption,
    timestep conditioning, scene conditioning, and x0 reconstruction training.
    """

    def __init__(self, input_dim: int, max_candidates: int = 24, d_model: int = 160, num_layers: int = 3,
                 num_heads: int = 5, dropout: float = 0.1, future_len: int = 20,
                 scene_layers: int = 2, energy_weight: float = 2.0, eval_noise_scale: float = 0.15,
                 beta_min: float = 0.1, beta_max: float = 20.0) -> None:
        super().__init__()
        self.max_candidates = int(max_candidates)
        self.d_model = int(d_model)
        self.future_len = int(future_len)
        self.energy_weight = float(energy_weight)
        self.eval_noise_scale = float(eval_noise_scale)
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.scene = CompactSceneEncoder(d_model, num_heads, scene_layers, dropout)
        self.state_in = nn.Linear(4, d_model)
        self.time_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.cond = nn.Linear(d_model, d_model)
        self.temporal = _make_encoder(d_model, num_heads, num_layers, dropout)
        self.state_out = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 4))
        self.prior = _CandidatePrior(input_dim, d_model, dropout)
        self.scalar_heads = _ScalarHeads(input_dim, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, *, prefix_traj: torch.Tensor | None = None,
                prefix_valid: torch.Tensor | None = None, source_agent_history: torch.Tensor | None = None,
                source_agent_valid: torch.Tensor | None = None, source_current_state: torch.Tensor | None = None,
                source_map_points: torch.Tensor | None = None, source_map_point_valid: torch.Tensor | None = None,
                source_map_meta: torch.Tensor | None = None, source_map_center: torch.Tensor | None = None,
                source_map_valid: torch.Tensor | None = None, source_centerline: torch.Tensor | None = None,
                **_: torch.Tensor) -> dict[str, torch.Tensor]:
        B, N, _ = x.shape
        if prefix_traj is None:
            raise ValueError("DiffusionPlannerPort requires prefix_traj")
        state, valid = _trajectory_state(prefix_traj, prefix_valid)
        T = state.shape[-2]
        scene = self.scene(
            source_agent_history=source_agent_history, source_agent_valid=source_agent_valid,
            source_current_state=source_current_state, source_map_points=source_map_points,
            source_map_point_valid=source_map_point_valid, source_map_meta=source_map_meta,
            source_map_center=source_map_center, source_map_valid=source_map_valid,
            source_centerline=source_centerline, batch_size=B, device=x.device,
        )
        if self.training:
            t = torch.rand(B, N, device=x.device, dtype=state.dtype).clamp_(0.02, 0.98)
            noise = torch.randn_like(state)
        else:
            t = torch.full((B, N), 0.55, device=x.device, dtype=state.dtype)
            base = torch.arange(T * 4, device=x.device, dtype=state.dtype).reshape(1, 1, T, 4)
            noise = self.eval_noise_scale * torch.sin(base * 0.731 + 0.17).expand_as(state)
        # Match the source Diffusion Planner's linear VP-SDE marginal instead
        # of the cosine schedule used by the earlier lightweight port.
        mean_log_coeff = -0.25 * t.square() * (self.beta_max - self.beta_min) - 0.5 * self.beta_min * t
        alpha = torch.exp(mean_log_coeff).clamp_min(1.0e-6)
        sigma = torch.sqrt((1.0 - torch.exp(2.0 * mean_log_coeff)).clamp_min(1.0e-8))
        noisy = alpha[..., None, None] * state + sigma[..., None, None] * noise
        h = self.state_in(noisy.reshape(B * N, T, 4))
        te = self.time_mlp(_time_embedding(t.reshape(B * N), self.d_model)).unsqueeze(1)
        sc = self.cond(scene[:, None, :].expand(B, N, -1).reshape(B * N, -1)).unsqueeze(1)
        h = self.temporal(h + te + sc, src_key_padding_mask=~valid.reshape(B * N, T))
        pred = self.state_out(h).reshape(B, N, T, 4)
        energy = _candidate_energy(pred, state, valid)
        logits = self.prior(x) - self.energy_weight * energy
        if mask is not None:
            logits = logits.masked_fill(~mask.bool(), -1.0e4)
        out: dict[str, torch.Tensor] = {
            "logits": logits,
            "diffusion_pred_x0": pred,
            "diffusion_target_x0": state,
            "diffusion_valid": valid,
            "diffusion_energy": energy,
        }
        out.update(self.scalar_heads(x))
        return out


class FlowPlannerPort(nn.Module):
    """Flow-Planner CondOT/CFG port with overlapping trajectory tokens.

    The source's fine-grained overlapping trajectory tokenization is retained via
    unfold/fold.  Training follows conditional optimal-transport flow matching;
    conditioning dropout implements classifier-free guidance training.  Evaluation
    uses one vectorized midpoint flow-energy pass over the executable lattice,
    which is substantially cheaper than an ODE solve per candidate.
    """

    def __init__(self, input_dim: int, max_candidates: int = 24, d_model: int = 160, num_layers: int = 3,
                 num_heads: int = 5, dropout: float = 0.1, future_len: int = 20, scene_layers: int = 2,
                 token_size: int = 5, token_stride: int = 3, cfg_dropout: float = 0.30,
                 cfg_weight: float = 1.8, consistency_weight: float = 0.5,
                 energy_weight: float = 1.5) -> None:
        super().__init__()
        self.max_candidates = int(max_candidates)
        self.d_model = int(d_model)
        self.future_len = int(future_len)
        self.token_size = max(2, int(token_size))
        self.token_stride = max(1, int(token_stride))
        self.cfg_dropout = float(cfg_dropout)
        self.cfg_weight = float(cfg_weight)
        self.consistency_weight = float(consistency_weight)
        self.energy_weight = float(energy_weight)
        self.scene = CompactSceneEncoder(d_model, num_heads, scene_layers, dropout)
        self.segment_in = nn.Linear(self.token_size * 4, d_model)
        self.time_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.blocks = _make_encoder(d_model, num_heads, num_layers, dropout)
        self.segment_out = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, self.token_size * 4))
        self.prior = _CandidatePrior(input_dim, d_model, dropout)
        self.scalar_heads = _ScalarHeads(input_dim, d_model)

    def _segments(self, z: torch.Tensor) -> tuple[torch.Tensor, list[int]]:
        # z [BN,T,4]; include a final anchored window so the tail is never dropped.
        T = z.shape[1]
        starts = list(range(0, max(T - self.token_size + 1, 1), self.token_stride))
        last = max(T - self.token_size, 0)
        if not starts or starts[-1] != last:
            starts.append(last)
        segs = []
        for s in starts:
            e = min(s + self.token_size, T)
            q = z[:, s:e]
            if q.shape[1] < self.token_size:
                q = F.pad(q, (0, 0, 0, self.token_size - q.shape[1]))
            segs.append(q.reshape(z.shape[0], -1))
        return torch.stack(segs, dim=1), starts

    def _fold(self, seg: torch.Tensor, starts: list[int], T: int) -> torch.Tensor:
        BN = seg.shape[0]
        out = seg.new_zeros(BN, T, 4)
        cnt = seg.new_zeros(BN, T, 1)
        q = seg.reshape(BN, len(starts), self.token_size, 4)
        for j, s in enumerate(starts):
            e = min(s + self.token_size, T)
            ln = e - s
            out[:, s:e] += q[:, j, :ln]
            cnt[:, s:e] += 1.0
        return out / cnt.clamp_min(1.0)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, *, prefix_traj: torch.Tensor | None = None,
                prefix_valid: torch.Tensor | None = None, source_agent_history: torch.Tensor | None = None,
                source_agent_valid: torch.Tensor | None = None, source_current_state: torch.Tensor | None = None,
                source_map_points: torch.Tensor | None = None, source_map_point_valid: torch.Tensor | None = None,
                source_map_meta: torch.Tensor | None = None, source_map_center: torch.Tensor | None = None,
                source_map_valid: torch.Tensor | None = None, source_centerline: torch.Tensor | None = None,
                **_: torch.Tensor) -> dict[str, torch.Tensor]:
        B, N, _ = x.shape
        if prefix_traj is None:
            raise ValueError("FlowPlannerPort requires prefix_traj")
        target, valid = _trajectory_state(prefix_traj, prefix_valid)
        T = target.shape[-2]
        scene = self.scene(
            source_agent_history=source_agent_history, source_agent_valid=source_agent_valid,
            source_current_state=source_current_state, source_map_points=source_map_points,
            source_map_point_valid=source_map_point_valid, source_map_meta=source_map_meta,
            source_map_center=source_map_center, source_map_valid=source_map_valid,
            source_centerline=source_centerline, batch_size=B, device=x.device,
        )
        if self.training:
            base = torch.randn_like(target)
            t = torch.rand(B, N, device=x.device, dtype=target.dtype).clamp_(0.02, 0.98)
        else:
            base = torch.zeros_like(target)
            t = torch.full((B, N), 0.5, device=x.device, dtype=target.dtype)
        xt = (1.0 - t[..., None, None]) * base + t[..., None, None] * target
        velocity_target = target - base
        flat = xt.reshape(B * N, T, 4)
        seg, starts = self._segments(flat)
        h0 = self.segment_in(seg)
        cond = scene[:, None, :].expand(B, N, -1).reshape(B * N, -1)
        time_cond = self.time_mlp(_time_embedding(t.reshape(B * N), self.d_model)).unsqueeze(1)
        if self.training:
            # Source Flow Planner samples one CFG mask per scene, then applies it
            # to that scene's trajectory tokens.  Candidate-wise dropout changes
            # the learned conditional/unconditional mixture and is avoided here.
            if self.cfg_dropout > 0:
                keep_scene = (torch.rand(B, 1, device=x.device) >= self.cfg_dropout).to(cond.dtype)
                keep = keep_scene[:, None, :].expand(B, N, 1).reshape(B * N, 1)
                cond = cond * keep
            h = self.blocks(h0 + cond.unsqueeze(1) + time_cond)
            vel_seg = self.segment_out(h)
        else:
            # Classifier-free guidance is a defining inference mechanism in Flow
            # Planner.  Evaluate conditional and unconditional branches in one
            # batched transformer call so fidelity does not require two serial
            # forwards. Source formula: (1-w) u_uncond + w u_cond.
            h_cond = h0 + cond.unsqueeze(1) + time_cond
            h_uncond = h0 + time_cond
            h_pair = self.blocks(torch.cat([h_cond, h_uncond], dim=0))
            seg_pair = self.segment_out(h_pair)
            seg_cond, seg_uncond = torch.chunk(seg_pair, 2, dim=0)
            vel_seg = (1.0 - self.cfg_weight) * seg_uncond + self.cfg_weight * seg_cond
        velocity_pred = self._fold(vel_seg, starts, T).reshape(B, N, T, 4)
        # Endpoint consistency of the conditional OT path at the sampled t.
        endpoint = xt + (1.0 - t[..., None, None]) * velocity_pred
        energy = _candidate_energy(endpoint, target, valid)
        logits = self.prior(x) - self.energy_weight * energy
        if mask is not None:
            logits = logits.masked_fill(~mask.bool(), -1.0e4)
        # Match the source overlap-consistency term on adjacent segment
        # predictions.  It is computed before folding/averaging the overlaps.
        q = vel_seg.reshape(B * N, len(starts), self.token_size, 4)
        consistency_terms = []
        for j in range(len(starts) - 1):
            overlap = max(0, starts[j] + self.token_size - starts[j + 1])
            if overlap > 0:
                consistency_terms.append((q[:, j, -overlap:] - q[:, j + 1, :overlap]).square().sum(dim=-1).mean())
        consistency = torch.stack(consistency_terms).mean() if consistency_terms else velocity_pred.new_zeros(())
        out: dict[str, torch.Tensor] = {
            "logits": logits,
            "flow_velocity_pred": velocity_pred,
            "flow_velocity_target": velocity_target,
            "flow_valid": valid,
            "flow_endpoint": endpoint,
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
