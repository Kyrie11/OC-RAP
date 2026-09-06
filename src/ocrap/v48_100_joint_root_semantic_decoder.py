from __future__ import annotations

import math

import torch
from torch import nn

from ocrap.v48_97_executable_recovery_state import ExecutableRecoverySufficientState

ENGINEERING_VERSION = "v48.100.0-OC-JRSD"
ALGORITHM_NAME = "Observation-Consistent Joint Root-Semantic Decoder"


class JointRootSemanticDecoder(nn.Module):
    """Minimal control-sufficient recovery chart over a frozen decoder body.

    V48.97 showed that a two-coordinate recovery chart can be statically useful,
    while V48.98--99 showed that fitting an action tangent *after* fixing that
    chart is insufficient.  V48.100 therefore changes the representation map
    itself, but only through the smallest root-decoder degrees of freedom that
    define which observation-conditioned hypotheses are queried.

    The frozen L80 root queries are treated as anchors q0 and only a zero-init
    query displacement dq is trained.  The cross/self-attention, FFN, root-logit
    head and structured encoder stay frozen.  The V48.97 two-coordinate ERSS
    chart is initialized from its certified state and is trained jointly with dq.

    The resulting commuting diagram is

        (observation, action) --frozen encoder--> memory
                 --(q0+dq, frozen decoder body)--> root measure
                 --trainable 2D semantic chart--> (D, R)

    so static recovery state and candidate-minus-nominal action change are
    learned in the same coordinate system.  No source/admission logit, regime id,
    root id, option id, root-slot correspondence, or teacher metadata is input.
    """

    def __init__(self, *, base_root_queries: torch.Tensor, d_model: int):
        super().__init__()
        q = base_root_queries.detach().float().clone()
        if q.ndim != 3 or q.shape[0] != 1 or q.shape[-1] != int(d_model):
            raise ValueError("V48.100 base root-query shape mismatch")
        self.d_model = int(d_model)
        self.num_roots = int(q.shape[1])
        self.register_buffer("base_root_queries", q, persistent=True)
        self.query_delta = nn.Parameter(torch.zeros_like(q))
        self.chart = ExecutableRecoverySufficientState(self.d_model)

    @property
    def query_parameter_count(self) -> int:
        return int(self.query_delta.numel())

    @property
    def chart_parameter_count(self) -> int:
        return int(self.chart.trainable_parameter_count)

    @property
    def trainable_parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def load_chart_state(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.chart.load_state_dict(state_dict, strict=True)

    def effective_queries(self, batch_size: int) -> torch.Tensor:
        return (self.base_root_queries + self.query_delta).expand(int(batch_size), -1, -1)

    def decode_roots(self, model: nn.Module, memory: torch.Tensor) -> torch.Tensor:
        """Use the frozen L80 decoder body with the learned root-query chart."""
        q0 = self.effective_queries(memory.shape[0]).to(dtype=memory.dtype, device=memory.device)
        q, _ = model.root_cross_attn(q0, memory, memory, need_weights=False)
        q = model.root_norm1(q0 + q)
        qs, _ = model.root_self_attn(q, q, q, need_weights=False)
        q = model.root_norm2(q + qs)
        q = model.root_norm3(q + model.root_ffn(q))
        return q

    def forward(
        self,
        *,
        model: nn.Module,
        memory: torch.Tensor,
        root_valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        rt = self.decode_roots(model, memory)
        logits = model.root_logit_head(rt).squeeze(-1).float()
        valid = root_valid.bool()
        logits = logits.masked_fill(~valid, -1.0e9)
        p = torch.softmax(logits, dim=-1) * valid.float()
        p = p / p.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
        out = self.chart(rt, p, valid)
        out["root_tokens"] = rt
        out["root_probs"] = p
        return out


def _safe_scale(x: torch.Tensor, floor: float = 1.0e-3) -> torch.Tensor:
    x = x.float()
    if x.numel() == 0:
        raise ValueError("V48.100 semantic scale requires non-empty target")
    return torch.sqrt(torch.mean(x * x)).clamp_min(float(floor)).detach()


def joint_semantic_scales(
    teacher_support: torch.Tensor,
    teacher_reserve: torch.Tensor,
    teacher_support_delta: torch.Tensor,
    teacher_reserve_delta: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Fixed coordinate metric from training labels only; never swept.

    Absolute terms are centered before RMS scaling so a constant offset cannot
    dominate the product-space metric.  Delta terms use RMS directly.
    """
    td = teacher_support.float()
    tr = teacher_reserve.float()
    return {
        "support": _safe_scale(td - td.mean()),
        "reserve": _safe_scale(tr - tr.mean()),
        "delta_support": _safe_scale(teacher_support_delta),
        "delta_reserve": _safe_scale(teacher_reserve_delta),
    }


def joint_semantic_loss(
    support: torch.Tensor,
    reserve: torch.Tensor,
    teacher_support: torch.Tensor,
    teacher_reserve: torch.Tensor,
    candidate_index: torch.Tensor,
    nominal_index: torch.Tensor,
    scales: dict[str, torch.Tensor | float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Coordinate-invariant static + action-equivariant recovery objective."""
    s = support.float()
    r = reserve.float()
    td = teacher_support.float().clamp(0.0, 1.0)
    tr = teacher_reserve.float()

    def huber(err: torch.Tensor, scale: torch.Tensor | float) -> torch.Tensor:
        sc = torch.as_tensor(scale, dtype=err.dtype, device=err.device).clamp_min(1.0e-6)
        z = err / sc
        return torch.nn.functional.smooth_l1_loss(z, torch.zeros_like(z), beta=1.0)

    l_s = huber(s - td, scales["support"])
    l_r = huber(r - tr, scales["reserve"])
    if candidate_index.numel() > 0:
        ci = candidate_index.long(); ni = nominal_index.long()
        ds = (s.index_select(0, ci) - s.index_select(0, ni)) - (td.index_select(0, ci) - td.index_select(0, ni))
        dr = (r.index_select(0, ci) - r.index_select(0, ni)) - (tr.index_select(0, ci) - tr.index_select(0, ni))
        l_ds = huber(ds, scales["delta_support"])
        l_dr = huber(dr, scales["delta_reserve"])
    else:
        z = s.sum() * 0.0
        l_ds = z; l_dr = z
    total = (l_s + l_r + l_ds + l_dr) / 4.0
    return total, {
        "support_normalized": l_s,
        "reserve_normalized": l_r,
        "delta_support_normalized": l_ds,
        "delta_reserve_normalized": l_dr,
    }


def zero_delta_decoder_identity_check(d_model: int = 16, num_roots: int = 5, num_heads: int = 4) -> bool:
    """Synthetic proof that zero query displacement reproduces frozen decoding."""
    torch.manual_seed(100)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.root_queries = nn.Parameter(torch.randn(1, num_roots, d_model) * 0.02)
            self.root_cross_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
            self.root_self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
            self.root_norm1 = nn.LayerNorm(d_model); self.root_norm2 = nn.LayerNorm(d_model)
            self.root_ffn = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Linear(4*d_model, d_model))
            self.root_norm3 = nn.LayerNorm(d_model)
            self.root_logit_head = nn.Linear(d_model, 1)
        def _decode_roots(self, memory):
            q0 = self.root_queries.expand(memory.shape[0], -1, -1)
            q,_ = self.root_cross_attn(q0,memory,memory,need_weights=False); q=self.root_norm1(q0+q)
            qs,_ = self.root_self_attn(q,q,q,need_weights=False); q=self.root_norm2(q+qs)
            return self.root_norm3(q+self.root_ffn(q))

    m = M().eval()
    for p in m.parameters(): p.requires_grad_(False)
    j = JointRootSemanticDecoder(base_root_queries=m.root_queries, d_model=d_model).eval()
    mem = torch.randn(3, 7, d_model)
    with torch.no_grad():
        a = m._decode_roots(mem); b = j.decode_roots(m, mem)
    return bool(torch.equal(a, b))


def trainable_contract_check(d_model: int = 192, num_roots: int = 8) -> bool:
    q = torch.zeros(1, num_roots, d_model)
    m = JointRootSemanticDecoder(base_root_queries=q, d_model=d_model)
    expected = num_roots * d_model + (4 * d_model + 2)
    return bool(m.query_parameter_count == num_roots*d_model and m.chart_parameter_count == 4*d_model+2 and m.trainable_parameter_count == expected)


def query_gradient_check(d_model: int = 16, num_roots: int = 5, num_heads: int = 4) -> bool:
    torch.manual_seed(101)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.root_queries = nn.Parameter(torch.randn(1, num_roots, d_model) * 0.02)
            self.root_cross_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
            self.root_self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
            self.root_norm1 = nn.LayerNorm(d_model); self.root_norm2 = nn.LayerNorm(d_model)
            self.root_ffn = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Linear(4*d_model, d_model))
            self.root_norm3 = nn.LayerNorm(d_model)
            self.root_logit_head = nn.Linear(d_model, 1)
    base=M().eval()
    for p in base.parameters(): p.requires_grad_(False)
    mod=JointRootSemanticDecoder(base_root_queries=base.root_queries,d_model=d_model)
    mem=torch.randn(4,7,d_model); rv=torch.ones(4,num_roots,dtype=torch.bool)
    out=mod(model=base,memory=mem,root_valid=rv)
    loss=out['support'].mean()+out['reserve_debt'].mean()
    loss.backward()
    g=mod.query_delta.grad
    return bool(g is not None and torch.isfinite(g).all() and float(g.abs().sum())>0.0)
