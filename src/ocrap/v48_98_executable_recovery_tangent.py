from __future__ import annotations

import math

import torch
from torch import nn

from ocrap.models.encoders import StructuredTokenEncoder

ENGINEERING_VERSION = "v48.98.0-OC-ERTA"
ALGORITHM_NAME = "Observation-Consistent Executable-Recovery Tangent Alignment"


class ExecutableRecoveryTangentAdapter(nn.Module):
    """Centered rank-2 Stage-I action-tangent update.

    V48.97 establishes that a nominal executable-recovery state can be learned
    from the frozen latent representation, while the candidate-induced support
    and reserve/debt deltas are not stable across Near/Contact.  This adapter
    therefore changes *representation*, not admission: it injects a two-
    dimensional recovery tangent into only the executable action-geometry
    tokens before the frozen structured transformer/root decoder.

    For candidate action a and nominal a0, each physical token j receives

        e'_j(a) = e_j(a) + U B_j [x_j(a) - x_j(a0)]

    where U has two orthonormal columns.  The semantic rank is exactly two
    because the registered recovery state has two coordinates (support and
    signed reserve/debt); rank is not a research hyperparameter.  No bias is
    allowed, so a=a0 gives exact zero update.  Shared observation tokens are
    untouched, and the frozen transformer supplies observation conditioning.
    """

    def __init__(self, *, d_model: int, prefix_param_dim: int, prefix_state_dim: int, control_dim: int):
        super().__init__()
        self.d_model = int(d_model)
        self.prefix_param_dim = int(prefix_param_dim)
        self.prefix_state_dim = int(prefix_state_dim)
        self.control_dim = int(control_dim)
        if min(self.d_model, self.prefix_param_dim, self.prefix_state_dim, self.control_dim) <= 0:
            raise ValueError("all V48.98 tangent dimensions must be positive")
        # Shared recovery tangent plane.  QR removes scale/rotation ambiguity in
        # the basis norm; semantic amplitude lives in the token coefficient maps.
        self.tangent_basis_raw = nn.Parameter(torch.empty(self.d_model, 2, dtype=torch.float32))
        nn.init.normal_(self.tangent_basis_raw, mean=0.0, std=1.0 / math.sqrt(self.d_model))
        # Candidate-relative physical token maps.  Zero init gives exact V48.97
        # representation at initialization and exact identity on nominal rows.
        self.prefix_param_map = nn.Parameter(torch.zeros(2, self.prefix_param_dim, dtype=torch.float32))
        self.prefix_state_map = nn.Parameter(torch.zeros(2, self.prefix_state_dim, dtype=torch.float32))
        self.control_map = nn.Parameter(torch.zeros(2, self.control_dim, dtype=torch.float32))

    @property
    def trainable_parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def basis(self) -> torch.Tensor:
        q, _ = torch.linalg.qr(self.tangent_basis_raw.float(), mode="reduced")
        return q[:, :2]

    def _residual(self, delta: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        coeff = torch.nn.functional.linear(delta.float(), weight.float())  # [B,2]
        return coeff @ self.basis().transpose(0, 1)  # [B,D]

    def forward(
        self,
        *,
        prefix_param_delta: torch.Tensor,
        prefix_state_delta: torch.Tensor,
        control_delta: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {
            "prefix_param": self._residual(prefix_param_delta, self.prefix_param_map),
            "prefix_state": self._residual(prefix_state_delta, self.prefix_state_map),
            "control": self._residual(control_delta, self.control_map),
        }


def scene_tokens_with_recovery_tangent(
    model: nn.Module,
    adapter: ExecutableRecoveryTangentAdapter,
    x_candidate: torch.Tensor,
    x_nominal: torch.Tensor,
) -> torch.Tensor:
    """Run the frozen structured Stage-I encoder with centered tangent injection."""
    if x_candidate.shape != x_nominal.shape:
        raise ValueError("candidate/nominal feature shapes differ")
    enc = getattr(model, "encoder", None)
    if not isinstance(enc, StructuredTokenEncoder):
        raise TypeError("V48.98 requires the structured transformer encoder")
    (
        ego, prefix_param, macro, scalar, prefix_state, control,
        agent_summary, agents, bev, route, maps, dyn,
    ) = enc._split(x_candidate)
    (
        _ego0, prefix_param0, _macro0, _scalar0, prefix_state0, control0,
        _agent_summary0, _agents0, _bev0, _route0, _maps0, _dyn0,
    ) = enc._split(x_nominal)

    residual = adapter(
        prefix_param_delta=prefix_param - prefix_param0,
        prefix_state_delta=prefix_state - prefix_state0,
        control_delta=control - control0,
    )

    B = x_candidate.shape[0]
    tokens = [
        enc.ego_proj(ego),
        enc.prefix_param_proj(prefix_param) + residual["prefix_param"],
        enc.macro_scalar_proj(torch.cat([macro, scalar], dim=-1)),
        enc.prefix_state_proj(prefix_state) + residual["prefix_state"],
        enc.control_proj(control) + residual["control"],
        enc.agent_summary_proj(agent_summary),
        enc.bev_proj(bev),
        enc.route_proj(route),
        enc.map_proj(maps),
        enc.dyn_proj(dyn),
    ]
    tok = torch.stack(tokens, dim=1)
    agent_tok = enc.agent_proj(agents)
    cls = enc.cls.expand(B, -1, -1)
    tok = torch.cat([cls, tok, agent_tok], dim=1)
    tok = tok + enc.pos[:, : tok.shape[1], :]
    return enc.norm(enc.encoder(tok))


def tangent_loss(
    *,
    support: torch.Tensor,
    reserve: torch.Tensor,
    teacher_drs: torch.Tensor,
    teacher_r_dep: torch.Tensor,
    candidate_index: torch.Tensor,
    nominal_index: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Only the unresolved action tangent is optimized.

    V48.97 state observability is already GO; V48.98 therefore freezes the
    state chart and allocates no capacity/loss to relearning nominal state.
    """
    ci = candidate_index.long()
    ni = nominal_index.long()
    if ci.numel() == 0:
        raise ValueError("V48.98 tangent training requires candidate/nominal pairs")
    ds = support.index_select(0, ci) - support.index_select(0, ni)
    dr = reserve.index_select(0, ci) - reserve.index_select(0, ni)
    tds = teacher_drs.float().index_select(0, ci) - teacher_drs.float().index_select(0, ni)
    tdr = teacher_r_dep.float().index_select(0, ci) - teacher_r_dep.float().index_select(0, ni)
    l_support = torch.nn.functional.smooth_l1_loss(ds, tds, beta=1.0)
    l_reserve = torch.nn.functional.smooth_l1_loss(dr, tdr, beta=1.0)
    total = (l_support + l_reserve) / 2.0
    return total, {"delta_support": l_support, "delta_reserve": l_reserve}


def nominal_identity_synthetic_check(d_model: int = 16) -> bool:
    torch.manual_seed(98)
    m = ExecutableRecoveryTangentAdapter(
        d_model=d_model, prefix_param_dim=5, prefix_state_dim=8, control_dim=4
    )
    z1 = torch.randn(7, 5)
    z2 = torch.randn(7, 8)
    z3 = torch.randn(7, 4)
    out = m(prefix_param_delta=z1 * 0.0, prefix_state_delta=z2 * 0.0, control_delta=z3 * 0.0)
    return bool(all(torch.count_nonzero(v).item() == 0 for v in out.values()))


def orthonormal_tangent_basis_synthetic_check(d_model: int = 16) -> bool:
    torch.manual_seed(98)
    m = ExecutableRecoveryTangentAdapter(
        d_model=d_model, prefix_param_dim=5, prefix_state_dim=8, control_dim=4
    )
    q = m.basis().detach()
    eye = q.transpose(0, 1) @ q
    return bool(torch.allclose(eye, torch.eye(2), atol=1e-5, rtol=0.0))
