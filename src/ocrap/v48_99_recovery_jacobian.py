from __future__ import annotations

import math

import torch
from torch import nn

ENGINEERING_VERSION = "v48.99.0-OC-RJCA"
ALGORITHM_NAME = "Observation-Consistent Recovery-Jacobian Control-Affine Alignment"


class ObservationConditionedRecoveryJacobian(nn.Module):
    """Rank-2 state-conditioned control-affine update of decoded root geometry.

    V48.97 establishes a stable two-coordinate nominal recovery chart, while
    V48.98 rejects a *global*, observation-independent Stage-I tangent plane.
    The minimal next object is therefore a local control Jacobian whose
    coefficients depend on the current observation-conditioned recovery state.

    Let h0=[h_D,h_R] be the two frozen V48.97 semantic pooled states and da the
    candidate-minus-nominal executable physical action.  The two semantic
    coefficients are

        eta = (A da) * (C [1, h0]),

    which is the rank-2 bilinear/control-affine interaction between action and
    nominal state.  eta is injected only into the decoded *candidate* root set,
    using the frozen semantic attention weights as a permutation-equivariant
    allocation measure.  There is no root-slot correspondence and no regime,
    option, macro, or teacher metadata input.

    da=0 makes eta exactly zero, so every nominal root set and the V48.97 state
    chart are preserved by construction.
    """

    def __init__(self, *, d_model: int, action_dim: int):
        super().__init__()
        self.d_model = int(d_model)
        self.action_dim = int(action_dim)
        if self.d_model <= 0 or self.action_dim <= 0:
            raise ValueError("V48.99 dimensions must be positive")
        self.context_dim = 1 + 2 * self.d_model
        # Two semantic directions, fixed by support/reserve dimensionality.
        self.basis_raw = nn.Parameter(torch.empty(self.d_model, 2, dtype=torch.float32))
        nn.init.normal_(self.basis_raw, mean=0.0, std=1.0 / math.sqrt(self.d_model))
        # State/action factors are rank exactly two.  The action map is zero-init
        # so epoch 0 is exactly the V48.97 representation while gradients to it
        # are non-zero through the non-zero state/basis factors.
        self.action_map = nn.Parameter(torch.zeros(2, self.action_dim, dtype=torch.float32))
        self.state_map = nn.Parameter(torch.empty(2, self.context_dim, dtype=torch.float32))
        nn.init.xavier_uniform_(self.state_map)

    @property
    def trainable_parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def basis(self) -> torch.Tensor:
        q, _ = torch.linalg.qr(self.basis_raw.float(), mode="reduced")
        return q[:, :2]

    def coefficients(
        self,
        *,
        action_delta: torch.Tensor,
        nominal_support_state: torch.Tensor,
        nominal_reserve_state: torch.Tensor,
    ) -> torch.Tensor:
        if action_delta.ndim != 2 or action_delta.shape[-1] != self.action_dim:
            raise ValueError("V48.99 action-delta shape mismatch")
        for name, h in (("support", nominal_support_state), ("reserve", nominal_reserve_state)):
            if h.ndim != 2 or h.shape[-1] != self.d_model or h.shape[0] != action_delta.shape[0]:
                raise ValueError(f"V48.99 nominal {name}-state shape mismatch")
        # Preserve the magnitude of the executable action delta.  These blocks
        # are already in the model's registered physical feature coordinates;
        # normalizing a delta to unit norm would erase actuation magnitude.
        a = action_delta.float()
        hs = torch.nn.functional.layer_norm(nominal_support_state.float(), (self.d_model,))
        hr = torch.nn.functional.layer_norm(nominal_reserve_state.float(), (self.d_model,))
        one = torch.ones((a.shape[0], 1), dtype=a.dtype, device=a.device)
        ctx = torch.cat([one, hs, hr], dim=-1)
        action_factor = torch.nn.functional.linear(a, self.action_map.float())
        state_factor = torch.nn.functional.linear(ctx, self.state_map.float())
        return action_factor * state_factor

    def forward(
        self,
        *,
        root_tokens: torch.Tensor,
        support_weights: torch.Tensor,
        reserve_weights: torch.Tensor,
        action_delta: torch.Tensor,
        nominal_support_state: torch.Tensor,
        nominal_reserve_state: torch.Tensor,
    ) -> torch.Tensor:
        if root_tokens.ndim != 3 or root_tokens.shape[-1] != self.d_model:
            raise ValueError("V48.99 root-token shape mismatch")
        if support_weights.shape != root_tokens.shape[:2] or reserve_weights.shape != root_tokens.shape[:2]:
            raise ValueError("V48.99 semantic-weight shape mismatch")
        eta = self.coefficients(
            action_delta=action_delta,
            nominal_support_state=nominal_support_state,
            nominal_reserve_state=nominal_reserve_state,
        )
        u = self.basis()
        # Allocate each semantic component through its *frozen* V48.97 semantic
        # measure.  This is permutation-equivariant and assumes no root bijection.
        ds = support_weights.float().unsqueeze(-1) * eta[:, 0:1].unsqueeze(1) * u[:, 0].view(1, 1, -1)
        dr = reserve_weights.float().unsqueeze(-1) * eta[:, 1:2].unsqueeze(1) * u[:, 1].view(1, 1, -1)
        return root_tokens + (ds + dr).to(dtype=root_tokens.dtype)


def semantic_attention_weights(
    erss: nn.Module,
    root_tokens: torch.Tensor,
    root_probs: torch.Tensor,
    root_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Frozen V48.97 support/reserve attention measures, without slot identity."""
    if root_tokens.ndim != 3:
        raise ValueError("root_tokens must be [B,K,D]")
    valid = root_valid.bool()
    p = root_probs.float().clamp_min(0.0) * valid.float()
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
    r = erss.root_norm(root_tokens.float())

    def one(query: torch.Tensor) -> torch.Tensor:
        score = torch.einsum("bkd,d->bk", r, query.float()) / math.sqrt(root_tokens.shape[-1])
        score = score + torch.log(p.clamp_min(1.0e-12))
        score = score.masked_fill(~valid, -1.0e9)
        a = torch.softmax(score, dim=-1) * valid.float()
        return a / a.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)

    return one(erss.support_query).detach(), one(erss.reserve_query).detach()


def semantic_delta_scales(
    teacher_support_delta: torch.Tensor,
    teacher_reserve_delta: torch.Tensor,
    *,
    floor: float = 1.0e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dimensionless metric scales fixed once from the training teacher deltas.

    D and R have different units/ranges.  Raw equal-weight Huber is not invariant
    to an arbitrary re-scaling of either semantic coordinate.  RMS normalization
    supplies a fixed product-space metric without a class/loss-weight sweep.
    """
    ds = teacher_support_delta.float()
    dr = teacher_reserve_delta.float()
    if ds.numel() == 0 or dr.numel() == 0:
        raise ValueError("V48.99 semantic scales require non-empty deltas")
    ss = torch.sqrt(torch.mean(ds * ds)).clamp_min(float(floor))
    sr = torch.sqrt(torch.mean(dr * dr)).clamp_min(float(floor))
    return ss.detach(), sr.detach()


def normalized_tangent_loss(
    pred_support_delta: torch.Tensor,
    pred_reserve_delta: torch.Tensor,
    teacher_support_delta: torch.Tensor,
    teacher_reserve_delta: torch.Tensor,
    support_scale: torch.Tensor | float,
    reserve_scale: torch.Tensor | float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ss = torch.as_tensor(support_scale, dtype=pred_support_delta.dtype, device=pred_support_delta.device).clamp_min(1.0e-6)
    sr = torch.as_tensor(reserve_scale, dtype=pred_reserve_delta.dtype, device=pred_reserve_delta.device).clamp_min(1.0e-6)
    es = (pred_support_delta.float() - teacher_support_delta.float()) / ss
    er = (pred_reserve_delta.float() - teacher_reserve_delta.float()) / sr
    z_s = torch.zeros_like(es)
    z_r = torch.zeros_like(er)
    ls = torch.nn.functional.smooth_l1_loss(es, z_s, beta=1.0)
    lr = torch.nn.functional.smooth_l1_loss(er, z_r, beta=1.0)
    return (ls + lr) / 2.0, {"delta_support_normalized": ls, "delta_reserve_normalized": lr}


def nominal_identity_synthetic_check(d_model: int = 16, action_dim: int = 9) -> bool:
    torch.manual_seed(99)
    m = ObservationConditionedRecoveryJacobian(d_model=d_model, action_dim=action_dim)
    r = torch.randn(4, 7, d_model)
    ws = torch.softmax(torch.randn(4, 7), dim=-1)
    wr = torch.softmax(torch.randn(4, 7), dim=-1)
    h1 = torch.randn(4, d_model)
    h2 = torch.randn(4, d_model)
    out = m(
        root_tokens=r,
        support_weights=ws,
        reserve_weights=wr,
        action_delta=torch.zeros(4, action_dim),
        nominal_support_state=h1,
        nominal_reserve_state=h2,
    )
    return bool(torch.equal(out, r))


def root_permutation_equivariance_synthetic_check(d_model: int = 16, action_dim: int = 9) -> bool:
    torch.manual_seed(99)
    m = ObservationConditionedRecoveryJacobian(d_model=d_model, action_dim=action_dim)
    with torch.no_grad():
        m.action_map.normal_(0.0, 0.1)
    r = torch.randn(3, 7, d_model)
    ws = torch.softmax(torch.randn(3, 7), dim=-1)
    wr = torch.softmax(torch.randn(3, 7), dim=-1)
    da = torch.randn(3, action_dim)
    h1 = torch.randn(3, d_model)
    h2 = torch.randn(3, d_model)
    perm = torch.tensor([4, 0, 6, 2, 1, 5, 3])
    a = m(root_tokens=r, support_weights=ws, reserve_weights=wr, action_delta=da,
          nominal_support_state=h1, nominal_reserve_state=h2)
    b = m(root_tokens=r[:, perm], support_weights=ws[:, perm], reserve_weights=wr[:, perm], action_delta=da,
          nominal_support_state=h1, nominal_reserve_state=h2)
    return bool(torch.allclose(a[:, perm], b, atol=1e-6, rtol=0.0))


def observation_conditioning_synthetic_check(d_model: int = 16, action_dim: int = 9) -> bool:
    torch.manual_seed(99)
    m = ObservationConditionedRecoveryJacobian(d_model=d_model, action_dim=action_dim)
    with torch.no_grad():
        m.action_map.normal_(0.0, 0.2)
    da = torch.randn(2, action_dim)
    da[1] = da[0]
    h1 = torch.randn(2, d_model)
    h2 = torch.randn(2, d_model)
    c = m.coefficients(action_delta=da, nominal_support_state=h1, nominal_reserve_state=h2)
    return bool(torch.max(torch.abs(c[0] - c[1])).item() > 1.0e-5)




def zero_init_nonzero_gradient_synthetic_check(d_model: int = 16, action_dim: int = 9) -> bool:
    torch.manual_seed(99)
    m = ObservationConditionedRecoveryJacobian(d_model=d_model, action_dim=action_dim)
    r = torch.randn(3, 5, d_model)
    ws = torch.softmax(torch.randn(3, 5), dim=-1)
    wr = torch.softmax(torch.randn(3, 5), dim=-1)
    da = torch.randn(3, action_dim)
    h1 = torch.randn(3, d_model)
    h2 = torch.randn(3, d_model)
    out = m(root_tokens=r, support_weights=ws, reserve_weights=wr, action_delta=da,
            nominal_support_state=h1, nominal_reserve_state=h2)
    target = r + 0.1 * torch.randn_like(r)
    loss = torch.nn.functional.mse_loss(out, target)
    loss.backward()
    g = m.action_map.grad
    return bool(g is not None and torch.isfinite(g).all() and torch.count_nonzero(g).item() > 0)

def action_magnitude_linearity_synthetic_check(d_model: int = 16, action_dim: int = 9) -> bool:
    torch.manual_seed(99)
    m = ObservationConditionedRecoveryJacobian(d_model=d_model, action_dim=action_dim)
    with torch.no_grad():
        m.action_map.normal_(0.0, 0.2)
    a = torch.randn(3, action_dim)
    h1 = torch.randn(3, d_model)
    h2 = torch.randn(3, d_model)
    c1 = m.coefficients(action_delta=a, nominal_support_state=h1, nominal_reserve_state=h2)
    c2 = m.coefficients(action_delta=2.0 * a, nominal_support_state=h1, nominal_reserve_state=h2)
    return bool(torch.allclose(c2, 2.0 * c1, atol=1e-6, rtol=1e-5))

def coordinate_scale_invariance_synthetic_check() -> bool:
    p1 = torch.tensor([0.1, -0.2, 0.3])
    t1 = torch.tensor([0.0, -0.1, 0.4])
    p2 = torch.tensor([1.0, -2.0, 3.0])
    t2 = torch.tensor([0.0, -1.0, 4.0])
    s1, s2 = semantic_delta_scales(t1, t2)
    a, _ = normalized_tangent_loss(p1, p2, t1, t2, s1, s2)
    b, _ = normalized_tangent_loss(10.0 * p1, 0.1 * p2, 10.0 * t1, 0.1 * t2, 10.0 * s1, 0.1 * s2)
    return bool(torch.allclose(a, b, atol=1e-7, rtol=0.0))
