from __future__ import annotations

"""V48.103 OC-FCSS: factorized control-sufficient recovery state.

V48.102 shows that a fixed architecture-aware summary of the frozen Stage-I
memory does not expose one linear statistic that simultaneously carries nominal
recovery state, support-establishment response and signed reserve/debt response.
That audit is deliberately weaker than an information-theoretic impossibility
claim because the full Stage-I token set still contains nonlinear/tokenwise
structure that the fixed summary can discard.

V48.103 is the preregistered *minimal Stage-I recovery representation objective*.
It keeps the historical Stage-I encoder and every planner/root/source parameter
frozen.  A tiny semantic bottleneck reads the full Stage-I token set and
factorizes the representation into:

  * a nominal recovery-state chart S(o), and
  * a candidate-induced response chart Delta(o, a; a0).

The two parts are composed in the same support/reserve coordinates, but no
learned state/delta mixing coefficient exists.  The nominal response is exactly
zero by construction.  This directly tests the durable V48.100--102 lesson
that static sufficiency and control sufficiency are distinct requirements.
"""

import math

import torch
from torch import nn

ENGINEERING_VERSION = "v48.103.0-OC-FCSS"
ALGORITHM_NAME = "Observation-Consistent Factorized Control-Sufficient State"


class FactorizedControlSufficientState(nn.Module):
    """Four fixed-capacity semantic set queries over frozen Stage-I memory.

    Channels:
      0: nominal support state
      1: nominal signed reserve/debt state
      2: support response potential
      3: reserve/debt response potential

    At d=192 the complete learned representation has exactly
    4 * (query[d] + readout[d] + bias[1]) = 1,540 parameters.
    There is no hidden MLP, rank, width, expert, root/option/regime id or source
    residual to sweep.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = int(d_model)
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        self.token_norm = nn.LayerNorm(self.d_model, elementwise_affine=False)
        self.queries = nn.Parameter(torch.empty(4, self.d_model, dtype=torch.float32))
        self.readouts = nn.Parameter(torch.empty(4, self.d_model, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(4, dtype=torch.float32))
        nn.init.normal_(self.queries, mean=0.0, std=1.0 / math.sqrt(self.d_model))
        nn.init.normal_(self.readouts, mean=0.0, std=1.0 / math.sqrt(self.d_model))

    @property
    def trainable_parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def semantic_potentials(self, memory: torch.Tensor) -> torch.Tensor:
        """Return [B,4] content-addressed semantic potentials.

        The pooling is invariant to a permutation of token rows at this
        interface and introduces no token IDs.  Historical token content may of
        course already encode the structured encoder's semantics/positions.
        """
        if memory.ndim != 3 or memory.shape[-1] != self.d_model:
            raise ValueError("V48.103 memory must have shape [B,T,d_model]")
        x = self.token_norm(memory.float())
        score = torch.einsum("btd,qd->bqt", x, self.queries.float()) / math.sqrt(self.d_model)
        attn = torch.softmax(score, dim=-1)
        pooled = torch.einsum("bqt,btd->bqd", attn, x)
        raw = torch.einsum("bqd,qd->bq", pooled, self.readouts.float()) / math.sqrt(self.d_model)
        return raw + self.bias.float().unsqueeze(0)

    def forward(self, memory: torch.Tensor, nominal_index: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compose candidate semantics from nominal state + action response.

        ``nominal_index[i]`` points to the unique nominal row of row i's
        scene-time group.  For nominal rows the response difference is exactly
        zero, so candidate-response learning cannot rewrite the base state by
        algebraic shortcut.
        """
        pot = self.semantic_potentials(memory)
        ni = nominal_index.long().to(device=pot.device)
        if ni.ndim != 1 or ni.shape[0] != pot.shape[0]:
            raise ValueError("nominal_index must be [B]")
        if int(ni.min()) < 0 or int(ni.max()) >= pot.shape[0]:
            raise ValueError("nominal_index out of range")
        anchor = pot.index_select(0, ni)
        state_support_logit = anchor[:, 0]
        state_reserve = anchor[:, 1]
        delta_support_logit = pot[:, 2] - anchor[:, 2]
        delta_reserve = pot[:, 3] - anchor[:, 3]
        support_logit = state_support_logit + delta_support_logit
        reserve = state_reserve + delta_reserve
        return {
            "support": torch.sigmoid(support_logit),
            "reserve_debt": reserve,
            "state_support_logit": state_support_logit,
            "state_reserve_debt": state_reserve,
            "delta_support_logit": delta_support_logit,
            "delta_reserve_debt": delta_reserve,
            "potentials": pot,
        }


def expected_parameter_count(d_model: int) -> int:
    d = int(d_model)
    return int(4 * (2 * d + 1))


def build_nominal_index(rows: list[dict]) -> torch.Tensor:
    """Return the unique same-group nominal row for every row, fail closed."""
    groups: dict[tuple[int, str, int], list[int]] = {}
    for i, r in enumerate(rows):
        key = (int(r["bucket"]), str(r["scene"]), int(r["time"]))
        groups.setdefault(key, []).append(i)
    out = torch.empty(len(rows), dtype=torch.long)
    for key, ids in groups.items():
        noms = [i for i in ids if bool(rows[i].get("nominal", False))]
        if len(noms) != 1:
            raise ValueError(f"V48.103 requires exactly one nominal per group: {key} has {len(noms)}")
        n = int(noms[0])
        out[torch.tensor(ids, dtype=torch.long)] = n
    return out


def nominal_response_zero_check(d_model: int = 16) -> bool:
    torch.manual_seed(48103)
    m = FactorizedControlSufficientState(d_model).eval()
    mem = torch.randn(5, 13, d_model)
    # groups: [0,1,2] anchored at 0; [3,4] anchored at 3
    ni = torch.tensor([0, 0, 0, 3, 3])
    with torch.no_grad():
        out = m(mem, ni)
    return bool(
        out["delta_support_logit"][0].item() == 0.0
        and out["delta_reserve_debt"][0].item() == 0.0
        and out["delta_support_logit"][3].item() == 0.0
        and out["delta_reserve_debt"][3].item() == 0.0
    )


def token_permutation_invariance_check(d_model: int = 16) -> bool:
    torch.manual_seed(48104)
    m = FactorizedControlSufficientState(d_model).eval()
    mem = torch.randn(6, 17, d_model)
    ni = torch.tensor([0, 0, 0, 3, 3, 3])
    perm = torch.randperm(mem.shape[1])
    with torch.no_grad():
        a = m(mem, ni)
        b = m(mem[:, perm], ni)
    return bool(
        torch.allclose(a["support"], b["support"], atol=1e-6, rtol=0.0)
        and torch.allclose(a["reserve_debt"], b["reserve_debt"], atol=1e-6, rtol=0.0)
    )


def state_response_parameter_disjoint_check(d_model: int = 16) -> bool:
    """The architecture has fixed separate channels, not a learned mixture."""
    m = FactorizedControlSufficientState(d_model)
    return bool(
        m.queries.shape == (4, d_model)
        and m.readouts.shape == (4, d_model)
        and m.bias.shape == (4,)
        and m.trainable_parameter_count == expected_parameter_count(d_model)
    )
