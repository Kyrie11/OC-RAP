from __future__ import annotations

import copy

import torch
from torch import nn

from ocrap.v48_100_joint_root_semantic_decoder import JointRootSemanticDecoder

ENGINEERING_VERSION = "v48.101.0-OC-RCSA"
ALGORITHM_NAME = "Observation-Consistent Root Cross-Attention Semantic Alignment"
V100_ENGINEERING_VERSION = "v48.100.0-OC-JRSD"


def expected_cross_attention_parameter_count(d_model: int) -> int:
    """Parameter count of PyTorch MHA with shared q/k/v embedding dimension.

    in_proj_weight: 3 d^2
    in_proj_bias:   3 d
    out_proj:       d^2 + d
    """
    d = int(d_model)
    return int(4 * d * d + 4 * d)


def configure_cross_attention_only(model: nn.Module, *, trainable: bool) -> int:
    """Freeze the complete historical model, then optionally open root cross-attention only."""
    for p in model.parameters():
        p.requires_grad_(False)
    if not hasattr(model, "root_cross_attn"):
        raise ValueError("V48.101 requires model.root_cross_attn")
    for p in model.root_cross_attn.parameters():
        p.requires_grad_(bool(trainable))
    return int(sum(p.numel() for p in model.root_cross_attn.parameters() if p.requires_grad))


def freeze_v100_semantic_state(module: JointRootSemanticDecoder) -> None:
    for p in module.parameters():
        p.requires_grad_(False)
    module.eval()


def cross_attention_parameter_contract(model: nn.Module, d_model: int) -> bool:
    n = int(sum(p.numel() for p in model.root_cross_attn.parameters()))
    return bool(n == expected_cross_attention_parameter_count(d_model))


def cross_attention_training_contract(model: nn.Module, module: JointRootSemanticDecoder, d_model: int) -> bool:
    n = configure_cross_attention_only(model, trainable=True)
    freeze_v100_semantic_state(module)
    model_trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    expected_names = {f"root_cross_attn.{name}" for name, _ in model.root_cross_attn.named_parameters()}
    module_trainable = [name for name, p in module.named_parameters() if p.requires_grad]
    return bool(
        n == expected_cross_attention_parameter_count(d_model)
        and model_trainable == expected_names
        and not module_trainable
    )


def _toy_model(d_model: int, num_roots: int, num_heads: int) -> nn.Module:
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.root_queries = nn.Parameter(torch.randn(1, num_roots, d_model) * 0.02)
            self.root_cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=0.1, batch_first=True)
            self.root_self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=0.1, batch_first=True)
            self.root_norm1 = nn.LayerNorm(d_model)
            self.root_norm2 = nn.LayerNorm(d_model)
            self.root_ffn = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Linear(4 * d_model, d_model),
            )
            self.root_norm3 = nn.LayerNorm(d_model)
            self.root_logit_head = nn.Linear(d_model, 1)

    return M()


def initial_attention_identity_check(d_model: int = 32, num_roots: int = 5, num_heads: int = 4) -> bool:
    """Opening requires_grad on the existing cross-attention must not change the initial function."""
    torch.manual_seed(10101)
    model = _toy_model(d_model, num_roots, num_heads).eval()
    module = JointRootSemanticDecoder(base_root_queries=model.root_queries, d_model=d_model).eval()
    memory = torch.randn(4, 7, d_model)
    valid = torch.ones(4, num_roots, dtype=torch.bool)
    with torch.no_grad():
        before = module(model=model, memory=memory, root_valid=valid)
        s0 = before["support"].clone()
        r0 = before["reserve_debt"].clone()
    configure_cross_attention_only(model, trainable=True)
    freeze_v100_semantic_state(module)
    with torch.no_grad():
        after = module(model=model, memory=memory, root_valid=valid)
    return bool(torch.equal(s0, after["support"]) and torch.equal(r0, after["reserve_debt"]))


def cross_attention_gradient_check(d_model: int = 32, num_roots: int = 5, num_heads: int = 4) -> bool:
    """Semantic loss reaches only root cross-attention, not the frozen V48.100 state."""
    torch.manual_seed(10102)
    model = _toy_model(d_model, num_roots, num_heads).eval()
    module = JointRootSemanticDecoder(base_root_queries=model.root_queries, d_model=d_model).eval()
    configure_cross_attention_only(model, trainable=True)
    freeze_v100_semantic_state(module)
    memory = torch.randn(6, 9, d_model)
    valid = torch.ones(6, num_roots, dtype=torch.bool)
    out = module(model=model, memory=memory, root_valid=valid)
    loss = out["support"].square().mean() + out["reserve_debt"].square().mean()
    loss.backward()
    attn_grads = [p.grad for p in model.root_cross_attn.parameters()]
    module_grads = [p.grad for p in module.parameters()]
    return bool(
        all(g is not None and torch.isfinite(g).all() for g in attn_grads)
        and sum(float(g.abs().sum()) for g in attn_grads if g is not None) > 0.0
        and all(g is None for g in module_grads)
    )


def non_attention_frozen_after_step_check(d_model: int = 32, num_roots: int = 5, num_heads: int = 4) -> bool:
    """One optimizer step changes cross-attention while every other historical tensor remains exact."""
    torch.manual_seed(10103)
    model = _toy_model(d_model, num_roots, num_heads).eval()
    module = JointRootSemanticDecoder(base_root_queries=model.root_queries, d_model=d_model).eval()
    configure_cross_attention_only(model, trainable=True)
    freeze_v100_semantic_state(module)
    before_model = {k: v.detach().clone() for k, v in model.state_dict().items()}
    before_module = {k: v.detach().clone() for k, v in module.state_dict().items()}
    opt = torch.optim.AdamW(model.root_cross_attn.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    memory = torch.randn(6, 9, d_model)
    valid = torch.ones(6, num_roots, dtype=torch.bool)
    out = module(model=model, memory=memory, root_valid=valid)
    loss = out["support"].mean() + out["reserve_debt"].mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    after_model = model.state_dict()
    changed_attn = False
    for k, v0 in before_model.items():
        changed = not torch.equal(v0, after_model[k])
        if k.startswith("root_cross_attn."):
            changed_attn = changed_attn or changed
        elif changed:
            return False
    if any(not torch.equal(v, module.state_dict()[k]) for k, v in before_module.items()):
        return False
    return bool(changed_attn)
