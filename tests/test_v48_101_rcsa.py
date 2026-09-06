from __future__ import annotations

import torch

from ocrap.v48_100_joint_root_semantic_decoder import JointRootSemanticDecoder
from ocrap.v48_101_root_cross_attention_semantic_alignment import (
    ENGINEERING_VERSION,
    cross_attention_gradient_check,
    cross_attention_training_contract,
    expected_cross_attention_parameter_count,
    initial_attention_identity_check,
    non_attention_frozen_after_step_check,
)


def _toy(d_model=32, num_roots=5, num_heads=4):
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.root_queries = torch.nn.Parameter(torch.randn(1, num_roots, d_model) * 0.02)
            self.root_cross_attn = torch.nn.MultiheadAttention(d_model, num_heads, batch_first=True)
            self.root_self_attn = torch.nn.MultiheadAttention(d_model, num_heads, batch_first=True)
            self.root_norm1 = torch.nn.LayerNorm(d_model)
            self.root_norm2 = torch.nn.LayerNorm(d_model)
            self.root_ffn = torch.nn.Sequential(torch.nn.Linear(d_model,4*d_model),torch.nn.GELU(),torch.nn.Linear(4*d_model,d_model))
            self.root_norm3 = torch.nn.LayerNorm(d_model)
            self.root_logit_head = torch.nn.Linear(d_model,1)
    return M()


def test_v48_101_parameter_count():
    assert expected_cross_attention_parameter_count(192) == 148224


def test_v48_101_initial_identity():
    assert initial_attention_identity_check(32, 5, 4)


def test_v48_101_cross_attention_gradient_only():
    assert cross_attention_gradient_check(32, 5, 4)


def test_v48_101_non_attention_frozen_after_step():
    assert non_attention_frozen_after_step_check(32, 5, 4)


def test_v48_101_trainable_contract():
    model = _toy()
    module = JointRootSemanticDecoder(base_root_queries=model.root_queries, d_model=32)
    assert cross_attention_training_contract(model, module, 32)
    assert not any(p.requires_grad for p in module.parameters())
    assert {n for n,p in model.named_parameters() if p.requires_grad} == {f"root_cross_attn.{n}" for n,_ in model.root_cross_attn.named_parameters()}


def test_v48_101_version():
    assert ENGINEERING_VERSION == "v48.101.0-OC-RCSA"
