from __future__ import annotations

import json
from pathlib import Path

import torch

from ocrap.v48_100_joint_root_semantic_decoder import (
    JointRootSemanticDecoder,
    joint_semantic_loss,
    joint_semantic_scales,
    query_gradient_check,
    trainable_contract_check,
    zero_delta_decoder_identity_check,
)


def test_v48_100_parameter_contract():
    assert trainable_contract_check(192, 8)
    m = JointRootSemanticDecoder(base_root_queries=torch.zeros(1, 8, 192), d_model=192)
    assert m.query_parameter_count == 1536
    assert m.chart_parameter_count == 770
    assert m.trainable_parameter_count == 2306


def test_v48_100_zero_query_delta_identity():
    assert zero_delta_decoder_identity_check(32, 5, 4)


def test_v48_100_query_gradient():
    assert query_gradient_check(32, 5, 4)


def test_v48_100_scales_are_positive_and_coordinate_normalized():
    td = torch.tensor([0.0, 1.0, 0.2, 0.7])
    tr = torch.tensor([-2.0, 1.0, 0.5, -0.5])
    ds = torch.tensor([1.0, -0.2, 0.5])
    dr = torch.tensor([3.0, -0.5, 1.0])
    sc = joint_semantic_scales(td, tr, ds, dr)
    assert set(sc) == {"support", "reserve", "delta_support", "delta_reserve"}
    assert all(float(v) > 0 for v in sc.values())


def test_v48_100_loss_zero_on_exact_semantics():
    s = torch.tensor([0.2, 0.8, 0.4])
    r = torch.tensor([-1.0, 0.5, 0.2])
    ci = torch.tensor([1, 2]); ni = torch.tensor([0, 0])
    sc = {"support":1.0,"reserve":1.0,"delta_support":1.0,"delta_reserve":1.0}
    loss, parts = joint_semantic_loss(s, r, s, r, ci, ni, sc)
    assert float(loss) == 0.0
    assert all(float(v) == 0.0 for v in parts.values())


def test_v48_100_engineering_version_present():
    from ocrap.v48_100_joint_root_semantic_decoder import ENGINEERING_VERSION
    assert ENGINEERING_VERSION == "v48.100.0-OC-JRSD"
