from __future__ import annotations

import torch

from ocrap.v48_97_executable_recovery_state import (
    ENGINEERING_VERSION,
    ExecutableRecoverySufficientState,
    root_permutation_invariance_check,
    semantic_loss,
)


def test_v48_97_engineering_version():
    assert ENGINEERING_VERSION == "v48.97.1-OC-ERSS-EVALFIX"


def test_v48_97_parameter_count_fixed():
    d = 24
    m = ExecutableRecoverySufficientState(d)
    assert m.trainable_parameter_count == 4 * d + 2


def test_v48_97_root_permutation_invariance():
    assert root_permutation_invariance_check(24)


def test_v48_97_output_semantics_and_shapes():
    torch.manual_seed(1)
    m = ExecutableRecoverySufficientState(16)
    r = torch.randn(5, 7, 16)
    p = torch.softmax(torch.randn(5, 7), dim=-1)
    v = torch.ones(5, 7, dtype=torch.bool)
    out = m(r, p, v)
    assert out["support"].shape == (5,)
    assert out["reserve_debt"].shape == (5,)
    assert torch.all((out["support"] >= 0) & (out["support"] <= 1))


def test_v48_97_semantic_loss_uses_absolute_and_delta_targets():
    torch.manual_seed(2)
    m = ExecutableRecoverySufficientState(8)
    r = torch.randn(4, 3, 8)
    p = torch.softmax(torch.randn(4, 3), dim=-1)
    v = torch.ones(4, 3, dtype=torch.bool)
    out = m(r, p, v)
    td = torch.tensor([0.0, 1.0, 0.3, 0.8])
    tr = torch.tensor([-0.7, 0.4, -0.2, 0.6])
    ci = torch.tensor([1, 3])
    ni = torch.tensor([0, 2])
    loss, parts = semantic_loss(out, td, tr, ci, ni)
    assert torch.isfinite(loss)
    assert set(parts) == {"support", "reserve", "delta_support", "delta_reserve"}
    assert all(torch.isfinite(x) for x in parts.values())


def test_v48_97_backward_nonzero():
    torch.manual_seed(3)
    m = ExecutableRecoverySufficientState(12)
    r = torch.randn(6, 4, 12)
    p = torch.softmax(torch.randn(6, 4), dim=-1)
    v = torch.ones(6, 4, dtype=torch.bool)
    out = m(r, p, v)
    td = torch.tensor([0.0, 1.0, 0.2, 0.9, 0.4, 0.8])
    tr = torch.tensor([-1.0, 0.5, -0.4, 0.7, -0.2, 0.9])
    loss, _ = semantic_loss(out, td, tr, torch.tensor([1, 3, 5]), torch.tensor([0, 2, 4]))
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in m.parameters())


def test_v48_97_candidate_only_v93_join_preserves_nominal_and_eval_contract():
    from tools.run_v48_97_executable_recovery_state import candidate_only_label_join_synthetic_check
    assert candidate_only_label_join_synthetic_check()


def test_v48_97_empty_eval_is_not_scientific_stop():
    from tools.compare_v48_97_erss import _evaluation_errors
    empty_cell = {
        "state": {"rows": 0, "auc": None},
        "support_true": {"positive_rows": 0, "negative_rows": 0, "auc": None},
        "reserve_true": {"positive_rows": 0, "negative_rows": 0, "auc": None},
    }
    obj = {
        "engineering_version": ENGINEERING_VERSION,
        "evaluation_contracts": {role: {"valid": False} for role in ("dev_near", "dev_contact", "certificate_near", "certificate_contact")},
        "cells": {role: dict(empty_cell) for role in ("dev_near", "dev_contact", "certificate_near", "certificate_contact")},
    }
    errors = _evaluation_errors(obj, "balanced")
    assert errors
    assert any("state_empty_or_auc_null" in e for e in errors)
