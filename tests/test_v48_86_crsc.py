from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from ocrap.cli.train import (
    _absolute_feasibility_counterfactual_response_interval_huber,
    _absolute_feasibility_counterfactual_selective_response,
    _counterfactual_response_group_losses,
)


def _batch():
    return {
        "r_dep_star": torch.zeros(5),
        "is_nominal": torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0]),
        "bucket_id": torch.tensor([1, 1, 1, 2, 2]),
        "scene_hash": torch.tensor([11, 11, 11, 22, 22]),
        "time_index": torch.tensor([3, 3, 3, 4, 4]),
        "action_response_truth_informative": torch.ones(5),
        "action_response_truth_lower": torch.tensor([0.0, 0.2, -0.6, 0.0, -0.4]),
        "action_response_truth_upper": torch.tensor([0.0, 0.5, -0.2, 0.0, -0.1]),
        "action_response_safe_positive": torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0]),
        "action_response_component_harmful": torch.tensor([0.0, 0.0, 1.0, 0.0, 1.0]),
        "action_response_deployable": torch.tensor([0.0, 1.0, 1.0, 0.0, 1.0]),
    }


def test_v4886_response_interval_is_candidate_minus_nominal_and_has_gradient():
    logits = torch.tensor([0.0, -0.2, 0.4, 0.1, 0.5], requires_grad=True)
    out = {"direct_recovery_absolute_feasibility_logit": logits}
    loss = _absolute_feasibility_counterfactual_response_interval_huber(out, _batch())
    assert torch.isfinite(loss)
    assert float(loss.detach()) > 0.0
    loss.backward()
    assert logits.grad is not None
    assert float(logits.grad.abs().sum()) > 0.0


def test_v4886_selective_pairwise_contract_prefers_safe_positive_and_rejects_harmful():
    batch = _batch()
    bad = torch.tensor([0.0, -0.2, 0.4, 0.1, 0.5])
    good = torch.tensor([0.0, 0.4, -0.4, 0.1, -0.2])
    p0, s0, h0 = _counterfactual_response_group_losses(
        {"direct_recovery_absolute_feasibility_logit": bad}, batch, selective=True
    )
    p1, s1, h1 = _counterfactual_response_group_losses(
        {"direct_recovery_absolute_feasibility_logit": good}, batch, selective=True
    )
    assert float(s1) < float(s0)
    assert float(h1) < float(h0)
    assert torch.isfinite(p0) and torch.isfinite(p1)
    assert float(_absolute_feasibility_counterfactual_selective_response(
        {"direct_recovery_absolute_feasibility_logit": good}, batch
    )) >= float(p1)


def test_v4886_group_contract_requires_exactly_one_nominal():
    batch = _batch()
    batch["is_nominal"] = torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0])
    try:
        _absolute_feasibility_counterfactual_response_interval_huber(
            {"direct_recovery_absolute_feasibility_logit": torch.zeros(5)}, batch
        )
    except ValueError as exc:
        assert "exactly one nominal" in str(exc)
    else:
        raise AssertionError("expected fail-closed nominal-group error")


def test_v4886_truth_builder_uses_interval_difference_and_macro6(tmp_path: Path):
    abs_idx = tmp_path / "abs.jsonl"
    pcd_idx = tmp_path / "pcd.jsonl"
    out = tmp_path / "response.jsonl"
    summary = tmp_path / "summary.json"
    nominal = tmp_path / "nominal.npz"
    cand = tmp_path / "cand.npz"
    nominal.touch(); cand.touch()
    abs_rows = [
        {"valid": True, "sample_path": str(nominal), "dataset_role": "train_contact", "scene_id": "s", "time_index": 7, "candidate_index": 0, "informative": True, "physical_lower": -0.2, "physical_upper": 0.1},
        {"valid": True, "sample_path": str(cand), "dataset_role": "train_contact", "scene_id": "s", "time_index": 7, "candidate_index": 1, "informative": True, "physical_lower": 0.3, "physical_upper": 0.8},
    ]
    pcd_rows = [
        {"path": str(nominal), "bucket": 2, "scene": "s", "time": 7, "candidate": 0, "macro": 0, "nominal": True, "teacher_pcd": 0.20, "component_harmful": False, "beneficial": False},
        {"path": str(cand), "bucket": 2, "scene": "s", "time": 7, "candidate": 1, "macro": 6, "nominal": False, "teacher_pcd": 0.25, "component_harmful": False, "beneficial": True},
    ]
    abs_idx.write_text("".join(json.dumps(r) + "\n" for r in abs_rows))
    pcd_idx.write_text("".join(json.dumps(r) + "\n" for r in pcd_rows))
    tool = Path(__file__).resolve().parents[1] / "tools" / "build_v48_86_action_response_truth_index.py"
    subprocess.run([sys.executable, str(tool), "--absolute-index", str(abs_idx), "--pcd-index", str(pcd_idx), "--output", str(out), "--summary", str(summary)], check=True)
    rows = [json.loads(x) for x in out.read_text().splitlines() if x.strip()]
    c = next(r for r in rows if not r["nominal"])
    assert abs(c["response_lower"] - 0.2) < 1e-9  # 0.3 - 0.1
    assert abs(c["response_upper"] - 1.0) < 1e-9  # 0.8 - (-0.2)
    assert c["deployable"] is True
    assert c["safe_positive"] is True
    sm = json.loads(summary.read_text())
    assert 6 in sm["deployable_macros"]
    assert sm["roles"]["train_contact"]["safe_positive"] == 1
