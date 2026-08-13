from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ocrap.evaluation.metrics import (
    best_option_indices,
    deployable_recovery_success,
    predicted_option_success,
)
from ocrap.models.data import _persistent_tensor_cache_key
from ocrap.models.losses import (
    observation_class_best_option_loss,
    observation_class_option_success_loss,
)


def test_observation_class_semantics_removes_only_global_option_false_veto() -> None:
    q = np.asarray([[1.0, -1.0], [-1.0, 1.0]], dtype=np.float32)
    m = q.copy()
    p = np.asarray([0.5, 0.5], dtype=np.float32)
    global_opt = best_option_indices(q, p, semantics="global")
    class_opt = best_option_indices(q, p, semantics="observation_class")
    assert isinstance(global_opt, int)
    assert np.asarray(class_opt).tolist() == [0, 1]
    assert deployable_recovery_success(m, p, global_opt) == 0.5
    assert deployable_recovery_success(m, p, class_opt) == 1.0
    assert predicted_option_success(q, p, semantics="global") == 0.5
    assert predicted_option_success(q, p, semantics="observation_class") == 1.0


def test_observation_class_losses_are_finite_and_train_pred_q() -> None:
    pred = torch.tensor([[[0.2, -0.1], [-0.2, 0.1]]], requires_grad=True)
    teacher = torch.tensor([[[1.0, -1.0], [-1.0, 1.0]]])
    probs = torch.tensor([[0.5, 0.5]])
    rv = torch.tensor([[True, True]])
    ov = torch.tensor([[True, True]])
    loss = observation_class_option_success_loss(pred, teacher, probs, rv, ov) + observation_class_best_option_loss(pred, teacher, probs, rv, ov)
    assert torch.isfinite(loss)
    loss.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
    assert float(pred.grad.abs().sum()) > 0.0


def test_persistent_tensor_cache_ignores_non_tensor_model_heads(tmp_path: Path) -> None:
    root = tmp_path / "calibration_near_contact"; samples = root / "samples"
    samples.mkdir(parents=True)
    p = samples / "a.npz"; p.write_bytes(b"placeholder")
    (root / "manifest.csv").write_text("path,split_id\nsamples/a.npz,calibration\n")
    base = {
        "prefix_param_dim": 5,
        "bev_channels": 7,
        "model": {"feature_max_agents": 32, "feature_prefix_flat_dim": 80, "direct_recovery_set_tournament": True},
        "training": {"direct_policy_metric_exact_eligibility": True},
    }
    changed_head = {
        **base,
        "model": {**base["model"], "direct_recovery_set_tournament": False, "direct_recovery_delta_mode": "ordinal_evidence"},
    }
    k1 = _persistent_tensor_cache_key([p], base, num_roots=8, num_options=8, d_signature=4, d_future_signature=4, feature_dim=676)
    k2 = _persistent_tensor_cache_key([p], changed_head, num_roots=8, num_options=8, d_signature=4, d_future_signature=4, feature_dim=676)
    assert k1 == k2
    changed_feature = {**base, "model": {**base["model"], "feature_max_agents": 16}}
    k3 = _persistent_tensor_cache_key([p], changed_feature, num_roots=8, num_options=8, d_signature=4, d_future_signature=4, feature_dim=516)
    assert k3 != k1


def test_v4846_shell_factor_matrix_and_no_new_regime_router() -> None:
    repo = Path(__file__).resolve().parents[1]
    arm = (repo / "scripts/run_v48_46_ocswic_ablation_arm.sh").read_text()
    witness = (repo / "scripts/adapt_ocrap_v48_46_ocswic_witness_stage.sh").read_text()
    launcher = (repo / "scripts/run_v48_46_ocswic_2x2_two_gpu.sh").read_text()
    assert "TRAIN_OPTION_EXECUTION_SEMANTICS=observation_class" in arm
    assert "EVAL_OPTION_EXECUTION_SEMANTICS=observation_class" in arm
    assert "V4846_SEQUENTIAL_WITNESS=1" in arm
    assert 'prefixes="obs_embed_head"' in witness
    assert 'prefixes="margin_head"' in witness
    assert "root_logit_head" not in witness.split('prefixes="margin_head"')[0].split('case "$WITNESS_STAGE" in', 1)[1]
    assert 'for pair in "A B" "C D"' in launcher
    assert 'run_arm "$left" "$GPU0"' in launcher and 'run_arm "$right" "$GPU1"' in launcher
    assert "strategy_regime_conditioning':False" in arm


def test_persistent_tensor_cache_loader_mmap_and_corrupt_fallback(tmp_path: Path) -> None:
    from ocrap.models.data import _load_persistent_tensor_cache_payload

    p = tmp_path / "cache.pt"
    torch.save({"schema": 3, "key": "k", "num_items": 2, "tensors": {"x": torch.arange(6).reshape(2, 3)}}, p)
    payload = _load_persistent_tensor_cache_payload(p)
    assert payload is not None
    assert torch.equal(payload["tensors"]["x"], torch.arange(6).reshape(2, 3))
    p.write_bytes(b"not-a-torch-cache")
    assert _load_persistent_tensor_cache_payload(p) is None


def test_v4846_certificate_sets_explicit_evaluation_semantics() -> None:
    repo = Path(__file__).resolve().parents[1]
    text = (repo / "scripts/calibrate_v48_36_shared_certificate_pool.sh").read_text()
    assert '--set calibration.option_execution_semantics="$OPTION_EXECUTION_SEMANTICS"' in text
    assert '--set evaluation.option_execution_semantics="$OPTION_EXECUTION_SEMANTICS"' in text


def test_v4846_runtime_telemetry_summary(tmp_path: Path) -> None:
    import json
    from tools.summarize_v48_46_runtime_telemetry import summarize

    p = tmp_path / "telemetry.jsonl"
    rows = [
        {"gpu": 0, "gpu_util_pct": 20, "gpu_mem_used_mb": 1000, "gpu_mem_total_mb": 10000, "power_w": 80, "mem_available_kb": 100000},
        {"gpu": 0, "gpu_util_pct": 90, "gpu_mem_used_mb": 4000, "gpu_mem_total_mb": 10000, "power_w": 160, "mem_available_kb": 90000},
    ]
    p.write_text("\n".join(json.dumps(x) for x in rows) + "\n")
    s = summarize(p)
    g = s["gpus"]["0"]
    assert g["samples"] == 2
    assert g["gpu_mem_peak_fraction"] == 0.4
    assert g["gpu_util_lt30_fraction"] == 0.5
    assert s["host"]["mem_available_min_kb"] == 90000
