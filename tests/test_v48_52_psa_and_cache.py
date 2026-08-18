from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from ocrap.models.losses import (
    _exact_teacher_recovery_success,
    boundary_complete_frontier_calibration_loss,
)


def test_v4852_physical_teacher_success_uses_q_selection_then_mstar_success() -> None:
    # One sample, two roots, two options. q selects option 0 for both roots, but
    # physical m_star says root 0 fails and root 1 succeeds. q>=0 alone would
    # incorrectly mark both roots successful.
    q = torch.tensor([[[0.8, -0.2], [0.4, -0.1]]], dtype=torch.float32)
    m = torch.tensor([[[-0.3, 0.5], [0.2, -0.4]]], dtype=torch.float32)
    p = torch.tensor([[0.7, 0.3]], dtype=torch.float32)
    rv = torch.ones((1, 2), dtype=torch.bool)
    ov = torch.ones((1, 2), dtype=torch.bool)
    physical = _exact_teacher_recovery_success(
        q, m, p, rv, ov, gamma=0.0, semantics="observation_class"
    )
    assert torch.allclose(physical, torch.tensor([0.3]))
    q_hard_proxy = (p * (q.amax(dim=-1) >= 0.0).float()).sum(dim=-1)
    assert torch.allclose(q_hard_proxy, torch.tensor([1.0]))


def _frontier_loss(*, physical: bool, teacher_m_star: torch.Tensor | None) -> torch.Tensor:
    # Two nominal/recovery groups. The teacher q-hard sign is deliberately
    # optimistic for the recovery candidate while m_star makes the selected
    # physical recovery fail, so PSA must alter the sign target.
    pred_r_dep = torch.tensor([0.2, 0.2, 0.1, 0.1], requires_grad=True)
    pred_gap = torch.tensor([0.1, 0.1, 0.2, 0.2], requires_grad=True)
    pred_q = torch.tensor(
        [
            [[0.3, -0.2], [0.2, -0.1]],
            [[0.4, -0.2], [0.3, -0.1]],
            [[0.3, -0.2], [0.2, -0.1]],
            [[0.5, -0.2], [0.4, -0.1]],
        ],
        requires_grad=True,
    )
    teacher_q = pred_q.detach().clone()
    if teacher_m_star is None:
        mstar = None
    else:
        mstar = teacher_m_star
    roots = torch.tensor([[0.6, 0.4]] * 4, dtype=torch.float32)
    rv = torch.ones((4, 2), dtype=torch.bool)
    ov = torch.ones((4, 2), dtype=torch.bool)
    return boundary_complete_frontier_calibration_loss(
        pred_r_dep,
        pred_gap,
        pred_q,
        torch.tensor([0.2, 0.2, 0.1, 0.1]),
        torch.tensor([0.3, 0.3, 0.3, 0.3]),
        teacher_q,
        roots,
        roots,
        rv,
        ov,
        torch.tensor([10, 10, 20, 20]),
        torch.tensor([1, 1, 2, 2]),
        torch.tensor([1.0, 0.0, 1.0, 0.0]),
        positive_gain=0.015,
        teacher_m_star=mstar,
        physical_teacher_sign_alignment=physical,
        option_execution_semantics="observation_class",
    )


def test_v4852_psa_changes_only_teacher_sign_target_and_requires_mstar() -> None:
    # Nominals physically succeed; recovery candidates' q-selected option fails.
    mstar = torch.tensor(
        [
            [[0.2, -0.1], [0.2, -0.1]],
            [[-0.2, 0.4], [-0.2, 0.4]],
            [[0.2, -0.1], [0.2, -0.1]],
            [[-0.3, 0.5], [-0.3, 0.5]],
        ],
        dtype=torch.float32,
    )
    legacy = _frontier_loss(physical=False, teacher_m_star=mstar)
    psa = _frontier_loss(physical=True, teacher_m_star=mstar)
    assert torch.isfinite(legacy) and torch.isfinite(psa)
    assert not torch.allclose(legacy, psa, atol=1e-8, rtol=0.0)

    try:
        _frontier_loss(physical=True, teacher_m_star=None)
    except ValueError as exc:
        assert "requires teacher_m_star" in str(exc)
    else:
        raise AssertionError("PSA must fail closed when teacher_m_star is missing")


def test_v4852_standard_calibration_prediction_cache_is_exact_and_skips_repeat_forward(tmp_path, monkeypatch) -> None:
    import ocrap.cli.calibrate as cal

    a = tmp_path / "a.npz"
    b = tmp_path / "b.npz"
    # Files only need stable identities for this unit test; load_npz is mocked.
    a.write_bytes(b"a")
    b.write_bytes(b"b")
    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"checkpoint-bytes")
    cache = tmp_path / "pred-cache.json"

    splits = {a: "certificate_pool", b: "certificate_pool"}
    data = {
        a: {"r_dep_star": torch.tensor(-0.4).numpy()},
        b: {"r_dep_star": torch.tensor(0.2).numpy()},
    }
    calls = {"predict": 0, "load_bundle": 0}

    monkeypatch.setattr(cal, "iter_sample_paths_many", lambda dataset: [a, b] if dataset == "mix" else [a])
    monkeypatch.setattr(cal, "scalar_metadata_for_path", lambda p, key, default="": splits[p])
    monkeypatch.setattr(cal, "load_npz", lambda p: data[p])

    def fake_bundle(checkpoint, cfg):
        calls["load_bundle"] += 1
        return object()

    def fake_predict(d, bundle, cfg):
        calls["predict"] += 1
        # Deterministic score tied to teacher sign; exact numeric reuse is tested.
        return SimpleNamespace(r_dep=-0.7 if float(d["r_dep_star"]) < 0 else 0.6, gap=0.0)

    monkeypatch.setattr(cal, "load_model_bundle", fake_bundle)
    monkeypatch.setattr(cal, "predict_sample", fake_predict)

    cfg1 = {
        "calibration": {
            "allowed_split_ids": "certificate_pool",
            "exact_split_ids": True,
            "allow_validation_fallback": False,
            "required_min_for_delta": 1,
            "prediction_cache_json": str(cache),
            "deltas": [0.05],
        }
    }
    first = cal.calibrate("mix", str(ckpt), cfg=cfg1)
    assert calls["predict"] == 2
    assert first["prediction_cache"]["misses"] == 2
    assert first["prediction_cache"]["hits"] == 0

    # Change only bookkeeping fields deliberately excluded from the inference
    # signature. The second pass is a subset of the first and must hit cache.
    cfg2 = {
        "calibration": {
            "allowed_split_ids": "certificate_pool",
            "exact_split_ids": True,
            "allow_validation_fallback": False,
            "required_min_for_delta": 99,
            "prediction_cache_json": str(cache),
            "deltas": [0.05],
        }
    }
    second = cal.calibrate("near", str(ckpt), cfg=cfg2)
    assert calls["predict"] == 2, "cache hit must not repeat model forward"
    assert second["prediction_cache"]["hits"] == 1
    assert second["prediction_cache"]["misses"] == 0
    assert second["num_samples"] == 1
    assert second["num_negative"] == 1
    assert first["gamma_rec"] == second["gamma_rec"]


def test_v4852_scripts_keep_single_regime_agnostic_psa_axis() -> None:
    root = Path(__file__).resolve().parents[1]
    arm = (root / "scripts/run_v48_52_dcp_drfc_bcde_psa_arm.sh").read_text(encoding="utf-8")
    launcher = (root / "scripts/run_v48_52_dcp_drfc_bcde_psa_two_gpu.sh").read_text(encoding="utf-8")
    post = (root / "scripts/run_v48_52_postgate_if_authorized.sh").read_text(encoding="utf-8")
    stage = (root / "scripts/adapt_ocrap_v48_47_dsofr_witness_stage.sh").read_text(encoding="utf-8")

    assert "V4851_BOUNDARY_COMPLETE_FRONTIER=true" in arm
    assert "EVIDENCE_NATIVE_BOUNDARY_COMPLETE_ADVANTAGE_PRESERVATION=false" in arm
    assert "EVIDENCE_NATIVE_EXACT_ADVANTAGE_PRESERVATION=false" in arm
    assert "V4852_PHYSICAL_TEACHER_SIGN_ALIGNMENT=true" in arm
    assert "strategy_regime_conditioning':False" in arm
    assert "check_v48_52_reference_reuse.py" in launcher
    assert "SERIAL_VARIANTS_ON_ONE_GPU=0" in launcher
    assert "physical_teacher_sign_alignment') is not True" in post
    assert "RECOVERY_FRONTIER_PHYSICAL_TEACHER_SIGN_ALIGNMENT" in stage
