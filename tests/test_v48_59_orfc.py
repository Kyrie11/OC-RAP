from __future__ import annotations

from pathlib import Path
import torch

from ocrap.models.ocrap import OCRAPModel
from ocrap.cli.train import _absolute_feasibility_bce


def _orfc_model() -> OCRAPModel:
    return OCRAPModel(
        input_dim=5,
        num_roots=3,
        num_options=2,
        d_model=16,
        d_obs=8,
        num_heads=4,
        dropout=0.0,
        direct_recovery_value_head=True,
        direct_recovery_absolute_option_margin_correction=True,
        direct_recovery_evidence_native_certificate_preservation=True,
    )


def _batch_inputs():
    x = torch.randn(4, 5)
    groups = torch.zeros((4, 1), dtype=torch.long)
    nominal = torch.tensor([1.0, 0.0, 0.0, 0.0])
    rv = torch.ones((4, 3), dtype=torch.bool)
    ov = torch.ones((4, 2), dtype=torch.bool)
    return x, groups, nominal, rv, ov


def test_orfc_zero_bias_is_exact_native_rdep_boundary() -> None:
    torch.manual_seed(4859)
    model = _orfc_model().eval()
    x, groups, nominal, rv, ov = _batch_inputs()
    out = model(
        x,
        group_index=groups,
        is_nominal=nominal,
        root_valid=rv,
        option_valid=ov,
    )
    native_p = out["direct_recovery_evidence_native_certificate"][:, 1]
    orfc_p = out["direct_recovery_absolute_feasibility_probability"]
    assert torch.allclose(orfc_p, native_p, atol=2e-6, rtol=0)
    assert torch.equal(model.direct_absolute_option_margin_bias, torch.zeros(2))


def test_orfc_gradient_isolated_to_option_margin_bias() -> None:
    torch.manual_seed(4860)
    model = _orfc_model().train()
    x, groups, nominal, rv, ov = _batch_inputs()
    out = model(
        x,
        group_index=groups,
        is_nominal=nominal,
        root_valid=rv,
        option_valid=ov,
    )
    batch = {
        "r_dep_star": torch.tensor([0.5, 0.5, -1.0, -0.5]),
        "is_nominal": nominal,
        "bucket_id": torch.tensor([1, 1, 2, 2]),
        "time_index": torch.arange(4),
    }
    loss = _absolute_feasibility_bce(out, batch)
    loss.backward()
    assert model.direct_absolute_option_margin_bias.grad is not None
    assert torch.isfinite(model.direct_absolute_option_margin_bias.grad).all()
    leaked = [
        name
        for name, param in model.named_parameters()
        if name != "direct_absolute_option_margin_bias"
        and param.grad is not None
        and torch.any(param.grad != 0)
    ]
    assert leaked == []


def test_orfc_adds_only_one_option_bias_state() -> None:
    torch.manual_seed(4861)
    base = OCRAPModel(
        input_dim=5,
        num_roots=3,
        num_options=2,
        d_model=16,
        d_obs=8,
        num_heads=4,
        dropout=0.0,
        direct_recovery_value_head=True,
        direct_recovery_evidence_native_certificate_preservation=True,
    )
    orfc = _orfc_model()
    shared = {k: v for k, v in base.state_dict().items() if k in orfc.state_dict()}
    missing, unexpected = orfc.load_state_dict(shared, strict=False)
    assert missing == ["direct_absolute_option_margin_bias"]
    assert unexpected == []


def test_orfc_and_afe_are_mutually_exclusive() -> None:
    try:
        OCRAPModel(
            input_dim=5,
            num_roots=3,
            num_options=2,
            d_model=16,
            d_obs=8,
            num_heads=4,
            direct_recovery_value_head=True,
            direct_recovery_absolute_feasibility_head=True,
            direct_recovery_absolute_option_margin_correction=True,
        )
    except ValueError as exc:
        assert "mutually exclusive" in str(exc)
    else:
        raise AssertionError("AFE+ORFC must fail closed")


def test_v4859_train_plumbing_exposes_only_orfc_flag() -> None:
    script = (Path(__file__).resolve().parents[1] / "scripts" / "train_ocrap_v48_trac_sr.sh").read_text()
    assert "ABSOLUTE_OPTION_MARGIN_CORRECTION" in script
    assert "model.direct_recovery_absolute_option_margin_correction" in script


def test_v4859_formal_launcher_is_single_axis_and_reuses_v58_controls() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts" / "run_v48_59_dcp_drfc_bcde_rifa_orfc_two_gpu.sh").read_text()
    assert "V4859_NATIVE_B" in script and "V4859_AFE_C" in script
    assert "OC-RAP-v48.58-PIPELINE_COMPLETE.json" in script
    assert "EVIDENCE_TRAINABLE_PREFIXES_OVERRIDE=direct_absolute_option_margin_bias" in script
    assert "STRICT_INIT_ALLOWED_MISSING_PREFIXES=direct_absolute_option_margin_bias" in script
    assert "ABSOLUTE_FEASIBILITY_HEAD=false ABSOLUTE_OPTION_MARGIN_CORRECTION=true" in script
    assert "MAX_EVIDENCE_CALIBRATOR_PARAMS=24" in script
    assert "PROPOSAL_TOP_K=5" in script
    assert "ABSOLUTE_FEASIBILITY_THRESHOLD=0.5" in script
    assert "check_v48_59_state_isolation.py" in script
    assert "check_v48_59_pipeline_complete.py" in script
    assert "EVIDENCE_CENTER" not in script.upper()
    assert "PRED_ADV_CENTER" not in script.upper()


def test_v4859_changelog_preregisters_stop_without_centering() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "ALGORITHM_CHANGELOG.md").read_text(encoding="utf-8")
    first = text.split("## v48.58.2", 1)[0]
    assert "v48.59" in first
    assert "ORFC" in first
    assert "Centering 仍未获授权" in first
    assert "不做 option-bias threshold/LR grid" in first
    assert "regime-specific" in first
