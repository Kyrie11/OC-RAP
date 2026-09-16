from __future__ import annotations

from pathlib import Path

import torch
import yaml

from ocrap.models.native_recovery_certificate import (
    native_certified_root_option_margins,
    semantic_witness_physical_viability,
)

ROOT = Path(__file__).resolve().parents[1]


def _features() -> tuple[torch.Tensor, torch.Tensor]:
    # [B=1,L=2,F=14].  All generic physical barriers are positive; route,
    # re-entry and historical unprojected-control barriers are deliberately
    # negative so each factor can be isolated causally.
    f = torch.zeros((1, 2, 14), dtype=torch.float32)
    for i in (0, 1, 2, 3, 5, 6, 7, 10):
        f[..., i] = 0.8
    f[..., 4] = -0.4      # historical post-hoc control barrier
    f[..., 8] = 0.1       # clearance floor gain
    f[..., 9] = 0.1       # stability floor gain
    f[..., 11] = 0.0      # stability inactive under active-set alignment
    f[..., 12] = -0.3     # route barrier
    f[..., 13] = -0.2     # persistent re-entry barrier

    of = torch.zeros((1, 2, 12), dtype=torch.float32)
    of[0, 0, 0] = 1.0     # stop
    of[0, 1, 2] = 1.0     # lateral_escape
    return f, of


def test_native_certificate_factors_are_true_pre_ocmero_knockouts() -> None:
    f, of = _features()
    full, _, _ = semantic_witness_physical_viability(
        f, of,
        path_stop_alignment=False,
        active_set_alignment=True,
        control_projection=True,
        route_alignment=True,
        reentry_alignment=True,
    )
    no_route, _, _ = semantic_witness_physical_viability(
        f, of,
        path_stop_alignment=False,
        active_set_alignment=True,
        control_projection=True,
        route_alignment=False,
        reentry_alignment=True,
    )
    no_route_reentry, _, _ = semantic_witness_physical_viability(
        f, of,
        path_stop_alignment=False,
        active_set_alignment=True,
        control_projection=True,
        route_alignment=False,
        reentry_alignment=False,
    )
    no_projection, _, _ = semantic_witness_physical_viability(
        f, of,
        path_stop_alignment=False,
        active_set_alignment=True,
        control_projection=False,
        route_alignment=False,
        reentry_alignment=False,
    )

    assert torch.allclose(full, torch.full_like(full, -0.3))
    assert torch.allclose(no_route, torch.full_like(no_route, -0.2))
    assert torch.allclose(no_route_reentry, torch.full_like(no_route_reentry, 0.8))
    assert torch.allclose(no_projection, torch.full_like(no_projection, -0.4))


def test_native_certificate_is_unit_aligned_and_conservative() -> None:
    viability = torch.tensor([[0.8, -0.3]], dtype=torch.float32)
    raw = torch.tensor([[[2.0, 0.7], [0.2, -1.0]]], dtype=torch.float32)
    option_valid = torch.tensor([[True, False]])
    certified, cert_margin = native_certified_root_option_margins(
        raw, viability, option_valid=option_valid
    )
    # tanh-bounded witness reserve is inverted back to normalized signed-margin
    # coordinates before it is combined with M_{k,l}.
    assert torch.allclose(cert_margin[0, 0], torch.atanh(torch.tensor(0.8)), atol=1e-6)
    assert torch.allclose(certified[0, 0, 0], cert_margin[0, 0], atol=1e-6)
    assert certified[0, 1, 0].item() == raw[0, 1, 0].item()  # already more conservative
    assert torch.equal(certified[..., 1], raw[..., 1])        # invalid option untouched


def test_submission_configs_enable_one_common_native_full_stack() -> None:
    names = {
        "submission_no_obs_consistency.yaml": "without_observation_kernel",
        "submission_mean_tail.yaml": "without_lower_tail",
        "submission_no_actuator_projection.yaml": "disable_control_projection",
        "submission_no_route_alignment.yaml": "disable_route_alignment",
        "submission_no_persistent_reentry.yaml": "disable_reentry_alignment",
        "submission_no_rifa_absolute_admission.yaml": "without_rifa_absolute_admission",
        "submission_no_nominal_abstention.yaml": "without_nominal_abstention",
    }
    base = yaml.safe_load((ROOT / "configs/ablations/submission_native_certified_full.yaml").read_text())
    assert base["ablation"] == {"native_recovery_certification": True}
    for fn, knockout in names.items():
        d = yaml.safe_load((ROOT / "configs/ablations" / fn).read_text())["ablation"]
        assert d["native_recovery_certification"] is True
        assert d[knockout] is True


def test_ablation_launcher_is_two_gpu_dynamic_and_builds_fresh_full_reference() -> None:
    text = (ROOT / "scripts/run_submission_ablations.sh").read_text()
    assert '--run-id "$ABLATION_RUN_ID"' in text
    assert 'worker "$gpu" &' in text
    assert 'if ((${#GPUS[@]} > 2)); then' in text
    assert '_native_full_reference' in text
    assert '--full-run "$NATIVE_FULL_REFERENCE_ROOT"' in text
    assert 'submission_no_obs_consistency.yaml' in text
    assert 'submission_mean_tail.yaml' in text
