from __future__ import annotations

import torch

from ocrap.models.ocrap import OCRAPModel


def test_cmri_broadcasts_unique_nominal_logits_with_group_isolation() -> None:
    logits = torch.tensor(
        [
            [3.0, 1.0, -2.0],   # group 10 nominal
            [-1.0, 4.0, 0.5],   # group 10 candidate
            [0.0, -2.0, 5.0],   # group 10 candidate
            [1.5, 0.0, -1.0],   # group 20 nominal
            [-3.0, 2.0, 1.0],   # group 20 candidate
        ]
    )
    groups = torch.tensor([[10], [10], [10], [20], [20]])
    nominal = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0])

    projected = OCRAPModel._common_measure_root_logits(logits, groups, nominal)

    assert torch.equal(projected[0], logits[0])
    assert torch.equal(projected[1], logits[0])
    assert torch.equal(projected[2], logits[0])
    assert torch.equal(projected[3], logits[3])
    assert torch.equal(projected[4], logits[3])


def test_cmri_fails_closed_for_malformed_groups() -> None:
    logits = torch.randn(5, 4)
    groups = torch.tensor([[1], [1], [2], [2], [3]])
    # group 1 has no nominal; group 2 has two nominals; group 3 is singleton.
    nominal = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0])

    projected = OCRAPModel._common_measure_root_logits(logits, groups, nominal)
    assert torch.equal(projected, logits)



def test_cmri_fails_closed_when_root_support_differs_within_group() -> None:
    logits = torch.tensor([[2.0, 0.0, -1.0], [-1.0, 3.0, 0.0]])
    groups = torch.zeros((2, 1), dtype=torch.long)
    nominal = torch.tensor([1.0, 0.0])
    root_valid = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)

    projected = OCRAPModel._common_measure_root_logits(
        logits, groups, nominal, root_valid
    )
    assert torch.equal(projected, logits)

def test_cmri_anchor_is_candidate_set_and_order_invariant() -> None:
    nominal_logits = torch.tensor([[0.2, 1.1, -0.5]])
    candidate_logits = torch.tensor([[3.0, -2.0, 0.0], [-1.0, 0.5, 2.5]])
    logits = torch.cat([nominal_logits, candidate_logits], dim=0)
    groups = torch.zeros((3, 1), dtype=torch.long)
    nominal = torch.tensor([1.0, 0.0, 0.0])
    projected = OCRAPModel._common_measure_root_logits(logits, groups, nominal)

    # Add a new counterfactual and permute all rows.  Every recovery measure must
    # still equal the exact same nominal posterior; no candidate-set average is used.
    extended = torch.cat([candidate_logits[1:], nominal_logits, torch.tensor([[9.0, 0.0, -4.0]]), candidate_logits[:1]], dim=0)
    ext_groups = torch.zeros((4, 1), dtype=torch.long)
    ext_nominal = torch.tensor([0.0, 1.0, 0.0, 0.0])
    projected_ext = OCRAPModel._common_measure_root_logits(extended, ext_groups, ext_nominal)

    assert torch.equal(projected, nominal_logits.expand_as(projected))
    assert torch.equal(projected_ext, nominal_logits.expand_as(projected_ext))


def test_forward_preserves_raw_root_logits_and_exposes_common_recovery_measure() -> None:
    torch.manual_seed(57)
    model = OCRAPModel(
        input_dim=7,
        num_roots=3,
        num_options=2,
        d_model=16,
        d_obs=8,
        num_heads=4,
        dropout=0.0,
        direct_recovery_evidence_common_measure_root_mass=True,
    ).eval()
    x = torch.randn(4, 7)
    groups = torch.zeros((4, 1), dtype=torch.long)
    nominal = torch.tensor([0.0, 1.0, 0.0, 0.0])

    out = model(x, group_index=groups, is_nominal=nominal)
    raw = out["root_logits"]
    recovery = out["recovery_root_logits"]

    assert raw.shape == recovery.shape == (4, 3)
    assert torch.equal(recovery, raw[1:2].expand_as(recovery))
    # CMRI is an aggregation projection, not a root-head rewrite.
    assert not torch.equal(raw, recovery)


def test_disabled_cmri_is_numerically_identical_on_legacy_outputs() -> None:
    torch.manual_seed(5701)
    legacy = OCRAPModel(
        input_dim=5, num_roots=3, num_options=2, d_model=16, d_obs=8,
        num_heads=4, dropout=0.0,
        direct_recovery_evidence_common_measure_root_mass=False,
    ).eval()
    cmri = OCRAPModel(
        input_dim=5, num_roots=3, num_options=2, d_model=16, d_obs=8,
        num_heads=4, dropout=0.0,
        direct_recovery_evidence_common_measure_root_mass=True,
    ).eval()
    cmri.load_state_dict(legacy.state_dict(), strict=True)
    x = torch.randn(3, 5)
    groups = torch.zeros((3, 1), dtype=torch.long)
    nominal = torch.tensor([1.0, 0.0, 0.0])

    with torch.inference_mode():
        a = legacy(x, group_index=groups, is_nominal=nominal)
        b = cmri(x, group_index=groups, is_nominal=nominal)

    assert torch.equal(a["root_logits"], b["root_logits"])
    assert torch.equal(a["margins"], b["margins"])
    assert torch.equal(a["c_star"], b["c_star"])
    assert "recovery_root_logits" not in a
    assert "recovery_root_logits" in b


def test_cmri_is_wired_into_native_recovery_certificate() -> None:
    torch.manual_seed(5757)
    model = OCRAPModel(
        input_dim=5,
        num_roots=3,
        num_options=2,
        d_model=16,
        d_obs=8,
        num_heads=4,
        dropout=0.0,
        direct_recovery_value_head=True,
        direct_recovery_evidence_native_certificate_preservation=True,
        direct_recovery_evidence_common_measure_root_mass=True,
    ).eval()
    x = torch.randn(3, 5)
    groups = torch.zeros((3, 1), dtype=torch.long)
    nominal = torch.tensor([1.0, 0.0, 0.0])
    root_valid = torch.ones((3, 3), dtype=torch.bool)
    option_valid = torch.ones((3, 2), dtype=torch.bool)

    with torch.inference_mode():
        out = model(
            x,
            group_index=groups,
            is_nominal=nominal,
            root_valid=root_valid,
            option_valid=option_valid,
        )
        _signature, expected_native = OCRAPModel._recovery_option_compatibility_signature(
            out["recovery_root_logits"],
            out["obs_embeddings"],
            out["margins"],
            model.tau_obs,
            model.direct_recovery_evidence_roct_alpha,
            model.direct_recovery_evidence_roct_beta,
            model.direct_recovery_evidence_roct_top_m,
            model.direct_recovery_evidence_roct_option_temperature,
            root_valid=root_valid,
            option_valid=option_valid,
            return_native_certificate=True,
            physical_student_drs=model.direct_recovery_evidence_physical_student_drs,
        )

    assert torch.allclose(
        out["direct_recovery_evidence_native_certificate"], expected_native, atol=1e-6, rtol=0.0
    )
    assert torch.equal(
        out["recovery_root_logits"], out["root_logits"][0:1].expand_as(out["root_logits"])
    )
