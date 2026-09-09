from types import SimpleNamespace

import numpy as np
import pytest

from ocrap.models.inference import _apply_runtime_mechanism_knockouts
from ocrap.planning.selector import calibrated_constrained_select


def _dummy_model(**overrides):
    base = dict(
        direct_recovery_semantic_witness_active_set_alignment=True,
        direct_recovery_semantic_witness_route_alignment=True,
        direct_recovery_semantic_witness_reentry_alignment=True,
        direct_recovery_semantic_witness_control_projection=True,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_runtime_mechanism_knockouts_update_model_and_feature_cfg():
    model = _dummy_model()
    cfg = {"model": {}, "ablation": {"disable_control_projection": True, "disable_reentry_alignment": True}}
    applied = _apply_runtime_mechanism_knockouts(model, cfg)
    assert applied == ["disable_reentry_alignment", "disable_control_projection"]
    assert model.direct_recovery_semantic_witness_reentry_alignment is False
    assert model.direct_recovery_semantic_witness_control_projection is False
    assert cfg["model"]["direct_recovery_semantic_witness_reentry_alignment"] is False
    assert cfg["model"]["direct_recovery_semantic_witness_control_projection"] is False


def test_runtime_mechanism_knockout_fails_on_noop():
    model = _dummy_model(direct_recovery_semantic_witness_route_alignment=False)
    with pytest.raises(RuntimeError, match="no-op"):
        _apply_runtime_mechanism_knockouts(model, {"model": {}, "ablation": {"disable_route_alignment": True}})


def test_rifa_absolute_admission_ablation_removes_only_set_gate():
    kwargs = dict(
        utility=np.array([1.0, 0.9]),
        r_dep=np.array([-0.5, -0.2]),
        hard=np.array([0.0, 0.0]),
        harm=np.array([0.0, 0.0]),
        feasible=np.array([True, True]),
        gamma_rec=0.0,
        gamma_H=0.0,
        gamma_D=0.0,
        nominal_index=0,
    )
    normal = calibrated_constrained_select(**kwargs)
    ablated = calibrated_constrained_select(**kwargs, ablation_without_absolute_admission=True)
    assert normal.admitted.tolist() == [False, False]
    assert ablated.admitted.tolist() == [True, True]


def test_rifa_ablation_still_respects_hard_harm_feasibility():
    out = calibrated_constrained_select(
        utility=np.array([1.0, 0.9, 0.8]),
        r_dep=np.array([-0.5, -0.2, -0.1]),
        hard=np.array([0.0, 0.2, 0.0]),
        harm=np.array([0.0, 0.0, 0.3]),
        feasible=np.array([True, True, True]),
        gamma_rec=0.0,
        gamma_H=0.0,
        gamma_D=0.0,
        nominal_index=0,
        ablation_without_absolute_admission=True,
    )
    assert out.admitted.tolist() == [True, False, False]
