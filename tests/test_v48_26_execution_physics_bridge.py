from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import numpy as np

from ocrap.cli.train import _finalize_direct_policy_stats


ROOT = Path(__file__).resolve().parents[1]


def test_certificate_json_safe_handles_path_and_numpy() -> None:
    spec = importlib.util.spec_from_file_location(
        "calibrate_policy_risk_v48", ROOT / "tools" / "calibrate_policy_risk_v48.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    value = {
        "path": Path("/tmp/rule.json"),
        "scalar": np.float32(1.5),
        "array": np.asarray([1, 2]),
    }
    assert module._json_safe(value) == {
        "path": "/tmp/rule.json", "scalar": 1.5, "array": [1, 2]
    }


def _constructor_keywords(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "OCRAPModel"
    ]
    assert calls
    return {kw.arg for kw in calls[-1].keywords if kw.arg}


def test_train_and_inference_forward_same_evidence_contract() -> None:
    required = {
        "direct_recovery_evidence_frontier",
        "direct_recovery_evidence_component_prior_logit",
        "direct_recovery_evidence_admission_bounded",
    }
    train_keys = _constructor_keywords(ROOT / "src" / "ocrap" / "cli" / "train.py")
    inference_keys = _constructor_keywords(ROOT / "src" / "ocrap" / "models" / "inference.py")
    assert required <= train_keys
    assert required <= inference_keys


def test_integrity_metric_uses_safe_positive_contract() -> None:
    stats: dict[str, float] = {}
    for regime in ("near", "contact"):
        stats[f"group_count_{regime}"] = 10.0
        stats[f"positive_count_{regime}"] = 8.0
        stats[f"positive_admission_hit_{regime}"] = 8.0  # misleading raw-positive metric
        stats[f"admission_count_{regime}"] = 2.0
        stats[f"safe_opportunity_count_{regime}"] = 2.0
        stats[f"safe_positive_admission_hit_{regime}"] = 1.0
        stats[f"valid_safe_admission_count_{regime}"] = 1.0
        stats[f"invalid_admission_count_{regime}"] = 1.0
        stats[f"evidence_safe_top1_hit_{regime}"] = 1.0
        stats[f"evidence_safe_top1_regret_sum_{regime}"] = 0.2
        stats[f"soft_safe_nll_sum_{regime}"] = 1.0
        stats[f"soft_safe_group_{regime}"] = 2.0
        stats[f"soft_safe_recall_sum_{regime}"] = 1.0
        stats[f"soft_false_admission_sum_{regime}"] = 0.1
        stats[f"soft_harmful_mass_sum_{regime}"] = 0.1
        stats[f"soft_frontier_harmful_mass_sum_{regime}"] = 0.1
        stats[f"soft_safe_mass_sum_{regime}"] = 1.0
        stats[f"soft_safe_regret_sum_{regime}"] = 0.1
    out = _finalize_direct_policy_stats(stats, {})
    assert out["direct_positive_admission_recall_near"] == 1.0
    assert out["direct_safe_positive_admission_recall_near"] == 0.5
    assert out["direct_safe_admission_precision_near"] == 0.5
    assert out["direct_invalid_admission_rate_near"] == 0.5
    assert out["direct_integrity_recall_min"] == 0.5
    assert out["direct_integrity_precision_min"] == 0.5


def test_safe_utility_uses_exact_runtime_scale() -> None:
    source = (ROOT / "src" / "ocrap" / "models" / "losses.py").read_text()
    assert "torch.sigmoid(admission_delta_logits[deployment_idx]) - 0.5" in source
    assert "safe_set_teacher_delta" in source
    assert ".clamp(-0.5, 0.5)" in source


def test_closed_loop_exports_missing_near_and_contact_physics() -> None:
    source = (ROOT / "src" / "ocrap" / "simulation" / "closed_loop_runner.py").read_text()
    for key in (
        "near_contact_exposure_episode_count",
        "near_contact_longest_exposure_run_s",
        "time_to_min_clearance_s",
        "clearance_recovery_gain_m",
        "time_to_min_ttc_s",
        "ttc_recovery_gain_s",
        "recontact_scene_rate",
        "post_contact_free_space_auc_normalized_m",
        "post_contact_clearance_deficit_auc_m_s",
        "new_stable_stop_quality_scene_rate",
        "time_to_stable_stop_quality_s",
    ):
        assert key in source


def test_v48_26_has_fail_closed_preflight_and_repair_path() -> None:
    controller = (ROOT / "scripts" / "run_v48_26_execution_physics_dedicated.sh").read_text()
    assert "check_v48_26_model_contract.py" in controller
    assert "model_inference_contract" in controller
    assert (ROOT / "scripts" / "repair_v48_25_certificate_with_v48_26.sh").is_file()


def test_standard_calibration_can_enforce_exact_split_ids(monkeypatch) -> None:
    from types import SimpleNamespace
    import ocrap.cli.calibrate as calibration_module

    paths = [Path('/tmp/calibration.npz'), Path('/tmp/certificate.npz')]
    split_by_path = {
        str(paths[0]): 'calibration',
        str(paths[1]): 'certificate_pool',
    }
    monkeypatch.setattr(calibration_module, 'iter_sample_paths_many', lambda _: paths)
    monkeypatch.setattr(
        calibration_module,
        'scalar_metadata_for_path',
        lambda path, key, default='': split_by_path.get(str(path), default),
    )
    monkeypatch.setattr(calibration_module, 'load_npz', lambda _: {'r_dep_star': np.asarray(-1.0)})
    monkeypatch.setattr(calibration_module, 'load_model_bundle', lambda checkpoint, cfg: object())
    monkeypatch.setattr(
        calibration_module,
        'predict_sample',
        lambda data, bundle, cfg: SimpleNamespace(r_dep=-0.1, gap=0.0, q=np.ones((1, 1)), root_probs=np.ones(1)),
    )
    cfg = {
        'calibration': {
            'allowed_split_ids': 'calibration',
            'exact_split_ids': True,
            'allow_validation_fallback': False,
            'required_min_for_delta': 1,
            'deltas': [0.5],
        }
    }
    out = calibration_module.calibrate('/tmp/fake', checkpoint='/tmp/model.pt', cfg=cfg)
    assert out['num_samples'] == 1
    assert out['splits'] == ['calibration']
    assert out['allowed_split_ids'] == ['calibration']
    assert out['exact_split_ids'] is True


def test_near_contact_is_not_misclassified_as_post_contact() -> None:
    from ocrap.simulation.closed_loop_runner import _is_post_contact_bucket_name

    assert not _is_post_contact_bucket_name('near_contact')
    assert not _is_post_contact_bucket_name('test_near_contact')
    assert _is_post_contact_bucket_name('contact')
    assert _is_post_contact_bucket_name('post-contact')
