from pathlib import Path
import importlib.util


def root():
    return Path(__file__).resolve().parents[1]


def _load_module():
    p = root() / 'tools/adjudicate_terminal_internal_closure.py'
    spec = importlib.util.spec_from_file_location('terminal_closure', p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)
    return mod


def test_stable_launcher_defaults_to_offline_terminal_closure():
    text = (root() / 'scripts/run_constraint_native_orientation_audit.sh').read_text()
    assert 'OCRAP_CONSTRAINT_AUDIT_MODE:-terminal_internal_closure' in text
    assert 'run_terminal_internal_closure.sh' in text
    assert 'run_near_nonfloor_one_shot_realization_two_gpu.sh' in text


def test_terminal_closure_is_no_gpu_and_requires_authoritative_one_shot_bundle():
    text = (root() / 'scripts/run_terminal_internal_closure.sh').read_text()
    assert 'OC-RAP-v48.124.10.7.1-ONE-SHOT-ACTION-REALIZATION-results.zip' in text
    assert 'adjudicate_terminal_internal_closure.py' in text
    assert 'GPU' not in text.replace('# No GPU work', '')


def test_terminal_closure_contract_closes_internal_search_but_not_main_freeze():
    text = (root() / 'tools/adjudicate_terminal_internal_closure.py').read_text()
    assert 'ONE_SHOT_ACTION_REALIZATION_MIXED' in text
    assert 'absolute_admission_repair_hypothesis' in text
    assert '"CLOSED" if valid' in text
    assert '"recovery_mechanism_search": "FROZEN" if valid' in text
    assert '"deployed_main_freeze_authorized": False' in text
    assert '"final_three_regime_submission_test_authorized": False' in text
    assert '"additional_deployed_realization_gpu_closure_authorized": False' in text


def test_variant_contract_requires_one_positive_and_one_worsened_and_duplicate_robustness():
    m = _load_module()
    effect = {
        'scene_effects': {
            'a': {'classification': 'locally_positive'},
            'b': {'classification': 'worsened'},
        }
    }
    seed = {'a': {'candidate_index': 1}, 'b': {'candidate_index': 2}}
    row = {'exact_one_shot_contract': True, 'hard_no_harm': True, 'seed_execution': seed, 'effects': effect}
    out = {'variants': {'balanced': row, 'precision': row}}
    assert m._variant_contract_errors(out) == []
