from pathlib import Path
import importlib.util


def root():
    return Path(__file__).resolve().parents[1]


def test_stable_launcher_keeps_one_shot_explicit_after_terminal_closure_default():
    text = (root() / 'scripts/run_constraint_native_orientation_audit.sh').read_text()
    assert 'OCRAP_CONSTRAINT_AUDIT_MODE:-terminal_internal_closure' in text
    assert 'run_near_nonfloor_one_shot_realization_two_gpu.sh' in text
    assert 'run_near_nonfloor_admission_seed_screen_two_gpu.sh' in text


def test_one_shot_launcher_is_predecessor_pinned_and_provenance_distinct():
    text = (root() / 'scripts/run_near_nonfloor_one_shot_realization_two_gpu.sh').read_text()
    assert 'NONFLOOR_ADMISSION_SEED_SCREEN_NOT_PROMISING' in text
    assert 'OC-RAP-v48.124.10.6-NONFLOOR-ADMISSION-SEED-SCREEN-results.zip' in text
    assert 'PRIVILEGED_NONFLOOR_ONE_SHOT_SEED_REALIZATION=true' in text
    assert 'PRIVILEGED_NONFLOOR_PCD_ORACLE_CEILING=false' in text
    assert 'OC-RAP-v48.124.10.7.1-ONE-SHOT-ACTION-REALIZATION-results.zip' in text
    assert 'create_fixed_main_execution_snapshot.py' in text
    assert 'check_fixed_main_execution_snapshot.py' in text


def test_runner_contract_forces_exact_nominal_except_single_seed_and_two_labels():
    text = (root() / 'src/ocrap/simulation/closed_loop_runner.py').read_text()
    assert 'Exact nominal everywhere except the single seed decision.' in text
    assert 'audit_ids = [0, int(target_cid)]' in text
    assert 'if len(one_shot_labeled) != 2' in text
    assert 'privileged_nonfloor_one_shot_seed_action' in text
    assert 'privileged_nonfloor_one_shot_exact_nominal' in text
    assert 'expected_exactly_one_trigger' not in text  # adjudicator owns final count check


def _load_adjudicator():
    p = root() / 'tools/adjudicate_near_nonfloor_one_shot_realization.py'
    spec = importlib.util.spec_from_file_location('one_shot_adj', p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)
    return mod


def test_effects_distinguish_locally_positive_from_mixed_scene_effects():
    m = _load_adjudicator()
    nominal = {'scenes': [
        {'target_key': 'a', 'overlap_any': 0, 'offroad_any': 0,
         'clearance_deficit_auc_m_s': 1.0, 'critical_ttc_exposure_duration_s': 1.0,
         'ttc_deficit_auc_s2': 1.0, 'min_clearance_m_min': 1.0, 'ttc_s_min': 1.0},
        {'target_key': 'b', 'overlap_any': 0, 'offroad_any': 0,
         'clearance_deficit_auc_m_s': 1.0, 'critical_ttc_exposure_duration_s': 1.0,
         'ttc_deficit_auc_s2': 1.0, 'min_clearance_m_min': 1.0, 'ttc_s_min': 1.0},
    ]}
    method = {'scenes': [
        {'target_key': 'a', 'overlap_any': 0, 'offroad_any': 0,
         'clearance_deficit_auc_m_s': .8, 'critical_ttc_exposure_duration_s': .8,
         'ttc_deficit_auc_s2': .8, 'min_clearance_m_min': 1.1, 'ttc_s_min': 1.1},
        {'target_key': 'b', 'overlap_any': 0, 'offroad_any': 0,
         'clearance_deficit_auc_m_s': 1.0, 'critical_ttc_exposure_duration_s': 1.0,
         'ttc_deficit_auc_s2': 1.0, 'min_clearance_m_min': .9, 'ttc_s_min': .9},
    ]}
    out = m.effects(nominal, method, {'a', 'b'})
    assert out['hard_no_harm']
    assert out['scene_effects']['a']['classification'] == 'locally_positive'
    assert out['scene_effects']['b']['classification'] == 'worsened'


def test_adjudicator_marks_engfix_version_and_diagnostic_only():
    text = (root() / 'tools/adjudicate_near_nonfloor_one_shot_realization.py').read_text()
    assert 'v48.124.10.7.1-ONE-SHOT-ENTRYPOINT-ENGFIX' in text
    assert "'algorithm_modified':False" in text
    assert "'publication_evidence':False" in text
