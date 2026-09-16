from pathlib import Path


def root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_support_localization_is_small_and_policy_preserving():
    r=root(); text=(r/'scripts/run_near_all_state_support_localization_two_gpu.sh').read_text()
    assert 'TARGET_KEYS_FILE="$keys"' in text
    assert 'PARTIAL_WRITE_EVERY_SCENES=11' in text
    assert 'LABEL_MODE=coverage' in text
    assert 'AUDIT_EVERY_N_STEPS=1' in text
    assert 'AUDIT_INTERVENTION_ONLY=false' in text
    assert 'AUDIT_CANDIDATE_SCOPE=all' in text
    assert 'AUDIT_STORE_CANDIDATE_RECORDS=true' in text
    assert 'PRIVILEGED_PCD_ORACLE_CEILING=false' in text
    assert 'trap on_exit EXIT' in text


def test_support_runtime_contract_freezes_scientific_runtime():
    text=(root()/'tools/check_near_all_state_support_contract.py').read_text()
    for rel in (
        'src/ocrap/planning/selector.py',
        'src/ocrap/evaluation/baselines.py',
        'src/ocrap/planning/prefix_generation.py',
        'src/ocrap/simulation/closed_loop_runner.py',
        'scripts/run_ocrap_closed_loop.sh',
    ):
        assert rel in text
    assert "'algorithm_modified':False" in text
    assert "'teacher_labels_are_audit_only_after_action_selection':True" in text


def test_support_adjudicator_has_non_overclaiming_branches():
    text=(root()/'tools/adjudicate_near_all_state_support_localization.py').read_text()
    assert 'BASE_TRIGGER_MISSES_POSITIVE_ADMITTED_PCD_OPPORTUNITIES_ON_FROZEN_INTERVENTION_COHORT' in text
    assert 'ABSOLUTE_ADMISSION_BLOCKS_ALL_NONTRIGGER_POSITIVE_PCD_OPPORTUNITIES_ON_FROZEN_INTERVENTION_COHORT' in text
    assert 'NO_NONTRIGGER_POSITIVE_PCD_SUPPORT_ON_FROZEN_INTERVENTION_COHORT_CANDIDATE_ACTION_OR_PCD_TARGET_SUPPORT_REMAINS_LIMITING' in text
    assert 'absence does not prove global absence outside the cohort' in text
    assert "'publication_evidence':False" in text


def test_stable_launcher_defaults_to_support_localization():
    text=(root()/'scripts/run_constraint_native_orientation_audit.sh').read_text()
    assert 'OCRAP_CONSTRAINT_AUDIT_MODE:-one_shot_action_realization' in text
    assert 'run_near_all_state_support_localization_two_gpu.sh' in text


def _load_tool():
    import importlib.util
    p=root()/'tools/adjudicate_near_all_state_support_localization.py'
    spec=importlib.util.spec_from_file_location('support_localization_tool',p)
    m=importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(m); return m


def _candidate(cid, pcd, *, admitted=False, selected=False):
    return {
        'candidate_index':cid,'teacher_pcd':pcd,'absolute_admitted':admitted,'selected':selected,
        'teacher_r_dep_star':0.0,'pred_direct_advantage_vs_nominal':0.0,
    }


def test_support_summary_detects_missed_positive_admitted_opportunity():
    m=_load_tool()
    # Decision 0: Base stays nominal but candidate 1 is positive and admitted.
    rows0=[_candidate(0,0.5,admitted=True,selected=True),_candidate(1,0.7,admitted=True)] + [_candidate(i,0.4) for i in range(2,24)]
    # Decision 1: Base intervenes; no positive PCD candidate exists.
    rows1=[_candidate(0,0.5,admitted=True),_candidate(1,0.4,admitted=True,selected=True)] + [_candidate(i,0.3) for i in range(2,24)]
    result={'scenes':[{
        'target_key':'k','num_decisions':2,'audit_intervention_only':False,'audit_candidate_scope':'all','audit_store_candidate_records':True,
        'candidate_quality_audit_records':[
            {'step_index':0,'selected_candidate_index':0,'num_candidates_labeled':24,'candidates':rows0},
            {'step_index':1,'selected_candidate_index':1,'num_candidates_labeled':24,'candidates':rows1},
        ],
    }]}
    s,e=m.summarize(result); assert not e
    assert s['by_base_decision']['nominal']['pcd_positive_admitted']==1
    assert m.branch({'balanced':s,'precision':s}).startswith('BASE_TRIGGER_MISSES')
