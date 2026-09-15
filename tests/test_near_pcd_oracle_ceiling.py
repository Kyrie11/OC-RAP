from pathlib import Path


def root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_oracle_ceiling_is_default_diagnostic_but_privileged_path_is_default_off():
    r=root()
    launcher=(r/'scripts/run_constraint_native_orientation_audit.sh').read_text()
    runsh=(r/'scripts/run_ocrap_closed_loop.sh').read_text()
    assert 'OCRAP_CONSTRAINT_AUDIT_MODE:-pcd_oracle_ceiling' in launcher
    assert 'run_near_pcd_oracle_ceiling_two_gpu.sh' in launcher
    assert 'PRIVILEGED_PCD_ORACLE_CEILING="${PRIVILEGED_PCD_ORACLE_CEILING:-false}"' in runsh


def test_oracle_ceiling_has_causal_trigger_and_absolute_admission_guard():
    text=(root()/'src/ocrap/simulation/closed_loop_runner.py').read_text()
    assert 'base_sel_idx = int(sel_idx)' in text
    assert 'if privileged_pcd_oracle_ceiling' in text
    assert 'bool(admitted[pos])' in text
    assert 'best_pcd > nominal_pcd + privileged_pcd_oracle_epsilon' in text
    assert 'privileged_pcd_oracle_nominal_no_positive_admitted_gain' in text
    assert 'privileged_pcd_oracle_best_admitted' in text
    assert 'info["nup"] = float(oracle_nup["bounded_NUP"])' in text


def test_oracle_pipeline_is_small_and_fail_closed():
    text=(root()/'scripts/run_near_pcd_oracle_ceiling_two_gpu.sh').read_text()
    assert 'TARGET_KEYS_FILE="$keys"' in text
    assert 'PARTIAL_WRITE_EVERY_SCENES=11' in text
    assert 'NUM_CANDIDATES=24' in text
    assert 'NUM_RECOVERY_OPTIONS=12' in text
    assert 'PRIVILEGED_PCD_ORACLE_CEILING=true' in text
    assert 'materialize_near_pcd_oracle_reference.py' in text
    assert 'check_near_pcd_oracle_ceiling_contract.py' in text
    assert 'trap on_exit EXIT' in text


def test_oracle_adjudicator_distinguishes_empirical_ceiling_from_deployed_method():
    text=(root()/'tools/adjudicate_near_pcd_oracle_ceiling.py').read_text()
    assert "'publication_evidence':False" in text
    assert 'PCD_ORACLE_CEILING_GO' in text
    assert 'PCD_ORACLE_CEILING_STOP' in text
    assert 'authorize_v48_125_deployable_relative_evidence_realignment_only' in text
    assert 'execution-consistent teacher-PCD preference' in text


def test_oracle_adjudicator_freezes_trigger_rule_not_historical_trigger_count():
    text=(root()/'tools/adjudicate_near_pcd_oracle_ceiling.py').read_text()
    assert 'trigger!=51' not in text
    assert 'no_base_trigger_on_fresh_oracle_replay' in text
    assert "num_candidates_labeled', -1)) != 24" in text
