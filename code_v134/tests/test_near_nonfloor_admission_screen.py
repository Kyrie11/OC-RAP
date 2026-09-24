from pathlib import Path
import importlib.util


def root(): return Path(__file__).resolve().parents[1]

def test_stable_launcher_defaults_to_terminal_one_shot_after_nonfloor_stop():
    text=(root()/'scripts/run_constraint_native_orientation_audit.sh').read_text()
    assert 'OCRAP_CONSTRAINT_AUDIT_MODE:-terminal_internal_closure' in text
    assert 'run_near_nonfloor_one_shot_realization_two_gpu.sh' in text
    assert 'run_near_nonfloor_admission_seed_screen_two_gpu.sh' in text

def test_nonfloor_screen_is_small_privileged_and_fail_closed():
    text=(root()/'scripts/run_near_nonfloor_admission_seed_screen_two_gpu.sh').read_text()
    assert 'build_near_nonfloor_seed_cohort.py' in text
    assert 'PRIVILEGED_NONFLOOR_PCD_ORACLE_CEILING=true' in text
    assert 'PRIVILEGED_NONFLOOR_RDEP_FLOOR=0.5' in text
    assert 'PRIVILEGED_NONFLOOR_SEED_PLAN_FILE="$PLAN"' in text
    assert 'NSEED' in text and '-le 8' in text
    assert 'trap on_exit EXIT' in text

def test_runner_nonfloor_oracle_excludes_structural_floor_and_is_default_off():
    text=(root()/'src/ocrap/simulation/closed_loop_runner.py').read_text()
    assert 'cl_cfg.get("privileged_nonfloor_pcd_oracle_ceiling", False)' in text
    assert 'abs(r_dep_i - float(privileged_nonfloor_rdep_floor))' in text
    assert 'and r_dep_i > 0.0' in text
    assert 'privileged_nonfloor_pcd_oracle_best_teacher_positive' in text
    assert 'privileged_nonfloor_pcd_oracle_nominal_no_nonfloor_positive_gain' in text
    assert 'Base intervened before preregistered non-floor seed start' in text

def test_runtime_contract_freezes_deployed_core_not_diagnostic_runner_hash():
    text=(root()/'tools/check_near_nonfloor_admission_screen_contract.py').read_text()
    for rel in ('src/ocrap/planning/selector.py','src/ocrap/evaluation/baselines.py','src/ocrap/planning/prefix_generation.py','src/ocrap/models/ocrap.py'):
        assert rel in text
    assert "'algorithm_modified':False" in text
    assert "'screen_cannot_authorize_v48_125_by_itself':True" in text

def _load(name):
    p=root()/f'tools/{name}.py'; spec=importlib.util.spec_from_file_location(name,p); m=importlib.util.module_from_spec(spec); assert spec.loader; spec.loader.exec_module(m); return m

def _cand(cid,adv,rd,macro='x',admitted=False):
    return {'candidate_index':cid,'macro':macro,'teacher_pcd_advantage_vs_nominal':adv,'teacher_r_dep_star':rd,'teacher_drs':1.0,'teacher_oracle_gap_star':0.0,'pred_r_dep':-1.0,'pred_gap':0.0,'absolute_admitted':admitted}

def test_seed_builder_excludes_exact_half_floor_and_requires_pre_first_trigger():
    m=_load('build_near_nonfloor_seed_cohort')
    result={'scenes':[{'target_key':'s','candidate_quality_audit_records':[
      {'step_index':0,'selected_candidate_index':0,'candidates':[_cand(0,0,0.5),_cand(1,0.4,0.5),_cand(2,0.2,0.3,'pull_over')]},
      {'step_index':5,'selected_candidate_index':3,'candidates':[_cand(0,0,0.5)]},
    ]}]}
    x=m.extract(result); assert len(x['witnesses'])==1
    w=x['witnesses'][0]; assert w['candidate_index']==2 and w['step_index']==0 and w['first_base_trigger_step']==5

def test_screen_adjudicator_explicitly_cannot_authorize_v48125():
    text=(root()/'tools/adjudicate_near_nonfloor_admission_screen.py').read_text()
    assert 'NONFLOOR_ADMISSION_SEED_SCREEN_PROMISING' in text
    assert 'NONFLOOR_ADMISSION_SEED_SCREEN_NOT_PROMISING' in text
    assert 'does not authorize V48.125 by itself' in text
    assert 'statistical_note' in text
