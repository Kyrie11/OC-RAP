from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

TOOL = Path(__file__).resolve().parents[1] / 'tools' / 'select_recovery_toy_examples.py'
spec = importlib.util.spec_from_file_location('select_recovery_toy_examples', TOOL)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def test_oracle_gap_proof_direct_sign_flip_and_conflicting_pair():
    # Two observationally indistinguishable roots.  Each hidden root can recover,
    # but they require incompatible options, so no shared deployable option works.
    d = {
        'root_probs': np.array([0.5, 0.5], dtype=np.float32),
        'root_valid': np.array([1, 1], dtype=bool),
        'option_valid': np.array([1, 1], dtype=bool),
        'c_star': np.ones((2, 2), dtype=np.float32),
        'm_star': np.array([[1.0, -2.0], [-2.0, 1.0]], dtype=np.float32),
    }
    proof = mod._oracle_gap_proof(d, beta=1.0, min_root_prob=0.05, min_compat=0.5)
    assert proof is not None
    assert proof['proof_grade'] == 'A'
    assert np.isclose(proof['local_oracle_margin'], 1.0)
    assert proof['local_deployable_margin'] < 0.0
    assert proof['pair']['strong_pair_conflict'] is True
    assert proof['pair']['option_i'] != proof['pair']['option_j']
    assert proof['pair']['pair_shared_margin'] <= 0.0


def test_prefix_state_render_mapping_uses_heading_not_length():
    ps = np.array([1.0, 2.0, 3.0, 4.0, 0.37, 0.1, 5.0, 4.8, 2.1])
    s = mod._prefix_state_to_agent_box(ps)
    assert np.isclose(s[0], 1.0) and np.isclose(s[1], 2.0)
    assert np.isclose(s[3], 3.0) and np.isclose(s[4], 4.0)
    assert np.isclose(s[7], 0.37)
    assert np.isclose(s[10], 4.8)
    assert np.isclose(s[11], 2.1)


def test_progress_selection_retries_after_unsafe_top_candidate(tmp_path: Path):
    def save(name: str, *, feasible: int, hard: float, harm: float, macro: str):
        p = tmp_path / name
        np.savez_compressed(
            p,
            feasible=np.array(feasible),
            hard_violation=np.array(hard),
            harm_proxy=np.array(harm),
            prefix_macro_name=np.array(macro),
        )
        return str(p)

    nominal = save('nom.npz', feasible=1, hard=0.0, harm=0.0, macro='nominal')
    unsafe = save('unsafe.npz', feasible=1, hard=1.0, harm=0.0, macro='aggressive')
    safe = save('safe.npz', feasible=1, hard=0.0, harm=0.0, macro='brake')
    rows = [
        {'_scene':'s','_time':5,'_nom':True,'_r_dep':-0.2,'_r_orc':0.0,'_gap':0.2,'_path':nominal,'_role':'near','_cand':0},
        {'_scene':'s','_time':5,'_nom':False,'_r_dep':0.4,'_r_orc':0.5,'_gap':0.1,'_path':unsafe,'_role':'near','_cand':1},
        {'_scene':'s','_time':5,'_nom':False,'_r_dep':0.2,'_r_orc':0.3,'_gap':0.1,'_path':safe,'_role':'near','_cand':2},
    ]
    cases = mod._progress_cases(rows, 'near_reserve', max_hard=0.0, max_harm=0.05, min_delta=0.05)
    assert len(cases) == 1
    assert cases[0].candidate_index == 2
    assert cases[0].nominal_path == nominal
    assert np.isclose(cases[0].delta_r_dep_vs_nominal, 0.4)


def test_compact_oracle_render_smoke(tmp_path: Path):
    case = mod.Case(
        case_type='oracle_gap', dataset_role='near', path='x', scene_id='s', time_index=0,
        candidate_index=1, macro='brake', r_dep=-1.0, r_orc=1.0, gap=2.0, score=1.0,
        proof_grade='A', anchor_root=0, local_oracle_margin=1.0,
        local_deployable_margin=-1.0, local_gap=2.0,
        conflict_root_i=0, conflict_root_j=1, compatibility=1.0,
        option_i=0, option_j=1, best_shared_option=0,
        pair_oracle_margin=1.0, pair_shared_margin=-1.0, strong_pair_conflict=True,
    )
    d = {
        'c_star': np.ones((2,2), dtype=np.float32),
        'm_star': np.array([[1.0,-1.0],[-1.0,1.0]], dtype=np.float32),
        'root_valid': np.ones(2, dtype=bool),
        'option_valid': np.ones(2, dtype=bool),
        'recovery_modes': np.array(['brake','yield']),
    }
    mod._render_ambiguity_matrix(case, d, tmp_path, dpi=80, full_matrix=False)
    p = tmp_path / 'toy_ambiguity_matrix.png'
    assert p.is_file() and p.stat().st_size > 0
