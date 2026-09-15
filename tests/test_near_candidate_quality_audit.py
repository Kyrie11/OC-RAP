from __future__ import annotations
from types import SimpleNamespace
from pathlib import Path
import json
import numpy as np

from ocrap.simulation.closed_loop_runner import _select_audit_candidate_indices


def _samples(n=5):
    return [SimpleNamespace(candidate_index=i, prefix=SimpleNamespace(macro_name=f"m{i}")) for i in range(n)]


def _info(admitted):
    n=len(admitted)
    return {
        "selection": SimpleNamespace(admitted=np.asarray(admitted,dtype=bool)),
        "utility": np.arange(n,dtype=float),
        "pred_r_dep": np.linspace(0.1,0.5,n),
        "pred_gap": np.zeros(n),
        "nominal_deviation": np.zeros(n),
    }


def test_audit_candidate_scope_all_is_exhaustive_and_ordered():
    samples=_samples(5)
    cfg={"closed_loop":{"audit_candidate_scope":"all"}}
    out=_select_audit_candidate_indices(samples,_info([False,True,False,True,True]),samples[3],cfg)
    assert out == [0,1,2,3,4]


def test_audit_candidate_scope_absolute_admitted_keeps_nominal_selected_and_all_admitted():
    samples=_samples(5)
    cfg={"closed_loop":{"audit_candidate_scope":"absolute_admitted_all"}}
    out=_select_audit_candidate_indices(samples,_info([False,True,False,True,True]),samples[3],cfg)
    assert out == [3,0,1,4]


def test_candidate_quality_launcher_is_small_intervention_only_and_two_gpu():
    root=Path(__file__).resolve().parents[1]
    text=(root/"scripts/run_near_candidate_quality_audit_two_gpu.sh").read_text()
    assert "AUDIT_INTERVENTION_ONLY=true" in text
    assert "AUDIT_CANDIDATE_SCOPE=all" in text
    assert "AUDIT_STORE_CANDIDATE_RECORDS=true" in text
    assert "cohort/balanced_keys.json" in text and "cohort/precision_keys.json" in text
    assert 'GPU0' in text and 'GPU1' in text
    assert "MAX_SCENARIOS=0" in text
    assert "NUM_CANDIDATES=24" in text


def test_candidate_quality_runtime_contract_freezes_scientific_selector_sources():
    root=Path(__file__).resolve().parents[1]
    text=(root/"tools/check_candidate_quality_diagnostic_contract.py").read_text()
    assert 'src/ocrap/planning/selector.py' in text
    assert 'src/ocrap/evaluation/baselines.py' in text
    assert 'algorithm_modified' in text and 'False' in text


def test_candidate_quality_adjudicator_branch_order_is_fail_closed():
    root=Path(__file__).resolve().parents[1]
    text=(root/"tools/adjudicate_near_candidate_quality_audit.py").read_text()
    assert "engineering_fix_required_before_scientific_diagnosis" in text
    assert "candidate_library_or_action_realization_bottleneck" in text
    assert "absolute_admission_bottleneck" in text
    assert "relative_evidence_alignment_bottleneck" in text


def _load_audit_tool():
    import importlib.util
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location("candidate_quality_tool", root/"tools/adjudicate_near_candidate_quality_audit.py")
    mod=importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(mod)
    return mod


def test_candidate_quality_summary_localizes_absolute_admission_failure():
    mod=_load_audit_tool()
    result={"scenes":[{
        "target_key":"k","num_decisions":2,"intervention_rate":0.5,
        "audit_intervention_only":True,"audit_candidate_scope":"all","audit_store_candidate_records":True,
        "candidate_quality_audit_records":[{
            "step_index":0,"selected_candidate_index":1,"num_candidates_labeled":3,
            "candidates":[
                {"candidate_index":0,"selected":False,"absolute_admitted":False,"teacher_r_dep_star":0.1,"teacher_pcd":0.1,"pred_direct_value":0.5,"pred_direct_advantage_vs_nominal":0.0},
                {"candidate_index":1,"selected":True,"absolute_admitted":True,"teacher_r_dep_star":0.0,"teacher_pcd":0.0,"pred_direct_value":0.2,"pred_direct_advantage_vs_nominal":-0.3},
                {"candidate_index":2,"selected":False,"absolute_admitted":False,"teacher_r_dep_star":0.4,"teacher_pcd":0.3,"pred_direct_value":0.7,"pred_direct_advantage_vs_nominal":0.2},
            ]
        }]
    }]}
    summary,errors=mod.summarize_variant(result)
    assert not errors
    assert summary["teacher_pcd_better_than_nominal_anywhere"] == 1
    assert summary["teacher_pcd_better_than_nominal_admitted"] == 0
    assert mod.branch_for({"balanced":summary,"precision":summary}).startswith("absolute_admission_bottleneck")


def test_candidate_quality_finite_equal_treats_matching_nan_as_equal():
    mod=_load_audit_tool()
    assert mod.finite_equal(float("nan"), float("nan"))
    assert not mod.finite_equal(float("nan"), 0.0)
    assert mod.finite_equal(float("inf"), float("inf"))
    assert not mod.finite_equal(float("inf"), float("-inf"))


def test_candidate_quality_replay_comparison_ignores_matching_undefined_metrics():
    mod=_load_audit_tool()
    scene={
        "target_key":"k", "num_decisions":1, "macro_counts":{"nominal":1},
        "selection_reason_counts":{"x":1}, "intervention_rate":0.0,
        "closed_loop_bounded_NUP":0.5, "closed_loop_direct_recovery_advantage":0.0,
        "metric_summary":{"clearance_deficit_auc_m_s":0.1, "contact_anchor_step":float("nan")},
    }
    assert mod.compare_behavior({"scenes":[scene]}, {"scenes":[dict(scene)]}) == []


def test_candidate_quality_primary_truth_matches_direct_value_training_target():
    mod=_load_audit_tool()
    result={"scenes":[{
        "target_key":"k","num_decisions":1,"intervention_rate":1.0,
        "audit_intervention_only":True,"audit_candidate_scope":"all","audit_store_candidate_records":True,
        "candidate_quality_audit_records":[{
            "step_index":0,"selected_candidate_index":1,"num_candidates_labeled":2,
            "candidates":[
                {"candidate_index":0,"selected":False,"absolute_admitted":False,"teacher_r_dep_star":0.0,"teacher_pcd":0.6,"pred_direct_advantage_vs_nominal":0.0},
                {"candidate_index":1,"selected":True,"absolute_admitted":True,"teacher_r_dep_star":1.0,"teacher_pcd":0.5,"pred_direct_advantage_vs_nominal":-0.1},
            ]
        }]
    }]}
    summary,errors=mod.summarize_variant(result)
    assert not errors
    # R_dep improved, but the direct relative head is trained on PCD delta, so
    # this is not a positive relative-recovery opportunity.
    assert summary["teacher_r_dep_better_than_nominal_anywhere"] == 1
    assert summary["teacher_pcd_better_than_nominal_anywhere"] == 0
