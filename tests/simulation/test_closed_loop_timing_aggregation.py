from __future__ import annotations

from ocrap.simulation.closed_loop_runner import _aggregate_scene_results


def test_closed_loop_timing_is_summed_and_normalized() -> None:
    scenes = [
        {
            "num_decisions": 2,
            "num_metric_steps": 2,
            "label_mode": "all",
            "timing": {"wall_s": 5.0, "totals_s": {"teacher_labels": 4.0, "policy_selection": 0.2}},
            "metric_summary": {},
            "macro_counts": {},
            "selection_reason_counts": {},
        },
        {
            "num_decisions": 3,
            "num_metric_steps": 3,
            "label_mode": "all",
            "timing": {"wall_s": 7.0, "totals_s": {"teacher_labels": 6.0, "policy_selection": 0.3}},
            "metric_summary": {},
            "macro_counts": {},
            "selection_reason_counts": {},
        },
    ]
    out = _aggregate_scene_results(scenes, "marc_lite", "test")
    assert out["timing"]["scene_wall_sum_s"] == 12.0
    assert out["timing"]["totals_s"]["teacher_labels"] == 10.0
    assert out["timing"]["per_decision_s"]["teacher_labels"] == 2.0


def test_deployed_latency_excludes_teacher_and_metric_bookkeeping() -> None:
    scenes = [
        {
            "num_decisions": 2,
            "num_metric_steps": 2,
            "label_mode": "selected",
            "timing": {
                "wall_s": 12.0,
                "totals_s": {
                    "state_history": 0.2,
                    "candidate_features": 0.4,
                    "policy_selection": 0.1,
                    "teacher_labels": 8.0,
                    "audit_labels": 2.0,
                    "waymax_step_metrics": 0.3,
                },
            },
            "metric_summary": {},
            "macro_counts": {},
            "selection_reason_counts": {},
        }
    ]
    out = _aggregate_scene_results(scenes, "marc_lite", "test")
    assert abs(out["timing"]["per_decision_s"]["deployed_planner"] - 0.35) < 1e-12
    assert abs(out["timing"]["per_decision_s"]["evaluation_overhead"] - 5.15) < 1e-12


def test_post_contact_overlap_rate_uses_contact_interval_denominator() -> None:
    scenes = [
        {
            "num_decisions": 10, "num_metric_steps": 10, "label_mode": "fast",
            "timing": {"wall_s": 1.0, "totals_s": {}},
            "metric_summary": {
                "post_contact_overlap_rate": 1.0,
                "post_contact_overlap_count": 1.0,
                "post_contact_num_intervals": 1.0,
            },
            "macro_counts": {}, "selection_reason_counts": {},
        },
        {
            "num_decisions": 10, "num_metric_steps": 10, "label_mode": "fast",
            "timing": {"wall_s": 1.0, "totals_s": {}},
            "metric_summary": {
                "post_contact_overlap_rate": 0.0,
                "post_contact_overlap_count": 0.0,
                "post_contact_num_intervals": 9.0,
            },
            "macro_counts": {}, "selection_reason_counts": {},
        },
    ]
    out = _aggregate_scene_results(scenes, "postimpact_mpc_lite", "test")
    assert abs(out["waymax_metrics"]["post_contact_overlap_rate"] - 0.1) < 1e-12


def test_steady_state_latency_samples_are_aggregated() -> None:
    scenes = [
        {
            "num_decisions": 4, "num_metric_steps": 4, "label_mode": "fast",
            "timing": {
                "wall_s": 1.0,
                "totals_s": {"state_history": 0.4, "candidate_features": 0.4, "policy_selection": 0.4},
                "steady_state_deployed_planner_samples_s": [0.20, 0.30],
            },
            "metric_summary": {}, "macro_counts": {}, "selection_reason_counts": {},
        },
        {
            "num_decisions": 3, "num_metric_steps": 3, "label_mode": "fast",
            "timing": {
                "wall_s": 1.0,
                "totals_s": {"state_history": 0.3, "candidate_features": 0.3, "policy_selection": 0.3},
                "steady_state_deployed_planner_samples_s": [0.10],
            },
            "metric_summary": {}, "macro_counts": {}, "selection_reason_counts": {},
        },
    ]
    out = _aggregate_scene_results(scenes, "diffusion_planner", "test")
    steady = out["timing"]["steady_state_deployed_planner_s"]
    assert steady["num_samples"] == 3
    assert abs(steady["mean"] - 0.20) < 1e-12
    assert out["timing"]["publication_latency_requires_single_job_per_gpu"] is True
