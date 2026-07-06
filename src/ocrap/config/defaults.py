from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 7,
    "data_source": "synthetic_artifact",
    "num_synthetic_scenarios": 4,
    "womd_patterns": None,
    "max_scenarios": None,
    "sample_rate_hz": 10,
    "history_horizon_s": 1.0,
    "prefix_horizon_s": 1.0,
    "recovery_horizon_s": 4.0,
    "planning_time_stride_s": 0.5,
    "max_times_per_scenario": 2,
    "max_biased_times_per_scenario": 2,
    "max_agents": 32,
    "max_map_polylines": 128,
    "max_polyline_points": 64,
    "max_dynamic_signals": 16,
    "route_points": 80,
    "local_radius_m": 80.0,
    "bev_resolution_m": 2.0,
    "bev_channels": 7,
    "num_candidate_prefixes": 24,
    "num_reactive_futures": 4,
    "num_targeted_futures": 8,
    "num_roots": 8,
    "num_recovery_options": 24,
    "prefix_param_dim": 5,
    "margin_clip": 5.0,
    "root_margin_aggregation": "lcvar",
    "intra_root_lcvar_alpha": 0.2,
    "hidden_emergence_delay_steps": 2,
    "min_unknown_ratio_for_hidden": 0.01,
    "eps_signature": 0.30,
    "epsilon_obs": 1.0,
    "tau_obs": 1.4426950408889634,
    "wheelbase_m": 2.8,
    "speed_limit_default": 13.4,
    "route_width": 3.5,
    "route_dev_max_m": 2.5,
    "default_available_distance_m": 60.0,
    "d_safe0_m": 1.0,
    "safe_time_headway_s": 0.5,
    "comfort_brake_mps2": -3.0,
    "control_delay_s_default": 0.2,
    "delta_v_max_mps": 5.0,
    "yaw_rate_max_rps": 0.6,
    "future_priors": {"replay": 0.25, "reactive": 0.35, "targeted": 0.40},
    "control_limits": {"a_max": 3.0, "a_min": -6.0, "j_max": 6.0, "delta_max": 0.55, "steer_rate_max": 0.5, "v_max": 20.0},
    "margin_scales": {"distance": 2.0, "stop": 5.0, "accel": 1.0, "decel": 1.0, "steer": 0.1, "jerk": 2.0, "steer_rate": 0.1, "route": 1.0, "delta_v": 2.0, "yaw": 0.2, "inactive": 10.0},
    "obs_distance": {"s_c": 2.0, "s_v": 2.0, "s_p": 2.0, "s_yaw": 0.2, "lambda_psi": 0.5, "lambda_v": 0.5, "lambda_type": 2.0, "lambda_occ": 1.0, "lambda_ego": 0.5, "lambda_map": 0.5, "p_unmatch": 5.0},
    "utility_weights": {"progress": 1.0, "comfort": 0.05, "route": 0.5, "logdiv": 0.05, "offroad": 5.0, "wrongway": 5.0},
    "split_ratios": {"train": 0.70, "val": 0.10, "calibration": 0.10, "test": 0.10},
    "split": {"force_id": None},
    "regime_thresholds": {
        "tau_high": 1.0,
        "tau_d": 2.0,
        "tau_ttc": 3.0,
        "tau_occ": 0.05,
        # Normal/NUP examples in WOMD should be safe, non-artifact samples, but
        # they do not need a completely empty unknown corridor. Occlusion remains
        # a separate overlapping regime through tau_occ.
        "tau_normal_occ": 0.75,
        "tau_normal_dep": 0.50,
        "tau_prefix_hard": 0.0,
        "tau_prefix_harm": 0.05,
        "tau_contact": 0.8,
        "include_prefix_collision_in_near": False,
        "include_prefix_contact_in_post": False,
        "use_paper_regime_definitions": True,
        "require_uniform_for_normal": False,
    },
    "ocmero": {"alpha": 0.2, "beta": 0.2, "top_m": 8},
    "artifact": {
        "gamma_orc": 0.0,
        "gamma_dep": 0.0,
        "delta_neg": 0.0,
        "min_fraction_train_warning": 0.05,
        "force_mine": True,
        "mine_probability": 1.0,
        "compatible_margin": 1.2,
        "incompatible_margin": -6.0,
        "enable_branch_intent_margin": False,
        "branch_intent_compatible_margin": 1.0,
        "branch_intent_incompatible_margin": -2.5,
        "gap_margin": 0.5,
        "admission_gamma": 0.0,
    },
    "selection": {
        "gamma_rec": 0.0, "fixed_gamma_rec": 0.0, "gamma_H": 0.0, "gamma_D": 5.0,
        "ocrap_selector": "lcb_constrained", "lcb_beta": 0.10, "nominal_slack": 0.03,
        "nominal_slack_gap_limit": 0.50, "intervention_penalty": 0.03,
        "deviation_penalty": 0.15, "recovery_bonus": 0.02, "fallback_rec_weight": 0.10,
        "fallback_lcb_margin": 0.05, "fallback_gap_margin": 0.25,
        "nominal_fallback_lcb_slack": 0.05,
        "calibrated_shortfall_penalty": 1.0,
        "calibrated_gap_penalty": 0.05,
        "calibrated_admission_bonus": 0.02,
        "nominal_utility_slack": 0.05,
        "safe_nominal_slack": 0.12,
        "intervention_budget_rate": None,
        "intervention_budget_penalty": 0.25,
        "gamma_rec_by_bucket": {},
        "gamma_rec_by_bucket_file": None,
    },
    "calibration": {"deltas": [0.01, 0.05, 0.10], "numerical_margin": 0.0, "strict": True, "required_min_for_delta": 100},
    "model": {"d_model": 128, "d_z": 128, "d_obs": 64, "d_signature": 32, "d_future_signature": 32, "num_macros": 16, "no_occlusion_bev": False, "encoder_type": "mlp", "transformer_layers": 2, "transformer_heads": 4, "dropout": 0.1},
    "training": {"device": "auto", "require_cuda": False, "progress": True, "batch_size": 16, "num_workers": 0, "epochs": 1, "lr": 0.001, "weight_decay": 0.0001, "grad_clip": 5.0, "artifact_sampler_weight": 0.25, "root_target_mode": "recovery_signature", "val_dataset": None, "balanced_obs_loss": True},
    "loss_weights": {"assign": 1.0, "sig": 0.5, "ib": 0.01, "obs": 1.0, "margin": 2.0, "anti_oracle": 1.0, "artifact_gap": 0.5, "admission": 0.2, "utility": 0.2},
    "evaluation": {"batch_size": 64, "delta": 0.05, "allow_infinite_gamma": False, "group_by_dataset": True, "methods": ["nominal", "log_replay", "idm_proxy", "mpc_proxy", "risk_aware", "backup_filter", "contingency", "oracle_filter", "ocrap"]},
    "closed_loop": {
        "max_scenarios": 8,
        "max_rollouts": None,
        "raw_max_scenarios": None,
        "bucket_dataset": None,
        "bucket_split": "",
        "max_bucket_targets": 0,
        "max_targets_per_scene": 1,
        "start_time_index": None,
        "max_steps": 40,
        "replan_interval_steps": 1,
        "method": "ocrap",
        "label_mode": "fast",
        "audit_every_n_steps": 1,
        "audit_max_labels": 0,
        "audit_top_k": 4,
        "audit_max_extra_candidates": 5,
        "num_candidate_prefixes": None,
        "num_recovery_options": None,
        "artifact_mine_probability": 0.0,
        "render": False,
        "save_trace_npz": False,
        "save_partial": True,
        "progress": True,
        "progress_every_steps": 5,
        "allow_infinite_gamma": False,
        "force_teacher_baselines": False,
    },
    "baselines": {"risk_lambda": 1.0, "idm_harm_lambda": 2.0, "idm_hard_lambda": 15.0},
    "metrics": {"sigma_u": 1.0},
    "dataset_quality": {
        "require_artifact_pairs": False,
        "artifact_pair_mode": "tag",
        "max_accepted_prefixes_per_scene_time": 0,
        "min_artifact_prefixes_per_scene_time": 1,
        "max_artifact_prefixes_per_scene_time": 2,
        "min_nonartifact_prefixes_per_scene_time": 1,
        "max_nonartifact_prefixes_per_scene_time": 6,
        "macro_diversity_first": True,
        "balanced_two_pass": True,
        "balanced_rotate_prefix_order": True,
        "balanced_keep_nominal_nonartifact": True,
        "require_nominal_per_scene_time": True,
        "keep_nominal_even_if_quality_fails": True,
        "min_accepted_prefixes_per_scene_time": 2,
        "require_nominal_regimes": [],
        "forbid_nominal_regimes": [],
        "require_any_regimes": [],
        "forbid_any_regimes": [],
        "min_uniform_times_per_scenario": 0,
        # For primary mixed builds, non-mined pass stays natural/nominal while
        # the mined pass may use the branch-specific oracle-artifact label.
        # Without this, mined hidden pairs often remain r_dep >= 0 under benign
        # Waymax metrics and the artifact quota silently becomes non-artifact.
        "artifact_pass_use_margin_override": True,
        # Count an artifact quota only when the final OC-MERO label is actually
        # an oracle artifact, not merely because a hidden pair was generated.
        "artifact_quota_uses_label": True,
        "warn_if_artifact_fraction_above": 0.80,
        "warn_if_scene_count_below": 50,
        "min_obs_negative_fraction_per_sample": 0.0,
        "require_negative_deployable_sample": False,
        "negative_deployable_threshold": 0.0,
    },
    "simulation_backend": "ocrap_surrogate",
    "progress": True,
    "io": {"compress_npz": True, "fsync_npz": True},
    "profiling": {"enabled": False, "log_every_sample": False, "slow_sample_s": 30.0, "slow_scene_time_s": 120.0, "profile_flush_scene_times": 1, "profile_csv_fsync": False},
    "scenario_start_index": 0,
    "scenario_stride": 1,
    "scenario_worker_index": 0,
    "waymax": {
        "append_scenario_index_to_id": True,
        "strict": True,
        "dataloader_include_sdc_paths": True,
        "num_paths": 45,
        "num_points_per_path": 800,
        "max_num_rg_points": 30000,
        "init_history_steps": 11,
        "prefix_dynamics": "invertible_bicycle",
        "debug_prefix_dynamics": "state",
        "control_non_sdc": "log_playback",
        "allow_new_objects_after_warmup": True,
        "enable_augmented_hidden_roots": True,
        "augmented_hidden_from_unknown_only": True,
        "metrics_to_run": ["log_divergence", "overlap", "offroad", "sdc_wrongway", "sdc_off_route", "sdc_progression", "kinematic_infeasibility"],
        "jax_platforms": "cuda,cpu",
        "preallocate_gpu_memory": False,
        "cache_identical_teacher_rollouts": True,
        "cache_env_objects": True,
        "cache_postprefix_rollouts": True,
        "cache_teacher_metric_rollouts": True,
        "use_jit_scan_rollouts": True,
        "compute_future_metrics": True,
        "detect_natural_hidden_emergence": True,
        "teacher_backend": "auto",
        "teacher_metrics_stride": 0,
        "teacher_rollout_top_k_options": 0,
        "teacher_rollout_option_modes": [],
        # In screened-hybrid mode, screened-out options still need the same
        # branch-specific mined-pair label as rolled options. Otherwise top-k
        # rollout can leave incompatible options structurally positive and erase
        # FRA/ODG artifacts.
        "apply_artifact_override_to_screened_options": True,
        # Optional smoke/debug acceleration: when a mined hidden pair is being
        # explicitly labeled by the branch override, skip Waymax recovery rollout
        # for that augmented future and use the override/structural label row.
        # Keep false for strict final runs unless this approximation is reported.
        "skip_waymax_rollout_for_augmented_override": False,
    },
    "ablation": {"without_observation_kernel": False, "without_lower_tail": False, "without_calibration": False, "without_anti_oracle": False, "full_future_roots": False, "no_occlusion_bev": False},
}


def repo_default_config_path() -> Path | None:
    p = Path(__file__).resolve().parents[3] / "configs" / "default.yaml"
    return p if p.exists() else None


def get_default_config() -> dict[str, Any]:
    p = repo_default_config_path()
    if p is not None:
        with p.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        return deep_update(DEFAULT_CONFIG.copy(), loaded)
    return DEFAULT_CONFIG.copy()


def deep_update(base: dict[str, Any], upd: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out
