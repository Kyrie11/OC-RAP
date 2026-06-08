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
    "regime_thresholds": {"tau_high": 1.0, "tau_d": 2.0, "tau_ttc": 3.0, "tau_occ": 0.05},
    "ocmero": {"alpha": 0.2, "beta": 0.2, "top_m": 8},
    "artifact": {"gamma_orc": 0.0, "gamma_dep": 0.0, "delta_neg": 0.0, "min_fraction_train_warning": 0.05, "force_mine": True},
    "selection": {"gamma_rec": 0.0, "fixed_gamma_rec": 0.0, "gamma_H": 0.0, "gamma_D": 5.0},
    "calibration": {"deltas": [0.01, 0.05, 0.10], "numerical_margin": 0.0, "strict": True, "required_min_for_delta": 100},
    "model": {"d_model": 128, "d_z": 128, "d_obs": 64, "d_signature": 32, "d_future_signature": 32, "num_macros": 16, "no_occlusion_bev": False},
    "training": {"device": "auto", "batch_size": 16, "num_workers": 0, "epochs": 1, "lr": 0.001, "weight_decay": 0.0001, "grad_clip": 5.0, "artifact_sampler_weight": 0.25, "root_target_mode": "recovery_signature"},
    "loss_weights": {"assign": 1.0, "sig": 0.5, "ib": 0.01, "obs": 1.0, "margin": 2.0, "anti_oracle": 1.0, "utility": 0.2},
    "evaluation": {"batch_size": 64, "delta": 0.05, "methods": ["nominal", "risk_aware", "backup_filter", "contingency", "oracle_filter", "ocrap"]},
    "baselines": {"risk_lambda": 1.0},
    "metrics": {"sigma_u": 1.0},
    "simulation_backend": "ocrap_surrogate",
    "progress": True,
    "waymax": {
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
