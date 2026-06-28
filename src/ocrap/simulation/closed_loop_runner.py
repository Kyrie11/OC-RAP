from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ocrap.data.build.builder import build_samples_for_history
from ocrap.data.build.history import construct_history
from ocrap.data.serialization import write_json
from ocrap.data.waymax_loader import iter_waymax_womd_scenarios, raw_scenario_from_waymax_state
from ocrap.evaluation.baselines import select_baseline
from ocrap.evaluation.metrics import deployable_recovery_success, false_recoverability_admission, nominal_utility_preservation
from ocrap.models.inference import load_model_bundle, predict_sample, teacher_prediction_from_sample
from ocrap.simulation.waymax_rollout import _as_np, _bicycle_action, _make_env, _metric_summary, _sdc_index


@dataclass
class ClosedLoopDecision:
    scene_id: str
    step_index: int
    time_index: int
    method: str
    selected_index: int
    selected_macro: str
    selected_candidate_index: int
    selected_utility: float
    selected_teacher_r_dep: float
    selected_teacher_r_orc: float
    selected_odg: float
    selected_artifact: bool
    fra_exec: float
    fra_cand: float
    drs: float
    nup: float
    metrics_after_step: dict[str, float]


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(np.asarray(x).reshape(-1)[0])
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _current_timestep(state: Any) -> int:
    try:
        return int(_as_np(state.timestep).reshape(()).item())
    except Exception:
        return 0


def _scene_done(state: Any) -> bool:
    try:
        return bool(_as_np(state.is_done).reshape(()).item())
    except Exception:
        try:
            return _current_timestep(state) >= int(_as_np(state.log_trajectory.valid).shape[-1]) - 2
        except Exception:
            return False


def _sample_to_dict(sample) -> dict[str, Any]:
    return sample.to_npz_dict()


def _select_prefix(samples: list, bundle, cfg: dict, method: str, gamma: float) -> tuple[int, dict[str, Any]]:
    items = []
    for s in samples:
        d = _sample_to_dict(s)
        pred = predict_sample(d, bundle, cfg)
        teacher = teacher_prediction_from_sample(d, cfg)
        items.append({"sample": s, "data": d, "pred": pred, "teacher": teacher})
    utility = np.asarray([_safe_float(x["data"].get("utility", 0.0)) for x in items], dtype=np.float32)
    pred_r_dep = np.asarray([float(x["pred"].r_dep) for x in items], dtype=np.float32)
    teacher_r_dep = np.asarray([_safe_float(x["data"].get("r_dep_star", 0.0)) for x in items], dtype=np.float32)
    teacher_r_orc = np.asarray([_safe_float(x["data"].get("r_orc_star", 0.0)) for x in items], dtype=np.float32)
    hard = np.asarray([_safe_float(x["data"].get("hard_violation", 0.0)) for x in items], dtype=np.float32)
    harm = np.asarray([_safe_float(x["data"].get("harm_proxy", 0.0)) for x in items], dtype=np.float32)
    feasible = np.asarray([bool(int(_safe_float(x["data"].get("feasible", 1.0), 1.0))) for x in items], dtype=bool)
    sel_cfg = cfg.get("selection", {}) if isinstance(cfg.get("selection", {}), dict) else {}
    selected = select_baseline(
        method,
        utility,
        pred_r_dep,
        teacher_r_dep,
        teacher_r_orc,
        hard,
        harm,
        feasible,
        gamma,
        float(sel_cfg.get("gamma_H", 0.0)),
        float(sel_cfg.get("gamma_D", 5.0)),
        cfg,
    )
    idx = int(selected.selected_index)
    chosen = items[idx]
    q_eval = chosen["pred"].q if method == "ocrap" else chosen["teacher"].q
    selected_options = np.argmax(q_eval, axis=1) if getattr(q_eval, "ndim", 0) == 2 else 0
    d = chosen["data"]
    drs = deployable_recovery_success(d["m_star"], d["root_probs"], selected_options, d.get("root_valid", None))
    nup = nominal_utility_preservation(utility[0] if len(utility) else 0.0, utility[idx], sigma_u=float((cfg.get("metrics", {}) or {}).get("sigma_u", 1.0)))
    info = {
        "items": items,
        "utility": utility,
        "teacher_r_dep": teacher_r_dep,
        "teacher_r_orc": teacher_r_orc,
        "selection": selected,
        "drs": float(drs),
        "nup": float(nup["bounded_NUP"]),
        "fra_cand": false_recoverability_admission(selected.admitted, teacher_r_dep),
    }
    return idx, info


def _rollout_one_scene(raw, scenario_rank: int, checkpoint: str | Path | None, cfg: dict, method: str, gamma: float) -> dict[str, Any]:
    import jax  # type: ignore

    cl_cfg = cfg.get("closed_loop", {}) if isinstance(cfg.get("closed_loop", {}), dict) else {}
    max_steps = int(cl_cfg.get("max_steps", 40))
    replan_interval = max(1, int(cl_cfg.get("replan_interval_steps", 1)))
    start_t = cl_cfg.get("start_time_index", None)
    if start_t is None:
        start_t = int((cfg.get("waymax", {}) or {}).get("init_history_steps", 11)) - 1
    start_t = int(start_t)

    state0 = raw.metadata.get("_waymax_state")
    if state0 is None:
        raise ValueError("Closed-loop runner requires Waymax RawScenario metadata['_waymax_state'].")
    local_cfg = dict(cfg)
    local_cfg["_waymax_init_steps_override"] = start_t + 1
    wx_env, _dyn_name = _make_env(state0, local_cfg, allow_new=bool((cfg.get("waymax", {}) or {}).get("allow_new_objects_after_warmup", True)))
    state = wx_env.reset(state0, rng=jax.random.PRNGKey(int((cfg.get("seed", 7) + scenario_rank) & 0x7FFFFFFF)))
    bundle = load_model_bundle(checkpoint, cfg)
    sdc = _sdc_index(state)
    decisions: list[ClosedLoopDecision] = []
    metric_trace: list[dict[str, float]] = []
    state_xy_trace: list[list[float]] = []

    for step_idx in range(max_steps):
        if _scene_done(state):
            break
        t = _current_timestep(state)
        spliced_raw = raw_scenario_from_waymax_state(
            state,
            f"{raw.scenario_id}__cl{scenario_rank:04d}",
            scenario_rank,
            cfg,
            trajectory_mode="closed_loop_splice",
            splice_until=t,
        )
        hist = construct_history(spliced_raw, t, cfg)
        hist.metadata["_waymax_state"] = state
        hist.metadata["_waymax_branch_from_current"] = True
        hist.metadata["waymax_planning_timestep"] = int(t)
        eval_cfg = dict(cfg)
        quality = dict(eval_cfg.get("dataset_quality", {}) or {})
        quality.update({
            "balanced_two_pass": False,
            "max_accepted_prefixes_per_scene_time": 0,
            "min_artifact_prefixes_per_scene_time": 0,
            "min_nonartifact_prefixes_per_scene_time": 0,
            "min_obs_negative_fraction_per_sample": 0.0,
            "require_negative_deployable_sample": False,
            "require_artifact_pairs": False,
            "artifact_pair_mode": "tag",
        })
        eval_cfg["dataset_quality"] = quality
        samples = build_samples_for_history(hist, "closed_loop", eval_cfg)
        if not samples:
            break
        sel_idx, info = _select_prefix(samples, bundle, cfg, method, gamma)
        selected_sample = samples[sel_idx]
        prefix = selected_sample.prefix
        controls = prefix.prefix_controls if prefix.prefix_controls.size else np.zeros((1, 4), dtype=np.float32)
        metrics_after: dict[str, float] = {}
        for k in range(min(replan_interval, max(1, controls.shape[0]))):
            ctrl = controls[min(k, controls.shape[0] - 1)]
            action = _bicycle_action(int(state.num_objects), sdc, float(ctrl[0]), float(ctrl[1]), float(cfg.get("wheelbase_m", 2.8)))
            state = wx_env.step(state, action)
            metrics_after = _metric_summary(wx_env, state, sdc)
            metric_trace.append(metrics_after)
            try:
                tr = state.sim_trajectory
                tt = _current_timestep(state)
                state_xy_trace.append([float(_as_np(tr.x)[sdc, tt]), float(_as_np(tr.y)[sdc, tt])])
            except Exception:
                pass
            if _scene_done(state):
                break
        teacher_r_dep = info["teacher_r_dep"]
        teacher_r_orc = info["teacher_r_orc"]
        utility = info["utility"]
        decisions.append(
            ClosedLoopDecision(
                scene_id=str(raw.scenario_id),
                step_index=int(step_idx),
                time_index=int(t),
                method=str(method),
                selected_index=int(sel_idx),
                selected_macro=str(prefix.macro_name),
                selected_candidate_index=int(selected_sample.candidate_index),
                selected_utility=float(utility[sel_idx]),
                selected_teacher_r_dep=float(teacher_r_dep[sel_idx]),
                selected_teacher_r_orc=float(teacher_r_orc[sel_idx]),
                selected_odg=float(selected_sample.oracle_gap_star),
                selected_artifact=bool(selected_sample.i_art_star),
                fra_exec=float(teacher_r_dep[sel_idx] < 0.0),
                fra_cand=float(info["fra_cand"]),
                drs=float(info["drs"]),
                nup=float(info["nup"]),
                metrics_after_step=metrics_after,
            )
        )

    decision_dicts = [d.__dict__ for d in decisions]
    metric_names = sorted({k for m in metric_trace for k in m.keys()})
    metric_summary: dict[str, float] = {}
    for name in metric_names:
        vals = [float(m.get(name, 0.0)) for m in metric_trace if np.isfinite(float(m.get(name, 0.0)))]
        if vals:
            metric_summary[f"{name}_mean"] = float(np.mean(vals))
            metric_summary[f"{name}_max"] = float(np.max(vals))
            metric_summary[f"{name}_any"] = float(np.max(vals) > 0.0)
    out = {
        "scene_id": str(raw.scenario_id),
        "num_decisions": int(len(decisions)),
        "num_metric_steps": int(len(metric_trace)),
        "method": method,
        "closed_loop_FRA_exec": float(np.mean([d.fra_exec for d in decisions])) if decisions else 0.0,
        "closed_loop_FRA_cand": float(np.mean([d.fra_cand for d in decisions])) if decisions else 0.0,
        "closed_loop_DRS": float(np.mean([d.drs for d in decisions])) if decisions else 0.0,
        "closed_loop_ODG": float(np.mean([d.selected_odg for d in decisions])) if decisions else 0.0,
        "closed_loop_artifact_selection_rate": float(np.mean([float(d.selected_artifact) for d in decisions])) if decisions else 0.0,
        "closed_loop_bounded_NUP": float(np.mean([d.nup for d in decisions])) if decisions else 0.0,
        "metric_summary": metric_summary,
        "macro_counts": {m: int(sum(d.selected_macro == m for d in decisions)) for m in sorted({d.selected_macro for d in decisions})},
        "decisions": decision_dicts,
    }
    if bool(cl_cfg.get("save_trace_npz", False)):
        out["state_xy_trace"] = state_xy_trace
    return out


def _aggregate_scene_results(scene_results: list[dict[str, Any]], method: str, source: str) -> dict[str, Any]:
    keys = ["closed_loop_FRA_exec", "closed_loop_FRA_cand", "closed_loop_DRS", "closed_loop_ODG", "closed_loop_artifact_selection_rate", "closed_loop_bounded_NUP"]
    agg: dict[str, Any] = {
        "source": source,
        "method": method,
        "num_scenes": int(len(scene_results)),
        "num_decisions": int(sum(int(s.get("num_decisions", 0)) for s in scene_results)),
        "num_metric_steps": int(sum(int(s.get("num_metric_steps", 0)) for s in scene_results)),
    }
    for k in keys:
        vals = [float(s.get(k, 0.0)) for s in scene_results if int(s.get("num_decisions", 0)) > 0]
        agg[k] = float(np.mean(vals)) if vals else 0.0
    metric_names = sorted({mk for s in scene_results for mk in (s.get("metric_summary", {}) or {}).keys()})
    agg["waymax_metrics"] = {}
    for mk in metric_names:
        vals = [float((s.get("metric_summary", {}) or {}).get(mk, 0.0)) for s in scene_results]
        agg["waymax_metrics"][mk] = float(np.mean(vals)) if vals else 0.0
    macro_counts: dict[str, int] = {}
    for s in scene_results:
        for m, c in (s.get("macro_counts", {}) or {}).items():
            macro_counts[m] = macro_counts.get(m, 0) + int(c)
    agg["macro_counts"] = macro_counts
    return agg


def closed_loop_evaluate(dataset_patterns: str, checkpoint: str | Path | None, output: str | Path, cfg: dict) -> dict[str, Any]:
    cl_cfg = cfg.get("closed_loop", {}) if isinstance(cfg.get("closed_loop", {}), dict) else {}
    max_scenes = int(cl_cfg.get("max_scenarios", cfg.get("max_scenarios") or 8))
    method = str(cl_cfg.get("method", "ocrap")).lower()
    gamma = float((cfg.get("selection", {}) or {}).get("gamma_rec", 0.0))
    local = dict(cfg)
    local["data_source"] = "womd"
    local["simulation_backend"] = "waymax_closed_loop"
    local["womd_patterns"] = dataset_patterns
    local["max_scenarios"] = max_scenes
    art = dict(local.get("artifact", {}) or {})
    mine_p = float(cl_cfg.get("artifact_mine_probability", 0.0) or 0.0)
    art["force_mine"] = mine_p > 0.0
    art["mine_probability"] = max(0.0, min(1.0, mine_p))
    local["artifact"] = art
    scene_results = []
    for i, raw in enumerate(iter_waymax_womd_scenarios(dataset_patterns, max_scenarios=max_scenes, parser_cfg=local)):
        scene_results.append(_rollout_one_scene(raw, i, checkpoint, local, method, gamma))
    source = "model" if checkpoint else "teacher_fallback"
    result = _aggregate_scene_results(scene_results, method, source)
    result["scenes"] = scene_results
    result["notes"] = [
        "This is a true Waymax receding-horizon loop: reset once, select an action from current SimulatorState, step the environment, then replan from the updated SimulatorState.",
        "Non-SDC actors use Waymax/default log-playback dynamics unless controlled by the environment configuration.",
    ]
    write_json(result, output)
    return result
