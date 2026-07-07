from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np

from ocrap.data.build.builder import build_feature_only_samples_for_history, build_labeled_samples_for_candidate_indices, build_samples_for_history
from ocrap.data.build.history import construct_history
from ocrap.data.serialization import write_json
from ocrap.data.waymax_loader import iter_waymax_womd_scenarios, raw_scenario_from_waymax_state
from ocrap.evaluation.baselines import select_baseline
from ocrap.evaluation.metrics import deployable_recovery_success, false_recoverability_admission, nominal_utility_preservation
from ocrap.models.data import iter_sample_paths_many, scalar_metadata_for_path
from ocrap.models.inference import ModelBundle, load_model_bundle, predict_sample, predict_samples, teacher_prediction_from_sample
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
    selection_reason: str
    selected_utility: float
    selected_teacher_r_dep: float | None
    selected_teacher_r_orc: float | None
    selected_pred_r_dep: float | None
    selected_pred_r_orc: float | None
    selected_pred_gap: float | None
    selected_nominal_deviation: float | None
    selected_odg: float | None
    selected_artifact: bool | None
    audit_candidate_count: int | None
    audit_best_candidate_index: int | None
    audit_best_teacher_r_dep: float | None
    audit_best_drs: float | None
    audit_selected_r_dep_regret: float | None
    audit_has_recoverable_candidate: bool | None
    audit_selector_miss: bool | None
    fra_exec: float | None
    fra_cand: float | None
    drs: float | None
    nup: float
    metrics_after_step: dict[str, float]


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(np.asarray(x).reshape(-1)[0])
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _safe_optional_float(x: Any) -> float | None:
    try:
        v = float(np.asarray(x).reshape(-1)[0])
        return v if np.isfinite(v) else None
    except Exception:
        return None


def _prefix_nominal_deviation(samples: list) -> np.ndarray:
    if not samples:
        return np.zeros((0,), dtype=np.float32)
    try:
        ref = np.asarray(samples[0].prefix.prefix_states, dtype=float)[:, :2]
    except Exception:
        return np.zeros((len(samples),), dtype=np.float32)
    vals: list[float] = []
    for sample in samples:
        try:
            xy = np.asarray(sample.prefix.prefix_states, dtype=float)[:, :2]
            T = min(len(ref), len(xy))
            vals.append(0.0 if T <= 0 else float(np.sqrt(np.mean(np.sum((xy[:T] - ref[:T]) ** 2, axis=-1))) / 5.0))
        except Exception:
            vals.append(0.0)
    return np.asarray(vals, dtype=np.float32)




def _candidate_lookup(samples: list) -> dict[int, int]:
    out: dict[int, int] = {}
    for i, sample in enumerate(samples):
        try:
            out[int(sample.candidate_index)] = int(i)
        except Exception:
            continue
    return out


def _select_audit_candidate_indices(samples: list, info: dict[str, Any], selected_sample, cfg: dict) -> list[int]:
    """Choose a small, diagnostic candidate set for online label audit.

    The selected-only audit answers whether the executed action was deployably
    recoverable, but not whether OC-RAP missed a better candidate.  This helper
    adds a few high-value alternatives without falling back to expensive
    all-candidate teacher labeling.
    """
    if not samples:
        return []
    cl_cfg = cfg.get("closed_loop", {}) if isinstance(cfg.get("closed_loop", {}), dict) else {}
    k = max(1, int(cl_cfg.get("audit_top_k", 4) or 4))
    max_extra = max(0, int(cl_cfg.get("audit_max_extra_candidates", k) or k))
    selected_cid = int(getattr(selected_sample, "candidate_index", 0))
    out: list[int] = []

    def add(cid: int | None) -> None:
        if cid is None:
            return
        try:
            c = int(cid)
        except Exception:
            return
        if c not in out:
            out.append(c)

    add(selected_cid)
    add(0)  # nominal anchor
    lookup = _candidate_lookup(samples)
    utility = np.asarray(info.get("utility", []), dtype=float).reshape(-1)
    pred_r = np.asarray(info.get("pred_r_dep", []), dtype=float).reshape(-1)
    pred_gap = np.maximum(0.0, np.asarray(info.get("pred_gap", np.zeros_like(pred_r)), dtype=float).reshape(-1))
    dev = np.maximum(0.0, np.asarray(info.get("nominal_deviation", np.zeros_like(pred_r)), dtype=float).reshape(-1))
    sel_cfg = cfg.get("selection", {}) if isinstance(cfg.get("selection", {}), dict) else {}
    beta = float(sel_cfg.get("lcb_beta", 0.10))
    rec_lcb = pred_r - beta * pred_gap

    def add_ranked(score: np.ndarray, largest: bool = True, count: int = 1) -> None:
        if score.size == 0:
            return
        vals = np.asarray(score, dtype=float)
        vals = np.where(np.isfinite(vals), vals, -np.inf if largest else np.inf)
        order = np.argsort(vals)
        if largest:
            order = order[::-1]
        added = 0
        for idx in order.tolist():
            if idx < 0 or idx >= len(samples):
                continue
            add(int(getattr(samples[idx], "candidate_index", idx)))
            added += 1
            if added >= count:
                break

    add_ranked(rec_lcb, largest=True, count=k)
    add_ranked(pred_r, largest=True, count=max(1, k // 2))
    add_ranked(utility - float(sel_cfg.get("deviation_penalty", 0.15)) * dev, largest=True, count=max(1, k // 2))
    add_ranked(pred_gap, largest=False, count=1)

    # Keep macro diversity among audited alternatives, which is helpful for
    # diagnosing whether failures come from candidate coverage or selection.
    seen_macro = {str(getattr(samples[lookup[c]], "prefix", None).macro_name) for c in out if c in lookup and getattr(samples[lookup[c]], "prefix", None) is not None}
    for sample in samples:
        if len(out) >= 1 + max_extra:
            break
        macro = str(getattr(sample.prefix, "macro_name", ""))
        if macro and macro not in seen_macro:
            add(int(sample.candidate_index))
            seen_macro.add(macro)
    return out[: 1 + max_extra]


def _prediction_q_for_candidate(samples: list, info: dict[str, Any], candidate_index: int, selected_fallback_idx: int) -> np.ndarray | int:
    lookup = _candidate_lookup(samples)
    idx = lookup.get(int(candidate_index), int(selected_fallback_idx))
    try:
        return info["items"][idx]["pred"].q
    except Exception:
        return 0

def _mean_finite(values: list[Any], default: float | None = None) -> float | None:
    vals: list[float] = []
    for x in values:
        if x is None:
            continue
        try:
            v = float(x)
        except Exception:
            continue
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return default
    return float(np.mean(vals))



def _load_json_mapping(path: str | Path) -> dict[str, float]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text())
    except Exception:
        return {}
    if isinstance(raw, dict) and isinstance(raw.get("gamma_rec_by_bucket"), dict):
        raw = raw["gamma_rec_by_bucket"]
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for k, v in raw.items():
        try:
            fv = float(v)
            if np.isfinite(fv):
                out[str(k)] = fv
        except Exception:
            continue
    return out


def _apply_gamma_rec_by_bucket_file(cfg: dict) -> dict:
    sel = cfg.get("selection", {}) if isinstance(cfg.get("selection", {}), dict) else {}
    path = sel.get("gamma_rec_by_bucket_file", sel.get("gamma_rec_by_bucket_path", None))
    if not path:
        return cfg
    mapping = _load_json_mapping(path)
    if not mapping:
        return cfg
    local = dict(cfg)
    new_sel = dict(sel)
    existing = new_sel.get("gamma_rec_by_bucket", {})
    merged = dict(existing) if isinstance(existing, dict) else {}
    merged.update(mapping)
    new_sel["gamma_rec_by_bucket"] = merged
    local["selection"] = new_sel
    return local


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


def _select_prefix(
    samples: list,
    bundle: ModelBundle | None,
    cfg: dict,
    method: str,
    gamma: float,
    *,
    compute_teacher_labels: bool = True,
) -> tuple[int, dict[str, Any]]:
    dicts = [_sample_to_dict(s) for s in samples]
    preds = predict_samples(dicts, bundle, cfg) if bundle is not None else [predict_sample(d, None, cfg) for d in dicts]
    items = []
    for s, d, pred in zip(samples, dicts, preds):
        teacher = teacher_prediction_from_sample(d, cfg) if compute_teacher_labels else None
        items.append({"sample": s, "data": d, "pred": pred, "teacher": teacher})

    utility = np.asarray([_safe_float(x["data"].get("utility", 0.0)) for x in items], dtype=np.float32)
    pred_r_dep = np.asarray([float(x["pred"].r_dep) for x in items], dtype=np.float32)
    pred_r_orc = np.asarray([float(x["pred"].r_orc) for x in items], dtype=np.float32)
    pred_gap = np.asarray([float(x["pred"].gap) for x in items], dtype=np.float32)
    nominal_deviation = _prefix_nominal_deviation(samples)
    if compute_teacher_labels:
        teacher_r_dep = np.asarray([_safe_float(x["data"].get("r_dep_star", 0.0)) for x in items], dtype=np.float32)
        teacher_r_orc = np.asarray([_safe_float(x["data"].get("r_orc_star", 0.0)) for x in items], dtype=np.float32)
    else:
        teacher_r_dep = np.full((len(items),), np.nan, dtype=np.float32)
        teacher_r_orc = np.full((len(items),), np.nan, dtype=np.float32)
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
        pred_r_orc=pred_r_orc,
        pred_gap=pred_gap,
        nominal_deviation=nominal_deviation,
    )
    idx = int(selected.selected_index)
    chosen = items[idx]
    nup = nominal_utility_preservation(utility[0] if len(utility) else 0.0, utility[idx], sigma_u=float((cfg.get("metrics", {}) or {}).get("sigma_u", 1.0)))

    if compute_teacher_labels:
        q_eval = chosen["pred"].q if method == "ocrap" else chosen["teacher"].q
        selected_options = np.argmax(q_eval, axis=1) if getattr(q_eval, "ndim", 0) == 2 else 0
        d = chosen["data"]
        drs = deployable_recovery_success(d["m_star"], d["root_probs"], selected_options, d.get("root_valid", None))
        fra_cand = false_recoverability_admission(selected.admitted, teacher_r_dep)
    else:
        drs = None
        fra_cand = None

    info = {
        "items": items,
        "utility": utility,
        "teacher_r_dep": teacher_r_dep,
        "teacher_r_orc": teacher_r_orc,
        "selection": selected,
        "pred_r_dep": pred_r_dep,
        "pred_r_orc": pred_r_orc,
        "pred_gap": pred_gap,
        "nominal_deviation": nominal_deviation,
        "labels_available": bool(compute_teacher_labels),
        "drs": None if drs is None else float(drs),
        "nup": float(nup["bounded_NUP"]),
        "fra_cand": None if fra_cand is None else float(fra_cand),
    }
    return idx, info


def _strip_version_suffix(name: str) -> str:
    base, sep, version = str(name).rpartition("_v")
    return base if sep and version.isdigit() and base else str(name)


def _bucket_gamma_aliases(name: str | None) -> list[str]:
    if not name:
        return []
    raw = str(name)
    aliases = [raw]
    for p in ("test_", "val_", "train_"):
        if raw.startswith(p):
            aliases.append(raw[len(p):])
    aliases.extend([_strip_version_suffix(x) for x in list(aliases)])
    out: list[str] = []
    for x in aliases:
        if x and x not in out:
            out.append(x)
    return out


def _gamma_for_bucket(base_gamma: float, cfg: dict, bucket_name: str | None) -> float:
    """Return a regime/bucket-specific calibrated recovery threshold when set."""
    sel_cfg = cfg.get("selection", {}) if isinstance(cfg.get("selection", {}), dict) else {}
    mapping = sel_cfg.get("gamma_rec_by_bucket", {})
    if not isinstance(mapping, dict) or not bucket_name:
        return float(base_gamma)
    for key in _bucket_gamma_aliases(bucket_name):
        if key in mapping and mapping[key] not in {None, ""}:
            try:
                val = float(mapping[key])
                if np.isfinite(val):
                    return val
            except Exception:
                continue
    return float(base_gamma)

def _rollout_one_scene(
    raw,
    scenario_rank: int,
    bundle: ModelBundle | None,
    cfg: dict,
    method: str,
    gamma: float,
    *,
    start_time_index_override: int | None = None,
    bucket_name: str | None = None,
    target_key: str | None = None,
) -> dict[str, Any]:
    import jax  # type: ignore

    cl_cfg = cfg.get("closed_loop", {}) if isinstance(cfg.get("closed_loop", {}), dict) else {}
    max_steps = int(cl_cfg.get("max_steps", 40))
    replan_interval = max(1, int(cl_cfg.get("replan_interval_steps", 1)))
    start_t = start_time_index_override if start_time_index_override is not None else cl_cfg.get("start_time_index", None)
    if start_t is None:
        start_t = int((cfg.get("waymax", {}) or {}).get("init_history_steps", 11)) - 1
    start_t = int(start_t)
    requested_label_mode = str(cl_cfg.get("label_mode", "fast")).lower()
    label_mode = requested_label_mode
    teacher_required_methods = {"ocrap_teacher"}
    branchwise_methods = {"backup_filter", "oracle_filter", "contingency"}
    force_teacher_baselines = bool(cl_cfg.get("force_teacher_baselines", False))
    if method in teacher_required_methods or (method == "ocrap" and bundle is None) or (force_teacher_baselines and method in branchwise_methods):
        # Full teacher labels are very expensive online. Branch-wise/oracle
        # baselines therefore run as predicted-oracle proxies in fast mode by
        # default, and use true teacher labels only when explicitly requested.
        label_mode = "all"
    selected_label_audit = label_mode in {"selected", "selected_only", "audit_selected", "executed"}
    coverage_label_audit = label_mode in {"selected_topk", "topk", "coverage", "audit_topk", "selected_coverage"}
    compute_teacher_labels = label_mode in {"all", "full", "teacher", "labels"}
    audit_every_n_steps = max(1, int(cl_cfg.get("audit_every_n_steps", 1) or 1))
    audit_max_labels = int(cl_cfg.get("audit_max_labels", 0) or 0)
    audit_labels_done = 0
    progress = bool(cl_cfg.get("progress", True))
    progress_every = max(1, int(cl_cfg.get("progress_every_steps", 5)))

    state0 = raw.metadata.get("_waymax_state")
    if state0 is None:
        raise ValueError("Closed-loop runner requires Waymax RawScenario metadata['_waymax_state'].")
    local_cfg = dict(cfg)
    local_cfg["_waymax_init_steps_override"] = start_t + 1
    wx_env, _dyn_name = _make_env(state0, local_cfg, allow_new=bool((cfg.get("waymax", {}) or {}).get("allow_new_objects_after_warmup", True)))
    state = wx_env.reset(state0, rng=jax.random.PRNGKey(int((cfg.get("seed", 7) + scenario_rank) & 0x7FFFFFFF)))
    sdc = _sdc_index(state)
    decisions: list[ClosedLoopDecision] = []
    metric_trace: list[dict[str, float]] = []
    state_xy_trace: list[list[float]] = []
    interventions_used = 0

    for step_idx in range(max_steps):
        if _scene_done(state):
            break
        t = _current_timestep(state)
        if progress and (step_idx == 0 or step_idx % progress_every == 0):
            print({"event": "closed_loop_step", "scene_rank": scenario_rank, "scene_id": str(raw.scenario_id), "step": step_idx, "time_index": int(t), "label_mode": label_mode}, flush=True)
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
        if cl_cfg.get("num_candidate_prefixes", None) is not None:
            eval_cfg["num_candidate_prefixes"] = int(cl_cfg["num_candidate_prefixes"])
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
        if compute_teacher_labels:
            samples = build_samples_for_history(hist, "closed_loop", eval_cfg)
        else:
            cl_num_options = cl_cfg.get("num_recovery_options", None)
            feature_num_options = int(cl_num_options) if cl_num_options is not None else int(eval_cfg.get("num_recovery_options", getattr(bundle.model, "num_options", 24) if bundle is not None else 24))
            samples = build_feature_only_samples_for_history(
                hist,
                "closed_loop",
                eval_cfg,
                num_roots=int(getattr(bundle.model, "num_roots", int(cfg.get("num_roots", 8)))) if bundle is not None else int(cfg.get("num_roots", 8)),
                num_options=feature_num_options,
            )
        if not samples:
            break
        select_cfg = cfg
        if method == "ocrap":
            # Give the selector the active regime/bucket and the running
            # intervention count.  This is intentionally not written back into
            # the top-level config so each scene rollout remains independent.
            select_cfg = dict(cfg)
            sel_local = dict(select_cfg.get("selection", {}) or {}) if isinstance(select_cfg.get("selection", {}), dict) else {}
            sel_local["active_bucket_name"] = bucket_name or ""
            sel_local["intervention_budget_used"] = int(interventions_used)
            # Number of decisions considered before the current selection.
            # Use step_idx + 1 so the early rollout does not look artificially
            # over-budget after a single intervention.
            sel_local["intervention_budget_steps"] = max(1, int(step_idx) + 1)
            select_cfg["selection"] = sel_local
        sel_idx, info = _select_prefix(samples, bundle, select_cfg, method, gamma, compute_teacher_labels=compute_teacher_labels)
        selected_sample = samples[sel_idx]
        try:
            if int(getattr(selected_sample, "candidate_index", sel_idx)) != 0:
                interventions_used += 1
        except Exception:
            if int(sel_idx) != 0:
                interventions_used += 1
        prefix = selected_sample.prefix
        selected_audit_sample = None
        selected_audit_data = None
        selected_audit_drs = None
        selected_audit_fra_exec = None
        audit_candidate_count = None
        audit_best_candidate_index = None
        audit_best_teacher_r_dep = None
        audit_best_drs = None
        audit_selected_r_dep_regret = None
        audit_has_recoverable_candidate = None
        audit_selector_miss = None
        if (selected_label_audit or coverage_label_audit) and (step_idx % audit_every_n_steps == 0) and (audit_max_labels <= 0 or audit_labels_done < audit_max_labels):
            try:
                cl_num_options = cl_cfg.get("num_recovery_options", None)
                feature_num_options = int(cl_num_options) if cl_num_options is not None else int(eval_cfg.get("num_recovery_options", getattr(bundle.model, "num_options", 24) if bundle is not None else 24))
                audit_indices = ([int(selected_sample.candidate_index)] if selected_label_audit else _select_audit_candidate_indices(samples, info, selected_sample, cfg))
                labeled = build_labeled_samples_for_candidate_indices(
                    hist,
                    "closed_loop",
                    eval_cfg,
                    audit_indices,
                    num_roots=int(getattr(bundle.model, "num_roots", int(cfg.get("num_roots", 8)))) if bundle is not None else int(cfg.get("num_roots", 8)),
                    num_options=feature_num_options,
                )
                if labeled:
                    by_cid = {int(s.candidate_index): s for s in labeled}
                    selected_audit_sample = by_cid.get(int(selected_sample.candidate_index), labeled[0])
                    selected_audit_data = selected_audit_sample.to_npz_dict()
                    pred_q = info["items"][sel_idx]["pred"].q
                    selected_options = np.argmax(pred_q, axis=1) if getattr(pred_q, "ndim", 0) == 2 else 0
                    selected_audit_drs = deployable_recovery_success(
                        selected_audit_data["m_star"],
                        selected_audit_data["root_probs"],
                        selected_options,
                        selected_audit_data.get("root_valid", None),
                    )
                    selected_r_dep_star = _safe_float(selected_audit_data.get("r_dep_star", 0.0))
                    selected_audit_fra_exec = float(selected_r_dep_star < 0.0)
                    audit_candidate_count = int(len(labeled))
                    # Coverage audit: among a small top-k subset, determine
                    # whether any alternative had better deployable headroom.
                    best_r = -float("inf")
                    best_drs = None
                    best_cid = None
                    for lab in labeled:
                        ld = lab.to_npz_dict()
                        cid = int(lab.candidate_index)
                        r_star = _safe_float(ld.get("r_dep_star", -float("inf")), -float("inf"))
                        pred_q_i = _prediction_q_for_candidate(samples, info, cid, sel_idx)
                        opt_i = np.argmax(pred_q_i, axis=1) if getattr(pred_q_i, "ndim", 0) == 2 else 0
                        drs_i = deployable_recovery_success(ld["m_star"], ld["root_probs"], opt_i, ld.get("root_valid", None))
                        if r_star > best_r:
                            best_r = float(r_star)
                            best_drs = float(drs_i)
                            best_cid = int(cid)
                    if best_cid is not None and np.isfinite(best_r):
                        audit_best_candidate_index = int(best_cid)
                        audit_best_teacher_r_dep = float(best_r)
                        audit_best_drs = None if best_drs is None else float(best_drs)
                        audit_selected_r_dep_regret = float(best_r - selected_r_dep_star)
                        audit_has_recoverable_candidate = bool(best_r >= 0.0)
                        audit_selector_miss = bool(selected_r_dep_star < 0.0 and best_r >= 0.0)
                    audit_labels_done += int(len(labeled))
            except Exception as exc:
                if progress:
                    print({"event": "closed_loop_selected_label_audit_failed", "scene_rank": scenario_rank, "step": step_idx, "error": str(exc)}, flush=True)
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
        selected_teacher_r_dep = _safe_optional_float(teacher_r_dep[sel_idx]) if compute_teacher_labels else (_safe_optional_float(selected_audit_data.get("r_dep_star")) if selected_audit_data is not None else None)
        selected_teacher_r_orc = _safe_optional_float(teacher_r_orc[sel_idx]) if compute_teacher_labels else (_safe_optional_float(selected_audit_data.get("r_orc_star")) if selected_audit_data is not None else None)
        selected_pred_r_dep = _safe_optional_float(info["pred_r_dep"][sel_idx])
        selected_pred_r_orc = _safe_optional_float(info["pred_r_orc"][sel_idx])
        selected_pred_gap = _safe_optional_float(info["pred_gap"][sel_idx])
        selected_nominal_deviation = _safe_optional_float(info["nominal_deviation"][sel_idx])
        selected_odg = _safe_optional_float(selected_sample.oracle_gap_star) if compute_teacher_labels else (_safe_optional_float(selected_audit_data.get("oracle_gap_star")) if selected_audit_data is not None else None)
        selected_artifact = bool(selected_sample.i_art_star) if compute_teacher_labels else (bool(int(_safe_float(selected_audit_data.get("i_art_star", 0.0)))) if selected_audit_data is not None else None)
        decisions.append(
            ClosedLoopDecision(
                scene_id=str(raw.scenario_id),
                step_index=int(step_idx),
                time_index=int(t),
                method=str(method),
                selected_index=int(sel_idx),
                selected_macro=str(prefix.macro_name),
                selected_candidate_index=int(selected_sample.candidate_index),
                selection_reason=str(info["selection"].reason),
                selected_utility=float(utility[sel_idx]),
                selected_teacher_r_dep=selected_teacher_r_dep,
                selected_teacher_r_orc=selected_teacher_r_orc,
                selected_pred_r_dep=selected_pred_r_dep,
                selected_pred_r_orc=selected_pred_r_orc,
                selected_pred_gap=selected_pred_gap,
                selected_nominal_deviation=selected_nominal_deviation,
                selected_odg=selected_odg,
                selected_artifact=selected_artifact,
                audit_candidate_count=audit_candidate_count,
                audit_best_candidate_index=audit_best_candidate_index,
                audit_best_teacher_r_dep=audit_best_teacher_r_dep,
                audit_best_drs=audit_best_drs,
                audit_selected_r_dep_regret=audit_selected_r_dep_regret,
                audit_has_recoverable_candidate=audit_has_recoverable_candidate,
                audit_selector_miss=audit_selector_miss,
                fra_exec=(selected_audit_fra_exec if selected_audit_fra_exec is not None else (None if selected_teacher_r_dep is None else float(selected_teacher_r_dep < 0.0))),
                fra_cand=info["fra_cand"],
                drs=(None if selected_audit_drs is None else float(selected_audit_drs)) if (selected_label_audit or coverage_label_audit) else info["drs"],
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
        "bucket_name": bucket_name,
        "target_key": target_key,
        "target_time_index": int(start_time_index_override) if start_time_index_override is not None else None,
        "num_decisions": int(len(decisions)),
        "num_metric_steps": int(len(metric_trace)),
        "method": method,
        "gamma_rec": float(gamma),
        "label_mode": label_mode,
        "labels_available": bool(compute_teacher_labels or selected_label_audit or coverage_label_audit),
        "selected_label_audit": bool(selected_label_audit or coverage_label_audit),
        "coverage_label_audit": bool(coverage_label_audit),
        "audit_every_n_steps": int(audit_every_n_steps),
        "audit_labels_done": int(audit_labels_done),
        "closed_loop_FRA_exec": _mean_finite([d.fra_exec for d in decisions]),
        "closed_loop_FRA_cand": _mean_finite([d.fra_cand for d in decisions]),
        "closed_loop_DRS": _mean_finite([d.drs for d in decisions]),
        "closed_loop_ODG": _mean_finite([d.selected_odg for d in decisions]),
        "closed_loop_artifact_selection_rate": _mean_finite([float(d.selected_artifact) for d in decisions if d.selected_artifact is not None]),
        "closed_loop_audit_candidate_count": _mean_finite([d.audit_candidate_count for d in decisions]),
        "closed_loop_audit_best_R_dep": _mean_finite([d.audit_best_teacher_r_dep for d in decisions]),
        "closed_loop_audit_best_DRS": _mean_finite([d.audit_best_drs for d in decisions]),
        "closed_loop_audit_selected_R_dep_regret": _mean_finite([d.audit_selected_r_dep_regret for d in decisions]),
        "closed_loop_audit_recoverable_candidate_rate": _mean_finite([float(d.audit_has_recoverable_candidate) for d in decisions if d.audit_has_recoverable_candidate is not None]),
        "closed_loop_audit_selector_miss_rate": _mean_finite([float(d.audit_selector_miss) for d in decisions if d.audit_selector_miss is not None]),
        "closed_loop_bounded_NUP": _mean_finite([d.nup for d in decisions], default=0.0),
        "closed_loop_pred_r_dep": _mean_finite([d.selected_pred_r_dep for d in decisions]),
        "closed_loop_pred_gap": _mean_finite([d.selected_pred_gap for d in decisions]),
        "closed_loop_nominal_deviation": _mean_finite([d.selected_nominal_deviation for d in decisions]),
        "intervention_rate": _mean_finite([float(d.selected_candidate_index != 0) for d in decisions], default=0.0),
        "metric_summary": metric_summary,
        "macro_counts": {m: int(sum(d.selected_macro == m for d in decisions)) for m in sorted({d.selected_macro for d in decisions})},
        "selection_reason_counts": {r: int(sum(d.selection_reason == r for d in decisions)) for r in sorted({d.selection_reason for d in decisions})},
        "decisions": decision_dicts,
    }
    if bool(cl_cfg.get("save_trace_npz", False)):
        out["state_xy_trace"] = state_xy_trace
    return out

def _aggregate_scene_results(scene_results: list[dict[str, Any]], method: str, source: str) -> dict[str, Any]:
    keys = ["closed_loop_FRA_exec", "closed_loop_FRA_cand", "closed_loop_DRS", "closed_loop_ODG", "closed_loop_artifact_selection_rate", "closed_loop_audit_candidate_count", "closed_loop_audit_best_R_dep", "closed_loop_audit_best_DRS", "closed_loop_audit_selected_R_dep_regret", "closed_loop_audit_recoverable_candidate_rate", "closed_loop_audit_selector_miss_rate", "closed_loop_bounded_NUP", "closed_loop_pred_r_dep", "closed_loop_pred_gap", "closed_loop_nominal_deviation", "intervention_rate"]
    agg: dict[str, Any] = {
        "source": source,
        "method": method,
        "num_scenes": int(len(scene_results)),
        "num_decisions": int(sum(int(s.get("num_decisions", 0)) for s in scene_results)),
        "num_metric_steps": int(sum(int(s.get("num_metric_steps", 0)) for s in scene_results)),
        "label_modes": sorted({str(s.get("label_mode", "unknown")) for s in scene_results}),
    }
    for k in keys:
        vals = [s.get(k, None) for s in scene_results if int(s.get("num_decisions", 0)) > 0]
        agg[k] = _mean_finite(vals)
    metric_names = sorted({mk for s in scene_results for mk in (s.get("metric_summary", {}) or {}).keys()})
    agg["waymax_metrics"] = {}
    for mk in metric_names:
        vals = [(s.get("metric_summary", {}) or {}).get(mk, None) for s in scene_results]
        agg["waymax_metrics"][mk] = _mean_finite(vals, default=0.0)
    macro_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    for s in scene_results:
        for m, c in (s.get("macro_counts", {}) or {}).items():
            macro_counts[m] = macro_counts.get(m, 0) + int(c)
        for r, c in (s.get("selection_reason_counts", {}) or {}).items():
            reason_counts[r] = reason_counts.get(r, 0) + int(c)
    agg["macro_counts"] = macro_counts
    agg["selection_reason_counts"] = reason_counts
    return agg




def _dataset_label_for_sample_path(path: Path) -> str:
    try:
        return path.parent.parent.name if path.parent.name == "samples" else path.parent.name
    except Exception:
        return "dataset"


def _load_closed_loop_targets(dataset_spec: str | None, cfg: dict) -> list[dict[str, Any]]:
    """Load unique (bucket, scene_id, time_index) targets from OC-RAP offline roots."""
    if not dataset_spec:
        return []
    cl_cfg = cfg.get("closed_loop", {}) if isinstance(cfg.get("closed_loop", {}), dict) else {}
    split_filter = str(cl_cfg.get("bucket_split", "") or "").strip()
    max_targets = int(cl_cfg.get("max_bucket_targets", 0) or 0)
    max_per_scene = int(cl_cfg.get("max_targets_per_scene", 1) or 1)
    paths = iter_sample_paths_many(dataset_spec)
    seen: set[tuple[str, str, int]] = set()
    per_scene: dict[str, int] = {}
    targets: list[dict[str, Any]] = []
    for p in paths:
        split = str(scalar_metadata_for_path(p, "split_id", ""))
        if split_filter and split != split_filter:
            continue
        scene_id = str(scalar_metadata_for_path(p, "scene_id", ""))
        if not scene_id:
            continue
        try:
            time_index = int(float(scalar_metadata_for_path(p, "time_index", 0)))
        except Exception:
            continue
        bucket = _dataset_label_for_sample_path(Path(p))
        key = (bucket, scene_id, time_index)
        if key in seen:
            continue
        if per_scene.get(scene_id, 0) >= max_per_scene:
            continue
        seen.add(key)
        per_scene[scene_id] = per_scene.get(scene_id, 0) + 1
        targets.append({
            "bucket_name": bucket,
            "scene_id": scene_id,
            "time_index": int(time_index),
            "target_key": f"{bucket}:{scene_id}:t{int(time_index)}",
        })
        if max_targets > 0 and len(targets) >= max_targets:
            break
    return targets


def _aggregate_with_buckets(scene_results: list[dict[str, Any]], method: str, source: str) -> dict[str, Any]:
    result = _aggregate_scene_results(scene_results, method, source)
    buckets = sorted({str(s.get("bucket_name")) for s in scene_results if s.get("bucket_name")})
    if buckets:
        result["per_bucket"] = {}
        for b in buckets:
            sub = [s for s in scene_results if str(s.get("bucket_name")) == b]
            result["per_bucket"][b] = _aggregate_scene_results(sub, method, source)
    return result

def closed_loop_evaluate(dataset_patterns: str, checkpoint: str | Path | None, output: str | Path, cfg: dict) -> dict[str, Any]:
    if not str(dataset_patterns).strip() or str(dataset_patterns).strip().startswith("@"):
        raise ValueError(
            "closed-loop --dataset is empty or only contains an @limit suffix. "
            "Pass an explicit WOMD TFRecord path, e.g. ${WOMD_VAL}@150 after exporting WOMD_VAL."
        )
    cl_cfg = cfg.get("closed_loop", {}) if isinstance(cfg.get("closed_loop", {}), dict) else {}
    max_scenes = int(cl_cfg.get("max_scenarios", cfg.get("max_scenarios") or 8))
    max_rollouts = int(cl_cfg.get("max_rollouts", max_scenes) or max_scenes)
    raw_max_scenarios = cl_cfg.get("raw_max_scenarios", cfg.get("max_scenarios", None))
    raw_max_scenarios = None if raw_max_scenarios in {None, "", 0, "0"} else int(raw_max_scenarios)
    method = str(cl_cfg.get("method", "ocrap")).lower()
    gamma = float((cfg.get("selection", {}) or {}).get("gamma_rec", 0.0))
    if not np.isfinite(gamma) and not bool(cl_cfg.get("allow_infinite_gamma", False)):
        raise ValueError(
            "closed-loop selection.gamma_rec is not finite. Re-read calibration.json with the requested delta "
            "or pass --set closed_loop.allow_infinite_gamma=true only for debugging."
        )
    local = _apply_gamma_rec_by_bucket_file(dict(cfg))
    local["data_source"] = "womd"
    local["simulation_backend"] = "waymax_closed_loop"
    local["womd_patterns"] = dataset_patterns
    local["max_scenarios"] = max_scenes

    # Closed-loop evaluation is frequently run on WOMD TFExample shards whose
    # records do not contain `path_samples/*` route features.  The offline
    # OC-RAP dataset-build commands already disable SDC-path parsing for the
    # held-out test buckets; keep the online evaluator consistent by default.
    # Set `--set closed_loop.use_sdc_paths=true` only when the raw WOMD shards
    # are known to contain path_samples/{xyz,valid,id,arc_length,on_route}.
    wx = dict(local.get("waymax", {}) or {})
    use_sdc_paths = bool(cl_cfg.get("use_sdc_paths", False))
    wx["dataloader_include_sdc_paths"] = use_sdc_paths
    if not use_sdc_paths:
        route_metrics = {"sdc_wrongway", "sdc_off_route", "sdc_progression"}
        metrics = wx.get("metrics_to_run", [])
        if metrics:
            wx["metrics_to_run"] = [str(m) for m in metrics if str(m) not in route_metrics]
    local["waymax"] = wx

    art = dict(local.get("artifact", {}) or {})
    mine_p = float(cl_cfg.get("artifact_mine_probability", 0.0) or 0.0)
    art["force_mine"] = mine_p > 0.0
    art["mine_probability"] = max(0.0, min(1.0, mine_p))
    local["artifact"] = art
    bundle = load_model_bundle(checkpoint, local)
    if checkpoint and bundle is None:
        raise FileNotFoundError(f"Could not load model checkpoint for closed-loop evaluation: {checkpoint}")
    source = "model" if bundle is not None else "teacher_fallback"
    output_path = Path(output)
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    save_partial = bool(cl_cfg.get("save_partial", True))
    progress = bool(cl_cfg.get("progress", True))
    target_spec = str(cl_cfg.get("bucket_dataset", cl_cfg.get("target_dataset", "")) or "").strip()
    targets = _load_closed_loop_targets(target_spec, local)
    target_map: dict[str, list[dict[str, Any]]] = {}
    for t in targets:
        target_map.setdefault(str(t["scene_id"]), []).append(t)
    scene_results = []
    raw_seen = 0
    matched_targets = 0
    if progress and targets:
        print({"event": "closed_loop_bucket_targets_loaded", "num_targets": len(targets), "num_target_scenes": len(target_map), "max_rollouts": max_rollouts, "raw_max_scenarios": raw_max_scenarios}, flush=True)
    for i, raw in enumerate(iter_waymax_womd_scenarios(dataset_patterns, max_scenarios=raw_max_scenarios if targets else max_scenes, parser_cfg=local)):
        raw_seen += 1
        raw_targets = target_map.get(str(raw.scenario_id), []) if targets else [{"bucket_name": None, "time_index": None, "target_key": None}]
        if targets and not raw_targets:
            continue
        for target in raw_targets:
            if len(scene_results) >= (max_rollouts if targets else max_scenes):
                break
            rank = len(scene_results)
            if progress:
                print({"event": "closed_loop_scene_start", "scene_rank": rank, "raw_rank": i, "scene_id": str(raw.scenario_id), "bucket": target.get("bucket_name"), "start_time_index": target.get("time_index"), "max_rollouts": max_rollouts if targets else max_scenes}, flush=True)
            gamma_i = _gamma_for_bucket(gamma, local, target.get("bucket_name"))
            scene_results.append(_rollout_one_scene(raw, rank, bundle, local, method, gamma_i, start_time_index_override=target.get("time_index"), bucket_name=target.get("bucket_name"), target_key=target.get("target_key")))
            matched_targets += int(bool(targets))
            if save_partial:
                partial = _aggregate_with_buckets(scene_results, method, source)
                partial["scenes"] = scene_results
                partial["partial"] = True
                partial["bucket_dataset"] = target_spec or None
                partial["bucket_target_count"] = len(targets)
                partial["bucket_matched_rollouts"] = matched_targets
                partial["raw_scenarios_seen"] = raw_seen
                write_json(partial, partial_path)
            if progress:
                print({"event": "closed_loop_scene_done", "scene_rank": rank, "num_decisions": scene_results[-1].get("num_decisions", 0)}, flush=True)
        if len(scene_results) >= (max_rollouts if targets else max_scenes):
            break
    result = _aggregate_with_buckets(scene_results, method, source)
    result["bucket_dataset"] = target_spec or None
    result["bucket_target_count"] = len(targets)
    result["bucket_matched_rollouts"] = matched_targets
    result["raw_scenarios_seen"] = raw_seen
    if targets and matched_targets == 0:
        result.setdefault("warnings", []).append("No offline bucket scene_id matched the supplied WOMD raw dataset/pattern. Check WOMD_VAL vs WOMD_VAL_INTERACTIVE and scenario_start_index/raw_max_scenarios.")
    result["scenes"] = scene_results
    result["gamma_rec"] = gamma
    result["gamma_rec_by_bucket"] = (local.get("selection", {}) or {}).get("gamma_rec_by_bucket", {}) if isinstance(local.get("selection", {}), dict) else {}
    result["notes"] = [
        "This is a true Waymax receding-horizon loop: reset once, select an action from current SimulatorState, step the environment, then replan from the updated SimulatorState.",
        "closed_loop.label_mode=fast skips expensive per-candidate OC-MERO teacher labels inside the online loop. Use closed_loop.label_mode=selected to label only the executed candidate; use label_mode=selected_topk/coverage to label selected plus a small diagnostic top-k subset; use label_mode=all only for tiny exhaustive audits.",
        "Non-SDC actors use Waymax/default log-playback dynamics unless controlled by the environment configuration.",
    ]
    write_json(result, output)
    return result
