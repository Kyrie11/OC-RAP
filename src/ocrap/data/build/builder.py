from __future__ import annotations

import csv
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from ocrap.algorithms.ocmero import oc_mero
from ocrap.data.schema import DatasetSample, RawScenario
from ocrap.data.serialization import ensure_dir, np_savez, write_json
from ocrap.data.split import resolve_split
from ocrap.data.validation import missing_fields
from ocrap.planning.prefix_generation import generate_candidate_prefixes
from ocrap.roots import aggregate_root_margins, cluster_roots, future_trajectory_signature
from ocrap.simulation.futures import generate_counterfactual_futures
from ocrap.simulation.observation import compatibility_labels, render_observation, unknown_ratio_in_corridor
from ocrap.simulation.teacher import compute_future_option_margins, default_recovery_options, option_valid_mask
from ocrap.utils.seed import stable_seed

from .history import construct_history
from .regimes import assign_regimes
from .synthetic import iter_synthetic_scenarios


MANIFEST_FIELDS = [
    "path",
    "scene_id",
    "original_scenario_id",
    "time_index",
    "candidate_index",
    "split_id",
    "is_nominal",
    "r_orc_star",
    "r_dep_star",
    "oracle_gap_star",
    "i_art_star",
    "regime_label",
]

PROFILE_FIELDS = [
    "sample_index",
    "scene_id",
    "time_index",
    "candidate_index",
    "macro_name",
    "num_futures",
    "num_options",
    "total_s",
    "future_generation_s",
    "teacher_margins_s",
    "root_clustering_s",
    "observation_s",
    "ocmero_s",
    "r_orc_star",
    "r_dep_star",
    "i_art_star",
    "waymax_teacher_rollouts_executed",
    "waymax_teacher_metric_cache_hits",
    "waymax_teacher_screened_hybrid",
]

SCENE_PROFILE_FIELDS = [
    "raw_scenario_index",
    "scene_time_group",
    "scene_id",
    "original_scenario_id",
    "time_index",
    "split_id",
    "num_selected_samples",
    "construct_history_s",
    "build_samples_s",
    "sample_compute_sum_s",
    "sample_future_generation_sum_s",
    "sample_teacher_margins_sum_s",
    "sample_observation_sum_s",
    "sample_root_clustering_sum_s",
    "sample_ocmero_sum_s",
    "npz_serialize_s",
    "npz_write_s",
    "manifest_checkpoint_s",
    "scene_time_total_s",
]


def _profiling_cfg(cfg: dict) -> dict:
    prof = cfg.get("profiling", {}) if isinstance(cfg.get("profiling", {}), dict) else {}
    return prof


def _profiling_enabled(cfg: dict) -> bool:
    return bool(_profiling_cfg(cfg).get("enabled", False))


def _profile_log(cfg: dict, msg: str) -> None:
    if _profiling_enabled(cfg):
        print(f"[ocrap-profile] {msg}", flush=True)


def _now() -> float:
    return time.perf_counter()


def _io_cfg(cfg: dict) -> dict:
    io = cfg.get("io", cfg.get("dataset_io", {}))
    return io if isinstance(io, dict) else {}


def _stage_add(stage: dict[str, float], key: str, value: float) -> None:
    stage[key] = float(stage.get(key, 0.0) + float(value))


def _sample_timing_sum(samples: list[DatasetSample], key: str) -> float:
    total = 0.0
    for sample in samples:
        try:
            timings = sample.diagnostics.get("build_timing_s", {}) if isinstance(sample.diagnostics, dict) else {}
            total += float(timings.get(key, 0.0))
        except Exception:
            continue
    return float(total)


def _future_metadata_sum(sample: DatasetSample, key: str) -> int:
    vals: list[int] = []
    for fut in sample.futures:
        try:
            if key in fut.metadata:
                vals.append(int(fut.metadata.get(key, 0)))
        except Exception:
            continue
    return int(sum(vals))


def _future_metadata_any(sample: DatasetSample, key: str) -> bool:
    for fut in sample.futures:
        try:
            if bool(fut.metadata.get(key, False)):
                return True
        except Exception:
            continue
    return False


def _score_planning_times(raw: RawScenario, cfg: dict) -> tuple[list[int], dict[int, list[str]]]:
    sr = float(cfg.get("sample_rate_hz", 10))
    H = max(1, int(round(float(cfg.get("history_horizon_s", 1.0)) * sr)))
    T_future = max(2, int(round((float(cfg.get("prefix_horizon_s", 1.0)) + float(cfg.get("recovery_horizon_s", 4.0))) * sr)))
    start = H
    end = raw.agent_states.shape[0] - T_future - 1
    if end <= start:
        return [], {}
    stride = max(1, int(round(float(cfg.get("planning_time_stride_s", 0.5)) * sr)))
    all_times = np.arange(start, end, dtype=np.int64)
    uniform_times = [int(t) for t in all_times[::stride].tolist()]
    reasons_by_time: dict[int, set[str]] = {int(t): {"uniform"} for t in uniform_times}
    # Interaction/low-headroom/near-contact biased scores.
    scored: list[tuple[float, int, list[str]]] = []
    for t in all_times:
        if not raw.agent_valid[t, raw.sdc_track_index]:
            continue
        ego = raw.agent_states[t, raw.sdc_track_index]
        reasons: list[str] = []
        score = 0.0
        other_valid = raw.agent_valid[t]
        for a in range(raw.agent_states.shape[1]):
            if a == raw.sdc_track_index or not other_valid[a]:
                continue
            d = float(np.linalg.norm(raw.agent_states[t, a, :2] - ego[:2]))
            if d < 20.0:
                score += 20.0 - d
                reasons.append("interaction_biased")
            if d < 8.0:
                score += 20.0
                reasons.append("near_contact")
        if raw.map_polylines.size:
            # crosswalk/merge/route bottleneck proxy near x=35 in synthetic fixture.
            if abs(float(ego[0]) - 35.0) < 15.0:
                score += 5.0
                reasons.append("route_bottleneck_crosswalk")
        if score == 0:
            reasons.append("uniform")
        scored.append((score, int(t), sorted(set(reasons))))
    scored.sort(reverse=True)
    max_biased = max(0, int(cfg.get("max_biased_times_per_scenario", 4)))
    biased_times: list[int] = []
    for score, t, reasons in scored:
        if len(biased_times) >= max_biased:
            break
        # Do not let zero-score frames consume the biased quota; uniform sampling
        # already covers them.
        if score <= 0.0:
            continue
        biased_times.append(int(t))
        reasons_by_time.setdefault(int(t), set()).update(reasons)

    max_times = max(0, int(cfg.get("max_times_per_scenario", 8)))
    quality = cfg.get("dataset_quality", {}) if isinstance(cfg.get("dataset_quality", {}), dict) else {}
    min_uniform = max(0, int(cfg.get("min_uniform_times_per_scenario", quality.get("min_uniform_times_per_scenario", 0))))
    min_uniform = min(min_uniform, max_times)
    ordered: list[int] = []
    seen: set[int] = set()

    def add_from(seq, limit: int | None = None) -> None:
        added = 0
        for t in seq:
            if len(ordered) >= max_times:
                break
            if limit is not None and added >= limit:
                break
            if int(t) in seen:
                continue
            ordered.append(int(t))
            seen.add(int(t))
            added += 1

    # Keep interaction-biased frames, but reserve a configurable number of
    # uniform frames so primary mixed datasets contain normal/NUP examples.
    add_from(biased_times, max(0, max_times - min_uniform) if min_uniform else None)
    add_from(uniform_times, min_uniform)
    add_from(uniform_times, None)
    add_from([int(t) for _, t, _ in scored], None)
    times = ordered
    return times, {t: sorted(reasons_by_time.get(t, {"uniform"})) for t in times}


def select_planning_times(raw: RawScenario, cfg: dict) -> list[int]:
    times, _ = _score_planning_times(raw, cfg)
    return times


def select_planning_times_with_reasons(raw: RawScenario, cfg: dict) -> tuple[list[int], dict[int, list[str]]]:
    return _score_planning_times(raw, cfg)


def _teacher_diag_to_jsonable(diags) -> dict:
    if not diags or not diags[0]:
        return {}
    return {
        "component_margins_sample": diags[0][0].component_margins,
        "active_sample": diags[0][0].active,
        "controller_sample": diags[0][0].controller_diagnostics,
    }

def _quality_requires_artifact_pair(cfg: dict) -> bool:
    quality = cfg.get("dataset_quality", {}) if isinstance(cfg.get("dataset_quality", {}), dict) else {}
    return bool(quality.get("require_artifact_pairs", False))


def _artifact_pair_mode(cfg: dict) -> str:
    quality = cfg.get("dataset_quality", {}) if isinstance(cfg.get("dataset_quality", {}), dict) else {}
    raw = str(quality.get("artifact_pair_mode", "filter" if _quality_requires_artifact_pair(cfg) else "tag")).lower()
    aliases = {"strict": "filter", "only": "filter", "keep_all": "tag", "mixed": "balanced"}
    mode = aliases.get(raw, raw)
    if mode not in {"filter", "tag", "balanced"}:
        raise ValueError(f"dataset_quality.artifact_pair_mode={raw!r} must be filter, tag, or balanced")
    return mode


def _quality_int(cfg: dict, key: str, default: int) -> int:
    quality = cfg.get("dataset_quality", {}) if isinstance(cfg.get("dataset_quality", {}), dict) else {}
    return max(0, int(quality.get(key, default)))


def _quality_bool(cfg: dict, key: str, default: bool) -> bool:
    quality = cfg.get("dataset_quality", {}) if isinstance(cfg.get("dataset_quality", {}), dict) else {}
    return bool(quality.get(key, default))


def _quality_attempt_cap(cfg: dict, key: str, default: int) -> int:
    # 0 or a missing value means no additional cap beyond the available prefix
    # list.  Positive values bound expensive quota hunting per scene-time.
    val = _quality_int(cfg, key, default)
    return int(default) if val <= 0 else min(int(val), int(default))


def _cfg_with_artifact_mining(cfg: dict, *, enable: bool) -> dict:
    """Copy cfg and force artifact mining on/off for one materialization.

    In balanced primary builds, using only stochastic mining can still yield an
    all-artifact or all-non-artifact scene-time by chance.  This helper lets the
    builder intentionally create a non-mined branch set and a mined branch set
    on different prefixes while leaving the caller's label mode intact.
    """
    local = dict(cfg)
    art = dict(local.get("artifact", {}) or {})
    art["force_mine"] = bool(enable)
    art["mine_probability"] = 1.0 if enable else 0.0
    quality_in = cfg.get("dataset_quality", {}) if isinstance(cfg.get("dataset_quality", {}), dict) else {}
    if enable and bool(quality_in.get("artifact_pass_use_margin_override", False)):
        # Primary mixed builds should keep the no-mine half natural, but the
        # deliberately mined hidden yield/accelerate pair needs the same
        # branch-specific oracle-artifact label used by the sanity dataset.
        # Otherwise benign Waymax metrics can make r_dep_star positive for every
        # mined sample, producing complete_artifact_pair_count>0 but
        # artifact_fraction==0.
        art["use_margin_override"] = True

        # Speed path for proof-artifact / balanced mined passes.  The hidden
        # augmented branch is explicitly labeled by the margin override, so a full
        # Waymax recovery rollout for that branch is redundant for quick-proof
        # label construction.  These knobs are local to the mined pass and keep
        # the no-mine/natural half untouched.
        wx = dict(local.get("waymax", {}) or {})
        if bool(quality_in.get("artifact_pass_skip_augmented_waymax", True)):
            wx["skip_waymax_rollout_for_augmented_override"] = True
        if bool(quality_in.get("artifact_pass_apply_override_to_screened", True)):
            wx["apply_artifact_override_to_screened_options"] = True
        if not bool(quality_in.get("artifact_pass_compute_future_metrics", False)):
            wx["compute_future_metrics"] = False
        if bool(quality_in.get("artifact_pass_structural_teacher", False)):
            # More aggressive debug/proof mode: no recovery-option Waymax rollout
            # is executed for the mined pass.  Use only when the artifact set is
            # reported as a proof/ablation set rather than strict rollout labels.
            wx["teacher_backend"] = "structural"
        local["waymax"] = wx
    local["artifact"] = art
    quality = dict(local.get("dataset_quality", {}) or {})
    # The forced materialization itself must not be skipped by the strict
    # preflight gate unless we are creating the artifact half.
    quality["require_artifact_pairs"] = bool(enable and quality.get("require_artifact_pairs", False))
    local["dataset_quality"] = quality
    return local


def _sample_is_artifact(sample: DatasetSample, cfg: dict | None = None) -> bool:
    quality = (cfg or {}).get("dataset_quality", {}) if isinstance((cfg or {}).get("dataset_quality", {}), dict) else {}
    if bool(quality.get("artifact_quota_uses_label", True)):
        return bool(sample.i_art_star)
    return bool(sample.i_art_star or sample.diagnostics.get("complete_artifact_pair", False))


def _sample_obs_negative_fraction(sample: DatasetSample) -> float:
    try:
        y = np.asarray(sample.y_obs, dtype=float)
        valid = np.asarray(getattr(sample, "root_valid", np.ones(y.shape[0], dtype=bool)), dtype=bool).reshape(-1)[: y.shape[0]]
        idx = np.where(valid)[0]
        if idx.size <= 1:
            return 0.0
        sub = y[np.ix_(idx, idx)]
        mask = ~np.eye(idx.size, dtype=bool)
        denom = int(mask.sum())
        if denom <= 0:
            return 0.0
        return float(np.mean(sub[mask] < 0.5))
    except Exception:
        return 0.0


def _sample_passes_quality_gates(sample: DatasetSample, cfg: dict) -> bool:
    quality = cfg.get("dataset_quality", {}) if isinstance(cfg.get("dataset_quality", {}), dict) else {}
    min_obs_neg = float(quality.get("min_obs_negative_fraction_per_sample", 0.0) or 0.0)
    if min_obs_neg > 0.0 and _sample_obs_negative_fraction(sample) + 1e-12 < min_obs_neg:
        sample.diagnostics["quality_drop_reason"] = "low_obs_negative_fraction"
        sample.diagnostics["obs_negative_fraction"] = float(_sample_obs_negative_fraction(sample))
        return False
    if bool(quality.get("require_negative_deployable_sample", False)):
        thr = float(quality.get("negative_deployable_threshold", 0.0))
        if not (float(sample.r_dep_star) < thr):
            sample.diagnostics["quality_drop_reason"] = "not_negative_deployable"
            return False
    sample.diagnostics["obs_negative_fraction"] = float(_sample_obs_negative_fraction(sample))
    return True


def _has_complete_artifact_pair(sample: DatasetSample) -> bool:
    branches: set[str] = set()
    for fut in sample.futures:
        meta = fut.metadata or {}
        if not meta.get("artifact_pair_key") or not meta.get("hidden_emergence", False):
            continue
        if not meta.get("from_unknown_mask", False) or meta.get("spawn_in_visible_free", False):
            continue
        branch = str(meta.get("artifact_branch", ""))
        if branch:
            branches.add(branch)
    return {"yield", "accelerate"}.issubset(branches)


def _max_accepted_prefixes_per_scene_time(cfg: dict) -> int:
    quality = cfg.get("dataset_quality", {}) if isinstance(cfg.get("dataset_quality", {}), dict) else {}
    return max(0, int(quality.get("max_accepted_prefixes_per_scene_time", 0)))


def _balanced_prefix_order(history, prefixes: list[CandidatePrefix], cfg: dict, *, salt: str, nominal_first: bool) -> list[CandidatePrefix]:
    """Return a deterministic per-scene-time prefix order for balanced builds.

    Without this, a small ``max_accepted_prefixes_per_scene_time`` always
    selects the first macros in ``generate_candidate_prefixes``.  That is why
    4-sample smoke sets collapse to nominal/keep/brake/yield and all artifacts
    can land on yield.  Rotating the non-nominal part preserves deterministic
    reproducibility while spreading macro coverage across scene-times.
    """
    quality = cfg.get("dataset_quality", {}) if isinstance(cfg.get("dataset_quality", {}), dict) else {}
    if not bool(quality.get("balanced_rotate_prefix_order", True)):
        return list(prefixes)
    nominal = [p for p in prefixes if p.macro_name == "nominal"]
    rest = [p for p in prefixes if p.macro_name != "nominal"]
    if not rest:
        return list(prefixes)
    start = stable_seed("balanced-prefix-order", history.scene_id, history.time_index, salt) % len(rest)
    rest = rest[start:] + rest[:start]
    if nominal_first and nominal:
        return nominal[:1] + rest + nominal[1:]
    return rest + nominal


def _artifact_pair_attempt_is_possible(history, prefix: CandidatePrefix, cfg: dict) -> bool:
    if not _quality_requires_artifact_pair(cfg):
        return True
    if str(cfg.get("simulation_backend", "ocrap_surrogate")) != "waymax_closed_loop":
        return True
    try:
        from ocrap.simulation.waymax_rollout import can_mine_augmented_hidden_pair

        return bool(can_mine_augmented_hidden_pair(history, prefix, cfg))
    except Exception:
        # Do not silently discard data when the optional Waymax path changes.
        # A later strict rollout/quality gate will fail or skip the sample.
        return True

def _compute_within_root_dispersion(root_assignments: np.ndarray, obs_by_future: list, K: int, cfg: dict) -> np.ndarray:
    from ocrap.simulation.observation.compatibility import observation_distance

    out = np.zeros(K, dtype=np.float32)
    for k in range(K):
        idx = np.where(root_assignments == k)[0]
        if len(idx) <= 1:
            continue
        vals = []
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                vals.append(observation_distance(obs_by_future[int(idx[a])], obs_by_future[int(idx[b])], cfg))
        out[k] = float(np.mean(vals)) if vals else 0.0
    return out


def _materialize_sample(history, split_id: str, prefix: CandidatePrefix, a_idx: int, cfg: dict, options, option_valid, K: int) -> DatasetSample:
    prof = _profiling_cfg(cfg)
    t_all = _now()
    timings: dict[str, float] = {}

    t = _now()
    futures = generate_counterfactual_futures(history, prefix, cfg)
    timings["future_generation"] = _now() - t

    t = _now()
    future_probs = np.asarray([f.prior for f in futures], dtype=np.float32)
    future_probs = future_probs / max(float(future_probs.sum()), 1e-8)
    M_future, teacher_diags = compute_future_option_margins(history, prefix, futures, options, cfg)
    timings["teacher_margins"] = _now() - t

    t = _now()
    root = cluster_roots(M_future, future_probs, futures, cfg)
    M_star = aggregate_root_margins(M_future, root.assignments, future_probs, K, cfg)
    root_future_signature = future_trajectory_signature(futures, root.assignments, future_probs, K, width=int(cfg.get("model", {}).get("d_future_signature", 32)))
    timings["root_clustering"] = _now() - t

    t = _now()
    obs_by_future = [render_observation(history, prefix, f, cfg) for f in futures]
    within_disp = _compute_within_root_dispersion(root.assignments, obs_by_future, K, cfg)
    observations = []
    for k in range(K):
        rep = int(root.representative_indices[k]) if root.root_valid[k] and root.representative_indices[k] >= 0 else 0
        observations.append(obs_by_future[rep])
    Y, C, Dobs = compatibility_labels(observations, cfg)
    timings["observation"] = _now() - t

    t = _now()
    use_lcvar = not bool(cfg.get("ablation", {}).get("without_lower_tail", False))
    use_obs_kernel = not bool(cfg.get("ablation", {}).get("without_observation_kernel", False))
    res = oc_mero(
        M_star,
        root.root_probs,
        C,
        alpha=float(cfg.get("ocmero", {}).get("alpha", 0.2)),
        beta=float(cfg.get("ocmero", {}).get("beta", 0.2)),
        option_valid=option_valid,
        root_valid=root.root_valid,
        use_lcvar=use_lcvar,
        use_obs_kernel=use_obs_kernel,
        top_m=int(cfg.get("ocmero", {}).get("top_m", 8)),
    )
    timings["ocmero"] = _now() - t
    timings["total"] = _now() - t_all

    gamma_orc = float(cfg.get("artifact", {}).get("gamma_orc", 0.0))
    gamma_dep = float(cfg.get("artifact", {}).get("gamma_dep", 0.0))
    sample = DatasetSample(
        scene_id=history.scene_id,
        original_scenario_id=history.original_scenario_id,
        time_index=history.time_index,
        candidate_index=a_idx,
        split_id=split_id,
        is_nominal=(a_idx == 0),
        h_t=history,
        prefix=prefix,
        futures=futures,
        future_probs=future_probs,
        root_assignments=root.assignments,
        root_probs=root.root_probs,
        root_signature=root.root_signature,
        root_future_signature=root_future_signature,
        root_valid=root.root_valid,
        root_representative_future_id=root.representative_indices,
        future_to_root_weight=root.future_to_root_weight,
        within_root_obs_dispersion=within_disp,
        obs_distance=Dobs,
        y_obs=Y,
        c_star=C,
        recovery_options=options,
        m_star=M_star,
        option_valid=option_valid,
        r_orc_star=res.r_orc,
        r_dep_star=res.r_dep,
        oracle_gap_star=res.gap,
        i_art_star=bool(res.r_orc >= gamma_orc and res.r_dep < gamma_dep),
        regime_label={},
        valid_masks={"root_valid": root.root_valid.astype(bool).tolist(), "option_valid": option_valid.astype(bool).tolist()},
        teacher_diagnostics=_teacher_diag_to_jsonable(teacher_diags),
        diagnostics={
            "future_sources": [f.source for f in futures],
            "root_clustering": root.metadata,
            "unknown_ratio_in_corridor": unknown_ratio_in_corridor(history.occ_mask),
            "time_sampling_reasons": history.metadata.get("time_sampling_reasons", []),
            "ocmero_best_option": res.best_option.tolist(),
            "odg_pos": res.odg_pos,
            "complete_artifact_pair": False,  # filled below
            "build_timing_s": {k: float(v) for k, v in timings.items()},
        },
    )
    if bool(prof.get("enabled", False)):
        # Write a live per-sample CSV immediately, before the full scene-time group
        # finishes.  On real Waymax builds a single scene-time can take many
        # minutes, so waiting until samples are returned/written can leave users
        # with no machine-readable profile while the build is still running.
        live_path = cfg.get("_profile_live_path")
        if live_path:
            try:
                live_idx = int(cfg.get("_profile_live_sample_counter", 0)) + 1
                cfg["_profile_live_sample_counter"] = live_idx
                _append_csv_rows(
                    Path(str(live_path)),
                    [_sample_profile_row(sample, live_idx)],
                    PROFILE_FIELDS,
                    fsync_file=bool(prof.get("profile_csv_fsync", False)),
                )
            except Exception:
                pass
        if bool(prof.get("log_every_sample", False)) or timings["total"] >= float(prof.get("slow_sample_s", 30.0)):
            _profile_log(
                cfg,
                "sample scene=%s t=%s prefix=%s/%s futures=%d opts=%d total=%.2fs future=%.2fs teacher=%.2fs obs=%.2fs root=%.2fs ocmero=%.2fs wx_rollouts=%d wx_cache_hits=%d screened=%d"
                % (
                    history.scene_id,
                    history.time_index,
                    a_idx,
                    prefix.macro_name,
                    len(futures),
                    len(options),
                    timings["total"],
                    timings["future_generation"],
                    timings["teacher_margins"],
                    timings["observation"],
                    timings["root_clustering"],
                    timings["ocmero"],
                    _future_metadata_sum(sample, "waymax_teacher_rollouts_executed"),
                    _future_metadata_sum(sample, "waymax_teacher_metric_cache_hits"),
                    int(_future_metadata_any(sample, "waymax_teacher_screened_hybrid")),
                ),
            )
    return sample

def _try_add_sample(
    selected: list[DatasetSample],
    deferred: list[DatasetSample],
    sample: DatasetSample,
    *,
    cfg: dict,
    max_accepted: int,
    max_art: int,
    max_non: int,
    macros_seen: set[str],
    macro_diversity_first: bool,
) -> tuple[int, int]:
    is_art = _sample_is_artifact(sample, cfg)
    n_art = sum(int(_sample_is_artifact(s, cfg)) for s in selected)
    n_non = len(selected) - n_art
    keep_now = True
    if is_art and max_art > 0 and n_art >= max_art:
        keep_now = False
    if (not is_art) and max_non > 0 and n_non >= max_non:
        keep_now = False
    if macro_diversity_first and sample.prefix.macro_name in macros_seen and max_accepted > 0 and len(selected) >= max_accepted:
        keep_now = False
    if keep_now and (max_accepted <= 0 or len(selected) < max_accepted):
        selected.append(sample)
        macros_seen.add(sample.prefix.macro_name)
    else:
        deferred.append(sample)
    n_art = sum(int(_sample_is_artifact(s, cfg)) for s in selected)
    return n_art, len(selected) - n_art




def _replace_or_append_sample(
    selected: list[DatasetSample],
    sample: DatasetSample,
    *,
    max_accepted: int,
    macros_seen: set[str] | None = None,
) -> None:
    """Insert a mandatory anchor sample without exceeding the scene-time cap."""
    if any(int(s.candidate_index) == int(sample.candidate_index) for s in selected):
        return
    if max_accepted <= 0 or len(selected) < max_accepted:
        selected.append(sample)
        if macros_seen is not None:
            macros_seen.add(sample.prefix.macro_name)
        return
    # Keep the nominal anchor by evicting the least critical non-nominal sample.
    for idx in range(len(selected) - 1, -1, -1):
        if not bool(selected[idx].is_nominal):
            selected[idx] = sample
            if macros_seen is not None:
                macros_seen.clear()
                macros_seen.update(s.prefix.macro_name for s in selected)
            return
    # Degenerate case: all retained samples are nominal duplicates; replace tail.
    selected[-1] = sample
    if macros_seen is not None:
        macros_seen.clear()
        macros_seen.update(s.prefix.macro_name for s in selected)


def _dedupe_nominal(selected: list[DatasetSample]) -> list[DatasetSample]:
    out: list[DatasetSample] = []
    seen_nominal = False
    seen_candidate: set[int] = set()
    for sample in selected:
        cid = int(sample.candidate_index)
        if cid in seen_candidate:
            continue
        if bool(sample.is_nominal):
            if seen_nominal:
                continue
            seen_nominal = True
        out.append(sample)
        seen_candidate.add(cid)
    return out

def build_samples_for_history(history, split_id: str, cfg: dict, progress_callback: Callable[[int], None] | None = None) -> list[DatasetSample]:
    prefixes = generate_candidate_prefixes(history, cfg)
    options = default_recovery_options(int(cfg.get("num_recovery_options", 24)), shoulder_available=bool(history.metadata.get("shoulder_available", True)), adjacent_available=bool(history.metadata.get("adjacent_available", True)))
    option_valid = option_valid_mask(options)
    K = int(cfg.get("num_roots", 8))
    max_accepted = _max_accepted_prefixes_per_scene_time(cfg)
    mode = _artifact_pair_mode(cfg)
    min_art = _quality_int(cfg, "min_artifact_prefixes_per_scene_time", 1 if mode == "balanced" else 0)
    max_art = _quality_int(cfg, "max_artifact_prefixes_per_scene_time", max_accepted if max_accepted else len(prefixes))
    min_non = _quality_int(cfg, "min_nonartifact_prefixes_per_scene_time", 1 if mode == "balanced" else 0)
    max_non = _quality_int(cfg, "max_nonartifact_prefixes_per_scene_time", max_accepted if max_accepted else len(prefixes))
    macro_diversity_first = _quality_bool(cfg, "macro_diversity_first", True)
    balanced_two_pass = _quality_bool(cfg, "balanced_two_pass", True)
    max_non_attempts = _quality_attempt_cap(cfg, "max_nonartifact_attempts_per_scene_time", len(prefixes))
    max_art_attempts = _quality_attempt_cap(cfg, "max_artifact_attempts_per_scene_time", len(prefixes))

    selected: list[DatasetSample] = []
    deferred: list[DatasetSample] = []
    macros_seen: set[str] = set()
    materialized_prefixes: set[int] = set()

    def materialize(prefix, local_cfg: dict, *, enforce_quality: bool = True, allow_filter_drop: bool = True) -> DatasetSample | None:
        a_idx = int(prefix.macro_id)
        if allow_filter_drop and mode == "filter" and not _artifact_pair_attempt_is_possible(history, prefix, local_cfg):
            return None
        sample = _materialize_sample(history, split_id, prefix, a_idx, local_cfg, options, option_valid, K)
        has_pair = _has_complete_artifact_pair(sample)
        sample.diagnostics["complete_artifact_pair"] = bool(has_pair)
        if allow_filter_drop and mode == "filter" and not has_pair:
            return None
        if enforce_quality and not _sample_passes_quality_gates(sample, local_cfg):
            return None
        if not enforce_quality:
            sample.diagnostics["quality_anchor_retained"] = True
            sample.diagnostics["obs_negative_fraction"] = float(_sample_obs_negative_fraction(sample))
        return sample

    if mode == "balanced" and balanced_two_pass:
        # Pass 1: guarantee a non-artifact / nominal-preservation side by turning
        # off hidden-pair mining for this materialization.  This is what prevents
        # a primary WOMD build from degenerating into the current stress-only set.
        no_mine_cfg = _cfg_with_artifact_mining(cfg, enable=False)
        no_mine_prefixes = _balanced_prefix_order(history, prefixes, cfg, salt="nonartifact", nominal_first=bool((cfg.get("dataset_quality", {}) or {}).get("balanced_keep_nominal_nonartifact", True)))
        non_attempts = 0
        for prefix in no_mine_prefixes:
            if non_attempts >= max_non_attempts:
                break
            if max_accepted > 0 and len(selected) >= max_accepted and len(selected) - sum(int(_sample_is_artifact(s, no_mine_cfg)) for s in selected) >= min_non:
                break
            non_attempts += 1
            sample = materialize(prefix, no_mine_cfg)
            materialized_prefixes.add(int(prefix.macro_id))
            if sample is not None and not _sample_is_artifact(sample, no_mine_cfg):
                _try_add_sample(selected, deferred, sample, cfg=no_mine_cfg, max_accepted=max_accepted, max_art=max_art, max_non=max_non, macros_seen=macros_seen, macro_diversity_first=macro_diversity_first)
            if progress_callback is not None:
                progress_callback(1)
            if len(selected) - sum(int(_sample_is_artifact(s, no_mine_cfg)) for s in selected) >= max(min_non, min(max_non, max_accepted or max_non)):
                break

        # Pass 2: add a bounded stress side by forcing hidden-pair mining on the
        # remaining prefixes.  The quota keeps artifact_fraction below 1.0.
        mine_cfg = _cfg_with_artifact_mining(cfg, enable=True)
        mine_prefixes = _balanced_prefix_order(history, prefixes, cfg, salt="artifact", nominal_first=False)
        art_attempts = 0
        for prefix in mine_prefixes:
            n_art = sum(int(_sample_is_artifact(s, mine_cfg)) for s in selected)
            if art_attempts >= max_art_attempts:
                break
            if max_accepted > 0 and len(selected) >= max_accepted and n_art >= min_art:
                break
            if max_art > 0 and n_art >= max_art:
                break
            if int(prefix.macro_id) in materialized_prefixes and len(prefixes) > max_accepted:
                continue
            art_attempts += 1
            sample = materialize(prefix, mine_cfg)
            materialized_prefixes.add(int(prefix.macro_id))
            if sample is not None and _sample_is_artifact(sample, mine_cfg):
                _try_add_sample(selected, deferred, sample, cfg=mine_cfg, max_accepted=max_accepted, max_art=max_art, max_non=max_non, macros_seen=macros_seen, macro_diversity_first=macro_diversity_first)
            if progress_callback is not None:
                progress_callback(1)

        # Account for prefixes not materialized because quotas were satisfied.
        if progress_callback is not None:
            progress_callback(max(0, len(prefixes) - len(materialized_prefixes)))
    else:
        for a_idx, prefix in enumerate(prefixes):
            if max_accepted > 0 and len(selected) >= max_accepted:
                if progress_callback is not None:
                    progress_callback(len(prefixes) - a_idx)
                break
            sample = materialize(prefix, cfg)
            if sample is None:
                if progress_callback is not None:
                    progress_callback(1)
                continue
            if mode == "balanced":
                _try_add_sample(selected, deferred, sample, cfg=cfg, max_accepted=max_accepted, max_art=max_art, max_non=max_non, macros_seen=macros_seen, macro_diversity_first=macro_diversity_first)
            else:
                selected.append(sample)
            if progress_callback is not None:
                progress_callback(1)

    # Fill underfull scene-times from deferred examples without violating the
    # hard max_accepted cap.  This is a fallback for scenes where one side of the
    # balanced quota is physically unavailable.
    if max_accepted > 0 and len(selected) < max_accepted:
        for sample in deferred:
            if len(selected) >= max_accepted:
                break
            selected.append(sample)
    if max_accepted > 0:
        selected = selected[:max_accepted]

    quality = cfg.get("dataset_quality", {}) if isinstance(cfg.get("dataset_quality", {}), dict) else {}
    require_nominal = bool(quality.get("require_nominal_per_scene_time", True))
    keep_nominal = bool(quality.get("keep_nominal_even_if_quality_fails", True))
    min_accepted_required = int(quality.get("min_accepted_prefixes_per_scene_time", 0) or 0)
    if require_nominal:
        nominal_prefix = next((p for p in prefixes if p.macro_name == "nominal" or int(p.macro_id) == 0), None)
        if nominal_prefix is not None and sum(int(bool(s.is_nominal)) for s in selected) != 1:
            anchor_cfg = _cfg_with_artifact_mining(cfg, enable=False)
            nominal = materialize(nominal_prefix, anchor_cfg, enforce_quality=not keep_nominal, allow_filter_drop=False)
            if nominal is not None:
                _replace_or_append_sample(selected, nominal, max_accepted=max_accepted, macros_seen=macros_seen)
        selected = _dedupe_nominal(selected)

    if min_accepted_required > 0 and (max_accepted <= 0 or min_accepted_required <= max_accepted):
        used = {int(s.candidate_index) for s in selected}
        # First use already materialized deferred examples.
        for sample in deferred:
            if len(selected) >= min_accepted_required:
                break
            if int(sample.candidate_index) in used:
                continue
            _replace_or_append_sample(selected, sample, max_accepted=max_accepted, macros_seen=macros_seen)
            used.add(int(sample.candidate_index))
        # If a strict per-sample quality gate made the group underfull, keep a
        # small number of additional anchors so offline candidate selection is
        # still well-defined.
        if len(selected) < min_accepted_required:
            anchor_cfg = _cfg_with_artifact_mining(cfg, enable=False)
            for prefix in prefixes:
                if len(selected) >= min_accepted_required:
                    break
                if int(prefix.macro_id) in used:
                    continue
                sample = materialize(prefix, anchor_cfg, enforce_quality=False, allow_filter_drop=False)
                if sample is None:
                    continue
                _replace_or_append_sample(selected, sample, max_accepted=max_accepted, macros_seen=macros_seen)
                used.add(int(sample.candidate_index))

    if max_accepted > 0:
        selected = selected[:max_accepted]
    selected = _dedupe_nominal(selected)
    selected.sort(key=lambda s: int(s.candidate_index))
    assign_regimes(selected, history, cfg)
    return selected

def scenario_iterator(cfg: dict) -> Iterator[RawScenario]:
    source = str(cfg.get("data_source", "synthetic_artifact"))
    if source in {"synthetic", "synthetic_artifact"}:
        yield from iter_synthetic_scenarios(int(cfg.get("num_synthetic_scenarios", 4)), seed=int(cfg.get("seed", 0)), cfg=cfg, artifact=(source == "synthetic_artifact"))
    elif source == "womd":
        patterns = cfg.get("womd_patterns")
        if not patterns:
            raise ValueError("data_source=womd requires womd_patterns")
        if str(cfg.get("simulation_backend", "ocrap_surrogate")) == "waymax_closed_loop":
            from ocrap.data.waymax_loader import iter_waymax_womd_scenarios

            yield from iter_waymax_womd_scenarios(patterns, max_scenarios=cfg.get("max_scenarios"), parser_cfg=cfg)
        else:
            from ocrap.data.womd.scenario_parser import iter_womd_scenarios

            yield from iter_womd_scenarios(patterns, max_scenarios=cfg.get("max_scenarios"), parser_cfg=cfg)
    else:
        raise ValueError(f"Unknown data_source {source}")


def _sample_filename(scene_id: str, time_index: int, candidate_index: int) -> str:
    return f"{scene_id}_t{int(time_index):04d}_a{int(candidate_index):02d}.npz".replace("/", "_")


def _scene_time_key(scene_id: str, time_index: int) -> str:
    return f"{scene_id}_t{int(time_index):04d}".replace("/", "_")


def _scene_time_key_from_sample_name(name: str) -> str | None:
    if not name.endswith(".npz"):
        return None
    stem = name[:-4]
    idx = stem.rfind("_a")
    if idx <= 0:
        return None
    return stem[:idx]


def _expected_samples_per_scene_time(cfg: dict) -> int:
    max_accepted = _max_accepted_prefixes_per_scene_time(cfg)
    if max_accepted > 0:
        return max_accepted
    return max(0, int(cfg.get("num_candidate_prefixes", 24)))


def _estimated_total_samples(cfg: dict) -> int | None:
    max_scenarios = cfg.get("max_scenarios")
    if max_scenarios is None:
        return None
    return int(max_scenarios) * int(cfg.get("max_times_per_scenario", 8)) * _expected_samples_per_scene_time(cfg)


def _is_complete_npz(path: Path) -> bool:
    try:
        if not path.is_file() or path.stat().st_size <= 0:
            return False
        # Validate both the npz container and the OC-RAP sample schema.  A run
        # interrupted between partial/debug writes and manifest rewrite can leave
        # a readable .npz with only a few arrays.  Treating that file as resumable
        # makes --skip-existing permanently skip the deterministic sample name,
        # and diagnose/papercheck will later fail on missing required fields.
        with np.load(path, allow_pickle=True) as z:
            return bool(z.files) and not missing_fields(set(z.files))
    except (OSError, EOFError, ValueError):
        return False


def _quarantine_invalid_existing_sample(path: Path, sample_dir: Path) -> bool:
    """Move a corrupt/incomplete resumable sample out of samples/*.npz.

    The dataset readers intentionally scan ``samples/*.npz`` directly, not only
    manifest rows.  Keeping a bad file in that directory means a repaired build
    can still fail validation even after the replacement sample is written.
    """
    try:
        bad_dir = ensure_dir(sample_dir / "invalid_samples")
        target = bad_dir / path.name
        if target.exists():
            stem = path.stem
            suffix = path.suffix
            i = 1
            while True:
                candidate = bad_dir / f"{stem}.bad{i}{suffix}"
                if not candidate.exists():
                    target = candidate
                    break
                i += 1
        os.replace(path, target)
        return True
    except OSError:
        return False


def _scalar_from_npz(z, key: str, default: object = "") -> object:
    if key not in z.files:
        return default
    arr = np.asarray(z[key])
    if arr.shape == ():
        return arr.item()
    return arr.tolist()


def _manifest_regime_label(raw: object) -> str:
    if raw is None:
        return ""
    try:
        obj = json.loads(str(raw))
        if isinstance(obj, dict):
            return ";".join(k for k, v in obj.items() if v)
    except Exception:
        pass
    return str(raw)


def _manifest_row_from_npz(path: Path, out: Path) -> dict:
    with np.load(path, allow_pickle=True) as z:
        return {
            "path": str(path.relative_to(out)),
            "scene_id": _scalar_from_npz(z, "scene_id", ""),
            "original_scenario_id": _scalar_from_npz(z, "original_scenario_id", ""),
            "time_index": _scalar_from_npz(z, "time_index", ""),
            "candidate_index": _scalar_from_npz(z, "candidate_index", ""),
            "split_id": _scalar_from_npz(z, "split_id", ""),
            "is_nominal": int(_scalar_from_npz(z, "is_nominal", 0)),
            "r_orc_star": _scalar_from_npz(z, "r_orc_star", ""),
            "r_dep_star": _scalar_from_npz(z, "r_dep_star", ""),
            "oracle_gap_star": _scalar_from_npz(z, "oracle_gap_star", ""),
            "i_art_star": int(_scalar_from_npz(z, "i_art_star", 0)),
            "regime_label": _manifest_regime_label(_scalar_from_npz(z, "regime_label", "{}")),
        }


def _read_existing_manifest_rows(out: Path, existing_names: set[str]) -> list[dict]:
    manifest_path = out / "manifest.csv"
    if not manifest_path.exists():
        return []
    rows: list[dict] = []
    try:
        with manifest_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = Path(row.get("path", "")).name
                if name in existing_names:
                    rows.append({field: row.get(field, "") for field in MANIFEST_FIELDS})
    except Exception:
        return []
    return rows


def _bootstrap_existing_samples(out: Path, sample_dir: Path) -> tuple[set[str], dict[str, int], list[dict], int]:
    existing_names: set[str] = set()
    existing_paths: dict[str, Path] = {}
    existing_counts_by_scene_time: dict[str, int] = defaultdict(int)
    invalid_quarantined = 0
    for path in sample_dir.glob("*.npz"):
        if not _is_complete_npz(path):
            invalid_quarantined += int(_quarantine_invalid_existing_sample(path, sample_dir))
            continue
        existing_names.add(path.name)
        existing_paths[path.name] = path
        key = _scene_time_key_from_sample_name(path.name)
        if key is not None:
            existing_counts_by_scene_time[key] += 1

    manifest_rows = _read_existing_manifest_rows(out, existing_names)
    manifest_names = {Path(row.get("path", "")).name for row in manifest_rows}
    for name in sorted(existing_names - manifest_names):
        try:
            manifest_rows.append(_manifest_row_from_npz(existing_paths[name], out))
        except Exception:
            # A file can pass the container/schema check but still fail while
            # reading metadata from an older incompatible run.  Remove it from the
            # resumable set and quarantine it so direct samples/*.npz validation
            # will not see the bad file.
            existing_names.discard(name)
            invalid_quarantined += int(_quarantine_invalid_existing_sample(existing_paths[name], sample_dir))
            key = _scene_time_key_from_sample_name(name)
            if key is not None and existing_counts_by_scene_time.get(key, 0) > 0:
                existing_counts_by_scene_time[key] -= 1
    return existing_names, dict(existing_counts_by_scene_time), manifest_rows, invalid_quarantined


def _append_manifest_row(manifest_rows: list[dict], out: Path, sample: DatasetSample, path: Path) -> None:
    manifest_rows.append({
        "path": str(path.relative_to(out)),
        "scene_id": sample.scene_id,
        "original_scenario_id": sample.original_scenario_id,
        "time_index": sample.time_index,
        "candidate_index": sample.candidate_index,
        "split_id": sample.split_id,
        "is_nominal": int(sample.is_nominal),
        "r_orc_star": sample.r_orc_star,
        "r_dep_star": sample.r_dep_star,
        "oracle_gap_star": sample.oracle_gap_star,
        "i_art_star": int(sample.i_art_star),
        "regime_label": ";".join(k for k, v in sample.regime_label.items() if v),
    })


def _write_manifest_atomic(manifest_path: Path, rows: list[dict]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest_path.with_name(f".{manifest_path.name}.tmp")
    try:
        with tmp.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in MANIFEST_FIELDS})
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, manifest_path)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        finally:
            raise


def _sample_profile_row(sample: DatasetSample, sample_index: int) -> dict[str, Any]:
    timings = sample.diagnostics.get("build_timing_s", {}) if isinstance(sample.diagnostics, dict) else {}
    row = {
        "sample_index": int(sample_index),
        "scene_id": sample.scene_id,
        "time_index": int(sample.time_index),
        "candidate_index": int(sample.candidate_index),
        "macro_name": sample.prefix.macro_name,
        "num_futures": int(len(sample.futures)),
        "num_options": int(len(sample.recovery_options)),
        "r_orc_star": float(sample.r_orc_star),
        "r_dep_star": float(sample.r_dep_star),
        "i_art_star": int(sample.i_art_star),
        "waymax_teacher_rollouts_executed": _future_metadata_sum(sample, "waymax_teacher_rollouts_executed"),
        "waymax_teacher_metric_cache_hits": _future_metadata_sum(sample, "waymax_teacher_metric_cache_hits"),
        "waymax_teacher_screened_hybrid": int(_future_metadata_any(sample, "waymax_teacher_screened_hybrid")),
    }
    for src, dst in [
        ("total", "total_s"),
        ("future_generation", "future_generation_s"),
        ("teacher_margins", "teacher_margins_s"),
        ("root_clustering", "root_clustering_s"),
        ("observation", "observation_s"),
        ("ocmero", "ocmero_s"),
    ]:
        row[dst] = float(timings.get(src, 0.0))
    return row


def _append_csv_rows(path: Path, rows: list[dict[str, Any]], fields: list[str], *, fsync_file: bool = False) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
        f.flush()
        if fsync_file:
            os.fsync(f.fileno())


def _append_profile_row(profile_path: Path, sample: DatasetSample, sample_index: int) -> None:
    """Append one build-time row for external watch diagnostics."""
    _append_csv_rows(profile_path, [_sample_profile_row(sample, sample_index)], PROFILE_FIELDS, fsync_file=True)


def _write_running_status(out: Path, **kwargs) -> None:
    """Write a lightweight status file while a long build is still running."""
    try:
        write_json({k: v for k, v in kwargs.items()}, out / "dataset_status.json")
    except Exception:
        # Status reporting must never make dataset generation fail.
        return


def _count_splits(rows: list[dict]) -> dict[str, int]:
    split_counts: dict[str, int] = {"train": 0, "val": 0, "calibration": 0, "test": 0}
    for row in rows:
        split_id = str(row.get("split_id", ""))
        if not split_id:
            continue
        split_counts[split_id] = split_counts.get(split_id, 0) + 1
    return split_counts


def build_dataset(output_dir: str | Path, cfg: dict, skip_existing: bool = False) -> dict:
    # Use a shallow copy so live profiling counters/private paths do not leak back
    # into caller-owned config objects, while nested user config remains intact.
    cfg = dict(cfg)
    build_t0 = _now()
    out = ensure_dir(output_dir)
    sample_dir = ensure_dir(out / "samples")
    skip_existing = bool(skip_existing or cfg.get("skip_existing", False))
    existing_sample_names: set[str] = set()
    existing_counts_by_scene_time: dict[str, int] = {}
    manifest_rows: list[dict] = []
    invalid_existing_samples_quarantined = 0
    if skip_existing:
        existing_sample_names, existing_counts_by_scene_time, manifest_rows, invalid_existing_samples_quarantined = _bootstrap_existing_samples(out, sample_dir)
    split_counts: dict[str, int] = _count_splits(manifest_rows)
    total = len(manifest_rows)
    initial_existing_total = len(existing_sample_names)
    skipped_existing_samples = 0
    skipped_existing_scene_time_groups = 0
    new_samples_written = 0
    raw_scenarios_seen = 0
    scene_time_groups = 0
    skipped_no_planning_times = 0
    raw_scene_ids: set[str] = set()
    raw_iter = iter(scenario_iterator(cfg))
    progress_bar = None
    profile_path = out / "build_profile.csv"
    scene_profile_path = out / "build_scene_time_profile.csv"
    live_profile_path = out / "build_profile_live.csv"
    stage_profile_path = out / "build_stage_profile.json"
    stage_totals: dict[str, float] = {}
    progress_mode = "samples" if skip_existing else "prefixes"
    prof_cfg = _profiling_cfg(cfg)
    profiling = _profiling_enabled(cfg)
    profile_flush_scene_times = max(1, int(prof_cfg.get("profile_flush_scene_times", 1)))
    profile_csv_fsync = bool(prof_cfg.get("profile_csv_fsync", False))
    sample_profile_rows: list[dict[str, Any]] = []
    scene_profile_rows: list[dict[str, Any]] = []
    if profiling:
        cfg["_profile_live_path"] = str(live_profile_path)
        cfg["_profile_live_sample_counter"] = 0
    if profiling and not skip_existing:
        for stale in [profile_path, scene_profile_path, live_profile_path, stage_profile_path]:
            try:
                stale.unlink(missing_ok=True)
            except Exception:
                pass
    io = _io_cfg(cfg)
    compress_npz = bool(io.get("compress_npz", cfg.get("compress_npz", True)))
    fsync_npz = bool(io.get("fsync_npz", cfg.get("fsync_npz", True)))
    if bool(cfg.get("progress", True)):
        try:
            from tqdm.auto import tqdm

            if skip_existing:
                total_samples = _estimated_total_samples(cfg)
                if total_samples is not None:
                    total_samples = max(int(total_samples), initial_existing_total)
                    initial = min(initial_existing_total, total_samples)
                else:
                    initial = initial_existing_total
                progress_bar = tqdm(total=total_samples, initial=initial, desc="OC-RAP build samples", unit="sample")
            else:
                max_scenarios = cfg.get("max_scenarios")
                total_prefixes = None
                if max_scenarios is not None:
                    total_prefixes = int(max_scenarios) * int(cfg.get("max_times_per_scenario", 8)) * int(cfg.get("num_candidate_prefixes", 24))
                progress_bar = tqdm(total=total_prefixes, desc="OC-RAP build prefixes", unit="prefix")
        except Exception:
            progress_bar = None

    def _progress(n: int) -> None:
        if progress_bar is not None:
            progress_bar.update(int(n))

    def _flush_profiles() -> None:
        nonlocal sample_profile_rows, scene_profile_rows
        if not profiling:
            return
        _append_csv_rows(profile_path, sample_profile_rows, PROFILE_FIELDS, fsync_file=profile_csv_fsync)
        _append_csv_rows(scene_profile_path, scene_profile_rows, SCENE_PROFILE_FIELDS, fsync_file=profile_csv_fsync)
        sample_profile_rows = []
        scene_profile_rows = []

    def _write_stage_profile() -> None:
        if not profiling:
            return
        elapsed = max(_now() - build_t0, 1e-9)
        payload = {
            "elapsed_wall_s": float(elapsed),
            "num_samples_total": int(total),
            "new_samples_written": int(new_samples_written),
            "raw_scenarios_seen": int(raw_scenarios_seen),
            "scene_time_groups": int(scene_time_groups),
            "samples_per_hour": float(new_samples_written / elapsed * 3600.0),
            "stage_totals_s": {k: float(v) for k, v in sorted(stage_totals.items())},
            "io": {"compress_npz": bool(compress_npz), "fsync_npz": bool(fsync_npz)},
            "profile_csv": str(profile_path),
            "profile_live_csv": str(live_profile_path),
            "scene_profile_csv": str(scene_profile_path),
        }
        write_json(payload, stage_profile_path)

    try:
        while True:
            next_t0 = _now()
            try:
                raw = next(raw_iter)
            except StopIteration:
                break
            _stage_add(stage_totals, "scenario_next_s", _now() - next_t0)
            raw_scenarios_seen += 1
            raw_scene_ids.add(str(raw.scenario_id))
            split_id = resolve_split(raw.scenario_id, cfg)
            select_t0 = _now()
            times, reasons_by_time = select_planning_times_with_reasons(raw, cfg)
            _stage_add(stage_totals, "planning_time_selection_s", _now() - select_t0)
            _profile_log(cfg, f"scenario {raw_scenarios_seen} id={raw.scenario_id} split={split_id} selected_times={len(times)}")
            if not times:
                skipped_no_planning_times += 1
                continue
            for t in times:
                scene_time_t0 = _now()
                scene_time_groups += 1
                hist_t0 = _now()
                history = construct_history(raw, t, cfg)
                construct_history_s = _now() - hist_t0
                _stage_add(stage_totals, "construct_history_s", construct_history_s)
                _profile_log(cfg, f"scene_time start scene={history.scene_id} t={int(t)} group={scene_time_groups}")
                # Retain only the reasons that actually selected this planning instant.
                history.metadata["time_sampling_reasons"] = reasons_by_time.get(int(t), ["uniform"])
                if skip_existing:
                    key = _scene_time_key(history.scene_id, int(t))
                    expected = _expected_samples_per_scene_time(cfg)
                    if expected > 0 and existing_counts_by_scene_time.get(key, 0) >= expected:
                        skipped_existing_scene_time_groups += 1
                        _profile_log(cfg, f"scene_time skip-existing scene={history.scene_id} t={int(t)} existing={existing_counts_by_scene_time.get(key, 0)} expected={expected}")
                        continue
                build_samples_t0 = _now()
                samples = build_samples_for_history(history, split_id, cfg, progress_callback=None if skip_existing else _progress)
                build_samples_s = _now() - build_samples_t0
                _stage_add(stage_totals, "build_samples_s", build_samples_s)
                npz_serialize_s = 0.0
                npz_write_s = 0.0
                for sample in samples:
                    fname = _sample_filename(sample.scene_id, sample.time_index, sample.candidate_index)
                    path = sample_dir / fname
                    if skip_existing and fname in existing_sample_names:
                        skipped_existing_samples += 1
                        continue
                    ser_t0 = _now()
                    sample_arrays = sample.to_npz_dict()
                    npz_serialize_s += _now() - ser_t0
                    write_t0 = _now()
                    np_savez(path, compressed=compress_npz, fsync=fsync_npz, **sample_arrays)
                    npz_write_s += _now() - write_t0
                    _append_manifest_row(manifest_rows, out, sample, path)
                    existing_sample_names.add(fname)
                    key = _scene_time_key(sample.scene_id, sample.time_index)
                    existing_counts_by_scene_time[key] = existing_counts_by_scene_time.get(key, 0) + 1
                    split_counts[sample.split_id] = split_counts.get(sample.split_id, 0) + 1
                    total += 1
                    new_samples_written += 1
                    if profiling:
                        sample_profile_rows.append(_sample_profile_row(sample, total))
                    if skip_existing:
                        _progress(1)
                    if profiling and bool(prof_cfg.get("log_writes", False)):
                        _profile_log(cfg, f"wrote sample #{total} path={path}")
                _stage_add(stage_totals, "npz_serialize_s", npz_serialize_s)
                _stage_add(stage_totals, "npz_write_s", npz_write_s)
                manifest_checkpoint_s = 0.0
                if profiling or bool(cfg.get("checkpoint_manifest_each_scene_time", False)):
                    manifest_t0 = _now()
                    _write_manifest_atomic(out / "manifest.csv", manifest_rows)
                    manifest_checkpoint_s = _now() - manifest_t0
                    _stage_add(stage_totals, "manifest_checkpoint_s", manifest_checkpoint_s)
                    _write_running_status(
                        out,
                        num_samples=int(total),
                        split_counts=split_counts,
                        sample_dir=str(sample_dir),
                        raw_scenarios_seen=int(raw_scenarios_seen),
                        scene_time_groups=int(scene_time_groups),
                        new_samples_written=int(new_samples_written),
                        skipped_existing_samples=int(skipped_existing_samples),
                        skipped_existing_scene_time_groups=int(skipped_existing_scene_time_groups),
                        progress_mode=progress_mode,
                        profile_csv=str(profile_path) if profiling else "",
                        profile_live_csv=str(live_profile_path) if profiling else "",
                        scene_profile_csv=str(scene_profile_path) if profiling else "",
                        stage_profile_json=str(stage_profile_path) if profiling else "",
                        stage_totals_s={k: float(v) for k, v in sorted(stage_totals.items())},
                    )
                scene_time_dt = _now() - scene_time_t0
                if profiling or scene_time_dt >= float(prof_cfg.get("slow_scene_time_s", 120.0)):
                    _profile_log(
                        cfg,
                        "scene_time done scene=%s t=%s samples=%d elapsed=%.2fs build=%.2fs write=%.2fs teacher_sum=%.2fs"
                        % (
                            history.scene_id,
                            int(t),
                            len(samples),
                            scene_time_dt,
                            build_samples_s,
                            npz_write_s,
                            _sample_timing_sum(samples, "teacher_margins"),
                        ),
                    )
                if profiling:
                    scene_profile_rows.append({
                        "raw_scenario_index": int(raw_scenarios_seen),
                        "scene_time_group": int(scene_time_groups),
                        "scene_id": history.scene_id,
                        "original_scenario_id": history.original_scenario_id,
                        "time_index": int(t),
                        "split_id": split_id,
                        "num_selected_samples": int(len(samples)),
                        "construct_history_s": float(construct_history_s),
                        "build_samples_s": float(build_samples_s),
                        "sample_compute_sum_s": _sample_timing_sum(samples, "total"),
                        "sample_future_generation_sum_s": _sample_timing_sum(samples, "future_generation"),
                        "sample_teacher_margins_sum_s": _sample_timing_sum(samples, "teacher_margins"),
                        "sample_observation_sum_s": _sample_timing_sum(samples, "observation"),
                        "sample_root_clustering_sum_s": _sample_timing_sum(samples, "root_clustering"),
                        "sample_ocmero_sum_s": _sample_timing_sum(samples, "ocmero"),
                        "npz_serialize_s": float(npz_serialize_s),
                        "npz_write_s": float(npz_write_s),
                        "manifest_checkpoint_s": float(manifest_checkpoint_s),
                        "scene_time_total_s": float(scene_time_dt),
                    })
                    if scene_time_groups % profile_flush_scene_times == 0:
                        _flush_profiles()
                        _write_stage_profile()
    finally:
        if progress_bar is not None:
            progress_bar.close()
    _flush_profiles()
    manifest_path = out / "manifest.csv"
    final_manifest_t0 = _now()
    _write_manifest_atomic(manifest_path, manifest_rows)
    _stage_add(stage_totals, "final_manifest_s", _now() - final_manifest_t0)
    elapsed_wall_s = max(_now() - build_t0, 1e-9)
    summary = {
        "num_samples": total,
        "split_counts": split_counts,
        "manifest": str(manifest_path),
        "sample_dir": str(sample_dir),
        "skip_existing": bool(skip_existing),
        "existing_samples_seen": int(initial_existing_total),
        "invalid_existing_samples_quarantined": int(invalid_existing_samples_quarantined),
        "skipped_existing_samples": int(skipped_existing_samples),
        "skipped_existing_scene_time_groups": int(skipped_existing_scene_time_groups),
        "new_samples_written": int(new_samples_written),
        "progress_mode": progress_mode,
        "raw_scenarios_seen": int(raw_scenarios_seen),
        "scene_time_groups": int(scene_time_groups),
        "skipped_no_planning_times": int(skipped_no_planning_times),
        "unique_raw_scene_ids": int(len(raw_scene_ids)),
        "elapsed_wall_s": float(elapsed_wall_s),
        "samples_per_hour": float(new_samples_written / elapsed_wall_s * 3600.0),
        "seconds_per_new_sample": float(elapsed_wall_s / max(new_samples_written, 1)),
        "profile_csv": str(profile_path) if profiling else "",
        "profile_live_csv": str(live_profile_path) if profiling else "",
        "scene_profile_csv": str(scene_profile_path) if profiling else "",
        "stage_profile_json": str(stage_profile_path) if profiling else "",
        "stage_totals_s": {k: float(v) for k, v in sorted(stage_totals.items())},
        "io": {"compress_npz": bool(compress_npz), "fsync_npz": bool(fsync_npz)},
        "dataset_quality": cfg.get("dataset_quality", {}),
        "artifact": cfg.get("artifact", {}),
        "waymax": cfg.get("waymax", {}),
    }
    summary_t0 = _now()
    write_json(summary, out / "dataset_summary.json")
    _stage_add(stage_totals, "summary_write_s", _now() - summary_t0)
    _write_stage_profile()
    return summary

