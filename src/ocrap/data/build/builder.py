from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from ocrap.algorithms.ocmero import oc_mero
from ocrap.data.schema import DatasetSample, RawScenario
from ocrap.data.serialization import ensure_dir, np_savez, write_json
from ocrap.data.split import scenario_split
from ocrap.planning.prefix_generation import generate_candidate_prefixes
from ocrap.roots import aggregate_root_margins, cluster_roots, future_trajectory_signature
from ocrap.simulation.futures import generate_counterfactual_futures
from ocrap.simulation.observation import compatibility_labels, render_observation, unknown_ratio_in_corridor
from ocrap.simulation.teacher import compute_future_option_margins, default_recovery_options, option_valid_mask

from .history import construct_history
from .regimes import assign_regimes
from .synthetic import iter_synthetic_scenarios


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
    ordered: list[int] = []
    seen: set[int] = set()
    # Priority order matters when max_times_per_scenario is small: keep the
    # interaction-biased frame instead of accidentally discarding it by sorting
    # the union and taking the earliest timestamp.
    for seq in (biased_times, uniform_times, [int(t) for _, t, _ in scored]):
        for t in seq:
            if t in seen:
                continue
            ordered.append(int(t))
            seen.add(int(t))
            if len(ordered) >= max_times:
                break
        if len(ordered) >= max_times:
            break
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
    futures = generate_counterfactual_futures(history, prefix, cfg)
    future_probs = np.asarray([f.prior for f in futures], dtype=np.float32)
    future_probs = future_probs / max(float(future_probs.sum()), 1e-8)
    M_future, teacher_diags = compute_future_option_margins(history, prefix, futures, options, cfg)
    root = cluster_roots(M_future, future_probs, futures, cfg)
    M_star = aggregate_root_margins(M_future, root.assignments, future_probs, K, cfg)
    root_future_signature = future_trajectory_signature(futures, root.assignments, future_probs, K, width=int(cfg.get("model", {}).get("d_future_signature", 32)))
    obs_by_future = [render_observation(history, prefix, f, cfg) for f in futures]
    within_disp = _compute_within_root_dispersion(root.assignments, obs_by_future, K, cfg)
    observations = []
    for k in range(K):
        rep = int(root.representative_indices[k]) if root.root_valid[k] and root.representative_indices[k] >= 0 else 0
        observations.append(obs_by_future[rep])
    Y, C, Dobs = compatibility_labels(observations, cfg)
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
    gamma_orc = float(cfg.get("artifact", {}).get("gamma_orc", 0.0))
    gamma_dep = float(cfg.get("artifact", {}).get("gamma_dep", 0.0))
    return DatasetSample(
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
        },
    )


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
    macro_diversity_first = bool((cfg.get("dataset_quality", {}) or {}).get("macro_diversity_first", True))

    selected: list[DatasetSample] = []
    deferred: list[DatasetSample] = []
    n_art = 0
    n_non = 0
    macros_seen: set[str] = set()

    for a_idx, prefix in enumerate(prefixes):
        if mode == "filter" and not _artifact_pair_attempt_is_possible(history, prefix, cfg):
            if progress_callback is not None:
                progress_callback(1)
            continue
        if max_accepted > 0 and len(selected) >= max_accepted and not (mode == "balanced" and (n_art < min_art or n_non < min_non)):
            if progress_callback is not None:
                progress_callback(len(prefixes) - a_idx)
            break

        sample = _materialize_sample(history, split_id, prefix, a_idx, cfg, options, option_valid, K)
        has_pair = _has_complete_artifact_pair(sample)
        sample.diagnostics["complete_artifact_pair"] = bool(has_pair)
        if mode == "filter" and not has_pair:
            if progress_callback is not None:
                progress_callback(1)
            continue

        is_art = bool(sample.i_art_star or has_pair)
        keep_now = True
        if mode == "balanced":
            if is_art and max_art > 0 and n_art >= max_art:
                keep_now = False
            if (not is_art) and max_non > 0 and n_non >= max_non:
                keep_now = False
            if macro_diversity_first and prefix.macro_name in macros_seen and max_accepted > 0 and len(selected) >= max_accepted:
                keep_now = False
        elif max_accepted > 0 and len(selected) >= max_accepted:
            keep_now = False

        if keep_now:
            selected.append(sample)
            macros_seen.add(prefix.macro_name)
            n_art += int(is_art)
            n_non += int(not is_art)
        else:
            deferred.append(sample)
        if progress_callback is not None:
            progress_callback(1)

    if max_accepted > 0 and len(selected) < max_accepted:
        for sample in deferred:
            if len(selected) >= max_accepted:
                break
            selected.append(sample)
    if max_accepted > 0:
        selected = selected[:max_accepted]
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


def build_dataset(output_dir: str | Path, cfg: dict) -> dict:
    out = ensure_dir(output_dir)
    sample_dir = ensure_dir(out / "samples")
    manifest_rows: list[dict] = []
    split_counts: dict[str, int] = {"train": 0, "val": 0, "calibration": 0, "test": 0}
    total = 0
    raw_scenarios_seen = 0
    scene_time_groups = 0
    skipped_no_planning_times = 0
    raw_iter = scenario_iterator(cfg)
    prefix_bar = None
    if bool(cfg.get("progress", True)):
        try:
            from tqdm.auto import tqdm

            max_scenarios = cfg.get("max_scenarios")
            total_prefixes = None
            if max_scenarios is not None:
                total_prefixes = int(max_scenarios) * int(cfg.get("max_times_per_scenario", 8)) * int(cfg.get("num_candidate_prefixes", 24))
            prefix_bar = tqdm(total=total_prefixes, desc="OC-RAP build prefixes", unit="prefix")
        except Exception:
            prefix_bar = None
    def _progress(n: int) -> None:
        if prefix_bar is not None:
            prefix_bar.update(int(n))
    try:
        for raw in raw_iter:
            raw_scenarios_seen += 1
            split_id = scenario_split(raw.scenario_id, cfg.get("split_ratios"))
            times, reasons_by_time = select_planning_times_with_reasons(raw, cfg)
            if not times:
                skipped_no_planning_times += 1
                continue
            for t in times:
                scene_time_groups += 1
                history = construct_history(raw, t, cfg)
                # Retain only the reasons that actually selected this planning instant.
                history.metadata["time_sampling_reasons"] = reasons_by_time.get(int(t), ["uniform"])
                samples = build_samples_for_history(history, split_id, cfg, progress_callback=_progress)
                for sample in samples:
                    fname = f"{sample.scene_id}_t{sample.time_index:04d}_a{sample.candidate_index:02d}.npz".replace("/", "_")
                    path = sample_dir / fname
                    np_savez(path, **sample.to_npz_dict())
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
                    split_counts[sample.split_id] = split_counts.get(sample.split_id, 0) + 1
                    total += 1
    finally:
        if prefix_bar is not None:
            prefix_bar.close()
    manifest_path = out / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        fields = list(manifest_rows[0].keys()) if manifest_rows else ["path"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_rows)
    summary = {
        "num_samples": total,
        "split_counts": split_counts,
        "manifest": str(manifest_path),
        "sample_dir": str(sample_dir),
        "raw_scenarios_seen": int(raw_scenarios_seen),
        "scene_time_groups": int(scene_time_groups),
        "skipped_no_planning_times": int(skipped_no_planning_times),
    }
    write_json(summary, out / "dataset_summary.json")
    return summary
