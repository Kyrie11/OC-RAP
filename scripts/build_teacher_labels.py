#!/usr/bin/env python
from __future__ import annotations

# Allow running scripts directly from a source checkout without pip install -e .
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import argparse
import json
from pathlib import Path
import time

import numpy as np

from ocrap.utils.datatypes import EgoState, ActorState, MapFeatures, RouteInfo
from ocrap.proposals.action_lattice import generate_lattice_actions, actions_to_tensors
from ocrap.proposals.recovery_options import generate_recovery_options, options_to_tensors
from ocrap.teacher.root_modes import generate_root_modes, mode_seed_params_array, normalized_mode_uncertainty, MODE_SLOT_SEMANTICS
from ocrap.teacher.rollout import synthetic_rollout
from ocrap.teacher.evidence_labels import evidence_from_trace, scene_uncertainty_from_action, combine_uncertainty
from ocrap.teacher.dataset_writer import ShardedDatasetWriter, read_dataset, write_dataset
from ocrap.evaluation.metrics import weighted_lcvar_np, upper_tail_cvar_np
from ocrap.teacher.observation_classes import build_obs_equivalence, beta_from_obs_equiv, class_consistent_witness, post_prefix_observation_signature
from ocrap.utils.progress import tqdm
from scripts._common import load_config


def _append_progress(progress_file: str | None, record: dict) -> None:
    """Append one JSONL progress record for external parallel supervisors.

    Each build_teacher_labels process writes to its own progress file.  A parent
    launcher can count these records and render a single global tqdm bar, while
    the worker's normal stdout/stderr can still be redirected to a log file.
    """
    if not progress_file:
        return
    p = Path(progress_file)
    p.parent.mkdir(parents=True, exist_ok=True)
    rec = {**record, "time": time.time()}
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()


def load_root(path: Path):
    obj = json.loads(path.read_text())
    ego = EgoState(**obj["ego_state"])
    actors = [ActorState(**a) for a in obj["actor_states"]]
    mfobj = obj["map_features"]
    mf = MapFeatures(
        [np.asarray(p, dtype=np.float32) for p in mfobj.get("drivable_polygons", [])],
        [np.asarray(p, dtype=np.float32) for p in mfobj.get("lane_centerlines", [])],
        [np.asarray(p, dtype=np.float32) for p in mfobj.get("lane_boundaries", [])],
        [np.asarray(p, dtype=np.float32) for p in mfobj.get("static_obstacles", [])],
        float(mfobj.get("speed_limit_mps", 13.9)),
    )
    robj = obj["route_info"]
    route = RouteInfo(np.asarray(robj["waypoints"], dtype=np.float32), np.asarray(robj.get("command_ids", np.zeros(len(robj["waypoints"]))), dtype=np.int64), float(robj.get("speed_limit_mps", 13.9)))
    return obj, ego, actors, mf, route


def _select_ids(root_dir: Path, split: str, max_roots: int | None) -> list[str]:
    if (root_dir / "splits.json").exists() and split != "all":
        ids = json.loads((root_dir / "splits.json").read_text()).get(split, [])
    else:
        ids = sorted(p.stem for p in root_dir.glob("*.json") if p.name not in ("metadata.json", "splits.json"))
    return ids[:max_roots] if max_roots else ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--root-dir", default="data/ocrap/roots_raw")
    ap.add_argument("--bev-dir", default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-roots", type=int, default=None)
    ap.add_argument("--rollout-backend", choices=["auto", "synthetic", "metadrive"], default="auto")
    ap.add_argument("--scenario-dir", default=None, help="ScenarioNet database directory for real MetaDrive rollouts; root metadata scenario_dir is used per-root when present.")
    ap.add_argument("--metadrive-reactive-traffic", type=str, default="true")
    ap.add_argument("--allow-temporal-root-rollout", action="store_true", help="Allow real MetaDrive rollout on roots sampled at multiple ticks from the same scenario. This is unsafe unless your MetaDrive runner restores the exact root tick; default is to fail fast for paper-final labels.")
    ap.add_argument("--disable-root-alignment-check", action="store_true", help="Debug only: do not verify that ScenarioEnv reset matches the stored root ego pose.")
    ap.add_argument("--disable-root-time-replay", action="store_true", help="Debug only: do not replay the stored WOMD/ScenarioNet ego history before counterfactual rollout. Without this, ScenarioEnv starts from tick 0 while WOMD roots are usually current_time_index=10.")
    ap.add_argument("--disable-root-state-restore", action="store_true", help="Debug only: do not snap the controllable MetaDrive SDC to the stored root pose after bounded root-time replay drift.")
    ap.add_argument("--root-state-restore-max-m", type=float, default=25.0, help="Maximum pre-snap replay drift for root-state restore. Larger drift is treated as scenario/coordinate mismatch.")
    ap.add_argument("--alignment-tolerance-m", type=float, default=5.0)
    ap.add_argument("--shard-size", type=int, default=4, help="Number of roots per label shard. Keep small because BEV is large.")
    ap.add_argument("--compress-shards", action="store_true", help="Use np.savez_compressed per shard. Saves disk but can be much slower.")
    ap.add_argument("--single-npz", action="store_true", help="Legacy mode: materialize all labels/BEV in RAM and write one arrays.npz. Only for tiny debug runs.")
    ap.add_argument("--inner-progress", action="store_true", help="Show nested action-level progress for long real MetaDrive teacher generation.")
    ap.add_argument("--root-start", type=int, default=0, help="Start index within the selected split. Useful for parallel CPU sharding.")
    ap.add_argument("--root-end", type=int, default=None, help="Exclusive end index within the selected split. Useful for parallel CPU sharding.")
    ap.add_argument("--root-stride", type=int, default=1, help="Keep every Nth root after root-start/root-end. Useful for parallel CPU sharding.")
    ap.add_argument("--progress-file", default=None, help="Optional JSONL file updated once per completed root. Used by the parallel launcher to show a single tqdm bar.")
    ap.add_argument("--no-reuse-metadrive-env", action="store_true", help="Debug fallback: create a fresh ScenarioEnv for every rollout. Default reuses one env per root and calls reset(), which is much faster.")
    ap.add_argument("--allow-json-only-hybrid-stress", action="store_true", help="Debug only: permit hybrid stress roots whose added actors are not injected into ScenarioEnv.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    K_raw = int(cfg.get("planner", {}).get("K_raw", 32 if cfg.get("implementation_level") == "mvp" else 64))
    K = int(cfg.get("planner", {}).get("K", 16 if cfg.get("implementation_level") == "mvp" else 32))
    L = int(cfg.get("planner", {}).get("L", 6 if cfg.get("implementation_level") == "mvp" else 12))
    M = int(cfg.get("planner", {}).get("M", 4 if cfg.get("implementation_level") == "mvp" else 8))
    H_p = int(cfg.get("planner", {}).get("H_p", 10))
    H_r = int(cfg.get("planner", {}).get("H_r", 25))
    dt = float(cfg.get("planner", {}).get("dt", 0.2))

    root_dir = Path(args.root_dir)
    root_meta = {}
    meta_path = root_dir / "metadata.json"
    if meta_path.exists():
        root_meta = json.loads(meta_path.read_text())
    rollout_backend = args.rollout_backend
    if rollout_backend == "auto":
        rollout_backend = "synthetic" if root_meta.get("is_synthetic", True) else "metadrive"

    md_runner = None
    if rollout_backend == "metadrive":
        from ocrap.teacher.metadrive_rollout import MetaDriveRolloutRunner
        scenario_dir = args.scenario_dir or root_meta.get("scenario_dir")
        if not scenario_dir:
            raise ValueError("--rollout-backend metadrive requires --scenario-dir or root metadata scenario_dir")
        multi_tick_roots = int(root_meta.get("max_samples_per_log_last_run", 1) or 1) > 1 or bool(root_meta.get("temporal_roots_require_state_restore_for_metadrive_rollout", False))
        if multi_tick_roots and not args.allow_temporal_root_rollout:
            raise ValueError(
                "This root set was collected with max-samples-per-log > 1. The current real MetaDrive "
                "rollout runner resets ScenarioEnv by scenario index and cannot guarantee restoration to each "
                "stored _tXXX root tick. For paper-final real teacher labels, re-run "
                "collect_metadrive_roots.py with --max-samples-per-log 1. Use "
                "--allow-temporal-root-rollout only for debug after independently verifying root-time restore."
            )
        md_runner = MetaDriveRolloutRunner(
            scenario_dir=str(scenario_dir),
            reactive_traffic=args.metadrive_reactive_traffic.lower() in ("1", "true", "yes"),
            # Allowing temporal roots should not weaken alignment validation.  It
            # only permits roots sampled at non-default ticks; the strict post-
            # replay root-pose check remains on unless explicitly disabled.
            strict_root_alignment=not bool(args.disable_root_alignment_check),
            alignment_tolerance_m=float(args.alignment_tolerance_m),
            restore_root_time=not bool(args.disable_root_time_replay),
            restore_root_state=not bool(args.disable_root_state_restore),
            root_state_restore_max_m=float(args.root_state_restore_max_m),
        )

    all_ids = _select_ids(root_dir, args.split, args.max_roots)
    root_end = len(all_ids) if args.root_end is None else min(len(all_ids), int(args.root_end))
    root_start = max(0, int(args.root_start))
    root_stride = max(1, int(args.root_stride))
    if root_start > root_end:
        raise ValueError(f"--root-start ({root_start}) must be <= --root-end ({root_end})")
    ids = all_ids[root_start:root_end:root_stride]
    if args.progress_file:
        pf = Path(args.progress_file)
        pf.parent.mkdir(parents=True, exist_ok=True)
        pf.write_text("", encoding="utf-8")

    if args.bev_dir:
        try:
            bev_arrays, bev_meta = read_dataset(args.bev_dir)
        except Exception as exc:
            raise RuntimeError(f"Failed to read BEV dataset from {args.bev_dir}") from exc
    else:
        bev_arrays, bev_meta = {}, {}
    bev_index_by_root = {}
    if "bev" in bev_arrays:
        if "root_ids" not in bev_arrays:
            raise ValueError("BEV dataset is missing root_ids; refusing positional root/BEV alignment.")
        bev_root_ids = np.asarray(bev_arrays["root_ids"]).astype(str)
        bev_index_by_root = {str(r): j for j, r in enumerate(bev_root_ids)}

    is_synthetic = bool(root_meta.get("is_synthetic", rollout_backend != "metadrive"))
    implementation_level = str(cfg.get("implementation_level", "mvp"))
    metadata = {
        "dataset_version": "metadrive_recovery_v1_real" if rollout_backend == "metadrive" else "metadrive_recovery_v0_synthetic",
        "split": args.split,
        "split_by": "root_scene_id",
        "implementation_level": implementation_level,
        "K": K,
        "L": L,
        "M": M,
        "H_p": H_p,
        "H_r": H_r,
        "dt": dt,
        "mode_slot_semantics": MODE_SLOT_SEMANTICS[:M],
        "mode_alignment": "fixed_semantic_index",
        "root_shared_mode_is_latent_context_not_open_loop_trajectory": True,
        "root_backend": root_meta.get("backend", "unknown"),
        "rollout_backend": rollout_backend,
        "scenario_dir": args.scenario_dir or root_meta.get("scenario_dir"),
        "is_synthetic": is_synthetic,
        "paper_final_ready": rollout_backend == "metadrive" and not is_synthetic and implementation_level == "final",
        "paper_final_ready_note": (
            "diagnostic/mvp/paper_check configs are for pipeline validation and idea debugging; use final config, full splits, "
            "and post-generation label-health checks for paper tables."
            if implementation_level != "final" else "requires post-generation label-health validation"
        ),
        "num_roots": len(ids),
        "num_roots_full_selected_split": len(all_ids),
        "root_start": root_start,
        "root_end": root_end,
        "root_stride": root_stride,
        "selected_root_count": len(ids),
        "bev_dir": args.bev_dir,
        "bev_metadata": bev_meta,
        "channel_names": bev_meta.get("channel_names", []),
        "format_note": "sharded_npz keeps memory bounded; use read_dataset() for lazy loading.",
        "allow_temporal_root_rollout": bool(args.allow_temporal_root_rollout),
        "root_alignment_check": not bool(args.disable_root_alignment_check),
        "alignment_tolerance_m": float(args.alignment_tolerance_m),
        "root_time_replay": rollout_backend == "metadrive" and not bool(args.disable_root_time_replay),
    }

    def build_one(idx: int, rid: str) -> dict:
        obj, ego, actors, mf, route = load_root(root_dir / f"{rid}.json")
        if rollout_backend == "metadrive" and (obj.get("scenario_data", {}) or {}).get("hybrid_stress_requires_simulator_injection", False) and not args.allow_json_only_hybrid_stress:
            raise RuntimeError(
                f"Root {rid} is a JSON-level hybrid stress perturbation. Its injected actors are not guaranteed to exist in MetaDrive ScenarioEnv. "
                "Write the stress actor into the ScenarioNet file or spawn/control it in the rollout runner; use --allow-json-only-hybrid-stress only for diagnostics."
            )
        actions = generate_lattice_actions(ego, route, mf, K_raw=K_raw, K=K, H_p=H_p, dt=dt)
        opts = generate_recovery_options(actions, route, mf, L=L, H_r=H_r, dt=dt)
        at = actions_to_tensors(actions)
        ot = options_to_tensors(opts)
        modes = generate_root_modes(int(obj["seed"]), M=M)
        mode_probs = np.ones(M, dtype=np.float32) / M
        P = np.zeros((K, L, M), np.float32)
        Praw = np.zeros_like(P)
        G = np.zeros_like(P)
        C = np.zeros_like(P)
        Kdef = np.zeros_like(P)
        margins = {name: np.zeros_like(P) for name in ["margin_option", "M_path_raw", "M_path_rec", "M_path_pre_no_first_contact", "M_secondary", "M_return", "M_ctrl", "M_post"]}
        g_star = np.zeros((K, L, M, 9), np.float32)
        spec_margin_star = np.zeros((K, L, M, 3), np.float32)
        spec_id_star = np.zeros((K, L, M), np.int64)
        # c_rule is generated per option first, then reduced through the
        # observation-consistent witness.  Initializing the action-level label
        # directly and taking max over invalid tokens made almost every action
        # look rule-violating whenever any padded/invalid token was present.
        c_rule_option = np.full((K, L, M), np.inf, np.float32)
        c_rule_star = np.zeros((K, M), np.float32)
        Y = np.zeros((K, L, M), bool)
        witness = np.zeros((K, M), np.int64)
        witness_gap = np.zeros((K, M), np.float32)
        H = np.zeros((K, M), np.float32)
        Hsrc = np.zeros((K, M), np.int8)
        U_scene = np.zeros(K, np.float32)
        U_interact = np.zeros((K, M), np.float32)
        # One post-prefix observable signature per (action, root mode).
        # It must be observable-only but include mode-dependent visible actors
        # when the rollout backend can expose them; otherwise OC labels collapse
        # to an over-conservative single class for every action.
        obs_signatures = [[None for _ in range(M)] for _ in range(K)]
        U_mode = normalized_mode_uncertainty(modes[:M])

        action_iter = range(len(actions))
        if args.inner_progress:
            action_iter = tqdm(action_iter, desc=f"{rid} actions", unit="action", leave=False)
        reusable_env = None
        if rollout_backend == "metadrive" and not args.no_reuse_metadrive_env:
            reusable_env = md_runner._make_env(obj, modes[0])
        try:
            for a_i in action_iter:
                a = actions[a_i]
                U_scene[a_i] = scene_uncertainty_from_action(a.states, 0.2 if obj["regime"] in ("near_contact", "contact_post_contact") else 0.0, len(actors) / 8)
                for m_i, mode in enumerate(modes[:M]):
                    best = -1e9
                    second = -1e9
                    h_values = []
                    h_sources = []
                    for r_i, o in enumerate(opts[a_i]):
                        if not (a.valid and o.valid):
                            for arr in margins.values():
                                arr[a_i, r_i, m_i] = -1.0
                            g_star[a_i, r_i, m_i, :] = -1.0
                            spec_margin_star[a_i, r_i, m_i, :] = -1.0
                            # Invalid/padded options should be excluded by option_mask
                            # and must not contaminate the action-level CRISP rule label.
                            continue
                        if rollout_backend == "metadrive":
                            trace = md_runner.rollout(obj, ego, a, o, mode, H_p, H_r, dt, root_map_features=mf, env=reusable_env)
                        else:
                            trace = synthetic_rollout(a, o, mode, H_p, H_r, dt, obj["regime"])
                        e = evidence_from_trace(trace, {"H_p": H_p, "dt": dt})
                        P[a_i, r_i, m_i] = e["P_star"]
                        Praw[a_i, r_i, m_i] = e["P_raw_star"]
                        G[a_i, r_i, m_i] = e["G_star"]
                        C[a_i, r_i, m_i] = e["C_star"]
                        Kdef[a_i, r_i, m_i] = e["K_star"]
                        for kmap, ekey in [("margin_option", "margin_option"), ("M_path_raw", "M_path_raw"), ("M_path_rec", "M_path_rec"), ("M_path_pre_no_first_contact", "M_path_pre_no_first_contact"), ("M_secondary", "M_secondary"), ("M_return", "M_return"), ("M_ctrl", "M_ctrl"), ("M_post", "M_post")]:
                            margins[kmap][a_i, r_i, m_i] = e[ekey]
                        g_star[a_i, r_i, m_i] = e["g_star"]
                        spec_margin_star[a_i, r_i, m_i] = e["spec_margin_star"]
                        spec_id_star[a_i, r_i, m_i] = e["spec_id_star"]
                        c_rule_option[a_i, r_i, m_i] = float(e["c_rule_star"])
                        if obs_signatures[a_i][m_i] is None:
                            obs_signatures[a_i][m_i] = post_prefix_observation_signature(trace, obj, mode)
                        Y[a_i, r_i, m_i] = bool(e["y_star"])
                        h_values.append(e["H_star"])
                        h_sources.append(e["H_source"])
                        val = e["margin_option"]
                        if val > best:
                            second = best
                            best = val
                            witness[a_i, m_i] = r_i
                        elif val > second:
                            second = val
                    H[a_i, m_i] = max(h_values) if h_values else 0.0
                    Hsrc[a_i, m_i] = max(h_sources) if h_sources else 0
                    witness_gap[a_i, m_i] = best - second if second > -1e8 else 0.0
                    U_interact[a_i, m_i] = 0.0
        finally:
            if reusable_env is not None:
                try:
                    reusable_env.close()
                except Exception:
                    pass

        U = np.zeros((K, M), np.float32)
        for a_i in range(K):
            for m_i in range(M):
                U[a_i, m_i] = combine_uncertainty(U_scene[a_i], U_mode[m_i], U_interact[a_i, m_i])
        obs_class = np.zeros((K, M), np.int64)
        obs_equiv = np.zeros((K, M, M), bool)
        beta_star = np.zeros((K, M, M), np.float32)
        witness_oc = np.zeros((K, M), np.int64)
        Y_oc = np.zeros((K, M), np.float32)
        for a_i in range(K):
            # Observation-consistent labels are based on information available
            # after executing the prefix: ego post-prefix state plus visible
            # actor summary when the rollout backend provides it.  Do not use
            # teacher success, future labels, or hidden mode parameters here.
            signatures = []
            for m_i in range(M):
                sig = obs_signatures[a_i][m_i]
                if sig is None:
                    sig = {"ego": actions[a_i].states[-1, :6]}
                signatures.append(sig)
            obs_class[a_i], obs_equiv[a_i] = build_obs_equivalence(signatures, eps_o=float(cfg.get("planner", {}).get("eps_o", 1e-3)))
            beta_star[a_i] = beta_from_obs_equiv(mode_probs, obs_equiv[a_i])
            witness_oc[a_i], Y_oc[a_i], _ = class_consistent_witness(Y[a_i].astype(np.float32), margins["margin_option"][a_i], mode_probs, obs_class[a_i])
            for m_i in range(M):
                j = int(witness_oc[a_i, m_i])
                val = c_rule_option[a_i, j, m_i]
                if not np.isfinite(val):
                    finite = c_rule_option[a_i, :, m_i][np.isfinite(c_rule_option[a_i, :, m_i])]
                    val = float(finite.min()) if finite.size else 1.0
                c_rule_star[a_i, m_i] = float(max(0.0, val))
        Y_action = Y.max(axis=1)  # oracle diagnostic only
        R_star = weighted_lcvar_np(Y_oc.astype(np.float32), mode_probs, float(cfg.get("planner", {}).get("alpha_R", 0.2)))
        H_action = upper_tail_cvar_np(H, mode_probs, float(cfg.get("planner", {}).get("alpha_H", 0.2))) if H.ndim == 2 else H

        sample = {}
        sample.update(at)
        sample.update(ot)
        sample.update({
            "mode_probs": mode_probs,
            "mode_seed_params": mode_seed_params_array(modes[:M]),
            "margin_option": margins["margin_option"],
            "Y_option": Y,
            "Y_action": Y_action,
            "R_star": R_star,
            "witness": witness_oc,
            "witness_raw_oracle": witness,
            "witness_oc": witness_oc,
            "witness_gap": witness_gap,
            "Y_oc": Y_oc.astype(np.float32),
            "obs_class": obs_class,
            "obs_equiv": obs_equiv,
            "beta_star": beta_star,
            "g_star": g_star.astype(np.float16),
            "y_star": Y.astype(np.float32),
            "h_star": H.astype(np.float16),
            "k_star": Kdef.astype(np.float16),
            "u_star": U.astype(np.float16),
            "c_rule_star": c_rule_star.astype(np.float16),
            "spec_margin_star": spec_margin_star.astype(np.float16),
            "spec_id_star": spec_id_star,
            "M_path_raw": margins["M_path_raw"],
            "M_path_rec": margins["M_path_rec"],
            "M_path_pre_no_first_contact": margins["M_path_pre_no_first_contact"],
            "M_secondary": margins["M_secondary"],
            "M_return": margins["M_return"],
            "M_ctrl": margins["M_ctrl"],
            "M_post": margins["M_post"],
            "P_star": P.astype(np.float16),
            "P_raw_star": Praw.astype(np.float16),
            "G_star": G.astype(np.float16),
            "C_star": C.astype(np.float16),
            "U_star": U.astype(np.float16),
            "U_scene": U_scene.astype(np.float16),
            "U_mode": U_mode.astype(np.float16),
            "U_interact": U_interact.astype(np.float16),
            "H_star": H.astype(np.float16),
            "H_action_star": H_action.astype(np.float16),
            "H_source": Hsrc,
            "K_star": Kdef.astype(np.float16),
        })
        if "bev" in bev_arrays:
            if rid not in bev_index_by_root:
                raise KeyError(f"root_id {rid} is missing from BEV dataset {args.bev_dir}")
            bidx = bev_index_by_root[rid]
            sample["bev"] = bev_arrays["bev"][bidx]
            sample["ego_info"] = bev_arrays["ego_info"][bidx]
            sample["route_command"] = bev_arrays["route_command"][bidx]
        else:
            sample["bev"] = np.zeros((5, 24, 256, 256), np.float16)
            sample["ego_info"] = np.zeros(11, np.float32)
            sample["route_command"] = np.zeros((20, 6), np.float32)
        sample["root_ids"] = str(rid)
        sample["regime"] = str(obj["regime"])
        return sample

    if args.single_npz:
        all_samples = []
        pbar = tqdm(ids, desc=f"teacher_labels[{args.split}:{root_start}:{root_end}:{root_stride}]", unit="root", total=len(ids))
        for idx, rid in enumerate(pbar):
            sample = build_one(idx, rid)
            all_samples.append(sample)
            _append_progress(args.progress_file, {"event": "root_done", "split": args.split, "root_id": str(rid), "local_index": idx, "global_index": root_start + idx * root_stride, "done": idx + 1, "total": len(ids)})
        arrays = {k: np.stack([s[k] for s in all_samples]) if not isinstance(all_samples[0][k], str) else np.asarray([s[k] for s in all_samples]) for k in all_samples[0].keys()} if all_samples else {}
        write_dataset(args.output, arrays, metadata)
    else:
        with ShardedDatasetWriter(args.output, metadata, shard_size=args.shard_size, compressed=args.compress_shards) as writer:
            pbar = tqdm(ids, desc=f"teacher_labels[{args.split}:{root_start}:{root_end}:{root_stride}]", unit="root", total=len(ids))
            for idx, rid in enumerate(pbar):
                sample = build_one(idx, rid)
                writer.append(sample)
                _append_progress(args.progress_file, {"event": "root_done", "split": args.split, "root_id": str(rid), "local_index": idx, "global_index": root_start + idx * root_stride, "done": idx + 1, "total": len(ids)})
    print(f"wrote teacher labels for {len(ids)} roots to {args.output}")


if __name__ == "__main__":
    main()
