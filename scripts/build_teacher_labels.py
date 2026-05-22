#!/usr/bin/env python
from __future__ import annotations

# Allow running scripts directly from a source checkout without pip install -e .
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))


import argparse, json
from pathlib import Path
import numpy as np

from recap.utils.datatypes import EgoState, ActorState, MapFeatures, RouteInfo
from recap.proposals.action_lattice import generate_lattice_actions, actions_to_tensors
from recap.proposals.recovery_options import generate_recovery_options, options_to_tensors
from recap.teacher.root_modes import generate_root_modes, mode_seed_params_array, normalized_mode_uncertainty, MODE_SLOT_SEMANTICS
from recap.teacher.rollout import synthetic_rollout
from recap.teacher.evidence_labels import evidence_from_trace, scene_uncertainty_from_action, combine_uncertainty
from recap.teacher.dataset_writer import write_dataset, read_dataset
from recap.evaluation.metrics import weighted_lcvar_np
from scripts._common import load_config


def load_root(path: Path):
    obj = json.loads(path.read_text())
    ego = EgoState(**obj["ego_state"])
    actors = [ActorState(**a) for a in obj["actor_states"]]
    mfobj = obj["map_features"]
    mf = MapFeatures([np.asarray(p, dtype=np.float32) for p in mfobj["drivable_polygons"]], [np.asarray(p, dtype=np.float32) for p in mfobj["lane_centerlines"]], [np.asarray(p, dtype=np.float32) for p in mfobj["lane_boundaries"]], [], float(mfobj.get("speed_limit_mps", 13.9)))
    robj = obj["route_info"]
    route = RouteInfo(np.asarray(robj["waypoints"], dtype=np.float32), np.asarray(robj.get("command_ids", np.zeros(len(robj["waypoints"]))), dtype=np.int64), float(robj.get("speed_limit_mps", 13.9)))
    return obj, ego, actors, mf, route


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--root-dir", default="data/recap/roots_raw")
    ap.add_argument("--bev-dir", default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-roots", type=int, default=None)
    ap.add_argument("--rollout-backend", choices=["auto", "synthetic", "metadrive"], default="auto")
    ap.add_argument("--scenario-dir", default=None, help="ScenarioNet database directory for real MetaDrive rollouts; defaults to root metadata scenario_dir.")
    ap.add_argument("--metadrive-reactive-traffic", type=str, default="true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    K = int(cfg.get("planner", {}).get("K", 16 if cfg.get("implementation_level") == "mvp" else 32))
    L = int(cfg.get("planner", {}).get("L", 6 if cfg.get("implementation_level") == "mvp" else 12))
    M = int(cfg.get("planner", {}).get("M", 4 if cfg.get("implementation_level") == "mvp" else 8))
    H_p = int(cfg.get("planner", {}).get("H_p", 10)); H_r = int(cfg.get("planner", {}).get("H_r", 25)); dt = float(cfg.get("planner", {}).get("dt", 0.2))
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
        from recap.teacher.metadrive_rollout import MetaDriveRolloutRunner
        scenario_dir = args.scenario_dir or root_meta.get("scenario_dir")
        if not scenario_dir:
            raise ValueError("--rollout-backend metadrive requires --scenario-dir or root metadata scenario_dir")
        md_runner = MetaDriveRolloutRunner(scenario_dir=str(scenario_dir), reactive_traffic=args.metadrive_reactive_traffic.lower() in ("1", "true", "yes"))
    if (root_dir / "splits.json").exists() and args.split != "all":
        ids = json.loads((root_dir / "splits.json").read_text()).get(args.split, [])
    else:
        ids = sorted(p.stem for p in root_dir.glob("root_*.json"))
    if args.max_roots:
        ids = ids[:args.max_roots]
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
        bev_index_by_root = {str(r): j for j, r in enumerate(bev_arrays["root_ids"])}
    all_arrays = {k: [] for k in ["actions_states","actions_controls","actions_params","action_mask","options_states_ref","options_controls_ref","options_params","option_mask","mode_probs","mode_seed_params","margin_option","Y_option","Y_action","R_star","witness","witness_gap","M_path_raw","M_path_rec","M_path_pre_no_first_contact","M_secondary","M_return","M_post","P_star","P_raw_star","G_star","C_star","U_star","U_scene","U_mode","U_interact","H_star","H_action_star","H_source","K_star"]}
    # Also pass through BEV arrays if available or zero-fill.
    bevs=[]; ego_infos=[]; route_cmds=[]; root_ids=[]; regimes=[]
    for idx, rid in enumerate(ids):
        obj, ego, actors, mf, route = load_root(root_dir / f"{rid}.json")
        actions = generate_lattice_actions(ego, route, mf, K=K, H_p=H_p, dt=dt)
        opts = generate_recovery_options(actions, route, mf, L=L, H_r=H_r, dt=dt)
        at = actions_to_tensors(actions); ot = options_to_tensors(opts)
        modes = generate_root_modes(int(obj["seed"]), M=M)
        mode_probs = np.ones(M, dtype=np.float32) / M
        P=np.zeros((K,L,M),np.float32); Praw=np.zeros_like(P); G=np.zeros_like(P); C=np.zeros_like(P); Kdef=np.zeros_like(P)
        margins={name: np.zeros_like(P) for name in ["margin_option","M_path_raw","M_path_rec","M_path_pre_no_first_contact","M_secondary","M_return","M_post"]}
        Y=np.zeros((K,L,M), bool); witness=np.zeros((K,M),np.int64); witness_gap=np.zeros((K,M),np.float32)
        H=np.zeros((K,M),np.float32); Hsrc=np.zeros((K,M),np.int8); U_scene=np.zeros(K,np.float32); U_interact=np.zeros((K,M),np.float32)
        U_mode = normalized_mode_uncertainty(modes[:M])
        for a_i,a in enumerate(actions):
            U_scene[a_i]=scene_uncertainty_from_action(a.states, 0.2 if obj["regime"] in ("near_contact","contact_post_contact") else 0.0, len(actors)/8)
            for m_i,mode in enumerate(modes[:M]):
                best=-1e9; second=-1e9
                h_values=[]; h_sources=[]
                for r_i,o in enumerate(opts[a_i]):
                    if not (a.valid and o.valid):
                        margins["margin_option"][a_i,r_i,m_i]=-1.0
                        continue
                    if rollout_backend == "metadrive":
                        trace = md_runner.rollout(obj, ego, a, o, mode, H_p, H_r, dt)
                    else:
                        trace = synthetic_rollout(a,o,mode,H_p,H_r,dt,obj["regime"])
                    e = evidence_from_trace(trace,{"H_p":H_p,"dt":dt})
                    P[a_i,r_i,m_i]=e["P_star"]; Praw[a_i,r_i,m_i]=e["P_raw_star"]; G[a_i,r_i,m_i]=e["G_star"]; C[a_i,r_i,m_i]=e["C_star"]; Kdef[a_i,r_i,m_i]=e["K_star"]
                    for kmap,ekey in [("margin_option","M_option"),("M_path_raw","M_path_raw"),("M_path_rec","M_path_rec"),("M_path_pre_no_first_contact","M_path_pre_no_first_contact"),("M_secondary","M_secondary"),("M_return","M_return"),("M_post","M_post")]:
                        margins[kmap][a_i,r_i,m_i]=e[ekey]
                    Y[a_i,r_i,m_i]=e["Y_option"]
                    h_values.append(e["H_star"]); h_sources.append(e["H_source"])
                    val=e["M_option"]
                    if val>best:
                        second=best; best=val; witness[a_i,m_i]=r_i
                    elif val>second:
                        second=val
                H[a_i,m_i]=max(h_values) if h_values else 0.0  # identical by construction for prefix-level synthetic rollouts
                Hsrc[a_i,m_i]=max(h_sources) if h_sources else 0
                witness_gap[a_i,m_i]=best-second if second>-1e8 else 0.0
                U_interact[a_i,m_i]=0.0
        U=np.zeros((K,M),np.float32)
        for a_i in range(K):
            for m_i in range(M):
                U[a_i,m_i]=combine_uncertainty(U_scene[a_i],U_mode[m_i],U_interact[a_i,m_i])
        Y_action=Y.max(axis=1)
        R_star=weighted_lcvar_np(Y_action.astype(np.float32), mode_probs, float(cfg.get("planner", {}).get("alpha_R",0.2)))
        H_action=H.max(axis=-1) if H.ndim==2 else H
        for k,v in at.items(): all_arrays[k].append(v)
        for k,v in ot.items(): all_arrays[k].append(v)
        for k,v in {"mode_probs":mode_probs,"mode_seed_params":mode_seed_params_array(modes[:M]),"margin_option":margins["margin_option"],"Y_option":Y,"Y_action":Y_action,"R_star":R_star,"witness":witness,"witness_gap":witness_gap,"M_path_raw":margins["M_path_raw"],"M_path_rec":margins["M_path_rec"],"M_path_pre_no_first_contact":margins["M_path_pre_no_first_contact"],"M_secondary":margins["M_secondary"],"M_return":margins["M_return"],"M_post":margins["M_post"],"P_star":P.astype(np.float16),"P_raw_star":Praw.astype(np.float16),"G_star":G.astype(np.float16),"C_star":C.astype(np.float16),"U_star":U.astype(np.float16),"U_scene":U_scene.astype(np.float16),"U_mode":U_mode.astype(np.float16),"U_interact":U_interact.astype(np.float16),"H_star":H.astype(np.float16),"H_action_star":H_action.astype(np.float16),"H_source":Hsrc,"K_star":Kdef.astype(np.float16)}.items():
            all_arrays[k].append(v)
        if "bev" in bev_arrays:
            if rid not in bev_index_by_root:
                raise KeyError(f"root_id {rid} is missing from BEV dataset {args.bev_dir}")
            bidx = bev_index_by_root[rid]
            bevs.append(bev_arrays["bev"][bidx]); ego_infos.append(bev_arrays["ego_info"][bidx]); route_cmds.append(bev_arrays["route_command"][bidx])
        else:
            bevs.append(np.zeros((10,24,256,256),np.float16)); ego_infos.append(np.zeros(11,np.float32)); route_cmds.append(np.zeros((20,6),np.float32))
        root_ids.append(rid); regimes.append(obj["regime"])
    arrays={k:np.stack(v) for k,v in all_arrays.items()}
    arrays.update({"bev":np.stack(bevs),"ego_info":np.stack(ego_infos),"route_command":np.stack(route_cmds),"root_ids":np.asarray(root_ids),"regime":np.asarray(regimes)})
    is_synthetic = bool(root_meta.get("is_synthetic", rollout_backend != "metadrive"))
    metadata={"dataset_version":"metadrive_recovery_v1_real" if rollout_backend == "metadrive" else "metadrive_recovery_v0_synthetic","split":args.split,"split_by":"root_scene_id","implementation_level":cfg.get("implementation_level","mvp"),"K":K,"L":L,"M":M,"H_p":H_p,"H_r":H_r,"dt":dt,"mode_slot_semantics":MODE_SLOT_SEMANTICS[:M],"mode_alignment":"fixed_semantic_index","root_shared_mode_is_latent_context_not_open_loop_trajectory":True,"root_backend":root_meta.get("backend","unknown"),"rollout_backend":rollout_backend,"scenario_dir":args.scenario_dir or root_meta.get("scenario_dir"),"is_synthetic":is_synthetic,"paper_final_ready":rollout_backend == "metadrive" and not is_synthetic}
    write_dataset(args.output, arrays, metadata)
    print(f"wrote teacher labels for {len(ids)} roots to {args.output}")

if __name__=="__main__": main()
