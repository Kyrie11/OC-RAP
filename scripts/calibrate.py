#!/usr/bin/env python
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import argparse, json
from pathlib import Path
import numpy as np
from ocrap.teacher.dataset_writer import read_dataset
from ocrap.training.calibrate_selector import calibrate_q
from ocrap.evaluation.offline_eval import nominal_utility

if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--split", default="calib")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--eta-R", type=float, default=0.70)
    ap.add_argument("--eta-H", type=float, default=0.50)
    ap.add_argument("--epsilon-H", type=float, default=0.05)
    ap.add_argument("--output", required=True)
    args=ap.parse_args()
    arrays,meta=read_dataset(args.dataset); arrays=dict(arrays)
    if args.checkpoint:
        from ocrap.evaluation.inference import predict_profiles
        arrays.update(predict_profiles(args.dataset, args.checkpoint, batch_size=args.batch_size))
    R_star=np.asarray(arrays["R_star"])
    R_pred=np.asarray(arrays.get("R_pred", R_star))
    H_star=np.asarray(arrays.get("H_action_star", np.asarray(arrays["H_star"]).max(axis=-1)))
    H_pred=np.asarray(arrays.get("H_pred", H_star))
    action_mask=np.asarray(arrays["action_mask"]).astype(bool)
    dH_star=H_star-np.min(np.where(action_mask,H_star,np.inf),axis=1,keepdims=True)
    dH_pred=np.asarray(arrays.get("dH_pred", H_pred-np.min(np.where(action_mask,H_pred,np.inf),axis=1,keepdims=True)))
    if "C_pred" in arrays:
        C_pred=np.asarray(arrays["C_pred"])
    elif "c_rule_star" in arrays:
        C_pred=np.asarray(arrays["c_rule_star"]).max(axis=-1)
    else:
        C_pred=np.zeros_like(R_pred)
    if "c_rule_star" in arrays:
        C_star=np.asarray(arrays["c_rule_star"]).max(axis=-1)
    else:
        C_star=np.zeros_like(R_star)
    U_drv=nominal_utility(np.asarray(arrays["actions_states"]), action_mask)
    q=calibrate_q(R_pred,R_star,dH_pred,dH_star,action_mask,H_pred=H_pred,H_star=H_star,C_pred=C_pred,C_star=C_star,U_drv=U_drv,eta_R=args.eta_R,eta_H=args.eta_H,epsilon_H=args.epsilon_H)
    q["dataset_version"]=meta.get("dataset_version",""); q["split"]=args.split; q["uses_learned_predictions"]=bool(args.checkpoint)
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True); (out/"q_values.json").write_text(json.dumps(q,indent=2)); print(json.dumps(q,indent=2))
