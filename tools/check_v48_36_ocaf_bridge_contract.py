#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, os, time
from pathlib import Path
import torch
from ocrap.models.ocrap import ObservationConditionedActionFrontierBridge, OCRAPModel

def atomic(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_name(f'.{path.name}.tmp.{os.getpid()}.{time.time_ns()}')
    tmp.write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); os.replace(tmp,path)

def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
    torch.manual_seed(7)
    bridge=ObservationConditionedActionFrontierBridge(13,17,32,0.0)
    bridge.eval()
    zero=torch.zeros(6,13); obs=torch.randn(6,17)
    z=bridge(zero,obs)
    exact_zero=bool(torch.equal(z,torch.zeros_like(z)))
    action=torch.randn(6,13,requires_grad=True)
    out=bridge(action,obs); loss=out.square().mean(); loss.backward()
    finite_grad=bool(action.grad is not None and torch.isfinite(action.grad).all() and action.grad.abs().sum()>0)
    with torch.no_grad():
        small=bridge(action.detach()*0.1,obs).norm().item(); large=bridge(action.detach(),obs).norm().item()
        obs_changed=(bridge(action.detach(),obs)-bridge(action.detach(),obs+torch.linspace(0,1,obs.shape[-1]))).abs().max().item()
        scene_only=bridge(torch.zeros_like(action.detach()),obs+3.0).abs().max().item()
        free=torch.tensor([4.0,-1.0,0.2]); cap=torch.tensor([-0.5,2.0,0.1])
        capped=OCRAPModel._noncompensatory_smooth_cap(free,cap,0.1)
        noncomp=bool(torch.all(capped<=free+1e-7) and torch.all(capped<=cap+1e-7))
    checks={
      'zero_action_exact_zero':exact_zero,
      'finite_nonzero_action_gradient':finite_grad,
      'action_magnitude_preserved':large>small*1.5,
      'observation_modulates_nonzero_action':obs_changed>1e-7,
      'observation_cannot_create_action_free_signal':scene_only==0.0,
      'noncompensatory_smooth_cap_upper_bounds_both_inputs':noncomp,
    }
    doc={'event':'v48_36_ocaf_bridge_contract','version':'v48.36-OCAF','created_unix':time.time(),
         'valid':all(checks.values()),'checks':checks,'measurements':{'small_action_norm':small,'large_action_norm':large,'observation_effect_max':obs_changed,'scene_only_max':scene_only},
         'regime_id_consumed':False,'test_roots_read':False}
    atomic(args.output,doc); print(json.dumps(doc,ensure_ascii=False)); return 0 if doc['valid'] else 4
if __name__=='__main__': raise SystemExit(main())
