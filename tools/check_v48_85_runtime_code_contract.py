#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,warnings
from pathlib import Path
import torch
from ocrap.models.ocrap import OCRAPModel
from ocrap.models.encoders import FlatFeatureLayout

def sha(p):
 h=hashlib.sha256();h.update(Path(p).read_bytes());return h.hexdigest()

def make(state):
 L=FlatFeatureLayout(feature_max_agents=2)
 return OCRAPModel(L.total_dim,num_roots=3,num_options=2,d_model=32,d_obs=8,encoder_type='structured_transformer',feature_layout=L.__dict__,direct_recovery_absolute_semantic_witness_correction=True,direct_recovery_semantic_witness_action_response_adapter=True,direct_recovery_semantic_witness_action_response_state_conditioning=state)

def _deterministic_synthetic(q,r):
    """Exercise the action-response/state-gate contract without random or degenerate weights.

    The original v48.85.0 preflight filled every action-projection coefficient with
    the same constant.  Because the adapter applies a bias-free LayerNorm first,
    every normalized action has (up to roundoff) zero feature mean, so a constant
    projection is mathematically orthogonal to the normalized action and can
    collapse to an exact zero response.  Whether torch.allclose then observed a
    state effect depended on numerical roundoff/PyTorch version.  Use a fixed,
    non-constant projection and deterministic inputs instead.
    """
    qa=q.direct_absolute_action_response_adapter
    ra=r.direct_absolute_action_response_adapter
    D=int(qa.action_dim); H=int(qa.d_model)
    # Non-constant feature pattern guarantees a non-zero dot product with the
    # deterministic normalized action below.  Channel/hidden scaling also
    # exercises both signed channels and multiple output coordinates.
    feat=torch.linspace(-0.037,0.061,D,dtype=torch.float32)
    hidden=(1.0+0.017*torch.arange(H,dtype=torch.float32)).view(H,1)
    with torch.no_grad():
        for c,scale in enumerate((1.0,1.37)):
            qa.action_projection[c].copy_(scale*hidden*feat.view(1,D))
        ra.action_projection.copy_(qa.action_projection)
    zero=torch.zeros(D,dtype=torch.float32)
    ramp=torch.linspace(-1.7,2.3,D,dtype=torch.float32)
    curved=torch.sin(torch.linspace(-1.2,2.0,D,dtype=torch.float32))+0.19*torch.linspace(-1,1,D)
    action=torch.stack((zero,ramp,curved),dim=0)
    # Every root has non-zero variance across hidden coordinates so the
    # parameter-free state gate is provably non-constant.
    base=torch.linspace(-1.5,1.9,H,dtype=torch.float32)
    roots=torch.stack([
        torch.stack((base,0.7*base.flip(0),torch.cos(base)),dim=0),
        torch.stack((1.2*base+0.1,-0.9*base+0.2,torch.sin(base)),dim=0),
        torch.stack((0.5*base-0.3,1.1*base.flip(0),torch.tanh(base)),dim=0),
    ],dim=0)
    margins=torch.tensor([[1.0,1.0,1.0],[-1.0,-1.0,-1.0],[1.0,-1.0,1.0]],dtype=torch.float32)
    with torch.no_grad():
        yq=qa(action,roots,margins)
        yr=ra(action,roots,margins)
    nominal_zero=bool(torch.equal(yq[0],torch.zeros_like(yq[0])) and torch.equal(yr[0],torch.zeros_like(yr[0])))
    finite=bool(torch.isfinite(yq).all() and torch.isfinite(yr).all())
    action_nonzero=float(yq[1:].abs().amax().item())
    state_delta=float((yr[1:]-yq[1:]).abs().amax().item())
    state_effect=bool(state_delta>1.0e-5 and action_nonzero>1.0e-5)
    return {
        'nominal_zero':nominal_zero,
        'finite':finite,
        'action_nonzero':action_nonzero,
        'state_delta_max_abs':state_delta,
        'state_effect':state_effect,
    }

def _zero_init_gradient_contract(state_conditioned: bool):
    m=make(state_conditioned)
    a=m.direct_absolute_action_response_adapter
    D=int(a.action_dim); H=int(a.d_model)
    action=torch.stack((
        torch.zeros(D,dtype=torch.float32),
        torch.linspace(-1.7,2.3,D,dtype=torch.float32),
        torch.sin(torch.linspace(-1.2,2.0,D,dtype=torch.float32))+0.19*torch.linspace(-1,1,D,dtype=torch.float32),
    ),dim=0)
    base=torch.linspace(-1.5,1.9,H,dtype=torch.float32)
    roots=torch.stack([
        torch.stack((base,0.7*base.flip(0),torch.cos(base)),dim=0),
        torch.stack((1.2*base+0.1,-0.9*base+0.2,torch.sin(base)),dim=0),
        torch.stack((0.5*base-0.3,1.1*base.flip(0),torch.tanh(base)),dim=0),
    ],dim=0)
    margins=torch.tensor([[1.0,1.0,1.0],[-1.0,-1.0,-1.0],[1.0,-1.0,1.0]],dtype=torch.float32)
    y=a(action,roots,margins)
    weight=torch.linspace(0.3,1.7,H,dtype=torch.float32).view(1,1,H)
    (y[1:]*weight).sum().backward()
    grad=a.action_projection.grad
    finite=bool(isinstance(grad,torch.Tensor) and torch.isfinite(grad).all())
    norm=float(grad.norm().detach().item()) if finite else 0.0
    return {'finite':finite,'grad_norm':norm,'nonzero':bool(finite and norm>1.0e-5)}

def main():
 warnings.filterwarnings('ignore', message='enable_nested_tensor is True.*norm_first was True', category=UserWarning)
 ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();repo=a.repo.resolve();errors=[]
 import ocrap,ocrap.cli.train,ocrap.models.ocrap
 mods={}
 for name,mod in [('ocrap',ocrap),('ocrap.cli.train',ocrap.cli.train),('ocrap.models.ocrap',ocrap.models.ocrap)]:
  p=Path(mod.__file__).resolve();inside=str(p).startswith(str(repo));mods[name]={'path':str(p),'inside_repo':inside,'sha256':sha(p)}
  if not inside:errors.append(f'{name} imported outside repo: {p}')
 q=make(False);r=make(True);qw=q.direct_absolute_action_response_adapter.action_projection;rw=r.direct_absolute_action_response_adapter.action_projection
 shape_ok=tuple(qw.shape)==tuple(rw.shape)==(2,32,q.direct_candidate_physical_feature_dim)
 zero_ok=torch.count_nonzero(qw).item()==0 and torch.count_nonzero(rw).item()==0
 cap_ok=sum(p.numel() for p in q.direct_absolute_action_response_adapter.parameters())==sum(p.numel() for p in r.direct_absolute_action_response_adapter.parameters())
 synth=_deterministic_synthetic(q,r)
 grad_q=_zero_init_gradient_contract(False);grad_r=_zero_init_gradient_contract(True)
 checks={
  'shape_ok':bool(shape_ok),'zero_init_ok':bool(zero_ok),'equal_capacity_ok':bool(cap_ok),
  'nominal_zero_ok':bool(synth['nominal_zero']),'finite_ok':bool(synth['finite']),
  'nonzero_action_response_ok':bool(synth['action_nonzero']>1.0e-5),
  'state_conditioning_effect_ok':bool(synth['state_effect']),
  'q_zero_init_gradient_ok':bool(grad_q['nonzero']),
  'r_zero_init_gradient_ok':bool(grad_r['nonzero']),
 }
 for name,ok in checks.items():
  if not ok:errors.append(f'action-response synthetic contract failed: {name}')
 doc={'schema':'ocrap-v48.85-sarr-runtime-code-contract-v2','engineering_version':'v48.85.1-OC-SARR-ENGFIX','valid':not errors,'attribution_ready':not errors,'errors':errors,'runtime_modules':mods,'representation_contract':{'narrow_absolute_only_action_response':True,'raw_action_candidate_minus_nominal':True,'signed_reserve_debt_channels':True,'q_r_equal_trainable_capacity':cap_ok,'action_projection_shape':list(qw.shape),'zero_init_execution_exact':zero_ok,'nominal_response_exact_zero':synth['nominal_zero'],'state_conditioning_parameter_free':True,'state_conditioning_changes_nonzero_action_response':synth['state_effect'],'synthetic_action_response_max_abs':synth['action_nonzero'],'synthetic_state_delta_max_abs':synth['state_delta_max_abs'],'synthetic_checks':checks,'zero_init_gradient_contract':{'Q85':grad_q,'R85':grad_r},'root_id_input':False,'option_id_input':False,'regime_id_input':False,'generic_mlp':False,'root_tail_adapter_family':False,'boundary_transport':False},'scientific_contract':{'truth_contract':'V48.80 structural_interval_bounds frozen scaffold','stage_i_shared_parameters_modified':False,'relative_ranker_modified':False,'dataset_reconstruction':False},'test_roots_read':False}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(doc,indent=2)+'\n');print(json.dumps({'valid':doc['valid'],'errors':errors,'synthetic':synth}));return 0 if doc['valid'] else 30
if __name__=='__main__':raise SystemExit(main())
