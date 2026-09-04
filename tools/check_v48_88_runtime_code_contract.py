#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import torch
from ocrap.models.ocrap import OCRAPModel
from ocrap.models.encoders import FlatFeatureLayout


def sha(p: Path):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for z in iter(lambda:f.read(1<<20),b''): h.update(z)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--repo',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    repo=a.repo.resolve(); errs=[]
    files=[
        'src/ocrap/cli/train.py','src/ocrap/models/ocrap.py','src/ocrap/models/inference.py',
        'scripts/train_ocrap_v48_trac_sr.sh','scripts/adapt_ocrap_v48_36_ocaf_single_stage.sh',
        'tools/check_v48_88_state_isolation.py','tools/audit_v48_88_qtrr.py','tools/compare_v48_88_qtrr.py',
        'scripts/run_v48_88_dcp_drfc_bcde_rifa_qtrr_two_gpu.sh',
    ]
    rf={}
    for rel in files:
        p=(repo/rel).resolve(); ok=p.is_file() and str(p).startswith(str(repo))
        rf[rel]={'path':str(p),'exists':p.is_file(),'inside_repo':str(p).startswith(str(repo)),'sha256':sha(p) if p.is_file() else None}
        if not ok: errs.append(f'runtime file invalid: {rel}')

    L=FlatFeatureLayout(feature_max_agents=2)
    m=OCRAPModel(
        L.total_dim,num_roots=4,num_options=3,d_model=32,d_obs=8,
        encoder_type='structured_transformer',feature_layout=L.__dict__,
        direct_recovery_absolute_semantic_witness_correction=True,
        direct_recovery_semantic_witness_quotient_tail_response=True,
        direct_recovery_semantic_witness_boundary_transport=False,
    )
    w=m.direct_absolute_quotient_tail_response_weight
    parameter_count=int(w.numel()) if w is not None else -1
    zero_init=bool(w is not None and torch.count_nonzero(w).item()==0)
    A=m.direct_candidate_physical_feature_dim
    with torch.no_grad():
        if w is not None:
            w[0].copy_(torch.linspace(-.2,.3,A)); w[1].copy_(torch.linspace(.25,-.15,A))
    nominal_zero=bool(torch.equal(
        m._quotient_tail_response_coefficient(torch.zeros(2,A),torch.tensor([1.0,-1.0])),
        torch.zeros(2)
    ))
    torch.manual_seed(88)
    B,K,NO=5,4,3
    p=torch.rand(B,K); p=p/p.sum(dim=1,keepdim=True); g=torch.randn(B,K,NO)
    d=OCRAPModel._quotient_tail_direction_from_cotangent(p,g)
    translation_residual=float((p.unsqueeze(-1)*d).sum(dim=1).abs().max())
    unit_norm=d.square().sum(dim=(1,2)).sqrt()
    unit_ok=bool(torch.allclose(unit_norm,torch.ones_like(unit_norm),atol=2e-6,rtol=2e-6))
    c=torch.randn(B,1,NO); pure=p.unsqueeze(-1)*c
    z=OCRAPModel._quotient_tail_direction_from_cotangent(p,pure)
    pure_translation_zero=float(z.abs().max()) < 2e-6
    checks={
        'parameter_count':parameter_count,'v48_87_barr_parameter_count':53550,
        'capacity_ratio_vs_barr':parameter_count/53550.0 if parameter_count>0 else None,
        'zero_init_exact_native':zero_init,'nominal_action_exact_zero':nominal_zero,
        'p_weighted_option_translation_residual_max':translation_residual,
        'zero_translation_contract':translation_residual<2e-6,
        'unit_quotient_direction':unit_ok,
        'pure_translation_projected_out':pure_translation_zero,
    }
    if parameter_count!=282: errs.append(f'parameter count {parameter_count} != 282')
    for k in ['zero_init_exact_native','nominal_action_exact_zero','zero_translation_contract','unit_quotient_direction','pure_translation_projected_out']:
        if not checks[k]: errs.append(f'synthetic contract failed: {k}')
    doc={
        'schema':'ocrap-v48.88-qtrr-runtime-code-contract-v1','engineering_version':'v48.88.0-OC-QTRR',
        'valid':not errs,'attribution_ready':not errs,'errors':errs,'runtime_files':rf,
        'representation_contract':{
            'name':'Observation-Consistent Quotient Tail-Recovery Response',
            'candidate_minus_nominal_executable_action':True,
            'exact_nested_ocmero_cotangent':True,
            'learned_root_local_target':False,
            'learned_nullspace_capacity':False,
            'option_translation_removed':True,
            'reserve_debt_channels':2,
            'regime_conditioning':False,'root_option_regime_ids':False,'generic_mlp':False,
            'broad_encoder_retraining':False,'boundary_transport':False,'dataset_reconstruction':False,
            'relative_ranker_modified':False,
        },
        'synthetic_checks':checks,'test_roots_read':False,
    }
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'valid':doc['valid'],'errors':errs,'checks':checks})); return 0 if doc['valid'] else 30
if __name__=='__main__': raise SystemExit(main())
