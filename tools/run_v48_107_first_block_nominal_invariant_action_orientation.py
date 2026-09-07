#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ocrap.models.data import OCRAPSampleDataset
from ocrap.models.inference import load_model_bundle
from ocrap.models.encoders import StructuredTokenEncoder
from ocrap.v48_96_support_reserve_root_observability import feature_only_dataset_cfg
from ocrap.v48_100_joint_root_semantic_decoder import joint_semantic_loss
from ocrap.v48_103_factorized_control_sufficient_state import FactorizedControlSufficientState
from ocrap.v48_107_first_block_nominal_invariant_action_orientation import (
    ALGORITHM_NAME,
    ENGINEERING_VERSION,
    NominalInvariantFirstBlockOrientation,
    ordinal_action_orientation_loss_sum,
)
from tools.run_v48_97_executable_recovery_state import (
    ROLES,
    _action_metric,
    _dense_metrics,
    _evaluation_contract,
    _index_rows,
    _pair_indices,
    _role_rows,
    _state_metric,
    build_v93_map,
    sha256,
)

V103_ENGINEERING_VERSION = "v48.103.0-OC-FCSS"


def _cache_key(checkpoint: Path, index_path: Path, variant: str) -> str:
    payload = {
        "checkpoint": sha256(checkpoint), "index": sha256(index_path), "variant": variant,
        "kind": "v48_107_frozen_preencoder_and_first_block_tokens",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _input_tokens(enc: StructuredTokenEncoder, x: torch.Tensor) -> torch.Tensor:
    ego, prefix_param, macro, scalar, prefix_state, control, agent_summary, agents, bev, route, maps, dyn = enc._split(x)
    B = x.shape[0]
    tokens = [
        enc.ego_proj(ego), enc.prefix_param_proj(prefix_param), enc.macro_scalar_proj(torch.cat([macro, scalar], dim=-1)),
        enc.prefix_state_proj(prefix_state), enc.control_proj(control), enc.agent_summary_proj(agent_summary),
        enc.bev_proj(bev), enc.route_proj(route), enc.map_proj(maps), enc.dyn_proj(dyn),
    ]
    tok = torch.stack(tokens, dim=1)
    tok = torch.cat([enc.cls.expand(B, -1, -1), tok, enc.agent_proj(agents)], dim=1)
    return tok + enc.pos[:, :tok.shape[1], :]


def _input_first_and_final(enc: StructuredTokenEncoder, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(enc.encoder.layers) != 2:
        raise RuntimeError(f"V48.107 preregistered for historical two-layer Stage-I, got {len(enc.encoder.layers)}")
    inp = _input_tokens(enc, x)
    first = enc.encoder.layers[0](inp)
    final = enc.norm(enc.encoder.layers[1](first))
    return inp, first, final


def extract_first_block_features(*, checkpoint: Path, index_path: Path, cache_dir: Path, device: str, variant: str,
                                 batch_size: int = 128) -> dict[str, Any]:
    key = _cache_key(checkpoint, index_path, variant)
    cache_dir.mkdir(parents=True, exist_ok=True); cp = cache_dir / f"{key}.pt"
    if cp.is_file():
        obj = torch.load(cp, map_location="cpu", weights_only=False)
        if obj.get("cache_key") == key:
            return obj
    rows = _index_rows(index_path); paths = [Path(str(r["path"])) for r in rows]
    bundle = load_model_bundle(checkpoint, {"training": {"device": device}})
    if bundle is None:
        raise RuntimeError(f"cannot load checkpoint {checkpoint}")
    model = bundle.model.eval(); [p.requires_grad_(False) for p in model.parameters()]
    if not isinstance(model.encoder, StructuredTokenEncoder):
        raise RuntimeError("V48.107 requires StructuredTokenEncoder")
    enc = model.encoder.eval(); dev = bundle.device
    if len(enc.encoder.layers) != 2:
        raise RuntimeError("V48.107 requires exactly two historical Stage-I Transformer layers")
    cfg, event = feature_only_dataset_cfg(bundle.cfg, cache_dir=str(cache_dir / "tensor"), workers=8)
    ds = OCRAPSampleDataset(paths, cfg)
    if ds.absolute_truth_contract_event.get("enabled") or ds.action_response_truth_event.get("enabled"):
        raise RuntimeError("V48.107 feature-only dataset unexpectedly attached truth sidecars")
    if [str(p.resolve()) for p in paths] != [str(p.resolve()) for p in ds.paths]:
        raise RuntimeError("V48.107 dataset path order differs from index")
    inputs=[]; bases=[]; root_valid=[]; max_identity=0.0
    with torch.no_grad():
        for st in range(0, len(ds), batch_size):
            items=[ds[i] for i in range(st,min(len(ds),st+batch_size))]
            x=torch.stack([it["x"] for it in items]).to(dev)
            rv=torch.stack([it["root_valid"] for it in items]).to(dev)
            inp, first, final = _input_first_and_final(enc, x)
            direct = model._scene_tokens(x)
            max_identity=max(max_identity,float((final-direct).abs().max().item()))
            inputs.append(inp.float().cpu()); bases.append(first.float().cpu()); root_valid.append(rv.bool().cpu())
    if max_identity > 1e-6:
        raise RuntimeError(f"V48.107 historical final reconstruction mismatch {max_identity}")
    obj={
        "cache_key":key,"checkpoint":str(checkpoint.resolve()),"index":str(index_path.resolve()),"rows":rows,
        "input_tokens":torch.cat(inputs),"base_after_first":torch.cat(bases),"root_valid":torch.cat(root_valid),
        "historical_final_identity_max_abs":max_identity,"feature_only_dataset_contract":event,
        "tensor_cache_event":ds.tensor_cache_event,"encoder_layer_count":2,"first_block_index":0,
    }
    torch.save(obj,cp); return obj


def _load_model(checkpoint: Path, device: str):
    b=load_model_bundle(checkpoint,{"training":{"device":device}})
    if b is None: raise RuntimeError(f"cannot load checkpoint {checkpoint}")
    m=b.model.eval(); [p.requires_grad_(False) for p in m.parameters()]
    if not isinstance(m.encoder,StructuredTokenEncoder): raise RuntimeError("V48.107 requires structured encoder")
    if len(m.encoder.encoder.layers)!=2: raise RuntimeError("V48.107 requires historical two-layer Stage-I")
    return b,m


def _load_v103_state(path: Path, variant: str, d_model: int) -> FactorizedControlSufficientState:
    obj=torch.load(path,map_location="cpu",weights_only=False)
    if obj.get("engineering_version")!=V103_ENGINEERING_VERSION or obj.get("variant")!=variant or int(obj.get("representation_parameter_count",-1))!=1540:
        raise ValueError("V48.107 authoritative V48.103 state mismatch")
    m=FactorizedControlSufficientState(d_model); m.load_state_dict(obj["state_dict"],strict=True); m.eval()
    for p in m.parameters(): p.requires_grad_(False)
    return m


def _load_v103_result(path: Path, variant: str) -> dict[str,Any]:
    x=json.loads(path.read_text())
    if not x.get("valid") or x.get("engineering_version")!=V103_ENGINEERING_VERSION or x.get("variant")!=variant:
        raise ValueError("V48.107 authoritative V48.103 result mismatch")
    return x


def _groups(rows:list[dict[str,Any]]) -> list[list[int]]:
    by={}
    for i,r in enumerate(rows): by.setdefault((int(r["bucket"]),str(r["scene"]),int(r["time"])),[]).append(i)
    out=list(by.values())
    for ids in out:
        if sum(bool(rows[i].get("nominal",False)) for i in ids)!=1: raise ValueError("V48.107 group nominal contract")
    return out


def _group_batches(rows:list[dict[str,Any]], max_rows:int=192) -> list[list[int]]:
    batches=[]; cur=[]
    for ids in _groups(rows):
        if cur and len(cur)+len(ids)>max_rows: batches.append(cur); cur=[]
        cur.extend(ids)
    if cur: batches.append(cur)
    return batches


def _local_nominal(rows, ids, device):
    local={g:i for i,g in enumerate(ids)}; ni=[]
    group_nom={}
    for j,g in enumerate(ids):
        r=rows[g]; key=(int(r["bucket"]),str(r["scene"]),int(r["time"]))
        if bool(r.get("nominal",False)): group_nom[key]=j
    for g in ids:
        r=rows[g]; key=(int(r["bucket"]),str(r["scene"]),int(r["time"]))
        if key not in group_nom: raise RuntimeError("V48.107 batch split group")
        ni.append(group_nom[key])
    return torch.tensor(ni,dtype=torch.long,device=device)


def _local_group_ids(rows, ids, device):
    mapping={}; out=[]
    for g in ids:
        r=rows[g]; key=(int(r["bucket"]),str(r["scene"]),int(r["time"]))
        if key not in mapping: mapping[key]=len(mapping)
        out.append(mapping[key])
    return torch.tensor(out,dtype=torch.long,device=device)


def _semantics(refiner, readout, inp, base1, ni):
    mem=refiner.refined_memory(inp,ni,base1)
    pot=readout.semantic_potentials(mem); anchor=pot.index_select(0,ni)
    support=torch.sigmoid(anchor[:,0]+pot[:,2]-anchor[:,2])
    reserve=anchor[:,1]+pot[:,3]-anchor[:,3]
    return support,reserve


def _teacher(rows, ids, device):
    td=torch.tensor([float(rows[i]["teacher_drs"]) for i in ids],dtype=torch.float32,device=device)
    tr=torch.tensor([float(rows[i]["teacher_r_dep"]) for i in ids],dtype=torch.float32,device=device)
    return td,tr


def _pair_counts(rows:list[dict[str,Any]]) -> tuple[int,int]:
    ns=nr=0
    for ids in _groups(rows):
        ds=[float(rows[i]["teacher_drs"]) for i in ids]
        rr=[float(rows[i]["teacher_r_dep"]) for i in ids]
        for a in range(len(ids)):
            for b in range(a+1,len(ids)):
                if abs(ds[a]-ds[b])>1e-6: ns+=1
                if abs(rr[a]-rr[b])>1e-6: nr+=1
    if ns<=0 or nr<=0: raise RuntimeError(f"V48.107 insufficient orientation pairs support={ns} reserve={nr}")
    return ns,nr


def _orientation_epoch(*,refiner,readout,obj,device,scales,train:bool,max_rows:int=192):
    total_s,total_r=_pair_counts(obj["rows"]); ss=0.0; rr=0.0
    for ids in _group_batches(obj["rows"],max_rows):
        inp=obj["input_tokens"][ids].to(device); base1=obj["base_after_first"][ids].to(device)
        ni=_local_nominal(obj["rows"],ids,device); gid=_local_group_ids(obj["rows"],ids,device)
        support,reserve=_semantics(refiner,readout,inp,base1,ni); td,tr=_teacher(obj["rows"],ids,device)
        _loss,parts=ordinal_action_orientation_loss_sum(support,reserve,td,tr,gid,scales)
        batch_loss=0.5*(parts["support_orientation_sum"]/float(total_s)+parts["reserve_orientation_sum"]/float(total_r))
        if train: batch_loss.backward()
        ss += float(parts["support_orientation_sum"].detach().item()); rr += float(parts["reserve_orientation_sum"].detach().item())
    ms=ss/float(total_s); mr=rr/float(total_r)
    return (ms+mr)/2.0,{"support_orientation_mean":ms,"reserve_orientation_mean":mr,"support_pairs":total_s,"reserve_pairs":total_r}


def train_refiner(*,refiner,readout,train_obj,dev_obj,device,scales,max_epochs=60,patience=10):
    torch.manual_seed(107); np.random.seed(107); random.seed(107)
    refiner.to(device).train(False); readout.to(device).eval(); params=list(refiner.adapted_first.parameters())
    opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=1e-4)
    best=float("inf"); best_state=None; best_epoch=-1; stale=0; hist=[]
    for epoch in range(max_epochs):
        opt.zero_grad(set_to_none=True)
        train_loss,train_parts=_orientation_epoch(refiner=refiner,readout=readout,obj=train_obj,device=device,scales=scales,train=True)
        grad=float(torch.nn.utils.clip_grad_norm_(params,5.0).item()); opt.step()
        with torch.no_grad(): dev_loss,dev_parts=_orientation_epoch(refiner=refiner,readout=readout,obj=dev_obj,device=device,scales=scales,train=False)
        hist.append({"epoch":epoch,"train_orientation_loss":train_loss,"train_parts":train_parts,"dev_orientation_loss":dev_loss,"dev_parts":dev_parts,"first_block_grad_norm":grad})
        if dev_loss < best-1e-5:
            best=dev_loss; best_epoch=epoch; stale=0; best_state={k:v.detach().cpu().clone() for k,v in refiner.adapted_first.state_dict().items()}
        else: stale+=1
        if stale>=patience: break
    if best_state is None: raise RuntimeError("V48.107 no best state")
    refiner.adapted_first.load_state_dict(best_state,strict=True); refiner.to(device).train(False)
    return {"best_epoch":best_epoch,"best_dev_orientation_loss":best,"epochs_completed":len(hist),"history":hist}


def _predict_all(*,refiner,readout,obj,device,max_rows=192,base_only:bool=False):
    s=np.zeros(len(obj["rows"]),dtype=np.float32); r=np.zeros(len(obj["rows"]),dtype=np.float32); max_nom=0.0
    with torch.no_grad():
        for ids in _group_batches(obj["rows"],max_rows):
            inp=obj["input_tokens"][ids].to(device); base1=obj["base_after_first"][ids].to(device); ni=_local_nominal(obj["rows"],ids,device)
            if base_only:
                mem=refiner.tail_memory(base1); pot=readout.semantic_potentials(mem); anchor=pot.index_select(0,ni)
                sp=torch.sigmoid(anchor[:,0]+pot[:,2]-anchor[:,2]); rs=anchor[:,1]+pot[:,3]-anchor[:,3]
            else:
                max_nom=max(max_nom,refiner.nominal_identity_error(inp,ni,base1)); sp,rs=_semantics(refiner,readout,inp,base1,ni)
            s[np.asarray(ids)]=sp.cpu().numpy(); r[np.asarray(ids)]=rs.cpu().numpy()
    return s,r,max_nom


def _semantic_full(obj,support,reserve,device,scales):
    s=torch.tensor(support,device=device); r=torch.tensor(reserve,device=device)
    td=torch.tensor([float(x["teacher_drs"]) for x in obj["rows"]],device=device)
    tr=torch.tensor([float(x["teacher_r_dep"]) for x in obj["rows"]],device=device)
    ci,ni=_pair_indices(obj["rows"]); ci=torch.tensor(ci,dtype=torch.long,device=device); ni=torch.tensor(ni,dtype=torch.long,device=device)
    loss,parts=joint_semantic_loss(s,r,td,tr,ci,ni,scales)
    return float(loss.item()),{k:float(v.item()) for k,v in parts.items()}


def _evaluate_cells(dev_obj,cert_obj,dev_s,dev_r,cert_s,cert_r,v93):
    cells={}; contracts={}
    for role in ROLES:
        obj=dev_obj if role.startswith("dev_") else cert_obj; sp=dev_s if role.startswith("dev_") else cert_s; rs=dev_r if role.startswith("dev_") else cert_r
        rr=_role_rows(obj,sp,rs,role,v93); c=_evaluation_contract(rr,role)
        if not c["valid"]: raise RuntimeError(f"V48.107 evaluation contract {role}: {c['errors']}")
        st=_state_metric(rr); su,ss=_action_metric(rr,"drs_activation","support"); re,rrs=_action_metric(rr,"deployability_gain","reserve")
        cells[role]={"state":st,"support_true":su,"support_shuffled":ss,"reserve_true":re,"reserve_shuffled":rrs}; contracts[role]=c
    return cells,contracts


def _state_identity(cells,ref,tol=1e-7):
    errors=[]
    for role in ROLES:
        a=cells[role]["state"]; b=ref["cells"][role]["state"]
        for k in ("rows","drs_state_rows","dep_state_rows"):
            if int(a.get(k,-1))!=int(b.get(k,-2)): errors.append(f"{role}:{k}")
        if a.get("auc") is None or b.get("auc") is None or abs(float(a["auc"])-float(b["auc"]))>tol: errors.append(f"{role}:auc")
    return {"valid":not errors,"tolerance":tol,"errors":errors}


def _full_metric_identity(cells,ref,tol=1e-7):
    errors=[]
    for role in ROLES:
        for name in ("state","support_true","support_shuffled","reserve_true","reserve_shuffled"):
            aa=cells[role][name]; bb=ref["cells"][role][name]
            for k in ("rows","positive_rows","negative_rows","powered_groups"):
                if k in aa or k in bb:
                    if int(aa.get(k,-1))!=int(bb.get(k,-2)): errors.append(f"{role}:{name}:{k}")
            for k in ("auc","auc_vs_shuffled","top1","top1_vs_shuffled"):
                if k in aa or k in bb:
                    av,bv=aa.get(k),bb.get(k)
                    if av is None and bv is None: continue
                    if av is None or bv is None or abs(float(av)-float(bv))>tol: errors.append(f"{role}:{name}:{k}")
    return {"valid":not errors,"tolerance":tol,"errors":errors}


def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",type=Path,required=True); ap.add_argument("--v103-state",type=Path,required=True); ap.add_argument("--v103-result",type=Path,required=True)
    ap.add_argument("--train-index",type=Path,required=True); ap.add_argument("--dev-index",type=Path,required=True); ap.add_argument("--certificate-index",type=Path,required=True)
    ap.add_argument("--v93-audit",type=Path,required=True); ap.add_argument("--cache-dir",type=Path,required=True); ap.add_argument("--device",default="cuda"); ap.add_argument("--variant",required=True)
    ap.add_argument("--output",type=Path,required=True); ap.add_argument("--state-output",type=Path,required=True)
    a=ap.parse_args(); t0=time.perf_counter(); resolved=a.device if (not a.device.startswith("cuda") or torch.cuda.is_available()) else "cpu"; device=torch.device(resolved)
    train_obj=extract_first_block_features(checkpoint=a.checkpoint,index_path=a.train_index,cache_dir=a.cache_dir/"train",device=resolved,variant=a.variant)
    dev_obj=extract_first_block_features(checkpoint=a.checkpoint,index_path=a.dev_index,cache_dir=a.cache_dir/"dev",device=resolved,variant=a.variant)
    cert_obj=extract_first_block_features(checkpoint=a.checkpoint,index_path=a.certificate_index,cache_dir=a.cache_dir/"certificate",device=resolved,variant=a.variant)
    _,model=_load_model(a.checkpoint,resolved); d=int(train_obj["input_tokens"].shape[-1]); readout=_load_v103_state(a.v103_state,a.variant,d).to(device); ref103=_load_v103_result(a.v103_result,a.variant)
    refiner=NominalInvariantFirstBlockOrientation(model.encoder.encoder.layers[0],list(model.encoder.encoder.layers[1:]),model.encoder.norm).to(device); refiner.train(False)
    scales={k:float(v) for k,v in (ref103.get("semantic_metric_scales") or {}).items()}
    if set(scales)!={"support","reserve","delta_support","delta_reserve"}: raise RuntimeError("V48.107 semantic scales missing")
    if refiner.parameter_count!=444864: raise RuntimeError(f"V48.107 first block parameter count {refiner.parameter_count}")
    for k,v in refiner.base_first.state_dict().items():
        if not torch.equal(v.detach().cpu(),refiner.adapted_first.state_dict()[k].detach().cpu()): raise RuntimeError("V48.107 adapted first block initialization mismatch")
    d0s,d0r,_=_predict_all(refiner=refiner,readout=readout,obj=dev_obj,device=device,base_only=True); c0s,c0r,_=_predict_all(refiner=refiner,readout=readout,obj=cert_obj,device=device,base_only=True)
    v93=build_v93_map(a.v93_audit); init_cells,_=_evaluate_cells(dev_obj,cert_obj,d0s,d0r,c0s,c0r,v93); init_id=_full_metric_identity(init_cells,ref103)
    if not init_id["valid"]: raise RuntimeError(f"V48.107 initial V103 function identity failed {init_id}")
    training=train_refiner(refiner=refiner,readout=readout,train_obj=train_obj,dev_obj=dev_obj,device=device,scales=scales)
    ts,tr,tn=_predict_all(refiner=refiner,readout=readout,obj=train_obj,device=device); ds,dr,dn=_predict_all(refiner=refiner,readout=readout,obj=dev_obj,device=device); cs,cr,cn=_predict_all(refiner=refiner,readout=readout,obj=cert_obj,device=device)
    cells,contracts=_evaluate_cells(dev_obj,cert_obj,ds,dr,cs,cr,v93); state_id=_state_identity(cells,ref103)
    if not state_id["valid"] or max(tn,dn,cn)!=0.0: raise RuntimeError(f"V48.107 nominal-state identity failed {state_id} {tn,dn,cn}")
    tl,tp=_semantic_full(train_obj,ts,tr,device,scales); dl,dp=_semantic_full(dev_obj,ds,dr,device,scales); cl,cp=_semantic_full(cert_obj,cs,cr,device,scales)
    with torch.no_grad():
        train_ol,train_op=_orientation_epoch(refiner=refiner,readout=readout,obj=train_obj,device=device,scales=scales,train=False)
        dev_ol,dev_op=_orientation_epoch(refiner=refiner,readout=readout,obj=dev_obj,device=device,scales=scales,train=False)
        cert_ol,cert_op=_orientation_epoch(refiner=refiner,readout=readout,obj=cert_obj,device=device,scales=scales,train=False)
    result={
        "schema":"ocrap-v48.107-fnao-result-v1","engineering_version":ENGINEERING_VERSION,"algorithm_name":ALGORITHM_NAME,"valid":True,"variant":a.variant,
        "planner_parameters_trained":0,"stage_i_first_block_parameters_trained":refiner.parameter_count,"stage_i_other_parameters_trained":0,
        "frozen_stage_i_second_block":True,"frozen_v103_readout_parameters":1540,"root_decoder_parameters_trained":0,"source_parameters_trained":0,
        "relative_ranker_modified":False,"boundary_transport":False,"regime_conditioning":False,"teacher_metadata_input_to_model":False,
        "ordinal_action_orientation_objective":True,"ordinal_target_magnitude_discarded_after_sign":True,"nominal_first_block_exact_identity":True,
        "nominal_final_memory_exact_identity":True,"state_metrics_exact_v103":state_id,"initial_v103_function_identity":init_id,
        "checkpoint":str(a.checkpoint.resolve()),"checkpoint_sha256":sha256(a.checkpoint),"v103_state":str(a.v103_state.resolve()),"v103_state_sha256":sha256(a.v103_state),
        "semantic_metric_scales":scales,"training":training,
        "orientation_loss":{"train":{"total":train_ol,**train_op},"dev":{"total":dev_ol,**dev_op},"certificate":{"total":cert_ol,**cert_op}},
        "dense_metrics":{"train":_dense_metrics(train_obj,ts,tr),"dev":_dense_metrics(dev_obj,ds,dr),"certificate":_dense_metrics(cert_obj,cs,cr)},
        "semantic_loss":{"train":{"total":tl,**tp},"dev":{"total":dl,**dp},"certificate":{"total":cl,**cp}},"cells":cells,"evaluation_contracts":contracts,
        "feature_contracts":{"train":train_obj.get("feature_only_dataset_contract"),"dev":dev_obj.get("feature_only_dataset_contract"),"certificate":cert_obj.get("feature_only_dataset_contract")},
        "historical_final_reconstruction_max_abs":max(float(train_obj["historical_final_identity_max_abs"]),float(dev_obj["historical_final_identity_max_abs"]),float(cert_obj["historical_final_identity_max_abs"])),
        "elapsed_seconds":float(time.perf_counter()-t0),
    }
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    torch.save({
        "schema":"ocrap-v48.107-fnao-state-v1","engineering_version":ENGINEERING_VERSION,"algorithm_name":ALGORITHM_NAME,"variant":a.variant,
        "adapted_first_block_state":{k:v.detach().cpu() for k,v in refiner.adapted_first.state_dict().items()},"stage_i_first_block_parameter_count":refiner.parameter_count,
        "v103_state_sha256":sha256(a.v103_state),"checkpoint_sha256":sha256(a.checkpoint),"semantic_metric_scales":scales,
        "training":{k:v for k,v in training.items() if k!="history"},
    },a.state_output)
    print(json.dumps({"valid":True,"variant":a.variant,"parameters":refiner.parameter_count,"best_epoch":training["best_epoch"]})); return 0

if __name__=="__main__": raise SystemExit(main())
