#!/usr/bin/env python3
from __future__ import annotations
import argparse, copy, json, math, os, random, time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ocrap.models.data import OCRAPSampleDataset, iter_sample_paths_many
from ocrap.models.inference import load_model_bundle

DEPLOYABLE_MACROS = {2, 3, 5, 7}
POSITIVE_GAIN = 0.015


def rows(path: Path) -> list[dict[str, Any]]:
    out=[]
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if line: out.append(json.loads(line))
    return out


def auc(labels, scores):
    y=np.asarray(labels, dtype=np.int64); s=np.asarray(scores, dtype=np.float64)
    ok=np.isfinite(s); y=y[ok]; s=s[ok]
    pos=int(y.sum()); neg=int(len(y)-pos)
    if pos==0 or neg==0: return None
    order=np.argsort(s, kind='mergesort'); ranks=np.empty(len(s), dtype=np.float64)
    i=0
    while i<len(s):
        j=i+1
        while j<len(s) and s[order[j]]==s[order[i]]: j+=1
        r=(i+j+1)/2.0; ranks[order[i:j]]=r; i=j
    return float((ranks[y==1].sum() - pos*(pos+1)/2.0)/(pos*neg))


def pearson(a,b):
    a=np.asarray(a,dtype=np.float64); b=np.asarray(b,dtype=np.float64)
    ok=np.isfinite(a)&np.isfinite(b); a=a[ok]; b=b[ok]
    if len(a)<2 or float(a.std())<1e-12 or float(b.std())<1e-12: return None
    return float(np.corrcoef(a,b)[0,1])


def _root_probs(model, root_tokens, root_valid):
    logits=model.root_logit_head(root_tokens).squeeze(-1).float()
    mask=root_valid.bool()
    logits=logits.masked_fill(~mask, -1.0e9)
    p=torch.softmax(logits, dim=-1)*mask.float()
    return p/p.sum(dim=-1,keepdim=True).clamp_min(1.0e-12)


def _stats(values: torch.Tensor, weights: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    # values [N,K,D], weights [N,K], valid [N,K]
    w=weights*valid.float(); w=w/w.sum(dim=1,keepdim=True).clamp_min(1.0e-12)
    mean=(w.unsqueeze(-1)*values).sum(dim=1)
    var=(w.unsqueeze(-1)*(values-mean.unsqueeze(1)).pow(2)).sum(dim=1)
    std=var.clamp_min(0).sqrt()
    inf=torch.tensor(float('inf'),device=values.device,dtype=values.dtype)
    ninf=torch.tensor(float('-inf'),device=values.device,dtype=values.dtype)
    vmax=torch.where(valid.unsqueeze(-1),values,ninf).amax(dim=1)
    vmin=torch.where(valid.unsqueeze(-1),values,inf).amin(dim=1)
    anyv=valid.any(dim=1,keepdim=True)
    vmax=torch.where(anyv,vmax,torch.zeros_like(vmax)); vmin=torch.where(anyv,vmin,torch.zeros_like(vmin))
    return torch.cat([mean,std,vmax,vmin],dim=-1)


def _stack(items: list[dict[str, torch.Tensor]], key: str) -> torch.Tensor:
    return torch.stack([x[key] for x in items],dim=0)


def _index_labels(index_rows: list[dict[str,Any]]):
    groups=defaultdict(list)
    for r in index_rows:
        key=(int(r['bucket']),str(r['scene']),int(r['time']))
        groups[key].append(r)
    out={}
    for key,rs in groups.items():
        nom=[r for r in rs if bool(r.get('nominal',False))]
        if len(nom)!=1: continue
        n=nom[0]; n_pcd=float(n['teacher_pcd'])
        eligible=[r for r in rs if (not bool(r.get('nominal',False))) and int(r.get('macro',-1)) in DEPLOYABLE_MACROS]
        if len(eligible)<2: continue  # paired permutation control needs >=2 actions
        eligible=sorted(eligible,key=lambda z:int(z.get('candidate',0)))
        for r in eligible:
            adv=float(r['teacher_pcd'])-n_pcd
            harm=bool(r.get('component_harmful',False))
            out[str(Path(r['path']).resolve())]={
                'group':key,'candidate':int(r.get('candidate',0)),'bucket':int(r['bucket']),
                'adv':adv,'harm':float(harm),'safe':float(adv>=POSITIVE_GAIN and not harm),
            }
    return out


def extract_features(*, checkpoint:Path, dataset_roots:str, label_index:Path, cache_dir:Path, device:str):
    runtime={'training':{'device':device}}
    bundle=load_model_bundle(checkpoint,runtime)
    if bundle is None: raise RuntimeError(f'cannot load checkpoint {checkpoint}')
    model=bundle.model.eval(); dev=bundle.device
    cfg=copy.deepcopy(bundle.cfg); cfg.setdefault('training',{})
    cfg['training']['persistent_tensor_cache']=True; cfg['training']['persistent_tensor_cache_dir']=str(cache_dir)
    cfg['training']['persistent_tensor_cache_build_workers']=8
    paths=iter_sample_paths_many(dataset_roots)
    label_map=_index_labels(rows(label_index))
    # Retain every sample in a group that contains at least two deployable candidate actions.
    all_rows=rows(label_index); group_for_path={str(Path(r['path']).resolve()):(int(r['bucket']),str(r['scene']),int(r['time'])) for r in all_rows}
    wanted_groups={v['group'] for v in label_map.values()}
    paths=[p for p in iter_sample_paths_many(dataset_roots) if group_for_path.get(str(p.resolve())) in wanted_groups]
    ds=OCRAPSampleDataset(paths,cfg)
    idx_by_path={str(p.resolve()):i for i,p in enumerate(ds.paths)}
    rows_by_group=defaultdict(list)
    for r in all_rows:
        key=(int(r['bucket']),str(r['scene']),int(r['time']))
        if key in wanted_groups and str(Path(r['path']).resolve()) in idx_by_path: rows_by_group[key].append(r)
    records=[]
    with torch.no_grad():
        for key,rs in rows_by_group.items():
            nom=[r for r in rs if bool(r.get('nominal',False))]
            elig=[r for r in rs if str(Path(r['path']).resolve()) in label_map]
            if len(nom)!=1 or len(elig)<2: continue
            ordered=[nom[0]]+sorted(elig,key=lambda z:int(z.get('candidate',0)))
            items=[ds[idx_by_path[str(Path(r['path']).resolve())]] for r in ordered]
            x=_stack(items,'x').to(dev); rv=_stack(items,'root_valid').to(dev)
            memory=model._scene_tokens(x); rt=model._decode_roots(memory.detach())
            p=_root_probs(model,rt,rv); p0=p[0:1].expand(rt.shape[0]-1,-1)
            r0=rt[0:1].expand(rt.shape[0]-1,-1,-1)
            delta=rt[1:]-r0; valid=rv[1:].bool() & rv[0:1].bool().expand_as(rv[1:])
            delta_stats=_stats(delta,p0,valid)
            state_stats=_stats(r0,p0,valid)
            # Same-dimensional deterministic state conditioning: probe capacity is identical
            # to delta-only, so any gain cannot be attributed to a larger linear head.
            context=delta_stats*(1.0+torch.tanh(state_stats))
            dnp=delta_stats.cpu().numpy(); cnp=context.cpu().numpy()
            for j,r in enumerate(ordered[1:]):
                lab=label_map[str(Path(r['path']).resolve())]
                records.append({**lab,'path':str(Path(r['path']).resolve()),'delta':dnp[j],'context':cnp[j]})
    return records, {'tensor_cache_event':ds.tensor_cache_event,'rows':len(records),'groups':len(set(r['group'] for r in records))}


def permute_within_group(recs, key):
    groups=defaultdict(list)
    for i,r in enumerate(recs): groups[tuple(r['group'])].append(i)
    out=np.stack([r[key] for r in recs]).copy()
    for ids in groups.values():
        ids=sorted(ids,key=lambda i:recs[i]['candidate'])
        vals=out[ids].copy(); out[ids]=np.roll(vals,1,axis=0)
    return out


class Probe(nn.Module):
    def __init__(self,d): super().__init__(); self.linear=nn.Linear(d,3); nn.init.zeros_(self.linear.weight); nn.init.zeros_(self.linear.bias)
    def forward(self,x): return self.linear(x)


def fit_probe(X,y_safe,y_harm,y_adv,Xv,ysv,yhv,yav,device,seed=84):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    mu=X.mean(0,keepdims=True); sd=X.std(0,keepdims=True); sd=np.where(sd>1e-6,sd,1.0)
    X=(X-mu)/sd; Xv=(Xv-mu)/sd
    dev=torch.device(device); model=Probe(X.shape[1]).to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    xs=torch.tensor(X,dtype=torch.float32); ss=torch.tensor(y_safe,dtype=torch.float32); hh=torch.tensor(y_harm,dtype=torch.float32); aa=torch.tensor(y_adv,dtype=torch.float32)
    xv=torch.tensor(Xv,dtype=torch.float32,device=dev); sv=torch.tensor(ysv,dtype=torch.float32,device=dev); hv=torch.tensor(yhv,dtype=torch.float32,device=dev); av=torch.tensor(yav,dtype=torch.float32,device=dev)
    ps=max(float((len(y_safe)-sum(y_safe))/max(sum(y_safe),1)),1.0); ph=max(float((len(y_harm)-sum(y_harm))/max(sum(y_harm),1)),1.0)
    bce_s=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(ps,device=dev)); bce_h=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(ph,device=dev))
    hub=nn.SmoothL1Loss(beta=0.05)
    best=None; best_loss=float('inf'); batch=1024
    for epoch in range(40):
        order=torch.randperm(len(xs))
        model.train()
        for start in range(0,len(xs),batch):
            ix=order[start:start+batch]; xb=xs[ix].to(dev); sb=ss[ix].to(dev); hb=hh[ix].to(dev); ab=aa[ix].to(dev)
            o=model(xb); loss=bce_s(o[:,0],sb)+bce_h(o[:,1],hb)+hub(o[:,2],ab)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            o=model(xv); vl=bce_s(o[:,0],sv)+bce_h(o[:,1],hv)+hub(o[:,2],av)
        if float(vl)<best_loss-1e-7:
            best_loss=float(vl); best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; best_epoch=epoch
    model.load_state_dict(best); model.eval()
    return model,mu,sd,best_epoch,best_loss


def metrics(model,mu,sd,recs,key,X_override=None,device='cpu'):
    X=np.stack([r[key] for r in recs]) if X_override is None else X_override
    X=(X-mu)/sd; dev=torch.device(device)
    with torch.no_grad(): o=model(torch.tensor(X,dtype=torch.float32,device=dev)).cpu().numpy()
    safe=np.asarray([r['safe'] for r in recs]); harm=np.asarray([r['harm'] for r in recs]); adv=np.asarray([r['adv'] for r in recs])
    groups=defaultdict(list)
    for i,r in enumerate(recs): groups[tuple(r['group'])].append(i)
    powered=[ids for ids in groups.values() if any(safe[i]>0.5 for i in ids)]
    top1=None
    if powered:
        top1=float(np.mean([safe[max(ids,key=lambda i:o[i,0])]>0.5 for ids in powered]))
    return {'rows':len(recs),'safe_positive_rows':int(safe.sum()),'harmful_rows':int(harm.sum()),'safe_auc':auc(safe,o[:,0]),'harm_auc':auc(harm,o[:,1]),'adv_pearson':pearson(adv,o[:,2]),'adv_mae':float(np.mean(np.abs(adv-o[:,2]))),'top1_safe_recall':top1,'powered_groups':len(powered)}


def split_records(recs):
    return {1:[r for r in recs if int(r['bucket'])==1],2:[r for r in recs if int(r['bucket'])==2]}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',type=Path,required=True); ap.add_argument('--train-dataset',required=True); ap.add_argument('--dev-dataset',required=True); ap.add_argument('--certificate-dataset',required=True); ap.add_argument('--train-index',type=Path,required=True); ap.add_argument('--dev-index',type=Path,required=True); ap.add_argument('--certificate-index',type=Path,required=True); ap.add_argument('--cache-dir',type=Path,required=True); ap.add_argument('--device',default='cuda'); ap.add_argument('--variant',required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    t0=time.perf_counter()
    tr,evt_tr=extract_features(checkpoint=a.checkpoint,dataset_roots=a.train_dataset,label_index=a.train_index,cache_dir=a.cache_dir,device=a.device)
    dv,evt_dv=extract_features(checkpoint=a.checkpoint,dataset_roots=a.dev_dataset,label_index=a.dev_index,cache_dir=a.cache_dir,device=a.device)
    ce,evt_ce=extract_features(checkpoint=a.checkpoint,dataset_roots=a.certificate_dataset,label_index=a.certificate_index,cache_dir=a.cache_dir,device=a.device)
    if not tr or not dv or not ce: raise SystemExit('empty observability feature set')
    result={'schema':'ocrap-v48.84-saop-probe-v1','engineering_version':'v48.84.0-OC-SAOP','variant':a.variant,'valid':True,'checkpoint':str(a.checkpoint),'events':{'train':evt_tr,'dev':evt_dv,'certificate':evt_ce},'probes':{}}
    for key in ['delta','context']:
        X=np.stack([r[key] for r in tr]); Xv=np.stack([r[key] for r in dv]); Xt_perm=permute_within_group(tr,key); Xv_perm=permute_within_group(dv,key)
        ys=np.asarray([r['safe'] for r in tr]); yh=np.asarray([r['harm'] for r in tr]); ya=np.asarray([r['adv'] for r in tr]); ysv=np.asarray([r['safe'] for r in dv]); yhv=np.asarray([r['harm'] for r in dv]); yav=np.asarray([r['adv'] for r in dv])
        m,mu,sd,be,bl=fit_probe(X,ys,yh,ya,Xv,ysv,yhv,yav,a.device,seed=84)
        mp,mup,sdp,bep,blp=fit_probe(Xt_perm,ys,yh,ya,Xv_perm,ysv,yhv,yav,a.device,seed=84)
        block={'best_epoch':be,'best_val_loss':bl,'shuffled_best_epoch':bep,'shuffled_best_val_loss':blp,'true':{},'shuffled':{}}
        for split_name,recs in [('dev_near',split_records(dv)[1]),('dev_contact',split_records(dv)[2]),('certificate_near',split_records(ce)[1]),('certificate_contact',split_records(ce)[2])]:
            block['true'][split_name]=metrics(m,mu,sd,recs,key,device=a.device)
            perm=permute_within_group(recs,key)
            block['shuffled'][split_name]=metrics(mp,mup,sdp,recs,key,X_override=perm,device=a.device)
        result['probes'][key]=block
    result['elapsed_seconds']=float(time.perf_counter()-t0); a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps({'valid':True,'variant':a.variant,'elapsed_seconds':result['elapsed_seconds']}))
if __name__=='__main__': main()
