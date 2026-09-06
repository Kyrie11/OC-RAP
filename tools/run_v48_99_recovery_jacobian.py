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
from ocrap.v48_96_support_reserve_root_observability import feature_only_dataset_cfg
from ocrap.v48_97_executable_recovery_state import ExecutableRecoverySufficientState
from ocrap.v48_99_recovery_jacobian import (
    ALGORITHM_NAME,
    ENGINEERING_VERSION,
    ObservationConditionedRecoveryJacobian,
    normalized_tangent_loss,
    semantic_attention_weights,
    semantic_delta_scales,
)
from tools.run_v48_97_executable_recovery_state import (
    ROLES,
    _action_metric,
    _dense_metrics,
    _evaluation_contract,
    _index_rows,
    _pair_indices,
    _role_rows,
    _root_probs,
    _state_metric,
    build_v93_map,
    sha256,
)


def _cache_key(checkpoint: Path, index_path: Path, erss_state: Path, variant: str) -> str:
    payload = {
        "engineering_version": ENGINEERING_VERSION,
        "checkpoint": sha256(checkpoint),
        "index": sha256(index_path),
        "erss_state": sha256(erss_state),
        "variant": variant,
        "kind": "frozen_root_semantic_features",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _load_erss(path: Path, device: torch.device) -> tuple[ExecutableRecoverySufficientState, dict[str, Any]]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict) or str(obj.get("engineering_version")) != "v48.97.2-OC-ERSS-STRATAFIX":
        raise ValueError("V48.99 requires a V48.97.2 ERSS state")
    d = int(obj.get("d_model", 0))
    m = ExecutableRecoverySufficientState(d)
    m.load_state_dict(obj["state_dict"], strict=True)
    m.eval().to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    return m, obj


def _physical_action_vector(model, x: torch.Tensor) -> torch.Tensor:
    enc = model.encoder
    if not hasattr(enc, "_split"):
        raise TypeError("V48.99 requires StructuredTokenEncoder._split")
    (
        _ego, prefix_param, _macro, _scalar, prefix_state, control,
        _agent_summary, _agents, _bev, _route, _maps, _dyn,
    ) = enc._split(x)
    return torch.cat([prefix_param.float(), prefix_state.float(), control.float()], dim=-1)


def extract_features(
    *, checkpoint: Path, erss_state: Path, index_path: Path, cache_dir: Path,
    device: str, variant: str, batch_size: int = 256,
) -> dict[str, Any]:
    key = _cache_key(checkpoint, index_path, erss_state, variant)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{key}.pt"
    if cache.is_file():
        obj = torch.load(cache, map_location="cpu", weights_only=False)
        if obj.get("cache_key") == key:
            return obj

    rows = _index_rows(index_path)
    paths = [Path(str(r["path"])) for r in rows]
    runtime = {"training": {"device": device}}
    bundle = load_model_bundle(checkpoint, runtime)
    if bundle is None:
        raise RuntimeError(f"cannot load checkpoint {checkpoint}")
    model = bundle.model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    dev = bundle.device
    erss, _ = _load_erss(erss_state, dev)

    cfg, feature_cfg_event = feature_only_dataset_cfg(
        bundle.cfg, cache_dir=str(cache_dir / "tensor"), workers=8
    )
    ds = OCRAPSampleDataset(paths, cfg)
    if ds.absolute_truth_contract_event.get("enabled") or ds.action_response_truth_event.get("enabled"):
        raise RuntimeError("V48.99 feature-only dataset unexpectedly attached truth sidecars")
    if [str(p.resolve()) for p in paths] != [str(p.resolve()) for p in ds.paths]:
        raise RuntimeError("V48.99 dataset path order differs from teacher-index order")

    xs: list[torch.Tensor] = []
    rvs: list[torch.Tensor] = []
    for i in range(len(ds)):
        it = ds[i]
        xs.append(it["x"].detach().float().cpu())
        rvs.append(it["root_valid"].detach().bool().cpu())
    x = torch.stack(xs, dim=0)
    rv = torch.stack(rvs, dim=0)

    roots: list[torch.Tensor] = []
    probs: list[torch.Tensor] = []
    sws: list[torch.Tensor] = []
    rws: list[torch.Tensor] = []
    sup: list[torch.Tensor] = []
    res: list[torch.Tensor] = []
    hs: list[torch.Tensor] = []
    hr: list[torch.Tensor] = []
    act: list[torch.Tensor] = []
    with torch.no_grad():
        for st in range(0, len(x), batch_size):
            xx = x[st: st + batch_size].to(dev)
            vv = rv[st: st + batch_size].to(dev)
            mem = model._scene_tokens(xx)
            rt = model._decode_roots(mem)
            rp = _root_probs(model, rt, vv)
            out = erss(rt, rp, vv)
            ws, wr = semantic_attention_weights(erss, rt, rp, vv)
            av = _physical_action_vector(model, xx)
            roots.append(rt.detach().float().cpu())
            probs.append(rp.detach().float().cpu())
            sws.append(ws.detach().float().cpu())
            rws.append(wr.detach().float().cpu())
            sup.append(out["support"].detach().float().cpu())
            res.append(out["reserve_debt"].detach().float().cpu())
            hs.append(out["support_state"].detach().float().cpu())
            hr.append(out["reserve_state"].detach().float().cpu())
            act.append(av.detach().float().cpu())

    obj = {
        "cache_key": key,
        "checkpoint": str(checkpoint.resolve()),
        "erss_state": str(erss_state.resolve()),
        "index": str(index_path.resolve()),
        "rows": rows,
        "root_tokens": torch.cat(roots, dim=0),
        "root_probs": torch.cat(probs, dim=0),
        "root_valid": rv,
        "support_weights": torch.cat(sws, dim=0),
        "reserve_weights": torch.cat(rws, dim=0),
        "base_support": torch.cat(sup, dim=0),
        "base_reserve": torch.cat(res, dim=0),
        "support_state": torch.cat(hs, dim=0),
        "reserve_state": torch.cat(hr, dim=0),
        "action": torch.cat(act, dim=0),
        "feature_only_dataset_contract": feature_cfg_event,
        "tensor_cache_event": ds.tensor_cache_event,
    }
    torch.save(obj, cache)
    return obj


def _load_frozen_heads(checkpoint: Path, erss_state: Path, device: str):
    runtime = {"training": {"device": device}}
    bundle = load_model_bundle(checkpoint, runtime)
    if bundle is None:
        raise RuntimeError(f"cannot load checkpoint {checkpoint}")
    model = bundle.model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    erss, erss_obj = _load_erss(erss_state, bundle.device)
    return bundle, model, erss, erss_obj


def _predict(
    *, model, erss, adapter: ObservationConditionedRecoveryJacobian,
    obj: dict[str, Any], device: torch.device, batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    rows = obj["rows"]
    ci, ni = _pair_indices(rows)
    n = len(rows)
    support = obj["base_support"].numpy().copy()
    reserve = obj["base_reserve"].numpy().copy()
    if len(ci) == 0:
        return support, reserve
    adapter.eval()
    with torch.no_grad():
        for st in range(0, len(ci), batch_size):
            ids = ci[st: st + batch_size]
            ns = ni[st: st + batch_size]
            rt = obj["root_tokens"][ids].to(device)
            vv = obj["root_valid"][ids].to(device)
            ws = obj["support_weights"][ids].to(device)
            wr = obj["reserve_weights"][ids].to(device)
            da = (obj["action"][ids] - obj["action"][ns]).to(device)
            h1 = obj["support_state"][ns].to(device)
            h2 = obj["reserve_state"][ns].to(device)
            new_rt = adapter(
                root_tokens=rt, support_weights=ws, reserve_weights=wr,
                action_delta=da, nominal_support_state=h1, nominal_reserve_state=h2,
            )
            rp = _root_probs(model, new_rt, vv)
            out = erss(new_rt, rp, vv)
            support[ids] = out["support"].detach().float().cpu().numpy()
            reserve[ids] = out["reserve_debt"].detach().float().cpu().numpy()
    return support, reserve


def _training_scales(obj: dict[str, Any]) -> tuple[float, float]:
    rows = obj["rows"]
    ci, ni = _pair_indices(rows)
    td = torch.tensor([float(r["teacher_drs"]) for r in rows], dtype=torch.float32)
    tr = torch.tensor([float(r["teacher_r_dep"]) for r in rows], dtype=torch.float32)
    ds = td[torch.from_numpy(ci)] - td[torch.from_numpy(ni)]
    dr = tr[torch.from_numpy(ci)] - tr[torch.from_numpy(ni)]
    ss, sr = semantic_delta_scales(ds, dr)
    return float(ss.item()), float(sr.item())


def _loss_eval(
    *, model, erss, adapter, obj: dict[str, Any], device: torch.device,
    support_scale: float, reserve_scale: float, batch_size: int = 512,
) -> tuple[float, dict[str, float]]:
    rows = obj["rows"]
    ci, ni = _pair_indices(rows)
    td = np.asarray([float(r["teacher_drs"]) for r in rows], dtype=np.float32)
    tr = np.asarray([float(r["teacher_r_dep"]) for r in rows], dtype=np.float32)
    total_s = total_r = 0.0
    total_n = 0
    adapter.eval()
    with torch.no_grad():
        for st in range(0, len(ci), batch_size):
            ids = ci[st: st + batch_size]
            ns = ni[st: st + batch_size]
            rt = obj["root_tokens"][ids].to(device)
            vv = obj["root_valid"][ids].to(device)
            ws = obj["support_weights"][ids].to(device)
            wr = obj["reserve_weights"][ids].to(device)
            da = (obj["action"][ids] - obj["action"][ns]).to(device)
            h1 = obj["support_state"][ns].to(device)
            h2 = obj["reserve_state"][ns].to(device)
            new_rt = adapter(root_tokens=rt, support_weights=ws, reserve_weights=wr,
                             action_delta=da, nominal_support_state=h1, nominal_reserve_state=h2)
            rp = _root_probs(model, new_rt, vv)
            out = erss(new_rt, rp, vv)
            ps = out["support"].float() - obj["base_support"][ns].to(device)
            pr = out["reserve_debt"].float() - obj["base_reserve"][ns].to(device)
            ts = torch.tensor(td[ids] - td[ns], device=device)
            trr = torch.tensor(tr[ids] - tr[ns], device=device)
            _, parts = normalized_tangent_loss(ps, pr, ts, trr, support_scale, reserve_scale)
            total_s += float(parts["delta_support_normalized"].item()) * len(ids)
            total_r += float(parts["delta_reserve_normalized"].item()) * len(ids)
            total_n += len(ids)
    ds = total_s / max(total_n, 1)
    dr = total_r / max(total_n, 1)
    return (ds + dr) / 2.0, {"delta_support_normalized": ds, "delta_reserve_normalized": dr}


def train_adapter(
    *, model, erss, adapter: ObservationConditionedRecoveryJacobian,
    train_obj: dict[str, Any], dev_obj: dict[str, Any], device: torch.device,
    support_scale: float, reserve_scale: float,
    max_epochs: int = 60, patience: int = 10, batch_size: int = 512,
) -> dict[str, Any]:
    torch.manual_seed(99); np.random.seed(99); random.seed(99)
    adapter.to(device)
    rows = train_obj["rows"]
    ci, ni = _pair_indices(rows)
    if len(ci) == 0:
        raise RuntimeError("V48.99 train split has no candidate/nominal pairs")
    td = np.asarray([float(r["teacher_drs"]) for r in rows], dtype=np.float32)
    tr = np.asarray([float(r["teacher_r_dep"]) for r in rows], dtype=np.float32)
    opt = torch.optim.AdamW(adapter.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    best = float("inf"); best_state = None; best_epoch = -1; stale = 0; history: list[dict[str, Any]] = []
    for epoch in range(max_epochs):
        adapter.train(); total_s = total_r = 0.0; total_n = 0
        for st in range(0, len(ci), batch_size):
            ids = ci[st: st + batch_size]
            ns = ni[st: st + batch_size]
            rt = train_obj["root_tokens"][ids].to(device)
            vv = train_obj["root_valid"][ids].to(device)
            ws = train_obj["support_weights"][ids].to(device)
            wr = train_obj["reserve_weights"][ids].to(device)
            da = (train_obj["action"][ids] - train_obj["action"][ns]).to(device)
            h1 = train_obj["support_state"][ns].to(device)
            h2 = train_obj["reserve_state"][ns].to(device)
            new_rt = adapter(root_tokens=rt, support_weights=ws, reserve_weights=wr,
                             action_delta=da, nominal_support_state=h1, nominal_reserve_state=h2)
            rp = _root_probs(model, new_rt, vv)
            out = erss(new_rt, rp, vv)
            ps = out["support"].float() - train_obj["base_support"][ns].to(device)
            pr = out["reserve_debt"].float() - train_obj["base_reserve"][ns].to(device)
            ts = torch.tensor(td[ids] - td[ns], device=device)
            trr = torch.tensor(tr[ids] - tr[ns], device=device)
            loss, parts = normalized_tangent_loss(ps, pr, ts, trr, support_scale, reserve_scale)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(adapter.parameters(), 5.0); opt.step()
            total_s += float(parts["delta_support_normalized"].item()) * len(ids)
            total_r += float(parts["delta_reserve_normalized"].item()) * len(ids)
            total_n += len(ids)
        dev_loss, dev_parts = _loss_eval(
            model=model, erss=erss, adapter=adapter, obj=dev_obj, device=device,
            support_scale=support_scale, reserve_scale=reserve_scale, batch_size=batch_size,
        )
        history.append({
            "epoch": epoch,
            "train_loss": (total_s + total_r) / (2.0 * max(total_n, 1)),
            "train_parts": {
                "delta_support_normalized": total_s / max(total_n, 1),
                "delta_reserve_normalized": total_r / max(total_n, 1),
            },
            "dev_loss": dev_loss,
            "dev_parts": dev_parts,
        })
        if dev_loss < best - 1.0e-5:
            best = dev_loss; best_epoch = epoch; stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()}
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("V48.99 training produced no checkpoint")
    adapter.load_state_dict(best_state); adapter.to(device)
    return {"best_epoch": best_epoch, "best_dev_loss": best, "epochs_completed": len(history), "history": history}


def _nominal_identity(rows: list[dict[str, Any]], base_s: np.ndarray, base_r: np.ndarray,
                      new_s: np.ndarray, new_r: np.ndarray) -> dict[str, float]:
    ids = np.asarray([i for i, r in enumerate(rows) if bool(r.get("nominal", False))], dtype=np.int64)
    if len(ids) == 0:
        raise RuntimeError("V48.99 nominal identity has no nominal rows")
    return {
        "rows": int(len(ids)),
        "support_max_abs_error": float(np.max(np.abs(new_s[ids] - base_s[ids]))),
        "reserve_max_abs_error": float(np.max(np.abs(new_r[ids] - base_r[ids]))),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--erss-state", type=Path, required=True)
    ap.add_argument("--train-index", type=Path, required=True)
    ap.add_argument("--dev-index", type=Path, required=True)
    ap.add_argument("--certificate-index", type=Path, required=True)
    ap.add_argument("--v93-audit", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--variant", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--state-output", type=Path, required=True)
    a = ap.parse_args(); t0 = time.perf_counter()

    train_obj = extract_features(checkpoint=a.checkpoint, erss_state=a.erss_state, index_path=a.train_index,
                                 cache_dir=a.cache_dir / "train", device=a.device, variant=a.variant)
    dev_obj = extract_features(checkpoint=a.checkpoint, erss_state=a.erss_state, index_path=a.dev_index,
                               cache_dir=a.cache_dir / "dev", device=a.device, variant=a.variant)
    cert_obj = extract_features(checkpoint=a.checkpoint, erss_state=a.erss_state, index_path=a.certificate_index,
                                cache_dir=a.cache_dir / "certificate", device=a.device, variant=a.variant)
    bundle, model, erss, erss_obj = _load_frozen_heads(a.checkpoint, a.erss_state, a.device)
    device = bundle.device
    d_model = int(train_obj["root_tokens"].shape[-1]); action_dim = int(train_obj["action"].shape[-1])
    adapter = ObservationConditionedRecoveryJacobian(d_model=d_model, action_dim=action_dim)
    s_scale, r_scale = _training_scales(train_obj)
    training = train_adapter(model=model, erss=erss, adapter=adapter, train_obj=train_obj, dev_obj=dev_obj,
                             device=device, support_scale=s_scale, reserve_scale=r_scale)

    train_support, train_reserve = _predict(model=model, erss=erss, adapter=adapter, obj=train_obj, device=device)
    dev_support, dev_reserve = _predict(model=model, erss=erss, adapter=adapter, obj=dev_obj, device=device)
    cert_support, cert_reserve = _predict(model=model, erss=erss, adapter=adapter, obj=cert_obj, device=device)
    v93 = build_v93_map(a.v93_audit)

    result: dict[str, Any] = {
        "schema": "ocrap-v48.99-ocrj-result-v1",
        "engineering_version": ENGINEERING_VERSION,
        "algorithm_name": ALGORITHM_NAME,
        "valid": True,
        "variant": a.variant,
        "planner_parameters_trained": 0,
        "source_parameters_trained": 0,
        "erss_parameters_trained": 0,
        "stage_i_parameters_trained": 0,
        "root_jacobian_parameters_trained": adapter.trainable_parameter_count,
        "semantic_rank": 2,
        "state_conditioned_control_affine": True,
        "root_slot_bijection_assumed": False,
        "regime_conditioning": False,
        "teacher_metadata_input_to_model": False,
        "checkpoint": str(a.checkpoint.resolve()),
        "erss_state": str(a.erss_state.resolve()),
        "semantic_metric_scales": {"support": s_scale, "reserve": r_scale},
        "training": training,
        "dense_metrics": {
            "train": _dense_metrics(train_obj, train_support, train_reserve),
            "dev": _dense_metrics(dev_obj, dev_support, dev_reserve),
            "certificate": _dense_metrics(cert_obj, cert_support, cert_reserve),
        },
        "cells": {}, "evaluation_contracts": {},
        "feature_contracts": {
            "train": train_obj.get("feature_only_dataset_contract"),
            "dev": dev_obj.get("feature_only_dataset_contract"),
            "certificate": cert_obj.get("feature_only_dataset_contract"),
        },
        "nominal_identity": {
            "dev": _nominal_identity(dev_obj["rows"], dev_obj["base_support"].numpy(), dev_obj["base_reserve"].numpy(), dev_support, dev_reserve),
            "certificate": _nominal_identity(cert_obj["rows"], cert_obj["base_support"].numpy(), cert_obj["base_reserve"].numpy(), cert_support, cert_reserve),
        },
    }
    for role in ROLES:
        obj = dev_obj if role.startswith("dev_") else cert_obj
        sp = dev_support if role.startswith("dev_") else cert_support
        rs = dev_reserve if role.startswith("dev_") else cert_reserve
        rr = _role_rows(obj, sp, rs, role, v93)
        contract = _evaluation_contract(rr, role)
        result["evaluation_contracts"][role] = contract
        if not contract["valid"]:
            raise RuntimeError(f"V48.99 evaluation join fail-closed for {role}: {contract['errors']}")
        state = _state_metric(rr)
        sup_true, sup_shuf = _action_metric(rr, "drs_activation", "support")
        res_true, res_shuf = _action_metric(rr, "deployability_gain", "reserve")
        for name, metric in (("state", state), ("support", sup_true), ("reserve", res_true)):
            if int(metric.get("rows", 0)) <= 0 or metric.get("auc") is None:
                raise RuntimeError(f"V48.99 {role} {name} metric empty/null")
        result["cells"][role] = {
            "state": state, "support_true": sup_true, "support_shuffled": sup_shuf,
            "reserve_true": res_true, "reserve_shuffled": res_shuf,
        }
    result["elapsed_seconds"] = float(time.perf_counter() - t0)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    torch.save({
        "schema": "ocrap-v48.99-ocrj-state-v1",
        "engineering_version": ENGINEERING_VERSION,
        "algorithm_name": ALGORITHM_NAME,
        "variant": a.variant,
        "d_model": d_model,
        "action_dim": action_dim,
        "state_dict": {k: v.detach().cpu() for k, v in adapter.state_dict().items()},
        "trainable_parameter_count": adapter.trainable_parameter_count,
        "semantic_metric_scales": {"support": s_scale, "reserve": r_scale},
        "erss_state_sha256": sha256(a.erss_state),
        "training": {k: v for k, v in training.items() if k != "history"},
    }, a.state_output)
    print(json.dumps({"valid": True, "variant": a.variant, "parameters": adapter.trainable_parameter_count,
                      "support_scale": s_scale, "reserve_scale": r_scale}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
