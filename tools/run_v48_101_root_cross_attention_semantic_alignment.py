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
from ocrap.v48_100_joint_root_semantic_decoder import JointRootSemanticDecoder, joint_semantic_loss
from ocrap.v48_101_root_cross_attention_semantic_alignment import (
    ALGORITHM_NAME,
    ENGINEERING_VERSION,
    V100_ENGINEERING_VERSION,
    configure_cross_attention_only,
    cross_attention_parameter_contract,
    expected_cross_attention_parameter_count,
    freeze_v100_semantic_state,
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


def _cache_key(checkpoint: Path, index_path: Path, variant: str) -> str:
    payload = {
        "checkpoint": sha256(checkpoint),
        "index": sha256(index_path),
        "variant": variant,
        "kind": "frozen_scene_memory_for_recovery_semantic_alignment",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def extract_memory_features(
    *, checkpoint: Path, index_path: Path, cache_dir: Path, device: str, variant: str, batch_size: int = 256,
) -> dict[str, Any]:
    key = _cache_key(checkpoint, index_path, variant)
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
    cfg, feature_cfg_event = feature_only_dataset_cfg(bundle.cfg, cache_dir=str(cache_dir / "tensor"), workers=8)
    ds = OCRAPSampleDataset(paths, cfg)
    if ds.absolute_truth_contract_event.get("enabled") or ds.action_response_truth_event.get("enabled"):
        raise RuntimeError("V48.101 feature-only dataset unexpectedly attached truth sidecars")
    if [str(p.resolve()) for p in paths] != [str(p.resolve()) for p in ds.paths]:
        raise RuntimeError("V48.101 dataset path order differs from teacher-index order")

    memories: list[torch.Tensor] = []
    root_valid: list[torch.Tensor] = []
    with torch.no_grad():
        for st in range(0, len(ds), batch_size):
            items = [ds[i] for i in range(st, min(len(ds), st + batch_size))]
            x = torch.stack([it["x"] for it in items], dim=0).to(dev)
            rv = torch.stack([it["root_valid"] for it in items], dim=0).to(dev)
            mem = model._scene_tokens(x)
            memories.append(mem.detach().float().cpu())
            root_valid.append(rv.detach().bool().cpu())
    obj = {
        "cache_key": key,
        "checkpoint": str(checkpoint.resolve()),
        "index": str(index_path.resolve()),
        "rows": rows,
        "memory": torch.cat(memories, dim=0),
        "root_valid": torch.cat(root_valid, dim=0),
        "feature_only_dataset_contract": feature_cfg_event,
        "tensor_cache_event": ds.tensor_cache_event,
    }
    torch.save(obj, cache)
    return obj


def _load_model(checkpoint: Path, device: str):
    bundle = load_model_bundle(checkpoint, {"training": {"device": device}})
    if bundle is None:
        raise RuntimeError(f"cannot load checkpoint {checkpoint}")
    model = bundle.model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return bundle, model


def _load_v100_state(path: Path, *, variant: str) -> dict[str, Any]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict) or str(obj.get("engineering_version")) != V100_ENGINEERING_VERSION:
        raise ValueError("V48.101 requires an authoritative V48.100 JRSD state")
    if str(obj.get("variant")) != str(variant):
        raise ValueError("V48.101 V48.100 state variant mismatch")
    if int(obj.get("joint_representation_parameter_count", -1)) != 2306:
        raise ValueError("V48.101 V48.100 parameter-count mismatch")
    if int(obj.get("root_query_parameter_count", -1)) != 1536 or int(obj.get("recovery_chart_parameter_count", -1)) != 770:
        raise ValueError("V48.101 V48.100 query/chart contract mismatch")
    scales = obj.get("semantic_metric_scales") or {}
    if set(scales) != {"support", "reserve", "delta_support", "delta_reserve"} or not all(float(v) > 0 for v in scales.values()):
        raise ValueError("V48.101 V48.100 semantic scales invalid")
    return obj


def _load_v100_result(path: Path, *, variant: str) -> dict[str, Any]:
    obj = json.loads(path.read_text())
    if not obj.get("valid") or str(obj.get("engineering_version")) != V100_ENGINEERING_VERSION or str(obj.get("variant")) != str(variant):
        raise ValueError("V48.101 requires authoritative V48.100 result JSON")
    return obj


def _pair_tensors(rows: list[dict[str, Any]], device: torch.device):
    ci_np, ni_np = _pair_indices(rows)
    return torch.tensor(ci_np, dtype=torch.long, device=device), torch.tensor(ni_np, dtype=torch.long, device=device)


def _teacher_tensors(rows: list[dict[str, Any]], device: torch.device):
    td = torch.tensor([float(r["teacher_drs"]) for r in rows], dtype=torch.float32, device=device)
    tr = torch.tensor([float(r["teacher_r_dep"]) for r in rows], dtype=torch.float32, device=device)
    return td, tr


def _forward_all(*, model, module, obj: dict[str, Any], device: torch.device, batch_size: int = 512, require_grad: bool):
    ss: list[torch.Tensor] = []
    rr: list[torch.Tensor] = []
    context = torch.enable_grad() if require_grad else torch.no_grad()
    with context:
        for st in range(0, len(obj["rows"]), batch_size):
            mem = obj["memory"][st:st + batch_size].to(device)
            rv = obj["root_valid"][st:st + batch_size].to(device)
            out = module(model=model, memory=mem, root_valid=rv)
            ss.append(out["support"])
            rr.append(out["reserve_debt"])
    return torch.cat(ss, dim=0), torch.cat(rr, dim=0)


def _loss_eval(*, model, module, obj, device, scales, batch_size=512):
    module.eval(); model.eval()
    s, r = _forward_all(model=model, module=module, obj=obj, device=device, batch_size=batch_size, require_grad=False)
    td, tr = _teacher_tensors(obj["rows"], device)
    ci, ni = _pair_tensors(obj["rows"], device)
    loss, parts = joint_semantic_loss(s, r, td, tr, ci, ni, scales)
    return float(loss.item()), {k: float(v.item()) for k, v in parts.items()}, s.detach().cpu().numpy(), r.detach().cpu().numpy()


def _evaluate_cells(*, dev_obj, cert_obj, dev_support, dev_reserve, cert_support, cert_reserve, v93):
    cells: dict[str, Any] = {}
    contracts: dict[str, Any] = {}
    for role in ROLES:
        obj = dev_obj if role.startswith("dev_") else cert_obj
        sp = dev_support if role.startswith("dev_") else cert_support
        rs = dev_reserve if role.startswith("dev_") else cert_reserve
        rr = _role_rows(obj, sp, rs, role, v93)
        contract = _evaluation_contract(rr, role)
        contracts[role] = contract
        if not contract["valid"]:
            raise RuntimeError(f"V48.101 evaluation join fail-closed {role}: {contract['errors']}")
        state = _state_metric(rr)
        sup_true, sup_shuf = _action_metric(rr, "drs_activation", "support")
        res_true, res_shuf = _action_metric(rr, "deployability_gain", "reserve")
        for name, metric in (("state", state), ("support", sup_true), ("reserve", res_true)):
            if int(metric.get("rows", 0)) <= 0 or metric.get("auc") is None:
                raise RuntimeError(f"V48.101 {role} {name} empty/null")
        cells[role] = {
            "state": state,
            "support_true": sup_true,
            "support_shuffled": sup_shuf,
            "reserve_true": res_true,
            "reserve_shuffled": res_shuf,
        }
    return cells, contracts


def _baseline_identity(current: dict[str, Any], reference: dict[str, Any], tol: float = 1.0e-7) -> dict[str, Any]:
    errors: list[str] = []
    fields = ("rows", "positive_rows", "negative_rows", "powered_groups")
    float_fields = ("auc", "auc_vs_shuffled", "top1", "top1_vs_shuffled")
    for role in ROLES:
        ca = current.get(role) or {}; cb = (reference.get("cells") or {}).get(role) or {}
        for name in ("state", "support_true", "support_shuffled", "reserve_true", "reserve_shuffled"):
            a = ca.get(name) or {}; b = cb.get(name) or {}
            for f in fields:
                if f in a or f in b:
                    if int(a.get(f, -777)) != int(b.get(f, -778)):
                        errors.append(f"{role}:{name}:{f}")
            for f in float_fields:
                av, bv = a.get(f), b.get(f)
                if av is None and bv is None:
                    continue
                if av is None or bv is None or abs(float(av) - float(bv)) > tol:
                    errors.append(f"{role}:{name}:{f}")
    return {"valid": not errors, "tolerance": tol, "errors": errors}


def _state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for k in sorted(state):
        t = state[k].detach().cpu().contiguous()
        h.update(k.encode()); h.update(str(t.dtype).encode()); h.update(str(tuple(t.shape)).encode()); h.update(t.numpy().tobytes())
    return h.hexdigest()


def train_cross_attention(
    *, model, module, train_obj, dev_obj, device: torch.device, scales: dict[str, float], max_epochs: int = 60,
    patience: int = 10, batch_size: int = 512,
) -> dict[str, Any]:
    torch.manual_seed(101); np.random.seed(101); random.seed(101)
    module.to(device); freeze_v100_semantic_state(module)
    model.to(device); model.eval()
    n = configure_cross_attention_only(model, trainable=True)
    expected = expected_cross_attention_parameter_count(int(module.d_model))
    if n != expected:
        raise RuntimeError(f"V48.101 root-cross-attention parameter contract {n}!={expected}")
    params = list(model.root_cross_attn.parameters())
    opt = torch.optim.AdamW(params, lr=1.0e-3, weight_decay=1.0e-4)
    td, tr = _teacher_tensors(train_obj["rows"], device)
    ci, ni = _pair_tensors(train_obj["rows"], device)
    best = float("inf"); best_state = None; best_epoch = -1; stale = 0; history = []
    for epoch in range(max_epochs):
        model.eval(); module.eval()
        s, r = _forward_all(model=model, module=module, obj=train_obj, device=device, batch_size=batch_size, require_grad=True)
        loss, parts = joint_semantic_loss(s, r, td, tr, ci, ni, scales)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 5.0).item())
        opt.step()
        dev_loss, dev_parts, _, _ = _loss_eval(model=model, module=module, obj=dev_obj, device=device, scales=scales, batch_size=batch_size)
        history.append({
            "epoch": epoch,
            "train_loss": float(loss.item()),
            "train_parts": {k: float(v.item()) for k, v in parts.items()},
            "dev_loss": dev_loss,
            "dev_parts": dev_parts,
            "cross_attention_grad_norm": grad_norm,
        })
        if dev_loss < best - 1.0e-5:
            best = dev_loss; best_epoch = epoch; stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.root_cross_attn.state_dict().items()}
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("V48.101 training produced no checkpoint")
    model.root_cross_attn.load_state_dict(best_state, strict=True)
    model.to(device); model.eval()
    return {"best_epoch": best_epoch, "best_dev_loss": best, "epochs_completed": len(history), "history": history}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--v100-state", type=Path, required=True)
    ap.add_argument("--v100-result", type=Path, required=True)
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

    train_obj = extract_memory_features(checkpoint=a.checkpoint, index_path=a.train_index, cache_dir=a.cache_dir / "train", device=a.device, variant=a.variant)
    dev_obj = extract_memory_features(checkpoint=a.checkpoint, index_path=a.dev_index, cache_dir=a.cache_dir / "dev", device=a.device, variant=a.variant)
    cert_obj = extract_memory_features(checkpoint=a.checkpoint, index_path=a.certificate_index, cache_dir=a.cache_dir / "certificate", device=a.device, variant=a.variant)
    bundle, model = _load_model(a.checkpoint, a.device); device = bundle.device
    v100_state = _load_v100_state(a.v100_state, variant=a.variant)
    v100_result = _load_v100_result(a.v100_result, variant=a.variant)

    d_model = int(model.root_queries.shape[-1]); num_roots = int(model.root_queries.shape[1])
    module = JointRootSemanticDecoder(base_root_queries=model.root_queries, d_model=d_model)
    module.load_state_dict(v100_state["state_dict"], strict=True)
    freeze_v100_semantic_state(module); module.to(device)
    if not cross_attention_parameter_contract(model, d_model):
        raise RuntimeError("V48.101 root-cross-attention module shape/parameter contract mismatch")
    scales = {k: float(v) for k, v in (v100_state.get("semantic_metric_scales") or {}).items()}

    # Before opening a single new parameter, prove exact V48.100 functional/evaluation identity.
    _, _, dev_support0, dev_reserve0 = _loss_eval(model=model, module=module, obj=dev_obj, device=device, scales=scales)
    _, _, cert_support0, cert_reserve0 = _loss_eval(model=model, module=module, obj=cert_obj, device=device, scales=scales)
    v93 = build_v93_map(a.v93_audit)
    cells0, _ = _evaluate_cells(dev_obj=dev_obj, cert_obj=cert_obj, dev_support=dev_support0, dev_reserve=dev_reserve0, cert_support=cert_support0, cert_reserve=cert_reserve0, v93=v93)
    identity = _baseline_identity(cells0, v100_result)
    if not identity["valid"]:
        raise RuntimeError(f"V48.101 initial V48.100 identity failed: {identity['errors'][:8]}")

    initial_attn_state = {k: v.detach().cpu().clone() for k, v in model.root_cross_attn.state_dict().items()}
    initial_attn_sha = _state_dict_sha256(initial_attn_state)
    training = train_cross_attention(model=model, module=module, train_obj=train_obj, dev_obj=dev_obj, device=device, scales=scales)
    final_attn_state = {k: v.detach().cpu().clone() for k, v in model.root_cross_attn.state_dict().items()}
    final_attn_sha = _state_dict_sha256(final_attn_state)

    train_loss, train_parts, train_support, train_reserve = _loss_eval(model=model, module=module, obj=train_obj, device=device, scales=scales)
    dev_loss, dev_parts, dev_support, dev_reserve = _loss_eval(model=model, module=module, obj=dev_obj, device=device, scales=scales)
    cert_loss, cert_parts, cert_support, cert_reserve = _loss_eval(model=model, module=module, obj=cert_obj, device=device, scales=scales)
    cells, contracts = _evaluate_cells(dev_obj=dev_obj, cert_obj=cert_obj, dev_support=dev_support, dev_reserve=dev_reserve, cert_support=cert_support, cert_reserve=cert_reserve, v93=v93)

    attn_params = expected_cross_attention_parameter_count(d_model)
    result = {
        "schema": "ocrap-v48.101-rcsa-result-v1",
        "engineering_version": ENGINEERING_VERSION,
        "algorithm_name": ALGORITHM_NAME,
        "valid": True,
        "variant": a.variant,
        "planner_parameters_trained": 0,
        "source_parameters_trained": 0,
        "stage_i_parameters_trained": 0,
        "root_query_parameters_trained": 0,
        "recovery_chart_parameters_trained": 0,
        "root_cross_attention_parameters_trained": attn_params,
        "root_self_attention_parameters_trained": 0,
        "root_ffn_parameters_trained": 0,
        "root_logit_head_parameters_trained": 0,
        "v100_root_query_parameters_reused_frozen": 1536,
        "v100_recovery_chart_parameters_reused_frozen": 770,
        "root_slot_bijection_assumed": False,
        "regime_conditioning": False,
        "teacher_metadata_input_to_model": False,
        "boundary_transport": False,
        "checkpoint": str(a.checkpoint.resolve()),
        "checkpoint_sha256": sha256(a.checkpoint),
        "v100_state": str(a.v100_state.resolve()),
        "v100_state_sha256": sha256(a.v100_state),
        "v100_result": str(a.v100_result.resolve()),
        "v100_result_sha256": sha256(a.v100_result),
        "v100_baseline_identity": identity,
        "semantic_metric_scales": scales,
        "initial_root_cross_attention_sha256": initial_attn_sha,
        "final_root_cross_attention_sha256": final_attn_sha,
        "root_cross_attention_changed": bool(initial_attn_sha != final_attn_sha),
        "training": training,
        "dense_metrics": {
            "train": _dense_metrics(train_obj, train_support, train_reserve),
            "dev": _dense_metrics(dev_obj, dev_support, dev_reserve),
            "certificate": _dense_metrics(cert_obj, cert_support, cert_reserve),
        },
        "semantic_loss": {
            "train": {"total": train_loss, **train_parts},
            "dev": {"total": dev_loss, **dev_parts},
            "certificate": {"total": cert_loss, **cert_parts},
        },
        "cells": cells,
        "evaluation_contracts": contracts,
        "feature_contracts": {
            "train": train_obj.get("feature_only_dataset_contract"),
            "dev": dev_obj.get("feature_only_dataset_contract"),
            "certificate": cert_obj.get("feature_only_dataset_contract"),
        },
        "elapsed_seconds": float(time.perf_counter() - t0),
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    torch.save({
        "schema": "ocrap-v48.101-rcsa-state-v1",
        "engineering_version": ENGINEERING_VERSION,
        "algorithm_name": ALGORITHM_NAME,
        "variant": a.variant,
        "d_model": d_model,
        "num_roots": num_roots,
        "root_cross_attention_state_dict": final_attn_state,
        "root_cross_attention_parameter_count": attn_params,
        "v100_state_sha256": sha256(a.v100_state),
        "v100_result_sha256": sha256(a.v100_result),
        "l80_checkpoint_sha256": sha256(a.checkpoint),
        "v100_frozen_state_dict_sha256": _state_dict_sha256({k: v.detach().cpu() for k, v in module.state_dict().items()}),
        "semantic_metric_scales": scales,
        "initial_root_cross_attention_sha256": initial_attn_sha,
        "final_root_cross_attention_sha256": final_attn_sha,
        "training": {k: v for k, v in training.items() if k != "history"},
    }, a.state_output)
    print(json.dumps({"valid": True, "variant": a.variant, "cross_attention_parameters": attn_params, "baseline_identity": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
