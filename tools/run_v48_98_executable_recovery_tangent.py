#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ocrap.models.data import OCRAPSampleDataset
from ocrap.models.inference import load_model_bundle
from ocrap.v48_96_support_reserve_root_observability import feature_only_dataset_cfg
from ocrap.v48_97_executable_recovery_state import ExecutableRecoverySufficientState
from ocrap.v48_98_executable_recovery_tangent import (
    ALGORITHM_NAME,
    ENGINEERING_VERSION,
    ExecutableRecoveryTangentAdapter,
    scene_tokens_with_recovery_tangent,
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
    read_jsonl,
    sha256,
)


def _input_cache_key(checkpoint: Path, index_path: Path, variant: str) -> str:
    payload = {
        "engineering_version": ENGINEERING_VERSION,
        "checkpoint": sha256(checkpoint),
        "index": sha256(index_path),
        "variant": variant,
        "kind": "feature_only_inputs",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def extract_inputs(
    *, checkpoint: Path, index_path: Path, cache_dir: Path, device: str, variant: str,
) -> dict[str, Any]:
    key = _input_cache_key(checkpoint, index_path, variant)
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
    cfg, feature_cfg_event = feature_only_dataset_cfg(
        bundle.cfg, cache_dir=str(cache_dir / "tensor"), workers=8
    )
    ds = OCRAPSampleDataset(paths, cfg)
    if ds.absolute_truth_contract_event.get("enabled") or ds.action_response_truth_event.get("enabled"):
        raise RuntimeError("V48.98 feature-only dataset unexpectedly attached supervision truth sidecars")
    resolved_index = [str(p.resolve()) for p in paths]
    resolved_ds = [str(p.resolve()) for p in ds.paths]
    if resolved_index != resolved_ds:
        raise RuntimeError("V48.98 dataset path order differs from teacher-index order")
    xs: list[torch.Tensor] = []
    rvs: list[torch.Tensor] = []
    for i in range(len(ds)):
        it = ds[i]
        xs.append(it["x"].detach().float().cpu())
        rvs.append(it["root_valid"].detach().bool().cpu())
    obj = {
        "cache_key": key,
        "checkpoint": str(checkpoint.resolve()),
        "index": str(index_path.resolve()),
        "rows": rows,
        "x": torch.stack(xs, dim=0),
        "root_valid": torch.stack(rvs, dim=0),
        "feature_only_dataset_contract": feature_cfg_event,
        "tensor_cache_event": ds.tensor_cache_event,
    }
    torch.save(obj, cache)
    return obj


def _nominal_for_each_row(rows: list[dict[str, Any]]) -> np.ndarray:
    by: dict[tuple[int, str, int], list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by[(int(r["bucket"]), str(r["scene"]), int(r["time"]))].append(i)
    out = np.full(len(rows), -1, dtype=np.int64)
    for key, ids in by.items():
        ns = [i for i in ids if bool(rows[i].get("nominal", False))]
        if len(ns) != 1:
            raise RuntimeError(f"V48.98 requires exactly one nominal per group, key={key} got={len(ns)}")
        ni = ns[0]
        for i in ids:
            out[i] = ni
    if np.any(out < 0):
        raise RuntimeError("V48.98 failed to assign nominal references")
    return out


def _load_erss_state(path: Path, device: torch.device) -> tuple[ExecutableRecoverySufficientState, dict[str, Any]]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise ValueError("invalid V48.97 ERSS state")
    if str(obj.get("engineering_version")) != "v48.97.2-OC-ERSS-STRATAFIX":
        raise ValueError(f"V48.98 requires V48.97.2 state, got {obj.get('engineering_version')!r}")
    d_model = int(obj.get("d_model", 0))
    module = ExecutableRecoverySufficientState(d_model)
    module.load_state_dict(obj["state_dict"], strict=True)
    module.eval().to(device)
    for p in module.parameters():
        p.requires_grad_(False)
    if int(obj.get("trainable_parameter_count", -1)) != 4 * d_model + 2:
        raise ValueError("V48.97 ERSS state parameter-count contract mismatch")
    return module, obj


def _load_frozen_model(checkpoint: Path, device: str):
    runtime = {"training": {"device": device}}
    bundle = load_model_bundle(checkpoint, runtime)
    if bundle is None:
        raise RuntimeError(f"cannot load checkpoint {checkpoint}")
    model = bundle.model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return bundle, model


def _predict_baseline(
    *, model, erss: ExecutableRecoverySufficientState, obj: dict[str, Any], device: torch.device,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    x = obj["x"]
    rv = obj["root_valid"]
    sup: list[torch.Tensor] = []
    res: list[torch.Tensor] = []
    with torch.no_grad():
        for st in range(0, len(x), batch_size):
            xx = x[st: st + batch_size].to(device)
            vv = rv[st: st + batch_size].to(device)
            mem = model._scene_tokens(xx)
            rt = model._decode_roots(mem)
            rp = _root_probs(model, rt, vv)
            out = erss(rt, rp, vv)
            sup.append(out["support"].detach().float().cpu())
            res.append(out["reserve_debt"].detach().float().cpu())
    return torch.cat(sup).numpy(), torch.cat(res).numpy()


def _predict_tangent(
    *, model, erss: ExecutableRecoverySufficientState, adapter: ExecutableRecoveryTangentAdapter,
    obj: dict[str, Any], device: torch.device, batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    x = obj["x"]
    rv = obj["root_valid"]
    nom = _nominal_for_each_row(obj["rows"])
    sup: list[torch.Tensor] = []
    res: list[torch.Tensor] = []
    adapter.eval()
    with torch.no_grad():
        for st in range(0, len(x), batch_size):
            ids = np.arange(st, min(len(x), st + batch_size), dtype=np.int64)
            xx = x[ids].to(device)
            xn = x[nom[ids]].to(device)
            vv = rv[ids].to(device)
            mem = scene_tokens_with_recovery_tangent(model, adapter, xx, xn)
            rt = model._decode_roots(mem)
            rp = _root_probs(model, rt, vv)
            out = erss(rt, rp, vv)
            sup.append(out["support"].detach().float().cpu())
            res.append(out["reserve_debt"].detach().float().cpu())
    return torch.cat(sup).numpy(), torch.cat(res).numpy()


def _delta_loss_eval(
    *, model, erss, adapter, obj: dict[str, Any], baseline_support: np.ndarray,
    baseline_reserve: np.ndarray, device: torch.device, batch_size: int = 256,
) -> tuple[float, dict[str, float]]:
    rows = obj["rows"]
    ci, ni = _pair_indices(rows)
    if len(ci) == 0:
        raise RuntimeError("V48.98 dev split has no candidate/nominal pairs")
    x = obj["x"]
    rv = obj["root_valid"]
    td = np.asarray([float(r["teacher_drs"]) for r in rows], dtype=np.float32)
    tr = np.asarray([float(r["teacher_r_dep"]) for r in rows], dtype=np.float32)
    losses_s: list[float] = []
    losses_r: list[float] = []
    adapter.eval()
    with torch.no_grad():
        for st in range(0, len(ci), batch_size):
            ids = ci[st: st + batch_size]
            ns = ni[st: st + batch_size]
            xx = x[ids].to(device)
            xn = x[ns].to(device)
            vv = rv[ids].to(device)
            mem = scene_tokens_with_recovery_tangent(model, adapter, xx, xn)
            rt = model._decode_roots(mem)
            rp = _root_probs(model, rt, vv)
            out = erss(rt, rp, vv)
            ps = out["support"].float() - torch.tensor(baseline_support[ns], device=device)
            pr = out["reserve_debt"].float() - torch.tensor(baseline_reserve[ns], device=device)
            ts = torch.tensor(td[ids] - td[ns], device=device)
            trr = torch.tensor(tr[ids] - tr[ns], device=device)
            ls = torch.nn.functional.smooth_l1_loss(ps, ts, beta=1.0, reduction="mean")
            lr = torch.nn.functional.smooth_l1_loss(pr, trr, beta=1.0, reduction="mean")
            losses_s.append(float(ls.item()) * len(ids))
            losses_r.append(float(lr.item()) * len(ids))
    n = float(len(ci))
    ds = sum(losses_s) / n
    dr = sum(losses_r) / n
    return (ds + dr) / 2.0, {"delta_support": ds, "delta_reserve": dr}


def train_tangent(
    *, model, erss, adapter: ExecutableRecoveryTangentAdapter,
    train_obj: dict[str, Any], dev_obj: dict[str, Any], device: torch.device,
    max_epochs: int = 60, patience: int = 10, batch_size: int = 256,
) -> dict[str, Any]:
    torch.manual_seed(98)
    np.random.seed(98)
    random.seed(98)
    adapter.to(device)
    rows = train_obj["rows"]
    ci, ni = _pair_indices(rows)
    if len(ci) == 0:
        raise RuntimeError("V48.98 train split has no candidate/nominal pairs")
    x = train_obj["x"]
    rv = train_obj["root_valid"]
    td = np.asarray([float(r["teacher_drs"]) for r in rows], dtype=np.float32)
    tr = np.asarray([float(r["teacher_r_dep"]) for r in rows], dtype=np.float32)
    base_train_s, base_train_r = _predict_baseline(model=model, erss=erss, obj=train_obj, device=device)
    base_dev_s, base_dev_r = _predict_baseline(model=model, erss=erss, obj=dev_obj, device=device)

    opt = torch.optim.AdamW(adapter.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    best = float("inf")
    best_state = None
    best_epoch = -1
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(int(max_epochs)):
        adapter.train()
        total_s = 0.0
        total_r = 0.0
        total_n = 0
        # Deterministic dense candidate order; no data/sampler sweep.
        for st in range(0, len(ci), batch_size):
            ids = ci[st: st + batch_size]
            ns = ni[st: st + batch_size]
            xx = x[ids].to(device)
            xn = x[ns].to(device)
            vv = rv[ids].to(device)
            mem = scene_tokens_with_recovery_tangent(model, adapter, xx, xn)
            rt = model._decode_roots(mem)
            rp = _root_probs(model, rt, vv)
            out = erss(rt, rp, vv)
            ps = out["support"].float() - torch.tensor(base_train_s[ns], device=device)
            pr = out["reserve_debt"].float() - torch.tensor(base_train_r[ns], device=device)
            ts = torch.tensor(td[ids] - td[ns], device=device)
            trr = torch.tensor(tr[ids] - tr[ns], device=device)
            ls = torch.nn.functional.smooth_l1_loss(ps, ts, beta=1.0)
            lr = torch.nn.functional.smooth_l1_loss(pr, trr, beta=1.0)
            loss = (ls + lr) / 2.0
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 5.0)
            opt.step()
            total_s += float(ls.item()) * len(ids)
            total_r += float(lr.item()) * len(ids)
            total_n += len(ids)
        dev_loss, dev_parts = _delta_loss_eval(
            model=model, erss=erss, adapter=adapter, obj=dev_obj,
            baseline_support=base_dev_s, baseline_reserve=base_dev_r, device=device,
            batch_size=batch_size,
        )
        history.append({
            "epoch": epoch,
            "train_loss": (total_s + total_r) / (2.0 * max(total_n, 1)),
            "train_parts": {
                "delta_support": total_s / max(total_n, 1),
                "delta_reserve": total_r / max(total_n, 1),
            },
            "dev_loss": dev_loss,
            "dev_parts": dev_parts,
        })
        if dev_loss < best - 1.0e-5:
            best = dev_loss
            best_epoch = epoch
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()}
        else:
            stale += 1
        if stale >= int(patience):
            break
    if best_state is None:
        raise RuntimeError("V48.98 tangent training produced no checkpoint")
    adapter.load_state_dict(best_state)
    adapter.to(device)
    return {
        "best_epoch": best_epoch,
        "best_dev_tangent_loss": best,
        "epochs_completed": len(history),
        "history": history,
    }


def _nominal_identity(
    rows: list[dict[str, Any]], base_s: np.ndarray, base_r: np.ndarray,
    new_s: np.ndarray, new_r: np.ndarray,
) -> dict[str, float]:
    ids = np.asarray([i for i, r in enumerate(rows) if bool(r.get("nominal", False))], dtype=np.int64)
    if len(ids) == 0:
        raise RuntimeError("V48.98 nominal identity has no nominal rows")
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
    args = ap.parse_args()
    t0 = time.perf_counter()

    train_obj = extract_inputs(
        checkpoint=args.checkpoint, index_path=args.train_index, cache_dir=args.cache_dir / "train",
        device=args.device, variant=args.variant,
    )
    dev_obj = extract_inputs(
        checkpoint=args.checkpoint, index_path=args.dev_index, cache_dir=args.cache_dir / "dev",
        device=args.device, variant=args.variant,
    )
    cert_obj = extract_inputs(
        checkpoint=args.checkpoint, index_path=args.certificate_index, cache_dir=args.cache_dir / "certificate",
        device=args.device, variant=args.variant,
    )
    bundle, model = _load_frozen_model(args.checkpoint, args.device)
    device = bundle.device
    erss, erss_obj = _load_erss_state(args.erss_state, device)
    enc = model.encoder
    layout = enc.layout
    adapter = ExecutableRecoveryTangentAdapter(
        d_model=int(enc.d_model),
        prefix_param_dim=int(layout.prefix_param_dim),
        prefix_state_dim=int(layout.prefix_flat_dim),
        control_dim=int(layout.control_flat_dim),
    )
    training = train_tangent(
        model=model, erss=erss, adapter=adapter, train_obj=train_obj, dev_obj=dev_obj, device=device
    )

    base_dev_s, base_dev_r = _predict_baseline(model=model, erss=erss, obj=dev_obj, device=device)
    base_cert_s, base_cert_r = _predict_baseline(model=model, erss=erss, obj=cert_obj, device=device)
    dev_s, dev_r = _predict_tangent(model=model, erss=erss, adapter=adapter, obj=dev_obj, device=device)
    cert_s, cert_r = _predict_tangent(model=model, erss=erss, adapter=adapter, obj=cert_obj, device=device)
    id_dev = _nominal_identity(dev_obj["rows"], base_dev_s, base_dev_r, dev_s, dev_r)
    id_cert = _nominal_identity(cert_obj["rows"], base_cert_s, base_cert_r, cert_s, cert_r)
    if max(id_dev["support_max_abs_error"], id_dev["reserve_max_abs_error"], id_cert["support_max_abs_error"], id_cert["reserve_max_abs_error"]) > 1.0e-7:
        raise RuntimeError(f"V48.98 nominal identity violated: dev={id_dev} cert={id_cert}")

    v93 = build_v93_map(args.v93_audit)
    result: dict[str, Any] = {
        "schema": "ocrap-v48.98-erta-result-v1",
        "engineering_version": ENGINEERING_VERSION,
        "algorithm_name": ALGORITHM_NAME,
        "valid": True,
        "variant": args.variant,
        "planner_parameters_trained": 0,
        "stage_i_tangent_parameters_trained": adapter.trainable_parameter_count,
        "erss_parameters_trained": 0,
        "source_parameters_trained": 0,
        "regime_conditioning": False,
        "teacher_metadata_input_to_model": False,
        "root_slot_bijection_assumed": False,
        "candidate_relative_stage_i_update": True,
        "candidate_physical_blocks": ["prefix_param", "prefix_state", "control"],
        "semantic_tangent_rank": 2,
        "checkpoint": str(args.checkpoint.resolve()),
        "erss_state": str(args.erss_state.resolve()),
        "erss_state_sha256": sha256(args.erss_state),
        "training": training,
        "nominal_identity": {"dev": id_dev, "certificate": id_cert},
        "dense_metrics": {
            "dev": _dense_metrics(dev_obj, dev_s, dev_r),
            "certificate": _dense_metrics(cert_obj, cert_s, cert_r),
        },
        "cells": {},
        "evaluation_contracts": {},
        "feature_contracts": {
            "train": train_obj.get("feature_only_dataset_contract"),
            "dev": dev_obj.get("feature_only_dataset_contract"),
            "certificate": cert_obj.get("feature_only_dataset_contract"),
        },
    }
    for role in ROLES:
        obj = dev_obj if role.startswith("dev_") else cert_obj
        sp = dev_s if role.startswith("dev_") else cert_s
        rs = dev_r if role.startswith("dev_") else cert_r
        rr = _role_rows(obj, sp, rs, role, v93)
        contract = _evaluation_contract(rr, role)
        result["evaluation_contracts"][role] = contract
        if not contract["valid"]:
            raise RuntimeError(f"V48.98 evaluation join fail-closed for {role}: {contract['errors']}")
        state = _state_metric(rr)
        sup_true, sup_shuf = _action_metric(rr, "drs_activation", "support")
        res_true, res_shuf = _action_metric(rr, "deployability_gain", "reserve")
        result["cells"][role] = {
            "state": state,
            "support_true": sup_true,
            "support_shuffled": sup_shuf,
            "reserve_true": res_true,
            "reserve_shuffled": res_shuf,
        }
    result["elapsed_seconds"] = float(time.perf_counter() - t0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    torch.save({
        "schema": "ocrap-v48.98-erta-state-v1",
        "engineering_version": ENGINEERING_VERSION,
        "algorithm_name": ALGORITHM_NAME,
        "variant": args.variant,
        "d_model": int(enc.d_model),
        "state_dict": {k: v.detach().cpu() for k, v in adapter.state_dict().items()},
        "trainable_parameter_count": adapter.trainable_parameter_count,
        "semantic_tangent_rank": 2,
        "erss_state_sha256": sha256(args.erss_state),
        "training": {k: v for k, v in training.items() if k != "history"},
    }, args.state_output)
    print(json.dumps({
        "valid": True,
        "variant": args.variant,
        "best_epoch": training["best_epoch"],
        "stage_i_tangent_parameters": adapter.trainable_parameter_count,
        "elapsed_seconds": result["elapsed_seconds"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
