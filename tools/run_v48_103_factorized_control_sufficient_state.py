#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ocrap.v48_100_joint_root_semantic_decoder import joint_semantic_loss
from ocrap.v48_103_factorized_control_sufficient_state import (
    ALGORITHM_NAME,
    ENGINEERING_VERSION,
    FactorizedControlSufficientState,
    build_nominal_index,
    expected_parameter_count,
)
from tools.run_v48_101_root_cross_attention_semantic_alignment import (
    extract_memory_features,
    _load_v100_state,
)
from tools.run_v48_97_executable_recovery_state import (
    ROLES,
    _action_metric,
    _dense_metrics,
    _evaluation_contract,
    _pair_indices,
    _role_rows,
    _state_metric,
    build_v93_map,
    sha256,
)


def _teacher_tensors(rows: list[dict[str, Any]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    td = torch.tensor([float(r["teacher_drs"]) for r in rows], dtype=torch.float32, device=device)
    tr = torch.tensor([float(r["teacher_r_dep"]) for r in rows], dtype=torch.float32, device=device)
    return td, tr


def _pair_tensors(rows: list[dict[str, Any]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    ci, ni = _pair_indices(rows)
    return (
        torch.tensor(ci, dtype=torch.long, device=device),
        torch.tensor(ni, dtype=torch.long, device=device),
    )


def _forward_all(
    *, module: FactorizedControlSufficientState, obj: dict[str, Any], device: torch.device,
    batch_size: int = 512, require_grad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    memory = obj["memory"]
    rows = obj["rows"]
    nominal = build_nominal_index(rows)
    # The module composition needs same-group nominal rows.  To keep the group
    # algebra exact, compute potentials in batches and compose after concatenation.
    pots: list[torch.Tensor] = []
    ctx = torch.enable_grad() if require_grad else torch.no_grad()
    with ctx:
        for st in range(0, len(memory), batch_size):
            mem = memory[st:st + batch_size].to(device)
            pots.append(module.semantic_potentials(mem))
        pot = torch.cat(pots, dim=0)
        ni = nominal.to(device)
        anchor = pot.index_select(0, ni)
        support = torch.sigmoid(anchor[:, 0] + pot[:, 2] - anchor[:, 2])
        reserve = anchor[:, 1] + pot[:, 3] - anchor[:, 3]
    return support, reserve


def _loss_eval(
    *, module: FactorizedControlSufficientState, obj: dict[str, Any], device: torch.device,
    scales: dict[str, float], batch_size: int = 512,
) -> tuple[float, dict[str, float], np.ndarray, np.ndarray]:
    support, reserve = _forward_all(module=module, obj=obj, device=device, batch_size=batch_size, require_grad=False)
    td, tr = _teacher_tensors(obj["rows"], device)
    ci, ni = _pair_tensors(obj["rows"], device)
    loss, parts = joint_semantic_loss(support, reserve, td, tr, ci, ni, scales)
    return (
        float(loss.item()),
        {k: float(v.item()) for k, v in parts.items()},
        support.detach().cpu().numpy(),
        reserve.detach().cpu().numpy(),
    )


def train_representation(
    *, module: FactorizedControlSufficientState, train_obj: dict[str, Any], dev_obj: dict[str, Any],
    device: torch.device, scales: dict[str, float], max_epochs: int = 60, patience: int = 10,
    batch_size: int = 512,
) -> dict[str, Any]:
    torch.manual_seed(103); np.random.seed(103); random.seed(103)
    module.to(device)
    params = list(module.parameters())
    opt = torch.optim.AdamW(params, lr=1.0e-3, weight_decay=1.0e-4)
    td, tr = _teacher_tensors(train_obj["rows"], device)
    ci, ni = _pair_tensors(train_obj["rows"], device)
    best = float("inf"); best_epoch = -1; best_state = None; stale = 0; history: list[dict[str, Any]] = []
    for epoch in range(max_epochs):
        module.train()
        support, reserve = _forward_all(module=module, obj=train_obj, device=device, batch_size=batch_size, require_grad=True)
        loss, parts = joint_semantic_loss(support, reserve, td, tr, ci, ni, scales)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 5.0).item())
        opt.step()
        module.eval()
        dev_loss, dev_parts, _, _ = _loss_eval(module=module, obj=dev_obj, device=device, scales=scales, batch_size=batch_size)
        history.append({
            "epoch": epoch,
            "train_loss": float(loss.item()),
            "train_parts": {k: float(v.item()) for k, v in parts.items()},
            "dev_loss": dev_loss,
            "dev_parts": dev_parts,
            "representation_grad_norm": grad_norm,
        })
        if dev_loss < best - 1.0e-5:
            best = dev_loss; best_epoch = epoch; stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("V48.103 training produced no best state")
    module.load_state_dict(best_state, strict=True)
    module.to(device).eval()
    return {
        "best_epoch": int(best_epoch),
        "best_dev_semantic_loss": float(best),
        "epochs_completed": len(history),
        "history": history,
    }


def _evaluate_cells(
    *, dev_obj: dict[str, Any], cert_obj: dict[str, Any], dev_support: np.ndarray, dev_reserve: np.ndarray,
    cert_support: np.ndarray, cert_reserve: np.ndarray, v93: dict,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cells: dict[str, Any] = {}; contracts: dict[str, Any] = {}
    for role in ROLES:
        obj = dev_obj if role.startswith("dev_") else cert_obj
        sp = dev_support if role.startswith("dev_") else cert_support
        rs = dev_reserve if role.startswith("dev_") else cert_reserve
        rr = _role_rows(obj, sp, rs, role, v93)
        contract = _evaluation_contract(rr, role)
        if not contract["valid"]:
            raise RuntimeError(f"V48.103 evaluation join fail-closed {role}: {contract['errors']}")
        state = _state_metric(rr)
        sup_true, sup_shuf = _action_metric(rr, "drs_activation", "support")
        res_true, res_shuf = _action_metric(rr, "deployability_gain", "reserve")
        cells[role] = {
            "state": state,
            "support_true": sup_true,
            "support_shuffled": sup_shuf,
            "reserve_true": res_true,
            "reserve_shuffled": res_shuf,
        }
        contracts[role] = contract
    return cells, contracts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--v100-state", type=Path, required=True)
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

    resolved_device = a.device if (not a.device.startswith("cuda") or torch.cuda.is_available()) else "cpu"
    train_obj = extract_memory_features(checkpoint=a.checkpoint, index_path=a.train_index, cache_dir=a.cache_dir / "train", device=resolved_device, variant=a.variant)
    dev_obj = extract_memory_features(checkpoint=a.checkpoint, index_path=a.dev_index, cache_dir=a.cache_dir / "dev", device=resolved_device, variant=a.variant)
    cert_obj = extract_memory_features(checkpoint=a.checkpoint, index_path=a.certificate_index, cache_dir=a.cache_dir / "certificate", device=resolved_device, variant=a.variant)
    d_model = int(train_obj["memory"].shape[-1])
    module = FactorizedControlSufficientState(d_model)
    if module.trainable_parameter_count != expected_parameter_count(d_model):
        raise RuntimeError("V48.103 parameter-count contract mismatch")
    device = torch.device(resolved_device)
    v100_state = _load_v100_state(a.v100_state, variant=a.variant)
    scales = {k: float(v) for k, v in (v100_state.get("semantic_metric_scales") or {}).items()}
    training = train_representation(module=module, train_obj=train_obj, dev_obj=dev_obj, device=device, scales=scales)
    train_loss, train_parts, train_support, train_reserve = _loss_eval(module=module, obj=train_obj, device=device, scales=scales)
    dev_loss, dev_parts, dev_support, dev_reserve = _loss_eval(module=module, obj=dev_obj, device=device, scales=scales)
    cert_loss, cert_parts, cert_support, cert_reserve = _loss_eval(module=module, obj=cert_obj, device=device, scales=scales)
    v93 = build_v93_map(a.v93_audit)
    cells, contracts = _evaluate_cells(
        dev_obj=dev_obj, cert_obj=cert_obj, dev_support=dev_support, dev_reserve=dev_reserve,
        cert_support=cert_support, cert_reserve=cert_reserve, v93=v93,
    )
    result = {
        "schema": "ocrap-v48.103-fcss-result-v1",
        "engineering_version": ENGINEERING_VERSION,
        "algorithm_name": ALGORITHM_NAME,
        "valid": True,
        "variant": a.variant,
        "planner_parameters_trained": 0,
        "stage_i_parameters_trained": 0,
        "root_decoder_parameters_trained": 0,
        "representation_parameters_trained": module.trainable_parameter_count,
        "source_parameters_trained": 0,
        "relative_ranker_modified": False,
        "boundary_transport": False,
        "regime_conditioning": False,
        "teacher_metadata_input_to_model": False,
        "root_slot_bijection_assumed": False,
        "nominal_response_exact_zero": True,
        "state_response_learned_mixing": False,
        "checkpoint": str(a.checkpoint.resolve()),
        "checkpoint_sha256": sha256(a.checkpoint),
        "v100_state": str(a.v100_state.resolve()),
        "v100_state_sha256": sha256(a.v100_state),
        "semantic_metric_scales": scales,
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
        "schema": "ocrap-v48.103-fcss-state-v1",
        "engineering_version": ENGINEERING_VERSION,
        "algorithm_name": ALGORITHM_NAME,
        "variant": a.variant,
        "d_model": d_model,
        "state_dict": {k: v.detach().cpu() for k, v in module.state_dict().items()},
        "representation_parameter_count": module.trainable_parameter_count,
        "checkpoint_sha256": sha256(a.checkpoint),
        "v100_state_sha256": sha256(a.v100_state),
        "semantic_metric_scales": scales,
        "training": {k: v for k, v in training.items() if k != "history"},
    }, a.state_output)
    print(json.dumps({"valid": True, "variant": a.variant, "parameters": module.trainable_parameter_count, "best_epoch": training["best_epoch"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
