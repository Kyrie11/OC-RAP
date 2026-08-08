#!/usr/bin/env python3
"""Runtime contract for v48.36.1 OCAF group-row gather/broadcast/scatter.

The original v48.36 A30 run passed CPU unit tests but failed on the first CUDA
batch inside ``_direct_nominal_observation_features``.  This preflight executes
the exact 141-D action / 529-D observation geometry, including backward, on the
selected runtime device before expensive adaptation starts.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import traceback

import torch

from ocrap.models.encoders import FlatFeatureLayout
from ocrap.models.ocrap import OCRAPModel


def _atomic_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _model(
    device: torch.device,
    hidden: int,
    dual: bool = False,
    factorized_harm: bool = False,
) -> OCRAPModel:
    model = OCRAPModel(
        input_dim=FlatFeatureLayout().total_dim,
        num_roots=2,
        num_options=3,
        d_model=8,
        d_obs=4,
        encoder_type="structured_transformer",
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        direct_recovery_value_head=True,
        direct_recovery_value_pooling="candidate_concat_raw",
        direct_recovery_delta_head=True,
        direct_recovery_delta_regime_experts=True,
        direct_recovery_evidence_calibrator=True,
        direct_recovery_evidence_calibrator_context=True,
        direct_recovery_evidence_calibrator_context_source="physical_interaction",
        direct_recovery_evidence_interaction_hidden=hidden,
        direct_recovery_evidence_interaction_dropout=0.0,
        direct_recovery_evidence_dual_interaction_bridge=dual,
        direct_recovery_evidence_factorized_harm_interaction=factorized_harm,
        direct_recovery_evidence_unified_experts=True,
        direct_recovery_evidence_component_heads=True,
        direct_recovery_evidence_component_count=5,
        direct_recovery_evidence_concord=True,
        direct_recovery_evidence_admission_head=False,
        direct_recovery_evidence_admission_bounded=False,
        direct_recovery_evidence_admission_prior_mode="frontier_capped_slack",
    ).to(device)
    return model.train()


def _exercise(
    device: torch.device,
    batch_size: int,
    group_size: int,
    hidden: int,
    dual: bool = False,
    factorized_harm: bool = False,
) -> dict:
    if batch_size < group_size or batch_size % group_size != 0:
        raise ValueError("batch-size must be a positive multiple of group-size")
    torch.manual_seed(48361)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(48361)

    model = _model(device, hidden, dual=dual, factorized_harm=factorized_harm)
    layout = FlatFeatureLayout()
    num_groups = batch_size // group_size
    x = torch.randn(batch_size, layout.total_dim, device=device, requires_grad=True)
    group_ids = torch.arange(num_groups, device=device, dtype=torch.long).repeat_interleave(group_size)
    group_index = torch.stack([group_ids, group_ids * 1009 + 17], dim=-1)
    nominal = torch.zeros(batch_size, device=device)
    nominal[::group_size] = 1.0

    action = model._direct_candidate_raw_relative_features(x, group_index, nominal)
    observation = model._direct_nominal_observation_features(x, group_index, nominal)
    if action.shape != (batch_size, 141):
        raise AssertionError(f"unexpected action shape: {tuple(action.shape)}")
    if observation.shape != (batch_size, 529):
        raise AssertionError(f"unexpected observation shape: {tuple(observation.shape)}")

    raw_action = torch.cat(
        [x[:, start:end] for start, end in model.direct_candidate_physical_slices], dim=-1
    )
    raw_observation = torch.cat(
        [x[:, start:end] for start, end in model.direct_observation_slices], dim=-1
    )
    for group_id in range(num_groups):
        start = group_id * group_size
        idx = torch.arange(start, start + group_size, device=device)
        expected_action = raw_action.index_select(0, idx) - raw_action[start : start + 1]
        expected_observation = raw_observation[start : start + 1].expand(group_size, -1)
        if not torch.allclose(action.index_select(0, idx), expected_action):
            raise AssertionError(f"candidate-relative action mismatch in group {group_id}")
        if not torch.allclose(observation.index_select(0, idx), expected_observation):
            raise AssertionError(f"nominal observation broadcast mismatch in group {group_id}")

    context_raw = model.direct_evidence_interaction_bridge(action, observation)
    contexts = list(context_raw) if isinstance(context_raw, tuple) else [context_raw]
    for context in contexts:
        if not torch.equal(context[::group_size], torch.zeros_like(context[::group_size])):
            raise AssertionError("zero-action nominal rows do not produce exact zero OCAF context")
    loss = sum(context.float().square().mean() for context in contexts) + action.float().square().mean() + observation.float().square().mean()
    context = contexts[0]
    loss.backward()
    if x.grad is None or not torch.isfinite(x.grad).all():
        raise AssertionError("non-finite group-row/OCAF input gradients")
    bridge_grads = [
        parameter.grad
        for parameter in model.direct_evidence_interaction_bridge.parameters()
        if parameter.requires_grad
    ]
    if not bridge_grads or any(grad is None or not torch.isfinite(grad).all() for grad in bridge_grads):
        raise AssertionError("non-finite OCAF bridge parameter gradients")
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    return {
        "batch_size": batch_size,
        "group_size": group_size,
        "num_groups": num_groups,
        "candidate_action_dim": action.shape[-1],
        "nominal_observation_dim": observation.shape[-1],
        "interaction_hidden": context.shape[-1],
        "dual_interaction_bridge": bool(dual),
        "factorized_harm_interaction": bool(factorized_harm),
        "zero_action_exact_zero": True,
        "forward_finite": bool(torch.isfinite(context).all()),
        "backward_finite": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--interaction-hidden", type=int, default=64)
    parser.add_argument("--dual-interaction-bridge", action="store_true")
    parser.add_argument("--factorized-harm-interaction", action="store_true")
    args = parser.parse_args()

    started = time.time()
    report = {
        "event": "v48_36_1_cuda_group_broadcast_contract",
        "version": "v48.36.1-RC30-CUDA-INDEX-HOTFIX",
        "created_unix": started,
        "device_request": args.device,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "test_roots_read": False,
    }
    try:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        if device.type == "cuda":
            torch.cuda.set_device(device)
            report["cuda_device_index"] = torch.cuda.current_device()
            report["cuda_device_name"] = torch.cuda.get_device_name(torch.cuda.current_device())
        report.update(
            _exercise(
                device=device,
                batch_size=args.batch_size,
                group_size=args.group_size,
                hidden=args.interaction_hidden,
                dual=args.dual_interaction_bridge,
                factorized_harm=args.factorized_harm_interaction,
            )
        )
        report["valid"] = True
        report["elapsed_s"] = time.time() - started
        _atomic_json(args.output, report)
        print(json.dumps(report, ensure_ascii=False))
        return 0
    except Exception as exc:  # fail-closed and preserve the first runtime traceback
        report.update(
            {
                "valid": False,
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "elapsed_s": time.time() - started,
            }
        )
        _atomic_json(args.output, report)
        print(json.dumps(report, ensure_ascii=False))
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
