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
from ocrap.models.encoders import StructuredTokenEncoder
from ocrap.models.inference import load_model_bundle
from ocrap.v48_96_support_reserve_root_observability import feature_only_dataset_cfg
from ocrap.v48_105_prelast_action_equivariance_localization import (
    ALGORITHM_NAME,
    ENGINEERING_VERSION,
    TOKEN_GROUP_ORDER,
    action_interaction_slice,
    prelast_action_features,
    summary_group_slices,
)
from tools.run_v48_102_stage_i_action_information_transport_audit import (
    LinearProbe,
    ROLES,
    action_metrics,
    action_subset,
    build_v93_map,
    fit_binary,
    label_groups,
    permute_within_group,
    read_jsonl,
    sha256,
    split_role,
    state_metrics,
    state_records,
)

V104_ENGINEERING_VERSION = "v48.104.0-OC-NICR"
SEMANTIC_TOKEN_COUNT = 11


def _input_tokens(enc: StructuredTokenEncoder, x: torch.Tensor) -> torch.Tensor:
    ego, prefix_param, macro, scalar, prefix_state, control, agent_summary, agents, bev, route, maps, dyn = enc._split(x)
    B = x.shape[0]
    tokens = [
        enc.ego_proj(ego),
        enc.prefix_param_proj(prefix_param),
        enc.macro_scalar_proj(torch.cat([macro, scalar], dim=-1)),
        enc.prefix_state_proj(prefix_state),
        enc.control_proj(control),
        enc.agent_summary_proj(agent_summary),
        enc.bev_proj(bev),
        enc.route_proj(route),
        enc.map_proj(maps),
        enc.dyn_proj(dyn),
    ]
    tok = torch.stack(tokens, dim=1)
    tok = torch.cat([enc.cls.expand(B, -1, -1), tok, enc.agent_proj(agents)], dim=1)
    return tok + enc.pos[:, : tok.shape[1], :]


def _prelast_memory(enc: StructuredTokenEncoder, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if len(enc.encoder.layers) < 1:
        raise RuntimeError("V48.105 requires at least one Stage-I Transformer layer")
    h = _input_tokens(enc, x)
    for layer in enc.encoder.layers[:-1]:
        h = layer(h)
    pre = h
    final = enc.norm(enc.encoder.layers[-1](pre))
    return pre, final


def feature_cache_key(checkpoint: Path, index_path: Path, role_filter: str | None, v93_path: Path | None) -> str:
    payload = {
        "version": ENGINEERING_VERSION,
        "checkpoint": sha256(checkpoint),
        "index": sha256(index_path),
        "role": role_filter,
        "v93": sha256(v93_path) if v93_path and v93_path.is_file() else None,
        "kind": "pre_last_stage_i_v102_summary",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _stack(items: list[dict[str, Any]], key: str) -> torch.Tensor:
    return torch.stack([x[key] for x in items], dim=0)


def extract_prelast_features(
    *,
    checkpoint: Path,
    index_path: Path,
    role_filter: str | None,
    v93_path: Path | None,
    cache_dir: Path,
    device: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    key = feature_cache_key(checkpoint, index_path, role_filter, v93_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cp = cache_dir / f"{key}.pt"
    if cp.is_file():
        obj = torch.load(cp, map_location="cpu", weights_only=False)
        if obj.get("cache_key") == key:
            return obj["records"], {"feature_cache": "hit", "cache_key": key, **(obj.get("event") or {})}

    v93 = build_v93_map(v93_path)
    groups = label_groups(index_path, role_filter=role_filter, v93_map=v93)
    needed: list[Path] = []
    for g in groups:
        needed.append(Path(g["nominal_path"]))
        needed.extend(Path(c["path"]) for c in g["candidates"])
    seen: set[str] = set()
    paths: list[Path] = []
    for p in needed:
        q = str(p.resolve())
        if q not in seen:
            seen.add(q)
            paths.append(p)

    runtime = {"training": {"device": device}}
    bundle = load_model_bundle(checkpoint, runtime)
    if bundle is None:
        raise RuntimeError(f"cannot load checkpoint {checkpoint}")
    model = bundle.model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if not isinstance(model.encoder, StructuredTokenEncoder):
        raise RuntimeError("V48.105 requires StructuredTokenEncoder")
    enc = model.encoder.eval()
    dev = bundle.device
    cfg, feature_cfg_event = feature_only_dataset_cfg(bundle.cfg, cache_dir=str(cache_dir / "tensor"), workers=8)
    ds = OCRAPSampleDataset(paths, cfg)
    if ds.absolute_truth_contract_event.get("enabled") or ds.action_response_truth_event.get("enabled"):
        raise RuntimeError("V48.105 feature-only dataset unexpectedly attached truth sidecars")
    idx = {str(p.resolve()): i for i, p in enumerate(ds.paths)}
    records: list[dict[str, Any]] = []
    memory_shape: tuple[int, int] | None = None
    max_reconstruction = 0.0
    with torch.no_grad():
        for g in groups:
            ordered = [g["nominal_path"]] + [c["path"] for c in g["candidates"]]
            if any(str(Path(p).resolve()) not in idx for p in ordered):
                continue
            items = [ds[idx[str(Path(p).resolve())]] for p in ordered]
            x = _stack(items, "x").to(dev)
            prelast, historical_final = _prelast_memory(enc, x)
            direct = model._scene_tokens(x)
            max_reconstruction = max(max_reconstruction, float((historical_final - direct).abs().max().item()))
            if memory_shape is None:
                memory_shape = (int(prelast.shape[1]), int(prelast.shape[2]))
            elif memory_shape != (int(prelast.shape[1]), int(prelast.shape[2])):
                raise RuntimeError("V48.105 pre-last Stage-I memory geometry drift")
            state, delta, context = prelast_action_features(prelast)
            snp, dnp, cnp = state.cpu().numpy(), delta.cpu().numpy(), context.cpu().numpy()
            for j, c in enumerate(g["candidates"]):
                records.append({
                    "group": tuple(g["key"]),
                    "candidate": int(c["candidate"]),
                    "group_mode": g["group_mode"],
                    "safe_positive": bool(c["safe_positive"]),
                    "teacher_harmful": bool(c["teacher_harmful"]),
                    "mediation_mode": c["mediation_mode"],
                    "state": snp[j],
                    "delta": dnp[j],
                    "context": cnp[j],
                })
    if max_reconstruction > 1.0e-6:
        raise RuntimeError(f"V48.105 pre-last reconstruction mismatch {max_reconstruction}")
    d_model = int(memory_shape[1]) if memory_shape else 0
    event = {
        "tensor_cache_event": ds.tensor_cache_event,
        "feature_only_dataset_contract": feature_cfg_event,
        "records": len(records),
        "groups": len(groups),
        "prelast_memory_shape": list(memory_shape or (0, 0)),
        "stage_i_summary_dim": int(records[0]["state"].shape[0]) if records else 0,
        "semantic_token_count": SEMANTIC_TOKEN_COUNT,
        "agent_token_summary": "mean_std_max_min",
        "encoder_layer_count": int(len(enc.encoder.layers)),
        "prelast_after_layer_count": max(int(len(enc.encoder.layers)) - 1, 0),
        "historical_final_reconstruction_max_abs": max_reconstruction,
        "d_model": d_model,
    }
    torch.save({"cache_key": key, "records": records, "event": event}, cp)
    return records, {"feature_cache": "miss", "cache_key": key, **event}


def _slice_matrix(records: list[dict[str, Any]], key: str, sl: slice) -> np.ndarray:
    return np.stack([np.asarray(r[key])[sl] for r in records])


def _fit_action_pair(records: list[dict[str, Any]], key: str, device: str, sl: slice | None = None):
    y = np.asarray([r["label"] for r in records], dtype=np.int64)
    X = np.stack([r[key] for r in records]) if sl is None else _slice_matrix(records, key, sl)
    model, mu, sd = fit_binary(X, y, device, seed=102)
    perm = permute_within_group(records, key)
    if sl is not None:
        perm = perm[:, sl]
    pmodel, pmu, psd = fit_binary(perm, y, device, seed=102)
    return (model, mu, sd), (pmodel, pmu, psd)


def _eval_action_pair(records, key, true_probe, shuf_probe, device, sl: slice | None = None):
    if not records:
        z = {"rows": 0, "positive_rows": 0, "negative_rows": 0, "auc": None, "top1": None, "powered_groups": 0}
        return z, dict(z)
    X = np.stack([r[key] for r in records])
    Xs = permute_within_group(records, key)
    if sl is not None:
        X = X[:, sl]
        Xs = Xs[:, sl]
    tm, tmu, tsd = true_probe
    pm, pmu, psd = shuf_probe
    true = action_metrics(records, key, tm, tmu, tsd, device, X_override=X)
    shuf = action_metrics(records, key, pm, pmu, psd, device, X_override=Xs)
    true["auc_vs_shuffled"] = None if true["auc"] is None or shuf["auc"] is None else float(true["auc"] - shuf["auc"])
    true["top1_vs_shuffled"] = None if true["top1"] is None or shuf["top1"] is None else float(true["top1"] - shuf["top1"])
    return true, shuf


def _linear_cka(X: np.ndarray, y: np.ndarray) -> float | None:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    if len(X) < 2 or len(np.unique(y)) < 2:
        return None
    X = X - X.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    # Linear CKA in Gram form.  Held-out target-specific cells are small while
    # semantic groups can be hundreds of dimensions, so N x N Gram matrices
    # are both exact and substantially cheaper than D x D covariance matrices.
    K = X @ X.T
    L = y @ y.T
    num = float((K * L).sum())
    den = float(np.sqrt((K * K).sum() * (L * L).sum()))
    if den <= 0.0:
        return None
    return float(num / den)


def _localization_metrics(records: list[dict[str, Any]], key: str, d_model: int) -> dict[str, Any]:
    if not records:
        return {}
    y = np.asarray([r["label"] for r in records], dtype=np.int64)
    X = np.stack([r[key] for r in records]).astype(np.float64, copy=False)
    Xs = permute_within_group(records, key).astype(np.float64, copy=False)
    total = np.square(X).sum(axis=1)
    groups = summary_group_slices(d_model)
    out: dict[str, Any] = {}
    for name in TOKEN_GROUP_ORDER:
        sl = groups[name]
        G, Gs = X[:, sl], Xs[:, sl]
        e = np.square(G).sum(axis=1)
        share = np.divide(e, total, out=np.zeros_like(e), where=total > 1.0e-12)
        cka = _linear_cka(G, y)
        ckas = _linear_cka(Gs, y)
        out[name] = {
            "dimension": int(sl.stop - sl.start),
            "mean_action_energy_share": float(share.mean()) if len(share) else None,
            "label_linear_cka": cka,
            "shuffled_linear_cka": ckas,
            "cka_minus_shuffled": None if cka is None or ckas is None else float(cka - ckas),
        }
    return out


def _probe_state_dict(m: LinearProbe, mu: np.ndarray, sd: np.ndarray) -> dict[str, Any]:
    return {"state_dict": {k: v.detach().cpu() for k, v in m.state_dict().items()}, "mu": mu, "sd": sd}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--train-index", type=Path, required=True)
    ap.add_argument("--dev-index", type=Path, required=True)
    ap.add_argument("--certificate-index", type=Path, required=True)
    ap.add_argument("--v93-audit", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--variant", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--state-output", type=Path, required=True)
    a = ap.parse_args()
    t0 = time.perf_counter()
    resolved = a.device if (not a.device.startswith("cuda") or torch.cuda.is_available()) else "cpu"

    tr, etr = extract_prelast_features(checkpoint=a.checkpoint, index_path=a.train_index, role_filter=None, v93_path=None, cache_dir=a.cache_dir / "train", device=resolved)
    dv: list[dict[str, Any]] = []
    events: dict[str, Any] = {"train": etr}
    for role in ("dev_near", "dev_contact"):
        rr, e = extract_prelast_features(checkpoint=a.checkpoint, index_path=a.dev_index, role_filter=role, v93_path=a.v93_audit, cache_dir=a.cache_dir / role, device=resolved)
        dv.extend(rr); events[role] = e
    ce: list[dict[str, Any]] = []
    for role in ("certificate_near", "certificate_contact"):
        rr, e = extract_prelast_features(checkpoint=a.checkpoint, index_path=a.certificate_index, role_filter=role, v93_path=a.v93_audit, cache_dir=a.cache_dir / role, device=resolved)
        ce.extend(rr); events[role] = e
    if not tr or not dv or not ce:
        raise RuntimeError("V48.105 empty pre-last audit feature set")

    d_model = int(etr.get("d_model", 0))
    if d_model <= 0 or int(etr.get("stage_i_summary_dim", -1)) != 15 * d_model:
        raise RuntimeError("V48.105 pre-last summary dimension contract")
    action_sl = action_interaction_slice(d_model)

    st_tr = state_records(tr)
    sup_tr = action_subset(tr, "drs_activation")
    res_tr = action_subset(tr, "deployability_gain")
    sm, smu, ssd = fit_binary(np.stack([r["state"] for r in st_tr]), np.asarray([r["label"] for r in st_tr]), resolved, seed=102)
    sup_true, sup_shuf = _fit_action_pair(sup_tr, "delta", resolved)
    res_true, res_shuf = _fit_action_pair(res_tr, "context", resolved)
    ais_true, ais_shuf = _fit_action_pair(sup_tr, "delta", resolved, action_sl)
    air_true, air_shuf = _fit_action_pair(res_tr, "context", resolved, action_sl)

    cells: dict[str, Any] = {}
    ai_cells: dict[str, Any] = {}
    localization: dict[str, Any] = {}
    for role in ROLES:
        src = dv if role.startswith("dev_") else ce
        rr = split_role(src, role)
        sr = state_records(rr)
        ur = action_subset(rr, "drs_activation")
        qr = action_subset(rr, "deployability_gain")
        smet = state_metrics(sr, sm, smu, ssd, resolved)
        utrue, ushuf = _eval_action_pair(ur, "delta", sup_true, sup_shuf, resolved)
        rtrue, rshuf = _eval_action_pair(qr, "context", res_true, res_shuf, resolved)
        autrue, aushuf = _eval_action_pair(ur, "delta", ais_true, ais_shuf, resolved, action_sl)
        artrue, arshuf = _eval_action_pair(qr, "context", air_true, air_shuf, resolved, action_sl)
        cells[role] = {"state": smet, "support_true": utrue, "support_shuffled": ushuf, "reserve_true": rtrue, "reserve_shuffled": rshuf}
        ai_cells[role] = {"support_true": autrue, "support_shuffled": aushuf, "reserve_true": artrue, "reserve_shuffled": arshuf}
        localization[role] = {
            "support": _localization_metrics(ur, "delta", d_model),
            "reserve": _localization_metrics(qr, "context", d_model),
        }

    result = {
        "schema": "ocrap-v48.105-prelast-action-equivariance-localization-audit-v1",
        "engineering_version": ENGINEERING_VERSION,
        "algorithm_name": ALGORITHM_NAME,
        "valid": True,
        "variant": a.variant,
        "audit_only": True,
        "checkpoint": str(a.checkpoint.resolve()),
        "checkpoint_sha256": sha256(a.checkpoint),
        "elapsed_seconds": time.perf_counter() - t0,
        "cells": cells,
        "action_interaction_cells": ai_cells,
        "token_localization": localization,
        "events": events,
        "train_counts": {"state": len(st_tr), "support": len(sup_tr), "reserve": len(res_tr)},
        "stage_i_parameters_trained": 0,
        "root_decoder_parameters_trained": 0,
        "source_parameters_trained": 0,
        "planner_parameters_trained": 0,
        "relative_ranker_modified": False,
        "boundary_transport": False,
        "regime_conditioning": False,
        "teacher_metadata_input_to_model": False,
        "test_roots_read": False,
        "same_v48_102_summary_operator": True,
        "same_v48_102_linear_probe_recipe": True,
        "prelast_only": True,
        "action_interaction_subspace": {
            "start": int(action_sl.start), "stop": int(action_sl.stop), "dimension": int(action_sl.stop - action_sl.start),
            "definition": "control_plus_scene_context_plus_agent_set_moments_excluding_cls_and_ego_history",
        },
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    # Probe checkpoint is audit provenance only; no planner/model parameter is stored.
    state_obj = {
        "schema": "ocrap-v48.105-pael-probe-state-v1",
        "engineering_version": ENGINEERING_VERSION,
        "algorithm_name": ALGORITHM_NAME,
        "variant": a.variant,
        "d_model": d_model,
        "global_state_probe": _probe_state_dict(sm, smu, ssd),
        "global_support_probe": _probe_state_dict(*sup_true),
        "global_support_shuffle_probe": _probe_state_dict(*sup_shuf),
        "global_reserve_probe": _probe_state_dict(*res_true),
        "global_reserve_shuffle_probe": _probe_state_dict(*res_shuf),
        "action_interaction_support_probe": _probe_state_dict(*ais_true),
        "action_interaction_support_shuffle_probe": _probe_state_dict(*ais_shuf),
        "action_interaction_reserve_probe": _probe_state_dict(*air_true),
        "action_interaction_reserve_shuffle_probe": _probe_state_dict(*air_shuf),
        "checkpoint_sha256": sha256(a.checkpoint),
        "train_counts": result["train_counts"],
    }
    torch.save(state_obj, a.state_output)
    print(json.dumps({"valid": True, "variant": a.variant, "elapsed_seconds": result["elapsed_seconds"], "output": str(a.output), "state": str(a.state_output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
