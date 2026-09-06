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
from ocrap.v48_97_executable_recovery_state import (
    ALGORITHM_NAME,
    ENGINEERING_VERSION,
    ExecutableRecoverySufficientState,
    semantic_loss,
)

ROLES = ("dev_near", "dev_contact", "certificate_near", "certificate_contact")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except Exception as exc:
                raise ValueError(f"invalid JSONL {path}:{i}: {exc}") from exc
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    pos = int(y.sum())
    neg = int(len(y) - pos)
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and s[order[j]] == s[order[i]]:
            j += 1
        r = (i + j + 1) / 2.0
        ranks[order[i:j]] = r
        i = j
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def _root_probs(model, root_tokens: torch.Tensor, root_valid: torch.Tensor) -> torch.Tensor:
    logits = model.root_logit_head(root_tokens).squeeze(-1).float()
    mask = root_valid.bool()
    logits = logits.masked_fill(~mask, -1.0e9)
    p = torch.softmax(logits, dim=-1) * mask.float()
    return p / p.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)


def _index_rows(index_path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(index_path)
    required = {"path", "bucket", "scene", "time", "candidate", "nominal", "teacher_drs", "teacher_r_dep"}
    for i, r in enumerate(rows):
        missing = required.difference(r)
        if missing:
            raise ValueError(f"teacher index {index_path} row {i} missing {sorted(missing)}")
    return rows


def _pair_indices(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    by: dict[tuple[int, str, int], list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by[(int(r["bucket"]), str(r["scene"]), int(r["time"]))].append(i)
    cand: list[int] = []
    nom: list[int] = []
    for key, ids in by.items():
        ns = [i for i in ids if bool(rows[i].get("nominal", False))]
        if len(ns) != 1:
            continue
        ni = ns[0]
        for i in ids:
            if i == ni:
                continue
            cand.append(i)
            nom.append(ni)
    return np.asarray(cand, dtype=np.int64), np.asarray(nom, dtype=np.int64)


def _feature_cache_key(checkpoint: Path, index_path: Path, variant: str) -> str:
    payload = {
        "engineering_version": ENGINEERING_VERSION,
        "checkpoint": sha256(checkpoint),
        "index": sha256(index_path),
        "variant": variant,
        "kind": "frozen_root_set",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def extract_root_set_features(
    *, checkpoint: Path, index_path: Path, cache_dir: Path, device: str, variant: str,
) -> dict[str, Any]:
    key = _feature_cache_key(checkpoint, index_path, variant)
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
    dev = bundle.device
    cfg, feature_cfg_event = feature_only_dataset_cfg(
        bundle.cfg, cache_dir=str(cache_dir / "tensor"), workers=8
    )
    ds = OCRAPSampleDataset(paths, cfg)
    if ds.absolute_truth_contract_event.get("enabled") or ds.action_response_truth_event.get("enabled"):
        raise RuntimeError("V48.97 feature-only dataset unexpectedly attached supervision truth sidecars")
    # The index order is part of the supervision contract.  Fail closed if the
    # Dataset canonicalizes it differently.
    resolved_index = [str(p.resolve()) for p in paths]
    resolved_ds = [str(p.resolve()) for p in ds.paths]
    if resolved_index != resolved_ds:
        raise RuntimeError("V48.97 dataset path order differs from teacher-index order")

    root_tokens: list[torch.Tensor] = []
    root_probs: list[torch.Tensor] = []
    root_valid: list[torch.Tensor] = []
    batch_size = 256
    with torch.no_grad():
        for st in range(0, len(ds), batch_size):
            items = [ds[i] for i in range(st, min(len(ds), st + batch_size))]
            x = torch.stack([it["x"] for it in items], dim=0).to(dev)
            rv = torch.stack([it["root_valid"] for it in items], dim=0).to(dev)
            memory = model._scene_tokens(x)
            rt = model._decode_roots(memory.detach())
            rp = _root_probs(model, rt, rv)
            root_tokens.append(rt.detach().float().cpu())
            root_probs.append(rp.detach().float().cpu())
            root_valid.append(rv.detach().bool().cpu())
    obj = {
        "cache_key": key,
        "checkpoint": str(checkpoint.resolve()),
        "index": str(index_path.resolve()),
        "rows": rows,
        "root_tokens": torch.cat(root_tokens, dim=0),
        "root_probs": torch.cat(root_probs, dim=0),
        "root_valid": torch.cat(root_valid, dim=0),
        "feature_only_dataset_contract": feature_cfg_event,
        "tensor_cache_event": ds.tensor_cache_event,
    }
    torch.save(obj, cache)
    return obj


def _semantic_loss_eval(
    module: ExecutableRecoverySufficientState, obj: dict[str, Any], device: str,
) -> tuple[float, dict[str, float], np.ndarray, np.ndarray]:
    dev = torch.device(device)
    rt = obj["root_tokens"].to(dev)
    rp = obj["root_probs"].to(dev)
    rv = obj["root_valid"].to(dev)
    rows = obj["rows"]
    td = torch.tensor([float(r["teacher_drs"]) for r in rows], dtype=torch.float32, device=dev)
    tr = torch.tensor([float(r["teacher_r_dep"]) for r in rows], dtype=torch.float32, device=dev)
    ci_np, ni_np = _pair_indices(rows)
    ci = torch.tensor(ci_np, dtype=torch.long, device=dev)
    ni = torch.tensor(ni_np, dtype=torch.long, device=dev)
    module.eval()
    with torch.no_grad():
        out = module(rt, rp, rv)
        total, parts = semantic_loss(out, td, tr, ci, ni)
    return (
        float(total.item()),
        {k: float(v.item()) for k, v in parts.items()},
        out["support"].detach().cpu().numpy(),
        out["reserve_debt"].detach().cpu().numpy(),
    )


def train_representation(
    *, module: ExecutableRecoverySufficientState, train_obj: dict[str, Any], dev_obj: dict[str, Any],
    device: str, max_epochs: int = 60, patience: int = 10,
) -> dict[str, Any]:
    torch.manual_seed(97)
    np.random.seed(97)
    random.seed(97)
    dev = torch.device(device)
    module.to(dev)
    rt = train_obj["root_tokens"].to(dev)
    rp = train_obj["root_probs"].to(dev)
    rv = train_obj["root_valid"].to(dev)
    rows = train_obj["rows"]
    td = torch.tensor([float(r["teacher_drs"]) for r in rows], dtype=torch.float32, device=dev)
    tr = torch.tensor([float(r["teacher_r_dep"]) for r in rows], dtype=torch.float32, device=dev)
    ci_np, ni_np = _pair_indices(rows)
    ci = torch.tensor(ci_np, dtype=torch.long, device=dev)
    ni = torch.tensor(ni_np, dtype=torch.long, device=dev)
    opt = torch.optim.AdamW(module.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    best = float("inf")
    best_state = None
    best_epoch = -1
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(int(max_epochs)):
        module.train()
        out = module(rt, rp, rv)
        loss, parts = semantic_loss(out, td, tr, ci, ni)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(module.parameters(), 5.0)
        opt.step()
        dev_loss, dev_parts, _, _ = _semantic_loss_eval(module, dev_obj, device)
        history.append({
            "epoch": epoch,
            "train_loss": float(loss.item()),
            "train_parts": {k: float(v.item()) for k, v in parts.items()},
            "dev_loss": dev_loss,
            "dev_parts": dev_parts,
        })
        if dev_loss < best - 1.0e-5:
            best = dev_loss
            best_epoch = epoch
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
        else:
            stale += 1
        if stale >= int(patience):
            break
    if best_state is None:
        raise RuntimeError("V48.97 representation training did not produce a checkpoint")
    module.load_state_dict(best_state)
    module.to(dev)
    return {
        "best_epoch": int(best_epoch),
        "best_dev_semantic_loss": float(best),
        "epochs_completed": len(history),
        "history": history,
    }


def build_v93_map(path: Path) -> dict[tuple[str, str, int, int], dict[str, Any]]:
    out: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    for r in read_jsonl(path):
        role = str(r.get("dataset_role"))
        if role not in ROLES:
            continue
        k = (role, str(r["scene_id"]), int(r["time_index"]), int(r["candidate_index"]))
        if k in out:
            raise ValueError(f"duplicate V48.93 key {k}")
        out[k] = r
    return out


def _role_rows(
    obj: dict[str, Any], support: np.ndarray, reserve: np.ndarray, role: str,
    v93_map: dict[tuple[str, str, int, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join candidate-only V48.93 labels without dropping nominal rows.

    V48.93 factor-mediation audit intentionally contains candidate rows only;
    nominal rows live in the dense teacher-PCD index.  The V48.97 evaluation
    needs both: candidate labels for safe/harmful/mode semantics and the
    matching nominal prediction for state/action deltas.  Therefore nominal
    rows are retained directly from the teacher index, while non-nominal rows
    are admitted only when their V48.93 label exists.
    """
    bucket = 1 if "near" in role else 2
    out: list[dict[str, Any]] = []
    for i, r in enumerate(obj["rows"]):
        if int(r["bucket"]) != bucket:
            continue
        nominal = bool(r.get("nominal", False))
        base = {
            "scene": str(r["scene"]), "time": int(r["time"]), "candidate": int(r["candidate"]),
            "nominal": nominal,
            "support": float(support[i]), "reserve": float(reserve[i]),
            "teacher_drs": float(r["teacher_drs"]), "teacher_r_dep": float(r["teacher_r_dep"]),
        }
        if nominal:
            # V48.93 has no nominal rows by construction.  Nominal values are
            # needed only as the observation-conditioned reference prediction;
            # they never contribute a positive/harmful mediation label.
            out.append({
                **base,
                "safe_positive": False,
                "teacher_harmful": False,
                "mediation_mode": "nominal",
            })
            continue
        key = (role, str(r["scene"]), int(r["time"]), int(r["candidate"]))
        v = v93_map.get(key)
        if v is None:
            continue
        out.append({
            **base,
            "safe_positive": bool(v.get("safe_positive")),
            "teacher_harmful": bool(v.get("teacher_harmful")),
            "mediation_mode": str(v.get("mediation_mode")),
        })
    return out


def _evaluation_contract(rows: list[dict[str, Any]], role: str) -> dict[str, Any]:
    by: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by[(str(r["scene"]), int(r["time"]))].append(r)
    candidate_groups = {
        g for g, rs in by.items() if any(not bool(r["nominal"]) for r in rs)
    }
    missing_nominal: list[str] = []
    duplicate_nominal: list[str] = []
    for g in sorted(candidate_groups):
        n = sum(bool(r["nominal"]) for r in by[g])
        if n == 0:
            missing_nominal.append(f"{g[0]}@{g[1]}")
        elif n != 1:
            duplicate_nominal.append(f"{g[0]}@{g[1]}:{n}")
    candidates = [r for r in rows if not bool(r["nominal"])]
    safe = [r for r in candidates if bool(r["safe_positive"])]
    harmful = [r for r in candidates if bool(r["teacher_harmful"])]
    drs = [r for r in safe if r["mediation_mode"] == "drs_activation"]
    dep = [r for r in safe if r["mediation_mode"] == "deployability_gain"]
    out = {
        "role": role,
        "joined_rows": len(rows),
        "nominal_rows": sum(bool(r["nominal"]) for r in rows),
        "matched_candidate_label_rows": len(candidates),
        "candidate_groups": len(candidate_groups),
        "safe_positive_rows": len(safe),
        "harmful_rows": len(harmful),
        "drs_activation_rows": len(drs),
        "deployability_gain_rows": len(dep),
        "missing_nominal_groups": missing_nominal[:10],
        "duplicate_nominal_groups": duplicate_nominal[:10],
        "valid": True,
        "errors": [],
    }
    errors: list[str] = []
    if not candidates:
        errors.append("no_matched_candidate_labels")
    if not safe:
        errors.append("no_safe_positive_rows")
    if not harmful:
        errors.append("no_harmful_rows")
    if not drs:
        errors.append("no_drs_activation_rows")
    if not dep:
        errors.append("no_deployability_gain_rows")
    if missing_nominal:
        errors.append(f"missing_nominal_groups={len(missing_nominal)}")
    if duplicate_nominal:
        errors.append(f"duplicate_nominal_groups={len(duplicate_nominal)}")
    out["errors"] = errors
    out["valid"] = not errors
    return out


def candidate_only_label_join_synthetic_check() -> bool:
    obj = {
        "rows": [
            {"bucket": 1, "scene": "s1", "time": 1, "candidate": 0, "nominal": True, "teacher_drs": 0.2, "teacher_r_dep": -0.1},
            {"bucket": 1, "scene": "s1", "time": 1, "candidate": 5, "nominal": False, "teacher_drs": 0.8, "teacher_r_dep": 0.4},
            {"bucket": 1, "scene": "s1", "time": 1, "candidate": 7, "nominal": False, "teacher_drs": 0.1, "teacher_r_dep": -0.4},
            {"bucket": 1, "scene": "s2", "time": 2, "candidate": 0, "nominal": True, "teacher_drs": 1.0, "teacher_r_dep": 0.1},
            {"bucket": 1, "scene": "s2", "time": 2, "candidate": 6, "nominal": False, "teacher_drs": 1.0, "teacher_r_dep": 0.7},
            {"bucket": 1, "scene": "s2", "time": 2, "candidate": 8, "nominal": False, "teacher_drs": 0.7, "teacher_r_dep": -0.3},
        ]
    }
    support = np.asarray([0.25, 0.75, 0.1, 0.9, 0.95, 0.6], dtype=np.float64)
    reserve = np.asarray([-0.2, 0.3, -0.5, 0.1, 0.8, -0.4], dtype=np.float64)
    labels = {
        ("dev_near", "s1", 1, 5): {"safe_positive": True, "teacher_harmful": False, "mediation_mode": "drs_activation"},
        ("dev_near", "s1", 1, 7): {"safe_positive": False, "teacher_harmful": True, "mediation_mode": "redundant_or_interaction"},
        ("dev_near", "s2", 2, 6): {"safe_positive": True, "teacher_harmful": False, "mediation_mode": "deployability_gain"},
        ("dev_near", "s2", 2, 8): {"safe_positive": False, "teacher_harmful": True, "mediation_mode": "redundant_or_interaction"},
    }
    rows = _role_rows(obj, support, reserve, "dev_near", labels)
    c = _evaluation_contract(rows, "dev_near")
    return bool(
        len(rows) == 6
        and sum(bool(r["nominal"]) for r in rows) == 2
        and c["valid"]
        and c["matched_candidate_label_rows"] == 4
        and c["drs_activation_rows"] == 1
        and c["deployability_gain_rows"] == 1
    )


def _permute_scores_within_group(rows: list[dict[str, Any]], scores: np.ndarray) -> np.ndarray:
    out = np.asarray(scores, dtype=np.float64).copy()
    by: dict[tuple[str, int], list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by[(r["scene"], int(r["time"]))].append(i)
    for ids in by.values():
        ids = sorted(ids, key=lambda i: int(rows[i]["candidate"]))
        vals = out[ids].copy()
        out[ids] = np.roll(vals, 1)
    return out


def _action_metric(rows: list[dict[str, Any]], mode: str, score_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    # Build nominal lookup first.
    nom: dict[tuple[str, int], dict[str, Any]] = {}
    for r in rows:
        if r["nominal"]:
            nom[(r["scene"], int(r["time"]))] = r
    rr: list[dict[str, Any]] = []
    scores: list[float] = []
    labels: list[int] = []
    groups: list[tuple[str, int]] = []
    for r in rows:
        if r["nominal"]:
            continue
        n = nom.get((r["scene"], int(r["time"])))
        if n is None:
            continue
        positive = bool(r["safe_positive"] and r["mediation_mode"] == mode)
        negative = bool(r["teacher_harmful"])
        if not (positive or negative):
            continue
        score = float(r[score_name] - n[score_name])
        rr.append(r)
        scores.append(score)
        labels.append(1 if positive else 0)
        groups.append((r["scene"], int(r["time"])))
    if not rr:
        empty = {"rows": 0, "positive_rows": 0, "negative_rows": 0, "auc": None, "top1": None, "powered_groups": 0}
        return empty, empty.copy()
    y = np.asarray(labels, dtype=np.int64)
    sc = np.asarray(scores, dtype=np.float64)
    sh = _permute_scores_within_group(rr, sc)

    def metric(values: np.ndarray) -> dict[str, Any]:
        by: dict[tuple[str, int], list[int]] = defaultdict(list)
        for i, g in enumerate(groups):
            by[g].append(i)
        powered = [ids for ids in by.values() if any(y[i] == 1 for i in ids) and any(y[i] == 0 for i in ids)]
        top1 = float(np.mean([y[max(ids, key=lambda i: float(values[i]))] == 1 for ids in powered])) if powered else None
        return {
            "rows": len(rr), "positive_rows": int(y.sum()), "negative_rows": int(len(y) - y.sum()),
            "auc": auc(y, values), "top1": top1, "powered_groups": len(powered),
        }
    a = metric(sc)
    b = metric(sh)
    a["auc_vs_shuffled"] = None if a["auc"] is None or b["auc"] is None else float(a["auc"] - b["auc"])
    a["top1_vs_shuffled"] = None if a["top1"] is None or b["top1"] is None else float(a["top1"] - b["top1"])
    return a, b


def _state_metric(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by[(r["scene"], int(r["time"]))].append(r)
    labels: list[int] = []
    scores: list[float] = []
    for rs in by.values():
        n = next((r for r in rs if r["nominal"]), None)
        if n is None:
            continue
        modes = {r["mediation_mode"] for r in rs if r["safe_positive"] and r["mediation_mode"] in {"drs_activation", "deployability_gain"}}
        if len(modes) != 1:
            continue
        mode = next(iter(modes))
        labels.append(1 if mode == "drs_activation" else 0)
        # Low nominal support predicts support-establishment mode.
        scores.append(1.0 - float(n["support"]))
    y = np.asarray(labels, dtype=np.int64)
    sc = np.asarray(scores, dtype=np.float64)
    return {
        "rows": len(y), "drs_state_rows": int(y.sum()) if len(y) else 0,
        "dep_state_rows": int(len(y) - y.sum()) if len(y) else 0,
        "auc": auc(y, sc) if len(y) else None,
    }


def _dense_metrics(obj: dict[str, Any], support: np.ndarray, reserve: np.ndarray) -> dict[str, float]:
    rows = obj["rows"]
    td = np.asarray([float(r["teacher_drs"]) for r in rows], dtype=np.float64)
    tr = np.asarray([float(r["teacher_r_dep"]) for r in rows], dtype=np.float64)
    support_mae = float(np.mean(np.abs(support - td)))
    d = np.abs(reserve - tr)
    reserve_huber = float(np.mean(np.where(d < 1.0, 0.5 * d * d, d - 0.5)))
    ci, ni = _pair_indices(rows)
    if len(ci):
        ds = (support[ci] - support[ni]) - (td[ci] - td[ni])
        dr = (reserve[ci] - reserve[ni]) - (tr[ci] - tr[ni])
        delta_support_mae = float(np.mean(np.abs(ds)))
        adr = np.abs(dr)
        delta_reserve_huber = float(np.mean(np.where(adr < 1.0, 0.5 * adr * adr, adr - 0.5)))
    else:
        delta_support_mae = float("nan")
        delta_reserve_huber = float("nan")
    return {
        "support_mae": support_mae,
        "reserve_huber": reserve_huber,
        "delta_support_mae": delta_support_mae,
        "delta_reserve_huber": delta_reserve_huber,
    }


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
    args = ap.parse_args()
    t0 = time.perf_counter()

    train_obj = extract_root_set_features(
        checkpoint=args.checkpoint, index_path=args.train_index, cache_dir=args.cache_dir / "train",
        device=args.device, variant=args.variant,
    )
    dev_obj = extract_root_set_features(
        checkpoint=args.checkpoint, index_path=args.dev_index, cache_dir=args.cache_dir / "dev",
        device=args.device, variant=args.variant,
    )
    cert_obj = extract_root_set_features(
        checkpoint=args.checkpoint, index_path=args.certificate_index, cache_dir=args.cache_dir / "certificate",
        device=args.device, variant=args.variant,
    )
    d_model = int(train_obj["root_tokens"].shape[-1])
    module = ExecutableRecoverySufficientState(d_model)
    training = train_representation(module=module, train_obj=train_obj, dev_obj=dev_obj, device=args.device)
    dev_loss, dev_parts, dev_support, dev_reserve = _semantic_loss_eval(module, dev_obj, args.device)
    cert_loss, cert_parts, cert_support, cert_reserve = _semantic_loss_eval(module, cert_obj, args.device)
    train_loss, train_parts, _, _ = _semantic_loss_eval(module, train_obj, args.device)
    v93 = build_v93_map(args.v93_audit)

    result: dict[str, Any] = {
        "schema": "ocrap-v48.97-erss-result-v1",
        "engineering_version": ENGINEERING_VERSION,
        "algorithm_name": ALGORITHM_NAME,
        "valid": True,
        "variant": args.variant,
        "planner_parameters_trained": 0,
        "representation_parameters_trained": module.trainable_parameter_count,
        "source_parameters_trained": 0,
        "regime_conditioning": False,
        "teacher_metadata_input_to_model": False,
        "root_slot_bijection_assumed": False,
        "checkpoint": str(args.checkpoint.resolve()),
        "training": training,
        "semantic_loss": {
            "train": {"total": train_loss, **train_parts},
            "dev": {"total": dev_loss, **dev_parts},
            "certificate": {"total": cert_loss, **cert_parts},
        },
        "dense_metrics": {
            "dev": _dense_metrics(dev_obj, dev_support, dev_reserve),
            "certificate": _dense_metrics(cert_obj, cert_support, cert_reserve),
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
        sp = dev_support if role.startswith("dev_") else cert_support
        rs = dev_reserve if role.startswith("dev_") else cert_reserve
        rr = _role_rows(obj, sp, rs, role, v93)
        contract = _evaluation_contract(rr, role)
        result["evaluation_contracts"][role] = contract
        if not contract["valid"]:
            raise RuntimeError(f"V48.97 evaluation join fail-closed for {role}: {contract['errors']}")
        state = _state_metric(rr)
        sup_true, sup_shuf = _action_metric(rr, "drs_activation", "support")
        res_true, res_shuf = _action_metric(rr, "deployability_gain", "reserve")
        metric_errors: list[str] = []
        if int(state.get("rows", 0)) <= 0 or state.get("auc") is None:
            metric_errors.append("state_empty_or_auc_null")
        for name, metric in (("support", sup_true), ("reserve", res_true)):
            if int(metric.get("positive_rows", 0)) <= 0 or int(metric.get("negative_rows", 0)) <= 0 or metric.get("auc") is None:
                metric_errors.append(f"{name}_empty_or_auc_null")
        if metric_errors:
            raise RuntimeError(f"V48.97 evaluation metric fail-closed for {role}: {metric_errors}")
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
        "schema": "ocrap-v48.97-erss-state-v1",
        "engineering_version": ENGINEERING_VERSION,
        "algorithm_name": ALGORITHM_NAME,
        "variant": args.variant,
        "d_model": d_model,
        "state_dict": {k: v.detach().cpu() for k, v in module.state_dict().items()},
        "trainable_parameter_count": module.trainable_parameter_count,
        "training": {k: v for k, v in training.items() if k != "history"},
    }, args.state_output)
    print(json.dumps({
        "valid": True, "variant": args.variant, "best_epoch": training["best_epoch"],
        "representation_parameters": module.trainable_parameter_count,
        "elapsed_seconds": result["elapsed_seconds"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
