#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ocrap.models.data import OCRAPSampleDataset
from ocrap.models.inference import load_model_bundle
from ocrap.v48_96_support_reserve_root_observability import (
    DEPLOYABLE_MACROS,
    POSITIVE_GAIN,
    VALID_MODES,
    derive_candidate_semantics,
    feature_only_dataset_cfg,
)
from ocrap.v48_102_action_information_transport_sufficiency import (
    ALGORITHM_NAME,
    ENGINEERING_VERSION,
    stage_i_action_features,
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
    pos, neg = int(y.sum()), int(len(y) - y.sum())
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and s[order[j]] == s[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j + 1) / 2.0
        i = j
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def build_v93_map(path: Path | None) -> dict[tuple[str, str, int, int], dict[str, Any]]:
    if path is None:
        return {}
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


def label_groups(
    index_path: Path,
    *,
    role_filter: str | None,
    v93_map: dict[tuple[str, str, int, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    raw = read_jsonl(index_path)
    by: dict[tuple[int, str, int], list[dict[str, Any]]] = defaultdict(list)
    for r in raw:
        by[(int(r["bucket"]), str(r["scene"]), int(r["time"]))].append(r)
    groups: list[dict[str, Any]] = []
    mismatches: list[Any] = []
    for key, rs in by.items():
        nom = [r for r in rs if bool(r.get("nominal", False))]
        if len(nom) != 1:
            continue
        n = nom[0]
        candidates: list[dict[str, Any]] = []
        for r in rs:
            if bool(r.get("nominal", False)) or int(r.get("macro", -1)) not in DEPLOYABLE_MACROS:
                continue
            sem = derive_candidate_semantics(n, r, positive_gain=POSITIVE_GAIN)
            rec = {
                "path": str(Path(r["path"]).resolve()),
                "candidate": int(r["candidate"]),
                "macro": int(r["macro"]),
                **sem,
            }
            if role_filter is not None:
                vk = (role_filter, str(r["scene"]), int(r["time"]), int(r["candidate"]))
                vr = v93_map.get(vk)
                if vr is None:
                    continue
                if bool(vr.get("safe_positive")) != bool(rec["safe_positive"]) or bool(vr.get("teacher_harmful")) != bool(rec["teacher_harmful"]):
                    mismatches.append((vk, "label"))
                if rec["safe_positive"] and str(vr.get("mediation_mode")) != str(rec["mediation_mode"]):
                    mismatches.append((vk, "mode"))
            candidates.append(rec)
        if role_filter is not None:
            candidates = [c for c in candidates if (role_filter, str(key[1]), int(key[2]), int(c["candidate"])) in v93_map]
        if not candidates:
            continue
        safe_modes = {str(c["mediation_mode"]) for c in candidates if c["safe_positive"] and c["mediation_mode"] in VALID_MODES}
        group_mode = next(iter(safe_modes)) if len(safe_modes) == 1 else None
        groups.append({
            "key": key,
            "nominal_path": str(Path(n["path"]).resolve()),
            "group_mode": group_mode,
            "candidates": candidates,
        })
    if mismatches:
        raise ValueError(f"V48.93 semantic mismatch examples={mismatches[:5]} total={len(mismatches)}")
    return groups


def feature_cache_key(checkpoint: Path, index_path: Path, role_filter: str | None, v93_path: Path | None) -> str:
    payload = {
        "version": ENGINEERING_VERSION,
        "checkpoint": sha256(checkpoint),
        "index": sha256(index_path),
        "role": role_filter,
        "v93": sha256(v93_path) if v93_path and v93_path.is_file() else None,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _stack(items: list[dict[str, Any]], key: str) -> torch.Tensor:
    return torch.stack([x[key] for x in items], dim=0)


def extract_stage_i_features(
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
    dev = bundle.device
    if getattr(model, "encoder_type", None) != "structured_transformer":
        raise RuntimeError("V48.102 requires structured_transformer Stage-I memory")
    cfg, feature_cfg_event = feature_only_dataset_cfg(bundle.cfg, cache_dir=str(cache_dir / "tensor"), workers=8)
    ds = OCRAPSampleDataset(paths, cfg)
    if ds.absolute_truth_contract_event.get("enabled") or ds.action_response_truth_event.get("enabled"):
        raise RuntimeError("V48.102 feature-only dataset unexpectedly attached supervision truth sidecars")
    idx = {str(p.resolve()): i for i, p in enumerate(ds.paths)}
    records: list[dict[str, Any]] = []
    memory_shape: tuple[int, int] | None = None
    with torch.no_grad():
        for g in groups:
            ordered = [g["nominal_path"]] + [c["path"] for c in g["candidates"]]
            if any(str(Path(p).resolve()) not in idx for p in ordered):
                continue
            items = [ds[idx[str(Path(p).resolve())]] for p in ordered]
            x = _stack(items, "x").to(dev)
            memory = model._scene_tokens(x)
            if memory_shape is None:
                memory_shape = (int(memory.shape[1]), int(memory.shape[2]))
            elif memory_shape != (int(memory.shape[1]), int(memory.shape[2])):
                raise RuntimeError("V48.102 Stage-I memory geometry drift")
            state, delta, context = stage_i_action_features(memory, semantic_token_count=11)
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
    event = {
        "tensor_cache_event": ds.tensor_cache_event,
        "feature_only_dataset_contract": feature_cfg_event,
        "records": len(records),
        "groups": len(groups),
        "stage_i_memory_shape": list(memory_shape or (0, 0)),
        "stage_i_summary_dim": int(records[0]["state"].shape[0]) if records else 0,
        "semantic_token_count": 11,
        "agent_token_summary": "mean_std_max_min",
    }
    torch.save({"cache_key": key, "records": records, "event": event}, cp)
    return records, {"feature_cache": "miss", "cache_key": key, **event}


class LinearProbe(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.linear = nn.Linear(d, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


def fit_binary(X: np.ndarray, y: np.ndarray, device: str, seed: int = 102):
    if len(X) < 4 or len(np.unique(y)) < 2:
        raise ValueError("V48.102 probe training requires both classes")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    mu = X.mean(0, keepdims=True)
    sd = X.std(0, keepdims=True)
    sd = np.where(sd > 1e-6, sd, 1.0)
    Z = (X - mu) / sd
    dev = torch.device(device)
    m = LinearProbe(Z.shape[1]).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    xt = torch.tensor(Z, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    pos = float(y.sum())
    neg = float(len(y) - pos)
    lossfn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(max(neg / max(pos, 1.0), 1.0), device=dev))
    batch = 512
    for _ in range(40):
        order = torch.randperm(len(xt))
        m.train()
        for st in range(0, len(xt), batch):
            ix = order[st:st + batch]
            o = m(xt[ix].to(dev))
            loss = lossfn(o, yt[ix].to(dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    m.eval()
    return m, mu, sd


def scores(model, mu, sd, X, device):
    Z = (X - mu) / sd
    with torch.no_grad():
        return model(torch.tensor(Z, dtype=torch.float32, device=torch.device(device))).cpu().numpy()


def permute_within_group(records: list[dict[str, Any]], key: str) -> np.ndarray:
    X = np.stack([r[key] for r in records]).copy()
    groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        groups[tuple(r["group"])].append(i)
    for ids in groups.values():
        ids = sorted(ids, key=lambda i: int(records[i]["candidate"]))
        vals = X[ids].copy()
        X[ids] = np.roll(vals, 1, axis=0)
    return X


def action_subset(records: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    out = []
    for r in records:
        if r.get("group_mode") != mode:
            continue
        positive = bool(r["safe_positive"] and r.get("mediation_mode") == mode)
        negative = bool(r["teacher_harmful"])
        if positive or negative:
            out.append({**r, "label": 1 if positive else 0})
    return out


def state_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by: dict[tuple[Any, ...], dict[str, Any]] = {}
    for r in records:
        mode = r.get("group_mode")
        if mode not in VALID_MODES:
            continue
        k = tuple(r["group"])
        rec = {"group": k, "state": r["state"], "label": 1 if mode == "drs_activation" else 0}
        if k in by:
            if by[k]["label"] != rec["label"]:
                raise ValueError(f"mixed state label {k}")
            if not np.array_equal(np.asarray(by[k]["state"]), np.asarray(rec["state"])):
                raise ValueError(f"nominal Stage-I state changed across candidates in group {k}")
        else:
            by[k] = rec
    return list(by.values())


def action_metrics(records, key, model, mu, sd, device, X_override=None):
    if not records:
        return {"rows": 0, "positive_rows": 0, "negative_rows": 0, "auc": None, "top1": None, "powered_groups": 0}
    X = np.stack([r[key] for r in records]) if X_override is None else X_override
    y = np.asarray([r["label"] for r in records], dtype=np.int64)
    sc = scores(model, mu, sd, X, device)
    groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        groups[tuple(r["group"])].append(i)
    powered = [ids for ids in groups.values() if any(y[i] == 1 for i in ids) and any(y[i] == 0 for i in ids)]
    top1 = float(np.mean([y[max(ids, key=lambda i: float(sc[i]))] == 1 for ids in powered])) if powered else None
    return {
        "rows": len(records),
        "positive_rows": int(y.sum()),
        "negative_rows": int(len(y) - y.sum()),
        "auc": auc(y, sc),
        "top1": top1,
        "powered_groups": len(powered),
    }


def state_metrics(records, model, mu, sd, device):
    if not records:
        return {"rows": 0, "drs_state_rows": 0, "dep_state_rows": 0, "auc": None}
    X = np.stack([r["state"] for r in records])
    y = np.asarray([r["label"] for r in records])
    sc = scores(model, mu, sd, X, device)
    return {
        "rows": len(records),
        "drs_state_rows": int(y.sum()),
        "dep_state_rows": int(len(y) - y.sum()),
        "auc": auc(y, sc),
    }


def split_role(records, role):
    bucket = 1 if "near" in role else 2
    return [r for r in records if int(r["group"][0]) == bucket]


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

    tr, etr = extract_stage_i_features(checkpoint=a.checkpoint, index_path=a.train_index, role_filter=None, v93_path=None, cache_dir=a.cache_dir / "train", device=a.device)
    dv: list[dict[str, Any]] = []
    ev: dict[str, Any] = {}
    for role in ("dev_near", "dev_contact"):
        rr, e = extract_stage_i_features(checkpoint=a.checkpoint, index_path=a.dev_index, role_filter=role, v93_path=a.v93_audit, cache_dir=a.cache_dir / role, device=a.device)
        dv.extend(rr)
        ev[role] = e
    ce: list[dict[str, Any]] = []
    for role in ("certificate_near", "certificate_contact"):
        rr, e = extract_stage_i_features(checkpoint=a.checkpoint, index_path=a.certificate_index, role_filter=role, v93_path=a.v93_audit, cache_dir=a.cache_dir / role, device=a.device)
        ce.extend(rr)
        ev[role] = e
    if not tr or not dv or not ce:
        raise RuntimeError("V48.102 empty Stage-I audit feature set")

    st_tr = state_records(tr)
    sup_tr = action_subset(tr, "drs_activation")
    res_tr = action_subset(tr, "deployability_gain")
    sm, smu, ssd = fit_binary(np.stack([r["state"] for r in st_tr]), np.asarray([r["label"] for r in st_tr]), a.device)
    um, umu, usd = fit_binary(np.stack([r["delta"] for r in sup_tr]), np.asarray([r["label"] for r in sup_tr]), a.device)
    rm, rmu, rsd = fit_binary(np.stack([r["context"] for r in res_tr]), np.asarray([r["label"] for r in res_tr]), a.device)
    sup_perm = permute_within_group(sup_tr, "delta")
    up, upmu, upsd = fit_binary(sup_perm, np.asarray([r["label"] for r in sup_tr]), a.device)
    res_perm = permute_within_group(res_tr, "context")
    rp, rpmu, rpsd = fit_binary(res_perm, np.asarray([r["label"] for r in res_tr]), a.device)

    cells: dict[str, Any] = {}
    for role in ROLES:
        src = dv if role.startswith("dev_") else ce
        rr = split_role(src, role)
        sr = state_records(rr)
        ur = action_subset(rr, "drs_activation")
        qr = action_subset(rr, "deployability_gain")
        smet = state_metrics(sr, sm, smu, ssd, a.device)
        utrue = action_metrics(ur, "delta", um, umu, usd, a.device)
        ushuf = action_metrics(ur, "delta", up, upmu, upsd, a.device, X_override=permute_within_group(ur, "delta") if ur else None)
        rtrue = action_metrics(qr, "context", rm, rmu, rsd, a.device)
        rshuf = action_metrics(qr, "context", rp, rpmu, rpsd, a.device, X_override=permute_within_group(qr, "context") if qr else None)
        for t, s in ((utrue, ushuf), (rtrue, rshuf)):
            t["auc_vs_shuffled"] = None if t["auc"] is None or s["auc"] is None else float(t["auc"] - s["auc"])
            t["top1_vs_shuffled"] = None if t["top1"] is None or s["top1"] is None else float(t["top1"] - s["top1"])
        cells[role] = {
            "state": smet,
            "support_true": utrue,
            "support_shuffled": ushuf,
            "reserve_true": rtrue,
            "reserve_shuffled": rshuf,
        }

    result = {
        "schema": "ocrap-v48.102-stage-i-action-information-transport-audit-v1",
        "engineering_version": ENGINEERING_VERSION,
        "algorithm_name": ALGORITHM_NAME,
        "valid": True,
        "variant": a.variant,
        "audit_only": True,
        "checkpoint": str(a.checkpoint.resolve()),
        "checkpoint_sha256": sha256(a.checkpoint),
        "planner_parameters_trained": 0,
        "stage_i_parameters_trained": 0,
        "root_decoder_parameters_trained": 0,
        "source_parameters_trained": 0,
        "relative_ranker_modified": False,
        "boundary_transport": False,
        "regime_conditioning": False,
        "teacher_metadata_input_to_model": False,
        "train_counts": {"state": len(st_tr), "support": len(sup_tr), "reserve": len(res_tr)},
        "events": {"train": etr, **ev},
        "cells": cells,
        "elapsed_seconds": float(time.perf_counter() - t0),
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    torch.save({
        "schema": "ocrap-v48.102-stage-i-probe-state-v1",
        "engineering_version": ENGINEERING_VERSION,
        "algorithm_name": ALGORITHM_NAME,
        "variant": a.variant,
        "checkpoint_sha256": sha256(a.checkpoint),
        "state_probe": sm.state_dict(),
        "support_probe": um.state_dict(),
        "reserve_probe": rm.state_dict(),
        "state_mu": smu,
        "state_sd": ssd,
        "support_mu": umu,
        "support_sd": usd,
        "reserve_mu": rmu,
        "reserve_sd": rsd,
    }, a.state_output)
    print(json.dumps({"valid": True, "variant": a.variant, "train_counts": result["train_counts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
