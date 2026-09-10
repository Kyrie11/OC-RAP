#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ocrap.models.data import OCRAPSampleDataset
from ocrap.models.inference import load_model_bundle
from ocrap.models.encoders import StructuredTokenEncoder
from ocrap.audits.constraint_native_orientation import (
    DEPLOYABLE_MACROS,
    POSITIVE_GAIN,
    RAW_CANDIDATE_DIM,
    VALID_MODES,
    action_features,
    derive_candidate_semantics,
    feature_only_dataset_cfg,
    fit_closed_form_ridge,
    raw_agent_set,
    raw_candidate_pathway,
    ridge_scores,
)
from ocrap.audits.heterogeneous_constraint_normal_cone import (
    ALGORITHM_NAME,
    CONE_GEOMETRY_DIM,
    ENGINEERING_VERSION,
    MATCHED_DIM,
    SCIENTIFIC_VERSION,
    active_constraint_indices,
    base_features,
    cone_config_from_mapping,
    fit_cone_scaler,
    heterogeneous_constraint_paths,
    matched_features,
    selector_diagnostics,
)

AUTHORITATIVE_V111_COMPARISON_SHA256 = "ee2a3f13f2793dd8d0a4a1bdf73192a188d21bfac31c1549b3d6d0ae63cb8373"
AUTHORITATIVE_V111_PIPELINE_SHA256 = "c155ac8277b2fe690be030eaaf4031e75873e1e22d2521ce56d10aabebb56187"
AUTHORITATIVE_V111_BALANCED_SHA256 = "df5dd1a3a590774255f09e92de3e5d4a9762272fb871a3e64abfe6c3d8eb7e19"
AUTHORITATIVE_V111_PRECISION_SHA256 = "ec431e27086a2fe2f30f278a1c3fbfbafca5e4d62094a87dc838e9b52e627b66"
ROLES = ("dev_near", "dev_contact", "certificate_near", "certificate_contact")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise ValueError(f"invalid JSONL {path}:{line_no}: {exc}") from exc
    return rows


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
        key = (role, str(r["scene_id"]), int(r["time_index"]), int(r["candidate_index"]))
        if key in out:
            raise ValueError(f"duplicate V48.93 key {key}")
        out[key] = r
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
            candidates = [
                c for c in candidates
                if (role_filter, str(key[1]), int(key[2]), int(c["candidate"])) in v93_map
            ]
        if not candidates:
            continue
        safe_modes = {
            str(c["mediation_mode"])
            for c in candidates
            if c["safe_positive"] and c["mediation_mode"] in VALID_MODES
        }
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


def split_role(records: list[dict[str, Any]], role: str) -> list[dict[str, Any]]:
    bucket = 1 if "near" in role else 2
    return [r for r in records if int(r["group"][0]) == bucket]


def _cache_key(checkpoint: Path, index_path: Path, role_filter: str | None, v93_path: Path | None) -> str:
    payload = {
        "version": ENGINEERING_VERSION,
        "checkpoint": sha256(checkpoint),
        "index": sha256(index_path),
        "role": role_filter,
        "v93": sha256(v93_path) if v93_path and v93_path.is_file() else None,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _stack(items: list[dict[str, Any]], key: str) -> torch.Tensor:
    return torch.stack([it[key] for it in items])


def _merge_selector_diag(diags: list[dict[str, Any]]) -> dict[str, Any]:
    if not diags:
        return {
            "selector_switch_fraction": 0.0,
            "candidate_any_switch_fraction": 0.0,
            "nominal_active_type_counts": {},
            "candidate_active_type_counts": {},
        }
    total_slots = sum(int(d["slots"]) for d in diags)
    total_candidates = sum(int(d["candidates"]) for d in diags)
    switch_slots = sum(float(d["selector_switch_fraction"]) * int(d["slots"]) for d in diags)
    any_switch = sum(float(d["candidate_any_switch_fraction"]) * int(d["candidates"]) for d in diags)
    names = ("clearance", "stopping", "route", "reentry")
    nom = {n: 0 for n in names}
    cand = {n: 0 for n in names}
    for d in diags:
        for n in names:
            nom[n] += int(d["nominal_active_type_counts"].get(n, 0))
            cand[n] += int(d["candidate_active_type_counts"].get(n, 0))
    return {
        "selector_switch_fraction": float(switch_slots / max(total_slots, 1)),
        "candidate_any_switch_fraction": float(any_switch / max(total_candidates, 1)),
        "nominal_active_type_counts": nom,
        "candidate_active_type_counts": cand,
    }


def extract_records(
    *, checkpoint: Path, index_path: Path, role_filter: str | None, v93_path: Path | None,
    cache_dir: Path, device: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    key = _cache_key(checkpoint, index_path, role_filter, v93_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cp = cache_dir / f"{key}.pt"
    if cp.is_file():
        obj = torch.load(cp, map_location="cpu", weights_only=False)
        if obj.get("cache_key") == key:
            return obj["records"], obj["event"]

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

    bundle = load_model_bundle(checkpoint, {"training": {"device": device}})
    if bundle is None:
        raise RuntimeError(f"cannot load checkpoint {checkpoint}")
    model = bundle.model.eval()
    [p.requires_grad_(False) for p in model.parameters()]
    if not isinstance(model.encoder, StructuredTokenEncoder):
        raise RuntimeError("V48.112 requires StructuredTokenEncoder")
    enc = model.encoder.eval()
    dev = bundle.device
    if len(enc.encoder.layers) != 2:
        raise RuntimeError("V48.112 requires historical two-layer Stage-I")
    sample_rate = float(bundle.cfg.get("sample_rate_hz", 10.0) or 10.0)
    cone_cfg = cone_config_from_mapping(bundle.cfg)

    cfg, feature_event = feature_only_dataset_cfg(bundle.cfg, cache_dir=str(cache_dir / "tensor"), workers=8)
    ds = OCRAPSampleDataset(paths, cfg)
    if ds.absolute_truth_contract_event.get("enabled") or ds.action_response_truth_event.get("enabled"):
        raise RuntimeError("V48.112 feature-only dataset unexpectedly attached truth sidecars")
    if [str(p.resolve()) for p in paths] != [str(p.resolve()) for p in ds.paths]:
        raise RuntimeError("V48.112 dataset path order differs from index")
    idx = {str(p.resolve()): i for i, p in enumerate(ds.paths)}

    records: list[dict[str, Any]] = []
    agent_delta_max = 0.0
    mask_delta_max = 0
    selector_diags: list[dict[str, Any]] = []
    path_diags: list[dict[str, Any]] = []

    for g in groups:
        ordered = [g["nominal_path"]] + [c["path"] for c in g["candidates"]]
        if any(str(Path(p).resolve()) not in idx for p in ordered):
            continue
        items = [ds[idx[str(Path(p).resolve())]] for p in ordered]
        x = _stack(items, "x").to(dev)
        with torch.no_grad():
            raw = raw_candidate_pathway(x, enc.layout)
            agents, mask = raw_agent_set(x, enc.layout)
            agent_delta_max = max(agent_delta_max, float((agents[1:] - agents[0:1]).abs().max().item()))
            mask_delta_max = max(mask_delta_max, int(torch.count_nonzero(mask[1:] != mask[0:1]).item()))
            state, delta, reserve_context = action_features(raw)
            nominal = raw[0:1].cpu().numpy()
            cand = raw[1:].cpu().numpy()
            an = agents[0:1].expand(delta.shape[0], -1, -1).cpu().numpy()
            mn = mask[0:1].expand(delta.shape[0], -1).cpu().numpy()
            stn = state.cpu().numpy()
            dn = delta.cpu().numpy()
            qn = reserve_context.cpu().numpy()
        nominal = np.repeat(nominal, len(cand), axis=0)

        hc, mc, dc = heterogeneous_constraint_paths(
            cand, an, mn, sample_rate_hz=sample_rate, config=cone_cfg
        )
        h0, m0, d0 = heterogeneous_constraint_paths(
            nominal, an, mn, sample_rate_hz=sample_rate, config=cone_cfg
        )
        csel = active_constraint_indices(hc, mc)
        nsel = active_constraint_indices(h0, m0)
        sd = selector_diagnostics(nsel, csel)
        sd["slots"] = int(np.prod(csel.shape))
        sd["candidates"] = int(csel.shape[0])
        selector_diags.append(sd)
        path_diags.append({"candidate": dc, "nominal": d0})

        for j, c in enumerate(g["candidates"]):
            records.append({
                "group": tuple(g["key"]),
                "candidate": int(c["candidate"]),
                "group_mode": g["group_mode"],
                "safe_positive": bool(c["safe_positive"]),
                "teacher_harmful": bool(c["teacher_harmful"]),
                "mediation_mode": c["mediation_mode"],
                "raw_state": stn[j],
                "candidate_raw": cand[j],
                "nominal_raw": nominal[j],
                "support_u": dn[j],
                "reserve_u": qn[j],
                "constraint_candidate": hc[j],
                "constraint_nominal": h0[j],
                "constraint_mask_candidate": mc[j],
                "constraint_mask_nominal": m0[j],
                "candidate_cone_selector": csel[j],
                "nominal_cone_selector": nsel[j],
            })

    if agent_delta_max > 1.0e-6 or mask_delta_max != 0:
        raise RuntimeError(
            f"V48.112 agent set changed across candidate actions values={agent_delta_max} mask={mask_delta_max}"
        )

    merged_selector = _merge_selector_diag(selector_diags)
    event = {
        "records": len(records),
        "groups": len(groups),
        "raw_candidate_dim": RAW_CANDIDATE_DIM,
        "cone_geometry_dim": CONE_GEOMETRY_DIM,
        "matched_dim": MATCHED_DIM,
        "agent_set_candidate_delta_max_abs": agent_delta_max,
        "agent_mask_candidate_delta_count": mask_delta_max,
        "sample_rate_hz": sample_rate,
        "constraint_names": ["clearance", "stopping", "route", "reentry"],
        "constraint_semantics": {
            "clearance": "cv_circle_signed_safety_reserve",
            "stopping": "remaining_prefix_path_to_first_safety_boundary_minus_braking_distance",
            "route": "route_corridor_signed_reserve",
            "reentry": "contact_activated_signed_debt_then_persistent_suffix_reserve",
        },
        "normal_cone_control": "nominal_prefix_active_constraint_type",
        "normal_cone_treatment": "candidate_prefix_active_constraint_type",
        "selector_diagnostics": merged_selector,
        "constraint_config": {
            "d_safe0_m": cone_cfg.d_safe0_m,
            "safe_time_headway_s": cone_cfg.safe_time_headway_s,
            "distance_scale": cone_cfg.distance_scale,
            "stop_scale": cone_cfg.stop_scale,
            "route_scale": cone_cfg.route_scale,
            "route_dev_max_m": cone_cfg.route_dev_max_m,
            "a_min_mps2": cone_cfg.a_min_mps2,
            "default_available_distance_m": cone_cfg.default_available_distance_m,
        },
        "feature_only_dataset_contract": feature_event,
        "tensor_cache_event": ds.tensor_cache_event,
        "encoder_layer_count": 2,
    }
    torch.save({"cache_key": key, "records": records, "event": event}, cp)
    return records, event


def _perm_indices(records: list[dict[str, Any]]) -> np.ndarray:
    groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        groups[tuple(r["group"])].append(i)
    idx = np.arange(len(records))
    for ids in groups.values():
        ids = sorted(ids, key=lambda i: int(records[i]["candidate"]))
        idx[ids] = np.roll(np.asarray(ids, dtype=np.int64), 1)
    return idx


def _arrays(records: list[dict[str, Any]], key: str):
    u = np.stack([r[key] for r in records]).astype(np.float64)
    hc = np.stack([r["constraint_candidate"] for r in records]).astype(np.float64)
    h0 = np.stack([r["constraint_nominal"] for r in records]).astype(np.float64)
    csel = np.stack([r["candidate_cone_selector"] for r in records]).astype(np.int64)
    nsel = np.stack([r["nominal_cone_selector"] for r in records]).astype(np.int64)
    y = np.asarray([r["label"] for r in records], dtype=np.int64)
    return u, hc, h0, nsel, csel, y


def _fit_axis(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    u, hc, h0, nsel, csel, y = _arrays(records, key)
    sc = fit_cone_scaler(u)
    pi = _perm_indices(records)
    fb = base_features(u, sc)
    fn = matched_features(u, hc, h0, nsel, sc)
    fc = matched_features(u, hc, h0, csel, sc)
    models = {
        "base_true": fit_closed_form_ridge(fb, y),
        "base_shuffle": fit_closed_form_ridge(fb[pi], y),
        "nominal_cone_true": fit_closed_form_ridge(fn, y),
        "nominal_cone_shuffle": fit_closed_form_ridge(fn[pi], y),
        "candidate_cone_true": fit_closed_form_ridge(fc, y),
        "candidate_cone_shuffle": fit_closed_form_ridge(fc[pi], y),
    }
    return {"scaler": sc, "models": models, "count": len(records)}


def _fit_family(records: list[dict[str, Any]]) -> dict[str, Any]:
    su = action_subset(records, "drs_activation")
    re = action_subset(records, "deployability_gain")
    return {
        "support": _fit_axis(su, "support_u"),
        "reserve": _fit_axis(re, "reserve_u"),
        "counts": {"support": len(su), "reserve": len(re)},
    }


def _metric(records: list[dict[str, Any]], scores: np.ndarray) -> dict[str, Any]:
    if not records:
        return {"rows": 0, "positive_rows": 0, "negative_rows": 0, "auc": None, "top1": None, "powered_groups": 0}
    y = np.asarray([r["label"] for r in records], dtype=np.int64)
    sc = np.asarray(scores, dtype=np.float64)
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


def _eval_axis(records: list[dict[str, Any]], key: str, fit: dict[str, Any]) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    if not records:
        return {s: (_metric([], np.array([])), _metric([], np.array([]))) for s in ("base", "nominal_cone", "candidate_cone")}
    u, hc, h0, nsel, csel, _ = _arrays(records, key)
    pi = _perm_indices(records)
    sc = fit["scaler"]
    m = fit["models"]
    feats = {
        "base": base_features(u, sc),
        "nominal_cone": matched_features(u, hc, h0, nsel, sc),
        "candidate_cone": matched_features(u, hc, h0, csel, sc),
    }
    out: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for space, f in feats.items():
        t = _metric(records, ridge_scores(m[f"{space}_true"], f))
        s = _metric(records, ridge_scores(m[f"{space}_shuffle"], f[pi]))
        t["auc_vs_shuffled"] = None if t["auc"] is None or s["auc"] is None else float(t["auc"] - s["auc"])
        t["top1_vs_shuffled"] = None if t["top1"] is None or s["top1"] is None else float(t["top1"] - s["top1"])
        out[space] = (t, s)
    return out


def _eval_family(dev_records: list[dict[str, Any]], cert_records: list[dict[str, Any]], family: dict[str, Any]) -> dict[str, Any]:
    cells = {k: {} for k in ("base", "nominal_cone", "candidate_cone")}
    for role in ROLES:
        src = dev_records if role.startswith("dev_") else cert_records
        rr = split_role(src, role)
        su = action_subset(rr, "drs_activation")
        re = action_subset(rr, "deployability_gain")
        sm = _eval_axis(su, "support_u", family["support"])
        rm = _eval_axis(re, "reserve_u", family["reserve"])
        for space in cells:
            cells[space][role] = {
                "support_true": sm[space][0],
                "support_shuffled": sm[space][1],
                "reserve_true": rm[space][0],
                "reserve_shuffled": rm[space][1],
            }
    return cells


def _pack_axis(fit: dict[str, Any]) -> dict[str, Any]:
    sc = fit["scaler"]
    return {
        "scaler": {"u_scale": sc.u_scale},
        "models": {
            k: {
                "coef": v.coef,
                "ridge_lambda": v.ridge_lambda,
                "objective": v.objective,
                "normal_equation_residual": v.normal_equation_residual,
            }
            for k, v in fit["models"].items()
        },
        "count": fit["count"],
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
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--state-output", type=Path, required=True)
    a = ap.parse_args()
    t0 = time.perf_counter()

    tr, etr = extract_records(
        checkpoint=a.checkpoint, index_path=a.train_index, role_filter=None, v93_path=None,
        cache_dir=a.cache_dir / "train", device=a.device,
    )
    dv: list[dict[str, Any]] = []
    ce: list[dict[str, Any]] = []
    events = {"train": etr}
    for role in ("dev_near", "dev_contact"):
        r, e = extract_records(
            checkpoint=a.checkpoint, index_path=a.dev_index, role_filter=role, v93_path=a.v93_audit,
            cache_dir=a.cache_dir / role, device=a.device,
        )
        dv += r
        events[role] = e
    for role in ("certificate_near", "certificate_contact"):
        r, e = extract_records(
            checkpoint=a.checkpoint, index_path=a.certificate_index, role_filter=role, v93_path=a.v93_audit,
            cache_dir=a.cache_dir / role, device=a.device,
        )
        ce += r
        events[role] = e
    if not tr or not dv or not ce:
        raise RuntimeError("V48.112 empty audit records")

    fam = _fit_family(tr)
    cells = _eval_family(dv, ce, fam)
    max_resid = max(
        v.normal_equation_residual
        for axis in ("support", "reserve")
        for v in fam[axis]["models"].values()
    )
    result = {
        "schema": "ocrap-v48.112-heterogeneous-constraint-normal-cone-audit-v1",
        "engineering_version": ENGINEERING_VERSION,
        "scientific_version": SCIENTIFIC_VERSION,
        "run_instance_id": a.run_id,
        "algorithm_name": ALGORITHM_NAME,
        "valid": True,
        "variant": a.variant,
        "audit_only": True,
        "checkpoint": str(a.checkpoint.resolve()),
        "checkpoint_sha256": sha256(a.checkpoint),
        "base_cells": cells["base"],
        "nominal_cone_cells": cells["nominal_cone"],
        "candidate_cone_cells": cells["candidate_cone"],
        "events": events,
        "train_counts": fam["counts"],
        "convex_closed_form_ridge": True,
        "strictly_convex_unique_solution": True,
        "iterative_optimizer_used": False,
        "ridge_lambda_rule": "1_over_axis_train_rows",
        "max_normal_equation_residual": max_resid,
        "score_family": "linear_on_fixed_heterogeneous_constraint_normal_cone_response_features",
        "nominal_zero_score_by_construction": True,
        "constraint_names": ["clearance", "stopping", "route", "reentry"],
        "constraint_response": "candidate_minus_nominal_normalized_signed_constraint_path",
        "nominal_cone_selector": "minimum_nominal_signed_constraint_per_prefix_time",
        "candidate_cone_selector": "minimum_candidate_signed_constraint_per_prefix_time",
        "cone_geometry_dimension": CONE_GEOMETRY_DIM,
        "matched_family_dimension": MATCHED_DIM,
        "capacity_matched_nominal_vs_candidate_cone": True,
        "candidate_identity_shuffle": "whole_feature_row_cyclic_permutation_within_scene_time_group",
        "planner_parameters_trained": 0,
        "stage_i_parameters_trained": 0,
        "root_decoder_parameters_trained": 0,
        "source_parameters_trained": 0,
        "relative_ranker_modified": False,
        "regime_conditioning": False,
        "boundary_transport": False,
        "teacher_metadata_input_to_model": False,
        "test_roots_read": False,
        "posthoc_feature_selection": False,
        "elapsed_seconds": float(time.perf_counter() - t0),
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    torch.save(
        {
            "schema": "ocrap-v48.112-heterogeneous-constraint-normal-cone-state-v1",
            "engineering_version": ENGINEERING_VERSION,
            "scientific_version": SCIENTIFIC_VERSION,
            "run_instance_id": a.run_id,
            "algorithm_name": ALGORITHM_NAME,
            "variant": a.variant,
            "support": _pack_axis(fam["support"]),
            "reserve": _pack_axis(fam["reserve"]),
            "convex_closed_form_ridge": True,
            "strictly_convex_unique_solution": True,
            "iterative_optimizer_used": False,
            "checkpoint_sha256": sha256(a.checkpoint),
        },
        a.state_output,
    )
    print(json.dumps({
        "valid": True,
        "variant": a.variant,
        "max_normal_equation_residual": max_resid,
        "elapsed_seconds": result["elapsed_seconds"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
