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

from ocrap.data.serialization import load_npz_selected
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
    raw_candidate_pathway,
    ridge_scores,
)
from ocrap.audits.executable_constraint_jacobian import (
    best_option_index,
    executable_constraint_field_from_sample,
    validate_group_contract,
)
from ocrap.audits.common_option_constraint_work import (
    ALGORITHM_NAME,
    ENGINEERING_VERSION,
    MATCHED_DIM,
    SCIENTIFIC_VERSION,
    WORK_GEOMETRY_DIM,
    base_features,
    constraint_work_geometry,
    fit_work_scaler,
    integral_response_geometry,
    matched_features,
    pair_work_diagnostics,
)

ROLES = ("dev_near", "dev_contact", "certificate_near", "certificate_contact")

# Deliberately excludes teacher m_star/root/future labels.  The scientific
# feature path reads only current observation, candidate prefix, and the fixed
# recovery option library.  Labels enter later through the historical indices.
CCW_SAMPLE_KEYS: frozenset[str] = frozenset({
    "scene_id", "time_index", "candidate_index", "is_nominal",
    "agent_history", "agent_valid", "ego_state",
    "prefix_states", "prefix_controls", "prefix_param", "prefix_macro_id", "prefix_macro_name",
    "utility", "hard_violation", "harm_proxy", "feasible",
    "recovery_modes", "recovery_params", "option_valid",
})


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


def _merge_pair_diags(diags: list[dict[str, Any]]) -> dict[str, Any]:
    if not diags:
        return {
            "candidate_count": 0,
            "option_switch_fraction": 0.0,
            "candidate_selected_mode_counts": {},
            "nominal_selected_mode_counts": {},
            "candidate_selected_active_type_counts": {},
            "candidate_selected_reentry_active_fraction": 0.0,
            "field_reentry_available_fraction": 0.0,
            "reserve_work_nonzero_fraction": 0.0,
            "debt_work_nonzero_fraction": 0.0,
            "integral_response_nonzero_fraction": 0.0,
            "max_work_conservation_error": 0.0,
        }
    candidate_modes: dict[str, int] = defaultdict(int)
    nominal_modes: dict[str, int] = defaultdict(int)
    active: dict[str, int] = defaultdict(int)
    switched = 0
    reentry_selected = 0
    field_reentry = 0
    reserve_work_nonzero = 0
    debt_work_nonzero = 0
    integral_response_nonzero = 0
    conservation_errors: list[float] = []
    for d in diags:
        switched += int(bool(d["option_switched"]))
        candidate_modes[str(d["candidate_mode"])] += 1
        nominal_modes[str(d["nominal_mode"])] += 1
        reentry_selected += int(bool(d["candidate_selected_reentry_active"]))
        field_reentry += int(int(d.get("candidate_field_reentry_active_options", 0)) > 0)
        reserve_work_nonzero += int(bool(d.get("reserve_work_nonzero", False)))
        debt_work_nonzero += int(bool(d.get("debt_work_nonzero", False)))
        integral_response_nonzero += int(bool(d.get("integral_response_nonzero", False)))
        conservation_errors.append(float(d.get("work_conservation_error", 0.0)))
        for k, v in (d.get("candidate_selected_active_type_counts") or {}).items():
            active[str(k)] += int(v)
    n = len(diags)
    return {
        "candidate_count": n,
        "option_switch_fraction": float(switched / n),
        "candidate_selected_mode_counts": dict(sorted(candidate_modes.items())),
        "nominal_selected_mode_counts": dict(sorted(nominal_modes.items())),
        "candidate_selected_active_type_counts": dict(sorted(active.items())),
        "candidate_selected_reentry_active_fraction": float(reentry_selected / n),
        "field_reentry_available_fraction": float(field_reentry / n),
        "reserve_work_nonzero_fraction": float(reserve_work_nonzero / n),
        "debt_work_nonzero_fraction": float(debt_work_nonzero / n),
        "integral_response_nonzero_fraction": float(integral_response_nonzero / n),
        "max_work_conservation_error": float(max(conservation_errors) if conservation_errors else 0.0),
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
        raise RuntimeError("V48.114 requires StructuredTokenEncoder")
    enc = model.encoder.eval()
    dev = bundle.device
    if len(enc.encoder.layers) != 2:
        raise RuntimeError("V48.114 requires historical two-layer Stage-I")

    cfg, feature_event = feature_only_dataset_cfg(bundle.cfg, cache_dir=str(cache_dir / "tensor"), workers=8)
    ds = OCRAPSampleDataset(paths, cfg)
    if ds.absolute_truth_contract_event.get("enabled") or ds.action_response_truth_event.get("enabled"):
        raise RuntimeError("V48.114 feature-only dataset unexpectedly attached truth sidecars")
    if [str(p.resolve()) for p in paths] != [str(p.resolve()) for p in ds.paths]:
        raise RuntimeError("V48.114 dataset path order differs from index")
    idx = {str(p.resolve()): i for i, p in enumerate(ds.paths)}

    # Independent raw-sample map used only by the deterministic executable
    # recovery constraint path.  Teacher root/margin arrays are not loaded.
    raw_sample = {str(p.resolve()): load_npz_selected(p, CCW_SAMPLE_KEYS) for p in paths}

    records: list[dict[str, Any]] = []
    pair_diags: list[dict[str, Any]] = []
    first_field_diag: dict[str, Any] | None = None

    for g in groups:
        ordered = [g["nominal_path"]] + [c["path"] for c in g["candidates"]]
        if any(str(Path(p).resolve()) not in idx for p in ordered):
            continue
        items = [ds[idx[str(Path(p).resolve())]] for p in ordered]
        x = _stack(items, "x").to(dev)
        with torch.no_grad():
            raw = raw_candidate_pathway(x, enc.layout)
            state, delta, reserve_context = action_features(raw)
            stn = state.cpu().numpy()
            dn = delta.cpu().numpy()
            qn = reserve_context.cpu().numpy()

        nominal_path = str(Path(g["nominal_path"]).resolve())
        d0 = raw_sample[nominal_path]
        f0 = executable_constraint_field_from_sample(d0, bundle.cfg)
        if first_field_diag is None:
            first_field_diag = dict(f0.diagnostics)

        for j, c in enumerate(g["candidates"]):
            cp_path = str(Path(c["path"]).resolve())
            dc = raw_sample[cp_path]
            validate_group_contract(d0, dc)
            fc = executable_constraint_field_from_sample(dc, bundle.cfg, num_options=len(f0.option_valid))
            ln = best_option_index(f0)
            lc = best_option_index(fc)
            nominal_integral = integral_response_geometry(fc, f0, ln)
            candidate_integral = integral_response_geometry(fc, f0, lc)
            nominal_work = constraint_work_geometry(fc, f0, ln)
            candidate_work = constraint_work_geometry(fc, f0, lc)
            diag = pair_work_diagnostics(fc, f0)
            pair_diags.append(diag)
            records.append({
                "group": tuple(g["key"]),
                "candidate": int(c["candidate"]),
                "group_mode": g["group_mode"],
                "safe_positive": bool(c["safe_positive"]),
                "teacher_harmful": bool(c["teacher_harmful"]),
                "mediation_mode": c["mediation_mode"],
                "raw_state": stn[j],
                "support_u": dn[j],
                "reserve_u": qn[j],
                "nominal_integral_geometry": nominal_integral,
                "candidate_integral_geometry": candidate_integral,
                "nominal_work_geometry": nominal_work,
                "candidate_work_geometry": candidate_work,
                "candidate_option": int(lc),
                "nominal_option": int(ln),
                "candidate_mode": diag["candidate_mode"],
                "nominal_mode": diag["nominal_mode"],
                "candidate_selected_reentry_active": bool(diag["candidate_selected_reentry_active"]),
            })

    merged = _merge_pair_diags(pair_diags)
    event = {
        "records": len(records),
        "groups": len(groups),
        "raw_candidate_dim": RAW_CANDIDATE_DIM,
        "work_geometry_dim": WORK_GEOMETRY_DIM,
        "matched_dim": MATCHED_DIM,
        "constraint_names": ["clearance", "stopping", "route", "reentry"],
        "constraint_semantics": {
            "clearance": "actuator_projected_recovery_cv_signed_safety_reserve",
            "stopping": "remaining_executable_recovery_path_to_safety_boundary_minus_braking_distance",
            "route": "recovery_route_corridor_signed_reserve",
            "reentry": "physical_contact_activated_persistent_suffix_signed_reserve",
        },
        "recovery_response": "same_recovery_option_full_horizon_constraint_work",
        "integral_control": "eight_contiguous_full_horizon_bins_of_delta_h_and_delta_h_times_h0",
        "constraint_work": "eight_contiguous_full_horizon_bins_of_positive_reserve_work_and_negative_debt_repayment",
        "work_identity": "reserve_work_plus_debt_work_equals_bin_mean_candidate_minus_nominal_signed_response",
        "option_control": "nominal_prefix_maximin_executable_recovery_option",
        "option_treatment": "candidate_prefix_maximin_executable_recovery_option",
        "option_selection_score": "max_over_valid_options_of_min_over_full_recovery_horizon_active_signed_constraints",
        "actuator_projection": True,
        "pair_diagnostics": merged,
        "field_contract_example": first_field_diag or {},
        "feature_only_dataset_contract": feature_event,
        "tensor_cache_event": ds.tensor_cache_event,
        "encoder_layer_count": 2,
        "teacher_npz_fields_loaded_into_feature_path": False,
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
    ni = np.stack([r["nominal_integral_geometry"] for r in records]).astype(np.float64)
    ci = np.stack([r["candidate_integral_geometry"] for r in records]).astype(np.float64)
    nw = np.stack([r["nominal_work_geometry"] for r in records]).astype(np.float64)
    cw = np.stack([r["candidate_work_geometry"] for r in records]).astype(np.float64)
    y = np.asarray([r["label"] for r in records], dtype=np.int64)
    return u, ni, ci, nw, cw, y

def _fit_axis(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    u, ni, ci, nw, cw, y = _arrays(records, key)
    sc = fit_work_scaler(u)
    pi = _perm_indices(records)
    feats = {
        "base": base_features(u, sc),
        "nominal_integral": matched_features(u, ni, sc),
        "candidate_integral": matched_features(u, ci, sc),
        "nominal_work": matched_features(u, nw, sc),
        "candidate_work": matched_features(u, cw, sc),
    }
    models: dict[str, Any] = {}
    for space, feat in feats.items():
        models[f"{space}_true"] = fit_closed_form_ridge(feat, y)
        models[f"{space}_shuffle"] = fit_closed_form_ridge(feat[pi], y)
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
    spaces = ("base", "nominal_integral", "candidate_integral", "nominal_work", "candidate_work")
    if not records:
        return {space: (_metric([], np.array([])), _metric([], np.array([]))) for space in spaces}
    u, ni, ci, nw, cw, _ = _arrays(records, key)
    pi = _perm_indices(records)
    sc = fit["scaler"]
    m = fit["models"]
    feats = {
        "base": base_features(u, sc),
        "nominal_integral": matched_features(u, ni, sc),
        "candidate_integral": matched_features(u, ci, sc),
        "nominal_work": matched_features(u, nw, sc),
        "candidate_work": matched_features(u, cw, sc),
    }
    out: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for space, f in feats.items():
        t = _metric(records, ridge_scores(m[f"{space}_true"], f))
        sh = _metric(records, ridge_scores(m[f"{space}_shuffle"], f[pi]))
        t["auc_vs_shuffled"] = None if t["auc"] is None or sh["auc"] is None else float(t["auc"] - sh["auc"])
        t["top1_vs_shuffled"] = None if t["top1"] is None or sh["top1"] is None else float(t["top1"] - sh["top1"])
        out[space] = (t, sh)
    return out

def _eval_family(dev_records: list[dict[str, Any]], cert_records: list[dict[str, Any]], family: dict[str, Any]) -> dict[str, Any]:
    cells = {k: {} for k in ("base", "nominal_integral", "candidate_integral", "nominal_work", "candidate_work")}
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
        raise RuntimeError("V48.114 empty audit records")

    fam = _fit_family(tr)
    cells = _eval_family(dv, ce, fam)
    max_resid = max(
        v.normal_equation_residual
        for axis in ("support", "reserve")
        for v in fam[axis]["models"].values()
    )
    result = {
        "schema": "ocrap-v48.114-common-option-constraint-work-audit-v1",
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
        "nominal_integral_cells": cells["nominal_integral"],
        "candidate_integral_cells": cells["candidate_integral"],
        "nominal_work_cells": cells["nominal_work"],
        "candidate_work_cells": cells["candidate_work"],
        "events": events,
        "train_counts": fam["counts"],
        "convex_closed_form_ridge": True,
        "strictly_convex_unique_solution": True,
        "iterative_optimizer_used": False,
        "ridge_lambda_rule": "1_over_axis_train_rows",
        "max_normal_equation_residual": max_resid,
        "score_family": "linear_on_fixed_same_option_full_horizon_integral_or_signed_constraint_work_features",
        "nominal_zero_score_by_construction": True,
        "constraint_names": ["clearance", "stopping", "route", "reentry"],
        "constraint_response": "same_option_actuator_projected_full_horizon_signed_constraint_work",
        "integral_response_channels": ["bin_mean_delta_h", "bin_mean_delta_h_times_h0"],
        "constraint_work_channels": ["positive_reserve_work", "negative_debt_repayment_work"],
        "work_conservation_identity": "reserve_work_plus_debt_work_equals_bin_mean_delta_h",
        "nominal_option_selector": "nominal_maximin_over_full_executable_recovery_constraint_path",
        "candidate_option_selector": "candidate_maximin_over_full_executable_recovery_constraint_path",
        "work_bins": 8,
        "work_geometry_dimension": WORK_GEOMETRY_DIM,
        "matched_family_dimension": MATCHED_DIM,
        "capacity_matched_all_work_families": True,
        "candidate_identity_shuffle": "whole_feature_row_cyclic_permutation_within_scene_time_group",
        "actuator_projection": True,
        "teacher_npz_fields_loaded_into_feature_path": False,
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
            "schema": "ocrap-v48.114-common-option-constraint-work-state-v1",
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
