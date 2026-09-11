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
    executable_constraint_field_from_sample,
    validate_group_contract,
)
from ocrap.audits.recovery_set_constraint_flow import (
    ALGORITHM_NAME,
    ENGINEERING_VERSION,
    MATCHED_DIM,
    SCIENTIFIC_VERSION,
    SET_GEOMETRY_DIM,
    base_features,
    fit_set_flow_scaler,
    matched_features,
    recovery_set_integral_geometry,
    recovery_set_work_geometry,
    set_flow_diagnostics,
)

ROLES = ("dev_near", "dev_contact", "certificate_near", "certificate_contact")

# Deliberately excludes teacher m_star/root/future labels.  The scientific
# feature path reads only current observation, candidate prefix, and the fixed
# recovery option library.  Labels enter later through the historical indices.
RSCF_SAMPLE_KEYS: frozenset[str] = frozenset({
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
            "mean_common_valid_option_count": 0.0,
            "min_common_valid_option_count": 0,
            "set_work_nonzero_fraction": 0.0,
            "set_integral_nonzero_fraction": 0.0,
            "option_flow_diverse_fraction": 0.0,
            "reentry_set_available_fraction": 0.0,
            "max_set_work_conservation_error": 0.0,
            "max_option_permutation_invariance_error": 0.0,
        }
    n = len(diags)
    counts = [int(d.get("common_valid_option_count", 0)) for d in diags]
    return {
        "candidate_count": n,
        "mean_common_valid_option_count": float(np.mean(counts)),
        "min_common_valid_option_count": int(min(counts)),
        "set_work_nonzero_fraction": float(np.mean([bool(d.get("set_work_nonzero", False)) for d in diags])),
        "set_integral_nonzero_fraction": float(np.mean([bool(d.get("set_integral_nonzero", False)) for d in diags])),
        "option_flow_diverse_fraction": float(np.mean([bool(d.get("option_flow_diverse", False)) for d in diags])),
        "reentry_set_available_fraction": float(np.mean([bool(d.get("reentry_available_in_set", False)) for d in diags])),
        "max_set_work_conservation_error": float(max(float(d.get("set_work_conservation_error", 0.0)) for d in diags)),
        "max_option_permutation_invariance_error": float(max(float(d.get("option_permutation_invariance_error", 0.0)) for d in diags)),
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
        raise RuntimeError("V48.115 requires StructuredTokenEncoder")
    enc = model.encoder.eval()
    dev = bundle.device
    if len(enc.encoder.layers) != 2:
        raise RuntimeError("V48.115 requires historical two-layer Stage-I")

    cfg, feature_event = feature_only_dataset_cfg(bundle.cfg, cache_dir=str(cache_dir / "tensor"), workers=8)
    ds = OCRAPSampleDataset(paths, cfg)
    if ds.absolute_truth_contract_event.get("enabled") or ds.action_response_truth_event.get("enabled"):
        raise RuntimeError("V48.115 feature-only dataset unexpectedly attached truth sidecars")
    if [str(p.resolve()) for p in paths] != [str(p.resolve()) for p in ds.paths]:
        raise RuntimeError("V48.115 dataset path order differs from index")
    idx = {str(p.resolve()): i for i, p in enumerate(ds.paths)}

    # Independent raw-sample map used only by the deterministic executable
    # recovery constraint path.  Teacher root/margin arrays are not loaded.
    raw_sample = {str(p.resolve()): load_npz_selected(p, RSCF_SAMPLE_KEYS) for p in paths}

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
            set_integral = recovery_set_integral_geometry(fc, f0)
            set_work = recovery_set_work_geometry(fc, f0)
            diag = set_flow_diagnostics(fc, f0)
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
                "set_integral_geometry": set_integral,
                "set_work_geometry": set_work,
            })

    merged = _merge_pair_diags(pair_diags)
    event = {
        "records": len(records),
        "groups": len(groups),
        "raw_candidate_dim": RAW_CANDIDATE_DIM,
        "set_geometry_dim": SET_GEOMETRY_DIM,
        "matched_dim": MATCHED_DIM,
        "constraint_names": ["clearance", "stopping", "route", "reentry"],
        "constraint_semantics": {
            "clearance": "actuator_projected_recovery_cv_signed_safety_reserve",
            "stopping": "remaining_executable_recovery_path_to_safety_boundary_minus_braking_distance",
            "route": "recovery_route_corridor_signed_reserve",
            "reentry": "physical_contact_activated_persistent_suffix_signed_reserve",
        },
        "recovery_response": "selector_free_empirical_mean_of_same_option_full_horizon_constraint_flows",
        "integral_control": "eight_contiguous_full_horizon_bins_of_delta_h_and_delta_h_times_h0",
        "constraint_work": "eight_contiguous_full_horizon_bins_of_positive_reserve_work_and_negative_debt_repayment",
        "work_identity": "mean_option_reserve_work_plus_mean_option_debt_work_equals_mean_option_bin_delta_h",
        "option_aggregation": "uniform_empirical_mean_over_all_common_valid_recovery_options",
        "selector_free_feature_path": True,
        "same_option_inside_each_set_summand": True,
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
    si = np.stack([r["set_integral_geometry"] for r in records]).astype(np.float64)
    sw = np.stack([r["set_work_geometry"] for r in records]).astype(np.float64)
    y = np.asarray([r["label"] for r in records], dtype=np.int64)
    return u, si, sw, y

def _fit_axis(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    u, si, sw, y = _arrays(records, key)
    sc = fit_set_flow_scaler(u)
    pi = _perm_indices(records)
    feats = {
        "base": base_features(u, sc),
        "set_integral": matched_features(u, si, sc),
        "set_work": matched_features(u, sw, sc),
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
    spaces = ("base", "set_integral", "set_work")
    if not records:
        return {space: (_metric([], np.array([])), _metric([], np.array([]))) for space in spaces}
    u, si, sw, _ = _arrays(records, key)
    pi = _perm_indices(records)
    sc = fit["scaler"]
    m = fit["models"]
    feats = {
        "base": base_features(u, sc),
        "set_integral": matched_features(u, si, sc),
        "set_work": matched_features(u, sw, sc),
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
    cells = {k: {} for k in ("base", "set_integral", "set_work")}
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
        raise RuntimeError("V48.115 empty audit records")

    fam = _fit_family(tr)
    cells = _eval_family(dv, ce, fam)
    max_resid = max(
        v.normal_equation_residual
        for axis in ("support", "reserve")
        for v in fam[axis]["models"].values()
    )
    result = {
        "schema": "ocrap-v48.115-recovery-set-constraint-flow-audit-v1",
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
        "set_integral_cells": cells["set_integral"],
        "set_work_cells": cells["set_work"],
        "events": events,
        "train_counts": fam["counts"],
        "convex_closed_form_ridge": True,
        "strictly_convex_unique_solution": True,
        "iterative_optimizer_used": False,
        "ridge_lambda_rule": "1_over_axis_train_rows",
        "max_normal_equation_residual": max_resid,
        "score_family": "linear_on_selector_free_recovery_set_full_horizon_integral_or_signed_constraint_flow_features",
        "nominal_zero_score_by_construction": True,
        "constraint_names": ["clearance", "stopping", "route", "reentry"],
        "constraint_response": "selector_free_empirical_recovery_set_of_same_option_actuator_projected_constraint_flows",
        "integral_response_channels": ["bin_mean_delta_h", "bin_mean_delta_h_times_h0"],
        "constraint_work_channels": ["positive_reserve_work", "negative_debt_repayment_work"],
        "work_conservation_identity": "reserve_work_plus_debt_work_equals_bin_mean_delta_h",
        "option_aggregation": "uniform_empirical_mean_over_all_common_valid_recovery_options",
        "selector_free_feature_path": True,
        "same_option_inside_each_set_summand": True,
        "work_bins": 8,
        "set_geometry_dimension": SET_GEOMETRY_DIM,
        "matched_family_dimension": MATCHED_DIM,
        "capacity_matched_all_set_families": True,
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
            "schema": "ocrap-v48.115-recovery-set-constraint-flow-state-v1",
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
