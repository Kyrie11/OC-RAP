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
from ocrap.models.data import OCRAPSampleDataset, option_features_from_sample
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
from ocrap.audits.recovery_set_constraint_flow import set_flow_diagnostics
from ocrap.audits.weak_root_recovery_set_flow import align_model_option_measure_to_physical_library
from ocrap.audits.tail_boundary_crossing_flow import (
    nominal_tail_boundary_measure,
    boundary_measure_diagnostics,
)
from ocrap.audits.signed_viability_rank_state import (
    ALGORITHM_NAME,
    ENGINEERING_VERSION,
    MATCHED_DIM,
    SCIENTIFIC_VERSION,
    STATE_GEOMETRY_DIM,
    STATE_MODE_NAMES,
    base_features,
    fit_set_flow_scaler,
    matched_features,
    exposed_signed_state_geometry,
    full_signed_state_geometry,
    option_permutation_invariance_error,
)

ROLES = ("dev_near", "dev_contact", "certificate_near", "certificate_contact")

# Deliberately excludes teacher m_star/root/future labels.  The scientific
# feature path reads only current observation, candidate prefix, and the fixed
# recovery option library.  Labels enter later through the historical indices.
SVRT_SAMPLE_KEYS: frozenset[str] = frozenset({
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
            "exposed_signed_state_nonzero_fraction": 0.0,
            "full_signed_state_nonzero_fraction": 0.0,
            "option_flow_diverse_fraction": 0.0,
            "reentry_set_available_fraction": 0.0,
            "max_option_permutation_invariance_error": 0.0,
            "mean_exposed_prefix_coupling_energy": 0.0,
            "mean_exposed_suffix_coupling_energy": 0.0,
            "mean_full_prefix_coupling_energy": 0.0,
            "mean_full_suffix_coupling_energy": 0.0,
            "mean_exposed_prefix_signed_state_energy": 0.0,
            "mean_exposed_suffix_signed_state_energy": 0.0,
            "mean_full_prefix_signed_state_energy": 0.0,
            "mean_full_suffix_signed_state_energy": 0.0,
            "mean_exposed_prefix_nonzero_fraction": 0.0,
            "mean_exposed_suffix_nonzero_fraction": 0.0,
            "mean_full_prefix_nonzero_fraction": 0.0,
            "mean_full_suffix_nonzero_fraction": 0.0,
            "mean_exposed_prefix_abs_nominal_state": 0.0,
            "mean_exposed_suffix_abs_nominal_state": 0.0,
            "mean_full_prefix_abs_nominal_state": 0.0,
            "mean_full_suffix_abs_nominal_state": 0.0,
            "mean_exposed_prefix_two_sided_state_fraction": 0.0,
            "mean_exposed_suffix_two_sided_state_fraction": 0.0,
            "mean_full_prefix_two_sided_state_fraction": 0.0,
            "mean_full_suffix_two_sided_state_fraction": 0.0,
            "mean_exposed_prefix_zero_crossing_fraction": 0.0,
            "mean_exposed_suffix_zero_crossing_fraction": 0.0,
            "mean_full_prefix_zero_crossing_fraction": 0.0,
            "mean_full_suffix_zero_crossing_fraction": 0.0,
            "mean_exposed_prefix_rank_inversion_fraction": 0.0,
            "mean_exposed_suffix_rank_inversion_fraction": 0.0,
            "mean_full_prefix_rank_inversion_fraction": 0.0,
            "mean_full_suffix_rank_inversion_fraction": 0.0,
            "mean_full_eligible_option_count": 0.0,
            "mean_full_prefix_coverage_fraction": 0.0,
            "mean_full_suffix_coverage_fraction": 0.0,
        }
    counts = [int(d.get("common_valid_option_count", 0)) for d in diags]
    mean = lambda key: float(np.mean([float(d.get(key, 0.0)) for d in diags]))
    return {
        "candidate_count": len(diags),
        "mean_common_valid_option_count": float(np.mean(counts)),
        "min_common_valid_option_count": int(min(counts)),
        "exposed_signed_state_nonzero_fraction": float(np.mean([bool(d.get("exposed_signed_state_nonzero", False)) for d in diags])),
        "full_signed_state_nonzero_fraction": float(np.mean([bool(d.get("full_signed_state_nonzero", False)) for d in diags])),
        "option_flow_diverse_fraction": float(np.mean([bool(d.get("option_flow_diverse", False)) for d in diags])),
        "reentry_set_available_fraction": float(np.mean([bool(d.get("reentry_available_in_set", False)) for d in diags])),
        "max_option_permutation_invariance_error": float(max(float(d.get("option_permutation_invariance_error", 0.0)) for d in diags)),
        "mean_exposed_prefix_coupling_energy": mean("exposed_prefix_coupling_energy"),
        "mean_exposed_suffix_coupling_energy": mean("exposed_suffix_coupling_energy"),
        "mean_full_prefix_coupling_energy": mean("full_prefix_coupling_energy"),
        "mean_full_suffix_coupling_energy": mean("full_suffix_coupling_energy"),
        "mean_exposed_prefix_signed_state_energy": mean("exposed_prefix_signed_state_energy"),
        "mean_exposed_suffix_signed_state_energy": mean("exposed_suffix_signed_state_energy"),
        "mean_full_prefix_signed_state_energy": mean("full_prefix_signed_state_energy"),
        "mean_full_suffix_signed_state_energy": mean("full_suffix_signed_state_energy"),
        "mean_exposed_prefix_nonzero_fraction": mean("exposed_prefix_nonzero_fraction"),
        "mean_exposed_suffix_nonzero_fraction": mean("exposed_suffix_nonzero_fraction"),
        "mean_full_prefix_nonzero_fraction": mean("full_prefix_nonzero_fraction"),
        "mean_full_suffix_nonzero_fraction": mean("full_suffix_nonzero_fraction"),
        "mean_exposed_prefix_abs_nominal_state": mean("exposed_prefix_abs_nominal_state"),
        "mean_exposed_suffix_abs_nominal_state": mean("exposed_suffix_abs_nominal_state"),
        "mean_full_prefix_abs_nominal_state": mean("full_prefix_abs_nominal_state"),
        "mean_full_suffix_abs_nominal_state": mean("full_suffix_abs_nominal_state"),
        "mean_exposed_prefix_two_sided_state_fraction": mean("exposed_prefix_two_sided_state_fraction"),
        "mean_exposed_suffix_two_sided_state_fraction": mean("exposed_suffix_two_sided_state_fraction"),
        "mean_full_prefix_two_sided_state_fraction": mean("full_prefix_two_sided_state_fraction"),
        "mean_full_suffix_two_sided_state_fraction": mean("full_suffix_two_sided_state_fraction"),
        "mean_exposed_prefix_zero_crossing_fraction": mean("exposed_prefix_zero_crossing_fraction"),
        "mean_exposed_suffix_zero_crossing_fraction": mean("exposed_suffix_zero_crossing_fraction"),
        "mean_full_prefix_zero_crossing_fraction": mean("full_prefix_zero_crossing_fraction"),
        "mean_full_suffix_zero_crossing_fraction": mean("full_suffix_zero_crossing_fraction"),
        "mean_exposed_prefix_rank_inversion_fraction": mean("exposed_prefix_rank_inversion_fraction"),
        "mean_exposed_suffix_rank_inversion_fraction": mean("exposed_suffix_rank_inversion_fraction"),
        "mean_full_prefix_rank_inversion_fraction": mean("full_prefix_rank_inversion_fraction"),
        "mean_full_suffix_rank_inversion_fraction": mean("full_suffix_rank_inversion_fraction"),
        "mean_full_eligible_option_count": mean("full_eligible_option_count"),
        "mean_full_prefix_coverage_fraction": mean("full_prefix_coverage_fraction"),
        "mean_full_suffix_coverage_fraction": mean("full_suffix_coverage_fraction"),
    }

def _merge_boundary_measure_diags(diags: list[dict[str, Any]]) -> dict[str, Any]:
    if not diags:
        return {
            "group_count": 0,
            "mean_positive_option_count": 0.0,
            "min_positive_option_count": 0,
            "mean_effective_option_count": 0.0,
            "min_effective_option_count": 0.0,
            "mean_positive_root_count": 0.0,
            "mean_witness_count_per_exposed_root": 0.0,
            "mean_two_sided_root_mass": 0.0,
            "mean_cotangent_positive_option_count": 0.0,
            "max_option_weight_sum_error": 0.0,
            "max_root_exposure_mass_error": 0.0,
            "max_model_padding_count": 0,
            "all_model_padding_invalid": True,
            "all_model_physical_valid_prefix_match": True,
            "max_padded_tail_mass": 0.0,
            "max_physical_tail_mass_error": 0.0,
        }
    return {
        "group_count": len(diags),
        "mean_positive_option_count": float(np.mean([d["boundary_positive_option_count"] for d in diags])),
        "min_positive_option_count": int(min(d["boundary_positive_option_count"] for d in diags)),
        "mean_effective_option_count": float(np.mean([d["boundary_effective_option_count"] for d in diags])),
        "min_effective_option_count": float(min(d["boundary_effective_option_count"] for d in diags)),
        "mean_positive_root_count": float(np.mean([d["boundary_positive_root_count"] for d in diags])),
        "mean_witness_count_per_exposed_root": float(np.mean([d["boundary_mean_witness_count_per_exposed_root"] for d in diags])),
        "mean_two_sided_root_mass": float(np.mean([d["boundary_two_sided_root_mass"] for d in diags])),
        "mean_cotangent_positive_option_count": float(np.mean([d["cotangent_positive_option_count"] for d in diags])),
        "max_option_weight_sum_error": float(max(abs(float(d["boundary_option_weight_sum"]) - 1.0) for d in diags)),
        "max_root_exposure_mass_error": float(max(abs(float(d["boundary_root_exposure_mass"]) - 1.0) for d in diags)),
        "max_model_padding_count": int(max(int(d.get("model_padding_count", 0)) for d in diags)),
        "all_model_padding_invalid": bool(all(bool(d.get("model_padding_all_invalid", False)) for d in diags)),
        "all_model_physical_valid_prefix_match": bool(all(bool(d.get("model_physical_valid_prefix_match", False)) for d in diags)),
        "max_padded_tail_mass": float(max(float(d.get("padded_tail_mass", 1.0)) for d in diags)),
        "max_physical_tail_mass_error": float(max(abs(float(d.get("physical_tail_mass", 0.0)) - 1.0) for d in diags)),
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
    for pth in needed:
        q = str(pth.resolve())
        if q not in seen:
            seen.add(q)
            paths.append(pth)

    bundle = load_model_bundle(checkpoint, {"training": {"device": device}})
    if bundle is None:
        raise RuntimeError(f"cannot load checkpoint {checkpoint}")
    model = bundle.model.eval()
    [par.requires_grad_(False) for par in model.parameters()]
    if not isinstance(model.encoder, StructuredTokenEncoder):
        raise RuntimeError("V48.122 requires StructuredTokenEncoder")
    enc = model.encoder.eval()
    dev = bundle.device
    if len(enc.encoder.layers) != 2:
        raise RuntimeError("V48.122 requires historical two-layer Stage-I")

    ocfg = bundle.cfg.get("ocmero", {}) if isinstance(bundle.cfg.get("ocmero", {}), dict) else {}
    ablation = bundle.cfg.get("ablation", {}) if isinstance(bundle.cfg.get("ablation", {}), dict) else {}
    if bool(ablation.get("without_lower_tail", False)) or not bool(ocfg.get("use_lcvar", True)):
        raise RuntimeError("V48.122 requires native lower-tail OC-MERO enabled")
    if bool(ablation.get("without_observation_kernel", False)) or not bool(ocfg.get("use_obs_kernel", True)):
        raise RuntimeError("V48.122 requires native observation-compatibility kernel enabled")
    alpha = float(ocfg.get("alpha", 0.2))
    beta = float(ocfg.get("beta", 0.2))
    top_m = int(ocfg.get("top_m", 8))

    cfg, feature_event = feature_only_dataset_cfg(bundle.cfg, cache_dir=str(cache_dir / "tensor"), workers=8)
    ds = OCRAPSampleDataset(paths, cfg)
    if ds.absolute_truth_contract_event.get("enabled") or ds.action_response_truth_event.get("enabled"):
        raise RuntimeError("V48.122 feature-only dataset unexpectedly attached truth sidecars")
    if [str(pth.resolve()) for pth in paths] != [str(pth.resolve()) for pth in ds.paths]:
        raise RuntimeError("V48.122 dataset path order differs from index")
    idx = {str(pth.resolve()): i for i, pth in enumerate(ds.paths)}

    # Deterministic physical path deliberately excludes teacher m_star/root/future
    # fields.  Weak-root weights come from the *frozen model's nominal prediction*,
    # never from teacher root_probs/m_star/c_star arrays or held-out labels.
    raw_sample = {str(pth.resolve()): load_npz_selected(pth, SVRT_SAMPLE_KEYS) for pth in paths}

    records: list[dict[str, Any]] = []
    pair_diags: list[dict[str, Any]] = []
    boundary_measure_diags: list[dict[str, Any]] = []
    first_field_diag: dict[str, Any] | None = None

    for g in groups:
        ordered = [g["nominal_path"]] + [c["path"] for c in g["candidates"]]
        if any(str(Path(pth).resolve()) not in idx for pth in ordered):
            continue
        items = [ds[idx[str(Path(pth).resolve())]] for pth in ordered]
        x = _stack(items, "x").to(dev)
        with torch.no_grad():
            raw = raw_candidate_pathway(x, enc.layout)
            state, delta, reserve_context = action_features(raw)
            stn = state.cpu().numpy(); dn = delta.cpu().numpy(); qn = reserve_context.cpu().numpy()

            nominal_item = items[0]
            x0 = nominal_item["x"].unsqueeze(0).to(dev)
            option_features0 = nominal_item["option_features"].unsqueeze(0).to(dev)
            root_valid0 = nominal_item["root_valid"].unsqueeze(0).to(dev)
            option_valid0 = nominal_item["option_valid"].unsqueeze(0).to(dev)
            native = model(
                x0, option_features0, root_valid=root_valid0,
                option_valid=option_valid0, witness_only=True,
            )
        if "margins" not in native or "root_logits" not in native or "c_star" not in native:
            raise RuntimeError("V48.122 frozen nominal model did not expose native OC-MERO fields")

        nominal_path = str(Path(g["nominal_path"]).resolve())
        d0 = raw_sample[nominal_path]
        f0 = executable_constraint_field_from_sample(d0, bundle.cfg)
        if first_field_diag is None:
            first_field_diag = dict(f0.diagnostics)
        ov_np = option_valid0[0].detach().cpu().numpy().astype(bool)
        model_option_features = option_features0[0].detach().cpu().numpy()
        raw_option_features = option_features_from_sample(dict(d0))
        physical_L = int(len(f0.option_valid))
        if raw_option_features.shape[0] != physical_L:
            raise RuntimeError("V48.122 raw option-feature/physical-library count mismatch")
        if model_option_features.shape[0] < physical_L:
            raise RuntimeError("V48.122 model option geometry shrinks physical recovery library")
        if not np.allclose(model_option_features[:physical_L], raw_option_features, rtol=0.0, atol=1.0e-7, equal_nan=True):
            raise RuntimeError("V48.122 model/physical recovery-option semantic ordering mismatch")
        if model_option_features.shape[0] > physical_L and not np.allclose(
            model_option_features[physical_L:], 0.0, rtol=0.0, atol=1.0e-12
        ):
            raise RuntimeError("V48.122 padded model option features are not structural zeros")

        margins_np = native["margins"][0].detach().cpu().numpy()
        root_logits_np = native["root_logits"][0].detach().cpu().numpy()
        compat_np = native["c_star"][0].detach().cpu().numpy()
        rv_np = root_valid0[0].detach().cpu().numpy().astype(bool)

        bm = nominal_tail_boundary_measure(
            margins_np, root_logits_np, compat_np,
            root_valid=rv_np, option_valid=ov_np, alpha=alpha, beta=beta, top_m=top_m,
        )
        physical_boundary_weights, boundary_align = align_model_option_measure_to_physical_library(
            ov_np, f0.option_valid, bm.option_weights
        )
        exposed_option_mask = np.asarray(physical_boundary_weights > 1.0e-12, dtype=bool)
        if not exposed_option_mask.any():
            raise RuntimeError("V48.122 weak-root boundary support is empty")
        bdiag = boundary_measure_diagnostics(bm)
        bdiag.update(boundary_align)
        boundary_measure_diags.append(bdiag)

        for j, c in enumerate(g["candidates"]):
            cp_path = str(Path(c["path"]).resolve())
            dc = raw_sample[cp_path]
            validate_group_contract(d0, dc)
            fc = executable_constraint_field_from_sample(dc, bundle.cfg, num_options=len(f0.option_valid))
            ee, eed = exposed_signed_state_geometry(fc, f0, exposed_option_mask)
            fe, fed = full_signed_state_geometry(fc, f0)
            physical_diag = set_flow_diagnostics(fc, f0)
            diag = dict(physical_diag)
            diag.update({
                "exposed_signed_state_nonzero": bool(np.any(np.abs(ee) > 1.0e-12)),
                "full_signed_state_nonzero": bool(np.any(np.abs(fe) > 1.0e-12)),
                "option_permutation_invariance_error": option_permutation_invariance_error(
                    fc, f0, exposed_option_mask
                ),
                "exposed_prefix_coupling_energy": eed.mean_prefix_coupling_energy,
                "exposed_suffix_coupling_energy": eed.mean_suffix_coupling_energy,
                "full_prefix_coupling_energy": fed.mean_prefix_coupling_energy,
                "full_suffix_coupling_energy": fed.mean_suffix_coupling_energy,
                "exposed_prefix_signed_state_energy": eed.mean_prefix_signed_state_energy,
                "exposed_suffix_signed_state_energy": eed.mean_suffix_signed_state_energy,
                "full_prefix_signed_state_energy": fed.mean_prefix_signed_state_energy,
                "full_suffix_signed_state_energy": fed.mean_suffix_signed_state_energy,
                "exposed_prefix_nonzero_fraction": eed.prefix_nonzero_fraction,
                "exposed_suffix_nonzero_fraction": eed.suffix_nonzero_fraction,
                "full_prefix_nonzero_fraction": fed.prefix_nonzero_fraction,
                "full_suffix_nonzero_fraction": fed.suffix_nonzero_fraction,
                "exposed_prefix_abs_nominal_state": eed.mean_prefix_abs_nominal_state,
                "exposed_suffix_abs_nominal_state": eed.mean_suffix_abs_nominal_state,
                "full_prefix_abs_nominal_state": fed.mean_prefix_abs_nominal_state,
                "full_suffix_abs_nominal_state": fed.mean_suffix_abs_nominal_state,
                "exposed_prefix_two_sided_state_fraction": eed.mean_prefix_two_sided_state_fraction,
                "exposed_suffix_two_sided_state_fraction": eed.mean_suffix_two_sided_state_fraction,
                "full_prefix_two_sided_state_fraction": fed.mean_prefix_two_sided_state_fraction,
                "full_suffix_two_sided_state_fraction": fed.mean_suffix_two_sided_state_fraction,
                "exposed_prefix_zero_crossing_fraction": eed.mean_prefix_zero_crossing_fraction,
                "exposed_suffix_zero_crossing_fraction": eed.mean_suffix_zero_crossing_fraction,
                "full_prefix_zero_crossing_fraction": fed.mean_prefix_zero_crossing_fraction,
                "full_suffix_zero_crossing_fraction": fed.mean_suffix_zero_crossing_fraction,
                "exposed_prefix_rank_inversion_fraction": eed.mean_prefix_rank_inversion_fraction,
                "exposed_suffix_rank_inversion_fraction": eed.mean_suffix_rank_inversion_fraction,
                "full_prefix_rank_inversion_fraction": fed.mean_prefix_rank_inversion_fraction,
                "full_suffix_rank_inversion_fraction": fed.mean_suffix_rank_inversion_fraction,
                "full_eligible_option_count": fed.mean_eligible_option_count,
                "full_prefix_coverage_fraction": fed.prefix_coverage_fraction,
                "full_suffix_coverage_fraction": fed.suffix_coverage_fraction,
            })
            pair_diags.append(diag)
            records.append({
                "group": tuple(g["key"]), "candidate": int(c["candidate"]),
                "group_mode": g["group_mode"], "safe_positive": bool(c["safe_positive"]),
                "teacher_harmful": bool(c["teacher_harmful"]), "mediation_mode": c["mediation_mode"],
                "raw_state": stn[j], "support_u": dn[j], "reserve_u": qn[j],
                "exposed_signed_state_geometry": ee,
                "full_signed_state_geometry": fe,
            })

    merged = _merge_pair_diags(pair_diags)
    boundary_merged = _merge_boundary_measure_diags(boundary_measure_diags)
    event = {
        "records": len(records), "groups": len(groups), "raw_candidate_dim": RAW_CANDIDATE_DIM,
        "state_geometry_dim": STATE_GEOMETRY_DIM, "matched_dim": MATCHED_DIM,
        "constraint_names": ["clearance", "stopping", "route", "reentry"],
        "constraint_semantics": {
            "clearance": "actuator_projected_recovery_cv_signed_safety_reserve",
            "stopping": "remaining_executable_recovery_path_to_safety_boundary_minus_braking_distance",
            "route": "recovery_route_corridor_signed_reserve",
            "reentry": "physical_contact_activated_persistent_suffix_signed_reserve",
        },
        "recovery_response": "same_option_signed_nominal_rank_state_prefix_and_suffix_viability_transport",
        "primary_option_set": "all_common_valid_recovery_options",
        "control_option_set": "support_of_frozen_weak_root_zero_boundary_witness_measure",
        "boundary_support_measure_source": "v48_117_outer_ocmero_weak_anchor_exposure_zero_margin_witness_support_only",
        "boundary_support_weights_used_in_signed_state": False,
        "state_channels": ["joint_prefix_signed_viability_rank_state_transport", "joint_suffix_persistent_reentry_signed_rank_state_transport"],
        "state_mode_names": [str(x) for x in STATE_MODE_NAMES],
        "signed_state_basis": "fixed_global_rank_signed_state_and_rank_x_signed_state_modes",
        "rank_coordinate": "candidate_independent_nominal_same_option_viability_midranks",
        "candidate_rank_sort_used_for_coordinate": False,
        "zero_boundary_threshold": 0.0,
        "pre_readout_fixed_option_selector": False,
        "active_option_identity_exported": False,
        "option_identity_exported": False,
        "actuator_projection": True,
        "pair_diagnostics": merged,
        "boundary_support_diagnostics": boundary_merged,
        "ocmero_contract": {"alpha": alpha, "beta": beta, "top_m": top_m, "use_lcvar": True, "use_obs_kernel": True},
        "field_contract_example": first_field_diag or {},
        "feature_only_dataset_contract": feature_event,
        "tensor_cache_event": ds.tensor_cache_event,
        "encoder_layer_count": 2,
        "frozen_root_decoder_read_only": True,
        "frozen_margin_head_read_only": True,
        "frozen_root_validity_mask_used": True,
        "teacher_npz_fields_loaded_into_feature_path": ["root_valid"],
        "teacher_margin_probability_compatibility_fields_used": False,
        "teacher_future_fields_used": False,
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
    ee = np.stack([r["exposed_signed_state_geometry"] for r in records]).astype(np.float64)
    fe = np.stack([r["full_signed_state_geometry"] for r in records]).astype(np.float64)
    y = np.asarray([r["label"] for r in records], dtype=np.int64)
    return u, ee, fe, y


def _fit_axis(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    u, ee, fe, y = _arrays(records, key)
    sc = fit_set_flow_scaler(u)
    pi = _perm_indices(records)
    feats = {
        "base": base_features(u, sc),
        "exposed_signed_state": matched_features(u, ee, sc),
        "full_signed_state": matched_features(u, fe, sc),
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
        "rows": len(records), "positive_rows": int(y.sum()),
        "negative_rows": int(len(y) - y.sum()), "auc": auc(y, sc),
        "top1": top1, "powered_groups": len(powered),
    }


def _eval_axis(records: list[dict[str, Any]], key: str, fit: dict[str, Any]) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    spaces = ("base", "exposed_signed_state", "full_signed_state")
    if not records:
        return {space: (_metric([], np.array([])), _metric([], np.array([]))) for space in spaces}
    u, ee, fe, _ = _arrays(records, key)
    pi = _perm_indices(records)
    sc = fit["scaler"]; m = fit["models"]
    feats = {
        "base": base_features(u, sc),
        "exposed_signed_state": matched_features(u, ee, sc),
        "full_signed_state": matched_features(u, fe, sc),
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
    cells = {k: {} for k in ("base", "exposed_signed_state", "full_signed_state")}
    for role in ROLES:
        src = dev_records if role.startswith("dev_") else cert_records
        rr = split_role(src, role)
        su = action_subset(rr, "drs_activation"); re = action_subset(rr, "deployability_gain")
        sm = _eval_axis(su, "support_u", family["support"])
        rm = _eval_axis(re, "reserve_u", family["reserve"])
        for space in cells:
            cells[space][role] = {
                "support_true": sm[space][0], "support_shuffled": sm[space][1],
                "reserve_true": rm[space][0], "reserve_shuffled": rm[space][1],
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
        raise RuntimeError("V48.122 empty audit records")

    fam = _fit_family(tr)
    cells = _eval_family(dv, ce, fam)
    max_resid = max(
        v.normal_equation_residual
        for axis in ("support", "reserve")
        for v in fam[axis]["models"].values()
    )
    result = {
        "schema": "ocrap-v48.122-signed-viability-rank-state-transport-audit-v1",
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
        "exposed_signed_state_cells": cells["exposed_signed_state"],
        "full_signed_state_cells": cells["full_signed_state"],
        "events": events,
        "train_counts": fam["counts"],
        "convex_closed_form_ridge": True,
        "strictly_convex_unique_solution": True,
        "iterative_optimizer_used": False,
        "ridge_lambda_rule": "1_over_axis_train_rows",
        "max_normal_equation_residual": max_resid,
        "score_family": "linear_on_same_option_signed_nominal_rank_state_transport_features",
        "nominal_zero_score_by_construction": True,
        "constraint_names": ["clearance", "stopping", "route", "reentry"],
        "constraint_response": "same_option_signed_nominal_rank_state_prefix_and_suffix_viability_transport",
        "state_channels": ["joint_prefix_signed_viability_rank_state_transport", "joint_suffix_persistent_reentry_signed_rank_state_transport"],
        "option_aggregation": "same_option_signed_margin_displacement_projected_on_nominal_rank_x_absolute_signed_state_basis",
        "primary_option_set": "all_common_valid_recovery_options",
        "control_option_set": "support_of_frozen_v48_117_weak_root_zero_boundary_witnesses",
        "boundary_support_weights_used_in_signed_state": False,
        "state_mode_names": [str(x) for x in STATE_MODE_NAMES],
        "signed_state_basis": "fixed_global_rank_signed_state_and_rank_x_signed_state_modes",
        "rank_coordinate": "candidate_independent_nominal_same_option_viability_midranks",
        "candidate_rank_sort_used_for_coordinate": False,
        "rank_cut_sweep": False,
        "signed_state_threshold_or_scale_sweep": False,
        "same_option_nominal_rank_correspondence": True,
        "absolute_signed_nominal_viability_state": True,
        "active_option_identity_exported": False,
        "option_identity_exported": False,
        "frozen_root_decoder_read_only": True,
        "frozen_margin_head_read_only": True,
        "work_bins": 8,
        "state_geometry_dimension": STATE_GEOMETRY_DIM,
        "matched_family_dimension": MATCHED_DIM,
        "capacity_matched_all_signed_state_families": True,
        "candidate_identity_shuffle": "whole_feature_row_cyclic_permutation_within_scene_time_group",
        "actuator_projection": True,
        "frozen_root_validity_mask_used": True,
        "teacher_npz_fields_loaded_into_feature_path": ["root_valid"],
        "teacher_margin_probability_compatibility_fields_used": False,
        "teacher_future_fields_used": False,
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
        "horizon_sweep": False,
        "threshold_sweep": False,
        "option_count_sweep": False,
        "elapsed_seconds": float(time.perf_counter() - t0),
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    torch.save(
        {
            "schema": "ocrap-v48.122-signed-viability-rank-state-transport-state-v1",
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
