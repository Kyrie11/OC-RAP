#!/usr/bin/env python3
"""v48.57 predicted-root source decomposition for counterfactual recovery.

This is a diagnostic, not a training or calibration procedure.  It compares the
legacy per-candidate OC-MERO integration measure with a counterfactual common
measure that reuses the unique nominal candidate's predicted root posterior for
all candidates in the same scene-time group.  No labels are fed to the model and
no threshold is fitted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ocrap.algorithms.ocmero import oc_mero
from ocrap.data.serialization import load_npz
from ocrap.models.data import fix_sample_geometry, iter_sample_paths_many, scalar_metadata_for_path
from ocrap.models.inference import load_model_bundle, predict_samples


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalise(p: np.ndarray) -> np.ndarray:
    x = np.asarray(p, dtype=np.float64).reshape(-1)
    x = np.where(np.isfinite(x) & (x > 0.0), x, 0.0)
    s = float(x.sum())
    if s <= 1.0e-12:
        return np.full_like(x, 1.0 / max(1, x.size))
    return x / s


def _js(p: np.ndarray, q: np.ndarray) -> float:
    p = _normalise(p)
    q = _normalise(q)
    m = 0.5 * (p + q)
    eps = 1.0e-12
    kl_pm = float(np.sum(np.where(p > 0, p * np.log((p + eps) / (m + eps)), 0.0)))
    kl_qm = float(np.sum(np.where(q > 0, q * np.log((q + eps) / (m + eps)), 0.0)))
    return 0.5 * (kl_pm + kl_qm)


def _quantiles(values: list[float]) -> dict[str, float | None]:
    a = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float64)
    if a.size == 0:
        return {"n": 0, "mean": None, "median": None, "p75": None, "p90": None, "p95": None}
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p75": float(np.quantile(a, 0.75)),
        "p90": float(np.quantile(a, 0.90)),
        "p95": float(np.quantile(a, 0.95)),
    }


def _corr(x: list[float], y: list[float]) -> float | None:
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < 3 or float(np.std(a[m])) < 1e-12 or float(np.std(b[m])) < 1e-12:
        return None
    return float(np.corrcoef(a[m], b[m])[0, 1])


def _auc(labels: list[int], scores: list[float]) -> float | None:
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    m = np.isfinite(s)
    y, s = y[m], s[m]
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and s[order[j]] == s[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + 1 + j)  # one-based average rank
        i = j
    rank_sum = float(ranks[y == 1].sum())
    return float((rank_sum - pos * (pos + 1) / 2.0) / (pos * neg))


def _dataset_spec(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("dataset must be BUCKET=PATH")
    bucket, path = raw.split("=", 1)
    bucket = bucket.strip().lower()
    if bucket not in {"safe", "near", "contact"} or not path.strip():
        raise argparse.ArgumentTypeError("dataset must be safe|near|contact=PATH")
    return bucket, path.strip()


def _scalar(d: dict[str, Any], key: str, default: Any) -> Any:
    if key not in d:
        return default
    a = np.asarray(d[key])
    return a.item() if a.size == 1 else a.reshape(-1)[0].item()


def _slot_alignment(
    candidate: np.ndarray, nominal: np.ndarray, candidate_valid: np.ndarray, nominal_valid: np.ndarray
) -> tuple[float | None, float | None, float | None]:
    """Return identity cosine, nearest-slot identity rate, and best-minus-identity gap.

    This is diagnostic only.  It asks whether a candidate root slot looks most like
    the nominal slot with the same index; CMRI is not authorized from this statistic
    alone.
    """
    a = np.asarray(candidate, dtype=np.float64)
    b = np.asarray(nominal, dtype=np.float64)
    va = np.asarray(candidate_valid, dtype=bool).reshape(-1)
    vb = np.asarray(nominal_valid, dtype=bool).reshape(-1)
    if a.ndim != 2 or b.ndim != 2 or a.shape != b.shape or a.shape[1] == 0:
        return None, None, None
    shared = np.where(va & vb)[0]
    nominal_idx = np.where(vb)[0]
    if shared.size == 0 or nominal_idx.size == 0:
        return None, None, None
    aa = a[shared]
    bb = b[nominal_idx]
    aa_n = aa / np.maximum(np.linalg.norm(aa, axis=1, keepdims=True), 1.0e-12)
    bb_n = bb / np.maximum(np.linalg.norm(bb, axis=1, keepdims=True), 1.0e-12)
    sim = aa_n @ bb_n.T
    index_to_col = {int(idx): j for j, idx in enumerate(nominal_idx.tolist())}
    identity = np.asarray([sim[row, index_to_col[int(idx)]] for row, idx in enumerate(shared)], dtype=np.float64)
    best_col = np.argmax(sim, axis=1)
    nearest_nominal_idx = nominal_idx[best_col]
    identity_rate = float(np.mean(nearest_nominal_idx == shared))
    best = np.max(sim, axis=1)
    return float(identity.mean()), identity_rate, float(np.mean(best - identity))


def audit_bucket(
    bucket: str,
    dataset: str,
    bundle,
    *,
    positive_gain: float,
) -> dict[str, Any]:
    paths = iter_sample_paths_many(dataset)
    grouped: dict[tuple[str, int], list[Path]] = defaultdict(list)
    for p in paths:
        scene = str(scalar_metadata_for_path(p, "scene_id", p.stem))
        time_index = int(scalar_metadata_for_path(p, "time_index", 0))
        grouped[(scene, time_index)].append(p)

    raw_rel: list[float] = []
    common_rel: list[float] = []
    teacher_rel: list[float] = []
    pred_js: list[float] = []
    teacher_js: list[float] = []
    pred_teacher_js_raw: list[float] = []
    pred_teacher_js_common: list[float] = []
    deployed_measure_js: list[float] = []
    native_dep_common_abs_error: list[float] = []
    abs_err_raw: list[float] = []
    abs_err_common: list[float] = []
    common_abs_error_gain: list[float] = []
    teacher_positive_raw_capture: list[int] = []
    teacher_positive_common_capture: list[int] = []
    teacher_positive_raw_rel: list[float] = []
    teacher_positive_common_rel: list[float] = []
    labels_positive: list[int] = []
    support_identical_pairs: list[int] = []
    root_valid_hamming: list[float] = []
    root_signature_identity_cosine: list[float] = []
    root_signature_nearest_identity_rate: list[float] = []
    root_signature_best_minus_identity_gap: list[float] = []
    root_future_signature_identity_cosine: list[float] = []
    root_future_signature_nearest_identity_rate: list[float] = []
    root_future_signature_best_minus_identity_gap: list[float] = []
    cmri_eligible_groups = 0
    malformed = defaultdict(int)
    group_count = 0

    oc_cfg = bundle.cfg.get("ocmero", {}) or {}
    ab_cfg = bundle.cfg.get("ablation", {}) or {}
    alpha = float(oc_cfg.get("alpha", 0.2))
    beta = float(oc_cfg.get("beta", 0.2))
    top_m = int(oc_cfg.get("top_m", 8))
    use_lcvar = not bool(ab_cfg.get("without_lower_tail", False))
    use_obs_kernel = not bool(ab_cfg.get("without_observation_kernel", False))

    for gi, ((scene, time_index), gp) in enumerate(sorted(grouped.items()), 1):
        ds = [load_npz(p) for p in gp]
        order = np.argsort([int(_scalar(d, "candidate_index", i)) for i, d in enumerate(ds)], kind="mergesort")
        ds = [ds[int(i)] for i in order]
        noms = [i for i, d in enumerate(ds) if float(_scalar(d, "is_nominal", 0.0)) > 0.5]
        if len(noms) != 1:
            malformed["non_unique_nominal"] += 1
            continue
        ni = noms[0]
        preds = predict_samples(ds, bundle, bundle.cfg, shared_scene_features=True)
        fixed = [
            fix_sample_geometry(
                d,
                num_roots=bundle.model.num_roots,
                num_options=bundle.model.num_options,
                d_signature=int(getattr(bundle.model, "d_signature", 0)),
                d_future_signature=int(getattr(bundle.model, "d_future_signature", 0)),
            )
            for d in ds
        ]
        p_nom = np.asarray(preds[ni].root_probs, dtype=np.float64)
        teacher_p_nom = np.asarray(fixed[ni]["root_probs"], dtype=np.float64)
        raw_nom = float(preds[ni].r_dep)
        teacher_nom = float(_scalar(ds[ni], "r_dep_star", np.nan))
        if not np.isfinite(teacher_nom):
            malformed["missing_teacher_r_dep"] += 1
            continue
        group_count += 1
        nominal_valid = np.asarray(fixed[ni]["root_valid"], dtype=bool)
        support_consistent = all(
            np.array_equal(np.asarray(fx["root_valid"], dtype=bool), nominal_valid)
            for fx in fixed
        )
        if support_consistent:
            cmri_eligible_groups += 1

        # CMRI must reproduce the raw nominal score exactly.  If root support is
        # not shared by every candidate, the deployed mechanism fails closed for
        # the whole group, so the diagnostic substitution does the same.
        cm_nom = oc_mero(
            preds[ni].margins, p_nom, preds[ni].c_star,
            alpha=alpha, beta=beta,
            option_valid=fixed[ni]["option_valid"], root_valid=fixed[ni]["root_valid"],
            use_lcvar=use_lcvar, use_obs_kernel=use_obs_kernel, top_m=top_m,
        ).r_dep
        if abs(float(cm_nom) - raw_nom) > 5e-5:
            malformed["nominal_recompute_mismatch"] += 1

        for i, (d, pred, fx) in enumerate(zip(ds, preds, fixed)):
            if i == ni:
                continue
            teacher_candidate = float(_scalar(d, "r_dep_star", np.nan))
            if not np.isfinite(teacher_candidate):
                malformed["missing_teacher_r_dep_candidate"] += 1
                continue
            candidate_valid = np.asarray(fx["root_valid"], dtype=bool)
            pair_support_identical = bool(np.array_equal(candidate_valid, nominal_valid))
            support_identical_pairs.append(int(pair_support_identical))
            root_valid_hamming.append(float(np.mean(candidate_valid != nominal_valid)))
            cm_p = p_nom if support_consistent else np.asarray(pred.root_probs, dtype=np.float64)
            cm = oc_mero(
                pred.margins, cm_p, pred.c_star,
                alpha=alpha, beta=beta,
                option_valid=fx["option_valid"], root_valid=fx["root_valid"],
                use_lcvar=use_lcvar, use_obs_kernel=use_obs_kernel, top_m=top_m,
            ).r_dep
            for key, dst_identity, dst_rate, dst_gap in (
                ("root_signature", root_signature_identity_cosine, root_signature_nearest_identity_rate, root_signature_best_minus_identity_gap),
                ("root_future_signature", root_future_signature_identity_cosine, root_future_signature_nearest_identity_rate, root_future_signature_best_minus_identity_gap),
            ):
                ident, rate, gap_align = _slot_alignment(
                    np.asarray(fx[key]), np.asarray(fixed[ni][key]), candidate_valid, nominal_valid
                )
                if ident is not None:
                    dst_identity.append(ident)
                    dst_rate.append(rate)
                    dst_gap.append(gap_align)
            tr = teacher_candidate - teacher_nom
            rr = float(pred.r_dep) - raw_nom
            cr = float(cm) - float(cm_nom)
            pj = _js(np.asarray(pred.root_probs), p_nom)
            tj = _js(np.asarray(fx["root_probs"]), teacher_p_nom)
            pt_raw = _js(np.asarray(pred.root_probs), np.asarray(fx["root_probs"]))
            pt_common = _js(cm_p, np.asarray(fx["root_probs"]))
            deployed_p = np.asarray(
                pred.recovery_root_probs if pred.recovery_root_probs is not None else pred.root_probs,
                dtype=np.float64,
            )
            expected_deployed_measure = p_nom if support_consistent else np.asarray(pred.root_probs, dtype=np.float64)
            deployed_measure_js.append(_js(deployed_p, expected_deployed_measure))
            if pred.direct_recovery_native_certificate is not None:
                native = np.asarray(pred.direct_recovery_native_certificate, dtype=np.float64).reshape(-1)
                if native.size >= 2 and 0.0 < float(native[1]) < 1.0:
                    native_r_dep = math.log(float(native[1]) / (1.0 - float(native[1])))
                    native_dep_common_abs_error.append(abs(native_r_dep - float(cm)))
            er = abs(rr - tr)
            ec = abs(cr - tr)

            teacher_rel.append(tr)
            raw_rel.append(rr)
            common_rel.append(cr)
            pred_js.append(pj)
            teacher_js.append(tj)
            pred_teacher_js_raw.append(pt_raw)
            pred_teacher_js_common.append(pt_common)
            abs_err_raw.append(er)
            abs_err_common.append(ec)
            common_abs_error_gain.append(er - ec)
            positive = int(tr >= positive_gain)
            labels_positive.append(positive)
            if positive:
                teacher_positive_raw_capture.append(int(rr >= positive_gain))
                teacher_positive_common_capture.append(int(cr >= positive_gain))
                teacher_positive_raw_rel.append(rr)
                teacher_positive_common_rel.append(cr)
        if gi == 1 or gi % 200 == 0 or gi == len(grouped):
            print({"event": "v48_57_root_source_audit_progress", "bucket": bucket, "groups": gi, "total": len(grouped)}, flush=True)

    raw = np.asarray(raw_rel, dtype=np.float64)
    common = np.asarray(common_rel, dtype=np.float64)
    teacher = np.asarray(teacher_rel, dtype=np.float64)
    sign_raw = float(np.mean(np.sign(raw) == np.sign(teacher))) if raw.size else None
    sign_common = float(np.mean(np.sign(common) == np.sign(teacher))) if raw.size else None
    pos_n = len(teacher_positive_raw_capture)
    return {
        "bucket": bucket,
        "dataset": str(dataset),
        "num_paths": int(len(paths)),
        "num_scene_time_groups": int(group_count),
        "num_candidate_pairs": int(len(raw_rel)),
        "malformed": dict(malformed),
        "common_measure_definition": "nominal predicted root posterior shared only when every candidate has the same root-valid support; candidate margins/C/option validity remain candidate-specific; otherwise fail closed",
        "cmri_eligibility": {
            "eligible_groups": int(cmri_eligible_groups),
            "group_coverage": (None if group_count == 0 else float(cmri_eligible_groups / group_count)),
            "pair_root_support_identity_rate": (None if not support_identical_pairs else float(np.mean(support_identical_pairs))),
            "root_valid_hamming": _quantiles(root_valid_hamming),
        },
        "root_slot_alignment_diagnostic": {
            "root_signature_identity_cosine": _quantiles(root_signature_identity_cosine),
            "root_signature_nearest_slot_identity_rate": _quantiles(root_signature_nearest_identity_rate),
            "root_signature_best_minus_identity_cosine_gap": _quantiles(root_signature_best_minus_identity_gap),
            "root_future_signature_identity_cosine": _quantiles(root_future_signature_identity_cosine),
            "root_future_signature_nearest_slot_identity_rate": _quantiles(root_future_signature_nearest_identity_rate),
            "root_future_signature_best_minus_identity_cosine_gap": _quantiles(root_future_signature_best_minus_identity_gap),
            "warning": "low identity alignment is a CMRI stop signal because root indices may not define a common latent-world coordinate system",
        },
        "root_source_drift": {
            "predicted_candidate_to_nominal_js": _quantiles(pred_js),
            "teacher_candidate_to_nominal_js": _quantiles(teacher_js),
            "excess_predicted_minus_teacher_js": _quantiles([a - b for a, b in zip(pred_js, teacher_js)]),
            "predicted_to_teacher_js_raw": _quantiles(pred_teacher_js_raw),
            "predicted_common_measure_to_teacher_candidate_js": _quantiles(pred_teacher_js_common),
            "deployed_recovery_measure_to_nominal_js": _quantiles(deployed_measure_js),
            "native_dep_vs_recomputed_common_r_dep_abs_error": _quantiles(native_dep_common_abs_error),
        },
        "relative_r_dep": {
            "teacher": _quantiles(teacher_rel),
            "legacy_predicted": _quantiles(raw_rel),
            "common_measure_predicted": _quantiles(common_rel),
            "legacy_abs_error": _quantiles(abs_err_raw),
            "common_measure_abs_error": _quantiles(abs_err_common),
            "common_measure_abs_error_gain_positive_is_better": _quantiles(common_abs_error_gain),
            "legacy_sign_accuracy": sign_raw,
            "common_measure_sign_accuracy": sign_common,
            "legacy_safe_positive_auc": _auc(labels_positive, raw_rel),
            "common_measure_safe_positive_auc": _auc(labels_positive, common_rel),
            "root_js_vs_common_measure_error_gain_correlation": _corr(pred_js, common_abs_error_gain),
        },
        "teacher_positive_gain_cohort": {
            "positive_gain": float(positive_gain),
            "n": int(pos_n),
            "legacy_capture_rate": (None if pos_n == 0 else float(np.mean(teacher_positive_raw_capture))),
            "common_measure_capture_rate": (None if pos_n == 0 else float(np.mean(teacher_positive_common_capture))),
            "legacy_predicted_relative_r_dep": _quantiles(teacher_positive_raw_rel),
            "common_measure_predicted_relative_r_dep": _quantiles(teacher_positive_common_rel),
        },
        "interpretation_contract": {
            "diagnostic_only": True,
            "fits_no_threshold": True,
            "updates_no_model_parameter": True,
            "uses_no_test_root": True,
            "go_signal": "predicted root drift exceeds teacher drift and common-measure substitution improves Near+Contact relative-R_dep error/capture without relying on regime-specific logic",
            "stop_signal": "root-support/slot alignment is too weak, teacher root drift is comparably large, or common-measure substitution does not improve relative-R_dep geometry",
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--dataset", action="append", type=_dataset_spec, required=True, help="repeat BUCKET=PATH")
    ap.add_argument("--positive-gain", type=float, default=0.015)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    bundle = load_model_bundle(str(args.checkpoint), {"training": {"device": args.device}})
    results = [
        audit_bucket(bucket, path, bundle, positive_gain=float(args.positive_gain))
        for bucket, path in args.dataset
    ]
    doc = {
        "schema": "ocrap-v48.57-root-source-decomposition-v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "model_common_measure_root_mass_enabled": bool(getattr(bundle.model, "direct_recovery_evidence_common_measure_root_mass", False)),
        "positive_gain": float(args.positive_gain),
        "buckets": {r["bucket"]: r for r in results},
        "test_roots_read": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "v48_57_root_source_audit_complete", "output": str(args.output), "buckets": list(doc["buckets"])}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
