from __future__ import annotations

from pathlib import Path
from time import perf_counter
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler

from ocrap.algorithms.ocmero import torch_oc_mero
from ocrap.data.serialization import ensure_dir, write_json
from ocrap.models.data import OCRAPSampleDataset, OPTION_FEATURE_DIM, bucket_id_for_path, iter_sample_paths_many, scalar_metadata_for_path, split_paths_by_npz_split, stable_scene_hash
from ocrap.models.losses import (
    anti_oracle_loss,
    artifact_gap_loss,
    best_shared_option_loss,
    deployability_classification_loss,
    shared_option_admission_loss,
    shared_option_q_regression_loss,
    shared_option_success_regression_loss,
    shared_option_success_bce_loss,
    groupwise_candidate_ranking_loss,
    groupwise_candidate_ce_loss,
    nominal_switch_consistency_loss,
    groupwise_score_distillation_loss,
    safe_nominal_preservation_loss,
    protective_macro_recovery_loss,
    deployability_dominance_calibration_loss,
    direct_teacher_pcd_loss,
    macro_shared_success_calibration_loss,
    observation_consistent_recovery_advantage_loss,
    direct_uncertainty_recovery_value_loss,
)
from ocrap.models.ocrap import OCRAPModel
from ocrap.utils.seed import seed_everything


def _device(cfg: dict) -> torch.device:
    tcfg = cfg.get("training", {}) if isinstance(cfg.get("training", {}), dict) else {}
    requested = str(tcfg.get("device", "auto"))
    if requested == "auto":
        if bool(tcfg.get("require_cuda", False)) and not torch.cuda.is_available():
            raise RuntimeError("training.require_cuda=true, but torch.cuda.is_available() is false")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if bool(tcfg.get("require_cuda", False)) and device.type != "cuda":
        raise RuntimeError("training.require_cuda=true, but selected device is not CUDA")
    if bool(tcfg.get("require_cuda", False)) and not torch.cuda.is_available():
        raise RuntimeError("training.require_cuda=true, but torch.cuda.is_available() is false")
    return device


def _device_summary(device: torch.device) -> dict[str, object]:
    summary: dict[str, object] = {
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_version": str(torch.__version__),
    }
    if torch.cuda.is_available():
        idx = device.index if device.type == "cuda" and device.index is not None else torch.cuda.current_device()
        try:
            summary.update({
                "cuda_device_index": int(idx),
                "cuda_device_name": torch.cuda.get_device_name(idx),
                "cuda_device_count": int(torch.cuda.device_count()),
            })
        except Exception:
            pass
    return summary


def _collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in batch[0].keys()}


def _masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
    mask = mask.bool() & torch.isfinite(target) & (target > -1e8)
    if not bool(mask.any()):
        return pred.sum() * 0.0
    return F.smooth_l1_loss(pred[mask], target[mask])


def _root_signature_loss(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], key: str, valid_key: str = "root_valid") -> torch.Tensor:
    if key not in out or key not in batch or batch[key].shape[-1] == 0:
        ref = next(iter(out.values()))
        return ref.sum() * 0.0
    pred = out[key]
    target = batch[key].float()
    mask = batch[valid_key].bool().unsqueeze(-1).expand_as(target)
    return _masked_smooth_l1(pred, target, mask)


def _obs_bce(pred_c: torch.Tensor, target_y: torch.Tensor, root_valid: torch.Tensor, *, balanced: bool = True) -> torch.Tensor:
    B, K, _ = pred_c.shape
    eye = torch.eye(K, dtype=torch.bool, device=pred_c.device).unsqueeze(0)
    pair_mask = root_valid.unsqueeze(1) & root_valid.unsqueeze(2) & (~eye)
    if not bool(pair_mask.any()):
        return pred_c.sum() * 0.0
    pred = pred_c.clamp(1e-5, 1.0 - 1e-5)[pair_mask]
    target = target_y[pair_mask].float().clamp(0.0, 1.0)
    if not balanced:
        return F.binary_cross_entropy(pred, target)
    pos = target >= 0.5
    neg = ~pos
    if not bool(pos.any()) or not bool(neg.any()):
        return F.binary_cross_entropy(pred, target)
    weights = torch.where(pos, 0.5 / pos.float().mean().clamp_min(1e-6), 0.5 / neg.float().mean().clamp_min(1e-6))
    return F.binary_cross_entropy(pred, target, weight=weights)


def _progress_iter(loader: DataLoader, *, enabled: bool, desc: str):
    if not enabled:
        return loader
    try:
        from tqdm.auto import tqdm  # type: ignore
        return tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    except Exception:
        return loader


def _keep_fully_frozen_modules_in_eval(model: torch.nn.Module) -> None:
    """Disable train-time dropout in modules whose parameters are all frozen.

    Head-only recovery-value training freezes the OC-MERO encoder. Calling
    ``model.train()`` nevertheless re-enabled dropout inside that frozen
    encoder, so the new head was trained on stochastic features that disappear
    at calibration/inference time. Keep only fully frozen subtrees in eval
    mode; trainable direct heads and routers remain in training mode.
    """
    for module in list(model.modules())[1:]:
        params = list(module.parameters())
        if params and all(not p.requires_grad for p in params):
            module.eval()


def _dataset_root_name(p: Path) -> str:
    parts = list(p.parts)
    for i in range(len(parts) - 2, -1, -1):
        name = parts[i]
        low = name.lower()
        if any(tok in low for tok in ("safe", "near", "contact", "train_", "val_", "test_")):
            return name
    return p.parent.parent.name if p.parent.name == "samples" else p.parent.name




def _parse_int_tuple(value, default: tuple[int, ...]) -> tuple[int, ...]:
    if value is None or value == "":
        return tuple(default)
    if isinstance(value, (list, tuple, set)):
        out = []
        for x in value:
            try:
                out.append(int(x))
            except Exception:
                continue
        return tuple(out) if out else tuple(default)
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    out = []
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except Exception:
            continue
    return tuple(out) if out else tuple(default)


def _profile_paths(paths: list[Path], *, stage: str, max_scalar_scan: int | None = None) -> dict[str, object]:
    total = len(paths)
    roots = Counter(_dataset_root_name(p) for p in paths)
    limit = total if max_scalar_scan is None or max_scalar_scan <= 0 else min(total, int(max_scalar_scan))
    num_art = num_neg = num_safe_pos = 0
    r_sum = 0.0
    scanned = 0
    for p in paths[:limit]:
        try:
            is_art = float(np.asarray(scalar_metadata_for_path(p, "i_art_star", 0)).item()) > 0.5
            r_dep = float(np.asarray(scalar_metadata_for_path(p, "r_dep_star", 0)).item())
            is_neg = r_dep < 0.0
            root = _dataset_root_name(p).lower()
            is_safe_pos = ("safe" in root) and (not is_neg) and (not is_art)
            num_art += int(is_art)
            num_neg += int(is_neg)
            num_safe_pos += int(is_safe_pos)
            r_sum += r_dep
            scanned += 1
        except Exception:
            pass
    return {
        "event": "dataset_profile",
        "stage": stage,
        "num_paths": int(total),
        "roots": dict(sorted(roots.items())),
        "scalar_scanned": int(scanned),
        "artifact_fraction": float(num_art / max(scanned, 1)),
        "negative_deployable_fraction": float(num_neg / max(scanned, 1)),
        "safe_positive_fraction": float(num_safe_pos / max(scanned, 1)),
        "r_dep_mean": float(r_sum / max(scanned, 1)),
    }


def _epoch(
    model: OCRAPModel,
    loader: DataLoader,
    cfg: dict,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    stage: str = "train",
    epoch: int | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    lw = cfg.get("loss_weights", {}) if isinstance(cfg.get("loss_weights", {}), dict) else {}
    ocfg = cfg.get("ocmero", {}) if isinstance(cfg.get("ocmero", {}), dict) else {}
    art_cfg = cfg.get("artifact", {}) if isinstance(cfg.get("artifact", {}), dict) else {}
    tcfg = cfg.get("training", {}) if isinstance(cfg.get("training", {}), dict) else {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
    if training and bool(tcfg.get("frozen_modules_eval", False)):
        _keep_fully_frozen_modules_in_eval(model)
    progress = bool(tcfg.get("progress", cfg.get("progress", True)))
    totals: dict[str, float] = {}
    n = 0
    desc = f"{stage} ep{epoch}" if epoch is not None else stage
    for batch in _progress_iter(loader, enabled=progress, desc=desc):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        out = model(batch["x"].float(), batch.get("option_features"), bucket_id=batch.get("bucket_id"))
        root_valid = batch["root_valid"].bool()
        masked_logits = out["root_logits"].masked_fill(~root_valid, -1.0e4)
        root_p = torch.softmax(masked_logits, dim=-1)
        root_target = batch["root_probs"].float() * root_valid.float()
        root_target = root_target / root_target.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        margin_target = torch.clamp(batch["m_star"].float(), min=-float(cfg.get("margin_clip", 5.0)), max=float(cfg.get("margin_clip", 5.0)))
        margin_mask = batch["root_valid"].unsqueeze(-1) & batch["option_valid"].unsqueeze(1)

        loss_root = -(root_target * F.log_softmax(masked_logits, dim=-1)).sum(dim=-1).mean()
        loss_margin = _masked_smooth_l1(out["margins"], margin_target, margin_mask)
        loss_sig = _root_signature_loss(out, batch, "root_signature")
        loss_future_sig = _root_signature_loss(out, batch, "root_future_signature")
        loss_obs = _obs_bce(
            out["c_star"],
            batch["y_obs"],
            batch["root_valid"],
            balanced=bool(tcfg.get("balanced_obs_loss", True)),
        )
        r_dep, r_orc, gap, pred_q = torch_oc_mero(
            out["margins"],
            root_p,
            out["c_star"],
            alpha=float(ocfg.get("alpha", 0.2)),
            beta=float(ocfg.get("beta", 0.2)),
            option_valid=batch["option_valid"],
            root_valid=root_valid,
            use_lcvar=not bool((cfg.get("ablation", {}) or {}).get("without_lower_tail", False)),
            use_obs_kernel=not bool((cfg.get("ablation", {}) or {}).get("without_observation_kernel", False)),
            top_m=int(ocfg.get("top_m", 8)),
        )
        with torch.no_grad():
            _teacher_r_dep, _teacher_r_orc, _teacher_gap, teacher_q = torch_oc_mero(
                batch["m_star"].float(),
                batch["root_probs"].float(),
                batch["c_star"].float(),
                alpha=float(ocfg.get("alpha", 0.2)),
                beta=float(ocfg.get("beta", 0.2)),
                option_valid=batch["option_valid"],
                root_valid=root_valid,
                use_lcvar=not bool((cfg.get("ablation", {}) or {}).get("without_lower_tail", False)),
                use_obs_kernel=not bool((cfg.get("ablation", {}) or {}).get("without_observation_kernel", False)),
                top_m=int(ocfg.get("top_m", 8)),
            )
        loss_dep = F.smooth_l1_loss(r_dep, batch["r_dep_star"].float())
        loss_orc = F.smooth_l1_loss(r_orc, batch["r_orc_star"].float())
        loss_art = anti_oracle_loss(r_orc, r_dep, batch["i_art_star"].float(), delta_neg=float(art_cfg.get("delta_neg", 0.0)))
        loss_gap = artifact_gap_loss(gap, (batch["r_orc_star"].float() - batch["r_dep_star"].float()).clamp_min(0.0), batch["i_art_star"].float(), margin=float(art_cfg.get("gap_margin", 0.5)))
        loss_admit = deployability_classification_loss(r_dep, batch["r_dep_star"].float(), gamma=float(art_cfg.get("admission_gamma", 0.0)))
        option_gamma = float(art_cfg.get("admission_gamma", 0.0))
        option_temperature = float(tcfg.get("option_success_temperature", 0.35))
        loss_option_q = shared_option_q_regression_loss(pred_q, teacher_q, batch["root_valid"], batch["option_valid"])
        loss_option_admit = shared_option_admission_loss(
            pred_q,
            teacher_q,
            batch["root_probs"].float(),
            batch["root_valid"],
            batch["option_valid"],
            gamma=option_gamma,
        )
        loss_option_success = shared_option_success_regression_loss(
            pred_q,
            teacher_q,
            batch["root_probs"].float(),
            batch["root_valid"],
            batch["option_valid"],
            gamma=option_gamma,
            temperature=option_temperature,
        )
        loss_option_success_bce = shared_option_success_bce_loss(
            pred_q,
            teacher_q,
            batch["root_probs"].float(),
            batch["root_valid"],
            batch["option_valid"],
            gamma=option_gamma,
            temperature=option_temperature,
        )
        loss_option_best = best_shared_option_loss(
            pred_q,
            teacher_q,
            batch["root_probs"].float(),
            batch["root_valid"],
            batch["option_valid"],
            gamma=option_gamma,
            temperature=option_temperature,
        )
        loss_group_rank = groupwise_candidate_ranking_loss(
            r_dep,
            gap,
            batch["r_dep_star"].float(),
            batch["r_orc_star"].float(),
            batch["i_art_star"].float(),
            batch["scene_hash"],
            batch["time_index"],
            batch["candidate_index"],
            margin=float(tcfg.get("group_ranking_margin", 0.25)),
            gap_weight=float(tcfg.get("group_ranking_gap_weight", 0.25)),
            teacher_gap_weight=float(tcfg.get("group_ranking_teacher_gap_weight", 0.25)),
            artifact_only=bool(tcfg.get("group_ranking_artifact_only", True)),
        )
        loss_group_ce = groupwise_candidate_ce_loss(
            r_dep,
            gap,
            batch["utility"].float(),
            batch["r_dep_star"].float(),
            batch["r_orc_star"].float(),
            batch["scene_hash"],
            batch["time_index"],
            temperature=float(tcfg.get("group_ce_temperature", 0.35)),
            pred_gap_weight=float(tcfg.get("group_ce_pred_gap_weight", 0.35)),
            teacher_gap_weight=float(tcfg.get("group_ce_teacher_gap_weight", 0.35)),
            utility_weight=float(tcfg.get("group_ce_utility_weight", 0.03)),
            require_deployable_target=bool(tcfg.get("group_ce_require_deployable_target", True)),
        )
        loss_nominal_switch = nominal_switch_consistency_loss(
            r_dep,
            gap,
            batch["utility"].float(),
            batch["r_dep_star"].float(),
            batch["r_orc_star"].float(),
            batch["scene_hash"],
            batch["time_index"],
            batch["is_nominal"].float(),
            margin=float(tcfg.get("nominal_switch_margin", 0.12)),
            pred_gap_weight=float(tcfg.get("nominal_switch_pred_gap_weight", 0.30)),
            teacher_gap_weight=float(tcfg.get("nominal_switch_teacher_gap_weight", 0.35)),
            utility_weight=float(tcfg.get("nominal_switch_utility_weight", 0.03)),
            teacher_gain_margin=float(tcfg.get("nominal_switch_teacher_gain_margin", 0.06)),
            nominal_deployable_gamma=float(tcfg.get("nominal_switch_deployable_gamma", 0.0)),
            nominal_gap_max=float(tcfg.get("nominal_switch_gap_max", 0.30)),
        )
        loss_group_distill = groupwise_score_distillation_loss(
            r_dep,
            gap,
            batch["utility"].float(),
            batch["r_dep_star"].float(),
            batch["r_orc_star"].float(),
            batch["scene_hash"],
            batch["time_index"],
            pred_gap_weight=float(tcfg.get("group_distill_pred_gap_weight", 0.45)),
            teacher_gap_weight=float(tcfg.get("group_distill_teacher_gap_weight", 0.45)),
            utility_weight=float(tcfg.get("group_distill_utility_weight", 0.02)),
            teacher_temperature=float(tcfg.get("group_distill_teacher_temperature", 0.20)),
            pred_temperature=float(tcfg.get("group_distill_pred_temperature", 0.30)),
        )
        loss_safe_nominal = safe_nominal_preservation_loss(
            r_dep,
            gap,
            batch["utility"].float(),
            pred_q,
            batch["root_probs"].float(),
            batch["root_valid"],
            batch["option_valid"],
            batch["scene_hash"],
            batch["time_index"],
            batch["is_nominal"].float(),
            batch.get("bucket_id", torch.full_like(batch["time_index"], 3)),
            margin=float(tcfg.get("safe_nominal_margin", 0.18)),
            pred_gap_weight=float(tcfg.get("safe_nominal_pred_gap_weight", 0.35)),
            utility_weight=float(tcfg.get("safe_nominal_utility_weight", 0.03)),
            drs_weight=float(tcfg.get("safe_nominal_drs_weight", 0.30)),
            min_nominal_success=float(tcfg.get("safe_nominal_min_success", 0.90)),
            success_gamma=option_gamma,
            success_temperature=option_temperature,
        )
        loss_protective_macro = protective_macro_recovery_loss(
            r_dep,
            gap,
            batch["utility"].float(),
            pred_q,
            batch["r_dep_star"].float(),
            batch["r_orc_star"].float(),
            teacher_q,
            batch["root_probs"].float(),
            batch["root_valid"],
            batch["option_valid"],
            batch["scene_hash"],
            batch["time_index"],
            batch.get("prefix_macro_type_id", batch.get("candidate_index", torch.zeros_like(batch["time_index"]))),
            batch["is_nominal"].float(),
            batch.get("bucket_id", torch.full_like(batch["time_index"], 3)),
            macro_ids=_parse_int_tuple(tcfg.get("protective_macro_ids", "2,7"), (2, 7)),
            bucket_ids=_parse_int_tuple(tcfg.get("protective_macro_bucket_ids", "2"), (2,)),
            margin=float(tcfg.get("protective_macro_margin", 0.14)),
            min_teacher_r_dep=float(tcfg.get("protective_macro_min_teacher_r_dep", 0.0)),
            min_teacher_drs=float(tcfg.get("protective_macro_min_teacher_drs", 0.50)),
            min_teacher_pcd_gain=float(tcfg.get("protective_macro_min_teacher_pcd_gain", 0.02)),
            max_nominal_teacher_pcd=float(tcfg.get("protective_macro_max_nominal_teacher_pcd", 0.90)),
            pred_gap_weight=float(tcfg.get("protective_macro_pred_gap_weight", 0.18)),
            pred_drs_weight=float(tcfg.get("protective_macro_pred_drs_weight", 0.65)),
            utility_weight=float(tcfg.get("protective_macro_utility_weight", 0.02)),
            teacher_gap_weight=float(tcfg.get("protective_macro_teacher_gap_weight", 0.10)),
            teacher_drs_weight=float(tcfg.get("protective_macro_teacher_drs_weight", 0.70)),
            success_gamma=option_gamma,
            success_temperature=option_temperature,
            target_min_pred_drs=float(tcfg.get("protective_macro_target_min_pred_drs", 0.62)),
        )
        loss_macro_drs = macro_shared_success_calibration_loss(
            pred_q,
            teacher_q,
            batch["root_probs"].float(),
            batch["root_valid"],
            batch["option_valid"],
            batch.get("prefix_macro_type_id", batch.get("candidate_index", torch.zeros_like(batch["time_index"]))),
            batch.get("bucket_id", torch.full_like(batch["time_index"], 3)),
            macro_ids=_parse_int_tuple(tcfg.get("macro_drs_ids", "2,3,5,7"), (2, 3, 5, 7)),
            bucket_ids=_parse_int_tuple(tcfg.get("macro_drs_bucket_ids", "1,2"), (1, 2)),
            gamma=option_gamma,
            temperature=option_temperature,
            pos_threshold=float(tcfg.get("macro_drs_pos_threshold", 0.80)),
            neg_threshold=float(tcfg.get("macro_drs_neg_threshold", 0.05)),
            pos_weight=float(tcfg.get("macro_drs_pos_weight", 4.0)),
            neg_weight=float(tcfg.get("macro_drs_neg_weight", 1.0)),
        )
        loss_ddc = deployability_dominance_calibration_loss(
            r_dep,
            gap,
            batch["utility"].float(),
            pred_q,
            batch["r_dep_star"].float(),
            batch["r_orc_star"].float(),
            teacher_q,
            batch["root_probs"].float(),
            batch["root_valid"],
            batch["option_valid"],
            batch["scene_hash"],
            batch["time_index"],
            batch.get("prefix_macro_type_id", batch.get("candidate_index", torch.zeros_like(batch["time_index"]))),
            batch["is_nominal"].float(),
            batch.get("bucket_id", torch.full_like(batch["time_index"], 3)),
            macro_ids=_parse_int_tuple(tcfg.get("ddc_macro_ids", "2,3,5,7"), (2, 3, 5, 7)),
            bucket_ids=_parse_int_tuple(tcfg.get("ddc_bucket_ids", "1,2"), (1, 2)),
            margin=float(tcfg.get("ddc_margin", 0.12)),
            min_teacher_pcd_gain=float(tcfg.get("ddc_min_teacher_pcd_gain", 0.04)),
            min_teacher_best_pcd=float(tcfg.get("ddc_min_teacher_best_pcd", 0.50)),
            max_nominal_teacher_pcd=float(tcfg.get("ddc_max_nominal_teacher_pcd", 0.62)),
            pred_gap_weight=float(tcfg.get("ddc_pred_gap_weight", 0.20)),
            pred_drs_weight=float(tcfg.get("ddc_pred_drs_weight", 0.35)),
            utility_weight=float(tcfg.get("ddc_utility_weight", 0.00)),
            success_gamma=option_gamma,
            success_temperature=option_temperature,
            target_min_pred_pcd=float(tcfg.get("ddc_target_min_pred_pcd", 0.45)),
            nominal_max_pred_pcd=float(tcfg.get("ddc_nominal_max_pred_pcd", 0.55)),
        )
        loss_teacher_pcd_direct = direct_teacher_pcd_loss(
            r_dep,
            gap,
            batch["utility"].float(),
            pred_q,
            batch["r_dep_star"].float(),
            batch["r_orc_star"].float(),
            teacher_q,
            batch["root_probs"].float(),
            batch["root_valid"],
            batch["option_valid"],
            batch["scene_hash"],
            batch["time_index"],
            batch.get("prefix_macro_type_id", batch.get("candidate_index", torch.zeros_like(batch["time_index"]))),
            batch["is_nominal"].float(),
            batch.get("bucket_id", torch.full_like(batch["time_index"], 3)),
            macro_ids=_parse_int_tuple(tcfg.get("teacher_pcd_direct_macro_ids", "2,3,5,7"), (2, 3, 5, 7)),
            positive_macro_ids=_parse_int_tuple(tcfg.get("teacher_pcd_direct_positive_macro_ids", tcfg.get("teacher_pcd_direct_macro_ids", "2,3,5,7")), (2, 3, 5, 7)),
            bucket_ids=_parse_int_tuple(tcfg.get("teacher_pcd_direct_bucket_ids", "2"), (2,)),
            success_gamma=option_gamma,
            success_temperature=option_temperature,
            regression_weight=float(tcfg.get("teacher_pcd_direct_regression_weight", 1.0)),
            ranking_weight=float(tcfg.get("teacher_pcd_direct_ranking_weight", 2.5)),
            nominal_penalty_weight=float(tcfg.get("teacher_pcd_direct_nominal_penalty_weight", 1.0)),
            false_positive_weight=float(tcfg.get("teacher_pcd_direct_false_positive_weight", 1.5)),
            margin=float(tcfg.get("teacher_pcd_direct_margin", 0.18)),
            min_teacher_pcd_gain=float(tcfg.get("teacher_pcd_direct_min_teacher_pcd_gain", 0.015)),
            min_teacher_best_pcd=float(tcfg.get("teacher_pcd_direct_min_teacher_best_pcd", 0.50)),
            max_nominal_teacher_pcd=float(tcfg.get("teacher_pcd_direct_max_nominal_teacher_pcd", 0.68)),
            target_min_pred_pcd=float(tcfg.get("teacher_pcd_direct_target_min_pred_pcd", 0.52)),
            nominal_max_pred_pcd=float(tcfg.get("teacher_pcd_direct_nominal_max_pred_pcd", 0.50)),
            focus_non_nominal_weight=float(tcfg.get("teacher_pcd_direct_focus_non_nominal_weight", 2.0)),
            false_positive_margin=float(tcfg.get("teacher_pcd_direct_false_positive_margin", 0.03)),
            component_weight=float(tcfg.get("teacher_pcd_direct_component_weight", 0.0)),
            positive_component_weight=float(tcfg.get("teacher_pcd_direct_positive_component_weight", 0.0)),
            nominal_cap_weight=float(tcfg.get("teacher_pcd_direct_nominal_cap_weight", 1.0)),
            positive_rank_all_weight=float(tcfg.get("teacher_pcd_direct_positive_rank_all_weight", 0.0)),
            positive_floor_weight=float(tcfg.get("teacher_pcd_direct_positive_floor_weight", 0.0)),
            positive_min_pred_r_dep=float(tcfg.get("teacher_pcd_direct_positive_min_pred_r_dep", -1.0e9)),
            positive_max_pred_gap=float(tcfg.get("teacher_pcd_direct_positive_max_pred_gap", -1.0)),
            positive_min_pred_drs=float(tcfg.get("teacher_pcd_direct_positive_min_pred_drs", -1.0)),
        )
        loss_recovery_advantage = observation_consistent_recovery_advantage_loss(
            r_dep, gap, pred_q,
            batch["r_dep_star"].float(), batch["r_orc_star"].float(), teacher_q,
            batch["root_probs"].float(), batch["root_valid"], batch["option_valid"],
            batch["scene_hash"], batch["time_index"],
            batch.get("prefix_macro_type_id", batch.get("candidate_index", torch.zeros_like(batch["time_index"]))),
            batch["is_nominal"].float(),
            batch.get("bucket_id", torch.full_like(batch["time_index"], 3)),
            macro_ids=_parse_int_tuple(tcfg.get("recovery_advantage_macro_ids", "2,3,5,7"), (2, 3, 5, 7)),
            bucket_ids=_parse_int_tuple(tcfg.get("recovery_advantage_bucket_ids", "1,2"), (1, 2)),
            positive_gain=float(tcfg.get("recovery_advantage_positive_gain", 0.03)),
            negative_gain=float(tcfg.get("recovery_advantage_negative_gain", 0.03)),
            advantage_margin=float(tcfg.get("recovery_advantage_margin", 0.10)),
            regression_weight=float(tcfg.get("recovery_advantage_regression_weight", 1.0)),
            ranking_weight=float(tcfg.get("recovery_advantage_ranking_weight", 1.0)),
            component_inversion_weight=float(tcfg.get("recovery_advantage_component_weight", 0.5)),
            false_positive_weight=float(tcfg.get("recovery_advantage_false_positive_weight", 0.75)),
            nominal_failure_pcd_max=float(tcfg.get("recovery_advantage_nominal_failure_pcd_max", 0.20)),
            target_min_pred_pcd=float(tcfg.get("recovery_advantage_target_min_pred_pcd", 0.50)),
            nominal_max_pred_pcd=float(tcfg.get("recovery_advantage_nominal_max_pred_pcd", 0.48)),
            near_weight=float(tcfg.get("recovery_advantage_near_weight", 1.5)),
            contact_weight=float(tcfg.get("recovery_advantage_contact_weight", 1.0)),
            success_gamma=option_gamma,
            success_temperature=option_temperature,
        )
        def _direct_value_loss(
            pred_logit: torch.Tensor,
            pred_logvar: torch.Tensor,
            pred_opportunity_logit: torch.Tensor | None,
            *,
            direct_bucket_ids: tuple[int, ...] | None = None,
        ) -> torch.Tensor:
            return direct_uncertainty_recovery_value_loss(
                pred_logit, pred_logvar,
                batch["r_dep_star"].float(), batch["r_orc_star"].float(), teacher_q,
                batch["root_probs"].float(), batch["root_valid"], batch["option_valid"],
                batch["scene_hash"], batch["time_index"],
                batch.get("prefix_macro_type_id", batch.get("candidate_index", torch.zeros_like(batch["time_index"]))),
                batch["is_nominal"].float(),
                batch.get("bucket_id", torch.full_like(batch["time_index"], 3)),
                macro_ids=_parse_int_tuple(tcfg.get("direct_value_macro_ids", "2,3,5,7"), (2, 3, 5, 7)),
                bucket_ids=(direct_bucket_ids if direct_bucket_ids is not None else _parse_int_tuple(tcfg.get("direct_value_bucket_ids", "1,2"), (1, 2))),
                temperature=float(tcfg.get("direct_value_temperature", 0.12)),
                positive_gain=float(tcfg.get("direct_value_positive_gain", 0.03)),
                negative_gain=float(tcfg.get("direct_value_negative_gain", 0.02)),
                rank_margin=float(tcfg.get("direct_value_rank_margin", 0.04)),
                point_weight=float(tcfg.get("direct_value_point_weight", 0.15)),
                listwise_weight=float(tcfg.get("direct_value_listwise_weight", 0.35)),
                advantage_weight=float(tcfg.get("direct_value_advantage_weight", 1.0)),
                centered_weight=float(tcfg.get("direct_value_centered_weight", 1.0)),
                positive_group_weight=float(tcfg.get("direct_value_positive_group_weight", 4.0)),
                negative_group_weight=float(tcfg.get("direct_value_negative_group_weight", 1.0)),
                ambiguous_group_weight=float(tcfg.get("direct_value_ambiguous_group_weight", 0.25)),
                near_weight=float(tcfg.get("direct_value_near_weight", 1.5)),
                contact_weight=float(tcfg.get("direct_value_contact_weight", 1.0)),
                min_group_range=float(tcfg.get("direct_value_min_group_range", 0.01)),
                false_positive_weight=float(tcfg.get("direct_value_false_positive_weight", 1.0)),
                variance_floor=float(tcfg.get("direct_value_variance_floor", 0.0025)),
                output_mode=str(tcfg.get("direct_value_output_mode", model_cfg.get("direct_recovery_value_output", "probability"))),
                pairwise_weight=float(tcfg.get("direct_value_pairwise_weight", 0.0)),
                top_rank_weight=float(tcfg.get("direct_value_top_rank_weight", 0.0)),
                success_gamma=option_gamma,
                success_temperature=option_temperature,
                pred_opportunity_logit=pred_opportunity_logit,
                opportunity_weight=float(tcfg.get("direct_value_opportunity_weight", 0.0)),
                opportunity_pos_weight=float(tcfg.get("direct_value_opportunity_pos_weight", 6.0)),
            )

        zero_direct = r_dep.sum() * 0.0
        loss_direct_value_mixture = zero_direct
        loss_direct_value_near = zero_direct
        loss_direct_value_contact = zero_direct
        if "direct_recovery_value_logit" in out:
            loss_direct_value_mixture = _direct_value_loss(
                out["direct_recovery_value_logit"],
                out["direct_recovery_value_logvar"],
                out.get("direct_recovery_opportunity_logit"),
            )
            expert_supervision = bool(tcfg.get("direct_value_expert_supervision", False))
            all_expert = out.get("direct_expert_outputs")
            expert_bucket_ids = _parse_int_tuple(
                tcfg.get("direct_value_expert_bucket_ids", "1,2"), (1, 2)
            )
            if expert_supervision and all_expert is not None:
                expert_losses: list[torch.Tensor] = []
                for expert_idx, direct_bucket in enumerate(expert_bucket_ids):
                    if expert_idx >= int(all_expert.shape[1]):
                        break
                    expert_out = all_expert[:, expert_idx]
                    opp = expert_out[:, 2] if expert_out.shape[-1] >= 3 else None
                    expert_loss = _direct_value_loss(
                        expert_out[:, 0], expert_out[:, 1], opp,
                        direct_bucket_ids=(int(direct_bucket),),
                    )
                    expert_losses.append(expert_loss)
                    if int(direct_bucket) == 1:
                        loss_direct_value_near = expert_loss
                    elif int(direct_bucket) == 2:
                        loss_direct_value_contact = expert_loss
                if expert_losses:
                    loss_expert = torch.stack(expert_losses).mean()
                    mixture_weight = float(tcfg.get("direct_value_mixture_weight", 0.25))
                    loss_direct_value = loss_expert + mixture_weight * loss_direct_value_mixture
                else:
                    loss_direct_value = loss_direct_value_mixture
            else:
                loss_direct_value = loss_direct_value_mixture
        else:
            loss_direct_value = zero_direct
        loss_direct_value_worst = torch.maximum(loss_direct_value_near, loss_direct_value_contact)

        loss_direct_router_supervision = zero_direct
        direct_router_accuracy = zero_direct
        if "direct_expert_logits" in out and bool(tcfg.get("direct_router_supervision", False)):
            router_logits = out["direct_expert_logits"]
            bucket = batch.get("bucket_id", torch.full_like(batch["time_index"], 3)).long().reshape(-1)
            router_target = bucket - 1
            router_valid = (router_target >= 0) & (router_target < router_logits.shape[-1])
            if bool(router_valid.any()):
                loss_direct_router_supervision = F.cross_entropy(
                    router_logits[router_valid], router_target[router_valid]
                )
                direct_router_accuracy = (
                    router_logits[router_valid].argmax(dim=-1) == router_target[router_valid]
                ).float().mean()
        if "direct_expert_weights" in out:
            # Small unsupervised anti-collapse regularizer.  It does not teach a
            # regime label; it only prevents one observation-conditioned expert
            # from receiving all traffic before the value loss differentiates
            # the experts.  Per-sample weights remain free to be sharp.
            mean_expert_weight = out["direct_expert_weights"].mean(dim=0)
            target_expert_weight = torch.full_like(
                mean_expert_weight, 1.0 / float(mean_expert_weight.numel())
            )
            loss_direct_router_balance = F.mse_loss(
                mean_expert_weight, target_expert_weight
            )
        else:
            loss_direct_router_balance = r_dep.sum() * 0.0
        if bool((cfg.get("ablation", {}) or {}).get("without_anti_oracle", False)):
            loss_art = loss_art * 0.0
            loss_gap = loss_gap * 0.0
        loss_util = F.smooth_l1_loss(out["utility"], batch["utility"].float())
        total = (
            float(lw.get("assign", 1.0)) * loss_root
            + float(lw.get("margin", 2.0)) * loss_margin
            + float(lw.get("sig", 0.5)) * (loss_sig + loss_future_sig)
            + float(lw.get("obs", 1.0)) * loss_obs
            + float(lw.get("dep", 0.5)) * loss_dep
            + float(lw.get("orc", 0.5)) * loss_orc
            + float(lw.get("anti_oracle", 1.0)) * loss_art
            + float(lw.get("artifact_gap", 0.5)) * loss_gap
            + float(lw.get("admission", 0.2)) * loss_admit
            + float(lw.get("option_q", 0.5)) * loss_option_q
            + float(lw.get("option_admission", 0.4)) * loss_option_admit
            + float(lw.get("option_success", 0.0)) * loss_option_success
            + float(lw.get("option_success_bce", 0.0)) * loss_option_success_bce
            + float(lw.get("option_best", 0.2)) * loss_option_best
            + float(lw.get("group_ranking", 0.0)) * loss_group_rank
            + float(lw.get("group_ce", 0.0)) * loss_group_ce
            + float(lw.get("nominal_switch", 0.0)) * loss_nominal_switch
            + float(lw.get("group_distill", 0.0)) * loss_group_distill
            + float(lw.get("safe_nominal", 0.0)) * loss_safe_nominal
            + float(lw.get("protective_macro", 0.0)) * loss_protective_macro
            + float(lw.get("macro_drs", 0.0)) * loss_macro_drs
            + float(lw.get("ddc", 0.0)) * loss_ddc
            + float(lw.get("teacher_pcd_direct", 0.0)) * loss_teacher_pcd_direct
            + float(lw.get("recovery_advantage", 0.0)) * loss_recovery_advantage
            + float(lw.get("direct_recovery_value", 0.0)) * loss_direct_value
            + float(lw.get("direct_router_balance", 0.0)) * loss_direct_router_balance
            + float(lw.get("direct_router_supervision", 0.0)) * loss_direct_router_supervision
            + float(lw.get("utility", 0.2)) * loss_util
        )
        if training:
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            grad_clip = float(tcfg.get("grad_clip", 5.0))
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        bsz = int(batch["x"].shape[0])
        n += bsz
        vals = {
            "loss": total.item(),
            "loss_root": loss_root.item(),
            "loss_margin": loss_margin.item(),
            "loss_sig": loss_sig.item(),
            "loss_future_sig": loss_future_sig.item(),
            "loss_obs": loss_obs.item(),
            "loss_dep": loss_dep.item(),
            "loss_orc": loss_orc.item(),
            "loss_art": loss_art.item(),
            "loss_gap": loss_gap.item(),
            "loss_admission": loss_admit.item(),
            "loss_option_q": loss_option_q.item(),
            "loss_option_admission": loss_option_admit.item(),
            "loss_option_success": loss_option_success.item(),
            "loss_option_success_bce": loss_option_success_bce.item(),
            "loss_option_best": loss_option_best.item(),
            "loss_group_ranking": loss_group_rank.item(),
            "loss_group_ce": loss_group_ce.item(),
            "loss_nominal_switch": loss_nominal_switch.item(),
            "loss_group_distill": loss_group_distill.item(),
            "loss_safe_nominal": loss_safe_nominal.item(),
            "loss_protective_macro": loss_protective_macro.item(),
            "loss_macro_drs": loss_macro_drs.item(),
            "loss_ddc": loss_ddc.item(),
            "loss_teacher_pcd_direct": loss_teacher_pcd_direct.item(),
            "loss_recovery_advantage": loss_recovery_advantage.item(),
            "loss_direct_recovery_value": loss_direct_value.item(),
            "loss_direct_recovery_value_mixture": loss_direct_value_mixture.item(),
            "loss_direct_recovery_value_near": loss_direct_value_near.item(),
            "loss_direct_recovery_value_contact": loss_direct_value_contact.item(),
            "loss_direct_recovery_value_worst": loss_direct_value_worst.item(),
            "loss_direct_router_balance": loss_direct_router_balance.item(),
            "loss_direct_router_supervision": loss_direct_router_supervision.item(),
            "direct_router_accuracy": direct_router_accuracy.item(),
            "loss_utility": loss_util.item(),
            "pred_r_dep_mean": r_dep.mean().item(),
            "teacher_r_dep_mean": batch["r_dep_star"].float().mean().item(),
        }
        for k, v in vals.items():
            totals[k] = totals.get(k, 0.0) + float(v) * bsz
    metrics = {k: float(v / max(n, 1)) for k, v in totals.items()}
    # Checkpoint selection must not hide a failing regime behind the mixed
    # validation average.  Recompute the worst expert loss from the epoch-level
    # Near/Contact aggregates (rather than averaging per-batch maxima).
    metrics["loss_direct_recovery_value_worst"] = max(
        float(metrics.get("loss_direct_recovery_value_near", 0.0)),
        float(metrics.get("loss_direct_recovery_value_contact", 0.0)),
    )
    return metrics | {"num_samples": int(n), "num_batches": int(len(loader))}



class SceneTimeBatchSampler(Sampler[list[int]]):
    """Batch sampler that keeps scene-time candidate sets together.

    Group-wise ranking losses need multiple candidates from the same scene-time
    in the same mini-batch.  Standard random sampling destroys that structure.
    """

    def __init__(self, groups: list[list[int]], batch_size: int, *, group_weights: list[float] | None = None, replacement: bool = True, shuffle_within_group: bool = True, shuffle_groups: bool = True):
        self.groups = [list(g) for g in groups if g]
        self.batch_size = max(1, int(batch_size))
        self.replacement = bool(replacement)
        self.shuffle_within_group = bool(shuffle_within_group)
        self.shuffle_groups = bool(shuffle_groups)
        if group_weights is None or len(group_weights) != len(self.groups):
            self.group_weights = torch.ones((len(self.groups),), dtype=torch.double)
        else:
            self.group_weights = torch.as_tensor(group_weights, dtype=torch.double).clamp_min(1.0e-8)

    def __len__(self) -> int:
        if not self.groups:
            return 0
        total = sum(len(g) for g in self.groups)
        return max(1, int(np.ceil(total / float(self.batch_size))))

    def __iter__(self):
        if not self.groups:
            return
        if self.replacement:
            order = torch.multinomial(self.group_weights, num_samples=len(self.groups), replacement=True).tolist()
        elif self.shuffle_groups:
            order = torch.randperm(len(self.groups)).tolist()
        else:
            order = list(range(len(self.groups)))
        batch: list[int] = []
        for gi in order:
            inds = list(self.groups[int(gi)])
            if self.shuffle_within_group and len(inds) > 1:
                perm = torch.randperm(len(inds)).tolist()
                inds = [inds[i] for i in perm]
            if len(batch) + len(inds) > self.batch_size and batch:
                yield batch
                batch = []
            if len(inds) > self.batch_size:
                for j in range(0, len(inds), self.batch_size):
                    chunk = inds[j : j + self.batch_size]
                    if chunk:
                        yield chunk
            else:
                batch.extend(inds)
        if batch:
            yield batch


def _sampler_weight_for_path(p: Path, cfg: dict, root_counts: Counter, total: int) -> tuple[float, bool, bool, bool]:
    tcfg = cfg.get("training", {}) if isinstance(cfg.get("training", {}), dict) else {}
    weight_art = float(tcfg.get("artifact_sampler_weight", 0.25))
    weight_neg = float(tcfg.get("negative_deployable_sampler_weight", 0.75))
    weight_safe_pos = float(tcfg.get("safe_positive_sampler_weight", 0.25))
    regime_balance_power = float(tcfg.get("regime_balance_power", 0.0))
    try:
        is_art = float(np.asarray(scalar_metadata_for_path(p, "i_art_star", 0)).item()) > 0.5
        r_dep = float(np.asarray(scalar_metadata_for_path(p, "r_dep_star", 0)).item())
        is_neg = r_dep < 0.0
        root_name = _dataset_root_name(p).lower()
        is_safe_pos = ("safe" in root_name) and (not is_neg) and (not is_art)
        w = 1.0
        if regime_balance_power > 0:
            root_count = max(1, int(root_counts.get(_dataset_root_name(p), 1)))
            w *= float((total / max(len(root_counts), 1) / root_count) ** regime_balance_power)
        if is_art:
            w += weight_art
        if is_neg:
            w += weight_neg
        if is_safe_pos:
            w += weight_safe_pos
        return float(w), bool(is_art), bool(is_neg), bool(is_safe_pos)
    except Exception:
        return 1.0, False, False, False


def _make_group_batch_sampler(ds: OCRAPSampleDataset, cfg: dict, batch_size: int) -> SceneTimeBatchSampler | None:
    tcfg = cfg.get("training", {}) if isinstance(cfg.get("training", {}), dict) else {}
    if not bool(tcfg.get("group_batching", False)):
        return None
    total = len(ds.paths)
    roots = [_dataset_root_name(p) for p in ds.paths]
    root_counts = Counter(roots)
    groups_by_key: dict[tuple[int, int, int], list[int]] = {}
    sample_weights: list[float] = []
    num_artifacts = num_negative = num_safe_pos = 0
    for i, p in enumerate(ds.paths):
        try:
            scene = scalar_metadata_for_path(p, "scene_id", "")
            t = int(np.asarray(scalar_metadata_for_path(p, "time_index", 0)).item())
            key = (bucket_id_for_path(p), stable_scene_hash(scene), t)
        except Exception:
            key = (3, i, 0)
        groups_by_key.setdefault(key, []).append(i)
        w, is_art, is_neg, is_safe_pos = _sampler_weight_for_path(p, cfg, root_counts, total)
        sample_weights.append(w)
        num_artifacts += int(is_art)
        num_negative += int(is_neg)
        num_safe_pos += int(is_safe_pos)
    groups = list(groups_by_key.values())
    hard_macro_ids = set(_parse_int_tuple(tcfg.get("group_batch_hard_macro_ids", ""), ()))
    hard_bucket_ids = set(_parse_int_tuple(tcfg.get("group_batch_hard_bucket_ids", ""), ()))
    hard_r_dep_min = float(tcfg.get("group_batch_hard_min_r_dep", 0.35))
    hard_boost = float(tcfg.get("group_batch_hard_boost", 0.0))

    # v43: oversample groups that contain a recovery candidate whose teacher
    # deployable value improves over the group's nominal candidate. v42 boosted
    # high absolute r_dep, which included many groups where nominal was equally
    # good or better and therefore diluted the rare positive-advantage signal.
    positive_macro_ids = set(_parse_int_tuple(tcfg.get("group_batch_positive_macro_ids", ""), ()))
    positive_bucket_ids = set(_parse_int_tuple(tcfg.get("group_batch_positive_bucket_ids", ""), ()))
    positive_gain_min = float(tcfg.get("group_batch_positive_r_dep_gain", 0.025))
    positive_boost = float(tcfg.get("group_batch_positive_advantage_boost", 0.0))

    group_weights = []
    hard_groups = 0
    positive_advantage_groups = 0
    for g in groups:
        gw = float(max(sample_weights[i] for i in g))
        if hard_boost > 0.0 and hard_macro_ids:
            is_hard = False
            for i in g:
                p = ds.paths[i]
                try:
                    mac = int(float(np.asarray(scalar_metadata_for_path(p, "prefix_macro_type_id", scalar_metadata_for_path(p, "prefix_macro_id", -1))).item()))
                    bid = bucket_id_for_path(p)
                    rdep = float(np.asarray(scalar_metadata_for_path(p, "r_dep_star", -99.0)).item())
                    if mac in hard_macro_ids and (not hard_bucket_ids or bid in hard_bucket_ids) and rdep >= hard_r_dep_min:
                        is_hard = True
                        break
                except Exception:
                    continue
            if is_hard:
                gw *= hard_boost
                hard_groups += 1

        if positive_boost > 0.0 and positive_macro_ids:
            nominal_rdep = None
            best_recovery_rdep = None
            for i in g:
                p = ds.paths[i]
                try:
                    bid = bucket_id_for_path(p)
                    if positive_bucket_ids and bid not in positive_bucket_ids:
                        continue
                    is_nominal = bool(float(np.asarray(scalar_metadata_for_path(p, "is_nominal", 0.0)).item()) > 0.5)
                    rdep = float(np.asarray(scalar_metadata_for_path(p, "r_dep_star", -99.0)).item())
                    if is_nominal:
                        nominal_rdep = rdep if nominal_rdep is None else max(nominal_rdep, rdep)
                        continue
                    mac = int(float(np.asarray(scalar_metadata_for_path(p, "prefix_macro_type_id", scalar_metadata_for_path(p, "prefix_macro_id", -1))).item()))
                    if mac in positive_macro_ids:
                        best_recovery_rdep = rdep if best_recovery_rdep is None else max(best_recovery_rdep, rdep)
                except Exception:
                    continue
            if nominal_rdep is not None and best_recovery_rdep is not None and (best_recovery_rdep - nominal_rdep) >= positive_gain_min:
                gw *= positive_boost
                positive_advantage_groups += 1
        group_weights.append(gw)
    print({
        "event": "group_batch_sampler_stats",
        "num_groups": int(len(groups)),
        "num_samples": int(total),
        "mean_group_size": float(np.mean([len(g) for g in groups])) if groups else 0.0,
        "max_group_size": int(max([len(g) for g in groups], default=0)),
        "replacement": bool(tcfg.get("group_batching_replacement", True)),
        "num_artifacts": int(num_artifacts),
        "num_negative_deployable": int(num_negative),
        "num_safe_positive": int(num_safe_pos),
        "hard_group_boost": float(hard_boost),
        "hard_groups": int(hard_groups),
        "positive_advantage_boost": float(positive_boost),
        "positive_advantage_gain_min": float(positive_gain_min),
        "positive_advantage_groups": int(positive_advantage_groups),
    }, flush=True)
    return SceneTimeBatchSampler(groups, batch_size, group_weights=group_weights, replacement=bool(tcfg.get("group_batching_replacement", True)))

def _make_sampler(ds: OCRAPSampleDataset, cfg: dict) -> WeightedRandomSampler | None:
    tcfg = cfg.get("training", {}) if isinstance(cfg.get("training", {}), dict) else {}
    weight_art = float(tcfg.get("artifact_sampler_weight", 0.25))
    weight_neg = float(tcfg.get("negative_deployable_sampler_weight", 0.75))
    weight_safe_pos = float(tcfg.get("safe_positive_sampler_weight", 0.25))
    regime_balance_power = float(tcfg.get("regime_balance_power", 0.0))
    if max(weight_art, weight_neg, weight_safe_pos, regime_balance_power) <= 0:
        return None
    weights = []
    num_artifacts = 0
    num_negative = 0
    num_safe_pos = 0
    total = len(ds.paths)
    roots = [_dataset_root_name(p) for p in ds.paths]
    root_counts = Counter(roots)
    print({
        "event": "sampler_weight_config",
        "artifact_sampler_weight": weight_art,
        "negative_deployable_sampler_weight": weight_neg,
        "safe_positive_sampler_weight": weight_safe_pos,
        "regime_balance_power": regime_balance_power,
        "root_counts": dict(sorted(root_counts.items())),
    }, flush=True)
    for idx, p in enumerate(ds.paths, 1):
        if idx == 1 or idx % 5000 == 0 or idx == total:
            print({"event": "sampler_scan_progress", "seen": idx, "total": total}, flush=True)
        try:
            is_art = float(np.asarray(scalar_metadata_for_path(p, "i_art_star", 0)).item()) > 0.5
            r_dep = float(np.asarray(scalar_metadata_for_path(p, "r_dep_star", 0)).item())
            is_neg = r_dep < 0.0
            root_name = _dataset_root_name(p).lower()
            is_safe_pos = ("safe" in root_name) and (not is_neg) and (not is_art)
            num_artifacts += int(is_art)
            num_negative += int(is_neg)
            num_safe_pos += int(is_safe_pos)
            w = 1.0
            if regime_balance_power > 0:
                root_count = max(1, int(root_counts.get(_dataset_root_name(p), 1)))
                w *= float((total / max(len(root_counts), 1) / root_count) ** regime_balance_power)
            if is_art:
                w += weight_art
            if is_neg:
                w += weight_neg
            if is_safe_pos:
                w += weight_safe_pos
            weights.append(w)
        except Exception:
            weights.append(1.0)
    print({
        "event": "sampler_scan_stats",
        "num_artifacts": int(num_artifacts),
        "artifact_fraction": float(num_artifacts / max(total, 1)),
        "num_negative_deployable": int(num_negative),
        "negative_deployable_fraction": float(num_negative / max(total, 1)),
        "num_safe_positive": int(num_safe_pos),
        "safe_positive_fraction": float(num_safe_pos / max(total, 1)),
    }, flush=True)
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=len(weights), replacement=True)


def train(dataset: str, output: str, cfg: dict, val_dataset: str | None = None) -> dict:
    seed_everything(int(cfg.get("seed", 7)))
    out = ensure_dir(output)
    print({"event": "dataset_scan_start", "dataset": str(dataset)}, flush=True)
    paths = iter_sample_paths_many(dataset)
    print({"event": "dataset_scan_done", "num_npz_paths": len(paths)}, flush=True)
    if not paths:
        raise ValueError(f"No OC-RAP sample .npz files found under {dataset}")
    tcfg_for_split = cfg.get("training", {}) if isinstance(cfg.get("training", {}), dict) else {}
    explicit_val_dataset = val_dataset or tcfg_for_split.get("val_dataset") or tcfg_for_split.get("validation_dataset")
    print({"event": "split_scan_start", "splits": ["train", "val"], "explicit_val_dataset": bool(explicit_val_dataset)}, flush=True)
    train_paths = split_paths_by_npz_split(paths, "train")
    if explicit_val_dataset:
        val_all_paths = iter_sample_paths_many(str(explicit_val_dataset))
        val_paths = split_paths_by_npz_split(val_all_paths, {"val", "calibration"})
        if not val_paths:
            val_paths = val_all_paths
    else:
        val_paths = split_paths_by_npz_split(paths, "val")
    print({"event": "split_scan_done", "num_train_paths": len(train_paths), "num_val_paths": len(val_paths)}, flush=True)
    profile_max = int(tcfg_for_split.get("dataset_profile_max_scalar_scan", 0))
    if bool(tcfg_for_split.get("dataset_profile", True)):
        print(_profile_paths(train_paths, stage="train", max_scalar_scan=profile_max), flush=True)
        print(_profile_paths(val_paths, stage="val", max_scalar_scan=profile_max), flush=True)
    if not train_paths:
        train_paths = paths
    if not val_paths:
        print({"event": "validation_fallback_warning", "reason": "no explicit validation split; using first 10pct of training paths"}, flush=True)
        val_paths = train_paths[: max(1, min(len(train_paths), len(train_paths) // 10 or 1))]

    train_ds = OCRAPSampleDataset(train_paths, cfg)
    val_ds = OCRAPSampleDataset(val_paths, cfg)
    first = train_ds[0]
    num_roots = int(train_ds.num_roots)
    num_options = int(train_ds.num_options)
    d_signature = int(train_ds.d_signature)
    d_future_signature = int(train_ds.d_future_signature)
    device = _device(cfg)
    device_info = _device_summary(device)
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
    d_model = int(model_cfg.get("d_model", 128))
    d_obs = int(model_cfg.get("d_obs", 64))
    tau_obs = float(model_cfg.get("tau_obs", (cfg.get("ocmero", {}) or {}).get("tau_obs", 1.0)))
    encoder_type = str(model_cfg.get("encoder_type", "mlp"))
    feature_layout = {
        "prefix_param_dim": int(cfg.get("prefix_param_dim", 5)),
        "num_macros": int(model_cfg.get("num_macros", 16)),
        "prefix_flat_dim": int(model_cfg.get("feature_prefix_flat_dim", 80)),
        "control_flat_dim": int(model_cfg.get("feature_control_flat_dim", 40)),
        "feature_max_agents": int(model_cfg.get("feature_max_agents", 32)),
        "bev_channels": int(cfg.get("bev_channels", 7)),
        "route_flat_dim": int(model_cfg.get("feature_route_flat_dim", 64)),
        "map_flat_dim": int(model_cfg.get("feature_map_flat_dim", 64)),
        "dyn_flat_dim": int(model_cfg.get("feature_dynamic_map_flat_dim", 32)),
    }
    model = OCRAPModel(
        train_ds.feature_dim,
        num_roots=num_roots,
        num_options=num_options,
        d_model=d_model,
        d_obs=d_obs,
        tau_obs=tau_obs,
        encoder_type=encoder_type,
        feature_layout=feature_layout,
        num_layers=int(model_cfg.get("transformer_layers", 2)),
        num_heads=int(model_cfg.get("transformer_heads", 4)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        d_signature=d_signature,
        d_future_signature=d_future_signature,
        option_feature_dim=OPTION_FEATURE_DIM,
        direct_recovery_value_head=bool(model_cfg.get("direct_recovery_value_head", False)),
        direct_recovery_value_pooling=str(model_cfg.get("direct_recovery_value_pooling", "scene")),
        direct_recovery_value_output=str(model_cfg.get("direct_recovery_value_output", "probability")),
        direct_recovery_value_regime_conditioning=bool(model_cfg.get("direct_recovery_value_regime_conditioning", False)),
        direct_recovery_value_num_regimes=int(model_cfg.get("direct_recovery_value_num_regimes", 4)),
        direct_recovery_value_regime_dim=int(model_cfg.get("direct_recovery_value_regime_dim", 16)),
        direct_recovery_opportunity_head=bool(model_cfg.get("direct_recovery_opportunity_head", False)),
        direct_recovery_value_experts=bool(model_cfg.get("direct_recovery_value_experts", False)),
        direct_recovery_value_num_experts=int(model_cfg.get("direct_recovery_value_num_experts", 2)),
        direct_recovery_value_expert_routing=str(model_cfg.get("direct_recovery_value_expert_routing", "bucket")),
        direct_recovery_value_router_temperature=float(model_cfg.get("direct_recovery_value_router_temperature", 1.0)),
        direct_recovery_value_router_pooling=str(model_cfg.get("direct_recovery_value_router_pooling", "candidate")),
    ).to(device)
    tcfg = cfg.get("training", {}) if isinstance(cfg.get("training", {}), dict) else {}

    # v28: residual DDC should fine-tune an already calibrated selector model
    # rather than relearning all heads from scratch.  The v27 trained run showed
    # that scratch training can minimize train loss while destroying the runtime
    # calibration scale used by the selector.
    init_checkpoint = str(tcfg.get("init_checkpoint", "") or "").strip()
    init_load_info: dict[str, object] = {}
    if init_checkpoint:
        init_path = Path(init_checkpoint)
        if not init_path.exists():
            raise FileNotFoundError(f"training.init_checkpoint does not exist: {init_checkpoint}")
        ckpt = torch.load(init_path, map_location=device)
        state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        init_load_info = {
            "event": "init_checkpoint_loaded",
            "path": str(init_path),
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
        }
        print(init_load_info, flush=True)

    freeze_prefixes = tuple(
        x.strip() for x in str(tcfg.get("freeze_param_prefixes", "") or "").split(",") if x.strip()
    )
    if freeze_prefixes:
        frozen = 0
        trainable = 0
        for name, param in model.named_parameters():
            if any(name.startswith(prefix) for prefix in freeze_prefixes):
                param.requires_grad_(False)
                frozen += int(param.numel())
            else:
                trainable += int(param.numel())
        print({
            "event": "freeze_param_prefixes",
            "prefixes": list(freeze_prefixes),
            "frozen_params": int(frozen),
            "trainable_params": int(trainable),
        }, flush=True)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters remain after training.freeze_param_prefixes")
    opt = torch.optim.AdamW(trainable_params, lr=float(tcfg.get("lr", 1e-3)), weight_decay=float(tcfg.get("weight_decay", 1e-4)))
    batch_size = int(tcfg.get("batch_size", 32))
    num_workers = int(tcfg.get("num_workers", 0))
    group_batch_sampler = _make_group_batch_sampler(train_ds, cfg, batch_size)
    if group_batch_sampler is not None:
        print({"event": "sampler_scan_done", "sampler": "scene_time_group_batch"}, flush=True)
        train_loader = DataLoader(train_ds, batch_sampler=group_batch_sampler, num_workers=num_workers, collate_fn=_collate, pin_memory=(device.type == "cuda"))
    else:
        print({"event": "sampler_scan_start", "artifact_sampler_weight": float(tcfg.get("artifact_sampler_weight", 0.25))}, flush=True)
        sampler = _make_sampler(train_ds, cfg)
        print({"event": "sampler_scan_done", "sampler": "weighted" if sampler is not None else "none"}, flush=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=sampler is None, sampler=sampler, num_workers=num_workers, collate_fn=_collate, pin_memory=(device.type == "cuda"))
    if bool(tcfg.get("group_batching", False)):
        val_cfg = dict(cfg)
        val_tcfg = dict(tcfg)
        val_tcfg["group_batching_replacement"] = False
        val_tcfg["group_batch_hard_boost"] = 0.0
        val_cfg["training"] = val_tcfg
        val_group_sampler = _make_group_batch_sampler(val_ds, val_cfg, batch_size)
        if val_group_sampler is not None:
            val_group_sampler.shuffle_within_group = False
            val_group_sampler.shuffle_groups = False
            val_loader = DataLoader(val_ds, batch_sampler=val_group_sampler, num_workers=num_workers, collate_fn=_collate, pin_memory=(device.type == "cuda"))
        else:
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=_collate, pin_memory=(device.type == "cuda"))
    else:
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=_collate, pin_memory=(device.type == "cuda"))
    epochs = int(tcfg.get("epochs", 10))
    best_val = float("inf")
    best_epoch = 0
    no_improve_epochs = 0
    history = []
    best_path = out / "best.pt"
    latest_path = out / "latest.pt"
    ckpt_dir = ensure_dir(out / "checkpoints")

    def _checkpoint_payload(ep: int, val_loss: float) -> dict:
        return {
            "cfg": cfg,
            "input_dim": train_ds.feature_dim,
            "num_roots": num_roots,
            "num_options": num_options,
            "d_model": d_model,
            "d_obs": d_obs,
            "tau_obs": tau_obs,
            "encoder_type": encoder_type,
            "feature_layout": feature_layout,
            "d_signature": d_signature,
            "d_future_signature": d_future_signature,
            "option_feature_dim": OPTION_FEATURE_DIM,
            "direct_recovery_value_head": bool(model_cfg.get("direct_recovery_value_head", False)),
            "direct_recovery_value_pooling": str(model_cfg.get("direct_recovery_value_pooling", "scene")),
            "direct_recovery_value_output": str(model_cfg.get("direct_recovery_value_output", "probability")),
            "direct_recovery_value_regime_conditioning": bool(model_cfg.get("direct_recovery_value_regime_conditioning", False)),
            "direct_recovery_value_num_regimes": int(model_cfg.get("direct_recovery_value_num_regimes", 4)),
            "direct_recovery_value_regime_dim": int(model_cfg.get("direct_recovery_value_regime_dim", 16)),
            "direct_recovery_opportunity_head": bool(model_cfg.get("direct_recovery_opportunity_head", False)),
            "direct_recovery_value_experts": bool(model_cfg.get("direct_recovery_value_experts", False)),
            "direct_recovery_value_num_experts": int(model_cfg.get("direct_recovery_value_num_experts", 2)),
            "direct_recovery_value_expert_routing": str(model_cfg.get("direct_recovery_value_expert_routing", "bucket")),
            "direct_recovery_value_router_temperature": float(model_cfg.get("direct_recovery_value_router_temperature", 1.0)),
            "direct_recovery_value_router_pooling": str(model_cfg.get("direct_recovery_value_router_pooling", "candidate")),
            "model_state": model.state_dict(),
            "optimizer_state": opt.state_dict(),
            "epoch": int(ep),
            "val_loss": float(val_loss),
            "device_info_at_train": device_info,
            "note": "OC-RAP neural checkpoint: predicts root probabilities, recovery margins, utility, and observation compatibility from scene-prefix features.",
            "init_checkpoint": init_checkpoint,
            "freeze_param_prefixes": list(freeze_prefixes),
        }
    print({
        "event": "train_start",
        **device_info,
        "num_train_samples": len(train_paths),
        "num_val_samples": len(val_paths),
        "train_batches_per_epoch": len(train_loader),
        "val_batches_per_epoch": len(val_loader),
        "epochs": epochs,
        "batch_size": batch_size,
        "d_model": d_model,
        "d_obs": d_obs,
        "encoder_type": encoder_type,
        "init_checkpoint": init_checkpoint or None,
        "freeze_param_prefixes": list(freeze_prefixes),
    }, flush=True)
    t0 = perf_counter()
    for ep in range(1, epochs + 1):
        ep_t0 = perf_counter()
        tr = _epoch(model, train_loader, cfg, device, opt, stage="train", epoch=ep)
        with torch.no_grad():
            va = _epoch(model, val_loader, cfg, device, None, stage="val", epoch=ep)
        row = {"epoch": ep, "train": tr, "val": va, "seconds": float(perf_counter() - ep_t0)}
        history.append(row)
        best_metric_name = str(tcfg.get("best_metric", "loss") or "loss")
        best_metric_mode = str(tcfg.get("best_metric_mode", "min") or "min").lower()
        current_metric = float(va.get(best_metric_name, va.get("loss", float("inf"))))
        compare_metric = current_metric if best_metric_mode != "max" else -current_metric
        improved = compare_metric <= best_val
        payload = _checkpoint_payload(ep, current_metric)
        save_every = bool(tcfg.get("save_every_epoch", True))
        if save_every:
            torch.save(payload, ckpt_dir / f"epoch_{ep:04d}.pt")
        if bool(tcfg.get("save_latest", True)):
            torch.save(payload, latest_path)
        if improved:
            best_val = compare_metric
            best_epoch = ep
            no_improve_epochs = 0
            payload["val_loss"] = float(va.get("loss", current_metric))
            payload["best_metric_name"] = best_metric_name
            payload["best_metric_value"] = float(current_metric)
            payload["is_best"] = True
            torch.save(payload, best_path)
        else:
            no_improve_epochs += 1
        print({
            "event": "epoch_end",
            "epoch": ep,
            "train_loss": round(float(tr.get("loss", 0.0)), 6),
            "val_loss": round(float(va.get("loss", 0.0)), 6),
            "best_val_loss": round(float(best_val), 6),
            "best_epoch": int(best_epoch),
            "improved": bool(improved),
            "seconds": round(float(row["seconds"]), 2),
        }, flush=True)
        patience = int(tcfg.get("early_stop_patience", 0) or 0)
        if patience > 0 and no_improve_epochs >= patience:
            print({
                "event": "early_stop",
                "epoch": ep,
                "best_epoch": int(best_epoch),
                "best_val_loss": float(best_val),
        "best_metric": str(tcfg.get("best_metric", "loss") or "loss"),
                "epochs_completed": int(len(history)),
                "patience": patience,
            }, flush=True)
            break
    result = {
        "checkpoint": str(best_path),
        "latest_checkpoint": str(latest_path),
        "checkpoint_dir": str(ckpt_dir),
        "num_train_samples": len(train_paths),
        "num_val_samples": len(val_paths),
        "input_dim": train_ds.feature_dim,
        "num_roots": num_roots,
        "num_options": num_options,
        "d_model": d_model,
        "d_obs": d_obs,
        "encoder_type": encoder_type,
        "init_checkpoint": init_checkpoint,
        "freeze_param_prefixes": list(freeze_prefixes),
        "best_val_loss": float(best_val),
        "best_metric": str(tcfg.get("best_metric", "loss") or "loss"),
        "device_info": device_info,
        "train_batches_per_epoch": len(train_loader),
        "val_batches_per_epoch": len(val_loader),
        "best_epoch": int(best_epoch),
        "epochs_completed": int(len(history)),
        "total_train_steps": int(len(train_loader) * len(history)),
        "elapsed_seconds": float(perf_counter() - t0),
        "history": history,
    }
    write_json(result, out / "train_summary.json")
    return result
