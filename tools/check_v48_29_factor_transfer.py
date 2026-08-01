#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def _load(path: Path) -> dict[str, Any]:
    doc = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(doc, dict) or not isinstance(doc.get("model_state"), dict):
        raise ValueError(f"invalid checkpoint: {path}")
    return doc


def _keys(state: dict[str, Any], prefix: str) -> list[str]:
    return sorted(k for k in state if k.startswith(prefix))


def _max_diff(a: dict[str, Any], b: dict[str, Any], keys: list[str]) -> float:
    values = []
    for key in keys:
        if key in a and key in b and tuple(a[key].shape) == tuple(b[key].shape):
            values.append(float((a[key].float() - b[key].float()).abs().max().item()))
    return max(values, default=0.0)


def _norm(state: dict[str, Any], keys: list[str]) -> float:
    return float(sum(float(state[k].float().square().sum().item()) for k in keys) ** 0.5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--factor", type=Path, required=True)
    ap.add_argument("--final", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--min-factor-norm", type=float, default=1.0e-6)
    ap.add_argument("--frozen-tolerance", type=float, default=0.0)
    args = ap.parse_args()

    source = _load(args.source)
    factor = _load(args.factor)
    final = _load(args.final)
    ss, fs, zs = source["model_state"], factor["model_state"], final["model_state"]
    benefit_prefix = "direct_evidence_concord_benefit_calibrator"
    harm_prefix = "direct_evidence_concord_harm_calibrator"
    admission_prefix = "direct_evidence_concord_admission_calibrator"
    benefit_keys = _keys(fs, benefit_prefix)
    harm_keys = _keys(fs, harm_prefix)
    admission_keys = _keys(zs, admission_prefix)
    missing_final = [k for k in benefit_keys + harm_keys if k not in zs]
    factor_to_final_diff = _max_diff(fs, zs, benefit_keys + harm_keys)
    factor_norm = _norm(fs, benefit_keys + harm_keys)
    admission_norm = _norm(zs, admission_keys)
    source_overlap = [k for k in benefit_keys + harm_keys if k in ss and tuple(ss[k].shape) == tuple(fs[k].shape)]
    source_to_factor_diff = _max_diff(ss, fs, source_overlap)
    factor_epoch = int(factor.get("epoch", -1))
    final_epoch = int(final.get("epoch", -1))
    valid = (
        factor_epoch > 0
        and final_epoch > 0
        and bool(benefit_keys)
        and bool(harm_keys)
        and bool(admission_keys)
        and not missing_final
        and factor_norm > float(args.min_factor_norm)
        and factor_to_final_diff <= float(args.frozen_tolerance)
        and admission_norm > float(args.min_factor_norm)
    )
    result = {
        "valid": valid,
        "source": str(args.source),
        "factor": str(args.factor),
        "final": str(args.final),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "factor_sha256": hashlib.sha256(args.factor.read_bytes()).hexdigest(),
        "final_sha256": hashlib.sha256(args.final.read_bytes()).hexdigest(),
        "factor_epoch": factor_epoch,
        "final_epoch": final_epoch,
        "benefit_parameter_count": len(benefit_keys),
        "harm_parameter_count": len(harm_keys),
        "admission_parameter_count": len(admission_keys),
        "factor_parameter_norm": factor_norm,
        "admission_parameter_norm": admission_norm,
        "source_overlap_parameter_count": len(source_overlap),
        "source_to_factor_max_abs_diff": source_to_factor_diff,
        "factor_to_final_frozen_max_abs_diff": factor_to_final_diff,
        "missing_factor_parameters_in_final": missing_final,
        "failure_reasons": [],
    }
    if factor_epoch <= 0:
        result["failure_reasons"].append("factor_checkpoint_is_epoch0")
    if final_epoch <= 0:
        result["failure_reasons"].append("admission_checkpoint_is_epoch0")
    if not benefit_keys or not harm_keys:
        result["failure_reasons"].append("factor_heads_missing")
    if missing_final:
        result["failure_reasons"].append("factor_heads_missing_from_final")
    if factor_to_final_diff > float(args.frozen_tolerance):
        result["failure_reasons"].append("factor_heads_changed_during_admission_stage")
    if admission_norm <= float(args.min_factor_norm):
        result["failure_reasons"].append("admission_head_not_trained")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if not valid:
        raise SystemExit(30)


if __name__ == "__main__":
    main()
