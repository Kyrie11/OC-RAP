#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def btext(x: str) -> bool:
    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "on"}: return True
    if s in {"0", "false", "no", "off"}: return False
    raise argparse.ArgumentTypeError(x)


def _csv(text: object) -> list[float]:
    if text is None: return []
    s = str(text).strip()
    if s.lower() in {"", "none", "null", "~"}: return []
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _same(a: list[float], b: list[float], tol: float = 1e-9) -> bool:
    return len(a) == len(b) and all(math.isfinite(x) and math.isfinite(y) and abs(x-y) <= tol * max(1.0, abs(x), abs(y)) for x,y in zip(a,b))


def main() -> int:
    ap = argparse.ArgumentParser(description="Fail-closed v48.55 coordinate-typed component calibration contract.")
    ap.add_argument("--run", type=Path, required=True, help="candidate run directory, e.g. candidates/precision")
    ap.add_argument("--expect-drs-sign-only", type=btext, required=True)
    ap.add_argument("--expect-continuous-canonicalization", type=btext, required=True)
    ap.add_argument("--scale-file", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()

    err: list[str] = []
    factor = a.run / "factor_stage"
    arch_p = factor / "STAGE_ARCHITECTURE.json"
    cache_p = factor / "FACTOR_CACHE_CONTRACT.json"
    for p, name in ((arch_p,"stage_architecture"),(cache_p,"factor_cache_contract")):
        if not p.is_file(): err.append(f"missing_{name}:{p}")
    arch = {}; cache = {}; scale = {}
    try:
        if arch_p.is_file(): arch=json.loads(arch_p.read_text())
    except Exception as exc: err.append(f"architecture_unreadable:{exc!r}")
    try:
        if cache_p.is_file(): cache=json.loads(cache_p.read_text())
    except Exception as exc: err.append(f"cache_unreadable:{exc!r}")
    settings = cache.get("settings", {}) if isinstance(cache.get("settings"), dict) else {}

    expected_reg = [0.0,1.0,1.0,0.0,0.0] if a.expect_drs_sign_only else [1.0,1.0,1.0,0.0,0.0]
    actual_reg_arch = _csv(arch.get("component_margin_regression_reliability"))
    actual_reg_cache = _csv(settings.get("component_margin_regression_reliability"))
    mode = "pooled_rms_linear" if a.expect_continuous_canonicalization else "raw"

    expected_scales: list[float] = []
    if a.expect_continuous_canonicalization:
        if a.scale_file is None or not a.scale_file.is_file():
            err.append(f"missing_scale_file:{a.scale_file}")
        else:
            try:
                scale=json.loads(a.scale_file.read_text())
                expected_scales=[float(x) for x in scale.get("canonical_scales", [])]
            except Exception as exc: err.append(f"scale_unreadable:{exc!r}")
    actual_scales_arch = _csv(arch.get("component_margin_canonical_scales"))
    actual_scales_cache = _csv(settings.get("component_margin_canonical_scales"))

    checks = {
        "algorithm_v48_55": str(arch.get("algorithm_variant", "")).startswith("v48.55-"),
        "no_regime_exposure": arch.get("regime_id_exposed_to_evidence_model") is False,
        "test_roots_not_read": arch.get("test_roots_read") is False,
        "component_reliability_unchanged": _same(_csv(arch.get("component_reliability")), [1,1,1,0,0]),
        "target_mode": arch.get("component_margin_target_mode") == mode and settings.get("component_margin_target_mode") == mode,
        "regression_reliability_arch": _same(actual_reg_arch, expected_reg),
        "regression_reliability_cache": _same(actual_reg_cache, expected_reg),
        "component_margin_weight_unchanged": abs(float(arch.get("component_margin_regression_weight", -1))-1.0) < 1e-12,
        "target_scale_unchanged": abs(float(arch.get("component_margin_target_scale", -1))-0.10) < 1e-12,
        "native_physical_student_off": settings.get("native_physical_student_drs", False) in {False, "false", "False", 0, "0"},
        "teacher_physical_off": settings.get("v4852_physical_teacher_sign_alignment", False) in {False, "false", "False", 0, "0"},
        "student_physical_off": settings.get("v4853_physical_student_sign_alignment", False) in {False, "false", "False", 0, "0"},
        "ipbd_off": settings.get("v4854_invariant_physical_boundary_distillation", False) in {False, "false", "False", 0, "0"},
    }
    if a.expect_continuous_canonicalization:
        checks.update({
            "scale_regime_free": scale.get("strategy_regime_conditioning") is False,
            "scale_no_test": scale.get("test_roots_read") is False,
            "scale_linear": scale.get("saturating_transform") is False,
            "scale_zero_preserved": scale.get("zero_crossing_preserved") is True,
            "scale_order_preserved": scale.get("within_component_order_preserved") is True,
            "scale_arch_matches": _same(actual_scales_arch, expected_scales),
            "scale_cache_matches": _same(actual_scales_cache, expected_scales),
            "drs_identity_scale": len(expected_scales)>=1 and abs(expected_scales[0]-0.10)<1e-12,
        })
    else:
        checks.update({
            "no_canonical_scales_arch": not actual_scales_arch,
            "no_canonical_scales_cache": not actual_scales_cache,
        })
    for name, ok in checks.items():
        if not ok: err.append("failed:"+name)

    doc={
        "event":"v48_55_tcbc_contract",
        "version":"v48.55-DCP-DRFC-BCDE-TCBC",
        "run":str(a.run),
        "expect_drs_sign_only":bool(a.expect_drs_sign_only),
        "expect_continuous_canonicalization":bool(a.expect_continuous_canonicalization),
        "expected_regression_reliability":expected_reg,
        "expected_canonical_scales":expected_scales,
        "checks":checks,
        "valid":not err,
        "errors":err,
        "strategy_regime_conditioning":False,
        "test_roots_read":False,
    }
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+"\n")
    return 0 if not err else 4

if __name__ == "__main__":
    raise SystemExit(main())
