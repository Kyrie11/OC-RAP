#!/usr/bin/env python3
"""Auto-discover V48.111 submission and external baseline closed-loop results.

Creates one paired table for each Safe/Near/Contact regime by delegating metric
formatting and pairing checks to build_regime_comparison_tables.py.  This wrapper
matches the single-regime baseline launchers shipped with the repository.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SAFE = ["gameformer_lite", "plantf", "pluto", "pdm_closed", "pdm_hybrid", "idm"]
NEAR = ["marc_lite", "racp_lite", "robust_scenario_mpc", "predictive_safety_filter", "dr_cvar_safety_filter", "conformal_predictive_safety_filter"]
CONTACT = ["postimpact_mpc_lite", "post_crash_braking", "postimpact_motion_tvlqr", "post_collision_restoration", "compensatory_postimpact_mpc", "robust_postimpact_control"]


def _require(p: Path, what: str) -> Path:
    if not p.is_file():
        raise SystemExit(f"missing {what}: {p}")
    return p


def _ocrap_result(root: Path, variant: str, regime: str) -> Path:
    return _require(root / variant / regime / "closed_loop_ocrap.json", f"OC-RAP {variant}/{regime} result")


def _baseline_results(root: Path, methods: list[str]) -> list[tuple[str, Path]]:
    out=[]
    for method in methods:
        out.append((method, _require(root / f"closed_loop_{method}.json", f"baseline {method}")))
    return out


def _build(regime: str, output: Path, entries: list[tuple[str, Path]], allow_unpaired: bool) -> None:
    script=Path(__file__).with_name("build_regime_comparison_tables.py")
    cmd=[sys.executable, str(script), "--regime", regime, "--output-dir", str(output)]
    for method,path in entries:
        cmd += ["--input", f"{method}={path}"]
    if allow_unpaired:
        cmd.append("--allow-unpaired")
    subprocess.run(cmd, check=True)


def main() -> int:
    ap=argparse.ArgumentParser(description="Build Safe/Near/Contact OC-RAP-vs-external-baseline tables from completed runs.")
    ap.add_argument("--ocrap-run", type=Path, required=True, help="runs/ocrap_v48_111_submission_three_regime")
    ap.add_argument("--safe-run", type=Path, required=True, help="RUN used by run_safe_regime_external_baselines.sh")
    ap.add_argument("--near-run", type=Path, required=True, help="RUN used by run_near_contact_external_baselines_2gpu_optimized.sh")
    ap.add_argument("--contact-run", type=Path, required=True, help="RUN used by run_contact_external_baselines.sh")
    ap.add_argument("--variants", default="balanced,precision")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--allow-unpaired", action="store_true", help="Only for diagnostics; paper tables should remain paired.")
    args=ap.parse_args()
    variants=[v.strip() for v in args.variants.split(",") if v.strip()]
    if not variants: raise SystemExit("--variants is empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    specs={
        "safe": (args.safe_run, SAFE),
        "near": (args.near_run, NEAR),
        "contact": (args.contact_run, CONTACT),
    }
    manifest={"schema_version":1,"ocrap_run":str(args.ocrap_run),"variants":variants,"tables":{}}
    for regime,(base_root,methods) in specs.items():
        entries=[(f"OC-RAP ({v})", _ocrap_result(args.ocrap_run,v,regime)) for v in variants]
        entries += _baseline_results(base_root,methods)
        out=args.output_dir/regime
        _build(regime,out,entries,args.allow_unpaired)
        manifest["tables"][regime]={"output":str(out/f"{regime}_comparison.md"),"methods":[m for m,_ in entries]}
    (args.output_dir/"submission_external_baseline_tables.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({"event":"submission_external_baseline_tables","output_dir":str(args.output_dir),"variants":variants,"regimes":list(specs)},ensure_ascii=False))
    return 0

if __name__ == "__main__": raise SystemExit(main())
