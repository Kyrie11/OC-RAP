#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import yaml

ENGINEERING_VERSION = "v48.124.10-OC-FMSA-NEAR-RIFA-SYSTEM-AXIS-AUDIT"
FILES = [
    "src/ocrap/planning/selector.py",
    "src/ocrap/evaluation/baselines.py",
    "configs/v48_124_10_near_relative_delta.yaml",
    "configs/v48_124_10_near_nested_evidence.yaml",
    "scripts/run_near_rifa_system_axis_two_gpu.sh",
    "tools/build_near_intervention_cohort.py",
    "tools/merge_monotone_near_subset.py",
    "tools/adjudicate_near_rifa_system_axis.py",
]

def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main() -> int:
    ap=argparse.ArgumentParser(description="Fail-closed V48.124.10 Near system-axis runtime contract.")
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args=ap.parse_args(); repo=args.repo.resolve(); errors=[]
    rows={}
    for rel in FILES:
        p=repo/rel
        if not p.is_file(): errors.append(f"missing:{rel}"); continue
        rows[rel]={"sha256":sha(p),"size":p.stat().st_size}
    try:
        d=yaml.safe_load((repo/"configs/v48_124_10_near_relative_delta.yaml").read_text()) or {}
        n=yaml.safe_load((repo/"configs/v48_124_10_near_nested_evidence.yaml").read_text()) or {}
        ds=d.get("selection") or {}; ns=n.get("selection") or {}
        if ds.get("ocrap_selector") != "lcb_constrained_relative_delta": errors.append("delta_selector_contract")
        if float(ds.get("rifa_relative_min_advantage")) != 0.0: errors.append("delta_sign_contract")
        if ns.get("ocrap_selector") != "lcb_constrained_nested_evidence": errors.append("nested_selector_contract")
        if int(ns.get("rifa_relative_proposal_top_k")) != 5: errors.append("nested_topk_contract")
        if float(ns.get("rifa_relative_min_advantage")) != 0.0: errors.append("nested_sign_contract")
        if float(ns.get("rifa_relative_opportunity_threshold")) != 0.65: errors.append("nested_opp_contract")
        if float(ns.get("rifa_relative_harm_threshold")) != 0.30: errors.append("nested_harm_contract")
    except Exception as e: errors.append(f"config_parse:{type(e).__name__}:{e}")
    # Require immutable snapshot when this checker is used by a scientific run.
    snap=repo/"EXECUTION_SNAPSHOT.json"
    snapshot={}
    if snap.is_file():
        try:
            snapshot=json.loads(snap.read_text())
            if snapshot.get("valid") is not True: errors.append("snapshot_invalid")
        except Exception as e: errors.append(f"snapshot_parse:{e}")
    else:
        errors.append("missing_execution_snapshot")
    out={
        "schema":"ocrap-v48.124.10-near-rifa-system-axis-runtime-contract-v1",
        "engineering_version":ENGINEERING_VERSION,
        "valid":not errors,
        "attribution_ready":not errors,
        "errors":errors,
        "scientific_contract":{
            "historical_main":"v48.124-OC-FMSA frozen",
            "scope":"Near deployed-selector/system-integration axis only",
            "training":False,"recalibration":False,"threshold_sweep":False,"capacity_sweep":False,
            "recovery_mechanism_family_frozen":True,
            "default_diagnostic_population":"historical V48.124.9 intervention cohort only",
            "full_250_required_after_promotion":True,
        },
        "snapshot":{"run_instance_id":snapshot.get("run_instance_id"),"tree_sha256":snapshot.get("tree_sha256")},
        "runtime_files":rows,
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"valid":out["valid"],"errors":errors,"output":str(args.output)},indent=2))
    return 0 if not errors else 30
if __name__=="__main__": raise SystemExit(main())
