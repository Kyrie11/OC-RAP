#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

ENGINEERING_VERSION = "v48.124.10.2-OBSERVATION-LEGAL-PROVENANCE-ENGFIX"
SCIENTIFIC_VERSION = "v48.124-OC-FMSA"

REQUIRED = {
    "src/ocrap/data/waymax_loader.py": [
        "womd_v1_3_1_sdc_paths_connectivity_only",
        "_connectivity_only_path_index",
        "refusing to fall back to validation log_trajectory future labels",
        "allow_logged_sdc_route_fallback",
    ],
    "src/ocrap/data/build/history.py": [
        "require_observation_legal_route",
        "refusing future_agent_states route fallback",
    ],
    "scripts/run_ocrap_closed_loop.sh": [
        'USE_SDC_PATHS="${USE_SDC_PATHS:-true}"',
        'REQUIRE_OBSERVATION_LEGAL_ROUTE="${REQUIRE_OBSERVATION_LEGAL_ROUTE:-true}"',
        'ALLOW_LOGGED_SDC_ROUTE_FALLBACK="${ALLOW_LOGGED_SDC_ROUTE_FALLBACK:-false}"',
    ],
}

def sha(p: Path) -> str:
    h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()

def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a=ap.parse_args()
    errors=[]; files={}
    root=a.repo.resolve()
    for rel, needles in REQUIRED.items():
        p=root/rel
        if not p.is_file():
            errors.append(f"missing:{rel}"); continue
        text=p.read_text(encoding='utf-8')
        missing=[x for x in needles if x not in text]
        if missing: errors.append(f"contract:{rel}:{missing}")
        files[rel]={"sha256":sha(p),"size":p.stat().st_size}
    snapshot=root/'EXECUTION_SNAPSHOT.json'
    if snapshot.exists(): files['EXECUTION_SNAPSHOT.json']={"sha256":sha(snapshot),"size":snapshot.stat().st_size}
    out={
        "schema":"ocrap-observation-legal-route-runtime-contract-v1",
        "engineering_version":ENGINEERING_VERSION,
        "scientific_version":SCIENTIFIC_VERSION,
        "valid":not errors,
        "attribution_ready":not errors,
        "algorithm_modified":False,
        "evaluation_protocol_modified":True,
        "route_contract":{
            "dataset":"WOMD v1.3.1 TFExample validation",
            "planner_route_source":"WOMD v1.3.1 sdc_paths connectivity geometry only",
            "planner_uses_path_samples_on_route":False,
            "logged_sdc_future_fallback":False,
            "future_agent_states_route_fallback":False,
            "fail_closed_if_sdc_paths_unusable":True,
        },
        "files":files,"errors":errors,
    }
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n",encoding='utf-8')
    print(json.dumps({"valid":out['valid'],"output":str(a.output)},indent=2))
    return 0 if out['valid'] else 30
if __name__=='__main__': raise SystemExit(main())
