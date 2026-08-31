#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


METRICS = {
    "safe": [
        ("collision_scene_rate", -1.5), ("offroad_scene_rate", -1.0),
        ("minimum_clearance_m", 0.75), ("minimum_ttc_s", 0.75),
        ("closed_loop_bounded_NUP", 1.5), ("intervention_rate", -0.5),
    ],
    "near": [
        ("collision_scene_rate", -1.0), ("offroad_scene_rate", -1.0),
        ("scene_min_clearance_m_p05", 1.0), ("scene_ttc_s_p05", 1.0),
        ("terminal_clearance_m", 0.5), ("terminal_ttc_s", 0.5),
        ("critical_ttc_exposure_duration_s", -0.75),
    ],
    "contact": [
        ("post_contact_terminal_clearance_m", 1.0),
        ("post_contact_free_space_auc_normalized_m", 1.0),
        ("post_contact_clearance_gain_m", 0.75),
        ("post_contact_escape_scene_rate", 1.0),
        ("recontact_scene_rate", -1.5), ("secondary_overlap_scene_rate", -1.5),
        ("new_stable_stop_quality_scene_rate", 0.75), ("offroad_scene_rate", -1.0),
    ],
}


def get(doc: dict[str, Any], key: str) -> float | None:
    x = doc.get(key, (doc.get("waymax_metrics", {}) or {}).get(key))
    try:
        v = float(x); return v if math.isfinite(v) else None
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Select a transparent aggregate-best external comparator for recovery videos.")
    ap.add_argument("--regime", choices=tuple(METRICS), required=True)
    ap.add_argument("--input", action="append", required=True, metavar="METHOD=RESULT.json")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    rows=[]
    for spec in args.input:
        name, raw = spec.split("=",1); path=Path(raw); doc=json.loads(path.read_text())
        vals={k:get(doc,k) for k,_ in METRICS[args.regime]}
        rows.append({"method":name,"path":str(path),"scene_journal":str(path)+".scenes.jsonl","values":vals})
    # Per-metric min-max normalization avoids arbitrary unit dominance. Missing
    # values receive no credit and are disclosed in the output.
    for key, direction in METRICS[args.regime]:
        finite=[r["values"][key] for r in rows if r["values"][key] is not None]
        lo=min(finite) if finite else 0.0; hi=max(finite) if finite else 0.0
        for r in rows:
            v=r["values"][key]
            if v is None: z=0.0
            elif hi-lo <= 1e-12: z=0.5
            else: z=(v-lo)/(hi-lo)
            if direction < 0: z=1.0-z
            r.setdefault("normalized",{})[key]=z
    for r in rows:
        num=sum(abs(w)*r["normalized"][k] for k,w in METRICS[args.regime])
        den=sum(abs(w) for _,w in METRICS[args.regime])
        r["score"]=num/max(den,1e-9)
    ranked=sorted(rows,key=lambda r:(-r["score"],r["method"]))
    doc={"event":"best_external_baseline_selection","regime":args.regime,"normalization":"per-metric min-max over supplied external baselines","metrics":[{"key":k,"direction":"higher" if w>0 else "lower","weight":abs(w)} for k,w in METRICS[args.regime]],"best":ranked[0],"ranking":ranked}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(doc,indent=2)+"\n")
    print(json.dumps({"event":doc["event"],"regime":args.regime,"best":ranked[0]["method"],"output":str(args.output)}))
    return 0

if __name__=="__main__": raise SystemExit(main())
