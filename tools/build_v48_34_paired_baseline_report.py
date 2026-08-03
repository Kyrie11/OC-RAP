#!/usr/bin/env python3
"""Build fail-closed paired closed-loop comparisons on an identical scene set."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _rows(path: Path) -> dict[str, dict[str, Any]]:
    if path.suffix == ".json" and not path.name.endswith(".scenes.jsonl"):
        alt=Path(str(path)+".scenes.jsonl")
        if alt.is_file(): path=alt
    out={}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            e=json.loads(line); s=e.get("scene",e)
            key=str(s.get("target_key") or s.get("scene_id") or e.get("resume_key"))
            if key in out: raise ValueError(f"duplicate key {key} in {path}")
            out[key]=s
    return out


def _f(x: Any) -> float | None:
    try:
        v=float(x); return v if math.isfinite(v) else None
    except Exception: return None

METRICS={
 "closed_loop_bounded_NUP":("top",1), "intervention_rate":("top",-1),
 "ttc_s_p05":("metric_summary",1), "terminal_ttc_s":("metric_summary",1),
 "min_clearance_m_p05":("metric_summary",1), "terminal_clearance_m":("metric_summary",1),
 "critical_ttc_exposure_duration_s":("metric_summary",-1), "near_zero_clearance_exposure_rate":("metric_summary",-1),
 "post_contact_terminal_clearance_m":("metric_summary",1),
 "post_contact_free_space_auc_normalized_m":("metric_summary",1),
 "clearance_recovery_gain_m":("metric_summary",1), "ttc_recovery_gain_s":("metric_summary",1),
 "recontact_event":("metric_summary",-1), "secondary_overlap_event":("metric_summary",-1),
 "overlap_any":("metric_summary",-1), "offroad_any":("metric_summary",-1),
}


def _value(scene: dict[str,Any], metric: str, loc: str) -> float | None:
    return _f(scene.get(metric) if loc=="top" else (scene.get("metric_summary",{}) or {}).get(metric))


def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--reference", required=True, help="NAME=SCENES_JSONL")
    ap.add_argument("--method", action="append", default=[], help="NAME=SCENES_JSONL")
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--output-csv", type=Path, required=True)
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=4834)
    args=ap.parse_args()
    def spec(text):
        if "=" not in text: raise ValueError(f"invalid method spec {text}")
        n,p=text.split("=",1); return n,Path(p)
    ref_name,ref_path=spec(args.reference); ref=_rows(ref_path)
    methods=[]
    for text in args.method:
        name,path=spec(text); methods.append((name,_rows(path),str(path)))
    if not methods: raise SystemExit("at least one --method is required")
    rng=np.random.default_rng(args.seed); reports=[]; csv_rows=[]
    for name,rows,path in methods:
        if set(rows)!=set(ref):
            raise SystemExit(f"scene set mismatch for {name}: method_only={len(set(rows)-set(ref))}, reference_only={len(set(ref)-set(rows))}")
        metric_reports={}
        for metric,(loc,direction) in METRICS.items():
            deltas=[]
            for key in sorted(ref):
                a=_value(rows[key],metric,loc); b=_value(ref[key],metric,loc)
                if a is not None and b is not None: deltas.append(direction*(a-b))
            if not deltas: continue
            arr=np.asarray(deltas,float); means=[]
            if len(arr)>0 and args.bootstrap>0:
                for _ in range(args.bootstrap): means.append(float(rng.choice(arr,size=len(arr),replace=True).mean()))
            ci=[float(np.quantile(means,.025)),float(np.quantile(means,.975))] if means else [None,None]
            rep={"n":len(arr),"oriented_delta_mean":float(arr.mean()),"oriented_delta_median":float(np.median(arr)),"ci95":ci,"higher_is_better_after_orientation":True,"raw_direction":direction}
            metric_reports[metric]=rep
            csv_rows.append({"reference":ref_name,"method":name,"metric":metric,**rep})
        reports.append({"method":name,"path":path,"num_scenes":len(rows),"metrics":metric_reports})
    doc={"event":"v48_34_paired_baseline_report","reference":ref_name,"reference_path":str(ref_path),"num_scenes":len(ref),"scene_set_exact_match":True,"exploratory_only":True,"paper_claim_allowed":False,"methods":reports}
    args.output_json.parent.mkdir(parents=True,exist_ok=True); args.output_csv.parent.mkdir(parents=True,exist_ok=True)
    args.output_json.write_text(json.dumps(doc,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    fields=["reference","method","metric","n","oriented_delta_mean","oriented_delta_median","ci95","higher_is_better_after_orientation","raw_direction"]
    with args.output_csv.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for row in csv_rows: row=dict(row); row["ci95"]=json.dumps(row["ci95"]); w.writerow(row)
    print(json.dumps({"event":doc["event"],"num_methods":len(methods),"num_scenes":len(ref)}))
    return 0

if __name__=="__main__": raise SystemExit(main())
