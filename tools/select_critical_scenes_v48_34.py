#!/usr/bin/env python3
"""Deterministically select paired Near/Contact qualitative examples.

The selector consumes *metric-only* scene journals.  Rendering traces are not
needed until after the target keys are selected, which avoids recording every
frame for every method in the full experiment.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

REQUIRED = {
    "near": {
        "ttc_s_p05", "terminal_ttc_s", "min_clearance_m_p05", "terminal_clearance_m",
        "critical_ttc_exposure_duration_s", "near_zero_clearance_exposure_rate",
        "overlap_any", "offroad_any",
    },
    "contact": {
        "post_contact_terminal_clearance_m", "post_contact_free_space_auc_normalized_m",
        "post_contact_clearance_gain_m", "post_contact_overlap_duration_s",
        "post_contact_escape_event", "recontact_event",
        "new_stable_stop_quality_event", "offroad_any",
    },
}


def _finite_or_none(x: Any) -> float | None:
    try:
        v=float(x); return v if math.isfinite(v) else None
    except Exception: return None


def _finite(x: Any, default: float=0.0) -> float:
    v=_finite_or_none(x); return default if v is None else v


def _scene_rows(path: Path) -> dict[str, dict[str, Any]]:
    if path.suffix == ".json" and not path.name.endswith(".scenes.jsonl"):
        candidate=Path(str(path)+".scenes.jsonl")
        if candidate.is_file(): path=candidate
    rows={}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            envelope=json.loads(line); scene=envelope.get("scene",envelope)
            key=str(scene.get("target_key") or envelope.get("resume_key") or "")
            if key.startswith("target:"): key=key[len("target:"):]
            if not key:
                sid=str(scene.get("scene_id") or ""); t=scene.get("target_time_index")
                key=f"{sid}:t{t}" if sid and t is not None else sid
            if not key: raise ValueError(f"scene row without target/scene key in {path}")
            if key in rows: raise ValueError(f"duplicate scene key {key} in {path}")
            # Population journals from v50.1 may still contain every decision
            # and render frame. Qualitative selection needs only scalar scene
            # summaries, so discard heavy payloads while streaming the file.
            rows[key]={
                k:v for k,v in scene.items()
                if k not in {"decisions","render_trace","render_context","render_trace_schema","state_xy_trace"}
                and not str(k).endswith("_trace")
            }
    if not rows: raise ValueError(f"empty paired scene journal: {path}")
    return rows


def _metric(scene: dict[str, Any], name: str) -> float | None:
    if name in scene: return _finite_or_none(scene.get(name))
    return _finite_or_none((scene.get("metric_summary",{}) or {}).get(name))


def _delta(method: dict[str, Any], control: dict[str, Any], name: str) -> float | None:
    a=_metric(method,name); b=_metric(control,name)
    return None if a is None or b is None else a-b


def _unsafe_regression(regime: str, method: dict[str, Any], control: dict[str, Any]) -> bool:
    # In the Contact bucket the initiating collision is the causal anchor, so
    # ``overlap_any`` is not a meaningful regression signal.  Later re-contact
    # and off-road events are.  Near-contact still treats any new overlap as a
    # hard safety regression.
    names = ("overlap_any", "offroad_any") if regime == "near" else ("offroad_any", "recontact_event")
    return any((_delta(method, control, name) or 0.0) > 1e-9 for name in names)


def _score(regime: str, method: dict[str, Any], control: dict[str, Any], args: argparse.Namespace) -> tuple[float,dict[str,float|None],bool,bool,list[str],list[str]]:
    missing=[n for n in sorted(REQUIRED[regime]) if _metric(method,n) is None or _metric(control,n) is None]
    common={
        "bounded_nup":_delta(method,control,"closed_loop_bounded_NUP"),
        "intervention_rate":_finite_or_none(method.get("intervention_rate")),
        "overlap_any":_delta(method,control,"overlap_any"),
        "offroad_any":_delta(method,control,"offroad_any"),
    }
    material=[]
    if regime=="near":
        terms=common|{
            "ttc_p05_s":_delta(method,control,"ttc_s_p05"),
            "terminal_ttc_s":_delta(method,control,"terminal_ttc_s"),
            "clearance_p05_m":_delta(method,control,"min_clearance_m_p05"),
            "terminal_clearance_m":_delta(method,control,"terminal_clearance_m"),
            "critical_ttc_exposure_s":_delta(method,control,"critical_ttc_exposure_duration_s"),
            "near_zero_clearance_rate":_delta(method,control,"near_zero_clearance_exposure_rate"),
        }
        score=(1.5*_finite(terms["ttc_p05_s"])+0.5*_finite(terms["terminal_ttc_s"])+1.0*_finite(terms["clearance_p05_m"])+0.5*_finite(terms["terminal_clearance_m"])-1.5*_finite(terms["critical_ttc_exposure_s"])-1.0*_finite(terms["near_zero_clearance_rate"])+0.25*_finite(terms["bounded_nup"]))
        if _finite(terms["ttc_p05_s"]) >= args.min_near_ttc_gain_s: material.append("ttc_p05")
        if _finite(terms["clearance_p05_m"]) >= args.min_near_clearance_gain_m: material.append("clearance_p05")
        if _finite(terms["terminal_clearance_m"]) >= args.min_near_clearance_gain_m: material.append("terminal_clearance")
        if _finite(terms["critical_ttc_exposure_s"]) <= -args.min_near_exposure_reduction_s: material.append("critical_exposure")
        material_regression=(
            _finite(terms["ttc_p05_s"]) < -args.max_near_ttc_regression_s or
            _finite(terms["clearance_p05_m"]) < -args.max_near_clearance_regression_m or
            _finite(terms["critical_ttc_exposure_s"]) > args.max_near_exposure_regression_s
        )
    elif regime=="contact":
        terms=common|{
            "post_contact_terminal_clearance_m":_delta(method,control,"post_contact_terminal_clearance_m"),
            "post_contact_free_space_auc_normalized_m":_delta(method,control,"post_contact_free_space_auc_normalized_m"),
            "post_contact_clearance_gain_m":_delta(method,control,"post_contact_clearance_gain_m"),
            "ttc_recovery_gain_s":_delta(method,control,"ttc_recovery_gain_s"),
            "post_contact_overlap_duration_s":_delta(method,control,"post_contact_overlap_duration_s"),
            "new_stable_stop_quality_event":_delta(method,control,"new_stable_stop_quality_event"),
            "post_contact_escape_event":_delta(method,control,"post_contact_escape_event"),
            "recontact_event":_delta(method,control,"recontact_event"),
        }
        score=(1.5*_finite(terms["post_contact_terminal_clearance_m"])+1.0*_finite(terms["post_contact_free_space_auc_normalized_m"])+0.75*_finite(terms["post_contact_clearance_gain_m"])+0.35*_finite(terms["ttc_recovery_gain_s"])-1.25*_finite(terms["post_contact_overlap_duration_s"])+2.0*_finite(terms["new_stable_stop_quality_event"])+1.5*_finite(terms["post_contact_escape_event"])-4.0*_finite(terms["recontact_event"])+0.25*_finite(terms["bounded_nup"]))
        if _finite(terms["post_contact_terminal_clearance_m"]) >= args.min_contact_terminal_clearance_gain_m: material.append("terminal_clearance")
        if _finite(terms["post_contact_free_space_auc_normalized_m"]) >= args.min_contact_auc_gain_m: material.append("free_space_auc")
        if _finite(terms["post_contact_clearance_gain_m"]) >= args.min_contact_clearance_gain_m: material.append("clearance_gain")
        if _finite(terms["post_contact_escape_event"]) > 0: material.append("escape_event")
        if _finite(terms["post_contact_overlap_duration_s"]) <= -args.min_contact_overlap_duration_reduction_s: material.append("overlap_duration_reduced")
        if _finite(terms["new_stable_stop_quality_event"]) > 0: material.append("new_stable_stop")
        if _finite(terms["recontact_event"]) < 0: material.append("recontact_avoided")
        material_regression=(
            _finite(terms["post_contact_terminal_clearance_m"]) < -args.max_contact_terminal_clearance_regression_m or
            _finite(terms["post_contact_free_space_auc_normalized_m"]) < -args.max_contact_auc_regression_m or
            _finite(terms["post_contact_overlap_duration_s"]) > args.max_contact_overlap_duration_regression_s
        )
    else: raise ValueError(regime)
    intervention=_finite_or_none(method.get("intervention_rate"))
    unsafe_regression = _unsafe_regression(regime,method,control)
    fallback_eligible = (
        not missing and intervention is not None and intervention > 0
        and score >= args.minimum_positive_score
        and not material_regression and not unsafe_regression
    )
    eligible = fallback_eligible and bool(material)
    return float(score),terms,eligible,fallback_eligible,missing,material


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--method-scenes",type=Path,required=True); ap.add_argument("--control-scenes",type=Path,required=True)
    ap.add_argument("--regime",choices=("near","contact"),required=True); ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--target-keys-output",type=Path)
    ap.add_argument("--num-positive",type=int,default=5); ap.add_argument("--num-failure",type=int,default=0)
    ap.add_argument("--max-per-scene",type=int,default=1); ap.add_argument("--minimum-positive-score",type=float,default=0.0)
    ap.add_argument("--require-exact-positive-count",action="store_true")
    ap.add_argument("--fallback-topk-nonregressive",action="store_true",help="Fill an exact 5-scene qualitative set with positive-score, non-regressive scenes when strict material thresholds yield fewer examples.")
    ap.add_argument("--min-near-ttc-gain-s",type=float,default=0.25); ap.add_argument("--min-near-clearance-gain-m",type=float,default=0.25); ap.add_argument("--min-near-exposure-reduction-s",type=float,default=0.20)
    ap.add_argument("--max-near-ttc-regression-s",type=float,default=0.10); ap.add_argument("--max-near-clearance-regression-m",type=float,default=0.10); ap.add_argument("--max-near-exposure-regression-s",type=float,default=0.10)
    ap.add_argument("--min-contact-terminal-clearance-gain-m",type=float,default=0.50); ap.add_argument("--min-contact-auc-gain-m",type=float,default=0.50); ap.add_argument("--min-contact-clearance-gain-m",type=float,default=0.25); ap.add_argument("--min-contact-overlap-duration-reduction-s",type=float,default=0.20)
    ap.add_argument("--max-contact-terminal-clearance-regression-m",type=float,default=0.10); ap.add_argument("--max-contact-auc-regression-m",type=float,default=0.25); ap.add_argument("--max-contact-overlap-duration-regression-s",type=float,default=0.10)
    args=ap.parse_args()
    method=_scene_rows(args.method_scenes); control=_scene_rows(args.control_scenes)
    if set(method)!=set(control): raise SystemExit(f"unpaired scene sets: method_only={sorted(set(method)-set(control))[:10]} control_only={sorted(set(control)-set(method))[:10]}")
    rows=[]
    for key in sorted(method):
        score,terms,eligible,fallback_eligible,missing,material=_score(args.regime,method[key],control[key],args)
        rows.append({"target_key":key,"scene_id":method[key].get("scene_id"),"target_time_index":method[key].get("target_time_index"),"regime":args.regime,"score":score,"eligible_positive_example":eligible,"fallback_nonregressive_example":fallback_eligible,"missing_required_metrics":missing,"material_improvements":material,"method_intervention_rate":_finite_or_none(method[key].get("intervention_rate")),"terms":terms})
    positive_pool=[r for r in rows if r["eligible_positive_example"]]
    positive=[]; scene_counts={}
    for row in sorted(positive_pool,key=lambda r:(-r["score"],str(r["target_key"]))):
        sid=str(row.get("scene_id") or row["target_key"])
        if scene_counts.get(sid,0)>=max(args.max_per_scene,1): continue
        positive.append(row); scene_counts[sid]=scene_counts.get(sid,0)+1
        if len(positive)>=max(args.num_positive,0): break
    strict_count=len(positive)
    if args.fallback_topk_nonregressive and len(positive)<max(args.num_positive,0):
        selected_keys={r["target_key"] for r in positive}
        fallback_pool=[r for r in rows if r["fallback_nonregressive_example"] and r["target_key"] not in selected_keys]
        for row in sorted(fallback_pool,key=lambda r:(-r["score"],str(r["target_key"]))):
            sid=str(row.get("scene_id") or row["target_key"])
            if scene_counts.get(sid,0)>=max(args.max_per_scene,1): continue
            row={**row,"selection_tier":"best_available_nonregressive"}
            positive.append(row); scene_counts[sid]=scene_counts.get(sid,0)+1
            if len(positive)>=max(args.num_positive,0): break
    positive=[({**r,"selection_tier":r.get("selection_tier","strict_material_improvement")}) for r in positive]
    if args.require_exact_positive_count and len(positive)!=args.num_positive:
        raise SystemExit(f"requested {args.num_positive} positive/non-regressive {args.regime} scenes, found {len(positive)} after diversity filtering (strict={strict_count})")
    pkeys={r["target_key"] for r in positive}; failure=[]; fcounts={}
    for row in sorted((r for r in rows if r["target_key"] not in pkeys),key=lambda r:(r["score"],str(r["target_key"]))):
        sid=str(row.get("scene_id") or row["target_key"])
        if fcounts.get(sid,0)>=max(args.max_per_scene,1): continue
        failure.append(row); fcounts[sid]=fcounts.get(sid,0)+1
        if len(failure)>=max(args.num_failure,0): break
    selected=[]
    for category,items in (("positive_toy_example",positive),("failure_case",failure)):
        for rank,row in enumerate(items,1): selected.append({**row,"category":category,"category_rank":rank})
    doc={
        "event":"v50_critical_scene_selection","regime":args.regime,"exploratory_qualitative_only":True,"paper_population_claim_allowed":False,
        "selection_process":"deterministic post-hoc selection from paired metric journals using published thresholds and target-key tie breaking",
        "not_population_level_evidence":True,"diversity_max_per_scene":args.max_per_scene,"minimum_positive_score":args.minimum_positive_score,
        "num_paired_scenes":len(rows),"num_positive_eligible_scenes":len(positive_pool),"num_strict_selected_scenes":strict_count,"num_fallback_selected_scenes":max(0,len(positive)-strict_count),"required_metrics":sorted(REQUIRED[args.regime]),
        "thresholds":{k:v for k,v in vars(args).items() if k.startswith(('min_','max_')) and k not in {'max_per_scene'}},
        "selected":selected,"target_keys":[r["target_key"] for r in selected],"all_scene_scores":sorted(rows,key=lambda r:(-r["score"],str(r["target_key"])))
    }
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if args.target_keys_output:
        args.target_keys_output.parent.mkdir(parents=True,exist_ok=True); args.target_keys_output.write_text(json.dumps({"regime":args.regime,"target_keys":doc["target_keys"]},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({"event":doc["event"],"output":str(args.output),"selected":len(selected),"positive_eligible":len(positive_pool)}))
    return 0
if __name__=='__main__': raise SystemExit(main())
