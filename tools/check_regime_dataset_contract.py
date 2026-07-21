#!/usr/bin/env python3
"""Validate OC-RAP regime dataset diagnostics before expensive experiments.

Consumes the JSON summaries produced by the dataset diagnosis utility. Development
mode reports publication blockers but only fails on semantic regime contamination.
Paper mode also requires adequate Safe scene count and bounded val/test drift.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
from typing import Any


def load(root: Path, name: str) -> dict[str, Any]:
    p = root / f"{name}.json"
    if not p.exists():
        raise FileNotFoundError(p)
    return json.load(p.open())


def mean(d: dict[str, Any], *keys: str) -> float | None:
    x: Any = d
    for k in keys:
        if not isinstance(x, dict): return None
        x = x.get(k)
    if isinstance(x, dict): x = x.get("mean")
    try: return float(x)
    except (TypeError, ValueError): return None


def frac(count: int, n: int) -> float:
    return float(count) / max(1, int(n))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("diagnostics_dir", type=Path)
    ap.add_argument("--mode", choices=["development", "paper"], default="development")
    ap.add_argument("--scope", choices=["all", "trainval"], default="all")
    ap.add_argument("--max-near-post-contact", type=float, default=0.10)
    ap.add_argument("--min-contact-post-contact", type=float, default=0.95)
    ap.add_argument("--min-safe-normal", type=float, default=0.95)
    ap.add_argument("--min-paper-safe-scenes", type=int, default=100)
    ap.add_argument("--max-rdep-mean-shift", type=float, default=0.75)
    ap.add_argument("--max-hard-mean-shift", type=float, default=0.20)
    args = ap.parse_args()
    root=args.diagnostics_dir
    splits = ("train", "val") if args.scope == "trainval" else ("train", "val", "test")
    names=[f"{s}_{b}" for s in splits for b in ("safe","near_contact","contact")]
    data={n:load(root,n) for n in names}
    failures=[]; warnings=[]; rows={}
    for split in splits:
        for bucket in ("safe","near_contact","contact"):
            key=f"{split}_{bucket}"; d=data[key]; n=int(d.get("num_samples",0)); regimes=(d.get("regimes") or {}).get("counts",{}) or {}
            post=frac(int(regimes.get("post_contact",0)),n)
            normal=frac(int(regimes.get("normal",0)),n)
            rows[key]={"samples":n,"groups":d.get("num_scene_time_groups"),"scenes":d.get("num_scenes"),"post_contact_fraction":post,"normal_fraction":normal,
                       "r_dep_mean":mean(d,"recovery_labels","r_dep_star"),"hard_mean":mean(d,"candidate_prefixes","hard_violation"),
                       "negative_deployable_fraction":mean(d,"recovery_labels","negative_deployable_fraction")}
            if bucket=="near_contact" and post>args.max_near_post_contact:
                failures.append(f"{key}: post_contact_fraction={post:.3f} > {args.max_near_post_contact:.3f}")
            if bucket=="contact" and post<args.min_contact_post_contact:
                failures.append(f"{key}: post_contact_fraction={post:.3f} < {args.min_contact_post_contact:.3f}")
            if bucket=="safe" and normal<args.min_safe_normal:
                failures.append(f"{key}: normal_fraction={normal:.3f} < {args.min_safe_normal:.3f}")
    if args.scope == "all":
        for bucket in ("near_contact", "contact"):
            v=rows[f"val_{bucket}"]; t=rows[f"test_{bucket}"]
            if v["r_dep_mean"] is not None and t["r_dep_mean"] is not None:
                shift=abs(float(v["r_dep_mean"])-float(t["r_dep_mean"]))
                if shift>args.max_rdep_mean_shift:
                    (failures if args.mode=="paper" else warnings).append(f"{bucket}: |test-val r_dep mean shift|={shift:.3f}")
            if v["hard_mean"] is not None and t["hard_mean"] is not None:
                shift=abs(float(v["hard_mean"])-float(t["hard_mean"]))
                if shift>args.max_hard_mean_shift:
                    (failures if args.mode=="paper" else warnings).append(f"{bucket}: |test-val hard mean shift|={shift:.3f}")
    for split in (("val", "test") if args.scope == "all" else ("val",)):
        scenes=int(rows[f"{split}_safe"]["scenes"] or 0)
        if scenes<args.min_paper_safe_scenes:
            (failures if args.mode=="paper" else warnings).append(f"{split}_safe: only {scenes} scenes; paper target >= {args.min_paper_safe_scenes}")
    result={"mode":args.mode,"scope":args.scope,"valid":not failures,"rows":rows,"failures":failures,"warnings":warnings}
    print(json.dumps(result,indent=2,ensure_ascii=False))
    if failures:
        print("RESULT: FAIL — rebuild/filter datasets before using them for this stage")
        return 2
    print("RESULT: PASS" if not warnings else "RESULT: PASS WITH WARNINGS")
    return 0

if __name__=="__main__": raise SystemExit(main())
