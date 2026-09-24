#!/usr/bin/env python3
"""Build paper-ready frozen-module ablation tables by regime and variant."""
from __future__ import annotations
import argparse, csv, json, subprocess, sys
from pathlib import Path

ARM_META = {
    "no_obs_consistency": ("w/o observation consistency", ("near", "contact"), "main"),
    "mean_tail": ("mean instead of lower-tail", ("near", "contact"), "main"),
    "no_actuator_projection": ("w/o actuator projection", ("near", "contact"), "main"),
    "no_persistent_reentry": ("w/o persistent re-entry", ("contact",), "main"),
    "no_rifa_absolute_admission": ("w/o RIFA absolute admission", ("safe", "near", "contact"), "main"),
    "no_nominal_abstention": ("w/o exact-nominal abstention", ("safe", "near", "contact"), "main"),
    "no_active_set_alignment": ("w/o active-set alignment", ("near", "contact"), "supplementary"),
    "no_route_alignment": ("w/o route alignment", ("near", "contact"), "main"),
}


def require(path: Path, what: str) -> Path:
    if not path.is_file(): raise SystemExit(f"missing {what}: {path}")
    return path


def build_one(
    regime: str,
    out: Path,
    entries: list[tuple[str, Path]],
    allow_unpaired: bool,
    latency_entries: list[tuple[str, Path]] | None,
) -> None:
    script = Path(__file__).with_name("build_regime_comparison_tables.py")
    cmd=[sys.executable,str(script),"--regime",regime,"--output-dir",str(out)]
    for name,path in entries:
        cmd += ["--input",f"{name}={path}"]
    if latency_entries is None:
        # Throughput timing from the two-GPU accuracy scheduler is not the same
        # contract as final publication latency.  Omit it rather than mixing
        # incomparable timing in an otherwise fair ablation table.
        cmd.append("--omit-latency")
    else:
        for name,path in latency_entries:
            cmd += ["--latency-input",f"{name}={path}"]
    if allow_unpaired:
        cmd.append("--allow-unpaired")
    subprocess.run(cmd,check=True)


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--full-run",type=Path,required=True,help="native-certified Full three-regime reference root")
    ap.add_argument("--ablation-root",type=Path,required=True)
    ap.add_argument("--variants",default="balanced,precision")
    ap.add_argument("--tier",choices=("main","all"),default="main")
    ap.add_argument("--output-dir",type=Path,required=True)
    ap.add_argument("--latency-root",type=Path,default=None,help="Optional mirrored isolated-latency root. If absent, latency is omitted rather than read from throughput accuracy runs.")
    ap.add_argument("--allow-unpaired",action="store_true",help="diagnostic only; paper tables should be paired")
    args=ap.parse_args(); variants=[x.strip() for x in args.variants.split(',') if x.strip()]
    args.output_dir.mkdir(parents=True,exist_ok=True)
    manifest={
        "schema_version":2,
        "full_run":str(args.full_run),
        "ablation_root":str(args.ablation_root),
        "latency_root":str(args.latency_root) if args.latency_root is not None else None,
        "latency_contract":"isolated_single_process_single_gpu" if args.latency_root is not None else "omitted_no_isolated_artifact",
        "tier":args.tier,
        "tables":[],
    }
    # Regime-selection matrix is useful even before every long closed-loop run is done.
    with (args.output_dir/"ablation_matrix.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["arm","reporting_name","tier","safe","near","contact"])
        for arm,(label,regimes,tier) in ARM_META.items():
            w.writerow([arm,label,tier,int("safe" in regimes),int("near" in regimes),int("contact" in regimes)])
    lines=["# Final frozen-module ablation matrix","","> Functional knockouts use the same frozen V48.80 checkpoint, frozen per-bucket calibration, immutable target lock, and native pre-OC-MERO executable-recovery certificate as the fresh Full reference. No ablation is retrained or recalibrated.","","| Ablation | Tier | Safe | Near | Contact |","|---|---|:---:|:---:|:---:|"]
    for arm,(label,regimes,tier) in ARM_META.items():
        lines.append(f"| {label} | {tier} | {'✓' if 'safe' in regimes else '—'} | {'✓' if 'near' in regimes else '—'} | {'✓' if 'contact' in regimes else '—'} |")
    (args.output_dir/"ablation_matrix.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

    for variant in variants:
        for regime in ("safe","near","contact"):
            full=require(args.full_run/variant/regime/"closed_loop_ocrap.json",f"Full {variant}/{regime}")
            entries=[("Full OC-RAP",full)]
            latency_entries: list[tuple[str, Path]] | None = None
            if args.latency_root is not None:
                latency_entries=[(
                    "Full OC-RAP",
                    require(args.latency_root/"_native_full_reference"/variant/regime/"closed_loop_ocrap.json",f"Full isolated latency {variant}/{regime}"),
                )]
            for arm,(label,regimes,tier) in ARM_META.items():
                if regime not in regimes or (args.tier=="main" and tier!="main"):
                    continue
                p=args.ablation_root/arm/variant/regime/"closed_loop_ocrap.json"
                if p.is_file():
                    entries.append((label,p))
                    if latency_entries is not None:
                        lp=require(args.latency_root/arm/variant/regime/"closed_loop_ocrap.json",f"isolated latency {arm} {variant}/{regime}")
                        latency_entries.append((label,lp))
            if len(entries)==1:
                continue
            out=args.output_dir/variant/regime
            build_one(regime,out,entries,args.allow_unpaired,latency_entries)
            manifest["tables"].append({
                "variant":variant,"regime":regime,"rows":[n for n,_ in entries],
                "isolated_latency":latency_entries is not None,
                "markdown":str(out/f"{regime}_comparison.md"),
            })
    (args.output_dir/"submission_ablation_tables.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"event":"submission_ablation_tables","tables":len(manifest["tables"]),"output_dir":str(args.output_dir)},ensure_ascii=False))
    return 0
if __name__=="__main__": raise SystemExit(main())
