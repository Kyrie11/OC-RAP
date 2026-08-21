#!/usr/bin/env python3
"""Canonicalize the v48.58 RIFA selector keys in POLICY_CONTRACT.env."""
from __future__ import annotations
import argparse
from pathlib import Path

KEYS = {
    "ABSOLUTE_FEASIBILITY_MODE",
    "ABSOLUTE_FEASIBILITY_THRESHOLD",
    "SELECTION_SEMANTICS",
}
RIFA_ORDER = "rank_topk_then_absolute_feasibility_then_relative_filter_then_evidence_rerank"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", type=Path, required=True)
    ap.add_argument("--mode", choices=("native", "learned"), required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()
    if abs(float(args.threshold) - 0.5) > 1e-12:
        raise SystemExit("v48.58 RIFA requires threshold=0.5")
    lines = args.contract.read_text(encoding="utf-8").splitlines()
    kept=[]
    for line in lines:
        stripped=line.strip()
        key=stripped.split("=",1)[0].strip() if "=" in stripped else None
        if key in KEYS:
            continue
        kept.append(line)
    while kept and not kept[-1].strip():
        kept.pop()
    kept.extend([
        f"ABSOLUTE_FEASIBILITY_MODE={args.mode}",
        "ABSOLUTE_FEASIBILITY_THRESHOLD=0.5",
        f"SELECTION_SEMANTICS={RIFA_ORDER}",
    ])
    text="\n".join(kept)+"\n"
    tmp=args.contract.with_name(f".{args.contract.name}.tmp")
    tmp.write_text(text,encoding="utf-8")
    tmp.replace(args.contract)
    # Fail closed on duplicate critical keys after rewrite.
    counts={k:0 for k in KEYS}
    values={}
    for raw in args.contract.read_text(encoding="utf-8").splitlines():
        if "=" not in raw: continue
        k,v=raw.split("=",1); k=k.strip(); v=v.strip()
        if k in KEYS:
            counts[k]+=1; values[k]=v
    if any(counts[k] != 1 for k in KEYS):
        raise SystemExit(f"non-canonical RIFA policy contract: counts={counts}")
    if values["ABSOLUTE_FEASIBILITY_MODE"] != args.mode or values["ABSOLUTE_FEASIBILITY_THRESHOLD"] != "0.5" or values["SELECTION_SEMANTICS"] != RIFA_ORDER:
        raise SystemExit(f"RIFA policy contract rewrite mismatch: {values}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
