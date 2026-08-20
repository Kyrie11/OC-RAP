#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

ARM_DIRS = {
    "A": "ocrap_v48_56_dcp_drfc_bcde_drac_ablation_A",
    "B": "ocrap_v48_56_dcp_drfc_bcde_drac_ablation_B",
    "C": "ocrap_v48_56_dcp_drfc_bcde_drac_ablation_C",
    "D": "ocrap_v48_56_dcp_drfc_bcde_drac_main",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out=[]
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try: out.append(json.loads(line))
        except Exception: continue
    return out


def end_durations(rows: list[dict[str, Any]]) -> dict[str, float]:
    out={}
    for row in rows:
        if row.get("event") == "end" and row.get("duration_seconds") is not None:
            try: out[str(row.get("stage"))] = float(row["duration_seconds"])
            except Exception: pass
    return out


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--base-out", type=Path, required=True)
    ap.add_argument("--global-log", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args=ap.parse_args()
    global_rows=read_jsonl(args.global_log)
    arms={}
    for arm,name in ARM_DIRS.items():
        run=args.base_out/name
        rows=read_jsonl(run/"logs/runtime_stage_timing.jsonl")
        if rows or run.exists():
            arms[arm]={
                "run": str(run),
                "durations_seconds": end_durations(rows),
                "events": rows,
            }
    doc={
        "event":"v48_56_stage_timing_summary",
        "created_unix":time.time(),
        "global_log":str(args.global_log),
        "global_durations_seconds":end_durations(global_rows),
        "global_events":global_rows,
        "arms":arms,
        "diagnostic_only":True,
        "algorithm_semantics_changed":False,
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(doc,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(doc,ensure_ascii=False),flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
