#!/usr/bin/env python3
"""Safely reuse only authoritative v48.45 arms after engineering-only failures.

The tool never touches the shared source or calibration protocol.  A previous
arm is reusable only when it is pipeline-valid RC=0/20 and its protocol seal and
both source checkpoint hashes still match the current immutable inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import time


def _json(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: pathlib.Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def _reusable(run: pathlib.Path, seal_sha: str, source_run: pathlib.Path) -> tuple[bool, int | None, list[str]]:
    reasons=[]
    try:
        status=_json(run/"AUTHORITATIVE_RUN_STATUS.json")
        complete=_json(run/"V48_36_COMPLETE.json")
        attempt=_json(run/"ATTEMPT_STARTED.json")
        source=_json(run/"SOURCE_CHECKPOINT_CONTRACT.json")
    except Exception as exc:
        return False, None, [f"missing_or_unreadable_terminal_artifact:{type(exc).__name__}"]
    rc=status.get("authoritative_exit_code")
    if rc not in (0,20): reasons.append(f"non_authoritative_rc:{rc}")
    if status.get("pipeline_valid") is not True: reasons.append("status_pipeline_invalid")
    if complete.get("pipeline_valid") is not True: reasons.append("complete_pipeline_invalid")
    if complete.get("pipeline_exit_code") != rc: reasons.append("terminal_rc_disagreement")
    if any(x.get("test_roots_read") is True for x in (status,complete,attempt,source)):
        reasons.append("test_root_read_detected")
    if attempt.get("protocol_seal_sha256") != seal_sha:
        reasons.append("protocol_seal_changed")
    checks=source.get("checks") or {}
    for variant in ("balanced","precision"):
        recorded=(checks.get(variant) or {}).get("sha256")
        ckpt=source_run/"candidates"/variant/"model_v48_trac_sr"/"best.pt"
        if not recorded:
            reasons.append(f"{variant}_source_hash_missing")
        elif not ckpt.is_file():
            reasons.append(f"{variant}_source_checkpoint_missing")
        elif _sha(ckpt) != recorded:
            reasons.append(f"{variant}_source_hash_changed")
    return not reasons, int(rc) if rc in (0,20) else None, reasons


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--base-out", type=pathlib.Path, required=True)
    ap.add_argument("--protocol-seal", type=pathlib.Path, required=True)
    ap.add_argument("--source-run", type=pathlib.Path, required=True)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--force-rerun-valid", action="store_true")
    ap.add_argument("--output", type=pathlib.Path, required=True)
    a=ap.parse_args()
    seal_sha=_sha(a.protocol_seal)
    names={"A":"ocrap_v48_45_sowr_ablation_A","B":"ocrap_v48_45_sowr_ablation_B",
           "C":"ocrap_v48_45_sowr_ablation_C","D":"ocrap_v48_45_sowr_main"}
    rows={}
    for arm,name in names.items():
        run=a.base_out/name
        if not run.exists():
            rows[arm]={"run":str(run),"action":"absent","reusable":False,"reasons":["run_absent"]}
            continue
        reusable,rc,reasons=_reusable(run,seal_sha,a.source_run)
        preserve=reusable and not a.force_rerun_valid
        action="preserve_authoritative" if preserve else "remove_for_clean_retry"
        if a.apply and not preserve:
            # run is constructed from a fixed basename under base-out; never
            # accept an arbitrary deletion target from a result artifact.
            shutil.rmtree(run)
        rows[arm]={"run":str(run),"action":action,"reusable":reusable,"authoritative_exit_code":rc,"reasons":reasons}
    for name in ("ocrap_v48_45_sowr_parallel_status.json","ocrap_v48_45_sowr_2x2_comparison.json"):
        p=a.base_out/name
        if a.apply:
            p.unlink(missing_ok=True)
    doc={"event":"v48_45_6_resume_plan","created_unix":time.time(),"apply":a.apply,
         "force_rerun_valid":a.force_rerun_valid,"protocol_seal_sha256":seal_sha,
         "source_run":str(a.source_run.resolve(strict=False)),"arms":rows,"test_roots_read":False}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(doc,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(doc,indent=2))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
