#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any


def read(path: Path) -> dict[str, Any] | None:
    try:
        x=json.loads(path.read_text(encoding='utf-8')); return x if isinstance(x,dict) else None
    except Exception: return None


def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--launcher-exit-code',type=int,default=0); args=ap.parse_args()
    regimes={}; failed=[]; summary={}
    for r in ('safe','near','contact'):
        d=args.root/r; p=d/'closed_loop_ocrap.json'; prog=p.with_suffix(p.suffix+'.progress.json'); journal=p.with_suffix(p.suffix+'.scenes.jsonl')
        phase=read(args.root/f'{r}.phase.json') or {'status':'not_started','exit_code':None}
        result=read(p); progress=read(prog)
        complete=bool(phase.get('status')=='complete' and result and progress and progress.get('status')=='complete' and journal.is_file())
        if complete and result.get('bucket_target_count') not in (None,0): complete=int(result.get('num_scenes') or 0)==int(result['bucket_target_count'])
        if not complete: failed.append(r)
        row={'phase':phase,'result':str(p),'result_exists':p.is_file(),'progress':str(prog),'progress_status':progress.get('status') if progress else None,'scene_journal':str(journal),'scene_journal_exists':journal.is_file(),'num_scenes':result.get('num_scenes') if result else None,'bucket_target_count':result.get('bucket_target_count') if result else None,'run_fingerprint':result.get('run_fingerprint') if result else None,'complete':complete}
        regimes[r]=row
        if result:
            keys=('method','source','num_scenes','num_decisions','collision_scene_rate','offroad_scene_rate','closed_loop_bounded_NUP','intervention_rate','scene_min_clearance_m_p05','scene_ttc_s_p05','critical_ttc_exposure_duration_s','post_contact_terminal_clearance_m','post_contact_free_space_auc_normalized_m','post_contact_clearance_gain_m','post_contact_escape_scene_rate','recontact_scene_rate','secondary_overlap_scene_rate','new_stable_stop_quality_scene_rate','post_contact_overlap_duration_s','timing')
            summary[r]={k:result.get(k) for k in keys}
    complete=not failed and args.launcher_exit_code==0
    doc={'event':'ocrap_three_regime_closed_loop_v50','schema_version':2,'root':str(args.root),'launcher_exit_code':args.launcher_exit_code,'status':'complete' if complete else 'failed_or_incomplete','complete':complete,'failed_or_incomplete_regimes':failed,'regimes':regimes}
    args.root.mkdir(parents=True,exist_ok=True)
    (args.root/'OCRAP_THREE_REGIME_RUN_INDEX.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (args.root/'SUMMARY.json').write_text(json.dumps({'event':'ocrap_three_regime_closed_loop_summary','complete':complete,'regimes':summary},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'event':doc['event'],'complete':complete,'failed':failed,'output':str(args.root/'OCRAP_THREE_REGIME_RUN_INDEX.json')}))
    return 0
if __name__=='__main__': raise SystemExit(main())
