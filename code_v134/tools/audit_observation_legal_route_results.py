#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from collections import Counter
from pathlib import Path

LEGAL="womd_v1_3_1_sdc_paths_connectivity_only"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--result', action='append', required=True, help='label=path')
    ap.add_argument('--output', required=True)
    a=ap.parse_args(); errors=[]; rows={}
    for spec in a.result:
        label,path=spec.split('=',1); d=json.loads(Path(path).read_text(encoding='utf-8'))
        scenes=d.get('scenes') or []
        sources=Counter(str(s.get('route_progression_source')) for s in scenes)
        illegal=[s.get('target_key') for s in scenes if str(s.get('route_progression_source')) != LEGAL]
        speed=d.get('closed_loop_speed_config') or {}
        if speed.get('dataloader_include_sdc_paths') is not True: errors.append(f'{label}:sdc_paths_not_enabled')
        if illegal: errors.append(f'{label}:illegal_route_sources:{len(illegal)}')
        rows[label]={"num_scenes":len(scenes),"route_progression_sources":dict(sources),"illegal_target_count":len(illegal),"illegal_target_examples":illegal[:10]}
    out={"schema":"ocrap-observation-legal-route-result-audit-v1","valid":not errors,"required_source":LEGAL,"errors":errors,"results":rows}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"valid":out['valid'],"errors":errors,"output":a.output},indent=2))
    return 0 if out['valid'] else 30
if __name__=='__main__': raise SystemExit(main())
