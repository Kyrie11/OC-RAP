#!/usr/bin/env python3
"""Extract the final Python/shell failure signature from a stage log."""
from __future__ import annotations
import argparse, json, re, time
from pathlib import Path


def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument('--log',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--stage',default='unknown'); ap.add_argument('--exit-code',type=int,required=True); args=ap.parse_args()
    text=args.log.read_text(errors='replace') if args.log.is_file() else ''
    lines=text.splitlines(); exception_type=None; message=None; location=None
    pattern=re.compile(r'^([A-Za-z_][\w.]*(?:Error|Exception|Interrupt)):\s*(.*)$')
    for line in reversed(lines):
        m=pattern.match(line.strip())
        if m:
            exception_type,message=m.group(1),m.group(2); break
    frame=re.compile(r'^\s*File "([^"]+)", line (\d+), in (.+)$')
    for line in reversed(lines):
        m=frame.match(line)
        if m:
            location={'file':m.group(1),'line':int(m.group(2)),'function':m.group(3)}; break
    doc={'event':'v48_32_1_failure_signature','created_unix':time.time(),'stage':args.stage,'exit_code':args.exit_code,'log':str(args.log),'exception_type':exception_type,'message':message,'location':location,'tail':'\n'.join(lines[-120:])}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps(doc,ensure_ascii=False)); return 0
if __name__=='__main__': raise SystemExit(main())
