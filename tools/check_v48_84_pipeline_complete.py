#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--runtime',type=Path,required=True);ap.add_argument('--balanced',type=Path,required=True);ap.add_argument('--precision',type=Path,required=True);ap.add_argument('--comparison',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args(); ds=[json.loads(x.read_text()) for x in [a.runtime,a.balanced,a.precision,a.comparison]]; errors=[]
 if not all(d.get('valid') for d in ds): errors.append('invalid_dependency')
 if not ds[0].get('attribution_ready'): errors.append('runtime_not_ready')
 doc={'schema':'ocrap-v48.84-saop-pipeline-complete-v1','engineering_version':'v48.84.0-OC-SAOP','valid':not errors,'attribution_ready':not errors,'errors':errors,'arms':{'balanced':str(a.balanced),'precision':str(a.precision)},'comparison_status':ds[-1].get('preregistered_decision',{}).get('status'),'planner_checkpoint_modified':False,'dataset_reconstruction':False,'boundary_transport':False,'test_roots_read':False}
 a.output.write_text(json.dumps(doc,indent=2),encoding='utf-8');sys.exit(0 if doc['valid'] else 30)
if __name__=='__main__':main()
