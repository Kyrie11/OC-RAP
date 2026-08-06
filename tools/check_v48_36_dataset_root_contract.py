#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os,time
from pathlib import Path

def atomic(path:Path,doc:dict)->None:
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(f'.{path.name}.tmp.{os.getpid()}.{time.time_ns()}')
    tmp.write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); os.replace(tmp,path)

def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--protocol-root',type=Path,required=True); ap.add_argument('--safe-root',type=Path,required=True)
    ap.add_argument('--train-near',type=Path,required=True); ap.add_argument('--train-contact',type=Path,required=True)
    ap.add_argument('--dev-near',type=Path,required=True); ap.add_argument('--dev-contact',type=Path,required=True)
    ap.add_argument('--cert-near',type=Path,required=True); ap.add_argument('--cert-contact',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
    expected={
      'train_near':'evidence_adapt_train_near_contact','train_contact':'evidence_adapt_train_contact',
      'dev_near':'evidence_adapt_dev_near_contact','dev_contact':'evidence_adapt_dev_contact',
      'cert_near':'certificate_pool_near_contact','cert_contact':'certificate_pool_contact'}
    paths={k:getattr(args,k.replace('train_','train_').replace('dev_','dev_').replace('cert_','cert_')) for k in expected}
    root=args.protocol_root.resolve(strict=False)
    checks={}
    details={}
    for key,leaf in expected.items():
        p=paths[key].resolve(strict=False); exp=(root/leaf).resolve(strict=False)
        checks[f'{key}_canonical_path']=p==exp
        checks[f'{key}_exists']=p.is_dir()
        details[key]={'actual':str(p),'expected':str(exp),'leaf':p.name}
    checks['safe_root_exists']=args.safe_root.resolve(strict=False).is_dir()
    forbidden={'traincontact','valcontact','calibrationcontact','testcontact','trainnearcontact','valnearcontact','calibrationnearcontact','testnearcontact'}
    aliases=[]
    for p in [root,*paths.values()]:
        if p.name.lower().replace('_','').replace('-','') in forbidden: aliases.append(str(p))
    checks['no_legacy_alias_selected']=not aliases
    checks['near_contact_and_contact_distinct']=len({str(paths[k].resolve(strict=False)) for k in expected})==len(expected)
    doc={'event':'v48_36_dataset_root_contract','version':'v48.36-OCAF','created_unix':time.time(),'valid':all(checks.values()),
         'checks':checks,'paths':details,'legacy_aliases':aliases,'test_roots_read':False}
    atomic(args.output,doc); print(json.dumps(doc,ensure_ascii=False)); return 0 if doc['valid'] else 4
if __name__=='__main__': raise SystemExit(main())
