#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path

def _sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

ROLE_TO_SOURCE = {
    'evidence_adapt_train_near_contact': 'near',
    'evidence_adapt_dev_near_contact': 'near',
    'certificate_pool_near_contact': 'near',
    'evidence_adapt_train_contact': 'contact',
    'evidence_adapt_dev_contact': 'contact',
    'certificate_pool_contact': 'contact',
}

def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--protocol-root',type=Path,required=True)
    ap.add_argument('--cal-near',type=Path,required=True)
    ap.add_argument('--cal-contact',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    errors=[]; roles={}; total=0; hardlinks=0; byte_copies=0; missing=0; stat_mismatch=0; content_mismatch=0
    sources={'near':a.cal_near.resolve(),'contact':a.cal_contact.resolve()}
    for k,p in [('protocol',a.protocol_root),('near',a.cal_near),('contact',a.cal_contact)]:
        if not p.is_dir(): errors.append(f'missing {k} root: {p}')
    if not errors:
        for role,kind in ROLE_TO_SOURCE.items():
            r=a.protocol_root/role/'samples'
            if not r.is_dir():
                continue
            src=sources[kind]/'samples'
            count=hl=copies=miss=bad=content_bad=0
            for p in r.glob('*.npz'):
                count+=1; total+=1
                q=src/p.name
                if not q.is_file():
                    miss+=1; missing+=1; continue
                sp=p.stat(); sq=q.stat()
                if sp.st_size != sq.st_size:
                    bad+=1; stat_mismatch+=1; continue
                if sp.st_dev==sq.st_dev and sp.st_ino==sq.st_ino:
                    hl+=1; hardlinks+=1
                else:
                    if _sha(p) == _sha(q):
                        copies+=1; byte_copies+=1
                    else:
                        content_bad+=1; content_mismatch+=1
            roles[role]={'source':kind,'samples':count,'hardlink_identical':hl,'byte_identical_copies':copies,'missing_in_source':miss,'size_mismatch':bad,'content_mismatch':content_bad}
            if miss or bad or content_bad:
                errors.append(f'{role}: source mismatch missing={miss} size_mismatch={bad} content_mismatch={content_bad}')
    seal=a.protocol_root/'V48_45_PROTOCOL_SEAL.json'
    seal_present=seal.is_file()
    doc={
        'schema':'ocrap-v48.91-calibration-source-contract-v1',
        'valid':not errors,'errors':errors,'protocol_root':str(a.protocol_root.resolve()),
        'cal_near':str(a.cal_near.resolve()),'cal_contact':str(a.cal_contact.resolve()),
        'protocol_seal_present':seal_present,'roles':roles,'total_protocol_samples_checked':total,
        'hardlink_identical_samples':hardlinks,'byte_identical_copy_samples':byte_copies,'missing_in_source':missing,'size_mismatch':stat_mismatch,'content_mismatch':content_mismatch,
        'interpretation':'protocol roles are deterministic views of calibration_near_contact/contact; rebuilding the protocol cannot repair a stored future realization mismatch when source NPZ bytes are unchanged',
    }
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'valid':doc['valid'],'errors':errors,'checked':total,'hardlinks':hardlinks}))
    return 0 if doc['valid'] else 30
if __name__=='__main__': raise SystemExit(main())
