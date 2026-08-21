#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import torch

ALLOWED_PREFIX = 'direct_absolute_feasibility_head.'

def load(path: Path):
    try: return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError: return torch.load(path, map_location='cpu')

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()

def main() -> int:
    ap=argparse.ArgumentParser(description='v48.58 fail-closed frozen-Stage-I state isolation audit')
    ap.add_argument('--reference', type=Path, required=True)
    ap.add_argument('--adapted', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a=ap.parse_args(); ra=load(a.reference); ca=load(a.adapted)
    rs=ra.get('model_state', ra); cs=ca.get('model_state', ca)
    if not isinstance(rs,dict) or not isinstance(cs,dict): raise SystemExit('checkpoint model_state missing')
    shared=sorted(set(rs)&set(cs)); changed=[]; shape=[]
    for k in shared:
        if tuple(rs[k].shape)!=tuple(cs[k].shape): shape.append(k); continue
        if not torch.equal(rs[k].cpu(), cs[k].cpu()): changed.append(k)
    removed=sorted(set(rs)-set(cs)); added=sorted(set(cs)-set(rs))
    disallowed_changed=[k for k in changed if not k.startswith(ALLOWED_PREFIX)]
    disallowed_added=[k for k in added if not k.startswith(ALLOWED_PREFIX)]
    allowed_added=[k for k in added if k.startswith(ALLOWED_PREFIX)]
    expected={'direct_absolute_feasibility_head.weight','direct_absolute_feasibility_head.bias'}
    valid=(not removed and not shape and not disallowed_changed and not disallowed_added and set(allowed_added)==expected)
    doc={
      'schema':'ocrap-v48.58-rifa-state-isolation-v1','valid':valid,
      'reference':str(a.reference.resolve(strict=False)),'adapted':str(a.adapted.resolve(strict=False)),
      'reference_sha256':sha(a.reference),'adapted_sha256':sha(a.adapted),
      'allowed_trainable_prefix':ALLOWED_PREFIX,
      'shared_state_tensors':len(shared),'changed_shared_state_keys':changed,
      'added_state_keys':added,'removed_state_keys':removed,'shape_mismatch_keys':shape,
      'disallowed_changed_state_keys':disallowed_changed,'disallowed_added_state_keys':disallowed_added,
      'expected_new_state_keys_present':set(allowed_added)==expected,
      'stage_i_bitwise_identity':not disallowed_changed and not removed and not shape and not disallowed_added,
    }
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'event':'v48_58_state_isolation','valid':valid,'output':str(a.output)}))
    return 0 if valid else 30
if __name__=='__main__': raise SystemExit(main())
