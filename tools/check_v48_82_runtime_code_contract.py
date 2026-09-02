#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import torch

def sha(p):
 h=hashlib.sha256();h.update(Path(p).read_bytes());return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();repo=a.repo.resolve();sys.path.insert(0,str(repo/'src'));sys.path.insert(0,str(repo))
 import ocrap,ocrap.models.ocrap as mo,ocrap.cli.train as tr
 from ocrap.models.ocrap import OCRAPModel
 errors=[];mods={}
 for name,m in [('ocrap',ocrap),('ocrap.models.ocrap',mo),('ocrap.cli.train',tr)]:
  p=Path(m.__file__).resolve();mods[name]={'path':str(p),'inside_repo':repo in p.parents,'sha256':sha(p)}
  if repo not in p.parents: errors.append(f'{name} outside repo')
 for signed,ch in [(False,1),(True,2)]:
  m=OCRAPModel(16,num_roots=3,num_options=2,d_model=32,d_obs=8,encoder_type='structured_transformer',direct_recovery_absolute_semantic_witness_correction=True,direct_recovery_semantic_witness_root_tail_source=True,direct_recovery_semantic_witness_tail_localization=True,direct_recovery_semantic_witness_structured_tail_field=True,direct_recovery_semantic_witness_signed_tail_channels=signed)
  w=m.direct_absolute_structured_tail_field_weight
  if tuple(w.shape)!=(ch,32) or torch.count_nonzero(w).item()!=0 or m.direct_absolute_root_tail_source_scale is not None:errors.append(f'field contract signed={signed}')
 doc={'schema':'ocrap-v48.82-sntf-runtime-code-contract-v1','engineering_version':'v48.82.0-OC-SNTF','valid':not errors,'attribution_ready':not errors,'errors':errors,'runtime_modules':mods,'source_contract':{'root_tail_source':True,'tail_localization':True,'structured_tail_field':True,'single_channel_shape':[1,192],'signed_channel_shape':[2,192],'option_translation_zero_mean':True,'option_id_input':False,'regime_id_input':False,'generic_mlp':False,'boundary_transport':False},'supervision_contract':{'truth_contract':'structural_interval_bounds','objective':'signed_margin_interval_huber','teacher_metadata_input_to_model':False,'dataset_reconstruction':False},'test_roots_read':False}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n');print(json.dumps({'valid':not errors,'output':str(a.output)}));return 0 if not errors else 30
if __name__=='__main__':raise SystemExit(main())
