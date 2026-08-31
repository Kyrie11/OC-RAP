#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, importlib, json, os, sys
from pathlib import Path
os.environ.setdefault("OCRAP_V48_74_SIGNED_VIABILITY", "1")

def sha(p: Path) -> str:
    h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()

def _model_cfg(response: bool) -> dict:
    return {
        'direct_recovery_absolute_semantic_witness_correction': True,
        'direct_recovery_semantic_witness_active_set_alignment': True,
        'direct_recovery_semantic_witness_route_alignment': True,
        'direct_recovery_semantic_witness_reentry_alignment': True,
        'direct_recovery_semantic_witness_control_projection': True,
        'direct_recovery_semantic_witness_projection_fidelity_weighting': True,
        'direct_recovery_semantic_witness_interaction_box_support': True,
        'direct_recovery_semantic_witness_interaction_hull_support': True,
        'direct_recovery_semantic_witness_interaction_anchor_support': True,
        'direct_recovery_semantic_witness_interaction_response_support': bool(response),
    }

def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("--repo",default="."); ap.add_argument("--output",required=True); a=ap.parse_args()
    repo=Path(a.repo).resolve()
    # The checker must prove the identity of the code that the launcher will
    # actually import, not merely that some import succeeded.
    sys.path.insert(0,str(repo/'src')); sys.path.insert(0,str(repo))
    expected={
        'ocrap': repo/'src/ocrap/__init__.py',
        'ocrap.data': repo/'src/ocrap/data/__init__.py',
        'ocrap.cli.train': repo/'src/ocrap/cli/train.py',
        'ocrap.models.data': repo/'src/ocrap/models/data.py',
        'ocrap.models.ocrap': repo/'src/ocrap/models/ocrap.py',
        'ocrap.models.inference': repo/'src/ocrap/models/inference.py',
        'ocrap.v48_74_signed_viability': repo/'src/ocrap/v48_74_signed_viability.py',
    }
    paths={}; errors=[]
    modules={}
    for name,want in expected.items():
        try:
            m=importlib.import_module(name); modules[name]=m; p=Path(m.__file__).resolve(); want=want.resolve()
            exact=p==want
            paths[name]={"path":str(p),"expected_path":str(want),"sha256":sha(p),"inside_repo":repo in p.parents,"exact_path":exact}
            if not exact: errors.append(f"runtime import path mismatch: {name}")
        except Exception as e:
            paths[name]={"error":repr(e),"expected_path":str(want.resolve()),"inside_repo":False,"exact_path":False}
            errors.append(f"runtime import failed: {name}")
    mod=modules.get('ocrap.v48_74_signed_viability')
    frag=mod.runtime_contract_fragment() if mod is not None else {}
    if not frag.get("enabled"): errors.append("V48.74 signed-viability mode not enabled")
    if frag.get("schema")!=10 or frag.get("feature_dim")!=22: errors.append("schema/dimension mismatch")
    if frag.get("source")!='signed_finite_time_viability_projected_recovery_witness': errors.append("feature source mismatch")
    if frag.get("engineering_version")!='v48.74.1-OC-SVBW-ENGFIX': errors.append("engineering version mismatch")

    train=modules.get('ocrap.cli.train'); data=modules.get('ocrap.models.data')
    serializer={}
    if train is not None:
        for label,response in (("P74_FIRST_ORDER_SVBW",False),("Q74_MAIN_OC_SVBW",True)):
            try:
                schema,source=train._semantic_witness_checkpoint_feature_contract(_model_cfg(response))
                serializer[label]={"schema":int(schema),"source":str(source)}
                if (int(schema),str(source))!=(10,'signed_finite_time_viability_projected_recovery_witness'):
                    errors.append(f"{label} checkpoint serializer contract mismatch")
            except Exception as e:
                serializer[label]={"error":repr(e)}; errors.append(f"{label} checkpoint serializer failed")
    if data is not None:
        if int(getattr(data,'DIRECT_SIGNED_VIABILITY_RECOVERY_WITNESS_FEATURE_SCHEMA',-1))!=10:
            errors.append('models.data schema-10 constant mismatch')
        if int(getattr(data,'DIRECT_SIGNED_VIABILITY_RECOVERY_WITNESS_FEATURE_DIM',-1))!=22:
            errors.append('models.data feature-dim constant mismatch')

    out={
        "valid":not errors,"attribution_ready":not errors,"errors":errors,
        "contract":frag,"runtime_modules":paths,"checkpoint_serializer":serializer,
        "test_roots_read":False,"dataset_reconstruction":False,
    }
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps(out,indent=2,sort_keys=True)); return 0 if not errors else 30
if __name__=="__main__": raise SystemExit(main())
