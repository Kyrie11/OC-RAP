#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from ocrap.v48_100_joint_root_semantic_decoder import ENGINEERING_VERSION
ROLES=('dev_near','dev_contact','certificate_near','certificate_contact')
def sha(p:Path)->str: return hashlib.sha256(p.read_bytes()).hexdigest()
def _result_errors(o:dict,v:str)->list[str]:
    e=[]
    if o.get('engineering_version')!=ENGINEERING_VERSION or not o.get('valid'): e.append(f'{v}_contract')
    if int(o.get('root_query_parameters_trained',-1))!=1536 or int(o.get('recovery_chart_parameters_trained',-1))!=770 or int(o.get('joint_representation_parameters_trained',-1))!=2306: e.append(f'{v}_parameter_contract')
    for k in ('planner_parameters_trained','source_parameters_trained','stage_i_parameters_trained','root_decoder_body_parameters_trained','root_logit_head_parameters_trained'):
        if int(o.get(k,-1))!=0: e.append(f'{v}_{k}')
    for role in ROLES:
        if not ((o.get('evaluation_contracts') or {}).get(role) or {}).get('valid'): e.append(f'{v}_{role}_eval_contract')
        for n in ('state','support_true','reserve_true'):
            m=(((o.get('cells') or {}).get(role) or {}).get(n) or {})
            if int(m.get('rows',0))<=0 or m.get('auc') is None: e.append(f'{v}_{role}_{n}_empty')
    return e

def main()->int:
    ap=argparse.ArgumentParser();
    for x in ('runtime','balanced','precision','balanced_state','precision_state','comparison','v48_99_pipeline','output'): ap.add_argument('--'+x.replace('_','-'),dest=x,type=Path,required=True)
    a=ap.parse_args(); errors=[]; objs={}
    for name,p in (('runtime',a.runtime),('balanced',a.balanced),('precision',a.precision),('comparison',a.comparison)):
        if not p.is_file(): errors.append('missing_'+name); continue
        try: objs[name]=json.loads(p.read_text())
        except Exception as exc: errors.append(f'invalid_{name}:{exc}')
    rt=objs.get('runtime',{})
    if rt.get('engineering_version')!=ENGINEERING_VERSION or not rt.get('valid') or not rt.get('attribution_ready'): errors.append('runtime_contract')
    errors += _result_errors(objs.get('balanced',{}),'balanced')+_result_errors(objs.get('precision',{}),'precision')
    comp=objs.get('comparison',{})
    if comp.get('engineering_version')!=ENGINEERING_VERSION or not comp.get('valid') or not comp.get('attribution_ready'): errors.append('comparison_contract')
    if not a.balanced_state.is_file() or not a.precision_state.is_file(): errors.append('missing_joint_state')
    if not a.v48_99_pipeline.is_file(): errors.append('missing_v48_99_pipeline')
    else:
        v99=json.loads(a.v48_99_pipeline.read_text())
        if not(v99.get('valid') and v99.get('attribution_ready') and v99.get('preregistered_status')=='RECOVERY_JACOBIAN_ALIGNMENT_STOP'): errors.append('v48_99_stop_prerequisite')
        expected=((v99.get('artifacts') or {}).get('comparison') or {}).get('sha256'); actual=comp.get('v48_99_comparison_sha256')
        if not expected or actual!=expected: errors.append('v48_99_provenance_mismatch')
    artifacts={}
    for name,p in (('runtime',a.runtime),('balanced',a.balanced),('precision',a.precision),('balanced_state',a.balanced_state),('precision_state',a.precision_state),('comparison',a.comparison)):
        if p.is_file(): artifacts[name]={'path':str(p.resolve()),'sha256':sha(p)}
    status=(comp.get('preregistered_decision') or {}).get('status')
    out={'schema':'ocrap-v48.100-jrsd-pipeline-complete-v1','engineering_version':ENGINEERING_VERSION,'valid':not errors,'attribution_ready':not errors,'errors':errors,
         'experiment_type':'joint_root_query_and_recovery_chart_semantic_learning','planner_parameters_trained':0,'source_parameters_trained':0,'stage_i_parameters_trained':0,
         'root_decoder_body_parameters_trained':0,'root_logit_head_parameters_trained':0,'root_query_parameters_trained':int((objs.get('balanced') or {}).get('root_query_parameters_trained',0)),
         'recovery_chart_parameters_trained':int((objs.get('balanced') or {}).get('recovery_chart_parameters_trained',0)),
         'joint_representation_parameters_trained':int((objs.get('balanced') or {}).get('joint_representation_parameters_trained',0)),
         'regime_conditioning':False,'relative_ranker_modified':False,'boundary_transport':False,'dataset_reconstruction':False,'dataset_reselection':False,
         'teacher_metadata_input_to_model':False,'test_roots_read':False,'preregistered_status':status,'artifacts':artifacts}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'valid':out['valid'],'status':status,'errors':errors})); return 0 if out['valid'] else 30
if __name__=='__main__': raise SystemExit(main())
