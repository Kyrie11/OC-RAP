#!/usr/bin/env python3
"""Audit teacher component semantics without reading test roots.

This diagnostic compares the legacy nominal-relative component veto with the
v48.56 decision-role proposal: DRS remains a nominal-relative boundary term,
DEP uses the absolute teacher R_dep=0 boundary, and GAP remains in PCD/order but
is not an independent hard veto.
"""
from __future__ import annotations
import argparse, json, math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

NAMES=("drs","deployability","gap","hard_rule","harm_proxy")

def rows(p:Path)->list[dict[str,Any]]:
    if not p.is_file(): return []
    return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]

def sig(x:float)->float:
    x=max(-60.0,min(60.0,float(x))); return 1.0/(1.0+math.exp(-x))

def terms(c:dict[str,Any], n:dict[str,Any], *, dep_abs:bool, gap_ord:bool)->list[float]:
    cdep=sig(c['teacher_r_dep']); ndep=sig(n['teacher_r_dep'])
    cg=math.exp(-max(0.0,min(float(c['teacher_gap']),20.0)))
    ng=math.exp(-max(0.0,min(float(n['teacher_gap']),20.0)))
    return [
      float(n['teacher_drs'])-float(c['teacher_drs'])-0.05,
      0.5-cdep if dep_abs else ndep-cdep-0.05,
      -0.05 if gap_ord else ng-cg-0.05,
      float(c.get('teacher_hard_violation',0.0))-float(n.get('teacher_hard_violation',0.0))-0.05,
      float(c.get('teacher_harm_proxy',0.0))-float(n.get('teacher_harm_proxy',0.0))-0.05,
    ]

def audit_index(rs:list[dict[str,Any]], gain:float)->dict[str,Any]:
    groups=defaultdict(list)
    for r in rs: groups[(int(r.get('bucket',-1)),str(r.get('scene')),int(r.get('time',-1)))].append(r)
    out={}
    for bucket,name in ((1,'near'),(2,'contact')):
      ctr=Counter(); conflict_causes=Counter(); drac_conflict_causes=Counter(); rdep_current=Counter()
      for (b,_,_),xs in groups.items():
        if b!=bucket: continue
        noms=[x for x in xs if x.get('nominal')]
        if not noms: continue
        nom=noms[0]
        for c in xs:
          if c.get('nominal'): continue
          if int(c.get('macro',-1)) not in {2,3,5,6,7}: continue
          ctr['candidates']+=1
          adv=float(c['teacher_pcd'])-float(nom['teacher_pcd'])
          raw=adv>=gain
          old=terms(c,nom,dep_abs=False,gap_ord=False); new=terms(c,nom,dep_abs=True,gap_ord=True)
          oh=max(old)>0; nh=max(new)>0
          abs_nondeploy=float(c['teacher_r_dep'])<0.0
          ctr['raw_beneficial']+=int(raw); ctr['legacy_harmful']+=int(oh); ctr['drac_harmful']+=int(nh)
          ctr['legacy_safe_beneficial']+=int(raw and not oh); ctr['drac_safe_beneficial']+=int(raw and not nh)
          ctr['legacy_raw_benefit_harm_conflict']+=int(raw and oh); ctr['drac_raw_benefit_harm_conflict']+=int(raw and nh)
          ctr['legacy_drac_harm_disagreement']+=int(oh!=nh)
          if oh: rdep_current['legacy_harmful']+=1
          if abs_nondeploy: rdep_current['rdep_lt_zero']+=1
          if oh and abs_nondeploy: rdep_current['overlap']+=1
          if raw and oh:
            for i,k in enumerate(NAMES): conflict_causes[k]+=int(old[i]>0)
          if raw and nh:
            for i,k in enumerate(NAMES): drac_conflict_causes[k]+=int(new[i]>0)
      out[name]={**dict(ctr),
        'legacy_conflict_component_counts':dict(conflict_causes),
        'drac_conflict_component_counts':dict(drac_conflict_causes),
        'legacy_harm_vs_absolute_rdep':dict(rdep_current),
      }
    return out

def audit_proposals(rs:list[dict[str,Any]], gain:float)->dict[str,Any]:
    ctr=Counter(); causes=Counter()
    for r in rs:
      adv=float(r.get('teacher_adv',0.0)); harmful=bool(r.get('teacher_harmful',False)); raw=adv>=gain
      ctr['candidates']+=1; ctr['raw_beneficial']+=int(raw); ctr['harmful']+=int(harmful); ctr['raw_benefit_harm_conflict']+=int(raw and harmful)
      ts=r.get('teacher_component_veto_terms')
      if raw and harmful and isinstance(ts,list):
        for i,k in enumerate(NAMES): causes[k]+=int(i<len(ts) and float(ts[i])>0)
    return {**dict(ctr),'conflict_component_counts':dict(causes)}

def main()->int:
  ap=argparse.ArgumentParser(); ap.add_argument('--run',type=Path,required=True); ap.add_argument('--positive-gain',type=float,default=.015); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
  train=rows(a.run/'evidence_adapt_teacher_pcd_index.jsonl'); dev=rows(a.run/'evidence_adapt_dev_teacher_pcd_index.jsonl')
  cert={}
  for reg in ('near','contact'):
    cert[reg]=audit_proposals(rows(a.run/'candidates'/'precision'/'calibration'/f'direct_value_risk_{reg}_v48.proposal_rows.jsonl'),a.positive_gain)
  doc={'event':'v48_56_teacher_component_correctness_audit','version':'v48.56-DCP-DRFC-BCDE-DRAC','source_run':str(a.run),'positive_gain':a.positive_gain,
       'train_index':audit_index(train,a.positive_gain),'dev_index':audit_index(dev,a.positive_gain),'certificate_legacy_proposal_conflicts':cert,
       'interpretation_contract':{
         'legacy_dep':'sigmoid(R_dep_nom)-sigmoid(R_dep_candidate)-0.05','drac_dep':'0.5-sigmoid(R_dep_candidate); positive iff R_dep<0',
         'legacy_gap':'exp(-G_nom)-exp(-G_candidate)-0.05 hard veto','drac_gap':'kept in PCD/order; neutral in independent hard veto',
         'drs':'nominal DRS - candidate DRS - 0.05','test_roots_read':False},'test_roots_read':False}
  a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(doc,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8'); print(json.dumps(doc,ensure_ascii=False)); return 0
if __name__=='__main__': raise SystemExit(main())
