#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np

KINDS = {
    'dev_near': 'dev_diagnostic_near_v48.proposal_rows.jsonl',
    'dev_contact': 'dev_diagnostic_contact_v48.proposal_rows.jsonl',
    'certificate_near': 'direct_value_risk_near_v48.proposal_rows.jsonl',
    'certificate_contact': 'direct_value_risk_contact_v48.proposal_rows.jsonl',
}
VARIANTS = ('balanced', 'precision')


def read_rows(run: Path, variant: str, split: str):
    p = run / 'candidates' / variant / 'calibration' / KINDS[split]
    if not p.is_file(): raise FileNotFoundError(p)
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]

def f(r, k, default=float('nan')):
    try: return float(r.get(k, default))
    except Exception: return default

def key(r): return (str(r.get('scene', '')), int(r.get('time', -1)), int(r.get('candidate', -1)))
def feasible(r): return f(r, 'teacher_candidate_r_dep') >= 0.0
def safe(r): return f(r, 'teacher_adv', -1e9) >= .015 and not bool(r.get('teacher_harmful', False))
def harmful(r): return bool(r.get('teacher_harmful', False))
def poscert(r): return f(r, 'semantic_best_common_viability', -1e9) > 0.0
def prob(r): return min(1-1e-12, max(1e-12, f(r, 'absolute_feasibility_probability')))
def margin(r):
    p = prob(r); return math.log(p/(1-p))
def huber(a, b):
    d = abs(float(a)-float(b)); return .5*d*d if d < 1. else d-.5

def auc(y, s):
    y = np.asarray(y, dtype=bool); s = np.asarray(s, dtype=float); ok = np.isfinite(s); y = y[ok]; s = s[ok]
    p = s[y]; n = s[~y]
    if not len(p) or not len(n): return None
    return float(((p[:,None] > n[None,:]).sum() + .5*(p[:,None] == n[None,:]).sum()) / (len(p)*len(n)))

def mean(rows, fn): return float(np.mean([fn(r) for r in rows])) if rows else None

def load_truth(path: Path):
    out = {}
    for ln, raw in enumerate(path.read_text().splitlines(), start=1):
        if not raw.strip(): continue
        r = json.loads(raw); role = str(r.get('dataset_role', '')); k = (role, str(r.get('scene_id','')), int(r.get('time_index',-1)), int(r.get('candidate_index',-1)))
        if k in out: raise ValueError(f'duplicate truth-index key at line {ln}: {k}')
        if not bool(r.get('valid', False)): raise ValueError(f'invalid truth-index row at line {ln}: {k}')
        out[k] = r
    return out

def attach(rows, split, truth):
    out = []
    missing = []
    for r in rows:
        k0 = key(r); t = truth.get((split, k0[0], k0[1], k0[2]))
        if t is None:
            missing.append(k0); continue
        out.append((r,t))
    if missing or len(out) != len(rows): raise ValueError(f'{split}: truth-index alignment failed missing={len(missing)} examples={missing[:3]}')
    return out

def subgroup_stats(pairs, physical: bool | None):
    z = [r for r,t in pairs if physical is None or bool(t.get('physical_identifiable', False)) == physical]
    pos = [r for r in z if feasible(r)]; neg = [r for r in z if not feasible(r)]
    safe_rows = [r for r in z if safe(r)]; harmful_rows = [r for r in z if harmful(r)]; ti = neg
    exact05 = [r for r in z if abs(f(r,'teacher_candidate_r_dep')-.5) <= 1e-8]
    return {
        'rows': len(z), 'teacher_feasible_rows': len(pos), 'teacher_infeasible_rows': len(neg),
        'source_auc': auc([feasible(r) for r in z], [prob(r) for r in z]),
        'signed_margin_huber': mean(z, lambda r: huber(margin(r), f(r,'teacher_candidate_r_dep'))),
        'signed_margin_mae': mean(z, lambda r: abs(margin(r)-f(r,'teacher_candidate_r_dep'))),
        'safe_positive_rows': len(safe_rows), 'safe_positive_pass_fraction': mean(safe_rows, lambda r: prob(r)>=.5),
        'harmful_rows': len(harmful_rows), 'harmful_pass_fraction': mean(harmful_rows, lambda r: prob(r)>=.5),
        'teacher_infeasible_pass_fraction': mean(ti, lambda r: prob(r)>=.5),
        'positive_certificate_rows': sum(poscert(r) for r in z),
        'positive_certificate_teacher_feasible_precision': mean([r for r in z if poscert(r)], feasible),
        'exact_0p5_rows': len(exact05), 'exact_0p5_fraction': len(exact05)/len(z) if z else None,
    }

def summarize(rows, split, truth):
    pairs = attach(rows, split, truth)
    exposures = [float(t.get('structural_exposure_mass', 0.0)) for _r,t in pairs]
    return {
        'full': subgroup_stats(pairs, None), 'physical_identifiable': subgroup_stats(pairs, True),
        'structurally_exposed': subgroup_stats(pairs, False),
        'physical_identifiable_fraction': float(np.mean([bool(t.get('physical_identifiable',False)) for _r,t in pairs])) if pairs else None,
        'mean_structural_exposure_mass': float(np.mean(exposures)) if exposures else None,
        'max_structural_exposure_mass': max(exposures, default=0.0),
    }

def compare(base, new, split, truth):
    bm = {key(r):r for r in base}; nm = {key(r):r for r in new}; common = sorted(set(bm)&set(nm))
    if set(bm) != set(nm): raise ValueError(f'{split}: row mismatch base={len(bm)} new={len(nm)} common={len(common)}')
    pairs = [(bm[k],nm[k],truth.get((split,k[0],k[1],k[2]))) for k in common]
    if any(t is None for _a,_b,t in pairs): raise ValueError(f'{split}: missing truth-index rows in comparison')
    labels = all(abs(f(a,'teacher_candidate_r_dep')-f(b,'teacher_candidate_r_dep'))<=1e-7 and bool(a.get('teacher_harmful',False))==bool(b.get('teacher_harmful',False)) for a,b,_t in pairs)
    cert = all(poscert(a)==poscert(b) for a,b,_t in pairs)
    def one(phys):
        z = [(a,b) for a,b,t in pairs if bool(t.get('physical_identifiable',False)) == phys]
        ba = auc([feasible(a) for a,b in z],[prob(a) for a,b in z]); na = auc([feasible(a) for a,b in z],[prob(b) for a,b in z])
        bh = float(np.mean([huber(margin(a),f(a,'teacher_candidate_r_dep')) for a,b in z])) if z else None
        nh = float(np.mean([huber(margin(b),f(b,'teacher_candidate_r_dep')) for a,b in z])) if z else None
        safez=[(a,b) for a,b in z if safe(a)]
        return {'rows':len(z), 'teacher_feasible_rows':sum(feasible(a) for a,b in z), 'teacher_infeasible_rows':sum(not feasible(a) for a,b in z),
                'auc_base':ba,'auc_new':na,'auc_delta':None if ba is None or na is None else na-ba,
                'huber_base':bh,'huber_new':nh,'huber_delta':None if bh is None or nh is None else nh-bh,
                'safe_positive_rows':len(safez),'safe_positive_pass_delta':float(np.mean([prob(b)>=.5 for a,b in safez])-np.mean([prob(a)>=.5 for a,b in safez])) if safez else None}
    return {'aligned_rows':len(pairs),'teacher_labels_equal':labels,'positive_certificate_set_equal':cert,
            'physical_identifiable':one(True),'structurally_exposed':one(False)}

def state(run: Path, v: str):
    p = run/'candidates'/v/'V48_79_STAGE_I_STATE_ISOLATION.json'
    if not p.is_file(): return None
    d = json.loads(p.read_text()); return {'valid':d.get('valid'),'root_tail_source_scale':d.get('raw_root_tail_source_scale'),'best_checkpoint':d.get('adapted')}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--j78',type=Path,required=True); ap.add_argument('--k79',type=Path,required=True); ap.add_argument('--truth-index',type=Path,required=True); ap.add_argument('--truth-summary',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    truth=load_truth(a.truth_index); summary=json.loads(a.truth_summary.read_text())
    arms={}
    for name,run in [('J78_RTSI',a.j78),('K79_PHYSICAL_TAIL_PROBE',a.k79)]:
        arms[name]={v:{'state':state(run,v) if name.startswith('K79') else None,'splits':{s:summarize(read_rows(run,v,s),s,truth) for s in KINDS}} for v in VARIANTS}
    comps={'K79_minus_J78':{v:{s:compare(read_rows(a.j78,v,s),read_rows(a.k79,v,s),s,truth) for s in KINDS} for v in VARIANTS}}
    doc={'schema':'ocrap-v48.79-pstc-audit-v1','engineering_version':'v48.79.0-OC-PSTC','arms':arms,'comparisons':comps,
         'truth_index_summary':summary,'truth_contract':'censor_structural_tail','objective':'signed_margin_huber','huber_beta':1.0,
         'intervention':'same J78 nested zero-translation one-scalar source; only absolute-source supervision is censored to candidates whose exact nested teacher OC-MERO active tail has zero conservative structural exposure',
         'teacher_labels_changed':False,'dataset_reconstruction':False,'teacher_future_input_to_model':False,'test_roots_read':False}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n'); print(json.dumps({'event':'v48_79_pstc_audit','output':str(a.output)}))

if __name__=='__main__': main()
