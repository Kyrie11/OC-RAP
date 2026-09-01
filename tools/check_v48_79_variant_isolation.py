#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import torch


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for z in iter(lambda: f.read(1 << 20), b''):
            h.update(z)
    return h.hexdigest()


def load_json(p: Path): return json.loads(p.read_text())
def load_ckpt(p: Path):
    try: return torch.load(p, map_location='cpu', weights_only=False)
    except TypeError: return torch.load(p, map_location='cpu')

def env(p: Path):
    out = {}
    for raw in p.read_text().splitlines():
        z = raw.strip()
        if z and not z.startswith('#') and '=' in z:
            k, v = z.split('=', 1); out[k] = v
    return out

def res(x): return str(Path(x).expanduser().resolve(strict=False))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference-run', type=Path, required=True)
    ap.add_argument('--reference-contract', type=Path, required=True)
    ap.add_argument('--pstc-run', type=Path, required=True)
    ap.add_argument('--truth-index', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args(); errors = []; rc = load_json(a.reference_contract); snap = rc.get('reference_candidate_checkpoint_sha256') or {}; variants = {}; paths = []
    if not rc.get('valid'): errors.append('reference contract invalid')
    for v in ('balanced', 'precision'):
        base = a.pstc_run / 'candidates' / v
        ref = a.reference_run / 'candidates' / v / 'model_v48_trac_sr' / 'best.pt'
        dst = base / 'model_v48_trac_sr' / 'best.pt'
        summ = base / 'model_v48_trac_sr' / 'train_summary.json'
        state = base / 'V48_79_STAGE_I_STATE_ISOLATION.json'
        pol = base / 'POLICY_CONTRACT.env'; complete = base / 'TRAINING_COMPLETE.json'; ecp = base / 'EVIDENCE_CORRECTION_COMPLETE.json'
        miss = [str(p) for p in (ref, dst, summ, state, pol, complete, ecp) if not p.is_file()]
        if miss:
            errors.append(f'{v}: missing {miss}'); variants[v] = {'valid': False}; continue
        sd, st, tc, ec = load_json(summ), load_json(state), load_json(complete), load_json(ecp); pe = env(pol)
        refsha, dstsha = sha(ref), sha(dst); ck = load_ckpt(dst); tcfg = ((ck.get('cfg') or {}).get('training') or {})
        trainable = list(sd.get('trainable_param_prefixes') or [])
        expected_idx = res(a.truth_index)
        checks = {
            'reference_snapshot_matches': str(snap.get(v, '')) == refsha,
            'init_checkpoint_matches': res(sd.get('init_checkpoint', '')) == res(ref),
            'output_checkpoint_matches': res(sd.get('checkpoint', '')) == res(dst),
            'trainable_prefix_exact': trainable == ['direct_absolute_root_tail_source_scale'],
            'stage_i_isolation': bool(st.get('valid')) and bool(st.get('stage_i_bitwise_identity')) and st.get('added_state_keys') == ['direct_absolute_root_tail_source_scale'],
            'feature_contract': bool(st.get('semantic_witness_feature_contract_valid')),
            'factor_flags': bool(st.get('factor_flags_valid')) and bool((st.get('factor_flags') or {}).get('root_tail_source')) and bool((st.get('factor_flags') or {}).get('tail_localization')),
            'truth_contract_checkpoint': str(st.get('absolute_feasibility_truth_contract')) == 'censor_structural_tail',
            'objective_checkpoint': str(st.get('absolute_feasibility_supervision_objective')) == 'signed_margin_huber',
            'truth_index_checkpoint': res(st.get('absolute_feasibility_truth_index', '')) == expected_idx,
            'truth_contract_summary': str(tcfg.get('direct_value_absolute_feasibility_truth_contract')) == 'censor_structural_tail',
            'objective_summary': str(tcfg.get('direct_value_absolute_feasibility_supervision_objective')) == 'signed_margin_huber',
            'truth_index_summary': res(tcfg.get('direct_value_absolute_feasibility_truth_index', '')) == expected_idx,
            'truth_contract_policy': pe.get('ABSOLUTE_FEASIBILITY_TRUTH_CONTRACT') == 'censor_structural_tail',
            'objective_policy': pe.get('ABSOLUTE_FEASIBILITY_SUPERVISION_OBJECTIVE') == 'signed_margin_huber',
            'truth_index_policy': res(pe.get('ABSOLUTE_FEASIBILITY_TRUTH_INDEX', '')) == expected_idx,
            'policy_mode': pe.get('ABSOLUTE_FEASIBILITY_MODE') == 'learned', 'policy_threshold': pe.get('ABSOLUTE_FEASIBILITY_THRESHOLD') == '0.5',
            'training_complete_sha_matches': str(tc.get('checkpoint_sha256', '')) == dstsha,
            'evidence_complete_sha_matches': str(ec.get('checkpoint_sha256', '')) == dstsha,
            'evidence_complete_source_matches': res(ec.get('source_checkpoint', '')) == res(ref),
            'evidence_complete_trainable_exact': list(ec.get('trainable_prefixes') or []) == ['direct_absolute_root_tail_source_scale'] and int(ec.get('trainable_state_params', -1)) == int(st.get('new_tensor_numel', -2)),
            'evidence_complete_no_regime_input': ec.get('regime_id_exposed_to_evidence_model') is False and ec.get('test_roots_read') is False,
        }
        ok = all(checks.values())
        if not ok: errors.append(f'{v}: failed {[k for k,z in checks.items() if not z]}')
        variants[v] = {'valid': ok, 'checks': checks, 'reference_sha256': refsha, 'pstc_checkpoint_sha256': dstsha, 'root_tail_source_scale': st.get('raw_root_tail_source_scale')}
        paths.append(res(dst))
    distinct = len(paths) == 2 and len(set(paths)) == 2
    if not distinct: errors.append('balanced/precision PSTC checkpoints not distinct')
    valid = not errors and all(z.get('valid') for z in variants.values()) and distinct
    doc = {'schema': 'ocrap-v48.79-pstc-variant-isolation-v1', 'valid': valid,
           'reference_run': res(a.reference_run), 'pstc_run': res(a.pstc_run),
           'absolute_feasibility_truth_contract': 'censor_structural_tail',
           'absolute_feasibility_supervision_objective': 'signed_margin_huber',
           'absolute_feasibility_truth_index': res(a.truth_index), 'variants': variants,
           'distinct_checkpoint_paths': distinct, 'errors': errors,
           'dataset_reconstruction': False, 'teacher_labels_changed': False, 'test_roots_read': False}
    a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(doc, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'event': 'v48_79_variant_isolation', 'valid': valid, 'output': str(a.output)}))
    return 0 if valid else 30


if __name__ == '__main__': raise SystemExit(main())
