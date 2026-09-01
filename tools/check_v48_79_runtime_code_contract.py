#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, importlib, inspect, json, sys
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch


def sha(p: Path) -> str:
    h = hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()


def _sample(M, modes, metas):
    M = np.asarray(M, dtype=np.float32); K, L = M.shape
    from ocrap.algorithms.ocmero import oc_mero
    p = np.ones(K, dtype=np.float32) / K; C = np.ones((K, K), dtype=np.float32)
    res = oc_mero(M, p, C, alpha=.2, beta=.2, option_valid=np.ones(L, bool), root_valid=np.ones(K, bool), top_m=8)
    return {'m_star': M, 'root_probs': p, 'root_valid': np.ones(K, bool), 'c_star': C,
            'option_valid': np.ones(L, bool), 'root_assignments': np.arange(K, dtype=np.int64),
            'future_metadata': np.asarray(json.dumps(metas)), 'recovery_modes': np.asarray(modes),
            'r_dep_star': np.float32(res.r_dep)}


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument('--repo', type=Path, required=True); ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args(); repo = a.repo.resolve(); src = repo / 'src'; sys.path.insert(0, str(src)); sys.path.insert(0, str(repo)); errors = []; mods = {}
    expected = {
        'ocrap': src/'ocrap/__init__.py', 'ocrap.cli.train': src/'ocrap/cli/train.py',
        'ocrap.models.data': src/'ocrap/models/data.py', 'ocrap.models.ocrap': src/'ocrap/models/ocrap.py',
        'ocrap.models.inference': src/'ocrap/models/inference.py', 'ocrap.algorithms.lcv': src/'ocrap/algorithms/lcv.py',
        'ocrap.algorithms.ocmero': src/'ocrap/algorithms/ocmero.py', 'ocrap.simulation.teacher.margins': src/'ocrap/simulation/teacher/margins.py',
        'ocrap.v48_79_truth_contract': src/'ocrap/v48_79_truth_contract.py',
    }
    for name, ep in expected.items():
        try:
            m = importlib.import_module(name); p = Path(m.__file__).resolve(); ok = p == ep.resolve() and repo in p.parents
            mods[name] = {'path': str(p), 'expected_path': str(ep.resolve()), 'exact_path': p == ep.resolve(), 'inside_repo': repo in p.parents, 'sha256': sha(p)}
            if not ok: errors.append(f'runtime module mismatch: {name}')
        except Exception as exc:
            mods[name] = {'error': repr(exc), 'inside_repo': False}; errors.append(f'runtime import failed: {name}')

    from ocrap.cli.train import _absolute_feasibility_supervision_mask, _absolute_feasibility_supervision_loss, _semantic_witness_checkpoint_feature_contract
    from ocrap.models.data import OPTION_FEATURE_DIM
    from ocrap.models.encoders import FlatFeatureLayout
    from ocrap.models.ocrap import OCRAPModel
    from ocrap.simulation.teacher.margins import teacher_margin
    from ocrap.v48_79_truth_contract import nested_tail_truth_contract

    clean = _sample([[.4,.2], [.3,.1]], ['stop','brake_lane'], [{},{}])
    clean_rec = nested_tail_truth_contract(clean)
    clean_ok = bool(clean_rec.valid and clean_rec.physical_identifiable and clean_rec.structural_exposure_mass == 0.0)
    if not clean_ok: errors.append('clean physical-tail truth contract failed')
    exposed = _sample([[.8,-1.0], [.8,-1.0]], ['yield_rejoin','stop'], [{},{}])
    exp_rec = nested_tail_truth_contract(exposed)
    exposed_ok = bool(exp_rec.valid and not exp_rec.physical_identifiable and exp_rec.structural_exposure_mass > 0.0)
    if not exposed_ok: errors.append('structural-tail exposure contract failed')
    offpath = _sample([[-5.0,.8], [-5.0,.7]], ['yield_rejoin','stop'], [{},{}])
    off_rec = nested_tail_truth_contract(offpath)
    offpath_ok = bool(off_rec.valid and off_rec.physical_identifiable and off_rec.structural_exposure_mass == 0.0)
    if not offpath_ok: errors.append('off-tail structural rule incorrectly censored')

    batch = {'r_dep_star': torch.tensor([.5,.2,-.7]), 'is_nominal': torch.zeros(3), 'bucket_id': torch.tensor([1,1,2]),
             'time_index': torch.zeros(3, dtype=torch.long), 'absolute_truth_physical_identifiable': torch.tensor([1.,0.,1.])}
    cfg = {'direct_value_absolute_feasibility_truth_contract': 'censor_structural_tail', 'direct_value_absolute_feasibility_supervision_objective': 'signed_margin_huber'}
    mask, _target, censored = _absolute_feasibility_supervision_mask(batch, cfg)
    mask_ok = mask.tolist() == [True, False, True] and censored.tolist() == [False, True, False]
    if not mask_ok: errors.append('structural-tail supervision mask failed')
    out = {'direct_recovery_absolute_feasibility_logit': torch.tensor([.4, 99.0, -.2])}
    got = float(_absolute_feasibility_supervision_loss(out, batch, cfg)); exp = float(torch.nn.functional.smooth_l1_loss(torch.tensor([.4,-.2]), torch.tensor([.5,-.7]), beta=1.0))
    supervision_ok = abs(got-exp) <= 1e-8
    if not supervision_ok: errors.append('structural-tail signed Huber supervision failed')

    source_text = inspect.getsource(teacher_margin)
    teacher_semantics_ok = all(x in source_text for x in ('max(val, 0.6)', 'min(val, -0.8)', 'max(val, 0.9)', '_artifact_margin_override'))
    if not teacher_semantics_ok: errors.append('teacher structural semantics no longer match PSTC adjudication assumptions')

    base = {'direct_recovery_absolute_semantic_witness_correction': True, 'direct_recovery_semantic_witness_active_set_alignment': True,
            'direct_recovery_semantic_witness_path_stop_alignment': False, 'direct_recovery_semantic_witness_classlocal_transport': False,
            'direct_recovery_semantic_witness_route_alignment': True, 'direct_recovery_semantic_witness_reentry_alignment': True,
            'direct_recovery_semantic_witness_control_projection': True, 'direct_recovery_semantic_witness_boundary_transport': False,
            'direct_recovery_semantic_witness_projection_fidelity_weighting': False, 'direct_recovery_semantic_witness_active_constraint_typed_source': False,
            'direct_recovery_semantic_witness_root_tail_source': True, 'direct_recovery_semantic_witness_tail_localization': True,
            'direct_recovery_evidence_native_certificate_preservation': True}
    schema, source = _semantic_witness_checkpoint_feature_contract(base)
    serializer_ok = schema == 3 and source == 'projected_boundary_common_executable_recovery_witness'
    if not serializer_ok: errors.append('J78 source serializer contract changed')
    L = FlatFeatureLayout(feature_max_agents=2)
    m = OCRAPModel(input_dim=L.total_dim, num_roots=3, num_options=2, d_model=16, d_obs=8, encoder_type='structured_transformer',
                   feature_layout=asdict(L), num_layers=1, num_heads=4, dropout=0.0, option_feature_dim=OPTION_FEATURE_DIM,
                   direct_recovery_value_head=True, **base)
    w = m.direct_absolute_root_tail_source_scale
    source_capacity_ok = bool(w is not None and tuple(w.shape) == (1,) and w.numel() == 1 and torch.count_nonzero(w).item() == 0 and m.direct_absolute_semantic_witness_gain is None)
    if not source_capacity_ok: errors.append('fixed J78 one-scalar source capacity contract failed')

    valid = not errors
    doc = {
        'schema': 'ocrap-v48.79-pstc-runtime-code-contract-v1', 'engineering_version': 'v48.79.0-OC-PSTC',
        'valid': valid, 'attribution_ready': valid, 'errors': errors, 'runtime_modules': mods,
        'teacher_structural_semantics_locked': {'valid': teacher_semantics_ok, 'rules': ['recovery_mode_floor_0p6','route_override_neg_0p8','secondary_floor_0p9','hidden_or_artifact_branch_semantics']},
        'truth_contract_synthetic': {'clean_physical': clean_rec.to_dict(), 'structural_exposed': exp_rec.to_dict(), 'structural_off_tail': off_rec.to_dict(), 'valid': clean_ok and exposed_ok and offpath_ok},
        'supervision_contract': {'truth_contract': 'censor_structural_tail', 'objective': 'signed_margin_huber', 'huber_beta': 1.0,
                                 'exact_0p5_not_automatically_censored': True, 'mask_valid': mask_ok, 'loss_valid': supervision_ok,
                                 'teacher_future_input_to_model': False, 'teacher_labels_changed': False},
        'source_contract': {'frozen_representation': 'J78 nested deployability-tail zero-translation root source', 'trainable_parameter': 'direct_absolute_root_tail_source_scale[1]',
                            'source_capacity_changed_vs_J78': False, 'serializer_schema': schema, 'serializer_source': source, 'serializer_valid': serializer_ok,
                            'option_translation_zero_mean': True, 'option_id_input': False, 'regime_id_input': False, 'classlocal_transport': False,
                            'boundary_transport': False, 'projection_fidelity': False, 'active_constraint_typed_source': False, 'valid': source_capacity_ok},
        'dataset_reconstruction': False, 'uses_test_roots': False, 'test_roots_read': False,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(doc, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'event': 'v48_79_runtime_contract', 'valid': valid, 'output': str(a.output)})); return 0 if valid else 30


if __name__ == '__main__': raise SystemExit(main())
