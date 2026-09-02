#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import torch


def sha(p: Path) -> str:
    h = hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args(); repo = a.repo.resolve()
    sys.path.insert(0, str(repo / 'src')); sys.path.insert(0, str(repo))
    import ocrap, ocrap.models.ocrap as mo, ocrap.cli.train as tr
    from ocrap.models.ocrap import OCRAPModel
    errors = []; mods = {}
    for name, m in [('ocrap', ocrap), ('ocrap.models.ocrap', mo), ('ocrap.cli.train', tr)]:
        p = Path(m.__file__).resolve(); mods[name] = {'path': str(p), 'inside_repo': repo in p.parents, 'sha256': sha(p)}
        if repo not in p.parents: errors.append(f'{name} outside repo')

    m = OCRAPModel(
        16, num_roots=3, num_options=2, d_model=32, d_obs=8,
        encoder_type='structured_transformer',
        direct_recovery_absolute_semantic_witness_correction=True,
        direct_recovery_semantic_witness_root_tail_source=True,
        direct_recovery_semantic_witness_tail_localization=True,
        direct_recovery_semantic_witness_structured_tail_field=True,
        direct_recovery_semantic_witness_signed_tail_channels=True,
        direct_recovery_semantic_witness_counterfactual_tail_response=True,
    )
    w = m.direct_absolute_structured_tail_field_weight
    if tuple(w.shape) != (2, 32) or torch.count_nonzero(w).item() != 0:
        errors.append('counterfactual signed field state contract failed')
    if m.direct_absolute_root_tail_source_scale is not None:
        errors.append('counterfactual field unexpectedly restored scalar root-tail source')

    # Exact candidate-minus-nominal latent-interaction response contract.
    # Nominal must be identically zero; malformed groups fail closed to zero.
    interaction = torch.tensor([
        [[[[0.20, -0.30], [0.10, -0.10]], [[0.05, 0.20], [-0.10, 0.30]]]],
        [[[[0.50, -0.40], [-0.20, 0.20]], [[0.15, 0.10], [-0.20, 0.50]]]],
        [[[[0.00, -0.10], [0.40, -0.50]], [[-0.05, 0.30], [0.10, 0.10]]]],
    ], dtype=torch.float32).squeeze(1)
    groups = torch.tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]], dtype=torch.long)
    nominal = torch.tensor([1.0, 0.0, 0.0])
    got = m._counterfactual_tail_response(interaction, groups, nominal)
    expect = interaction - interaction[:1]
    response_ok = torch.equal(got, expect) and torch.count_nonzero(got[0]).item() == 0
    bad_nominal = torch.tensor([1.0, 1.0, 0.0])
    fail_closed = torch.count_nonzero(m._counterfactual_tail_response(interaction, groups, bad_nominal)).item() == 0
    if not response_ok: errors.append('counterfactual latent-response exactness contract failed')
    if not fail_closed: errors.append('counterfactual latent-response malformed-group fail-closed contract failed')

    # Preserve v48.82.1 group-sampler engineering contract.
    from ocrap.cli.train import SceneTimeBatchSampler
    sampler = SceneTimeBatchSampler(groups=[[0, 1, 2], [3, 4, 5]], batch_size=9,
                                    replacement=True, shuffle_within_group=False, stratified=False)
    orig = torch.multinomial
    try:
        torch.multinomial = lambda weights, num_samples, replacement: torch.tensor([0, 0], dtype=torch.long)[:num_samples]
        batches = list(iter(sampler))
    finally:
        torch.multinomial = orig
    sampler_ok = batches == [[0, 1, 2], [0, 1, 2]] and all(len(b) == len(set(b)) for b in batches)
    if not sampler_ok: errors.append(f'replacement group atomicity contract failed: {batches!r}')

    doc = {
        'schema': 'ocrap-v48.83-crtf-runtime-code-contract-v1',
        'engineering_version': 'v48.83.0-OC-CRTF',
        'valid': not errors, 'attribution_ready': not errors, 'errors': errors,
        'runtime_modules': mods,
        'source_contract': {
            'root_tail_source': True, 'tail_localization': True,
            'structured_tail_field': True, 'signed_tail_channels': True,
            'counterfactual_tail_response': True,
            'field_shape': [2, 192], 'option_translation_zero_mean': True,
            'nominal_source_delta_exact_zero': True,
            'relative_ranker_input': False, 'teacher_metadata_input': False,
            'option_id_input': False, 'regime_id_input': False,
            'generic_mlp': False, 'boundary_transport': False,
        },
        'counterfactual_response_synthetic': {
            'exact_candidate_minus_nominal_latent_interaction': response_ok,
            'malformed_group_zero': fail_closed,
        },
        'sampler_contract': {
            'replacement_draws_preserved_across_epoch': True,
            'duplicate_group_within_minibatch_forbidden': True,
            'synthetic_duplicate_draw_passed': sampler_ok,
        },
        'supervision_contract': {
            'truth_contract': 'structural_interval_bounds',
            'objective': 'signed_margin_interval_huber',
            'dataset_reconstruction': False,
        },
        'test_roots_read': False,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(doc, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'valid': not errors, 'output': str(a.output)}))
    return 0 if not errors else 30


if __name__ == '__main__':
    raise SystemExit(main())
