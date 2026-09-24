"""Proof obligations for throughput-only changes (no changed experiment math)."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ocrap.simulation import closed_loop_runner as runner


def _state(seed: int, *, overlapping: bool) -> SimpleNamespace:
    rng = np.random.default_rng(seed)
    n, steps, t = 15, 9, 4
    xy = rng.normal(size=(n, steps, 2)).astype(np.float32) * 12
    if overlapping:
        xy[1, t] = xy[0, t] + np.float32([0.1, 0.2])
    tr = SimpleNamespace(
        x=xy[:, :, 0], y=xy[:, :, 1],
        vel_x=rng.normal(size=(n, steps)).astype(np.float32),
        vel_y=rng.normal(size=(n, steps)).astype(np.float32),
        yaw=rng.normal(size=(n, steps)).astype(np.float32),
        length=rng.uniform(3., 5., size=(n, steps)).astype(np.float32),
        width=rng.uniform(1., 2., size=(n, steps)).astype(np.float32),
        height=np.full((n, steps), 1.6, dtype=np.float32),
        valid=rng.uniform(size=(n, steps)) > .13,
    )
    tr.valid[0, t] = True
    return SimpleNamespace(sim_trajectory=tr, timestep=np.int32(t))


def test_current_frame_reuse_preserves_exact_metric_and_overlap_values() -> None:
    for seed in range(8):
        state = _state(seed, overlapping=seed % 2 == 0)
        reference = runner._state_geometry_metrics(state, 0)
        reference_overlap = runner._overlapping_object_indices(state, 0)
        frame = runner._state_trajectory_frame(state)
        assert runner._state_geometry_metrics(state, 0, _frame=frame) == reference
        assert runner._overlapping_object_indices(state, 0, _frame=frame) == reference_overlap
        assert [float(frame[1][0]), float(frame[2][0])] == [
            float(state.sim_trajectory.x[0, 4]), float(state.sim_trajectory.y[0, 4])
        ]


def test_frame_reuse_avoids_duplicate_host_conversions(monkeypatch) -> None:
    state = _state(21, overlapping=True)
    real = runner._as_np
    calls = []

    def counted(x):
        calls.append(x)
        return real(x)

    monkeypatch.setattr(runner, '_as_np', counted)
    frame = runner._state_trajectory_frame(state)
    # One scalar timestep read and nine trajectory fields; no per-consumer copies.
    assert len(calls) == 10
    runner._state_geometry_metrics(state, 0, _frame=frame)
    runner._overlapping_object_indices(state, 0, _frame=frame)
    assert len(calls) == 10


def test_timing_report_reads_artifacts_without_changing_them(tmp_path: Path) -> None:
    import importlib.util
    script = Path(__file__).resolve().parents[1] / 'tools' / 'analyze_ablation_bottlenecks.py'
    spec = importlib.util.spec_from_file_location('ablation_timing_report', script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    p = tmp_path / 'no_route_alignment' / 'balanced'
    (p / 'near').mkdir(parents=True)
    artifact = p / 'near' / 'closed_loop_ocrap.json'
    artifact.write_text(json.dumps({
        'num_scenes': 2, 'num_decisions': 5, 'timing': {
            'scene_wall_sum_s': 8., 'totals_s': {
                'state_history': 1., 'candidate_features': 2.,
                'policy_selection': 3., 'waymax_step_metrics': 1.,
            }, 'execution_contract': 'throughput_or_unspecified',
        }
    }))
    phase = p / 'near.phase.json'
    phase.write_text(json.dumps({
        'status': 'complete', 'started_at': '2026-09-18T00:00:00+00:00',
        'ended_at': '2026-09-18T00:00:20+00:00',
    }))
    before = (artifact.read_bytes(), phase.read_bytes())
    rows = mod.collect(tmp_path, tmp_path / '_native_full_reference')
    assert len(rows) == 1
    assert rows[0]['outside_scene_s'] == 12.
    assert rows[0]['unmeasured_inside_scene_s'] == 1.
    assert rows[0]['dominant_measured_component'] == 'policy_selection'
    assert rows[0]['run_kind'] == 'accuracy'
    assert before == (artifact.read_bytes(), phase.read_bytes())


def test_scientific_comparison_checks_every_target_and_ignores_timing(tmp_path: Path) -> None:
    import importlib.util
    script = Path(__file__).resolve().parents[1] / 'tools' / 'compare_ablation_outputs.py'
    spec = importlib.util.spec_from_file_location('ablation_output_comparison', script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    old = tmp_path / 'old' / 'no_obs_consistency' / 'balanced' / 'contact'
    new = tmp_path / 'new' / 'no_obs_consistency' / 'balanced' / 'contact'
    old.mkdir(parents=True)
    new.mkdir(parents=True)
    scene = {'target_key': 'target-1', 'scene_id': 'scenario-1', 'target_time_index': 18,
             'num_decisions': 2, 'metric_summary': {'ttc': 0.},
             'decisions': [{'selected_candidate_index': 0}],
             'timing': {'wall_s': 42.}}
    summary = {'num_scenes': 1, 'num_decisions': 2,
               'closed_loop_FRA_exec': 1.5, 'scenes': [scene],
               'timing': {'scene_wall_sum_s': 42.}, 'run_fingerprint': 'different-code-sha'}
    p_old, p_new = old / 'closed_loop_ocrap.json', new / 'closed_loop_ocrap.json'
    p_old.write_text(json.dumps(summary))
    altered = json.loads(json.dumps(summary))
    altered['timing']['scene_wall_sum_s'] = 10.
    altered['run_fingerprint'] = 'new-code-sha'
    altered['scenes'][0]['timing']['wall_s'] = 10.
    p_new.write_text(json.dumps(altered))
    assert mod.compare_pair(p_old, p_new) == []
    assert len(mod._artifacts(tmp_path / 'new', False)) == 1
    altered['scenes'][0]['decisions'][0]['selected_candidate_index'] = 1
    p_new.write_text(json.dumps(altered))
    assert any('decisions' in d for d in mod.compare_pair(p_old, p_new))
    altered['scenes'][0]['decisions'][0]['selected_candidate_index'] = 0
    altered['scenes'][0]['metric_summary']['ttc'] = 0.01
    p_new.write_text(json.dumps(altered))
    assert any('metric_summary' in d for d in mod.compare_pair(p_old, p_new))


def test_latency_timing_report_does_not_invent_phase_duration(tmp_path: Path) -> None:
    import importlib.util
    script = Path(__file__).resolve().parents[1] / 'tools' / 'analyze_ablation_bottlenecks.py'
    spec = importlib.util.spec_from_file_location('ablation_timing_latency', script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    root = tmp_path / 'accuracy'
    root.mkdir()
    latency = tmp_path / 'latency' / '_native_full_reference' / 'precision' / 'safe'
    latency.mkdir(parents=True)
    artifact = latency / 'closed_loop_ocrap.json'
    artifact.write_text(json.dumps({'num_scenes': 1, 'num_decisions': 40,
        'timing': {'scene_wall_sum_s': 12., 'execution_contract': 'isolated_single_process_single_gpu',
                   'totals_s': {'policy_selection': 3., 'waymax_step_metrics': 2.}}}))
    rows = mod.collect(root, root / '_native_full_reference', tmp_path / 'latency')
    assert len(rows) == 1
    assert rows[0]['run_kind'] == 'isolated_latency'
    assert rows[0]['phase_wall_s'] is None
    assert rows[0]['outside_scene_s'] is None
    assert rows[0]['policy_selection_s'] == 3.
    assert rows[0]['timing_contract'] == 'isolated_single_process_single_gpu'


def test_concurrent_job_claims_are_unique_and_complete(tmp_path: Path) -> None:
    import subprocess
    script = (Path(__file__).resolve().parents[1] / 'scripts/run_submission_ablations.sh').read_text()
    func = 'claim_job() {' + script.split('claim_job() {', 1)[1].split('\nrun_claimed_job() {', 1)[0]
    root = tmp_path / 'queue'
    for dirname in ('pending', 'running', 'done', 'failed'):
        (root / dirname).mkdir(parents=True)
    expected = [f'0.{i:04d}' for i in range(20)]
    for name in expected:
        (root / 'pending' / f'{name}.job').write_text('job')
    command = ('set -euo pipefail\nQUEUE_ROOT="$1"\n' + func + '\n'
               'for gpu in 0 1; do for slot in 0 1; do (\n'
               '  while job="$(claim_job "$gpu" "$slot")"; do basename "$job"; done\n'
               ') & done; done\nwait\n')
    completed = subprocess.run(['bash', '-c', command, 'bash', str(root)],
                               capture_output=True, text=True, timeout=20, check=True)
    claimed = completed.stdout.splitlines()
    assert len(claimed) == len(expected)
    assert len(set(claimed)) == len(expected)
    assert sorted(name.split('.gpu')[0] for name in claimed) == sorted(expected)
    assert list((root / 'pending').glob('*.job')) == []


def test_ablation_bottleneck_report_reads_in_progress_scene_journal(tmp_path: Path) -> None:
    import importlib.util
    script = Path(__file__).resolve().parents[1] / 'tools' / 'analyze_ablation_bottlenecks.py'
    spec = importlib.util.spec_from_file_location('ablation_timing_partial', script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    root = tmp_path / 'accuracy'
    variant_root = root / 'no_obs_consistency' / 'balanced'
    run_dir = variant_root / 'near'
    run_dir.mkdir(parents=True)
    (variant_root / 'near.phase.json').write_text(json.dumps({
        'regime': 'near', 'status': 'running', 'exit_code': 0,
        'started_at': '2026-09-20T10:00:00+00:00', 'ended_at': ''
    }))
    artifact = run_dir / 'closed_loop_ocrap.json'
    journal = Path(str(artifact) + '.scenes.jsonl')
    scenes = [
        {'target_key': f'near:s{i}:t10', 'num_decisions': 4,
         'timing': {'wall_s': 10.0, 'execution_contract': 'concurrent_accuracy',
                    'totals_s': {'state_history': 1.0, 'candidate_features': 2.0,
                                 'policy_selection': 4.0, 'waymax_step_metrics': 1.0}}}
        for i in range(2)
    ]
    journal.write_text(''.join(json.dumps({'scene': s}) + '\n' for s in scenes))
    Path(str(artifact) + '.progress.json').write_text(json.dumps({
        'completed_rollouts': 2, 'total_rollouts': 10
    }))

    rows = mod.collect(root, root / '_native_full_reference')
    assert len(rows) == 1
    row = rows[0]
    assert row['status'] == 'running_partial_journal'
    assert row['scenes'] == 2
    assert row['decisions'] == 8
    assert row['policy_selection_s'] == 8.0
    assert row['completion_fraction'] == 0.2
    assert row['deployed_planner_s_per_decision'] == 14.0 / 8.0
    assert row['artifact'].endswith('.scenes.jsonl')
