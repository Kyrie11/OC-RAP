from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
import pytest
import torch
from ocrap.models.ocrap import ObservationConditionedActionFrontierBridge, OCRAPModel
from ocrap.models.encoders import FlatFeatureLayout

ROOT=Path(__file__).resolve().parents[1]

def test_ocaf_zero_action_exactly_zero():
    b=ObservationConditionedActionFrontierBridge(7,11,16,0.0).eval()
    out=b(torch.zeros(4,7),torch.randn(4,11))
    assert torch.equal(out,torch.zeros_like(out))

def test_ocaf_scene_modulates_action_but_not_scene_only():
    torch.manual_seed(2); b=ObservationConditionedActionFrontierBridge(7,11,16,0.0).eval()
    a=torch.randn(4,7); o=torch.randn(4,11)
    assert not torch.allclose(b(a,o),b(a,o+torch.linspace(0,1,o.shape[-1])))
    assert torch.equal(b(torch.zeros_like(a),o+10),torch.zeros(4,16))

def test_ocaf_preserves_action_magnitude_and_gradient():
    torch.manual_seed(3); b=ObservationConditionedActionFrontierBridge(7,11,16,0.0).eval()
    a=torch.randn(8,7,requires_grad=True); o=torch.randn(8,11)
    small=b(a*0.1,o).norm(); large=b(a,o).norm(); large.backward()
    assert large.item()>small.item()*1.5
    assert a.grad is not None and torch.isfinite(a.grad).all() and a.grad.abs().sum()>0


def _ocaf_model() -> OCRAPModel:
    return OCRAPModel(
        input_dim=FlatFeatureLayout().total_dim,
        num_roots=2,
        num_options=3,
        d_model=8,
        d_obs=4,
        encoder_type="structured_transformer",
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        direct_recovery_value_head=True,
        direct_recovery_value_pooling="candidate_concat_raw",
        direct_recovery_delta_head=True,
        direct_recovery_delta_regime_experts=True,
        direct_recovery_evidence_calibrator=True,
        direct_recovery_evidence_calibrator_context=True,
        direct_recovery_evidence_calibrator_context_source="physical_interaction",
        direct_recovery_evidence_interaction_hidden=16,
        direct_recovery_evidence_interaction_dropout=0.0,
        direct_recovery_evidence_unified_experts=True,
        direct_recovery_evidence_component_heads=True,
        direct_recovery_evidence_component_count=5,
        direct_recovery_evidence_concord=True,
        direct_recovery_evidence_admission_head=True,
        direct_recovery_evidence_admission_bounded=False,
        direct_recovery_evidence_admission_prior_mode="frontier_capped_slack",
    ).eval()


def test_ocaf_model_uses_nominal_scene_anchor_and_expected_slices():
    torch.manual_seed(7)
    model = _ocaf_model()
    layout = FlatFeatureLayout()
    assert layout.total_dim == 676
    assert model.direct_candidate_physical_feature_dim == 141
    assert model.direct_observation_feature_dim == 529
    assert model.direct_evidence_interaction_bridge is not None

    x = torch.zeros((3, layout.total_dim))
    for start, end in model.direct_candidate_physical_slices:
        x[1, start:end] = 0.5
        x[2, start:end] = 0.5
    # Candidate-only scene corruption must not replace the nominal anchor.
    for start, end in model.direct_observation_slices:
        x[2, start:end] = 9.0
    group = torch.tensor([[4, 9], [4, 9], [4, 9]])
    nominal = torch.tensor([1.0, 0.0, 0.0])
    action = model._direct_candidate_raw_relative_features(x, group, nominal)
    observation = model._direct_nominal_observation_features(x, group, nominal)
    context = model.direct_evidence_interaction_bridge(action, observation)
    assert torch.allclose(action[1], action[2])
    assert torch.allclose(observation[1], observation[2])
    assert torch.allclose(context[1], context[2])

    # Changing the nominal observation continuously changes the same action effect.
    x_changed = x.clone()
    for start, end in model.direct_observation_slices:
        x_changed[0, start:end] = torch.linspace(-1.0, 1.0, end - start)
    observation_changed = model._direct_nominal_observation_features(x_changed, group, nominal)
    context_changed = model.direct_evidence_interaction_bridge(action, observation_changed)
    assert not torch.allclose(context[1], context_changed[1])
    assert torch.equal(context_changed[0], torch.zeros_like(context_changed[0]))


def _exercise_group_row_broadcast(device: torch.device) -> None:
    """Match the real A30 batch geometry that triggered the v48.36 RC=30."""
    torch.manual_seed(36)
    model = _ocaf_model().to(device)
    layout = FlatFeatureLayout()
    group_size = 8
    num_groups = 12
    batch = group_size * num_groups

    x = torch.randn(batch, layout.total_dim, device=device, requires_grad=True)
    group_ids = torch.arange(num_groups, device=device).repeat_interleave(group_size)
    # Composite group keys exercise the same two-column path used by training.
    group_index = torch.stack([group_ids, group_ids * 17 + 3], dim=-1)
    nominal = torch.zeros(batch, device=device)
    nominal[::group_size] = 1.0

    action = model._direct_candidate_raw_relative_features(x, group_index, nominal)
    observation = model._direct_nominal_observation_features(x, group_index, nominal)
    raw_action = torch.cat(
        [x[:, start:end] for start, end in model.direct_candidate_physical_slices], dim=-1
    )
    raw_observation = torch.cat(
        [x[:, start:end] for start, end in model.direct_observation_slices], dim=-1
    )

    assert action.shape == (batch, 141)
    assert observation.shape == (batch, 529)
    for group_id in range(num_groups):
        start = group_id * group_size
        idx = torch.arange(start, start + group_size, device=device)
        expected_action = raw_action.index_select(0, idx) - raw_action[start : start + 1]
        expected_observation = raw_observation[start : start + 1].expand(group_size, -1)
        assert torch.allclose(action.index_select(0, idx), expected_action)
        assert torch.allclose(observation.index_select(0, idx), expected_observation)

    context = model.direct_evidence_interaction_bridge(action, observation)
    loss = context.square().mean() + action.square().mean() + observation.square().mean()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def test_group_row_broadcast_matches_real_batch_geometry_cpu():
    _exercise_group_row_broadcast(torch.device("cpu"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_group_row_broadcast_matches_real_batch_geometry_cuda():
    _exercise_group_row_broadcast(torch.device("cuda"))


def test_group_broadcast_runtime_contract_runs_on_cpu(tmp_path: Path):
    output = tmp_path / "contract.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "check_v48_36_cuda_group_broadcast_contract.py"),
            "--device",
            "cpu",
            "--batch-size",
            "96",
            "--group-size",
            "8",
            "--output",
            str(output),
        ],
        cwd=ROOT,
    )
    assert completed.returncode == 0
    report = json.loads(output.read_text())
    assert report["valid"] is True
    assert report["candidate_action_dim"] == 141
    assert report["nominal_observation_dim"] == 529
    assert report["forward_finite"] is True
    assert report["backward_finite"] is True


def test_main_runner_preflights_group_broadcast_on_both_training_gpus():
    text = (ROOT / "scripts" / "run_v48_36_ocaf_dedicated.sh").read_text()
    assert 'for gpu_spec in "gpu0:$GPU0" "gpu1:$GPU1"' in text
    assert "check_v48_36_cuda_group_broadcast_contract.py" in text
    assert "ocaf_cuda_group_broadcast_preflight_" in text

def test_noncompensatory_cap_never_exceeds_any_component():
    free=torch.tensor([5.0,-1.0,0.2]); cap=torch.tensor([-0.5,3.0,0.1])
    x=OCRAPModel._noncompensatory_smooth_cap(free,cap,0.1)
    assert torch.all(x<=free+1e-7) and torch.all(x<=cap+1e-7)

def test_scripts_use_one_shared_rule_and_no_regime_policy_input():
    text='\n'.join((ROOT/'scripts'/n).read_text() for n in [
        'adapt_ocrap_v48_36_ocaf_single_stage.sh','adapt_ocrap_v48_36_ocaf_variant.sh',
        'run_v48_36_ocaf_dedicated.sh','calibrate_v48_36_shared_certificate_pool.sh'])
    assert 'physical_interaction' in text
    assert 'regime_id_exposed_to_evidence_model": false' in text
    assert 'final_thresholds_fit_by_single_shared_rule": true' in text
    assert 'run_v48_36_safe_noninferiority.sh' in text
    assert 'run_v48_36_stress_if_authorized.sh' in text

def test_dataset_root_contract_rejects_legacy_alias(tmp_path: Path):
    protocol=tmp_path/'protocol'; safe=tmp_path/'safe'; safe.mkdir(); protocol.mkdir()
    names=['evidence_adapt_train_near_contact','evidence_adapt_train_contact','evidence_adapt_dev_near_contact','evidence_adapt_dev_contact','certificate_pool_near_contact','certificate_pool_contact']
    for n in names: (protocol/n).mkdir()
    out=tmp_path/'out.json'
    cmd=[sys.executable,str(ROOT/'tools/check_v48_36_dataset_root_contract.py'),'--protocol-root',str(protocol),'--safe-root',str(safe),
         '--train-near',str(protocol/names[0]),'--train-contact',str(protocol/'traincontact'),
         '--dev-near',str(protocol/names[2]),'--dev-contact',str(protocol/names[3]),
         '--cert-near',str(protocol/names[4]),'--cert-contact',str(protocol/names[5]),'--output',str(out)]
    assert subprocess.run(cmd,cwd=ROOT).returncode!=0
    assert json.loads(out.read_text())['valid'] is False

def _write_terminal(run: Path, rc: int):
    run.mkdir(); attempt='a1'
    complete={'event':'v48_36_ocaf_controller_complete','version':'v48.36-OCAF','attempt_id':attempt,'created_unix':2,
      'pipeline_exit_code':rc,'certificate_exit_code':rc,'pipeline_valid':rc in (0,20),'certificate_executed':rc in (0,20),
      'gate_evaluated':rc in (0,20),'gate_passed':rc==0,'next_commands_generated':rc==0,'test_roots_read':False}
    (run/'V48_36_COMPLETE.json').write_text(json.dumps(complete))
    if rc==0:
        (run/'NEXT_COMMANDS.txt').write_text('ok\n'); (run/'NEXT_COMMANDS_STATUS.json').write_text(json.dumps({'generated':True}))
    elif rc==20:
        marker={'attempt_id':attempt,'created_unix':1}; (run/'GATE_FAILED.json').write_text(json.dumps(marker))
        blocked={'attempt_id':attempt,'created_unix':1,'reason':'natural_gate_failed','exit_code':20,'gate_evaluated':True}
        (run/'NEXT_COMMANDS_BLOCKED.json').write_text(json.dumps(blocked)); (run/'NEXT_COMMANDS_STATUS.json').write_text(json.dumps(blocked))

def test_authoritative_resolver_accepts_rc0_and_rc20(tmp_path: Path):
    for rc in (0,20):
        run=tmp_path/f'r{rc}'; _write_terminal(run,rc); out=run/'status.json'
        cp=subprocess.run([sys.executable,str(ROOT/'tools/resolve_v48_36_authoritative_result.py'),'--run',str(run),'--output',str(out),'--expect-exit-code',str(rc)],cwd=ROOT)
        assert cp.returncode==0, out.read_text() if out.exists() else ''
        assert json.loads(out.read_text())['valid'] is True

def test_ablation_runner_is_independent_2x2():
    t=(ROOT/'scripts/run_v48_36_ocaf_ablations.sh').read_text()
    for x in ('A_action_only_soft_slack','B_ocaf_soft_slack','C_action_only_frontier','D_ocaf_frontier_main'): assert x in t
    assert 'engineering_failures_do_not_abort_independent_tasks' in t


def test_resume_contract_is_ocaf_specific_not_legacy_action_only():
    t=(ROOT/'tools/check_v48_36_resume_contract.py').read_text()
    assert 'physical_interaction_context' in t
    assert 'evidence_context_source") == "physical_interaction"' in t
    assert 'v48.36.1-RC30-TRAINING-CONTRACT-HOTFIX' not in t


def test_ablation_contracts_do_not_force_ocaf_or_frontier_into_controls():
    stage=(ROOT/'scripts/adapt_ocrap_v48_36_ocaf_single_stage.sh').read_text()
    checker=(ROOT/'tools/check_v48_36_ocaf_training_contract.py').read_text()
    variant=(ROOT/'scripts/adapt_ocrap_v48_36_ocaf_variant.sh').read_text()
    assert 'OCAF_ENABLED=false' in stage
    assert 'NONCOMPENSATORY_CAP=false' in stage
    assert '"observation_conditioned_action_frontier": $OCAF_ENABLED' in stage
    assert '"noncompensatory_frontier_cap": $NONCOMPENSATORY_CAP' in stage
    assert 'expected_ocaf = args.expect_context_source == "physical_interaction"' in checker
    assert 'expected_noncompensatory_cap = args.expect_prior_mode == "frontier_capped_slack"' in checker
    assert 'interaction_hidden=${EVIDENCE_INTERACTION_HIDDEN:-64}' in variant
    assert 'admission_prior_mode=${EVIDENCE_ADMISSION_PRIOR_MODE:-frontier_capped_slack}' in variant
