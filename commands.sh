#!/usr/bin/env bash
set -euo pipefail

# Run from OC-RAP repository root after applying ocrap_v12_selector_training.patch,
# or directly inside the OC-RAP-v12-suggested code directory.
export OCRAP_ROOT=/data0/senzeyu2/dataset/OCRAP
export TRAIN_MIX="$OCRAP_ROOT/train_safe,$OCRAP_ROOT/train_near_contact,$OCRAP_ROOT/train_contact"
export VAL_MIX="$OCRAP_ROOT/val_safe,$OCRAP_ROOT/val_near_contact,$OCRAP_ROOT/val_contact"
export SAFE_TEST="$OCRAP_ROOT/test_safe"
export NEAR_TEST="$OCRAP_ROOT/test_near_contact"
export CONTACT_TEST="$OCRAP_ROOT/test_contact"
export WOMD_VAL=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation/validation_tfexample.tfrecord
export WOMD_VAL_INTERACTIVE=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation_interactive/validation_interactive_tfexample.tfrecord
export RUN=runs/ocrap_v12
mkdir -p "$RUN"

# 0) Sanity check the new losses/selectors are visible.
PYTHONPATH=src python - <<'PY'
from ocrap.models.losses import groupwise_candidate_ce_loss, nominal_switch_consistency_loss
from ocrap.planning.selector import calibrated_constrained_select
print('v12 imports ok')
PY

# 1) Train v12.
# Key changes vs v11:
# - group_ranking_artifact_only=false, so contact teacher-best non-artifact actions are ranked too.
# - group_ce directly classifies the teacher-best candidate within a scene-time set.
# - nominal_switch learns when nominal should be preserved instead of relying only on hard safe lock.
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python -u -m ocrap.cli train \
  --dataset "$TRAIN_MIX" \
  --val-dataset "$VAL_MIX" \
  --output "$RUN/model_v12" \
  --set training.epochs=45 \
  --set training.batch_size=72 \
  --set training.lr=0.00008 \
  --set training.weight_decay=0.00035 \
  --set training.artifact_sampler_weight=2.5 \
  --set training.negative_deployable_sampler_weight=1.6 \
  --set training.safe_positive_sampler_weight=2.5 \
  --set training.regime_balance_power=0.8 \
  --set training.group_batching=true \
  --set training.group_batching_replacement=true \
  --set training.group_ranking_margin=0.30 \
  --set training.group_ranking_gap_weight=0.35 \
  --set training.group_ranking_teacher_gap_weight=0.35 \
  --set training.group_ranking_artifact_only=false \
  --set training.group_ce_temperature=0.30 \
  --set training.group_ce_pred_gap_weight=0.35 \
  --set training.group_ce_teacher_gap_weight=0.35 \
  --set training.group_ce_utility_weight=0.03 \
  --set training.group_ce_require_deployable_target=true \
  --set training.nominal_switch_margin=0.12 \
  --set training.nominal_switch_teacher_gain_margin=0.06 \
  --set training.nominal_switch_gap_max=0.30 \
  --set training.option_success_temperature=0.25 \
  --set training.early_stop_patience=9 \
  --set training.dataset_profile=true \
  --set training.num_workers=8 \
  --set training.progress=true \
  --set training.require_cuda=true \
  --set training.save_every_epoch=false \
  --set model.encoder_type=structured_transformer \
  --set model.d_model=192 \
  --set model.d_obs=64 \
  --set model.transformer_layers=2 \
  --set model.transformer_heads=4 \
  --set model.dropout=0.30 \
  --set loss_weights.margin=2.0 \
  --set loss_weights.obs=1.0 \
  --set loss_weights.anti_oracle=1.2 \
  --set loss_weights.artifact_gap=1.2 \
  --set loss_weights.admission=1.0 \
  --set loss_weights.option_q=0.5 \
  --set loss_weights.option_admission=1.0 \
  --set loss_weights.option_success=0.5 \
  --set loss_weights.option_success_bce=1.3 \
  --set loss_weights.option_best=1.6 \
  --set loss_weights.group_ranking=1.5 \
  --set loss_weights.group_ce=1.0 \
  --set loss_weights.nominal_switch=0.8 \
  --set loss_weights.utility=0.30 \
  | tee "$RUN/train_v12.log"

python - <<'PY' "$RUN/model_v12/train_summary.json" | tee "$RUN/loss_variance_v12.txt"
import json, sys
d=json.load(open(sys.argv[1]))
print('best_epoch:', d.get('best_epoch'))
print('best_val_loss:', d.get('best_val_loss'))
print('epochs_completed:', d.get('epochs_completed'))
for h in d.get('history', []):
    ep=h['epoch']
    if ep in [1,2,3,4,5,8,10,15,20,25,30,35,40,45] or ep==d.get('best_epoch'):
        tr=h['train']; va=h['val']
        print(
            f"ep{ep:02d} train={tr.get('loss'):.3f} val={va.get('loss'):.3f} | "
            f"opt_bce={tr.get('loss_option_success_bce',0):.3f}/{va.get('loss_option_success_bce',0):.3f} "
            f"opt_best={tr.get('loss_option_best',0):.3f}/{va.get('loss_option_best',0):.3f} "
            f"group_rank={tr.get('loss_group_ranking',0):.3f}/{va.get('loss_group_ranking',0):.3f} "
            f"group_ce={tr.get('loss_group_ce',0):.3f}/{va.get('loss_group_ce',0):.3f} "
            f"nom_switch={tr.get('loss_nominal_switch',0):.3f}/{va.get('loss_nominal_switch',0):.3f} "
            f"dep={tr.get('loss_dep',0):.3f}/{va.get('loss_dep',0):.3f} gap={tr.get('loss_gap',0):.3f}/{va.get('loss_gap',0):.3f}"
        )
PY

# 2) Calibrate v12.
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python -u -m ocrap.cli calibrate \
  --dataset "$OCRAP_ROOT/val_near_contact" \
  --checkpoint "$RUN/model_v12/best.pt" \
  --output "$RUN/calibration_near_v12.json" \
  --set calibration.deltas='[0.01,0.05,0.10]' \
  --set calibration.required_min_for_delta=20 \
  | tee "$RUN/calibration_near_v12.log"

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python -u -m ocrap.cli calibrate \
  --dataset "$OCRAP_ROOT/val_contact" \
  --checkpoint "$RUN/model_v12/best.pt" \
  --output "$RUN/calibration_contact_v12.json" \
  --set calibration.deltas='[0.01,0.05,0.10]' \
  --set calibration.required_min_for_delta=20 \
  | tee "$RUN/calibration_contact_v12.log"

python - <<'PY'
import json, math
from pathlib import Path
run=Path('runs/ocrap_v12')
def load_gamma(path, delta='0.05', default=0.0):
    d=json.load(open(path)); th=d.get('thresholds', {})
    for k,v in th.items():
        try:
            if abs(float(k)-float(delta))<1e-12 and math.isfinite(float(v)):
                return float(v)
        except Exception:
            pass
    return float(d.get('gamma_rec', default))
near=load_gamma(run/'calibration_near_v12.json')
contact=load_gamma(run/'calibration_contact_v12.json')
def write(contact_gamma, name):
    m={
      'safe':0.0,'safe_v2':0.0,'test_safe':0.0,'test_safe_v2':0.0,'train_safe':0.0,'val_safe':0.0,
      'near_contact':near,'test_near_contact':near,'train_near_contact':near,'val_near_contact':near,
      'contact':contact_gamma,'test_contact':contact_gamma,'train_contact':contact_gamma,'val_contact':contact_gamma,
    }
    out=run/name
    json.dump(m, open(out,'w'), indent=2, sort_keys=True)
    print(out, json.dumps(m, sort_keys=True))
write(contact, 'gamma_rec_by_bucket_v12_delta05.json')
write(max(0.0, contact), 'gamma_rec_by_bucket_v12_delta05_contact_floor0.json')
PY

make_common_sel () {
  local MAP="$1"
  local SAFE_MODE="$2"      # always or certified
  local ANCHOR="$3"         # true or false
  COMMON_SEL=(
    --set selection.gamma_rec_by_bucket_file="$MAP"
    --set selection.ocrap_selector=calibrated_constrained
    --set selection.drs_success_gamma=0.0
    --set selection.drs_success_gamma_by_bucket.safe=0.0
    --set selection.drs_success_gamma_by_bucket.near_contact=0.0
    --set selection.drs_success_gamma_by_bucket.contact=0.0
    --set closed_loop.require_calibrated_selector=true
    --set closed_loop.require_gamma_by_bucket=true
    --set evaluation.require_calibrated_selector=true
    --set evaluation.require_gamma_by_bucket=true
    --set selection.safe_force_nominal_when_feasible_by_bucket.safe=true
    --set selection.safe_force_nominal_when_feasible_by_bucket.safe_v2=true
    --set selection.safe_force_nominal_when_feasible_by_bucket.test_safe=true
    --set selection.safe_force_nominal_when_feasible_by_bucket.test_safe_v2=true
    --set selection.safe_force_nominal_mode_by_bucket.safe="$SAFE_MODE"
    --set selection.safe_force_nominal_mode_by_bucket.safe_v2="$SAFE_MODE"
    --set selection.safe_force_nominal_mode_by_bucket.test_safe="$SAFE_MODE"
    --set selection.safe_force_nominal_mode_by_bucket.test_safe_v2="$SAFE_MODE"
    --set selection.safe_cert_min_pred_drs_by_bucket.safe=0.95
    --set selection.safe_cert_min_pred_drs_by_bucket.test_safe=0.95
    --set selection.safe_cert_max_pred_gap_by_bucket.safe=0.20
    --set selection.safe_cert_max_pred_gap_by_bucket.test_safe=0.20
    --set selection.safe_cert_rec_slack_by_bucket.safe=0.20
    --set selection.safe_cert_rec_slack_by_bucket.test_safe=0.20
    --set selection.calibrated_shortfall_penalty_by_bucket.safe=0.02
    --set selection.calibrated_shortfall_penalty_by_bucket.near_contact=1.20
    --set selection.calibrated_shortfall_penalty_by_bucket.contact=2.20
    --set selection.calibrated_gap_penalty_by_bucket.safe=0.00
    --set selection.calibrated_gap_penalty_by_bucket.near_contact=0.15
    --set selection.calibrated_gap_penalty_by_bucket.contact=0.35
    --set selection.deployability_bonus_by_bucket.near_contact=0.35
    --set selection.deployability_bonus_by_bucket.contact=0.75
    --set selection.contact_deployability_bonus_by_bucket.contact=1.30
    --set selection.contact_gap_penalty_by_bucket.contact=0.45
    --set selection.prefer_admitted_by_bucket.safe=false
    --set selection.prefer_admitted_by_bucket.near_contact=true
    --set selection.prefer_admitted_by_bucket.contact=true
    --set selection.intervention_budget_rate_by_bucket.safe=0.0
    --set selection.intervention_budget_rate_by_bucket.near_contact=0.18
    --set selection.intervention_budget_rate_by_bucket.contact=0.45
    --set selection.intervention_budget_penalty_by_bucket.safe=50.0
    --set selection.intervention_budget_penalty_by_bucket.near_contact=1.0
    --set selection.intervention_budget_penalty_by_bucket.contact=0.4
    --set selection.deviation_penalty_by_bucket.safe=3.0
    --set selection.deviation_penalty_by_bucket.near_contact=0.10
    --set selection.deviation_penalty_by_bucket.contact=0.03
    --set selection.intervention_penalty_by_bucket.safe=2.0
    --set selection.intervention_penalty_by_bucket.near_contact=0.02
    --set selection.intervention_penalty_by_bucket.contact=0.005
    --set selection.stress_nominal_anchor_by_bucket.near_contact="$ANCHOR"
    --set selection.stress_nominal_anchor_by_bucket.contact="$ANCHOR"
    --set selection.stress_anchor_drs_floor_by_bucket.near_contact=0.90
    --set selection.stress_anchor_drs_floor_by_bucket.contact=0.84
    --set selection.stress_anchor_max_gap_by_bucket.near_contact=0.25
    --set selection.stress_anchor_max_gap_by_bucket.contact=0.75
    --set selection.stress_anchor_rec_slack_by_bucket.near_contact=0.12
    --set selection.stress_anchor_rec_slack_by_bucket.contact=0.20
    --set selection.stress_anchor_min_drs_gain_by_bucket.near_contact=0.06
    --set selection.stress_anchor_min_drs_gain_by_bucket.contact=0.07
    --set selection.stress_anchor_min_rec_gain_by_bucket.near_contact=0.08
    --set selection.stress_anchor_min_rec_gain_by_bucket.contact=0.10
    --set selection.stress_anchor_min_gap_reduction_by_bucket.near_contact=0.08
    --set selection.stress_anchor_min_gap_reduction_by_bucket.contact=0.08
  )
}

# 3) Offline eval grid.
# hard: old safe hard lock, no stress anchor. certified: learned safe nominal certificate.
# cautious: learned safe certificate + stress nominal anchor; use this if DRS/NUP are harmed by aggressive switching.
for SPEC in \
  "$RUN/gamma_rec_by_bucket_v12_delta05.json hard always false" \
  "$RUN/gamma_rec_by_bucket_v12_delta05.json certified certified false" \
  "$RUN/gamma_rec_by_bucket_v12_delta05.json cautious certified true" \
  "$RUN/gamma_rec_by_bucket_v12_delta05_contact_floor0.json cautious_floor0 certified true"; do
  read -r MAP TAG SAFE_MODE ANCHOR <<< "$SPEC"
  make_common_sel "$MAP" "$SAFE_MODE" "$ANCHOR"
  for D in safe near_contact contact; do
    case "$D" in
      safe) DATASET="$SAFE_TEST" ;;
      near_contact) DATASET="$NEAR_TEST" ;;
      contact) DATASET="$CONTACT_TEST" ;;
    esac
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python -u -m ocrap.cli evaluate \
      --dataset "$DATASET" \
      --checkpoint "$RUN/model_v12/best.pt" \
      --calibration "$RUN/calibration_near_v12.json" \
      --split test \
      --output "$RUN/eval_${D}_v12_${TAG}.json" \
      --set evaluation.delta=0.05 \
      --set evaluation.group_by_dataset=true \
      --set evaluation.fallback_to_all_if_empty_split=true \
      "${COMMON_SEL[@]}" \
      --set 'evaluation.methods=[nominal,log_replay,mpc_proxy,backup_filter,contingency,oracle_filter,ocrap,ocrap_teacher]' \
      | tee "$RUN/eval_${D}_v12_${TAG}.log"
  done
done

# 4) Closed-loop fast: run both certified and cautious; keep the one that preserves safe NUP and improves near/contact.
for TAG in certified cautious; do
  if [[ "$TAG" == "certified" ]]; then ANCHOR=false; else ANCHOR=true; fi
  make_common_sel "$RUN/gamma_rec_by_bucket_v12_delta05.json" certified "$ANCHOR"
  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python -u -m ocrap.cli closed-loop \
    --dataset "$WOMD_VAL@150" \
    --checkpoint "$RUN/model_v12/best.pt" \
    --output "$RUN/closed_loop_safe_fast_v12_${TAG}.json" \
    "${COMMON_SEL[@]}" \
    --set closed_loop.method=ocrap \
    --set closed_loop.bucket_dataset="$SAFE_TEST" \
    --set closed_loop.bucket_split=test \
    --set closed_loop.max_bucket_targets=100 \
    --set closed_loop.max_targets_per_scene=1 \
    --set closed_loop.max_rollouts=40 \
    --set closed_loop.raw_max_scenarios=700 \
    --set closed_loop.max_steps=40 \
    --set closed_loop.replan_interval_steps=1 \
    --set closed_loop.num_candidate_prefixes=16 \
    --set closed_loop.num_recovery_options=8 \
    --set closed_loop.label_mode=fast \
    --set closed_loop.progress_every_steps=5 \
    | tee "$RUN/closed_loop_safe_fast_v12_${TAG}.log"

  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python -u -m ocrap.cli closed-loop \
    --dataset "$WOMD_VAL_INTERACTIVE@150" \
    --checkpoint "$RUN/model_v12/best.pt" \
    --output "$RUN/closed_loop_near_contact_fast_v12_${TAG}.json" \
    "${COMMON_SEL[@]}" \
    --set closed_loop.method=ocrap \
    --set closed_loop.bucket_dataset="$NEAR_TEST" \
    --set closed_loop.bucket_split=test \
    --set closed_loop.max_bucket_targets=100 \
    --set closed_loop.max_targets_per_scene=1 \
    --set closed_loop.max_rollouts=40 \
    --set closed_loop.raw_max_scenarios=700 \
    --set closed_loop.max_steps=40 \
    --set closed_loop.replan_interval_steps=1 \
    --set closed_loop.num_candidate_prefixes=16 \
    --set closed_loop.num_recovery_options=8 \
    --set closed_loop.label_mode=fast \
    --set closed_loop.progress_every_steps=5 \
    | tee "$RUN/closed_loop_near_contact_fast_v12_${TAG}.log" &

  CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 python -u -m ocrap.cli closed-loop \
    --dataset "$WOMD_VAL_INTERACTIVE@150" \
    --checkpoint "$RUN/model_v12/best.pt" \
    --output "$RUN/closed_loop_contact_fast_v12_${TAG}.json" \
    "${COMMON_SEL[@]}" \
    --set closed_loop.method=ocrap \
    --set closed_loop.bucket_dataset="$CONTACT_TEST" \
    --set closed_loop.bucket_split=test \
    --set closed_loop.max_bucket_targets=100 \
    --set closed_loop.max_targets_per_scene=1 \
    --set closed_loop.max_rollouts=40 \
    --set closed_loop.raw_max_scenarios=700 \
    --set closed_loop.max_steps=40 \
    --set closed_loop.replan_interval_steps=1 \
    --set closed_loop.num_candidate_prefixes=16 \
    --set closed_loop.num_recovery_options=8 \
    --set closed_loop.label_mode=fast \
    --set closed_loop.progress_every_steps=5 \
    | tee "$RUN/closed_loop_contact_fast_v12_${TAG}.log" &
  wait
done

# 5) Selected/top-k audit for candidate quality and selector miss diagnosis.
for TAG in certified cautious; do
  if [[ "$TAG" == "certified" ]]; then ANCHOR=false; else ANCHOR=true; fi
  make_common_sel "$RUN/gamma_rec_by_bucket_v12_delta05.json" certified "$ANCHOR"
  for B in near_contact contact; do
    case "$B" in
      near_contact) DATASET_RAW="$WOMD_VAL_INTERACTIVE@150"; BUCKET="$NEAR_TEST"; GPU=0 ;;
      contact) DATASET_RAW="$WOMD_VAL_INTERACTIVE@150"; BUCKET="$CONTACT_TEST"; GPU=1 ;;
    esac
    CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python -u -m ocrap.cli closed-loop \
      --dataset "$DATASET_RAW" \
      --checkpoint "$RUN/model_v12/best.pt" \
      --output "$RUN/audit_${B}_topk_v12_${TAG}.json" \
      "${COMMON_SEL[@]}" \
      --set closed_loop.method=ocrap \
      --set closed_loop.bucket_dataset="$BUCKET" \
      --set closed_loop.bucket_split=test \
      --set closed_loop.max_bucket_targets=24 \
      --set closed_loop.max_rollouts=8 \
      --set closed_loop.raw_max_scenarios=700 \
      --set closed_loop.max_steps=20 \
      --set closed_loop.num_candidate_prefixes=12 \
      --set closed_loop.num_recovery_options=8 \
      --set closed_loop.label_mode=selected \
      --set closed_loop.audit_every_n_steps=4 \
      --set closed_loop.audit_max_labels=96 \
      --set closed_loop.audit_top_k=6 \
      --set closed_loop.audit_max_extra_candidates=5 \
      --set closed_loop.progress_every_steps=1 \
      | tee "$RUN/audit_${B}_topk_v12_${TAG}.log"
  done
done

# 6) Compact summary.
python - <<'PY' "$RUN" | tee "$RUN/summary_all_v12.txt"
import json, pathlib, sys
root=pathlib.Path(sys.argv[1])
print('\n===== Eval =====')
for p in sorted(root.glob('eval_*_v12_*.json')):
    d=json.load(open(p))
    print('\n', p.name)
    print('  selector_config:', d.get('selector_config'))
    print('  gamma_rec_by_bucket_empty:', not bool(d.get('gamma_rec_by_bucket')))
    for m,r in d.get('methods',{}).items():
        print(
            f"  {m:16s} FRA={r.get('FRA_exec')} DRS={r.get('DRS')} NUP={r.get('bounded_NUP')} "
            f"ODG={r.get('ODG')} artifact={r.get('artifact_selection_rate')} "
            f"PCD={r.get('post_contact_deployability')} predDRS={r.get('mean_selected_pred_DRS_proxy')} "
            f"predR={r.get('mean_selected_pred_R_dep')} predGap={r.get('mean_selected_pred_gap')} reason={r.get('selection_reason_counts')}"
        )
print('\n===== Closed-loop / audit =====')
keys=['num_decisions','intervention_rate','closed_loop_bounded_NUP','closed_loop_FRA_exec','closed_loop_DRS','closed_loop_ODG','closed_loop_post_contact_deployability','closed_loop_artifact_selection_rate','closed_loop_audit_selected_R_dep_regret','closed_loop_audit_selector_miss_rate','closed_loop_audit_candidate_count','closed_loop_audit_recoverable_candidate_rate','closed_loop_pred_DRS_proxy']
for p in sorted(root.glob('*v12*.json')):
    try: d=json.load(open(p))
    except Exception: continue
    if 'closed_loop_DRS' not in d and 'closed_loop_bounded_NUP' not in d:
        continue
    print('\n', p.name)
    print('  selector_config:', d.get('selector_config'))
    for k in keys:
        if k in d: print(f'  {k}: {d[k]}')
    print('  selection_reason_counts:', d.get('selection_reason_counts'))
PY
