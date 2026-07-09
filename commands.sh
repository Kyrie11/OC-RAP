#!/usr/bin/env bash
set -euo pipefail

export OCRAP_ROOT=${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}
export TRAIN_MIX="$OCRAP_ROOT/train_safe,$OCRAP_ROOT/train_near_contact,$OCRAP_ROOT/train_contact"
export VAL_MIX="$OCRAP_ROOT/val_safe,$OCRAP_ROOT/val_near_contact,$OCRAP_ROOT/val_contact"

export SAFE_TEST="$OCRAP_ROOT/test_safe"
export NEAR_TEST="$OCRAP_ROOT/test_near_contact"
export CONTACT_TEST="$OCRAP_ROOT/test_contact"

export WOMD_VAL=${WOMD_VAL:-/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation/validation_tfexample.tfrecord}
export WOMD_VAL_INTERACTIVE=${WOMD_VAL_INTERACTIVE:-/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation_interactive/validation_interactive_tfexample.tfrecord}

export RUN=${RUN:-runs/ocrap_v14}
mkdir -p "$RUN"

# v14 selector: strict certified-abstention + optional evidence-gain gate.
# Key paper principle: OC-RAP may intervene only when a calibrated recovery
# action is admitted AND there is predicted evidence that switching from nominal
# improves deployable recovery or reduces the oracle-deployability gap.
make_common_sel() {
  local gamma_file="$1"      # gamma_rec_by_bucket json
  local require_admit="$2"   # true/false
  local evidence_gate="$3"   # true/false
  local anchor="$4"          # true/false
  local safe_mode="$5"       # certified/always/feasible

  COMMON_SEL=(
    --set selection.gamma_rec_by_bucket_file="$gamma_file"
    --set selection.ocrap_selector=calibrated_constrained
    --set selection.drs_success_gamma=0.0
    --set selection.drs_success_gamma_by_bucket.safe=0.0
    --set selection.drs_success_gamma_by_bucket.near_contact=0.0
    --set selection.drs_success_gamma_by_bucket.contact=0.0

    --set closed_loop.require_calibrated_selector=true
    --set closed_loop.require_gamma_by_bucket=true
    --set evaluation.require_calibrated_selector=true
    --set evaluation.require_gamma_by_bucket=true

    # Core v14 guarantee: no uncertified recovery fallback.
    --set selection.require_admitted_intervention_by_bucket.safe="$require_admit"
    --set selection.require_admitted_intervention_by_bucket.safe_v2="$require_admit"
    --set selection.require_admitted_intervention_by_bucket.test_safe="$require_admit"
    --set selection.require_admitted_intervention_by_bucket.test_safe_v2="$require_admit"
    --set selection.require_admitted_intervention_by_bucket.near_contact="$require_admit"
    --set selection.require_admitted_intervention_by_bucket.test_near_contact="$require_admit"
    --set selection.require_admitted_intervention_by_bucket.contact="$require_admit"
    --set selection.require_admitted_intervention_by_bucket.test_contact="$require_admit"
    --set selection.unadmitted_fallback_to_nominal=true

    # New v14 evidence gate.  Thresholds are intentionally mild: they prevent
    # low-DRS switches but should not suppress genuinely useful early recovery.
    --set selection.require_intervention_evidence_by_bucket.safe="$evidence_gate"
    --set selection.require_intervention_evidence_by_bucket.test_safe="$evidence_gate"
    --set selection.require_intervention_evidence_by_bucket.near_contact="$evidence_gate"
    --set selection.require_intervention_evidence_by_bucket.test_near_contact="$evidence_gate"
    --set selection.require_intervention_evidence_by_bucket.contact="$evidence_gate"
    --set selection.require_intervention_evidence_by_bucket.test_contact="$evidence_gate"
    --set selection.intervention_min_rec_lcb_gain_by_bucket.safe=0.10
    --set selection.intervention_min_rec_lcb_gain_by_bucket.test_safe=0.10
    --set selection.intervention_min_rec_lcb_gain_by_bucket.near_contact=0.025
    --set selection.intervention_min_rec_lcb_gain_by_bucket.test_near_contact=0.025
    --set selection.intervention_min_rec_lcb_gain_by_bucket.contact=0.035
    --set selection.intervention_min_rec_lcb_gain_by_bucket.test_contact=0.035
    --set selection.intervention_min_drs_gain_by_bucket.safe=0.05
    --set selection.intervention_min_drs_gain_by_bucket.test_safe=0.05
    --set selection.intervention_min_drs_gain_by_bucket.near_contact=0.00
    --set selection.intervention_min_drs_gain_by_bucket.test_near_contact=0.00
    --set selection.intervention_min_drs_gain_by_bucket.contact=0.015
    --set selection.intervention_min_drs_gain_by_bucket.test_contact=0.015
    --set selection.intervention_min_gap_reduction_by_bucket.safe=0.10
    --set selection.intervention_min_gap_reduction_by_bucket.test_safe=0.10
    --set selection.intervention_min_gap_reduction_by_bucket.near_contact=0.025
    --set selection.intervention_min_gap_reduction_by_bucket.test_near_contact=0.025
    --set selection.intervention_min_gap_reduction_by_bucket.contact=0.040
    --set selection.intervention_min_gap_reduction_by_bucket.test_contact=0.040

    # Absolute shared-action guards.  These are loose enough for recall, but
    # stricter than v13 to protect executed DRS.
    --set selection.intervention_min_pred_drs_by_bucket.safe=0.60
    --set selection.intervention_min_pred_drs_by_bucket.test_safe=0.60
    --set selection.intervention_min_pred_drs_by_bucket.near_contact=0.68
    --set selection.intervention_min_pred_drs_by_bucket.test_near_contact=0.68
    --set selection.intervention_min_pred_drs_by_bucket.contact=0.70
    --set selection.intervention_min_pred_drs_by_bucket.test_contact=0.70
    --set selection.intervention_max_pred_gap_by_bucket.safe=0.90
    --set selection.intervention_max_pred_gap_by_bucket.test_safe=0.90
    --set selection.intervention_max_pred_gap_by_bucket.near_contact=1.15
    --set selection.intervention_max_pred_gap_by_bucket.test_near_contact=1.15
    --set selection.intervention_max_pred_gap_by_bucket.contact=1.20
    --set selection.intervention_max_pred_gap_by_bucket.test_contact=1.20

    # Safe: certified nominal preservation, not unconditional hard lock.
    --set selection.safe_force_nominal_when_feasible_by_bucket.safe=true
    --set selection.safe_force_nominal_when_feasible_by_bucket.safe_v2=true
    --set selection.safe_force_nominal_when_feasible_by_bucket.test_safe=true
    --set selection.safe_force_nominal_when_feasible_by_bucket.test_safe_v2=true
    --set selection.safe_force_nominal_mode_by_bucket.safe="$safe_mode"
    --set selection.safe_force_nominal_mode_by_bucket.safe_v2="$safe_mode"
    --set selection.safe_force_nominal_mode_by_bucket.test_safe="$safe_mode"
    --set selection.safe_force_nominal_mode_by_bucket.test_safe_v2="$safe_mode"
    --set selection.safe_cert_min_pred_drs_by_bucket.safe=0.55
    --set selection.safe_cert_min_pred_drs_by_bucket.test_safe=0.55
    --set selection.safe_cert_max_pred_gap_by_bucket.safe=1.50
    --set selection.safe_cert_max_pred_gap_by_bucket.test_safe=1.50
    --set selection.safe_cert_rec_slack_by_bucket.safe=1.25
    --set selection.safe_cert_rec_slack_by_bucket.test_safe=1.25

    --set selection.calibrated_shortfall_penalty_by_bucket.safe=0.05
    --set selection.calibrated_shortfall_penalty_by_bucket.near_contact=1.60
    --set selection.calibrated_shortfall_penalty_by_bucket.contact=2.10
    --set selection.calibrated_gap_penalty_by_bucket.safe=0.00
    --set selection.calibrated_gap_penalty_by_bucket.near_contact=0.25
    --set selection.calibrated_gap_penalty_by_bucket.contact=0.45
    --set selection.deployability_bonus_by_bucket.near_contact=0.50
    --set selection.deployability_bonus_by_bucket.contact=0.75
    --set selection.contact_deployability_bonus_by_bucket.contact=1.20
    --set selection.contact_gap_penalty_by_bucket.contact=0.50

    --set selection.prefer_admitted_by_bucket.safe=true
    --set selection.prefer_admitted_by_bucket.near_contact=true
    --set selection.prefer_admitted_by_bucket.contact=true

    --set selection.intervention_budget_rate_by_bucket.safe=0.0
    --set selection.intervention_budget_rate_by_bucket.near_contact=0.16
    --set selection.intervention_budget_rate_by_bucket.contact=0.22
    --set selection.intervention_budget_penalty_by_bucket.safe=50.0
    --set selection.intervention_budget_penalty_by_bucket.near_contact=1.4
    --set selection.intervention_budget_penalty_by_bucket.contact=1.1
    --set selection.deviation_penalty_by_bucket.safe=3.0
    --set selection.deviation_penalty_by_bucket.near_contact=0.12
    --set selection.deviation_penalty_by_bucket.contact=0.08
    --set selection.intervention_penalty_by_bucket.safe=2.0
    --set selection.intervention_penalty_by_bucket.near_contact=0.035
    --set selection.intervention_penalty_by_bucket.contact=0.030

    --set selection.stress_nominal_anchor_by_bucket.near_contact="$anchor"
    --set selection.stress_nominal_anchor_by_bucket.contact="$anchor"
    --set selection.stress_anchor_drs_floor_by_bucket.near_contact=0.86
    --set selection.stress_anchor_drs_floor_by_bucket.contact=0.84
    --set selection.stress_anchor_max_gap_by_bucket.near_contact=1.20
    --set selection.stress_anchor_max_gap_by_bucket.contact=1.25
    --set selection.stress_anchor_rec_slack_by_bucket.near_contact=0.18
    --set selection.stress_anchor_rec_slack_by_bucket.contact=0.20
    --set selection.stress_anchor_min_drs_gain_by_bucket.near_contact=0.02
    --set selection.stress_anchor_min_drs_gain_by_bucket.contact=0.03
    --set selection.stress_anchor_min_rec_gain_by_bucket.near_contact=0.04
    --set selection.stress_anchor_min_rec_gain_by_bucket.contact=0.05
    --set selection.stress_anchor_min_gap_reduction_by_bucket.near_contact=0.03
    --set selection.stress_anchor_min_gap_reduction_by_bucket.contact=0.04
  )
}

summarize_eval() {
  python - <<'PY' "$RUN" | tee "$RUN/summary_all_v14.txt"
import json, pathlib, sys
root=pathlib.Path(sys.argv[1])
print('\n===== Selector-only diagnostics on v13 checkpoint =====')
for p in sorted(root.glob('eval_*_v14selector_on_v13ckpt_*.json')):
    d=json.load(open(p)); r=d.get('methods',{}).get('ocrap',{})
    print('\n', p.name)
    print('  ocrap:', {k:r.get(k) for k in ['FRA_exec','DRS','bounded_NUP','ODG','artifact_selection_rate','post_contact_deployability','intervention_rate','selected_admitted_rate','mean_num_admitted_interventions','selection_reason_counts']})
print('\n===== Eval v14 =====')
for p in sorted(root.glob('eval_*_v14_*.json')):
    d=json.load(open(p)); print('\n', p.name)
    for m,r in d.get('methods',{}).items():
        if m in ['nominal','backup_filter','contingency','oracle_filter','ocrap','ocrap_teacher']:
            print(f"  {m:14s} FRA={r.get('FRA_exec')} DRS={r.get('DRS')} NUP={r.get('bounded_NUP')} ODG={r.get('ODG')} artifact={r.get('artifact_selection_rate')} PCD={r.get('post_contact_deployability')} int={r.get('intervention_rate')} admRate={r.get('selected_admitted_rate')} admNonNom={r.get('mean_num_admitted_interventions')} reason={r.get('selection_reason_counts')}")
print('\n===== Closed-loop/audit v14 =====')
keys=['num_decisions','intervention_rate','closed_loop_bounded_NUP','closed_loop_FRA_exec','closed_loop_DRS','closed_loop_ODG','closed_loop_post_contact_deployability','closed_loop_artifact_selection_rate','closed_loop_audit_candidate_count','closed_loop_audit_best_DRS','closed_loop_audit_best_R_dep','closed_loop_audit_selected_R_dep_regret','closed_loop_audit_selector_miss_rate','closed_loop_audit_recoverable_candidate_rate','closed_loop_pred_r_dep','closed_loop_pred_gap','closed_loop_pred_DRS_proxy']
for p in sorted(root.glob('*v14*.json')):
    try: d=json.load(open(p))
    except Exception: continue
    if 'closed_loop_bounded_NUP' not in d: continue
    print('\n', p.name)
    for k in keys:
        if k in d: print(f'  {k}: {d[k]}')
    print('  selection_reason_counts:', d.get('selection_reason_counts'))
    print('  label_modes:', d.get('label_modes'))
PY
}

# ---------------------------------------------------------------------------
# 0) Selector-only diagnostics on the existing v13 checkpoint. This is the most
# important first run: it tells whether v13's remaining failures were mostly the
# soft fallback selector bug or require retraining.
# ---------------------------------------------------------------------------
if [[ -f runs/ocrap_v13/model_v13/best.pt && -f runs/ocrap_v13/gamma_rec_by_bucket_v13_floor0.json ]]; then
  for TAG in abstain_evidence abstain_noevidence anchor_evidence; do
    case "$TAG" in
      abstain_evidence) make_common_sel runs/ocrap_v13/gamma_rec_by_bucket_v13_floor0.json true true false certified ;;
      abstain_noevidence) make_common_sel runs/ocrap_v13/gamma_rec_by_bucket_v13_floor0.json true false false certified ;;
      anchor_evidence) make_common_sel runs/ocrap_v13/gamma_rec_by_bucket_v13_floor0.json true true true certified ;;
    esac
    for D in safe near_contact contact; do
      case "$D" in
        safe) DATASET="$SAFE_TEST" ;;
        near_contact) DATASET="$NEAR_TEST" ;;
        contact) DATASET="$CONTACT_TEST" ;;
      esac
      CUDA_VISIBLE_DEVICES=${GPU_EVAL:-0} PYTHONUNBUFFERED=1 python -u -m ocrap.cli evaluate \
        --dataset "$DATASET" \
        --checkpoint runs/ocrap_v13/model_v13/best.pt \
        --calibration runs/ocrap_v13/calibration_near_v13.json \
        --split test \
        --output "$RUN/eval_${D}_v14selector_on_v13ckpt_${TAG}.json" \
        --set evaluation.delta=0.05 \
        --set evaluation.group_by_dataset=true \
        --set evaluation.fallback_to_all_if_empty_split=true \
        "${COMMON_SEL[@]}" \
        --set 'evaluation.methods=[nominal,log_replay,mpc_proxy,backup_filter,contingency,oracle_filter,ocrap,ocrap_teacher]' \
        | tee "$RUN/eval_${D}_v14selector_on_v13ckpt_${TAG}.log"
    done
  done
fi

# ---------------------------------------------------------------------------
# 1) Train v14.  The main algorithmic change is selector-side certified
# abstention, but we also make training slightly more conservative around the
# admission boundary and reduce group-distillation overfitting seen in v13.
# ---------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=${GPU_TRAIN:-0} PYTHONUNBUFFERED=1 python -u -m ocrap.cli train \
  --dataset "$TRAIN_MIX" \
  --val-dataset "$VAL_MIX" \
  --output "$RUN/model_v14" \
  --set training.epochs=35 \
  --set training.batch_size=96 \
  --set training.lr=0.00008 \
  --set training.weight_decay=0.0005 \
  --set training.artifact_sampler_weight=2.5 \
  --set training.negative_deployable_sampler_weight=1.8 \
  --set training.safe_positive_sampler_weight=6.0 \
  --set training.regime_balance_power=0.85 \
  --set training.option_success_temperature=0.20 \
  --set training.early_stop_patience=9 \
  --set training.dataset_profile=true \
  --set training.num_workers=8 \
  --set training.progress=true \
  --set training.require_cuda=true \
  --set training.group_batching=true \
  --set training.group_batching_replacement=true \
  --set training.group_ranking_artifact_only=false \
  --set training.group_ranking_margin=0.28 \
  --set training.group_ce_temperature=0.30 \
  --set training.group_ce_pred_gap_weight=0.50 \
  --set training.group_ce_teacher_gap_weight=0.55 \
  --set training.group_distill_pred_gap_weight=0.40 \
  --set training.group_distill_teacher_gap_weight=0.50 \
  --set training.group_distill_teacher_temperature=0.25 \
  --set training.group_distill_pred_temperature=0.35 \
  --set training.nominal_switch_margin=0.18 \
  --set training.nominal_switch_teacher_gain_margin=0.09 \
  --set training.nominal_switch_gap_max=0.60 \
  --set training.safe_nominal_margin=0.26 \
  --set training.safe_nominal_min_success=0.88 \
  --set artifact.admission_gamma=0.05 \
  --set artifact.delta_neg=-0.03 \
  --set model.encoder_type=structured_transformer \
  --set model.d_model=192 \
  --set model.d_obs=64 \
  --set model.transformer_layers=2 \
  --set model.transformer_heads=4 \
  --set model.dropout=0.30 \
  --set loss_weights.margin=2.0 \
  --set loss_weights.obs=1.0 \
  --set loss_weights.anti_oracle=1.5 \
  --set loss_weights.artifact_gap=1.4 \
  --set loss_weights.admission=2.2 \
  --set loss_weights.option_q=0.7 \
  --set loss_weights.option_admission=1.6 \
  --set loss_weights.option_success=0.8 \
  --set loss_weights.option_success_bce=3.0 \
  --set loss_weights.option_best=3.0 \
  --set loss_weights.group_ranking=1.0 \
  --set loss_weights.group_ce=1.2 \
  --set loss_weights.group_distill=0.5 \
  --set loss_weights.nominal_switch=2.0 \
  --set loss_weights.safe_nominal=3.0 \
  --set loss_weights.utility=0.25 \
  | tee "$RUN/train_v14.log"

python - <<'PY' "$RUN/model_v14/train_summary.json" | tee "$RUN/loss_variance_v14.txt"
import json, sys
d=json.load(open(sys.argv[1]))
print('best_epoch:', d.get('best_epoch'))
print('best_val_loss:', d.get('best_val_loss'))
print('epochs_completed:', d.get('epochs_completed'))
for h in d.get('history', []):
    ep=h['epoch']
    if ep in [1,2,3,4,5,8,10,15,20,25,30,35] or ep==d.get('best_epoch'):
        tr=h['train']; va=h['val']
        print(
            f"ep{ep:02d} train={tr.get('loss'):.3f} val={va.get('loss'):.3f} | "
            f"adm={tr.get('loss_admission',0):.3f}/{va.get('loss_admission',0):.3f} "
            f"opt_bce={tr.get('loss_option_success_bce',0):.3f}/{va.get('loss_option_success_bce',0):.3f} "
            f"opt_best={tr.get('loss_option_best',0):.3f}/{va.get('loss_option_best',0):.3f} "
            f"group_rank={tr.get('loss_group_ranking',0):.3f}/{va.get('loss_group_ranking',0):.3f} "
            f"group_ce={tr.get('loss_group_ce',0):.3f}/{va.get('loss_group_ce',0):.3f} "
            f"group_distill={tr.get('loss_group_distill',0):.3f}/{va.get('loss_group_distill',0):.3f} "
            f"nom_switch={tr.get('loss_nominal_switch',0):.3f}/{va.get('loss_nominal_switch',0):.3f} "
            f"safe_nom={tr.get('loss_safe_nominal',0):.3f}/{va.get('loss_safe_nominal',0):.3f}"
        )
PY

# ---------------------------------------------------------------------------
# 2) Calibration and gamma maps.
# ---------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=${GPU_EVAL:-0} PYTHONUNBUFFERED=1 python -u -m ocrap.cli calibrate \
  --dataset "$OCRAP_ROOT/val_near_contact" \
  --checkpoint "$RUN/model_v14/best.pt" \
  --output "$RUN/calibration_near_v14.json" \
  --set calibration.deltas='[0.01,0.03,0.05,0.10]' \
  --set calibration.required_min_for_delta=20 \
  | tee "$RUN/calibration_near_v14.log"

CUDA_VISIBLE_DEVICES=${GPU_EVAL:-0} PYTHONUNBUFFERED=1 python -u -m ocrap.cli calibrate \
  --dataset "$OCRAP_ROOT/val_contact" \
  --checkpoint "$RUN/model_v14/best.pt" \
  --output "$RUN/calibration_contact_v14.json" \
  --set calibration.deltas='[0.01,0.03,0.05,0.10]' \
  --set calibration.required_min_for_delta=20 \
  | tee "$RUN/calibration_contact_v14.log"

python - <<'PY' "$RUN"
import json, math, sys
from pathlib import Path
run=Path(sys.argv[1])
def load_gamma(path, delta='0.05', default=0.0):
    d=json.load(open(path)); th=d.get('thresholds', {})
    for k,v in th.items():
        try:
            if abs(float(k)-float(delta)) < 1e-12 and math.isfinite(float(v)):
                return float(v)
        except Exception:
            pass
    return float(d.get('gamma_rec', default))
near=load_gamma(run/'calibration_near_v14.json')
contact=load_gamma(run/'calibration_contact_v14.json')
base={
 'safe':0.0,'safe_v2':0.0,'test_safe':0.0,'test_safe_v2':0.0,'train_safe':0.0,'val_safe':0.0,
 'near_contact':near,'test_near_contact':near,'train_near_contact':near,'val_near_contact':near,
 'contact':contact,'test_contact':contact,'train_contact':contact,'val_contact':contact,
}
floor0=dict(base)
for k in ['near_contact','test_near_contact','train_near_contact','val_near_contact','contact','test_contact','train_contact','val_contact']:
    floor0[k]=max(0.0, float(floor0[k]))
strict=dict(floor0)
for k in ['contact','test_contact','train_contact','val_contact']:
    strict[k]=max(0.08, float(strict[k]))
for name,m in [('delta05',base),('floor0',floor0),('contact_strict',strict)]:
    out=run/f'gamma_rec_by_bucket_v14_{name}.json'
    json.dump(m, open(out,'w'), indent=2, sort_keys=True)
    print('\n', out)
    print(json.dumps(m, indent=2, sort_keys=True))
PY

# ---------------------------------------------------------------------------
# 3) Offline eval grid.
# ---------------------------------------------------------------------------
for TAG in abstain_evidence anchor_evidence noevidence noabstain; do
  case "$TAG" in
    abstain_evidence) GAMMA="$RUN/gamma_rec_by_bucket_v14_floor0.json"; REQ=true; EVID=true; ANCHOR=false ;;
    anchor_evidence) GAMMA="$RUN/gamma_rec_by_bucket_v14_contact_strict.json"; REQ=true; EVID=true; ANCHOR=true ;;
    noevidence) GAMMA="$RUN/gamma_rec_by_bucket_v14_floor0.json"; REQ=true; EVID=false; ANCHOR=false ;;
    noabstain) GAMMA="$RUN/gamma_rec_by_bucket_v14_delta05.json"; REQ=false; EVID=false; ANCHOR=false ;;
  esac
  make_common_sel "$GAMMA" "$REQ" "$EVID" "$ANCHOR" certified
  for D in safe near_contact contact; do
    case "$D" in
      safe) DATASET="$SAFE_TEST" ;;
      near_contact) DATASET="$NEAR_TEST" ;;
      contact) DATASET="$CONTACT_TEST" ;;
    esac
    CUDA_VISIBLE_DEVICES=${GPU_EVAL:-0} PYTHONUNBUFFERED=1 python -u -m ocrap.cli evaluate \
      --dataset "$DATASET" \
      --checkpoint "$RUN/model_v14/best.pt" \
      --calibration "$RUN/calibration_near_v14.json" \
      --split test \
      --output "$RUN/eval_${D}_v14_${TAG}.json" \
      --set evaluation.delta=0.05 \
      --set evaluation.group_by_dataset=true \
      --set evaluation.fallback_to_all_if_empty_split=true \
      "${COMMON_SEL[@]}" \
      --set 'evaluation.methods=[nominal,log_replay,mpc_proxy,backup_filter,contingency,oracle_filter,ocrap,ocrap_teacher]' \
      | tee "$RUN/eval_${D}_v14_${TAG}.log"
  done
done

# ---------------------------------------------------------------------------
# 4) Closed-loop/audits.  Raw WOMD patterns must keep @150.
# ---------------------------------------------------------------------------
make_common_sel "$RUN/gamma_rec_by_bucket_v14_floor0.json" true true false certified
CUDA_VISIBLE_DEVICES=${GPU_EVAL:-0} PYTHONUNBUFFERED=1 python -u -m ocrap.cli closed-loop \
  --dataset "$WOMD_VAL@150" \
  --checkpoint "$RUN/model_v14/best.pt" \
  --output "$RUN/closed_loop_safe_fast_v14_abstain_evidence.json" \
  "${COMMON_SEL[@]}" \
  --set closed_loop.method=ocrap \
  --set closed_loop.bucket_dataset="$SAFE_TEST" \
  --set closed_loop.bucket_split=test \
  --set closed_loop.max_bucket_targets=80 \
  --set closed_loop.max_targets_per_scene=1 \
  --set closed_loop.max_rollouts=32 \
  --set closed_loop.raw_max_scenarios=600 \
  --set closed_loop.max_steps=40 \
  --set closed_loop.replan_interval_steps=1 \
  --set closed_loop.num_candidate_prefixes=16 \
  --set closed_loop.num_recovery_options=8 \
  --set closed_loop.label_mode=fast \
  --set closed_loop.progress_every_steps=5 \
  | tee "$RUN/closed_loop_safe_fast_v14_abstain_evidence.log"

for B in near_contact contact; do
  case "$B" in
    near_contact) DATASET_RAW="$WOMD_VAL_INTERACTIVE@150"; BUCKET="$NEAR_TEST"; GPU=${GPU_AUDIT_NEAR:-0} ;;
    contact) DATASET_RAW="$WOMD_VAL_INTERACTIVE@150"; BUCKET="$CONTACT_TEST"; GPU=${GPU_AUDIT_CONTACT:-0} ;;
  esac
  CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python -u -m ocrap.cli closed-loop \
    --dataset "$DATASET_RAW" \
    --checkpoint "$RUN/model_v14/best.pt" \
    --output "$RUN/audit_${B}_selected_topk_v14_abstain_evidence.json" \
    "${COMMON_SEL[@]}" \
    --set closed_loop.method=ocrap \
    --set closed_loop.bucket_dataset="$BUCKET" \
    --set closed_loop.bucket_split=test \
    --set closed_loop.max_bucket_targets=24 \
    --set closed_loop.max_rollouts=8 \
    --set closed_loop.raw_max_scenarios=800 \
    --set closed_loop.max_steps=20 \
    --set closed_loop.num_candidate_prefixes=12 \
    --set closed_loop.num_recovery_options=8 \
    --set closed_loop.label_mode=selected_topk \
    --set closed_loop.audit_every_n_steps=4 \
    --set closed_loop.audit_max_labels=240 \
    --set closed_loop.audit_top_k=8 \
    --set closed_loop.audit_max_extra_candidates=7 \
    --set closed_loop.progress_every_steps=1 \
    | tee "$RUN/audit_${B}_selected_topk_v14_abstain_evidence.log"
done

summarize_eval
