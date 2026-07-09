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

export RUN=${RUN:-runs/ocrap_v15_dualcert}
mkdir -p "$RUN"

# v15 selector principle:
#   certified intervention = scalar OC-MERO LCB certificate OR shared-option DRS certificate.
# This addresses v14's over-abstention in near/contact while preserving the v14 guarantee:
# no uncertified soft-constraint recovery fallback.
make_common_sel() {
  local gamma_file="$1"      # scalar gamma_rec_by_bucket json
  local drs_map_file="$2"    # option_drs_certificate_threshold_by_bucket json
  local require_admit="$3"   # true/false
  local evidence_gate="$4"   # true/false
  local dual_cert="$5"       # true/false
  local anchor="$6"          # true/false
  local safe_mode="$7"       # certified/always/feasible

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

    # v14/v15 invariant: no uncertified non-nominal fallback.
    --set selection.require_admitted_intervention_by_bucket.safe="$require_admit"
    --set selection.require_admitted_intervention_by_bucket.safe_v2="$require_admit"
    --set selection.require_admitted_intervention_by_bucket.test_safe="$require_admit"
    --set selection.require_admitted_intervention_by_bucket.test_safe_v2="$require_admit"
    --set selection.require_admitted_intervention_by_bucket.near_contact="$require_admit"
    --set selection.require_admitted_intervention_by_bucket.test_near_contact="$require_admit"
    --set selection.require_admitted_intervention_by_bucket.contact="$require_admit"
    --set selection.require_admitted_intervention_by_bucket.test_contact="$require_admit"
    --set selection.unadmitted_fallback_to_nominal=true

    # v15 dual certificate: disabled in safe, enabled only in stress regimes.
    --set selection.option_drs_certificate_by_bucket.safe=false
    --set selection.option_drs_certificate_by_bucket.safe_v2=false
    --set selection.option_drs_certificate_by_bucket.test_safe=false
    --set selection.option_drs_certificate_by_bucket.near_contact="$dual_cert"
    --set selection.option_drs_certificate_by_bucket.test_near_contact="$dual_cert"
    --set selection.option_drs_certificate_by_bucket.contact="$dual_cert"
    --set selection.option_drs_certificate_by_bucket.test_contact="$dual_cert"
    --set selection.option_drs_certificate_threshold_by_bucket_file="$drs_map_file"
    --set selection.option_drs_certificate_counts_as_evidence=true

    # Gap/LCB guards for the DRS certificate.  These prevent pure high-DRS but
    # oracle-gap-heavy actions from being admitted.
    --set selection.option_drs_certificate_max_gap_by_bucket.near_contact=1.20
    --set selection.option_drs_certificate_max_gap_by_bucket.test_near_contact=1.20
    --set selection.option_drs_certificate_max_gap_by_bucket.contact=1.35
    --set selection.option_drs_certificate_max_gap_by_bucket.test_contact=1.35
    --set selection.option_drs_certificate_rec_slack_by_bucket.near_contact=0.65
    --set selection.option_drs_certificate_rec_slack_by_bucket.test_near_contact=0.65
    --set selection.option_drs_certificate_rec_slack_by_bucket.contact=0.40
    --set selection.option_drs_certificate_rec_slack_by_bucket.test_contact=0.40
    --set selection.option_drs_certificate_min_rec_lcb_by_bucket.near_contact=-0.35
    --set selection.option_drs_certificate_min_rec_lcb_by_bucket.test_near_contact=-0.35
    --set selection.option_drs_certificate_min_rec_lcb_by_bucket.contact=-0.25
    --set selection.option_drs_certificate_min_rec_lcb_by_bucket.test_contact=-0.25

    # Evidence gate remains on, but the option-DRS certificate itself counts as
    # evidence because it is a direct observation-consistent shared-recovery test.
    --set selection.require_intervention_evidence_by_bucket.safe="$evidence_gate"
    --set selection.require_intervention_evidence_by_bucket.test_safe="$evidence_gate"
    --set selection.require_intervention_evidence_by_bucket.near_contact="$evidence_gate"
    --set selection.require_intervention_evidence_by_bucket.test_near_contact="$evidence_gate"
    --set selection.require_intervention_evidence_by_bucket.contact="$evidence_gate"
    --set selection.require_intervention_evidence_by_bucket.test_contact="$evidence_gate"
    --set selection.intervention_min_rec_lcb_gain_by_bucket.safe=0.10
    --set selection.intervention_min_rec_lcb_gain_by_bucket.test_safe=0.10
    --set selection.intervention_min_rec_lcb_gain_by_bucket.near_contact=0.015
    --set selection.intervention_min_rec_lcb_gain_by_bucket.test_near_contact=0.015
    --set selection.intervention_min_rec_lcb_gain_by_bucket.contact=0.020
    --set selection.intervention_min_rec_lcb_gain_by_bucket.test_contact=0.020
    --set selection.intervention_min_drs_gain_by_bucket.safe=0.05
    --set selection.intervention_min_drs_gain_by_bucket.test_safe=0.05
    --set selection.intervention_min_drs_gain_by_bucket.near_contact=0.00
    --set selection.intervention_min_drs_gain_by_bucket.test_near_contact=0.00
    --set selection.intervention_min_drs_gain_by_bucket.contact=0.005
    --set selection.intervention_min_drs_gain_by_bucket.test_contact=0.005
    --set selection.intervention_min_gap_reduction_by_bucket.safe=0.10
    --set selection.intervention_min_gap_reduction_by_bucket.test_safe=0.10
    --set selection.intervention_min_gap_reduction_by_bucket.near_contact=0.015
    --set selection.intervention_min_gap_reduction_by_bucket.test_near_contact=0.015
    --set selection.intervention_min_gap_reduction_by_bucket.contact=0.020
    --set selection.intervention_min_gap_reduction_by_bucket.test_contact=0.020

    # Absolute shared-action guards.
    --set selection.intervention_min_pred_drs_by_bucket.safe=0.60
    --set selection.intervention_min_pred_drs_by_bucket.test_safe=0.60
    --set selection.intervention_min_pred_drs_by_bucket.near_contact=0.70
    --set selection.intervention_min_pred_drs_by_bucket.test_near_contact=0.70
    --set selection.intervention_min_pred_drs_by_bucket.contact=0.74
    --set selection.intervention_min_pred_drs_by_bucket.test_contact=0.74
    --set selection.intervention_max_pred_gap_by_bucket.safe=0.90
    --set selection.intervention_max_pred_gap_by_bucket.test_safe=0.90
    --set selection.intervention_max_pred_gap_by_bucket.near_contact=1.20
    --set selection.intervention_max_pred_gap_by_bucket.test_near_contact=1.20
    --set selection.intervention_max_pred_gap_by_bucket.contact=1.35
    --set selection.intervention_max_pred_gap_by_bucket.test_contact=1.35

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

    # Ranking: prioritize deployability and low ODG more strongly than v14.
    --set selection.calibrated_shortfall_penalty_by_bucket.safe=0.05
    --set selection.calibrated_shortfall_penalty_by_bucket.near_contact=0.95
    --set selection.calibrated_shortfall_penalty_by_bucket.contact=1.10
    --set selection.calibrated_gap_penalty_by_bucket.safe=0.00
    --set selection.calibrated_gap_penalty_by_bucket.near_contact=0.30
    --set selection.calibrated_gap_penalty_by_bucket.contact=0.55
    --set selection.deployability_bonus_by_bucket.near_contact=0.95
    --set selection.deployability_bonus_by_bucket.contact=1.10
    --set selection.contact_deployability_bonus_by_bucket.contact=1.60
    --set selection.contact_gap_penalty_by_bucket.contact=0.55

    --set selection.prefer_admitted_by_bucket.safe=true
    --set selection.prefer_admitted_by_bucket.near_contact=true
    --set selection.prefer_admitted_by_bucket.contact=true

    --set selection.intervention_budget_rate_by_bucket.safe=0.0
    --set selection.intervention_budget_rate_by_bucket.near_contact=0.20
    --set selection.intervention_budget_rate_by_bucket.contact=0.28
    --set selection.intervention_budget_penalty_by_bucket.safe=50.0
    --set selection.intervention_budget_penalty_by_bucket.near_contact=1.0
    --set selection.intervention_budget_penalty_by_bucket.contact=0.85
    --set selection.deviation_penalty_by_bucket.safe=3.0
    --set selection.deviation_penalty_by_bucket.near_contact=0.08
    --set selection.deviation_penalty_by_bucket.contact=0.05
    --set selection.intervention_penalty_by_bucket.safe=2.0
    --set selection.intervention_penalty_by_bucket.near_contact=0.020
    --set selection.intervention_penalty_by_bucket.contact=0.018

    --set selection.stress_nominal_anchor_by_bucket.near_contact="$anchor"
    --set selection.stress_nominal_anchor_by_bucket.contact="$anchor"
    --set selection.stress_anchor_drs_floor_by_bucket.near_contact=0.88
    --set selection.stress_anchor_drs_floor_by_bucket.contact=0.86
    --set selection.stress_anchor_max_gap_by_bucket.near_contact=1.20
    --set selection.stress_anchor_max_gap_by_bucket.contact=1.35
    --set selection.stress_anchor_rec_slack_by_bucket.near_contact=0.25
    --set selection.stress_anchor_rec_slack_by_bucket.contact=0.25
    --set selection.stress_anchor_min_drs_gain_by_bucket.near_contact=0.015
    --set selection.stress_anchor_min_drs_gain_by_bucket.contact=0.020
    --set selection.stress_anchor_min_rec_gain_by_bucket.near_contact=0.025
    --set selection.stress_anchor_min_rec_gain_by_bucket.contact=0.030
    --set selection.stress_anchor_min_gap_reduction_by_bucket.near_contact=0.020
    --set selection.stress_anchor_min_gap_reduction_by_bucket.contact=0.025
  )
}

summarize_eval() {
  python - <<'PY' "$RUN" | tee "$RUN/summary_all_v15.txt"
import json, pathlib, sys
root=pathlib.Path(sys.argv[1])
print('\n===== Eval v15 =====')
for p in sorted(root.glob('eval_*_v15*.json')):
    d=json.load(open(p)); print('\n', p.name)
    for m,r in d.get('methods',{}).items():
        if m in ['nominal','backup_filter','contingency','oracle_filter','ocrap','ocrap_teacher']:
            print(f"  {m:14s} FRA={r.get('FRA_exec')} DRS={r.get('DRS')} NUP={r.get('bounded_NUP')} ODG={r.get('ODG')} artifact={r.get('artifact_selection_rate')} PCD={r.get('post_contact_deployability')} int={r.get('intervention_rate')} admRate={r.get('selected_admitted_rate')} admNonNom={r.get('mean_num_admitted_interventions')} reason={r.get('selection_reason_counts')}")
print('\n===== Closed-loop/audit v15 =====')
keys=['num_decisions','intervention_rate','closed_loop_bounded_NUP','closed_loop_FRA_exec','closed_loop_DRS','closed_loop_ODG','closed_loop_post_contact_deployability','closed_loop_artifact_selection_rate','closed_loop_audit_candidate_count','closed_loop_audit_best_DRS','closed_loop_audit_best_R_dep','closed_loop_audit_selected_R_dep_regret','closed_loop_audit_selector_miss_rate','closed_loop_audit_recoverable_candidate_rate','closed_loop_pred_r_dep','closed_loop_pred_gap','closed_loop_pred_DRS_proxy']
for p in sorted(root.glob('*v15*.json')):
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
# 0) Reuse v14 scalar calibration if available; otherwise build simple fallbacks.
# ---------------------------------------------------------------------------
python - <<'PY' "$RUN"
import json, pathlib, sys
run=pathlib.Path(sys.argv[1]); run.mkdir(parents=True, exist_ok=True)
# fallback maps: enough to run selector-only diagnostics before v15 calibration.
gamma={
 'safe':0.0,'safe_v2':0.0,'test_safe':0.0,'test_safe_v2':0.0,'train_safe':0.0,'val_safe':0.0,
 'near_contact':0.3710269331932068,'test_near_contact':0.3710269331932068,'train_near_contact':0.3710269331932068,'val_near_contact':0.3710269331932068,
 'contact':0.1553991585969925,'test_contact':0.1553991585969925,'train_contact':0.1553991585969925,'val_contact':0.1553991585969925,
}
if pathlib.Path('runs/ocrap_v14/gamma_rec_by_bucket_v14_floor0.json').exists():
    gamma=json.load(open('runs/ocrap_v14/gamma_rec_by_bucket_v14_floor0.json'))
json.dump(gamma, open(run/'gamma_rec_by_bucket_v15_from_v14_or_default.json','w'), indent=2, sort_keys=True)
# Three operating points for the auxiliary DRS certificate.
for name, near, contact in [('dual_loose',0.72,0.76),('dual_balanced',0.76,0.80),('dual_strict',0.82,0.86)]:
    m={
      'safe':1.01,'safe_v2':1.01,'test_safe':1.01,'test_safe_v2':1.01,
      'near_contact':near,'test_near_contact':near,'train_near_contact':near,'val_near_contact':near,
      'contact':contact,'test_contact':contact,'train_contact':contact,'val_contact':contact,
    }
    json.dump(m, open(run/f'option_drs_threshold_by_bucket_v15_{name}.json','w'), indent=2, sort_keys=True)
PY

# ---------------------------------------------------------------------------
# 1) Selector-only diagnostics on existing checkpoints.  This is cheap and tells
# whether v15's dual certificate fixes v14's over-abstention before retraining.
# ---------------------------------------------------------------------------
for CKPT_TAG in v14 v13; do
  case "$CKPT_TAG" in
    v14) CKPT="runs/ocrap_v14/model_v14/best.pt"; CAL="runs/ocrap_v14/calibration_near_v14.json" ;;
    v13) CKPT="runs/ocrap_v13/model_v13/best.pt"; CAL="runs/ocrap_v13/calibration_near_v13.json" ;;
  esac
  [[ -f "$CKPT" ]] || continue
  for TAG in dual_balanced dual_strict scalar_abstain; do
    case "$TAG" in
      dual_balanced) DRS_MAP="$RUN/option_drs_threshold_by_bucket_v15_dual_balanced.json"; DUAL=true; ANCHOR=false ;;
      dual_strict)   DRS_MAP="$RUN/option_drs_threshold_by_bucket_v15_dual_strict.json";   DUAL=true; ANCHOR=false ;;
      scalar_abstain) DRS_MAP="$RUN/option_drs_threshold_by_bucket_v15_dual_strict.json";  DUAL=false; ANCHOR=false ;;
    esac
    make_common_sel "$RUN/gamma_rec_by_bucket_v15_from_v14_or_default.json" "$DRS_MAP" true true "$DUAL" "$ANCHOR" certified
    for D in safe near_contact contact; do
      case "$D" in
        safe) DATASET="$SAFE_TEST" ;;
        near_contact) DATASET="$NEAR_TEST" ;;
        contact) DATASET="$CONTACT_TEST" ;;
      esac
      CUDA_VISIBLE_DEVICES=${GPU_EVAL:-0} PYTHONUNBUFFERED=1 python -u -m ocrap.cli evaluate \
        --dataset "$DATASET" \
        --checkpoint "$CKPT" \
        --calibration "$CAL" \
        --split test \
        --output "$RUN/eval_${D}_v15selector_on_${CKPT_TAG}ckpt_${TAG}.json" \
        --set evaluation.delta=0.05 \
        --set evaluation.group_by_dataset=true \
        --set evaluation.fallback_to_all_if_empty_split=true \
        "${COMMON_SEL[@]}" \
        --set 'evaluation.methods=[nominal,backup_filter,contingency,oracle_filter,ocrap,ocrap_teacher]' \
        | tee "$RUN/eval_${D}_v15selector_on_${CKPT_TAG}ckpt_${TAG}.log"
    done
  done
 done

# ---------------------------------------------------------------------------
# 2) Train v15.  Compared with v14, this de-emphasizes the scalar admission
# boundary that caused over-abstention and strengthens shared-option success.
# ---------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=${GPU_TRAIN:-0} PYTHONUNBUFFERED=1 python -u -m ocrap.cli train \
  --dataset "$TRAIN_MIX" \
  --val-dataset "$VAL_MIX" \
  --output "$RUN/model_v15" \
  --set training.epochs=28 \
  --set training.batch_size=96 \
  --set training.lr=0.00006 \
  --set training.weight_decay=0.0006 \
  --set training.artifact_sampler_weight=2.2 \
  --set training.negative_deployable_sampler_weight=1.5 \
  --set training.safe_positive_sampler_weight=6.0 \
  --set training.regime_balance_power=0.85 \
  --set training.option_success_temperature=0.18 \
  --set training.early_stop_patience=7 \
  --set training.dataset_profile=true \
  --set training.num_workers=8 \
  --set training.progress=true \
  --set training.require_cuda=true \
  --set training.group_batching=true \
  --set training.group_batching_replacement=true \
  --set training.group_ranking_artifact_only=false \
  --set training.group_ranking_margin=0.24 \
  --set training.group_ce_temperature=0.28 \
  --set training.group_ce_pred_gap_weight=0.35 \
  --set training.group_ce_teacher_gap_weight=0.45 \
  --set training.group_distill_pred_gap_weight=0.25 \
  --set training.group_distill_teacher_gap_weight=0.35 \
  --set training.group_distill_teacher_temperature=0.22 \
  --set training.group_distill_pred_temperature=0.32 \
  --set training.nominal_switch_margin=0.14 \
  --set training.nominal_switch_teacher_gain_margin=0.07 \
  --set training.nominal_switch_gap_max=0.70 \
  --set training.safe_nominal_margin=0.26 \
  --set training.safe_nominal_min_success=0.88 \
  --set artifact.admission_gamma=0.0 \
  --set artifact.delta_neg=0.0 \
  --set model.encoder_type=structured_transformer \
  --set model.d_model=192 \
  --set model.d_obs=64 \
  --set model.transformer_layers=2 \
  --set model.transformer_heads=4 \
  --set model.dropout=0.32 \
  --set loss_weights.margin=2.0 \
  --set loss_weights.obs=1.0 \
  --set loss_weights.anti_oracle=1.4 \
  --set loss_weights.artifact_gap=1.2 \
  --set loss_weights.admission=1.7 \
  --set loss_weights.option_q=0.8 \
  --set loss_weights.option_admission=1.7 \
  --set loss_weights.option_success=1.0 \
  --set loss_weights.option_success_bce=4.2 \
  --set loss_weights.option_best=4.0 \
  --set loss_weights.group_ranking=0.9 \
  --set loss_weights.group_ce=0.8 \
  --set loss_weights.group_distill=0.30 \
  --set loss_weights.nominal_switch=2.2 \
  --set loss_weights.safe_nominal=3.0 \
  --set loss_weights.utility=0.20 \
  | tee "$RUN/train_v15.log"

python - <<'PY' "$RUN/model_v15/train_summary.json" | tee "$RUN/loss_variance_v15.txt"
import json, sys
d=json.load(open(sys.argv[1]))
print('best_epoch:', d.get('best_epoch'))
print('best_val_loss:', d.get('best_val_loss'))
print('epochs_completed:', d.get('epochs_completed'))
for h in d.get('history', []):
    ep=h['epoch']
    if ep in [1,2,3,4,5,8,10,12,15,20,25,28] or ep==d.get('best_epoch'):
        tr=h['train']; va=h['val']
        print(
            f"ep{ep:02d} train={tr.get('loss'):.3f} val={va.get('loss'):.3f} | "
            f"adm={tr.get('loss_admission',0):.3f}/{va.get('loss_admission',0):.3f} "
            f"opt_bce={tr.get('loss_option_success_bce',0):.3f}/{va.get('loss_option_success_bce',0):.3f} "
            f"opt_best={tr.get('loss_option_best',0):.3f}/{va.get('loss_option_best',0):.3f} "
            f"group_ce={tr.get('loss_group_ce',0):.3f}/{va.get('loss_group_ce',0):.3f} "
            f"group_distill={tr.get('loss_group_distill',0):.3f}/{va.get('loss_group_distill',0):.3f} "
            f"nom_switch={tr.get('loss_nominal_switch',0):.3f}/{va.get('loss_nominal_switch',0):.3f} "
            f"safe_nom={tr.get('loss_safe_nominal',0):.3f}/{va.get('loss_safe_nominal',0):.3f}"
        )
PY

# ---------------------------------------------------------------------------
# 3) Scalar and shared-option calibration.
# ---------------------------------------------------------------------------
for B in near contact; do
  case "$B" in
    near) DATA="$OCRAP_ROOT/val_near_contact" ;;
    contact) DATA="$OCRAP_ROOT/val_contact" ;;
  esac
  CUDA_VISIBLE_DEVICES=${GPU_EVAL:-0} PYTHONUNBUFFERED=1 python -u -m ocrap.cli calibrate \
    --dataset "$DATA" \
    --checkpoint "$RUN/model_v15/best.pt" \
    --output "$RUN/calibration_${B}_rdep_v15.json" \
    --set calibration.score=r_dep \
    --set calibration.deltas='[0.03,0.05,0.10,0.15]' \
    --set calibration.required_min_for_delta=20 \
    | tee "$RUN/calibration_${B}_rdep_v15.log"

  CUDA_VISIBLE_DEVICES=${GPU_EVAL:-0} PYTHONUNBUFFERED=1 python -u -m ocrap.cli calibrate \
    --dataset "$DATA" \
    --checkpoint "$RUN/model_v15/best.pt" \
    --output "$RUN/calibration_${B}_drs_v15.json" \
    --set calibration.score=pred_drs \
    --set calibration.deltas='[0.05,0.10,0.15,0.20]' \
    --set calibration.required_min_for_delta=20 \
    | tee "$RUN/calibration_${B}_drs_v15.log"
 done

python - <<'PY' "$RUN"
import json, math, sys
from pathlib import Path
run=Path(sys.argv[1])
def load(path, delta, default):
    d=json.load(open(path)); th=d.get('thresholds', {})
    for k,v in th.items():
        try:
            if abs(float(k)-float(delta))<1e-12 and math.isfinite(float(v)):
                return float(v)
        except Exception: pass
    return float(d.get('gamma_rec', default))
near=load(run/'calibration_near_rdep_v15.json',0.05,0.0)
contact=load(run/'calibration_contact_rdep_v15.json',0.05,0.0)
gamma={
 'safe':0.0,'safe_v2':0.0,'test_safe':0.0,'test_safe_v2':0.0,'train_safe':0.0,'val_safe':0.0,
 'near_contact':max(0.0,near),'test_near_contact':max(0.0,near),'train_near_contact':max(0.0,near),'val_near_contact':max(0.0,near),
 'contact':max(0.0,contact),'test_contact':max(0.0,contact),'train_contact':max(0.0,contact),'val_contact':max(0.0,contact),
}
json.dump(gamma, open(run/'gamma_rec_by_bucket_v15_floor0.json','w'), indent=2, sort_keys=True)
# Strict = conformal-ish auxiliary threshold; balanced = validation operating point with recall cap.
near_drs_strict=load(run/'calibration_near_drs_v15.json',0.10,0.82)
contact_drs_strict=load(run/'calibration_contact_drs_v15.json',0.10,0.86)
near_drs_bal=min(max(load(run/'calibration_near_drs_v15.json',0.15,0.76),0.74),0.86)
contact_drs_bal=min(max(load(run/'calibration_contact_drs_v15.json',0.15,0.80),0.78),0.88)
for name, near_drs, contact_drs in [('calibrated_strict',near_drs_strict,contact_drs_strict),('balanced',near_drs_bal,contact_drs_bal),('loose',0.72,0.76)]:
    m={
      'safe':1.01,'safe_v2':1.01,'test_safe':1.01,'test_safe_v2':1.01,
      'near_contact':near_drs,'test_near_contact':near_drs,'train_near_contact':near_drs,'val_near_contact':near_drs,
      'contact':contact_drs,'test_contact':contact_drs,'train_contact':contact_drs,'val_contact':contact_drs,
    }
    json.dump(m, open(run/f'option_drs_threshold_by_bucket_v15_{name}.json','w'), indent=2, sort_keys=True)
    print(name, json.dumps(m, indent=2, sort_keys=True))
print('gamma', json.dumps(gamma, indent=2, sort_keys=True))
PY

# ---------------------------------------------------------------------------
# 4) Offline eval grid.
# ---------------------------------------------------------------------------
for TAG in dual_balanced dual_strict dual_loose scalar_abstain noabstain; do
  case "$TAG" in
    dual_balanced) DRS_MAP="$RUN/option_drs_threshold_by_bucket_v15_balanced.json"; REQ=true; EVID=true; DUAL=true; ANCHOR=false ;;
    dual_strict) DRS_MAP="$RUN/option_drs_threshold_by_bucket_v15_calibrated_strict.json"; REQ=true; EVID=true; DUAL=true; ANCHOR=false ;;
    dual_loose) DRS_MAP="$RUN/option_drs_threshold_by_bucket_v15_loose.json"; REQ=true; EVID=true; DUAL=true; ANCHOR=false ;;
    scalar_abstain) DRS_MAP="$RUN/option_drs_threshold_by_bucket_v15_calibrated_strict.json"; REQ=true; EVID=true; DUAL=false; ANCHOR=false ;;
    noabstain) DRS_MAP="$RUN/option_drs_threshold_by_bucket_v15_loose.json"; REQ=false; EVID=false; DUAL=true; ANCHOR=false ;;
  esac
  make_common_sel "$RUN/gamma_rec_by_bucket_v15_floor0.json" "$DRS_MAP" "$REQ" "$EVID" "$DUAL" "$ANCHOR" certified
  for D in safe near_contact contact; do
    case "$D" in
      safe) DATASET="$SAFE_TEST" ;;
      near_contact) DATASET="$NEAR_TEST" ;;
      contact) DATASET="$CONTACT_TEST" ;;
    esac
    CUDA_VISIBLE_DEVICES=${GPU_EVAL:-0} PYTHONUNBUFFERED=1 python -u -m ocrap.cli evaluate \
      --dataset "$DATASET" \
      --checkpoint "$RUN/model_v15/best.pt" \
      --calibration "$RUN/calibration_near_rdep_v15.json" \
      --split test \
      --output "$RUN/eval_${D}_v15_${TAG}.json" \
      --set evaluation.delta=0.05 \
      --set evaluation.group_by_dataset=true \
      --set evaluation.fallback_to_all_if_empty_split=true \
      "${COMMON_SEL[@]}" \
      --set 'evaluation.methods=[nominal,backup_filter,contingency,oracle_filter,ocrap,ocrap_teacher]' \
      | tee "$RUN/eval_${D}_v15_${TAG}.log"
  done
 done

# ---------------------------------------------------------------------------
# 5) Closed-loop/audits for the recommended balanced operating point.
# ---------------------------------------------------------------------------
make_common_sel "$RUN/gamma_rec_by_bucket_v15_floor0.json" "$RUN/option_drs_threshold_by_bucket_v15_balanced.json" true true true false certified
CUDA_VISIBLE_DEVICES=${GPU_EVAL:-0} PYTHONUNBUFFERED=1 python -u -m ocrap.cli closed-loop \
  --dataset "$WOMD_VAL@150" \
  --checkpoint "$RUN/model_v15/best.pt" \
  --output "$RUN/closed_loop_safe_fast_v15_dual_balanced.json" \
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
  | tee "$RUN/closed_loop_safe_fast_v15_dual_balanced.log"

for B in near_contact contact; do
  case "$B" in
    near_contact) DATASET_RAW="$WOMD_VAL_INTERACTIVE@150"; BUCKET="$NEAR_TEST"; GPU=${GPU_AUDIT_NEAR:-0} ;;
    contact) DATASET_RAW="$WOMD_VAL_INTERACTIVE@150"; BUCKET="$CONTACT_TEST"; GPU=${GPU_AUDIT_CONTACT:-0} ;;
  esac
  CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python -u -m ocrap.cli closed-loop \
    --dataset "$DATASET_RAW" \
    --checkpoint "$RUN/model_v15/best.pt" \
    --output "$RUN/audit_${B}_selected_topk_v15_dual_balanced.json" \
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
    | tee "$RUN/audit_${B}_selected_topk_v15_dual_balanced.log"
 done

summarize_eval
