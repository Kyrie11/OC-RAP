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

export RUN=${RUN:-runs/ocrap_v16_relative_cert}
export V15_RUN=${V15_RUN:-runs/ocrap_v15_dualcert}
export CKPT=${CKPT:-$V15_RUN/model_v15/best.pt}
export CAL=${CAL:-$V15_RUN/calibration_near_rdep_v15.json}
export GAMMA=${GAMMA:-$V15_RUN/gamma_rec_by_bucket_v15_floor0.json}
export DRS_MAP=${DRS_MAP:-$V15_RUN/option_drs_threshold_by_bucket_v15_balanced.json}
mkdir -p "$RUN"

# v16 principle:
#   intervention = absolute certificate OR relative recovery-opportunity certificate.
# The relative certificate is enabled only when nominal is low-headroom, then
# admits a non-nominal action only if it improves predicted deployable recovery
# over nominal while satisfying DRS and gap guards.  This targets v15's contact
# failure mode: long runs of nominal_no_certified_intervention_preserved despite
# recoverable candidate availability.
make_common_sel() {
  local tag="$1"       # scalar | rel | rel_dual | rel_aggr
  local anchor="$2"    # true/false
  local safe_mode="$3" # certified/always/feasible

  local dual=false
  local rel=false
  case "$tag" in
    scalar) dual=false; rel=false ;;
    rel) dual=false; rel=true ;;
    rel_dual) dual=true; rel=true ;;
    rel_aggr) dual=true; rel=true ;;
    *) echo "unknown selector tag: $tag" >&2; exit 2 ;;
  esac

  COMMON_SEL=(
    --set selection.gamma_rec_by_bucket_file="$GAMMA"
    --set selection.ocrap_selector=calibrated_constrained
    --set closed_loop.require_calibrated_selector=true
    --set closed_loop.require_gamma_by_bucket=true
    --set evaluation.require_calibrated_selector=true
    --set evaluation.require_gamma_by_bucket=true

    # Preserve the v14/v15 invariant: no unadmitted recovery fallback.
    --set selection.require_admitted_intervention_by_bucket.safe=true
    --set selection.require_admitted_intervention_by_bucket.safe_v2=true
    --set selection.require_admitted_intervention_by_bucket.test_safe=true
    --set selection.require_admitted_intervention_by_bucket.test_safe_v2=true
    --set selection.require_admitted_intervention_by_bucket.near_contact=true
    --set selection.require_admitted_intervention_by_bucket.test_near_contact=true
    --set selection.require_admitted_intervention_by_bucket.contact=true
    --set selection.require_admitted_intervention_by_bucket.test_contact=true
    --set selection.unadmitted_fallback_to_nominal=true

    # Safe remains nominal-preserving and certification-based.
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

    # Absolute option-DRS certificate from v15: keep as ablation, but not enough alone.
    --set selection.option_drs_certificate_by_bucket.safe=false
    --set selection.option_drs_certificate_by_bucket.safe_v2=false
    --set selection.option_drs_certificate_by_bucket.test_safe=false
    --set selection.option_drs_certificate_by_bucket.near_contact="$dual"
    --set selection.option_drs_certificate_by_bucket.test_near_contact="$dual"
    --set selection.option_drs_certificate_by_bucket.contact="$dual"
    --set selection.option_drs_certificate_by_bucket.test_contact="$dual"
    --set selection.option_drs_certificate_threshold_by_bucket_file="$DRS_MAP"
    --set selection.option_drs_certificate_counts_as_evidence=true
    --set selection.option_drs_certificate_max_gap_by_bucket.near_contact=1.25
    --set selection.option_drs_certificate_max_gap_by_bucket.test_near_contact=1.25
    --set selection.option_drs_certificate_max_gap_by_bucket.contact=1.60
    --set selection.option_drs_certificate_max_gap_by_bucket.test_contact=1.60
    --set selection.option_drs_certificate_rec_slack_by_bucket.near_contact=0.75
    --set selection.option_drs_certificate_rec_slack_by_bucket.test_near_contact=0.75
    --set selection.option_drs_certificate_rec_slack_by_bucket.contact=1.15
    --set selection.option_drs_certificate_rec_slack_by_bucket.test_contact=1.15
    --set selection.option_drs_certificate_min_rec_lcb_by_bucket.near_contact=-0.45
    --set selection.option_drs_certificate_min_rec_lcb_by_bucket.test_near_contact=-0.45
    --set selection.option_drs_certificate_min_rec_lcb_by_bucket.contact=-1.05
    --set selection.option_drs_certificate_min_rec_lcb_by_bucket.test_contact=-1.05

    # v16 relative recovery-opportunity certificate.
    --set selection.relative_recovery_certificate_by_bucket.safe=false
    --set selection.relative_recovery_certificate_by_bucket.safe_v2=false
    --set selection.relative_recovery_certificate_by_bucket.test_safe=false
    --set selection.relative_recovery_certificate_by_bucket.near_contact="$rel"
    --set selection.relative_recovery_certificate_by_bucket.test_near_contact="$rel"
    --set selection.relative_recovery_certificate_by_bucket.contact="$rel"
    --set selection.relative_recovery_certificate_by_bucket.test_contact="$rel"
    --set selection.relative_recovery_counts_as_evidence=true
    --set selection.relative_recovery_nominal_rec_lcb_max_by_bucket.near_contact=-0.22
    --set selection.relative_recovery_nominal_rec_lcb_max_by_bucket.test_near_contact=-0.22
    --set selection.relative_recovery_nominal_rec_lcb_max_by_bucket.contact=-0.55
    --set selection.relative_recovery_nominal_rec_lcb_max_by_bucket.test_contact=-0.55
    --set selection.relative_recovery_nominal_gap_min_by_bucket.near_contact=1.05
    --set selection.relative_recovery_nominal_gap_min_by_bucket.test_near_contact=1.05
    --set selection.relative_recovery_nominal_gap_min_by_bucket.contact=1.10
    --set selection.relative_recovery_nominal_gap_min_by_bucket.test_contact=1.10
    --set selection.relative_recovery_nominal_drs_max_by_bucket.near_contact=0.80
    --set selection.relative_recovery_nominal_drs_max_by_bucket.test_near_contact=0.80
    --set selection.relative_recovery_nominal_drs_max_by_bucket.contact=0.82
    --set selection.relative_recovery_nominal_drs_max_by_bucket.test_contact=0.82
    --set selection.relative_recovery_min_rec_gain_by_bucket.near_contact=0.18
    --set selection.relative_recovery_min_rec_gain_by_bucket.test_near_contact=0.18
    --set selection.relative_recovery_min_rec_gain_by_bucket.contact=0.18
    --set selection.relative_recovery_min_rec_gain_by_bucket.test_contact=0.18
    --set selection.relative_recovery_min_drs_by_bucket.near_contact=0.78
    --set selection.relative_recovery_min_drs_by_bucket.test_near_contact=0.78
    --set selection.relative_recovery_min_drs_by_bucket.contact=0.70
    --set selection.relative_recovery_min_drs_by_bucket.test_contact=0.70
    --set selection.relative_recovery_min_drs_gain_by_bucket.near_contact=0.00
    --set selection.relative_recovery_min_drs_gain_by_bucket.test_near_contact=0.00
    --set selection.relative_recovery_min_drs_gain_by_bucket.contact=-1.0
    --set selection.relative_recovery_min_drs_gain_by_bucket.test_contact=-1.0
    --set selection.relative_recovery_max_gap_by_bucket.near_contact=1.35
    --set selection.relative_recovery_max_gap_by_bucket.test_near_contact=1.35
    --set selection.relative_recovery_max_gap_by_bucket.contact=1.75
    --set selection.relative_recovery_max_gap_by_bucket.test_contact=1.75
    --set selection.relative_recovery_max_gap_increase_by_bucket.near_contact=0.10
    --set selection.relative_recovery_max_gap_increase_by_bucket.test_near_contact=0.10
    --set selection.relative_recovery_max_gap_increase_by_bucket.contact=0.35
    --set selection.relative_recovery_max_gap_increase_by_bucket.test_contact=0.35
    --set selection.relative_recovery_bonus_by_bucket.near_contact=0.70
    --set selection.relative_recovery_bonus_by_bucket.test_near_contact=0.70
    --set selection.relative_recovery_bonus_by_bucket.contact=1.30
    --set selection.relative_recovery_bonus_by_bucket.test_contact=1.30

    # Mild absolute guards: contact can use relative recovery even when scalar r_dep is conservative.
    --set selection.require_intervention_evidence_by_bucket.safe=true
    --set selection.require_intervention_evidence_by_bucket.test_safe=true
    --set selection.require_intervention_evidence_by_bucket.near_contact=true
    --set selection.require_intervention_evidence_by_bucket.test_near_contact=true
    --set selection.require_intervention_evidence_by_bucket.contact=true
    --set selection.require_intervention_evidence_by_bucket.test_contact=true
    --set selection.intervention_min_pred_drs_by_bucket.safe=0.60
    --set selection.intervention_min_pred_drs_by_bucket.test_safe=0.60
    --set selection.intervention_min_pred_drs_by_bucket.near_contact=0.70
    --set selection.intervention_min_pred_drs_by_bucket.test_near_contact=0.70
    --set selection.intervention_min_pred_drs_by_bucket.contact=0.68
    --set selection.intervention_min_pred_drs_by_bucket.test_contact=0.68
    --set selection.intervention_max_pred_gap_by_bucket.safe=0.90
    --set selection.intervention_max_pred_gap_by_bucket.test_safe=0.90
    --set selection.intervention_max_pred_gap_by_bucket.near_contact=1.35
    --set selection.intervention_max_pred_gap_by_bucket.test_near_contact=1.35
    --set selection.intervention_max_pred_gap_by_bucket.contact=1.75
    --set selection.intervention_max_pred_gap_by_bucket.test_contact=1.75
    --set selection.intervention_min_rec_lcb_gain_by_bucket.near_contact=0.015
    --set selection.intervention_min_rec_lcb_gain_by_bucket.test_near_contact=0.015
    --set selection.intervention_min_rec_lcb_gain_by_bucket.contact=0.015
    --set selection.intervention_min_rec_lcb_gain_by_bucket.test_contact=0.015
    --set selection.intervention_min_drs_gain_by_bucket.near_contact=0.00
    --set selection.intervention_min_drs_gain_by_bucket.test_near_contact=0.00
    --set selection.intervention_min_drs_gain_by_bucket.contact=0.00
    --set selection.intervention_min_drs_gain_by_bucket.test_contact=0.00
    --set selection.intervention_min_gap_reduction_by_bucket.near_contact=0.015
    --set selection.intervention_min_gap_reduction_by_bucket.test_near_contact=0.015
    --set selection.intervention_min_gap_reduction_by_bucket.contact=0.015
    --set selection.intervention_min_gap_reduction_by_bucket.test_contact=0.015

    # Ranking/budget tuned to let contact recovery happen, without reintroducing soft fallback.
    --set selection.prefer_admitted_by_bucket.safe=true
    --set selection.prefer_admitted_by_bucket.near_contact=true
    --set selection.prefer_admitted_by_bucket.contact=true
    --set selection.calibrated_shortfall_penalty_by_bucket.safe=0.05
    --set selection.calibrated_shortfall_penalty_by_bucket.near_contact=0.90
    --set selection.calibrated_shortfall_penalty_by_bucket.contact=0.80
    --set selection.calibrated_gap_penalty_by_bucket.safe=0.00
    --set selection.calibrated_gap_penalty_by_bucket.near_contact=0.35
    --set selection.calibrated_gap_penalty_by_bucket.contact=0.35
    --set selection.deployability_bonus_by_bucket.near_contact=0.90
    --set selection.deployability_bonus_by_bucket.contact=1.20
    --set selection.contact_deployability_bonus_by_bucket.contact=1.40
    --set selection.contact_gap_penalty_by_bucket.contact=0.35
    --set selection.intervention_budget_rate_by_bucket.safe=0.0
    --set selection.intervention_budget_rate_by_bucket.near_contact=0.20
    --set selection.intervention_budget_rate_by_bucket.contact=0.34
    --set selection.intervention_budget_penalty_by_bucket.safe=50.0
    --set selection.intervention_budget_penalty_by_bucket.near_contact=0.80
    --set selection.intervention_budget_penalty_by_bucket.contact=0.55
    --set selection.deviation_penalty_by_bucket.safe=3.0
    --set selection.deviation_penalty_by_bucket.near_contact=0.08
    --set selection.deviation_penalty_by_bucket.contact=0.04
    --set selection.intervention_penalty_by_bucket.safe=2.0
    --set selection.intervention_penalty_by_bucket.near_contact=0.020
    --set selection.intervention_penalty_by_bucket.contact=0.012

    --set selection.stress_nominal_anchor_by_bucket.near_contact="$anchor"
    --set selection.stress_nominal_anchor_by_bucket.contact="$anchor"
    --set selection.stress_anchor_drs_floor_by_bucket.near_contact=0.90
    --set selection.stress_anchor_drs_floor_by_bucket.contact=0.92
    --set selection.stress_anchor_max_gap_by_bucket.near_contact=1.05
    --set selection.stress_anchor_max_gap_by_bucket.contact=1.00
    --set selection.stress_anchor_rec_slack_by_bucket.near_contact=0.18
    --set selection.stress_anchor_rec_slack_by_bucket.contact=0.15
    --set selection.stress_anchor_min_drs_gain_by_bucket.near_contact=0.015
    --set selection.stress_anchor_min_drs_gain_by_bucket.contact=0.00
    --set selection.stress_anchor_min_rec_gain_by_bucket.near_contact=0.025
    --set selection.stress_anchor_min_rec_gain_by_bucket.contact=0.020
    --set selection.stress_anchor_min_gap_reduction_by_bucket.near_contact=0.020
    --set selection.stress_anchor_min_gap_reduction_by_bucket.contact=0.020
  )

  if [[ "$tag" == "rel_aggr" ]]; then
    COMMON_SEL+=(
      --set selection.relative_recovery_min_rec_gain_by_bucket.contact=0.10
      --set selection.relative_recovery_min_rec_gain_by_bucket.test_contact=0.10
      --set selection.relative_recovery_min_drs_by_bucket.contact=0.66
      --set selection.relative_recovery_min_drs_by_bucket.test_contact=0.66
      --set selection.relative_recovery_max_gap_by_bucket.contact=1.90
      --set selection.relative_recovery_max_gap_by_bucket.test_contact=1.90
      --set selection.relative_recovery_max_gap_increase_by_bucket.contact=0.50
      --set selection.relative_recovery_max_gap_increase_by_bucket.test_contact=0.50
      --set selection.intervention_max_pred_gap_by_bucket.contact=1.90
      --set selection.intervention_max_pred_gap_by_bucket.test_contact=1.90
    )
  fi
}

summarize_eval() {
  python - <<'PY' "$RUN" | tee "$RUN/summary_all_v16.txt"
import json, pathlib, sys
root=pathlib.Path(sys.argv[1])
print('\n===== Offline eval v16 =====')
for p in sorted(root.glob('eval_*_v16_*.json')):
    d=json.load(open(p)); print('\n', p.name)
    for m,r in d.get('methods',{}).items():
        if m in ['nominal','backup_filter','contingency','oracle_filter','ocrap','ocrap_teacher']:
            print(f"  {m:14s} FRA={r.get('FRA_exec')} DRS={r.get('DRS')} NUP={r.get('bounded_NUP')} ODG={r.get('ODG')} artifact={r.get('artifact_selection_rate')} PCD={r.get('post_contact_deployability')} int={r.get('intervention_rate')} reason={r.get('selection_reason_counts')}")
print('\n===== Closed-loop/audit v16 =====')
keys=['num_decisions','intervention_rate','closed_loop_bounded_NUP','closed_loop_FRA_exec','closed_loop_DRS','closed_loop_ODG','closed_loop_post_contact_deployability','closed_loop_artifact_selection_rate','closed_loop_audit_best_DRS','closed_loop_audit_best_R_dep','closed_loop_audit_selected_R_dep_regret','closed_loop_audit_selector_miss_rate','closed_loop_audit_recoverable_candidate_rate','closed_loop_pred_r_dep','closed_loop_pred_gap','closed_loop_pred_DRS_proxy']
for p in sorted(root.glob('*v16*.json')):
    try: d=json.load(open(p))
    except Exception: continue
    if 'closed_loop_bounded_NUP' not in d: continue
    print('\n', p.name)
    for k in keys:
        if k in d: print(f'  {k}: {d[k]}')
    print('  selection_reason_counts:', d.get('selection_reason_counts'))
PY
}

[[ -f "$CKPT" ]] || { echo "missing checkpoint $CKPT; set CKPT=/path/to/best.pt" >&2; exit 2; }
[[ -f "$GAMMA" ]] || { echo "missing gamma map $GAMMA; run v15 first or set GAMMA" >&2; exit 2; }
[[ -f "$CAL" ]] || { echo "missing calibration $CAL; run v15 first or set CAL" >&2; exit 2; }

# 1) Selector-only offline grid on the v15 checkpoint.
for TAG in scalar rel rel_dual rel_aggr; do
  case "$TAG" in
    scalar) ANCHOR=false ;;
    rel) ANCHOR=true ;;
    rel_dual) ANCHOR=true ;;
    rel_aggr) ANCHOR=false ;;
  esac
  make_common_sel "$TAG" "$ANCHOR" certified
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
      --output "$RUN/eval_${D}_v16_${TAG}.json" \
      --set evaluation.delta=0.05 \
      --set evaluation.group_by_dataset=true \
      --set evaluation.fallback_to_all_if_empty_split=true \
      "${COMMON_SEL[@]}" \
      --set 'evaluation.methods=[nominal,backup_filter,contingency,oracle_filter,ocrap,ocrap_teacher]' \
      | tee "$RUN/eval_${D}_v16_${TAG}.log"
  done
done

# 2) Closed-loop safe for the recommended rel_dual point.
make_common_sel rel_dual true certified
CUDA_VISIBLE_DEVICES=${GPU_EVAL:-0} PYTHONUNBUFFERED=1 python -u -m ocrap.cli closed-loop \
  --dataset "$WOMD_VAL@150" \
  --checkpoint "$CKPT" \
  --output "$RUN/closed_loop_safe_fast_v16_rel_dual.json" \
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
  | tee "$RUN/closed_loop_safe_fast_v16_rel_dual.log"

# 3) Selected-topk audits.  Run scalar, relative, and aggressive relative so we can
# directly isolate whether the new relative certificate fixes contact miss rate.
for TAG in scalar rel rel_dual rel_aggr; do
  case "$TAG" in
    scalar) ANCHOR=false ;;
    rel) ANCHOR=true ;;
    rel_dual) ANCHOR=true ;;
    rel_aggr) ANCHOR=false ;;
  esac
  make_common_sel "$TAG" "$ANCHOR" certified
  for B in near_contact contact; do
    case "$B" in
      near_contact) DATASET_RAW="$WOMD_VAL_INTERACTIVE@150"; BUCKET="$NEAR_TEST"; GPU=${GPU_AUDIT_NEAR:-0} ;;
      contact) DATASET_RAW="$WOMD_VAL_INTERACTIVE@150"; BUCKET="$CONTACT_TEST"; GPU=${GPU_AUDIT_CONTACT:-0} ;;
    esac
    CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python -u -m ocrap.cli closed-loop \
      --dataset "$DATASET_RAW" \
      --checkpoint "$CKPT" \
      --output "$RUN/audit_${B}_selected_topk_v16_${TAG}.json" \
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
      | tee "$RUN/audit_${B}_selected_topk_v16_${TAG}.log"
  done
done

summarize_eval
