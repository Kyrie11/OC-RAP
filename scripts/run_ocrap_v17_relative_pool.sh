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

export RUN=${RUN:-runs/ocrap_v17_relative_pool}
export V15_RUN=${V15_RUN:-runs/ocrap_v15_dualcert}
export CKPT=${CKPT:-$V15_RUN/model_v15/best.pt}
export CAL=${CAL:-$V15_RUN/calibration_near_rdep_v15.json}
export GAMMA=${GAMMA:-$V15_RUN/gamma_rec_by_bucket_v15_floor0.json}
mkdir -p "$RUN"

[[ -f "$CKPT" ]] || { echo "missing checkpoint $CKPT; set CKPT=/path/to/best.pt" >&2; exit 2; }
[[ -f "$GAMMA" ]] || { echo "missing gamma map $GAMMA; set GAMMA=/path/to/gamma.json" >&2; exit 2; }
[[ -f "$CAL" ]] || { echo "missing calibration $CAL; set CAL=/path/to/calibration.json" >&2; exit 2; }

make_sel() {
  local tag="$1"  # scalar | v17
  local rel=false
  local rec_pool=false
  case "$tag" in
    scalar) rel=false; rec_pool=false ;;
    v17) rel=true; rec_pool=true ;;
    *) echo "unknown selector tag: $tag" >&2; exit 2 ;;
  esac

  COMMON_SEL=(
    --set selection.gamma_rec_by_bucket_file="$GAMMA"
    --set selection.ocrap_selector=calibrated_constrained
    --set closed_loop.require_calibrated_selector=true
    --set closed_loop.require_gamma_by_bucket=true
    --set evaluation.require_calibrated_selector=true
    --set evaluation.require_gamma_by_bucket=true

    # Safety invariant: no soft/unadmitted recovery fallback in paper runs.
    --set selection.require_admitted_intervention_by_bucket.safe=true
    --set selection.require_admitted_intervention_by_bucket.safe_v2=true
    --set selection.require_admitted_intervention_by_bucket.test_safe=true
    --set selection.require_admitted_intervention_by_bucket.test_safe_v2=true
    --set selection.require_admitted_intervention_by_bucket.near_contact=true
    --set selection.require_admitted_intervention_by_bucket.test_near_contact=true
    --set selection.require_admitted_intervention_by_bucket.contact=true
    --set selection.require_admitted_intervention_by_bucket.test_contact=true
    --set selection.unadmitted_fallback_to_nominal=true

    # Safe/normal regime: explicitly nominal-preserving when nominal is feasible.
    --set selection.safe_force_nominal_when_feasible_by_bucket.safe=true
    --set selection.safe_force_nominal_when_feasible_by_bucket.safe_v2=true
    --set selection.safe_force_nominal_when_feasible_by_bucket.test_safe=true
    --set selection.safe_force_nominal_when_feasible_by_bucket.test_safe_v2=true
    --set selection.safe_force_nominal_mode_by_bucket.safe=feasible
    --set selection.safe_force_nominal_mode_by_bucket.safe_v2=feasible
    --set selection.safe_force_nominal_mode_by_bucket.test_safe=feasible
    --set selection.safe_force_nominal_mode_by_bucket.test_safe_v2=feasible

    # v17: no absolute option-DRS shortcut in the main run.  v16 dual/aggr raised
    # interventions but reduced DRS, so the next run isolates the relative-pool fix.
    --set selection.option_drs_certificate_by_bucket.safe=false
    --set selection.option_drs_certificate_by_bucket.test_safe=false
    --set selection.option_drs_certificate_by_bucket.near_contact=false
    --set selection.option_drs_certificate_by_bucket.test_near_contact=false
    --set selection.option_drs_certificate_by_bucket.contact=false
    --set selection.option_drs_certificate_by_bucket.test_contact=false

    # Relative recovery certificate.  It is active only in stress regimes and uses
    # a recovery-feasibility pool so post-contact stabilization/pull-over actions
    # are not rejected solely because they violate a nominal hard-rule flag.
    --set selection.relative_recovery_certificate_by_bucket.safe=false
    --set selection.relative_recovery_certificate_by_bucket.safe_v2=false
    --set selection.relative_recovery_certificate_by_bucket.test_safe=false
    --set selection.relative_recovery_certificate_by_bucket.near_contact="$rel"
    --set selection.relative_recovery_certificate_by_bucket.test_near_contact="$rel"
    --set selection.relative_recovery_certificate_by_bucket.contact="$rel"
    --set selection.relative_recovery_certificate_by_bucket.test_contact="$rel"
    --set selection.relative_recovery_use_recovery_pool_by_bucket.near_contact="$rec_pool"
    --set selection.relative_recovery_use_recovery_pool_by_bucket.test_near_contact="$rec_pool"
    --set selection.relative_recovery_use_recovery_pool_by_bucket.contact="$rec_pool"
    --set selection.relative_recovery_use_recovery_pool_by_bucket.test_contact="$rec_pool"
    --set selection.relative_recovery_gate_by_bucket.near_contact=rec_or_gap
    --set selection.relative_recovery_gate_by_bucket.test_near_contact=rec_or_gap
    --set selection.relative_recovery_gate_by_bucket.contact=rec_or_gap
    --set selection.relative_recovery_gate_by_bucket.test_contact=rec_or_gap
    --set selection.relative_recovery_counts_as_evidence=true

    # Trigger condition: only when nominal is low-headroom or high-gap.
    --set selection.relative_recovery_nominal_rec_lcb_max_by_bucket.near_contact=-0.25
    --set selection.relative_recovery_nominal_rec_lcb_max_by_bucket.test_near_contact=-0.25
    --set selection.relative_recovery_nominal_rec_lcb_max_by_bucket.contact=-0.45
    --set selection.relative_recovery_nominal_rec_lcb_max_by_bucket.test_contact=-0.45
    --set selection.relative_recovery_nominal_gap_min_by_bucket.near_contact=1.00
    --set selection.relative_recovery_nominal_gap_min_by_bucket.test_near_contact=1.00
    --set selection.relative_recovery_nominal_gap_min_by_bucket.contact=0.95
    --set selection.relative_recovery_nominal_gap_min_by_bucket.test_contact=0.95
    --set selection.relative_recovery_nominal_drs_max_by_bucket.near_contact=0.86
    --set selection.relative_recovery_nominal_drs_max_by_bucket.test_near_contact=0.86
    --set selection.relative_recovery_nominal_drs_max_by_bucket.contact=0.88
    --set selection.relative_recovery_nominal_drs_max_by_bucket.test_contact=0.88

    # Candidate dominance and guards.  Near-contact is stricter; contact permits
    # controlled recovery with a nominal hard-rule violation up to 2.
    --set selection.recovery_cert_max_hard_by_bucket.near_contact=1.0
    --set selection.recovery_cert_max_hard_by_bucket.test_near_contact=1.0
    --set selection.recovery_cert_max_hard_by_bucket.contact=2.0
    --set selection.recovery_cert_max_hard_by_bucket.test_contact=2.0
    --set selection.recovery_cert_max_harm_by_bucket.near_contact=0.60
    --set selection.recovery_cert_max_harm_by_bucket.test_near_contact=0.60
    --set selection.recovery_cert_max_harm_by_bucket.contact=1.20
    --set selection.recovery_cert_max_harm_by_bucket.test_contact=1.20
    --set selection.relative_recovery_min_rec_gain_by_bucket.near_contact=0.05
    --set selection.relative_recovery_min_rec_gain_by_bucket.test_near_contact=0.05
    --set selection.relative_recovery_min_rec_gain_by_bucket.contact=0.02
    --set selection.relative_recovery_min_rec_gain_by_bucket.test_contact=0.02
    --set selection.relative_recovery_min_gap_reduction_by_bucket.near_contact=0.08
    --set selection.relative_recovery_min_gap_reduction_by_bucket.test_near_contact=0.08
    --set selection.relative_recovery_min_gap_reduction_by_bucket.contact=0.05
    --set selection.relative_recovery_min_gap_reduction_by_bucket.test_contact=0.05
    --set selection.relative_recovery_min_drs_by_bucket.near_contact=0.78
    --set selection.relative_recovery_min_drs_by_bucket.test_near_contact=0.78
    --set selection.relative_recovery_min_drs_by_bucket.contact=0.66
    --set selection.relative_recovery_min_drs_by_bucket.test_contact=0.66
    --set selection.relative_recovery_min_drs_gain_by_bucket.near_contact=-1.0
    --set selection.relative_recovery_min_drs_gain_by_bucket.test_near_contact=-1.0
    --set selection.relative_recovery_min_drs_gain_by_bucket.contact=-1.0
    --set selection.relative_recovery_min_drs_gain_by_bucket.test_contact=-1.0
    --set selection.relative_recovery_max_gap_by_bucket.near_contact=1.45
    --set selection.relative_recovery_max_gap_by_bucket.test_near_contact=1.45
    --set selection.relative_recovery_max_gap_by_bucket.contact=1.70
    --set selection.relative_recovery_max_gap_by_bucket.test_contact=1.70
    --set selection.relative_recovery_max_gap_increase_by_bucket.near_contact=0.05
    --set selection.relative_recovery_max_gap_increase_by_bucket.test_near_contact=0.05
    --set selection.relative_recovery_max_gap_increase_by_bucket.contact=0.15
    --set selection.relative_recovery_max_gap_increase_by_bucket.test_contact=0.15
    --set selection.relative_recovery_bonus_by_bucket.near_contact=1.20
    --set selection.relative_recovery_bonus_by_bucket.test_near_contact=1.20
    --set selection.relative_recovery_bonus_by_bucket.contact=2.00
    --set selection.relative_recovery_bonus_by_bucket.test_contact=2.00

    # Evidence and ranking.  These remain calibrated/observation-consistent; they
    # only determine which certified candidate is selected.
    --set selection.require_intervention_evidence_by_bucket.safe=true
    --set selection.require_intervention_evidence_by_bucket.test_safe=true
    --set selection.require_intervention_evidence_by_bucket.near_contact=true
    --set selection.require_intervention_evidence_by_bucket.test_near_contact=true
    --set selection.require_intervention_evidence_by_bucket.contact=true
    --set selection.require_intervention_evidence_by_bucket.test_contact=true
    --set selection.intervention_min_pred_drs_by_bucket.near_contact=0.72
    --set selection.intervention_min_pred_drs_by_bucket.test_near_contact=0.72
    --set selection.intervention_min_pred_drs_by_bucket.contact=0.64
    --set selection.intervention_min_pred_drs_by_bucket.test_contact=0.64
    --set selection.intervention_max_pred_gap_by_bucket.near_contact=1.45
    --set selection.intervention_max_pred_gap_by_bucket.test_near_contact=1.45
    --set selection.intervention_max_pred_gap_by_bucket.contact=1.70
    --set selection.intervention_max_pred_gap_by_bucket.test_contact=1.70
    --set selection.intervention_min_rec_lcb_gain_by_bucket.near_contact=0.00
    --set selection.intervention_min_rec_lcb_gain_by_bucket.test_near_contact=0.00
    --set selection.intervention_min_rec_lcb_gain_by_bucket.contact=0.00
    --set selection.intervention_min_rec_lcb_gain_by_bucket.test_contact=0.00
    --set selection.intervention_min_gap_reduction_by_bucket.near_contact=0.00
    --set selection.intervention_min_gap_reduction_by_bucket.test_near_contact=0.00
    --set selection.intervention_min_gap_reduction_by_bucket.contact=0.00
    --set selection.intervention_min_gap_reduction_by_bucket.test_contact=0.00
    --set selection.prefer_admitted_by_bucket.safe=true
    --set selection.prefer_admitted_by_bucket.near_contact=true
    --set selection.prefer_admitted_by_bucket.contact=true
    --set selection.calibrated_shortfall_penalty_by_bucket.safe=0.05
    --set selection.calibrated_shortfall_penalty_by_bucket.near_contact=0.80
    --set selection.calibrated_shortfall_penalty_by_bucket.contact=0.65
    --set selection.calibrated_gap_penalty_by_bucket.safe=0.00
    --set selection.calibrated_gap_penalty_by_bucket.near_contact=0.30
    --set selection.calibrated_gap_penalty_by_bucket.contact=0.25
    --set selection.deployability_bonus_by_bucket.near_contact=0.70
    --set selection.deployability_bonus_by_bucket.contact=0.90
    --set selection.contact_deployability_bonus_by_bucket.contact=0.80
    --set selection.contact_gap_penalty_by_bucket.contact=0.20
    --set selection.intervention_budget_rate_by_bucket.safe=0.0
    --set selection.intervention_budget_rate_by_bucket.near_contact=0.22
    --set selection.intervention_budget_rate_by_bucket.contact=0.36
    --set selection.intervention_budget_penalty_by_bucket.safe=50.0
    --set selection.intervention_budget_penalty_by_bucket.near_contact=0.75
    --set selection.intervention_budget_penalty_by_bucket.contact=0.50
    --set selection.deviation_penalty_by_bucket.safe=3.0
    --set selection.deviation_penalty_by_bucket.near_contact=0.08
    --set selection.deviation_penalty_by_bucket.contact=0.04
    --set selection.intervention_penalty_by_bucket.safe=2.0
    --set selection.intervention_penalty_by_bucket.near_contact=0.018
    --set selection.intervention_penalty_by_bucket.contact=0.010
  )
}

run_eval() {
  local tag="$1"; local d="$2"; local gpu="$3"
  make_sel "$tag"
  local dataset="$SAFE_TEST"
  [[ "$d" == "near_contact" ]] && dataset="$NEAR_TEST"
  [[ "$d" == "contact" ]] && dataset="$CONTACT_TEST"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 python -u -m ocrap.cli evaluate \
    --dataset "$dataset" --checkpoint "$CKPT" --calibration "$CAL" --split test \
    --output "$RUN/eval_${d}_v17_${tag}.json" \
    --set evaluation.delta=0.05 \
    --set evaluation.group_by_dataset=true \
    --set evaluation.fallback_to_all_if_empty_split=true \
    "${COMMON_SEL[@]}" \
    --set 'evaluation.methods=[nominal,backup_filter,contingency,oracle_filter,ocrap,ocrap_teacher]' \
    | tee "$RUN/eval_${d}_v17_${tag}.log"
}

run_audit() {
  local tag="$1"; local b="$2"; local gpu="$3"
  make_sel "$tag"
  local bucket="$NEAR_TEST"
  [[ "$b" == "contact" ]] && bucket="$CONTACT_TEST"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 python -u -m ocrap.cli closed-loop \
    --dataset "$WOMD_VAL_INTERACTIVE@150" --checkpoint "$CKPT" \
    --output "$RUN/audit_${b}_selected_topk_v17_${tag}.json" \
    "${COMMON_SEL[@]}" \
    --set closed_loop.method=ocrap \
    --set closed_loop.bucket_dataset="$bucket" \
    --set closed_loop.bucket_split=test \
    --set closed_loop.max_bucket_targets=32 \
    --set closed_loop.max_rollouts=12 \
    --set closed_loop.raw_max_scenarios=900 \
    --set closed_loop.max_steps=20 \
    --set closed_loop.num_candidate_prefixes=12 \
    --set closed_loop.num_recovery_options=8 \
    --set closed_loop.label_mode=selected_topk \
    --set closed_loop.audit_every_n_steps=4 \
    --set closed_loop.audit_max_labels=384 \
    --set closed_loop.audit_top_k=10 \
    --set closed_loop.audit_max_extra_candidates=9 \
    --set closed_loop.progress_every_steps=1 \
    | tee "$RUN/audit_${b}_selected_topk_v17_${tag}.log"
}

run_safe_closed_loop() {
  make_sel v17
  CUDA_VISIBLE_DEVICES=${GPU_SAFE:-0} PYTHONUNBUFFERED=1 python -u -m ocrap.cli closed-loop \
    --dataset "$WOMD_VAL@150" --checkpoint "$CKPT" \
    --output "$RUN/closed_loop_safe_fast_v17.json" \
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
    | tee "$RUN/closed_loop_safe_fast_v17.log"
}

summarize() {
  python - <<'PY' "$RUN" | tee "$RUN/summary_all_v17.txt"
import json, pathlib, sys
root=pathlib.Path(sys.argv[1])
print("\n===== Offline eval v17 =====")
for p in sorted(root.glob('eval_*_v17_*.json')):
    d=json.load(open(p)); print('\n', p.name)
    for m,r in d.get('methods',{}).items():
        if m in ['nominal','backup_filter','contingency','oracle_filter','ocrap','ocrap_teacher']:
            print(f"  {m:14s} FRA={r.get('FRA_exec')} DRS={r.get('DRS')} NUP={r.get('bounded_NUP')} ODG={r.get('ODG')} artifact={r.get('artifact_selection_rate')} PCD={r.get('post_contact_deployability')} int={r.get('intervention_rate')} reason={r.get('selection_reason_counts')}")
print("\n===== Closed-loop/audit v17 =====")
keys=['num_decisions','intervention_rate','closed_loop_bounded_NUP','closed_loop_FRA_exec','closed_loop_DRS','closed_loop_ODG','closed_loop_post_contact_deployability','closed_loop_artifact_selection_rate','closed_loop_audit_best_DRS','closed_loop_audit_best_R_dep','closed_loop_audit_selected_R_dep_regret','closed_loop_audit_selector_miss_rate','closed_loop_audit_recoverable_candidate_rate','closed_loop_pred_r_dep','closed_loop_pred_gap','closed_loop_pred_DRS_proxy']
for p in sorted(root.glob('*v17*.json')):
    try: d=json.load(open(p))
    except Exception: continue
    if 'closed_loop_bounded_NUP' not in d: continue
    print('\n', p.name)
    for k in keys:
        if k in d: print(f'  {k}: {d[k]}')
    print('  macro_counts:', d.get('macro_counts'))
    print('  selection_reason_counts:', d.get('selection_reason_counts'))
PY
}

# Offline selector grid. Run safe/contact on GPU0 and near-contact on GPU1 where possible.
run_eval scalar safe 0 &
run_eval v17 safe 1 &
wait
run_eval scalar near_contact 0 &
run_eval v17 near_contact 1 &
wait
run_eval scalar contact 0 &
run_eval v17 contact 1 &
wait

# Closed-loop/audit. Two GPUs are used concurrently for the stress buckets.
run_audit scalar near_contact 0 &
run_audit v17 near_contact 1 &
wait
run_audit scalar contact 0 &
run_audit v17 contact 1 &
wait

# Safe audit is cheaper and uses GPU_SAFE (default GPU0) after stress audits to avoid GPU contention.
run_safe_closed_loop

summarize
