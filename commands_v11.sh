#!/usr/bin/env bash
set -euo pipefail

# Run from OC-RAP repository root after applying ocrap_v11_group_ranking_shared_audit.patch
export OCRAP_ROOT=/data0/senzeyu2/dataset/OCRAP
export TRAIN_MIX="$OCRAP_ROOT/train_safe,$OCRAP_ROOT/train_near_contact,$OCRAP_ROOT/train_contact"
export VAL_MIX="$OCRAP_ROOT/val_safe,$OCRAP_ROOT/val_near_contact,$OCRAP_ROOT/val_contact"
export SAFE_TEST="$OCRAP_ROOT/test_safe"
export NEAR_TEST="$OCRAP_ROOT/test_near_contact"
export CONTACT_TEST="$OCRAP_ROOT/test_contact"
export WOMD_VAL=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation/validation_tfexample.tfrecord
export WOMD_VAL_INTERACTIVE=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation_interactive/validation_interactive_tfexample.tfrecord

# 0) Patch code if you are not using the patched zip.
# git apply /path/to/ocrap_v11_group_ranking_shared_audit.patch

# 1) Re-audit v10 checkpoint with patched shared-action selected audit.
export RUN10=runs/ocrap_v10
export CKPT10="$RUN10/model_v10/best.pt"
COMMON_SEL_AUDIT_V10=(
  --set selection.gamma_rec_by_bucket_file="$RUN10/gamma_rec_by_bucket_v10_delta05.json"
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
  --set selection.safe_force_nominal_mode_by_bucket.safe=always
  --set selection.safe_force_nominal_mode_by_bucket.safe_v2=always
  --set selection.safe_force_nominal_mode_by_bucket.test_safe=always
  --set selection.safe_force_nominal_mode_by_bucket.test_safe_v2=always
  --set selection.calibrated_shortfall_penalty_by_bucket.safe=0.02
  --set selection.calibrated_shortfall_penalty_by_bucket.near_contact=1.20
  --set selection.calibrated_shortfall_penalty_by_bucket.contact=1.60
  --set selection.calibrated_gap_penalty_by_bucket.safe=0.00
  --set selection.calibrated_gap_penalty_by_bucket.near_contact=0.15
  --set selection.calibrated_gap_penalty_by_bucket.contact=0.25
  --set selection.deployability_bonus_by_bucket.near_contact=0.35
  --set selection.deployability_bonus_by_bucket.contact=0.45
  --set selection.contact_deployability_bonus_by_bucket.contact=0.80
  --set selection.contact_gap_penalty_by_bucket.contact=0.25
  --set selection.prefer_admitted_by_bucket.safe=false
  --set selection.prefer_admitted_by_bucket.near_contact=true
  --set selection.prefer_admitted_by_bucket.contact=true
  --set selection.intervention_budget_rate_by_bucket.safe=0.0
  --set selection.intervention_budget_rate_by_bucket.near_contact=0.18
  --set selection.intervention_budget_rate_by_bucket.contact=0.35
  --set selection.intervention_budget_penalty_by_bucket.safe=50.0
  --set selection.intervention_budget_penalty_by_bucket.near_contact=1.0
  --set selection.intervention_budget_penalty_by_bucket.contact=0.6
  --set selection.deviation_penalty_by_bucket.safe=3.0
  --set selection.deviation_penalty_by_bucket.near_contact=0.10
  --set selection.deviation_penalty_by_bucket.contact=0.05
  --set selection.intervention_penalty_by_bucket.safe=2.0
  --set selection.intervention_penalty_by_bucket.near_contact=0.02
  --set selection.intervention_penalty_by_bucket.contact=0.01
)

mkdir -p "$RUN10/shared_audit_fix"
for B in safe near_contact contact; do
  case "$B" in
    safe) DATASET_RAW="$WOMD_VAL@150"; BUCKET="$SAFE_TEST"; GPU=0 ;;
    near_contact) DATASET_RAW="$WOMD_VAL_INTERACTIVE@150"; BUCKET="$NEAR_TEST"; GPU=0 ;;
    contact) DATASET_RAW="$WOMD_VAL_INTERACTIVE@150"; BUCKET="$CONTACT_TEST"; GPU=1 ;;
  esac
  CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python -u -m ocrap.cli closed-loop \
    --dataset "$DATASET_RAW" \
    --checkpoint "$CKPT10" \
    --output "$RUN10/shared_audit_fix/audit_${B}_selected_sharedmetric_on_v10.json" \
    "${COMMON_SEL_AUDIT_V10[@]}" \
    --set selection.stress_preserve_nominal_min_drs_drop_by_bucket.near_contact=0.05 \
    --set selection.stress_preserve_nominal_min_drs_drop_by_bucket.contact=0.05 \
    --set closed_loop.method=ocrap \
    --set closed_loop.bucket_dataset="$BUCKET" \
    --set closed_loop.bucket_split=test \
    --set closed_loop.max_bucket_targets=20 \
    --set closed_loop.max_rollouts=6 \
    --set closed_loop.raw_max_scenarios=500 \
    --set closed_loop.max_steps=20 \
    --set closed_loop.num_candidate_prefixes=12 \
    --set closed_loop.num_recovery_options=8 \
    --set closed_loop.label_mode=selected \
    --set closed_loop.audit_every_n_steps=4 \
    --set closed_loop.audit_max_labels=80 \
    --set closed_loop.progress_every_steps=1 \
    | tee "$RUN10/shared_audit_fix/audit_${B}_selected_sharedmetric_on_v10.log"
done

# 2) Train v11 with scene-time group batching and group-wise candidate ranking.
export RUN=runs/ocrap_v11
mkdir -p "$RUN"
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python -u -m ocrap.cli train \
  --dataset "$TRAIN_MIX" \
  --val-dataset "$VAL_MIX" \
  --output "$RUN/model_v11" \
  --set training.epochs=40 \
  --set training.batch_size=72 \
  --set training.lr=0.00010 \
  --set training.weight_decay=0.0003 \
  --set training.artifact_sampler_weight=2.5 \
  --set training.negative_deployable_sampler_weight=1.5 \
  --set training.safe_positive_sampler_weight=2.0 \
  --set training.regime_balance_power=0.8 \
  --set training.group_batching=true \
  --set training.group_batching_replacement=true \
  --set training.group_ranking_margin=0.25 \
  --set training.group_ranking_gap_weight=0.35 \
  --set training.group_ranking_teacher_gap_weight=0.35 \
  --set training.group_ranking_artifact_only=true \
  --set training.option_success_temperature=0.25 \
  --set training.early_stop_patience=8 \
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
  --set model.dropout=0.25 \
  --set loss_weights.margin=2.0 \
  --set loss_weights.obs=1.0 \
  --set loss_weights.anti_oracle=1.2 \
  --set loss_weights.artifact_gap=1.2 \
  --set loss_weights.admission=1.0 \
  --set loss_weights.option_q=0.5 \
  --set loss_weights.option_admission=1.0 \
  --set loss_weights.option_success=0.5 \
  --set loss_weights.option_success_bce=1.5 \
  --set loss_weights.option_best=2.0 \
  --set loss_weights.group_ranking=1.0 \
  --set loss_weights.utility=0.35 \
  | tee "$RUN/train_v11.log"

python - <<'PY' "$RUN/model_v11/train_summary.json" | tee "$RUN/loss_variance_v11.txt"
import json, sys
d=json.load(open(sys.argv[1]))
print("best_epoch:", d.get("best_epoch"))
print("best_val_loss:", d.get("best_val_loss"))
print("epochs_completed:", d.get("epochs_completed"))
for h in d.get("history", []):
    ep=h["epoch"]
    if ep in [1,2,3,4,5,8,10,15,20,25,30,35,40] or ep==d.get("best_epoch"):
        tr=h["train"]; va=h["val"]
        print(
            f"ep{ep:02d} train={tr.get('loss'):.3f} val={va.get('loss'):.3f} | "
            f"opt_bce={tr.get('loss_option_success_bce',0):.3f}/{va.get('loss_option_success_bce',0):.3f} "
            f"opt_best={tr.get('loss_option_best',0):.3f}/{va.get('loss_option_best',0):.3f} "
            f"group_rank={tr.get('loss_group_ranking',0):.3f}/{va.get('loss_group_ranking',0):.3f} "
            f"dep={tr.get('loss_dep',0):.3f}/{va.get('loss_dep',0):.3f} gap={tr.get('loss_gap',0):.3f}/{va.get('loss_gap',0):.3f}"
        )
PY

# 3) Calibrate v11, then create two gamma maps: calibrated and contact-floor-0.
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python -u -m ocrap.cli calibrate \
  --dataset "$OCRAP_ROOT/val_near_contact" \
  --checkpoint "$RUN/model_v11/best.pt" \
  --output "$RUN/calibration_near_v11.json" \
  --set calibration.deltas='[0.01,0.05,0.10]' \
  --set calibration.required_min_for_delta=20 \
  | tee "$RUN/calibration_near_v11.log"

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python -u -m ocrap.cli calibrate \
  --dataset "$OCRAP_ROOT/val_contact" \
  --checkpoint "$RUN/model_v11/best.pt" \
  --output "$RUN/calibration_contact_v11.json" \
  --set calibration.deltas='[0.01,0.05,0.10]' \
  --set calibration.required_min_for_delta=20 \
  | tee "$RUN/calibration_contact_v11.log"

python - <<'PY'
import json, math
from pathlib import Path
run=Path("runs/ocrap_v11")
def load_gamma(path, delta="0.05", default=0.0):
    d=json.load(open(path)); th=d.get("thresholds", {})
    for k,v in th.items():
        try:
            if abs(float(k)-float(delta))<1e-12 and math.isfinite(float(v)):
                return float(v)
        except Exception:
            pass
    return float(d.get("gamma_rec", default))
near=load_gamma(run/"calibration_near_v11.json")
contact=load_gamma(run/"calibration_contact_v11.json")
def make(contact_gamma, name):
    m={
      "safe":0.0,"safe_v2":0.0,"test_safe":0.0,"test_safe_v2":0.0,"train_safe":0.0,"val_safe":0.0,
      "near_contact":near,"test_near_contact":near,"train_near_contact":near,"val_near_contact":near,
      "contact":contact_gamma,"test_contact":contact_gamma,"train_contact":contact_gamma,"val_contact":contact_gamma,
    }
    out=run/name; json.dump(m, open(out,"w"), indent=2, sort_keys=True); print(out, json.dumps(m, sort_keys=True))
make(contact, "gamma_rec_by_bucket_v11_delta05.json")
make(max(0.0, contact), "gamma_rec_by_bucket_v11_delta05_contact_floor0.json")
PY

# 4) Offline eval for calibrated gamma and contact-floor-0 gamma.
run_eval_set () {
  local MAP=$1
  local TAG=$2
  COMMON_SEL_V11=(
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
    --set selection.safe_force_nominal_mode_by_bucket.safe=always
    --set selection.safe_force_nominal_mode_by_bucket.safe_v2=always
    --set selection.safe_force_nominal_mode_by_bucket.test_safe=always
    --set selection.safe_force_nominal_mode_by_bucket.test_safe_v2=always
    --set selection.calibrated_shortfall_penalty_by_bucket.safe=0.02
    --set selection.calibrated_shortfall_penalty_by_bucket.near_contact=1.20
    --set selection.calibrated_shortfall_penalty_by_bucket.contact=2.20
    --set selection.calibrated_gap_penalty_by_bucket.safe=0.00
    --set selection.calibrated_gap_penalty_by_bucket.near_contact=0.15
    --set selection.calibrated_gap_penalty_by_bucket.contact=0.35
    --set selection.deployability_bonus_by_bucket.near_contact=0.35
    --set selection.deployability_bonus_by_bucket.contact=0.70
    --set selection.contact_deployability_bonus_by_bucket.contact=1.20
    --set selection.contact_gap_penalty_by_bucket.contact=0.40
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
  )
  for D in safe near_contact contact; do
    case "$D" in
      safe) DATASET="$SAFE_TEST" ;;
      near_contact) DATASET="$NEAR_TEST" ;;
      contact) DATASET="$CONTACT_TEST" ;;
    esac
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python -u -m ocrap.cli evaluate \
      --dataset "$DATASET" \
      --checkpoint "$RUN/model_v11/best.pt" \
      --calibration "$RUN/calibration_near_v11.json" \
      --split test \
      --output "$RUN/eval_${D}_v11_${TAG}.json" \
      --set evaluation.delta=0.05 \
      --set evaluation.group_by_dataset=true \
      --set evaluation.fallback_to_all_if_empty_split=true \
      "${COMMON_SEL_V11[@]}" \
      --set 'evaluation.methods=[nominal,log_replay,mpc_proxy,backup_filter,contingency,oracle_filter,ocrap,ocrap_teacher]' \
      | tee "$RUN/eval_${D}_v11_${TAG}.log"
  done
}
run_eval_set "$RUN/gamma_rec_by_bucket_v11_delta05.json" calibrated
run_eval_set "$RUN/gamma_rec_by_bucket_v11_delta05_contact_floor0.json" contactfloor0

# 5) Selected-topk coverage audit. This is more useful than selected-only audit for diagnosing selector mistakes.
# Use the better gamma map after inspecting offline eval; default below uses contact-floor-0.
export GAMMA_MAP="$RUN/gamma_rec_by_bucket_v11_delta05_contact_floor0.json"
COMMON_SEL_V11_AUDIT=(
  --set selection.gamma_rec_by_bucket_file="$GAMMA_MAP"
  --set selection.ocrap_selector=calibrated_constrained
  --set selection.drs_success_gamma=0.0
  --set selection.drs_success_gamma_by_bucket.safe=0.0
  --set selection.drs_success_gamma_by_bucket.near_contact=0.0
  --set selection.drs_success_gamma_by_bucket.contact=0.0
  --set closed_loop.require_calibrated_selector=true
  --set closed_loop.require_gamma_by_bucket=true
  --set selection.safe_force_nominal_when_feasible_by_bucket.safe=true
  --set selection.safe_force_nominal_when_feasible_by_bucket.test_safe=true
  --set selection.safe_force_nominal_mode_by_bucket.safe=always
  --set selection.safe_force_nominal_mode_by_bucket.test_safe=always
  --set selection.calibrated_shortfall_penalty_by_bucket.near_contact=1.20
  --set selection.calibrated_shortfall_penalty_by_bucket.contact=2.20
  --set selection.calibrated_gap_penalty_by_bucket.near_contact=0.15
  --set selection.calibrated_gap_penalty_by_bucket.contact=0.35
  --set selection.deployability_bonus_by_bucket.near_contact=0.35
  --set selection.deployability_bonus_by_bucket.contact=0.70
  --set selection.contact_deployability_bonus_by_bucket.contact=1.20
  --set selection.contact_gap_penalty_by_bucket.contact=0.40
  --set selection.prefer_admitted_by_bucket.near_contact=true
  --set selection.prefer_admitted_by_bucket.contact=true
  --set selection.intervention_budget_rate_by_bucket.near_contact=0.18
  --set selection.intervention_budget_rate_by_bucket.contact=0.45
  --set selection.deviation_penalty_by_bucket.near_contact=0.10
  --set selection.deviation_penalty_by_bucket.contact=0.03
  --set selection.intervention_penalty_by_bucket.near_contact=0.02
  --set selection.intervention_penalty_by_bucket.contact=0.005
)
for B in near_contact contact; do
  case "$B" in
    near_contact) DATASET_RAW="$WOMD_VAL_INTERACTIVE@150"; BUCKET="$NEAR_TEST"; GPU=0 ;;
    contact) DATASET_RAW="$WOMD_VAL_INTERACTIVE@150"; BUCKET="$CONTACT_TEST"; GPU=1 ;;
  esac
  CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 python -u -m ocrap.cli closed-loop \
    --dataset "$DATASET_RAW" \
    --checkpoint "$RUN/model_v11/best.pt" \
    --output "$RUN/audit_${B}_topk_v11.json" \
    "${COMMON_SEL_V11_AUDIT[@]}" \
    --set selection.stress_preserve_nominal_min_drs_drop_by_bucket.near_contact=0.05 \
    --set selection.stress_preserve_nominal_min_drs_drop_by_bucket.contact=0.05 \
    --set closed_loop.method=ocrap \
    --set closed_loop.bucket_dataset="$BUCKET" \
    --set closed_loop.bucket_split=test \
    --set closed_loop.max_bucket_targets=20 \
    --set closed_loop.max_rollouts=6 \
    --set closed_loop.raw_max_scenarios=500 \
    --set closed_loop.max_steps=20 \
    --set closed_loop.num_candidate_prefixes=12 \
    --set closed_loop.num_recovery_options=8 \
    --set closed_loop.label_mode=selected_topk \
    --set closed_loop.audit_every_n_steps=4 \
    --set closed_loop.audit_max_labels=240 \
    --set closed_loop.progress_every_steps=1 \
    | tee "$RUN/audit_${B}_topk_v11.log"
done

# 6) Compact summary.
python - <<'PY' "$RUN" | tee "$RUN/summary_all_v11.txt"
import json, pathlib, sys
root=pathlib.Path(sys.argv[1])
print("===== Eval =====")
for p in sorted(root.glob("eval_*_v11_*.json")):
    d=json.load(open(p)); print("\n", p.name)
    for m,r in d.get("methods",{}).items():
        print(f"  {m:16s} FRA={r.get('FRA_exec')} DRS={r.get('DRS')} NUP={r.get('bounded_NUP')} ODG={r.get('ODG')} artifact={r.get('artifact_selection_rate')} PCD={r.get('post_contact_deployability')} predDRS={r.get('mean_selected_pred_DRS_proxy')} reason={r.get('selection_reason_counts')}")
print("\n===== Closed-loop audit =====")
for p in sorted(root.glob("audit_*_v11.json")):
    d=json.load(open(p)); print("\n", p.name)
    for k in ["num_decisions","intervention_rate","closed_loop_FRA_exec","closed_loop_DRS","closed_loop_ODG","closed_loop_post_contact_deployability","closed_loop_artifact_selection_rate","closed_loop_audit_selected_R_dep_regret","closed_loop_audit_selector_miss_rate","closed_loop_audit_candidate_count","closed_loop_audit_recoverable_candidate_rate","closed_loop_pred_DRS_proxy"]:
        print(" ", k, d.get(k))
    print("  reason", d.get("selection_reason_counts"))
PY
