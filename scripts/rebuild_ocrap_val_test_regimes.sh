#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# OC-RAP clean validation / internal-test reconstruction on WOMD v1.3.1.
#
# Design:
#   * use only standard WOMD validation TFExamples (future labels are present);
#   * never use official testing/testing_interactive for teacher labels;
#   * never mix validation and validation_interactive in the primary IID table;
#   * isolate every regime and split by disjoint TFRecord shard ranges;
#   * use identical construction parameters for val and test within a regime;
#   * avoid scenario_start_index entirely (the supplied code had a double-skip).
#
# Apply ocrap_dataset_integrity.patch first.  In particular, the contact build
# uses post_contact_counterfactual to avoid conflating observed contact with a
# generated contact-surrogate branch.
# -----------------------------------------------------------------------------

export WOMD_ROOT=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example
export OCRAP_ROOT=/data0/senzeyu2/dataset/OCRAP
VAL_DIR="${WOMD_ROOT}/validation"

# Validation bank: 00000--00039.  Held-out internal-test bank: 00100--00139.
# Remaining validation shards stay untouched for calibration/extra audits.
export VAL_SAFE_PATTERN="${VAL_DIR}/validation_tfexample.tfrecord-000[01][0-9]-of-00150"   # 00000--00019, 20 shards
export VAL_NEAR_PATTERN="${VAL_DIR}/validation_tfexample.tfrecord-0002[0-9]-of-00150"      # 00020--00029, 10 shards
export VAL_CONTACT_PATTERN="${VAL_DIR}/validation_tfexample.tfrecord-0003[0-9]-of-00150"   # 00030--00039, 10 shards
export TEST_SAFE_PATTERN="${VAL_DIR}/validation_tfexample.tfrecord-001[01][0-9]-of-00150"  # 00100--00119, 20 shards
export TEST_NEAR_PATTERN="${VAL_DIR}/validation_tfexample.tfrecord-0012[0-9]-of-00150"     # 00120--00129, 10 shards
export TEST_CONTACT_PATTERN="${VAL_DIR}/validation_tfexample.tfrecord-0013[0-9]-of-00150"  # 00130--00139, 10 shards

check_glob() {
  local pattern="$1"
  local expected="$2"
  local label="$3"
  local count
  count=$(python - "$pattern" <<'PY'
import glob, sys
print(len(glob.glob(sys.argv[1])))
PY
)
  if [[ "$count" -ne "$expected" ]]; then
    echo "ERROR: ${label}: expected ${expected} shard files, found ${count}: ${pattern}" >&2
    exit 2
  fi
  echo "${label}: ${count} shards"
}

check_glob "$VAL_SAFE_PATTERN" 20 val_safe
check_glob "$VAL_NEAR_PATTERN" 10 val_near_contact
check_glob "$VAL_CONTACT_PATTERN" 10 val_contact
check_glob "$TEST_SAFE_PATTERN" 20 test_safe
check_glob "$TEST_NEAR_PATTERN" 10 test_near_contact
check_glob "$TEST_CONTACT_PATTERN" 10 test_contact

for d in val_safe val_near_contact val_contact test_safe test_near_contact test_contact; do
  if [[ -e "${OCRAP_ROOT}/${d}" ]]; then
    echo "ERROR: ${OCRAP_ROOT}/${d} already exists. Remove/rename it deliberately before rebuilding." >&2
    exit 3
  fi
done

COMMON=(
  --set data_source=womd
  --set simulation_backend=waymax_closed_loop
  --set num_candidate_prefixes=24
  --set num_reactive_futures=2
  --set num_roots=8
  --set num_recovery_options=12
  --set waymax.dataloader_include_sdc_paths=true
  --set 'waymax.metrics_to_run=[log_divergence,overlap,offroad,sdc_wrongway,sdc_off_route,sdc_progression,kinematic_infeasibility]'
  --set waymax.teacher_backend=hybrid
  --set waymax.use_jit_scan_rollouts=true
  --set waymax.enable_augmented_hidden_roots=true
  --set waymax.augmented_hidden_from_unknown_only=true
  --set dataset_quality.require_nominal_per_scene_time=true
  --set dataset_quality.keep_nominal_even_if_quality_fails=true
  --set dataset_quality.min_accepted_prefixes_per_scene_time=2
  --set regime_thresholds.include_prefix_collision_in_near=false
  --set regime_thresholds.include_prefix_contact_in_post=false
  --set regime_thresholds.use_paper_regime_definitions=true
  --set io.compress_npz=false
  --set io.fsync_npz=false
)

build_safe() {
  local split="$1" pattern="$2" max_scenarios="$3" output="$4"
  python -m ocrap.cli build-dataset \
    "${COMMON[@]}" \
    --set womd_patterns="${pattern}" \
    --set max_scenarios="${max_scenarios}" \
    --set split.force_id="${split}" \
    --set max_times_per_scenario=3 \
    --set max_biased_times_per_scenario=0 \
    --set dataset_quality.min_uniform_times_per_scenario=3 \
    --set num_targeted_futures=0 \
    --set waymax.compute_future_metrics=false \
    --set waymax.teacher_rollout_top_k_options=4 \
    --set waymax.enable_augmented_hidden_roots=false \
    --set waymax.enable_visible_perturbation_roots=false \
    --set artifact.force_mine=false \
    --set artifact.mine_probability=0.0 \
    --set artifact.use_margin_override=false \
    --set dataset_quality.balanced_two_pass=false \
    --set dataset_quality.artifact_pair_mode=tag \
    --set dataset_quality.max_accepted_prefixes_per_scene_time=8 \
    --set 'dataset_quality.require_nominal_regimes=[normal]' \
    --set 'dataset_quality.forbid_nominal_regimes=[near_contact,post_contact,oracle_artifact,prefix_collision,prefix_contact]' \
    --set 'dataset_quality.forbid_any_regimes=[near_contact,post_contact,oracle_artifact,prefix_collision,prefix_contact]' \
    --set regime_thresholds.tau_occ=0.75 \
    --set regime_thresholds.tau_normal_occ=0.90 \
    --set regime_thresholds.require_uniform_for_normal=true \
    --output "${output}"
}

build_near() {
  local split="$1" pattern="$2" max_scenarios="$3" output="$4"
  python -m ocrap.cli build-dataset \
    "${COMMON[@]}" \
    --set womd_patterns="${pattern}" \
    --set max_scenarios="${max_scenarios}" \
    --set split.force_id="${split}" \
    --set max_times_per_scenario=3 \
    --set max_biased_times_per_scenario=3 \
    --set num_targeted_futures=8 \
    --set 'targeted_future_kinds=[hidden_vehicle_yields,hidden_vehicle_accelerates,low_friction_braking,control_delay_noise]' \
    --set waymax.compute_future_metrics=true \
    --set waymax.teacher_rollout_top_k_options=0 \
    --set waymax.teacher_metrics_stride=0 \
    --set waymax.enable_visible_perturbation_roots=true \
    --set artifact.force_mine=true \
    --set artifact.mine_probability=0.30 \
    --set artifact.use_margin_override=false \
    --set artifact.enable_branch_intent_margin=true \
    --set artifact.branch_intent_compatible_margin=1.0 \
    --set artifact.branch_intent_incompatible_margin=-2.5 \
    --set dataset_quality.balanced_two_pass=true \
    --set dataset_quality.artifact_pair_mode=balanced \
    --set dataset_quality.artifact_quota_uses_label=true \
    --set dataset_quality.require_artifact_pairs=true \
    --set dataset_quality.max_accepted_prefixes_per_scene_time=8 \
    --set dataset_quality.min_artifact_prefixes_per_scene_time=1 \
    --set dataset_quality.max_artifact_prefixes_per_scene_time=2 \
    --set dataset_quality.min_nonartifact_prefixes_per_scene_time=4 \
    --set dataset_quality.max_nonartifact_prefixes_per_scene_time=6 \
    --set dataset_quality.max_artifact_attempts_per_scene_time=24 \
    --set dataset_quality.max_nonartifact_attempts_per_scene_time=12 \
    --set 'dataset_quality.require_nominal_regimes=[near_contact]' \
    --set 'dataset_quality.require_any_regimes=[oracle_artifact]' \
    --set 'dataset_quality.forbid_nominal_regimes=[post_contact,prefix_collision,prefix_contact]' \
    --set 'dataset_quality.forbid_any_regimes=[post_contact,prefix_collision,prefix_contact]' \
    --set dataset_quality.artifact_pass_use_margin_override=true \
    --set dataset_quality.artifact_pass_skip_augmented_waymax=true \
    --set dataset_quality.artifact_pass_apply_override_to_screened=true \
    --set dataset_quality.artifact_pass_compute_future_metrics=false \
    --output "${output}"
}

build_contact() {
  local split="$1" pattern="$2" max_scenarios="$3" output="$4"
  python -m ocrap.cli build-dataset \
    "${COMMON[@]}" \
    --set womd_patterns="${pattern}" \
    --set max_scenarios="${max_scenarios}" \
    --set split.force_id="${split}" \
    --set max_times_per_scenario=4 \
    --set max_biased_times_per_scenario=4 \
    --set num_targeted_futures=10 \
    --set 'targeted_future_kinds=[hidden_vehicle_yields,hidden_vehicle_accelerates,contact_impulse_surrogate,secondary_collision_approach,low_friction_braking,control_delay_noise]' \
    --set waymax.compute_future_metrics=true \
    --set waymax.teacher_rollout_top_k_options=0 \
    --set waymax.teacher_metrics_stride=0 \
    --set waymax.enable_visible_perturbation_roots=true \
    --set artifact.force_mine=true \
    --set artifact.mine_probability=0.25 \
    --set artifact.use_margin_override=false \
    --set artifact.enable_branch_intent_margin=true \
    --set artifact.branch_intent_compatible_margin=1.0 \
    --set artifact.branch_intent_incompatible_margin=-2.5 \
    --set dataset_quality.balanced_two_pass=true \
    --set dataset_quality.artifact_pair_mode=balanced \
    --set dataset_quality.artifact_quota_uses_label=true \
    --set dataset_quality.require_artifact_pairs=true \
    --set dataset_quality.max_accepted_prefixes_per_scene_time=9 \
    --set dataset_quality.min_artifact_prefixes_per_scene_time=1 \
    --set dataset_quality.max_artifact_prefixes_per_scene_time=2 \
    --set dataset_quality.min_nonartifact_prefixes_per_scene_time=5 \
    --set dataset_quality.max_nonartifact_prefixes_per_scene_time=7 \
    --set dataset_quality.max_artifact_attempts_per_scene_time=24 \
    --set dataset_quality.max_nonartifact_attempts_per_scene_time=12 \
    --set 'dataset_quality.require_nominal_regimes=[post_contact_counterfactual]' \
    --set 'dataset_quality.require_any_regimes=[oracle_artifact]' \
    --set 'dataset_quality.forbid_nominal_regimes=[post_contact_observed,prefix_collision,prefix_contact]' \
    --set 'dataset_quality.forbid_any_regimes=[post_contact_observed,prefix_collision,prefix_contact]' \
    --set dataset_quality.artifact_pass_use_margin_override=true \
    --set dataset_quality.artifact_pass_skip_augmented_waymax=true \
    --set dataset_quality.artifact_pass_apply_override_to_screened=true \
    --set dataset_quality.artifact_pass_compute_future_metrics=false \
    --output "${output}"
}

# Safe acceptance is low, so scan more raw scenarios.  Near/contact use strict
# regime + artifact gates and therefore also receive a larger raw budget than the
# previous 100/160-scenario commands.
build_safe val  "$VAL_SAFE_PATTERN"     800  "$OCRAP_ROOT/val_safe"
build_near val  "$VAL_NEAR_PATTERN"     400  "$OCRAP_ROOT/val_near_contact"
build_contact val "$VAL_CONTACT_PATTERN" 400 "$OCRAP_ROOT/val_contact"

build_safe test "$TEST_SAFE_PATTERN"     1200 "$OCRAP_ROOT/test_safe"
build_near test "$TEST_NEAR_PATTERN"     600  "$OCRAP_ROOT/test_near_contact"
build_contact test "$TEST_CONTACT_PATTERN" 600 "$OCRAP_ROOT/test_contact"

mkdir -p "$OCRAP_ROOT/reports"
for d in val_safe val_near_contact val_contact test_safe test_near_contact test_contact; do
  python -m ocrap.cli diagnose \
    --dataset "$OCRAP_ROOT/$d" \
    --output "$OCRAP_ROOT/reports/diagnose_${d}.json"
  python -m ocrap.cli papercheck \
    --dataset "$OCRAP_ROOT/$d" \
    --output "$OCRAP_ROOT/reports/papercheck_${d}.json"
done

echo "Rebuild complete. Review ${OCRAP_ROOT}/reports before training/evaluation."
