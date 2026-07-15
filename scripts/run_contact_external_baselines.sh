#!/usr/bin/env bash
set -euo pipefail

: "${OCRAP_ROOT:=/data0/senzeyu2/dataset/OCRAP}"
: "${TRAIN_CONTACT:=$OCRAP_ROOT/train_contact}"
: "${VAL_CONTACT:=$OCRAP_ROOT/val_contact}"
: "${TEST_CONTACT:=$OCRAP_ROOT/test_contact}"
: "${RUN:=runs/contact_external_baselines}"
: "${WOMD_VAL_INTERACTIVE:=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation_interactive/validation_interactive_tfexample.tfrecord}"
: "${CL_WOMD:=${WOMD_VAL_INTERACTIVE}@150}"
: "${CL_MAX_SCENARIOS:=50}"
: "${CL_MAX_STEPS:=40}"
: "${CL_LABEL_MAX_CANDIDATES:=10}"
: "${CL_NUM_RECOVERY_OPTIONS:=12}"
: "${CL_TEACHER_TOP_K_OPTIONS:=4}"
: "${CL_EXHAUSTIVE_TEACHER_LABELS:=false}"
: "${CL_SAVE_PARTIAL:=false}"
: "${OMP_NUM_THREADS:=8}"
: "${MKL_NUM_THREADS:=8}"
export OMP_NUM_THREADS MKL_NUM_THREADS

mkdir -p "$RUN"

CONTACT_METHODS=(
  postimpact_mpc_lite
  post_crash_braking
  post_collision_restoration
  severity_minimization
)
ALL_CONTACT_METHODS="postimpact_mpc_lite,post_crash_braking,post_collision_restoration,severity_minimization"

# These contact-regime external baselines are optimization/rule/heuristic
# planners over the OC-RAP candidate lattice, so train-baseline performs a
# deterministic registration and dataset sanity pass rather than neural fitting.
for method in "${CONTACT_METHODS[@]}"; do
  python -u -m ocrap.cli train-baseline \
    --config configs/external_baselines/contact_external_baselines.yaml \
    --dataset "$TRAIN_CONTACT" \
    --val-dataset "$VAL_CONTACT" \
    --baseline "$method" \
    --output "$RUN/$method" \
    2>&1 | tee "$RUN/train_${method}.log"
done

# Offline grouped candidate-set evaluation on the held-out contact bucket.
python -u -m ocrap.cli evaluate-baseline \
  --config configs/external_baselines/contact_external_baselines.yaml \
  --dataset "$TEST_CONTACT" \
  --split test \
  --output "$RUN/eval_contact_external_baselines.json" \
  --baselines "$ALL_CONTACT_METHODS" \
  2>&1 | tee "$RUN/eval_contact_external_baselines.log"

# Waymax receding-horizon closed-loop evaluation.  label_mode=all is deliberate:
# all four contact baselines use branch/root post-contact labels or hard/harm
# labels directly as part of their paper-defined planner/filter score.
for method in "${CONTACT_METHODS[@]}"; do
  python -u -m ocrap.cli closed-loop \
    --config configs/external_baselines/contact_external_baselines.yaml \
    --dataset "$CL_WOMD" \
    --output "$RUN/closed_loop_${method}.json" \
    --set closed_loop.method="$method" \
    --set closed_loop.max_scenarios="$CL_MAX_SCENARIOS" \
    --set closed_loop.max_steps="$CL_MAX_STEPS" \
    --set closed_loop.replan_interval_steps=1 \
    --set closed_loop.label_mode=all \
    --set closed_loop.external_sparse_labels=true \
    --set closed_loop.external_label_max_candidates="$CL_LABEL_MAX_CANDIDATES" \
    --set closed_loop.external_label_macro_diversity=true \
    --set closed_loop.exhaustive_teacher_labels="$CL_EXHAUSTIVE_TEACHER_LABELS" \
    --set closed_loop.num_candidate_prefixes=24 \
    --set closed_loop.num_recovery_options="$CL_NUM_RECOVERY_OPTIONS" \
    --set closed_loop.save_partial="$CL_SAVE_PARTIAL" \
    --set waymax.compute_future_metrics=false \
    --set waymax.teacher_metrics_stride=0 \
    --set waymax.teacher_rollout_top_k_options="$CL_TEACHER_TOP_K_OPTIONS" \
    2>&1 | tee "$RUN/closed_loop_${method}.log"
done

python - <<'PY'
import glob, json, os
run = os.environ.get('RUN', 'runs/contact_external_baselines')
rows = []
for p in sorted(glob.glob(os.path.join(run, 'closed_loop_*.json'))):
    try:
        d = json.load(open(p))
    except Exception:
        continue
    rows.append({
        k: d.get(k)
        for k in [
            'method', 'num_scenes', 'num_decisions',
            'closed_loop_FRA_exec', 'closed_loop_FRA_cand', 'closed_loop_DRS',
            'closed_loop_post_contact_deployability', 'closed_loop_ODG',
            'closed_loop_bounded_NUP', 'intervention_rate'
        ]
    })
out = os.path.join(run, 'closed_loop_summary.json')
json.dump(rows, open(out, 'w'), indent=2)
print({'event': 'contact_closed_loop_summary', 'output': out, 'num_methods': len(rows)})
PY
