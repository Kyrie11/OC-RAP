#!/usr/bin/env bash
# Build a treatment-independent Contact cohort from actual Waymax overlap states.
# Evaluation engineering only: the prelude always executes exact upstream a0.
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
# shellcheck source=scripts/lib/runtime.sh
source scripts/lib/runtime.sh

: "${OCRAP_ROOT:=/data0/senzeyu2/dataset/OCRAP}"
: "${CONTACT_BUCKET:=$OCRAP_ROOT/test_contact}"
: "${BUCKET_SPLIT:=test}"
: "${WOMD_ROOT:=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example}"
: "${WOMD_NUM_SHARDS:=150}"
: "${CONTACT_WOMD:=auto}"
: "${WOMD_ROLE:=validation}"
: "${OUT:?OUT is required}"
: "${GPU:=0}"
: "${PRELUDE_MAX_STEPS:=60}"
: "${PRELUDE_REPLAN_INTERVAL:=1}"
: "${MIN_POST_STEPS:=40}"
: "${MAX_TARGETS_PER_SCENE:=100000}"
: "${MAX_SCENARIOS:=0}"
: "${SOURCE_TARGET_KEYS_FILE:=}"

if [[ "${CONTACT_WOMD,,}" == auto ]]; then
  CONTACT_WOMD="$(runtime_resolve_bucket_womd_spec "$CONTACT_BUCKET" "$BUCKET_SPLIT" "$WOMD_ROOT" "$WOMD_NUM_SHARDS" "$WOMD_ROLE")"
else
  CONTACT_WOMD="$(runtime_normalize_womd_spec "$CONTACT_WOMD" "$WOMD_NUM_SHARDS")"
fi
if [[ -n "$SOURCE_TARGET_KEYS_FILE" ]]; then
  [[ -s "$SOURCE_TARGET_KEYS_FILE" ]] || { echo "missing SOURCE_TARGET_KEYS_FILE: $SOURCE_TARGET_KEYS_FILE" >&2; exit 30; }
fi
MINE="$OUT/mining"
MINING_JSON="$MINE/closed_loop_nominal.json"
MANIFEST="$OUT/contact_anchor_manifest.json"
KEYS="$OUT/contact_anchor_target_keys.json"
mkdir -p "$MINE" "$OUT"
rm -f "$MINING_JSON" "$MINING_JSON.progress.json" "$MINING_JSON.scenes.jsonl" "$MANIFEST" "$KEYS"

target_support_args=()
target_eval_args=()
if [[ -n "$SOURCE_TARGET_KEYS_FILE" ]]; then
  target_support_args=(--target-keys-file "$SOURCE_TARGET_KEYS_FILE" --require-target-keys)
  target_eval_args=(
    --set "closed_loop.target_keys_file=$SOURCE_TARGET_KEYS_FILE"
    --set closed_loop.require_target_keys=true
  )
fi

python tools/check_closed_loop_dataset_support.py --dataset "$CONTACT_BUCKET" --split "$BUCKET_SPLIT" \
  --womd-pattern "$CONTACT_WOMD" --expected-source-role "$WOMD_ROLE" \
  "${target_support_args[@]}" \
  --output "$MINE/closed_loop_dataset_support.json"

export CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR="$MINE/.jax_compilation_cache"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
python -u -m ocrap.cli closed-loop \
  --config configs/external_baselines/nominal_log_replay.yaml \
  --dataset "$CONTACT_WOMD" --output "$MINING_JSON" \
  --set closed_loop.method=nominal \
  --set "closed_loop.max_scenarios=$MAX_SCENARIOS" \
  --set "closed_loop.max_bucket_targets=$MAX_SCENARIOS" \
  --set "closed_loop.bucket_dataset=$CONTACT_BUCKET" \
  --set "closed_loop.bucket_split=$BUCKET_SPLIT" \
  --set closed_loop.require_bucket_targets=true \
  --set "closed_loop.max_targets_per_scene=$MAX_TARGETS_PER_SCENE" \
  --set closed_loop.max_steps=1 \
  --set closed_loop.label_mode=fast \
  --set closed_loop.contact_anchor_prelude_enabled=true \
  --set "closed_loop.contact_anchor_prelude_max_steps=$PRELUDE_MAX_STEPS" \
  --set "closed_loop.contact_anchor_prelude_replan_interval_steps=$PRELUDE_REPLAN_INTERVAL" \
  --set closed_loop.contact_anchor_require_found=false \
  --set closed_loop.contact_anchor_mining_only=true \
  --set closed_loop.include_scenes_in_result=true \
  --set closed_loop.result_scene_detail=metrics \
  --set closed_loop.scene_journal_detail=metrics \
  --set closed_loop.memory_scene_detail=metrics \
  --set closed_loop.render_trace=false \
  --set closed_loop.resume=false \
  --set closed_loop.use_sdc_paths=true \
  --set closed_loop.require_observation_legal_route=true \
  --set closed_loop.allow_future_route_proxy=false \
  --set waymax.dataloader_include_sdc_paths=true \
  --set waymax.allow_logged_sdc_route_fallback=false \
  --set waymax.compute_future_metrics=false \
  --set waymax.teacher_metrics_stride=0 \
  --set waymax.use_jit_scan_rollouts=true \
  "${target_eval_args[@]}" \
  2>&1 | tee -a "$MINE/contact_anchor_mining.log"

python tools/build_contact_anchor_manifest.py --mining-result "$MINING_JSON" \
  --min-post-steps "$MIN_POST_STEPS" --output "$MANIFEST" --target-keys-output "$KEYS"
python - "$MANIFEST" "$SOURCE_TARGET_KEYS_FILE" "$CONTACT_WOMD" "$WOMD_ROLE" <<'PY'
import hashlib,json,pathlib,sys
p=pathlib.Path(sys.argv[1])
m=json.loads(p.read_text(encoding='utf-8'))
source=sys.argv[2]
m['source_target_keys_file']=str(pathlib.Path(source).resolve()) if source else None
m['source_target_keys_sha256']=(hashlib.sha256(pathlib.Path(source).read_bytes()).hexdigest() if source else None)
m['route_contract']={
    'route_source':'womd_v1_3_1_sdc_paths_connectivity_only',
    'require_observation_legal_route':True,
    'allow_logged_sdc_route_fallback':False,
    'allow_future_route_proxy':False,
}
m['womd_pattern']=sys.argv[3]
m['womd_source_role']=sys.argv[4]
p.write_text(json.dumps(m,indent=2,sort_keys=True)+'\n',encoding='utf-8')
if not m.get('valid') or int(m.get('num_selected_anchors') or 0)<=0:
    raise SystemExit('no causally valid Contact anchors were mined; fail closed')
if int(m.get('num_selected_anchors')) != int(m.get('num_selected_scenes')):
    raise SystemExit('Contact anchor manifest is not scene-disjoint')
print(json.dumps({k:m.get(k) for k in ('num_mining_targets','num_selected_anchors','num_selected_scenes','min_post_steps','source_target_keys_sha256')},indent=2))
PY
printf '%s\n' "$MANIFEST"
printf '%s\n' "$KEYS"
