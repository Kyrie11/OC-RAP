#!/usr/bin/env bash
set -euo pipefail

# Exploratory-only closed-loop after an RC=20 Natural-gate rejection.
# This script never authorizes deployment or paper claims. It re-runs OC-RAP,
# scalar control, and external baselines on exactly the same target roots, then
# creates paired reports and auditable critical-scene videos.

: "${MODEL_RUN:?set MODEL_RUN to the v48.34 dedicated main-run directory}"
: "${OUT:=runs/ocrap_v48_34_exploratory_closed_loop}"
: "${ALLOW_DIAGNOSTIC_RC20:=0}"
: "${EXPLORATORY_DATA_SCOPE:=adaptation_dev}" # adaptation_dev | heldout_test
: "${ALLOW_HELDOUT_TEST_DIAGNOSTIC:=0}"
: "${VARIANTS:=balanced,precision}"
: "${VIDEO_VARIANTS:=precision}"
: "${CUDA_DEVICES:=0,1}"
: "${BUCKET_SPLIT:=test}"
: "${MAX_TARGETS:=50}"
: "${MAX_STEPS:=40}"
: "${NUM_CANDIDATES:=24}"
: "${NUM_RECOVERY_OPTIONS:=12}"
: "${AUDIT_LABELS:=800}"
: "${NUM_POSITIVE_VIDEOS:=3}"
: "${NUM_FAILURE_VIDEOS:=2}"
: "${VIDEO_FPS:=10}"
: "${RUN_SAFE:=1}"
: "${RUN_NEAR:=1}"
: "${RUN_CONTACT:=1}"
: "${RUN_EXTERNAL_BASELINES:=1}"
: "${RUN_VIDEOS:=1}"
: "${REQUIRE_ALL_BASELINES:=1}"

: "${OCRAP_ROOT:=/data0/senzeyu2/dataset/OCRAP}"
: "${PROTOCOL_ROOT:=$OCRAP_ROOT/calibration_v48_14_prism_4814}"
# Resolve data roots from the declared exploratory scope.  This prevents the
# nominally adaptation-dev command from silently reading held-out test roots.
if [[ "$EXPLORATORY_DATA_SCOPE" == "adaptation_dev" ]]; then
  : "${SAFE_TARGET_DATASET:=${SAFE_ADAPTATION_DEV_DATASET:-}}"
  : "${NEAR_TARGET_DATASET:=$PROTOCOL_ROOT/evidence_adapt_dev_near_contact}"
  : "${CONTACT_TARGET_DATASET:=$PROTOCOL_ROOT/evidence_adapt_dev_contact}"
else
  : "${SAFE_TARGET_DATASET:=$OCRAP_ROOT/test_safe}"
  : "${NEAR_TARGET_DATASET:=$OCRAP_ROOT/test_near_contact}"
  : "${CONTACT_TARGET_DATASET:=$OCRAP_ROOT/test_contact}"
fi
: "${WOMD_VAL:=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation/validation_tfexample.tfrecord}"
: "${WOMD_VAL_INTERACTIVE:=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation_interactive/validation_interactive_tfexample.tfrecord}"
: "${SAFE_WOMD_SOURCE:=${WOMD_VAL}@150}"
: "${NEAR_WOMD_SOURCE:=${WOMD_VAL}@150}"
: "${CONTACT_WOMD_SOURCE:=${WOMD_VAL_INTERACTIVE}@150}"

: "${SAFE_METHODS:=wayformer_bc,gameformer_lite,betopnet_lite}"
: "${NEAR_METHODS:=gameformer_lite,marc_lite,racp_lite,predictive_safety_filter,cvar_risk_filter}"
: "${CONTACT_METHODS:=postimpact_mpc_lite,post_crash_braking,post_collision_restoration,severity_minimization}"

if [[ "$ALLOW_DIAGNOSTIC_RC20" != "1" ]]; then
  echo "Refusing exploratory post-gate closed-loop. Set ALLOW_DIAGNOSTIC_RC20=1 explicitly." >&2
  exit 2
fi
if [[ "$EXPLORATORY_DATA_SCOPE" == "heldout_test" && "$ALLOW_HELDOUT_TEST_DIAGNOSTIC" != "1" ]]; then
  echo "Held-out test inspection after RC=20 contaminates future model selection. Set ALLOW_HELDOUT_TEST_DIAGNOSTIC=1 explicitly." >&2
  exit 2
fi
if [[ "$EXPLORATORY_DATA_SCOPE" != "adaptation_dev" && "$EXPLORATORY_DATA_SCOPE" != "heldout_test" ]]; then
  echo "EXPLORATORY_DATA_SCOPE must be adaptation_dev or heldout_test" >&2
  exit 2
fi
if [[ "$RUN_SAFE" == "1" && -z "$SAFE_TARGET_DATASET" ]]; then
  echo "Safe has no implicit adaptation-dev target root. Set SAFE_ADAPTATION_DEV_DATASET or RUN_SAFE=0." >&2
  exit 2
fi
for pair in "near:$RUN_NEAR:$NEAR_TARGET_DATASET" "contact:$RUN_CONTACT:$CONTACT_TARGET_DATASET" "safe:$RUN_SAFE:$SAFE_TARGET_DATASET"; do
  IFS=: read -r regime enabled dataset <<< "$pair"
  if [[ "$enabled" == "1" && ! -d "$dataset" ]]; then
    echo "missing $regime target dataset for scope $EXPLORATORY_DATA_SCOPE: $dataset" >&2
    exit 3
  fi
done

mkdir -p "$OUT" "$OUT/logs" "$OUT/reports" "$OUT/baselines" "$OUT/models" "$OUT/videos"
IFS=',' read -r -a GPU_LIST <<< "$CUDA_DEVICES"
[[ ${#GPU_LIST[@]} -ge 1 ]] || GPU_LIST=(0)
GPU0="${GPU_LIST[0]}"; GPU1="${GPU_LIST[1]:-${GPU_LIST[0]}}"

python - "$MODEL_RUN" "$OUT" "$EXPLORATORY_DATA_SCOPE" <<'PY'
import glob, json, os, pathlib, sys, time
root=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[2]); scope=sys.argv[3]
files=sorted(root.glob('V48_*_COMPLETE.json'))
if not files:
    raise SystemExit(f'no V48_*_COMPLETE.json under {root}')
d=json.load(open(files[-1]))
rc=int(d.get('pipeline_exit_code',-1))
if rc not in (0,20) or not d.get('certificate_executed') or not d.get('gate_evaluated'):
    raise SystemExit(f'exploratory run requires a completed RC=0/20 certificate, got {d}')
notice={
 'event':'v48_34_exploratory_closed_loop_authorization', 'created_unix':time.time(),
 'source_completion':str(files[-1]), 'source_pipeline_exit_code':rc,
 'natural_gate_passed':bool(d.get('gate_passed')), 'data_scope':scope,
 'exploratory_only':True, 'paper_claim_allowed':False, 'deployment_allowed':False,
 'may_not_be_used_for_checkpoint_threshold_or_algorithm_selection':True,
 'heldout_test_contaminated_for_future_model_selection':scope=='heldout_test',
 'required_reporting':'report all paired scenes, positive examples, and failure cases; do not cherry-pick videos',
}
(out/'EXPLORATORY_ONLY_DISCLOSURE.json').write_text(json.dumps(notice,indent=2)+'\n')
print(notice)
PY

python - "$OUT" "$EXPLORATORY_DATA_SCOPE" \
  "$RUN_SAFE" "$SAFE_TARGET_DATASET" "$SAFE_WOMD_SOURCE" \
  "$RUN_NEAR" "$NEAR_TARGET_DATASET" "$NEAR_WOMD_SOURCE" \
  "$RUN_CONTACT" "$CONTACT_TARGET_DATASET" "$CONTACT_WOMD_SOURCE" <<'PY'
import hashlib, json, pathlib, sys, time

def hash_path(path_text):
    p=pathlib.Path(path_text)
    if not p.exists():
        return None
    h=hashlib.sha256()
    if p.is_file():
        h.update(p.read_bytes())
        return h.hexdigest()
    files=[]
    for candidate in p.rglob('*'):
        if candidate.is_file() and candidate.name in {
            'manifest.json','targets.json','target_manifest.json','bucket_manifest.json',
            'metadata.json','dataset_manifest.json','index.json'
        }:
            files.append(candidate)
    if not files:
        # Dataset directories can be very large.  Hash the exact resolved path and
        # a sorted shallow inventory rather than silently traversing TFRecords.
        h.update(str(p.resolve()).encode())
        for candidate in sorted(p.iterdir(), key=lambda x: x.name):
            try:
                st=candidate.stat()
            except OSError:
                continue
            h.update(candidate.name.encode()); h.update(str(st.st_size).encode()); h.update(str(st.st_mtime_ns).encode())
        return h.hexdigest()
    for candidate in sorted(files):
        h.update(str(candidate.relative_to(p)).encode()); h.update(candidate.read_bytes())
    return h.hexdigest()

out=pathlib.Path(sys.argv[1]); scope=sys.argv[2]
raw=sys.argv[3:]
regimes={}
for i,regime in enumerate(('safe','near','contact')):
    enabled=raw[i*3]=='1'; target=raw[i*3+1]; womd=raw[i*3+2]
    regimes[regime]={
        'enabled':enabled,
        'target_dataset':str(pathlib.Path(target).resolve()) if target else '',
        'target_contract_sha256':hash_path(target) if enabled else None,
        'womd_source':womd,
    }
contract={
    'event':'v48_34_exploratory_data_scope_contract',
    'created_unix':time.time(),
    'data_scope':scope,
    'heldout_test_contaminated_for_future_model_selection':scope=='heldout_test',
    'regimes':regimes,
}
(out/'EXPLORATORY_DATA_SCOPE_CONTRACT.json').write_text(json.dumps(contract,indent=2)+'\n')
print(contract)
PY

sha_file() { sha256sum "$1" | awk '{print $1}'; }
require_file() { [[ -f "$1" ]] || { echo "missing required file: $1" >&2; exit 3; }; }

model_ckpt() { printf '%s/candidates/%s/model_v48_trac_sr/best.pt' "$MODEL_RUN" "$1"; }
model_base() { printf '%s/dedicated_candidates/%s' "$MODEL_RUN" "$1"; }

run_ocrap_regime() {
  local variant="$1" regime="$2" target="$3" womd="$4" run_dir="$5" render="${6:-false}"
  local ckpt base
  ckpt="$(model_ckpt "$variant")"; base="$(model_base "$variant")"
  require_file "$ckpt"
  require_file "$base/calibration/direct_value_risk_near_v48.json"
  require_file "$base/calibration/direct_value_risk_contact_v48.json"
  mkdir -p "$run_dir"
  local common=(
    env RUN="$run_dir" BASE_RUN="$base" CKPT="$ckpt"
    DEV_SHADOW_DIAGNOSTIC=1 RUN_OFFLINE_EVAL=0 RUN_SAFE_CLOSED_LOOP=0
    RUN_AUDITS=1 RUN_SCALAR_BASELINES=1 PAIR_BUCKETS_ON_TWO_GPUS=0
    AUDIT_TARGETS="$MAX_TARGETS" AUDIT_MAX_ROLLOUTS="$MAX_TARGETS"
    AUDIT_MAX_STEPS="$MAX_STEPS" AUDIT_NUM_CANDIDATES="$NUM_CANDIDATES"
    AUDIT_NUM_RECOVERY_OPTIONS="$NUM_RECOVERY_OPTIONS" AUDIT_LABELS="$AUDIT_LABELS"
    AUDIT_LABEL_MODE=selected_topk AUDIT_EVERY_N_STEPS=4 AUDIT_TOP_K=10
    CL_RENDER_TRACE="$render" CL_RENDER_MAX_AGENTS=64 CL_RESUME="${CL_RESUME:-true}"
    BUCKET_SPLIT="$BUCKET_SPLIT" DEV_SHADOW_WOMD_SOURCE="$womd"
  )
  if [[ "$regime" == "near" ]]; then
    "${common[@]}" NEAR_TEST="$target" CONTACT_TEST="$CONTACT_TARGET_DATASET" \
      RUN_NEAR_AUDIT=1 RUN_CONTACT_AUDIT=0 bash scripts/run_ocrap_v48_trac_sr.sh \
      > >(tee "$run_dir/controller.log") 2>&1
  elif [[ "$regime" == "contact" ]]; then
    "${common[@]}" NEAR_TEST="$NEAR_TARGET_DATASET" CONTACT_TEST="$target" \
      RUN_NEAR_AUDIT=0 RUN_CONTACT_AUDIT=1 bash scripts/run_ocrap_v48_trac_sr.sh \
      > >(tee "$run_dir/controller.log") 2>&1
  else
    echo "unsupported OC-RAP regime $regime" >&2; return 2
  fi
}

run_ocrap_safe() {
  local variant="$1" target="$2" womd="$3" run_dir="$4" render="${5:-false}"
  local ckpt; ckpt="$(model_ckpt "$variant")"; require_file "$ckpt"; mkdir -p "$run_dir"
  env RUN="$run_dir" CKPT="$ckpt" SAFE_NOMINAL_ONLY=1 \
    RUN_OFFLINE_EVAL=0 RUN_AUDITS=0 RUN_SAFE_CLOSED_LOOP=1 RUN_SAFE_PAIRED_SCALAR=1 \
    SAFE_TEST="$target" SAFE_WOMD_SOURCE="$womd" SAFE_MAX_TARGETS="$MAX_TARGETS" \
    SAFE_MAX_ROLLOUTS="$MAX_TARGETS" SAFE_MAX_STEPS="$MAX_STEPS" \
    SAFE_NUM_CANDIDATES="$NUM_CANDIDATES" SAFE_NUM_RECOVERY_OPTIONS="$NUM_RECOVERY_OPTIONS" \
    CL_RENDER_TRACE="$render" CL_RENDER_MAX_AGENTS=64 CL_RESUME="${CL_RESUME:-true}" \
    GPU_SAFE_BASELINE="$GPU0" GPU_SAFE="$GPU1" \
    bash scripts/run_ocrap_v48_trac_sr.sh > >(tee "$run_dir/controller.log") 2>&1
}

external_checkpoint() {
  case "$1" in
    wayformer_bc) printf '%s' "${SAFE_WAYFORMER_CHECKPOINT:-}" ;;
    gameformer_lite)
      if [[ "$2" == "safe" ]]; then printf '%s' "${SAFE_GAMEFORMER_CHECKPOINT:-}"; else printf '%s' "${NEAR_GAMEFORMER_CHECKPOINT:-}"; fi ;;
    betopnet_lite) printf '%s' "${SAFE_BETOPNET_CHECKPOINT:-}" ;;
    *) printf '' ;;
  esac
}
external_config() {
  case "$1:$2" in
    safe:wayformer_bc) echo configs/external_baselines/wayformer_bc.yaml ;;
    safe:gameformer_lite) echo configs/external_baselines/gameformer_lite.yaml ;;
    safe:betopnet_lite) echo configs/external_baselines/betopnet_lite.yaml ;;
    near:gameformer_lite) echo configs/external_baselines/near_contact_gameformer_lite.yaml ;;
    near:*) echo configs/external_baselines/near_contact_external_baselines.yaml ;;
    contact:*) echo configs/external_baselines/contact_external_baselines.yaml ;;
    *) return 2 ;;
  esac
}

run_external_one() {
  local regime="$1" method="$2" target="$3" womd="$4" gpu="$5" render="${6:-false}"
  local cfg ckpt out_dir out_file
  cfg="$(external_config "$regime" "$method")"
  ckpt="$(external_checkpoint "$method" "$regime")"
  if [[ "$method" =~ ^(wayformer_bc|gameformer_lite|betopnet_lite)$ ]]; then
    if [[ -z "$ckpt" || ! -f "$ckpt" ]]; then
      if [[ "$REQUIRE_ALL_BASELINES" == "1" ]]; then
        echo "missing $regime baseline checkpoint for $method: ${ckpt:-unset}" >&2; return 3
      fi
      echo "[SKIP] missing $regime baseline checkpoint for $method" >&2; return 0
    fi
  fi
  out_dir="$OUT/baselines/$regime"; mkdir -p "$out_dir"
  out_file="$out_dir/closed_loop_${method}.json"
  local checkpoint_args=(); [[ -z "$ckpt" ]] || checkpoint_args=(--checkpoint "$ckpt")
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 python -u -m ocrap.cli closed-loop \
    --config "$cfg" --dataset "$womd" "${checkpoint_args[@]}" --output "$out_file" \
    --set closed_loop.method="$method" \
    --set closed_loop.bucket_dataset="$target" --set closed_loop.bucket_split="$BUCKET_SPLIT" \
    --set closed_loop.require_bucket_targets=true --set closed_loop.allow_legacy_source_index_targets=true \
    --set closed_loop.max_bucket_targets="$MAX_TARGETS" --set closed_loop.max_targets_per_scene=1 \
    --set closed_loop.max_rollouts="$MAX_TARGETS" --set closed_loop.raw_max_scenarios=0 \
    --set closed_loop.max_steps="$MAX_STEPS" --set closed_loop.replan_interval_steps=1 \
    --set closed_loop.label_mode=selected --set closed_loop.force_teacher_baselines=false \
    --set closed_loop.external_sparse_labels=false --set closed_loop.exhaustive_teacher_labels=false \
    --set closed_loop.num_candidate_prefixes="$NUM_CANDIDATES" \
    --set closed_loop.num_recovery_options="$NUM_RECOVERY_OPTIONS" \
    --set closed_loop.save_partial=true --set closed_loop.resume=true \
    --set closed_loop.render_trace="$render" --set closed_loop.render_max_agents=64 \
    --set closed_loop.profile_timing=true --set closed_loop.audit_every_n_steps=1 \
    --set waymax.retain_official_scenario_id=true \
    --set waymax.dataloader_include_sdc_paths=false --set waymax.compute_future_metrics=false \
    --set waymax.teacher_metrics_stride=0 --set waymax.use_jit_scan_rollouts=true \
    > >(tee "$out_dir/closed_loop_${method}.log") 2>&1
}

run_method_list() {
  local regime="$1" csv="$2" target="$3" womd="$4"
  IFS=',' read -r -a methods <<< "$csv"
  local i=0 pids=() names=()
  for method in "${methods[@]}"; do
    method="${method//[[:space:]]/}"; [[ -n "$method" ]] || continue
    gpu="$GPU0"; (( i % 2 == 0 )) || gpu="$GPU1"
    run_external_one "$regime" "$method" "$target" "$womd" "$gpu" false &
    pids+=("$!"); names+=("$regime:$method"); i=$((i+1))
    if (( ${#pids[@]} == 2 )); then
      failed=0
      for j in "${!pids[@]}"; do wait "${pids[$j]}" || { echo "[ERROR] ${names[$j]}" >&2; failed=1; }; done
      (( failed == 0 )) || return 1
      pids=(); names=()
    fi
  done
  failed=0
  for j in "${!pids[@]}"; do wait "${pids[$j]}" || { echo "[ERROR] ${names[$j]}" >&2; failed=1; }; done
  (( failed == 0 ))
}

IFS=',' read -r -a VARIANT_LIST <<< "$VARIANTS"
for variant in "${VARIANT_LIST[@]}"; do
  variant="${variant//[[:space:]]/}"; [[ -n "$variant" ]] || continue
  if [[ "$RUN_SAFE" == "1" ]]; then run_ocrap_safe "$variant" "$SAFE_TARGET_DATASET" "$SAFE_WOMD_SOURCE" "$OUT/models/$variant/safe" false; fi
  if [[ "$RUN_NEAR" == "1" ]]; then run_ocrap_regime "$variant" near "$NEAR_TARGET_DATASET" "$NEAR_WOMD_SOURCE" "$OUT/models/$variant/near" false; fi
  if [[ "$RUN_CONTACT" == "1" ]]; then run_ocrap_regime "$variant" contact "$CONTACT_TARGET_DATASET" "$CONTACT_WOMD_SOURCE" "$OUT/models/$variant/contact" false; fi
done

if [[ "$RUN_EXTERNAL_BASELINES" == "1" ]]; then
  [[ "$RUN_SAFE" != "1" ]] || run_method_list safe "$SAFE_METHODS" "$SAFE_TARGET_DATASET" "$SAFE_WOMD_SOURCE"
  [[ "$RUN_NEAR" != "1" ]] || run_method_list near "$NEAR_METHODS" "$NEAR_TARGET_DATASET" "$NEAR_WOMD_SOURCE"
  [[ "$RUN_CONTACT" != "1" ]] || run_method_list contact "$CONTACT_METHODS" "$CONTACT_TARGET_DATASET" "$CONTACT_WOMD_SOURCE"
fi

make_report() {
  local variant="$1" regime="$2" reference="$3" model="$4" methods_csv="$5"
  local args=(--reference "scalar=$reference" --method "ocrap_${variant}=$model")
  IFS=',' read -r -a methods <<< "$methods_csv"
  for method in "${methods[@]}"; do
    method="${method//[[:space:]]/}"; [[ -n "$method" ]] || continue
    p="$OUT/baselines/$regime/closed_loop_${method}.json"
    [[ -s "$p.scenes.jsonl" ]] && args+=(--method "$method=$p")
  done
  python tools/build_v48_34_paired_baseline_report.py "${args[@]}" \
    --output-json "$OUT/reports/${variant}_${regime}_paired.json" \
    --output-csv "$OUT/reports/${variant}_${regime}_paired.csv"
}

for variant in "${VARIANT_LIST[@]}"; do
  variant="${variant//[[:space:]]/}"; [[ -n "$variant" ]] || continue
  if [[ "$RUN_SAFE" == "1" ]]; then
    make_report "$variant" safe \
      "$OUT/models/$variant/safe/closed_loop_safe_fast_v48_scalar.json" \
      "$OUT/models/$variant/safe/closed_loop_safe_fast_v48.json" "$SAFE_METHODS"
  fi
  if [[ "$RUN_NEAR" == "1" ]]; then
    make_report "$variant" near \
      "$OUT/models/$variant/near/audit_near_contact_selected_topk_v48_scalar.json" \
      "$OUT/models/$variant/near/audit_near_contact_selected_topk_v48_v48.json" "$NEAR_METHODS"
  fi
  if [[ "$RUN_CONTACT" == "1" ]]; then
    make_report "$variant" contact \
      "$OUT/models/$variant/contact/audit_contact_selected_topk_v48_scalar.json" \
      "$OUT/models/$variant/contact/audit_contact_selected_topk_v48_v48.json" "$CONTACT_METHODS"
  fi
done

run_video_regime() {
  local variant="$1" regime="$2" target="$3" womd="$4"
  local full="$OUT/models/$variant/$regime"
  local model_file control_file
  if [[ "$regime" == "near" ]]; then
    model_file="$full/audit_near_contact_selected_topk_v48_v48.json"
    control_file="$full/audit_near_contact_selected_topk_v48_scalar.json"
  else
    model_file="$full/audit_contact_selected_topk_v48_v48.json"
    control_file="$full/audit_contact_selected_topk_v48_scalar.json"
  fi
  local video_root="$OUT/videos/$variant/$regime"
  local selection="$video_root/critical_scene_selection.json"
  local subset="$video_root/target_subset"
  mkdir -p "$video_root"
  python tools/select_critical_scenes_v48_34.py \
    --method-scenes "$model_file" --control-scenes "$control_file" --regime "$regime" \
    --num-positive "$NUM_POSITIVE_VIDEOS" --num-failure "$NUM_FAILURE_VIDEOS" --output "$selection"
  python tools/subset_dataset_targets_v48_34.py \
    --input "$target" --selection "$selection" --output "$subset" --overwrite
  local n
  n="$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1])).get("selected",[])))' "$selection")"
  if [[ "$n" == "0" ]]; then echo "[WARN] no selected scenes for $variant/$regime"; return 0; fi
  local trace_run="$video_root/trace_run"
  MAX_TARGETS="$n" CL_RESUME=false run_ocrap_regime "$variant" "$regime" "$subset" "$womd" "$trace_run" true
  local trace_model trace_control
  if [[ "$regime" == "near" ]]; then
    trace_model="$trace_run/audit_near_contact_selected_topk_v48_v48.json"
    trace_control="$trace_run/audit_near_contact_selected_topk_v48_scalar.json"
  else
    trace_model="$trace_run/audit_contact_selected_topk_v48_v48.json"
    trace_control="$trace_run/audit_contact_selected_topk_v48_scalar.json"
  fi
  python tools/render_critical_scenes_v48_34.py \
    --method-scenes "$trace_model" --control-scenes "$trace_control" \
    --selection "$selection" --output-dir "$video_root/rendered" --fps "$VIDEO_FPS"
}

if [[ "$RUN_VIDEOS" == "1" ]]; then
  IFS=',' read -r -a VIDEO_LIST <<< "$VIDEO_VARIANTS"
  for variant in "${VIDEO_LIST[@]}"; do
    variant="${variant//[[:space:]]/}"; [[ -n "$variant" ]] || continue
    [[ "$RUN_NEAR" != "1" ]] || run_video_regime "$variant" near "$NEAR_TARGET_DATASET" "$NEAR_WOMD_SOURCE"
    [[ "$RUN_CONTACT" != "1" ]] || run_video_regime "$variant" contact "$CONTACT_TARGET_DATASET" "$CONTACT_WOMD_SOURCE"
  done
fi

python - "$OUT" <<'PY'
import json, pathlib, time, sys
out=pathlib.Path(sys.argv[1])
d={
 'event':'v48_34_exploratory_closed_loop_complete', 'created_unix':time.time(),
 'exploratory_only':True, 'paper_claim_allowed':False, 'deployment_allowed':False,
 'paired_reports':sorted(str(p) for p in (out/'reports').glob('*_paired.json')),
 'video_indices':sorted(str(p) for p in (out/'videos').rglob('VIDEO_INDEX.json')),
}
(out/'EXPLORATORY_CLOSED_LOOP_COMPLETE.json').write_text(json.dumps(d,indent=2)+'\n')
print(d)
PY
