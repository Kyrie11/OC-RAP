#!/usr/bin/env bash
set -euo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"

ROOT="${ABLATION_ROOT:-runs/ocrap_v48_32_identity_utility_bridge_ablations_4832}"
SOURCE_RUN="${SOURCE_RUN:-runs/ocrap_v48_13_terra_proxy_4801}"
OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
PROTOCOL_ROOT="${PROTOCOL_ROOT:-$OCRAP_ROOT/calibration_v48_14_prism_4814}"
CAL_SAFE="${CAL_SAFE:-$OCRAP_ROOT/calibration_safe}"
GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
BASE_FACTOR_CACHE_BALANCED="${V4832_ABLATION_FACTOR_CACHE_BALANCED:-}"
BASE_FACTOR_CACHE_PRECISION="${V4832_ABLATION_FACTOR_CACHE_PRECISION:-}"
ALPHA="${ALPHA:-0.2}"; BETA="${BETA:-0.2}"; TOP_M="${TOP_M:-8}"
POSITIVE_GAIN="${POSITIVE_GAIN:-0.015}"
DEPLOYABLE_MACRO_IDS="${DEPLOYABLE_MACRO_IDS:-2,3,5,6,7}"
COMPONENT_HARM_DRS_TOLERANCE="${COMPONENT_HARM_DRS_TOLERANCE:-0.05}"
COMPONENT_HARM_DEP_TOLERANCE="${COMPONENT_HARM_DEP_TOLERANCE:-0.05}"
COMPONENT_HARM_GAP_TOLERANCE="${COMPONENT_HARM_GAP_TOLERANCE:-0.05}"
COMPONENT_HARM_HARD_TOLERANCE="${COMPONENT_HARM_HARD_TOLERANCE:-0.05}"
COMPONENT_HARM_PROXY_TOLERANCE="${COMPONENT_HARM_PROXY_TOLERANCE:-0.05}"
mkdir -p "$ROOT/tasks" "$ROOT/logs"
rm -f "$ROOT/ABLATIONS_STATUS.json" "$ROOT/ABLATIONS_COMPLETE.json"
: > "$ROOT/logs/task_wait_status.log"

TRAIN_NEAR="$PROTOCOL_ROOT/evidence_adapt_train_near_contact"
TRAIN_CONTACT="$PROTOCOL_ROOT/evidence_adapt_train_contact"
DEV_NEAR="$PROTOCOL_ROOT/evidence_adapt_dev_near_contact"
DEV_CONTACT="$PROTOCOL_ROOT/evidence_adapt_dev_contact"
CERT_NEAR="$PROTOCOL_ROOT/certificate_pool_near_contact"
CERT_CONTACT="$PROTOCOL_ROOT/certificate_pool_contact"
GROUP_INDEX="$ROOT/evidence_adapt_teacher_pcd_index.jsonl"
GROUP_SUMMARY="$ROOT/evidence_adapt_teacher_pcd_index_summary.json"
VAL_GROUP_INDEX="$ROOT/evidence_adapt_dev_teacher_pcd_index.jsonl"
VAL_GROUP_SUMMARY="$ROOT/evidence_adapt_dev_teacher_pcd_index_summary.json"

write_root_failed() {
  local stage="$1" rc="$2" log="${3:-}" detail="${4:-}"
  python - "$ROOT" "$stage" "$rc" "$log" "$detail" <<'PY_ROOT_FAILURE'
import json,pathlib,sys,time
root=pathlib.Path(sys.argv[1]); log=pathlib.Path(sys.argv[4]) if sys.argv[4] else None
tail='\n'.join(log.read_text(errors='replace').splitlines()[-120:]) if log and log.is_file() else ''
doc={'complete':False,'event':'v48_32_ablations_failed','stage':sys.argv[2],
     'raw_exit_code':int(sys.argv[3]),'normalized_exit_code':30,'detail':sys.argv[5],
     'log':str(log) if log else None,'log_tail':tail,'created_unix':time.time(),
     'test_roots_read':False}
(root/'ABLATIONS_FAILED.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n')
(root/'ABLATIONS_STATUS.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n')
PY_ROOT_FAILURE
}

set +e
python tools/audit_dedicated_protocol_v48_16.py --protocol-root "$PROTOCOL_ROOT" \
  --output "$ROOT/dedicated_protocol_audit.json" >"$ROOT/logs/dedicated_protocol_audit.log" 2>&1
protocol_rc=$?
set -e
if [[ "$protocol_rc" != 0 ]]; then
  write_root_failed protocol_audit "$protocol_rc" "$ROOT/logs/dedicated_protocol_audit.log" "dedicated protocol audit failed"
  exit 30
fi

index_common=(
  --alpha "$ALPHA" --beta "$BETA" --top-m "$TOP_M" --positive-gain "$POSITIVE_GAIN"
  --deployable-macro-ids "$DEPLOYABLE_MACRO_IDS" --quality-mode warn
  --component-harm-drs-tolerance "$COMPONENT_HARM_DRS_TOLERANCE"
  --component-harm-dep-tolerance "$COMPONENT_HARM_DEP_TOLERANCE"
  --component-harm-gap-tolerance "$COMPONENT_HARM_GAP_TOLERANCE"
  --component-harm-hard-tolerance "$COMPONENT_HARM_HARD_TOLERANCE"
  --component-harm-proxy-tolerance "$COMPONENT_HARM_PROXY_TOLERANCE"
)

ensure_index() {
  local label="$1" dataset="$2" index="$3" summary="$4"
  local contract="$ROOT/${label}_INDEX_CONTRACT.json"
  local check_log="$ROOT/logs/check_${label}_index_contract.log"
  local build_log="$ROOT/logs/build_${label}_index.log"
  local rebuild="${REBUILD_ABLATION_INDEX:-0}"
  if [[ "$rebuild" != 1 && -s "$index" && -s "$summary" ]]; then
    set +e
    python tools/check_v48_19_target_support.py \
      --summary "$summary" --expected-dataset "$dataset" \
      --alpha "$ALPHA" --beta "$BETA" --top-m "$TOP_M" --positive-gain "$POSITIVE_GAIN" \
      --deployable-macro-ids "$DEPLOYABLE_MACRO_IDS" \
      --component-harm-drs-tolerance "$COMPONENT_HARM_DRS_TOLERANCE" \
      --component-harm-dep-tolerance "$COMPONENT_HARM_DEP_TOLERANCE" \
      --component-harm-gap-tolerance "$COMPONENT_HARM_GAP_TOLERANCE" \
      --component-harm-hard-tolerance "$COMPONENT_HARM_HARD_TOLERANCE" \
      --component-harm-proxy-tolerance "$COMPONENT_HARM_PROXY_TOLERANCE" \
      --mode contract --output "$contract" >"$check_log" 2>&1
    local check_rc=$?
    set -e
    [[ "$check_rc" == 0 ]] || rebuild=1
  else
    rebuild=1
  fi
  if [[ "$rebuild" == 1 ]]; then
    rm -f "$index" "$summary"
    set +e
    python tools/build_teacher_pcd_index_v48.py --dataset "$dataset" \
      --output "$index" --summary-output "$summary" "${index_common[@]}" >"$build_log" 2>&1
    local build_rc=$?
    set -e
    if [[ "$build_rc" != 0 ]]; then
      write_root_failed "${label}_index_build" "$build_rc" "$build_log" "teacher index build failed"
      return 30
    fi
  fi
  set +e
  python tools/check_v48_19_target_support.py \
    --summary "$summary" --expected-dataset "$dataset" \
    --alpha "$ALPHA" --beta "$BETA" --top-m "$TOP_M" --positive-gain "$POSITIVE_GAIN" \
    --deployable-macro-ids "$DEPLOYABLE_MACRO_IDS" \
    --component-harm-drs-tolerance "$COMPONENT_HARM_DRS_TOLERANCE" \
    --component-harm-dep-tolerance "$COMPONENT_HARM_DEP_TOLERANCE" \
    --component-harm-gap-tolerance "$COMPONENT_HARM_GAP_TOLERANCE" \
    --component-harm-hard-tolerance "$COMPONENT_HARM_HARD_TOLERANCE" \
    --component-harm-proxy-tolerance "$COMPONENT_HARM_PROXY_TOLERANCE" \
    --mode contract --output "$contract" >"$check_log" 2>&1
  local final_rc=$?
  set -e
  if [[ "$final_rc" != 0 ]]; then
    write_root_failed "${label}_index_contract" "$final_rc" "$check_log" "$contract"
    return 30
  fi
}

ensure_index train "$TRAIN_NEAR,$TRAIN_CONTACT" "$GROUP_INDEX" "$GROUP_SUMMARY" || exit 30
ensure_index validation "$DEV_NEAR,$DEV_CONTACT" "$VAL_GROUP_INDEX" "$VAL_GROUP_SUMMARY" || exit 30
set +e
python tools/check_v48_19_target_support.py \
  --summary "$GROUP_SUMMARY" --expected-dataset "$TRAIN_NEAR,$TRAIN_CONTACT" \
  --alpha "$ALPHA" --beta "$BETA" --top-m "$TOP_M" --positive-gain "$POSITIVE_GAIN" \
  --deployable-macro-ids "$DEPLOYABLE_MACRO_IDS" \
  --component-harm-drs-tolerance "$COMPONENT_HARM_DRS_TOLERANCE" \
  --component-harm-dep-tolerance "$COMPONENT_HARM_DEP_TOLERANCE" \
  --component-harm-gap-tolerance "$COMPONENT_HARM_GAP_TOLERANCE" \
  --component-harm-hard-tolerance "$COMPONENT_HARM_HARD_TOLERANCE" \
  --component-harm-proxy-tolerance "$COMPONENT_HARM_PROXY_TOLERANCE" \
  --mode all --output "$ROOT/SUPPORT_TARGET_SUPPORT.json" >"$ROOT/logs/check_support_target_support.log" 2>&1
support_rc=$?
set -e
if [[ "$support_rc" != 0 ]]; then
  write_root_failed target_support "$support_rc" "$ROOT/logs/check_support_target_support.log" "$ROOT/SUPPORT_TARGET_SUPPORT.json"
  exit 30
fi

write_task_failed() {
  local out="$1" stage="$2" rc="$3" log="${4:-}" detail="${5:-}"
  python - "$out" "$stage" "$rc" "$log" "$detail" <<'PY'
import json,pathlib,sys,time
out=pathlib.Path(sys.argv[1]); out.mkdir(parents=True,exist_ok=True)
log=pathlib.Path(sys.argv[4]) if sys.argv[4] else None
tail=''
if log and log.is_file(): tail='\n'.join(log.read_text(errors='replace').splitlines()[-120:])
doc={'complete':False,'event':'v48_32_ablation_task_failed','stage':sys.argv[2],
     'raw_exit_code':int(sys.argv[3]),'normalized_exit_code':30,
     'log':str(log) if log else None,'detail':sys.argv[5],'log_tail':tail,
     'created_unix':time.time(),'test_roots_read':False}
(out/'TASK_FAILED.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n')
PY
}

run_task() {
  local group="$1" variant="$2" gpu="$3" identity_all="$4" coupled="$5" adaptive="$6" cache="$7"
  local out="$ROOT/tasks/${group}_${variant}"
  local run="$out/candidates/$variant"
  local source="$SOURCE_RUN/candidates/$variant/model_v48_trac_sr/best.pt"
  [[ -f "$source" ]] || { write_task_failed "$out" source 30 "" "missing source checkpoint $source"; return 30; }
  if [[ -n "$cache" && ! -f "$cache/model_v48_trac_sr/best.pt" ]]; then
    write_task_failed "$out" factor_cache 30 "" "missing factor cache $cache/model_v48_trac_sr/best.pt"
    return 30
  fi
  rm -rf "$out"; mkdir -p "$out/logs"
  local factor_cache_env=""
  [[ -n "$cache" ]] && factor_cache_env="$cache"
  set +e
  RUN="$run" INIT_CKPT="$source" VARIANT="$variant" TRAIN_GPU="$gpu" \
  TRAIN_MIX="$TRAIN_NEAR,$TRAIN_CONTACT" VAL_MIX="$DEV_NEAR,$DEV_CONTACT" \
  GROUP_INDEX="$GROUP_INDEX" VAL_GROUP_INDEX="$VAL_GROUP_INDEX" \
  TRAIN_OCRAP_ROOT="$OCRAP_ROOT" EVAL_OCRAP_ROOT="$OCRAP_ROOT" \
  NUM_WORKERS="${NUM_WORKERS:-3}" PREFETCH_FACTOR="${PREFETCH_FACTOR:-3}" BATCH_SIZE="${BATCH_SIZE:-96}" \
  FACTOR_EPOCHS="${FACTOR_EPOCHS:-20}" FACTOR_PATIENCE="${FACTOR_PATIENCE:-6}" \
  IDENTITY_EPOCHS="${IDENTITY_EPOCHS:-18}" IDENTITY_PATIENCE="${IDENTITY_PATIENCE:-6}" \
  FINAL_EPOCHS="${FINAL_EPOCHS:-10}" FINAL_PATIENCE="${FINAL_PATIENCE:-4}" \
  IDENTITY_LR="${IDENTITY_LR:-0.00010}" FINAL_LR="${FINAL_LR:-0.00003}" \
  V4832_ENABLE_SUPPORT_RELIABILITY=1 V4832_IDENTITY_TRAIN_ALL="$identity_all" \
  V4832_COUPLE_ADMISSION_PRIOR="$coupled" V4832_ADAPTIVE_IDENTITY_MARGIN="$adaptive" \
  V4832_ENABLE_FINAL_CALIBRATION=1 V4832_FACTOR_CACHE_RUN="$factor_cache_env" \
  PROPOSAL_TOP_K=3 EVIDENCE_COMPONENT_COUNT=5 EVIDENCE_COMPONENT_SCALE="${EVIDENCE_COMPONENT_SCALE:-6.0}" \
  EVIDENCE_ADMISSION_PRIOR_MODE=safety_slack EVIDENCE_ADMISSION_SCALE=2.0 \
  EVIDENCE_SLACK_TEMPERATURE=0.025 EVIDENCE_SLACK_PENALTY=1.0 \
  COMPONENT_HARM_DRS_TOLERANCE="$COMPONENT_HARM_DRS_TOLERANCE" \
  COMPONENT_HARM_DEP_TOLERANCE="$COMPONENT_HARM_DEP_TOLERANCE" \
  COMPONENT_HARM_GAP_TOLERANCE="$COMPONENT_HARM_GAP_TOLERANCE" \
  COMPONENT_HARM_HARD_TOLERANCE="$COMPONENT_HARM_HARD_TOLERANCE" \
  COMPONENT_HARM_PROXY_TOLERANCE="$COMPONENT_HARM_PROXY_TOLERANCE" \
    bash scripts/adapt_ocrap_v48_32_identity_utility_variant.sh >"$out/logs/adapt.log" 2>&1
  local adapt_rc=$?
  set -e
  if [[ "$adapt_rc" != 0 ]]; then
    write_task_failed "$out" adaptation "$adapt_rc" "$out/logs/adapt.log" "identity utility training failed"
    return 30
  fi

  set +e
  python tools/check_v48_32_training_contract.py --run "$run" \
    --output "$run/TRAINING_CONTRACT.json" \
    --expect-identity-all "$([[ "$identity_all" == 1 ]] && echo true || echo false)" \
    --expect-prior-coupled "$([[ "$coupled" == 1 ]] && echo true || echo false)" \
    --expect-adaptive-margin "$([[ "$adaptive" == 1 ]] && echo true || echo false)" \
    --expect-final-enabled true >"$out/logs/training_contract.log" 2>&1
  local train_contract_rc=$?
  set -e
  if [[ "$train_contract_rc" != 0 ]]; then
    write_task_failed "$out" training_contract "$train_contract_rc" "$out/logs/training_contract.log" "$run/TRAINING_CONTRACT.json"
    return 30
  fi

  set +e
  python tools/check_v48_32_model_contract.py \
    --checkpoint "$run/model_v48_trac_sr/best.pt" --support-contract "$run/FACTOR_SUPPORT_CONTRACT.json" \
    --output "$run/MODEL_INFERENCE_CONTRACT.json" --expect-frontier true --expect-admission-bounded true \
    --expect-component-count 5 --expect-component-scale "${EVIDENCE_COMPONENT_SCALE:-6.0}" \
    --expect-admission-prior-detach any --expect-admission-prior-mode safety_slack \
    --expect-slack-temperature 0.025 --expect-slack-penalty 1.0 \
    >"$out/logs/model_contract.log" 2>&1
  local model_contract_rc=$?
  set -e
  if [[ "$model_contract_rc" != 0 ]]; then
    write_task_failed "$out" model_contract "$model_contract_rc" "$out/logs/model_contract.log" "$run/MODEL_INFERENCE_CONTRACT.json"
    return 30
  fi

  set +e
  OUTPUTDIR="$out" CAL_SAFE="$CAL_SAFE" CERT_NEAR="$CERT_NEAR" CERT_CONTACT="$CERT_CONTACT" \
  DEV_NEAR="$DEV_NEAR" DEV_CONTACT="$DEV_CONTACT" GPU0="$gpu" GPU1="$gpu" VARIANTS="$variant" \
  OPPORTUNITY_LABEL_MODE=raw_benefit GATE_POSITIVE_MODE=safe_benefit \
    bash scripts/calibrate_v48_32_certificate_pool.sh >"$out/logs/certificate.log" 2>&1
  local cert_rc=$?
  set -e
  if [[ "$cert_rc" != 0 && "$cert_rc" != 20 ]]; then
    write_task_failed "$out" certificate "$cert_rc" "$out/logs/certificate.log" "certificate artifact/protocol failure"
    return 30
  fi

  python - "$out" "$group" "$variant" "$cert_rc" "$identity_all" "$coupled" "$adaptive" "$cache" <<'PY'
import hashlib,json,pathlib,sys,time
out=pathlib.Path(sys.argv[1]); group=sys.argv[2]; variant=sys.argv[3]; rc=int(sys.argv[4])
identity_all=bool(int(sys.argv[5])); coupled=bool(int(sys.argv[6])); adaptive=bool(int(sys.argv[7])); cache=sys.argv[8]
base=out/'candidates'/variant; ckpt=base/'model_v48_trac_sr'/'best.pt'; cal=base/'calibration'
regimes={}
for regime in ('near','contact'):
    d=json.load(open(cal/f'direct_value_risk_{regime}_v48.json'))
    regimes[regime]={
      'valid':bool(d.get('valid_for_deployment',False)), 'rejection_kind':d.get('rejection_kind'),
      'candidate_positive_auc':d.get('candidate_positive_auc'),
      'candidate_safe_positive_auc':d.get('candidate_safe_positive_auc'),
      'proposal_evidence_top1_correlation':d.get('proposal_evidence_top1_correlation'),
      'proposal_evidence_top1_safe_positive_auc':d.get('proposal_evidence_top1_safe_positive_auc'),
      'proposal_evidence_top1_harm_auc':d.get('proposal_evidence_top1_harm_auc'),
      'verify':d.get('verify'), 'oracle':d.get('proposal_constrained_oracle_gate')}
doc={'complete':True,'version':'v48.32-IDENTITY-UTILITY-BRIDGE','group':group,'variant':variant,
     'support_reliability_enabled':True,'identity_all_heads':identity_all,
     'safe_utility_gradient_coupled':coupled,'adaptive_teacher_gap_margin':adaptive,
     'final_admission_calibration':True,'factor_cache_source':cache or None,
     'certificate_exit':rc,'gate_passed':rc==0,'regimes':regimes,
     'checkpoint_sha256':hashlib.sha256(ckpt.read_bytes()).hexdigest(),
     'created_unix':time.time(),'test_roots_read':False}
(out/'TASK_COMPLETE.json').write_text(json.dumps(doc,indent=2)+'\n')
(out/'TASK_FAILED.json').unlink(missing_ok=True)
PY
  return 0
}

groups=(
  A_admission_only_detached_fixed_margin
  B_joint_identity_detached_fixed_margin
  C_joint_identity_coupled_fixed_margin
  D_full_identity_utility_bridge
)
identity_flags=(0 1 1 1)
coupled_flags=(0 0 1 1)
adaptive_flags=(0 0 0 1)
: >"$ROOT/TASK_GPU_ASSIGNMENT.txt"
failures=0
# Stage 1 is identical across all four groups. Wave A builds one factor stage per
# variant; waves B-D copy those audited artifacts instead of retraining them.
for i in "${!groups[@]}"; do
  group="${groups[$i]}"; identity_all="${identity_flags[$i]}"; coupled="${coupled_flags[$i]}"; adaptive="${adaptive_flags[$i]}"
  cache_bal="$BASE_FACTOR_CACHE_BALANCED"; cache_pre="$BASE_FACTOR_CACHE_PRECISION"
  if [[ "$i" != 0 ]]; then
    cache_bal="$ROOT/tasks/${groups[0]}_balanced/candidates/balanced/factor_stage"
    cache_pre="$ROOT/tasks/${groups[0]}_precision/candidates/precision/factor_stage"
  fi
  echo "${group}_balanced:gpu${GPU0}:factor_cache=${cache_bal:-none}" >>"$ROOT/TASK_GPU_ASSIGNMENT.txt"
  echo "${group}_precision:gpu${GPU1}:factor_cache=${cache_pre:-none}" >>"$ROOT/TASK_GPU_ASSIGNMENT.txt"
  run_task "$group" balanced "$GPU0" "$identity_all" "$coupled" "$adaptive" "$cache_bal" & p0=$!
  run_task "$group" precision "$GPU1" "$identity_all" "$coupled" "$adaptive" "$cache_pre" & p1=$!
  set +e
  wait "$p0"; r0=$?
  wait "$p1"; r1=$?
  set -e
  printf '%s_balanced=%s %s_precision=%s\n' "$group" "$r0" "$group" "$r1" | tee -a "$ROOT/logs/task_wait_status.log"
  [[ "$r0" == 0 ]] || failures=$((failures+1))
  [[ "$r1" == 0 ]] || failures=$((failures+1))
done

python - "$ROOT" "$failures" <<'PY'
import json,pathlib,sys,time
root=pathlib.Path(sys.argv[1]); failures=int(sys.argv[2])
groups=['A_admission_only_detached_fixed_margin','B_joint_identity_detached_fixed_margin',
        'C_joint_identity_coupled_fixed_margin','D_full_identity_utility_bridge']
expected=[f'{g}_{v}' for g in groups for v in ('balanced','precision')]
missing=[x for x in expected if not (root/'tasks'/x/'TASK_COMPLETE.json').is_file()]
failed={x:json.load(open(root/'tasks'/x/'TASK_FAILED.json')) for x in expected if (root/'tasks'/x/'TASK_FAILED.json').is_file()}
doc={'complete':not missing and failures==0,'version':'v48.32-IDENTITY-UTILITY-BRIDGE',
     'execution':'four waves; one Balanced task on GPU0 and one Precision task on GPU1 per wave',
     'factor_stage_cache':'A factor stage reused by B/C/D; A may reuse exact-contract main-run factor stages',
     'max_concurrent_tasks':2,'expected_tasks':expected,'missing_tasks':missing,
     'failed_waits':failures,'failures':failed,'created_unix':time.time(),'test_roots_read':False}
(root/'ABLATIONS_STATUS.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n')
if doc['complete']:
    (root/'ABLATIONS_COMPLETE.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n')
else:
    raise SystemExit(30)
PY
