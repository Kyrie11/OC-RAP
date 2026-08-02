#!/usr/bin/env bash
set -euo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
# Eight small jobs run concurrently. Keep host threading/data workers bounded so
# CPU and TFRecord I/O, rather than GPU memory, do not become the bottleneck.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

ROOT="${ABLATION_ROOT:-runs/ocrap_v48_30_slack_rank_bridge_ablations_4830}"
SOURCE_RUN="${SOURCE_RUN:-runs/ocrap_v48_13_terra_proxy_4801}"
OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
PROTOCOL_ROOT="${PROTOCOL_ROOT:-$OCRAP_ROOT/calibration_v48_14_prism_4814}"
CAL_SAFE="${CAL_SAFE:-$OCRAP_ROOT/calibration_safe}"
GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
TASKS_PER_GPU="${TASKS_PER_GPU:-4}"
[[ "$TASKS_PER_GPU" -ge 4 ]] || { echo "v48.30 needs TASKS_PER_GPU>=4 for all-eight concurrency" >&2; exit 30; }
ALPHA="${ALPHA:-0.2}"; BETA="${BETA:-0.2}"; TOP_M="${TOP_M:-8}"
POSITIVE_GAIN="${POSITIVE_GAIN:-0.015}"
DEPLOYABLE_MACRO_IDS="${DEPLOYABLE_MACRO_IDS:-2,3,5,6,7}"
COMPONENT_HARM_DRS_TOLERANCE="${COMPONENT_HARM_DRS_TOLERANCE:-0.05}"
COMPONENT_HARM_DEP_TOLERANCE="${COMPONENT_HARM_DEP_TOLERANCE:-0.05}"
COMPONENT_HARM_GAP_TOLERANCE="${COMPONENT_HARM_GAP_TOLERANCE:-0.05}"
COMPONENT_HARM_HARD_TOLERANCE="${COMPONENT_HARM_HARD_TOLERANCE:-0.05}"
COMPONENT_HARM_PROXY_TOLERANCE="${COMPONENT_HARM_PROXY_TOLERANCE:-0.05}"
mkdir -p "$ROOT/tasks" "$ROOT/logs"

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

fail_controller() {
  local stage="$1" raw_rc="$2" detail="${3:-}"
  python - "$ROOT" "$stage" "$raw_rc" "$detail" <<'PY'
import json,pathlib,sys,time
root=pathlib.Path(sys.argv[1])
doc={'complete':False,'version':'v48.30-SLACK-RANK-BRIDGE','stage':sys.argv[2],
     'raw_exit_code':int(sys.argv[3]),'normalized_exit_code':30,'detail':sys.argv[4],
     'created_unix':time.time(),'test_roots_read':False}
(root/'ABLATIONS_FAILED.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n')
PY
}

set +e
python tools/audit_dedicated_protocol_v48_16.py --protocol-root "$PROTOCOL_ROOT" \
  --output "$ROOT/dedicated_protocol_audit.json" >"$ROOT/logs/dedicated_protocol_audit.log" 2>&1
protocol_rc=$?
set -e
[[ "$protocol_rc" == 0 ]] || { fail_controller protocol_audit "$protocol_rc" "$ROOT/logs/dedicated_protocol_audit.log"; exit 30; }

index_common=(
  --alpha "$ALPHA" --beta "$BETA" --top-m "$TOP_M" --positive-gain "$POSITIVE_GAIN"
  --deployable-macro-ids "$DEPLOYABLE_MACRO_IDS" --quality-mode warn
  --component-harm-drs-tolerance "$COMPONENT_HARM_DRS_TOLERANCE"
  --component-harm-dep-tolerance "$COMPONENT_HARM_DEP_TOLERANCE"
  --component-harm-gap-tolerance "$COMPONENT_HARM_GAP_TOLERANCE"
  --component-harm-hard-tolerance "$COMPONENT_HARM_HARD_TOLERANCE"
  --component-harm-proxy-tolerance "$COMPONENT_HARM_PROXY_TOLERANCE"
)
rm -f "$GROUP_INDEX" "$GROUP_SUMMARY" "$VAL_GROUP_INDEX" "$VAL_GROUP_SUMMARY"
python tools/build_teacher_pcd_index_v48.py --dataset "$TRAIN_NEAR,$TRAIN_CONTACT" \
  --output "$GROUP_INDEX" --summary-output "$GROUP_SUMMARY" "${index_common[@]}" \
  >"$ROOT/logs/build_teacher_index.log" 2>&1 || { fail_controller teacher_index_build $? "$ROOT/logs/build_teacher_index.log"; exit 30; }
python tools/build_teacher_pcd_index_v48.py --dataset "$DEV_NEAR,$DEV_CONTACT" \
  --output "$VAL_GROUP_INDEX" --summary-output "$VAL_GROUP_SUMMARY" "${index_common[@]}" \
  >"$ROOT/logs/build_dev_teacher_index.log" 2>&1 || { fail_controller dev_teacher_index_build $? "$ROOT/logs/build_dev_teacher_index.log"; exit 30; }

run_task() {
  local group="$1" variant="$2" gpu="$3"
  local out="$ROOT/tasks/${group}_${variant}"
  local run="$out/candidates/$variant"
  local source="$SOURCE_RUN/candidates/$variant/model_v48_trac_sr/best.pt"
  [[ -f "$source" ]] || { echo "missing $source" >&2; return 30; }
  rm -rf "$out"; mkdir -p "$out/logs"
  local component_count=5 component_scale=6.0 margin_reg=0.0 hard_negative=0.0 prior_mode=benefit_only
  case "$group" in
    A_natural_population_reference)
      prior_mode=benefit_only; margin_reg=0.0; hard_negative=0.0 ;;
    B_add_signed_component_margin)
      prior_mode=benefit_only; margin_reg=0.50; hard_negative=0.0 ;;
    C_add_safety_slack_projection)
      prior_mode=safety_slack; margin_reg=0.50; hard_negative=0.0 ;;
    D_full_slack_rank)
      prior_mode=safety_slack; margin_reg=0.50; hard_negative=1.0 ;;
    *) echo "unknown ablation $group" >&2; return 30 ;;
  esac

  set +e
  RUN="$run" INIT_CKPT="$source" VARIANT="$variant" TRAIN_GPU="$gpu" \
  TRAIN_MIX="$TRAIN_NEAR,$TRAIN_CONTACT" VAL_MIX="$DEV_NEAR,$DEV_CONTACT" GROUP_INDEX="$GROUP_INDEX" VAL_GROUP_INDEX="$VAL_GROUP_INDEX" \
  TRAIN_OCRAP_ROOT="$OCRAP_ROOT" EVAL_OCRAP_ROOT="$OCRAP_ROOT" \
  NUM_WORKERS="${NUM_WORKERS:-1}" PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}" BATCH_SIZE="${BATCH_SIZE:-72}" \
  FACTOR_EPOCHS="${FACTOR_EPOCHS:-20}" FACTOR_PATIENCE="${FACTOR_PATIENCE:-6}" \
  ADMISSION_EPOCHS="${ADMISSION_EPOCHS:-18}" ADMISSION_PATIENCE="${ADMISSION_PATIENCE:-6}" \
  EVIDENCE_ADAPT_LR="${EVIDENCE_ADAPT_LR:-0.00015}" PROPOSAL_TOP_K=3 \
  EVIDENCE_COMPONENT_COUNT="$component_count" EVIDENCE_COMPONENT_SCALE="$component_scale" \
  EVIDENCE_ADMISSION_PRIOR_MODE="$prior_mode" EVIDENCE_ADMISSION_SCALE=2.0 \
  ADMISSION_SAFE_UTILITY_REGRESSION_WEIGHT=0.75 \
  FACTOR_COMPONENT_MARGIN_REGRESSION_WEIGHT="$margin_reg" \
  ADMISSION_SAFE_UTILITY_LISTWISE_WEIGHT=0 \
  ADMISSION_SAFE_HARD_NEGATIVE_WEIGHT="$hard_negative" \
  ADMISSION_SAFE_HARD_NEGATIVE_MARGIN=0.05 \
  ADMISSION_FRONTIER_PAIRWISE_WEIGHT=0 \
  ADMISSION_SETWISE_WEIGHT=0.10 ADMISSION_BINARY_WEIGHT=0.10 \
  EVIDENCE_SLACK_TEMPERATURE=0.025 EVIDENCE_SLACK_PENALTY=1.0 \
    bash scripts/adapt_ocrap_v48_30_slack_rank_variant.sh >"$out/logs/adapt.log" 2>&1
  local adapt_rc=$?
  set -e
  [[ "$adapt_rc" == 0 ]] || { printf '{"complete":false,"stage":"adaptation","raw_exit_code":%s}\n' "$adapt_rc" >"$out/TASK_FAILED.json"; return 30; }

  set +e
  OUTPUTDIR="$out" CAL_SAFE="$CAL_SAFE" CERT_NEAR="$CERT_NEAR" CERT_CONTACT="$CERT_CONTACT" DEV_NEAR="$DEV_NEAR" DEV_CONTACT="$DEV_CONTACT" \
  GPU0="$gpu" GPU1="$gpu" VARIANTS="$variant" OPPORTUNITY_LABEL_MODE=raw_benefit GATE_POSITIVE_MODE=safe_benefit \
    bash scripts/calibrate_v48_30_certificate_pool.sh >"$out/logs/certificate.log" 2>&1
  local cert_rc=$?
  set -e
  [[ "$cert_rc" == 0 || "$cert_rc" == 20 ]] || { printf '{"complete":false,"stage":"certificate","raw_exit_code":%s}\n' "$cert_rc" >"$out/TASK_FAILED.json"; return 30; }

  python - "$out" "$group" "$variant" "$cert_rc" "$component_count" "$component_scale" "$margin_reg" "$hard_negative" "$prior_mode" <<'PY'
import hashlib,json,pathlib,sys,time
out=pathlib.Path(sys.argv[1]); group=sys.argv[2]; variant=sys.argv[3]; rc=int(sys.argv[4])
component_count=int(sys.argv[5]); component_scale=float(sys.argv[6]); margin_reg=float(sys.argv[7]); hard_negative=float(sys.argv[8]); prior_mode=sys.argv[9]
base=out/'candidates'/variant; ckpt=base/'model_v48_trac_sr'/'best.pt'; cal=base/'calibration'
required=[ckpt,cal/'CERTIFICATE_CALIBRATION_COMPLETE.json',cal/'direct_value_risk_near_v48.json',cal/'direct_value_risk_contact_v48.json',base/'FACTOR_TRANSFER_INTEGRITY.json']
missing=[str(x) for x in required if not x.is_file()]
if missing: raise SystemExit('missing: '+','.join(missing))
regimes={}
for regime in ('near','contact'):
    d=json.load(open(cal/f'direct_value_risk_{regime}_v48.json'))
    regimes[regime]={'valid':bool(d.get('valid_for_deployment',False)),'rejection_kind':d.get('rejection_kind'),
                     'verify':d.get('verify'),'oracle':d.get('proposal_constrained_oracle_gate')}
doc={'complete':True,'version':'v48.30-SLACK-RANK-BRIDGE','group':group,'variant':variant,
     'component_count':component_count,'component_scale':component_scale,'two_stage':True,
     'component_margin_regression_weight':margin_reg,'safe_utility_listwise_weight':0.0,'frontier_pairwise_weight':0.0,
     'safe_hard_negative_weight':hard_negative,'admission_prior_mode':prior_mode,
     'certificate_exit':rc,'gate_passed':rc==0,'regimes':regimes,
     'checkpoint_sha256':hashlib.sha256(ckpt.read_bytes()).hexdigest(),
     'created_unix':time.time(),'test_roots_read':False}
(out/'TASK_COMPLETE.json').write_text(json.dumps(doc,indent=2)+'\n')
PY
}

groups=(A_natural_population_reference B_add_signed_component_margin C_add_safety_slack_projection D_full_slack_rank)
: >"$ROOT/TASK_GPU_ASSIGNMENT.txt"
declare -a pids names
for group in "${groups[@]}"; do
  echo "${group}_balanced:gpu${GPU0}" >>"$ROOT/TASK_GPU_ASSIGNMENT.txt"
  echo "${group}_precision:gpu${GPU1}" >>"$ROOT/TASK_GPU_ASSIGNMENT.txt"
  run_task "$group" balanced "$GPU0" & pids+=("$!"); names+=("${group}_balanced")
  run_task "$group" precision "$GPU1" & pids+=("$!"); names+=("${group}_precision")
done

failures=0
set +e
for i in "${!pids[@]}"; do
  wait "${pids[$i]}"; rc=$?
  printf '%s=%s\n' "${names[$i]}" "$rc" | tee -a "$ROOT/logs/task_wait_status.log"
  [[ "$rc" == 0 ]] || failures=$((failures+1))
done
set -e

python - "$ROOT" "$failures" <<'PY'
import json,pathlib,sys,time
root=pathlib.Path(sys.argv[1]); failures=int(sys.argv[2])
groups=['A_natural_population_reference','B_add_signed_component_margin','C_add_safety_slack_projection','D_full_slack_rank']
expected=[f'{g}_{v}' for g in groups for v in ('balanced','precision')]
missing=[x for x in expected if not (root/'tasks'/x/'TASK_COMPLETE.json').is_file()]
doc={'complete':not missing and failures==0,'version':'v48.30-SLACK-RANK-BRIDGE',
     'max_concurrent_tasks':8,'execution':'all eight tasks launched together; four jobs per 24GB A30',
     'host_contention_controls':{'num_workers_per_task':1,'omp_threads':1,'mkl_threads':1,'openblas_threads':1},
     'expected_tasks':expected,'missing_tasks':missing,'failed_waits':failures,
     'created_unix':time.time(),'test_roots_read':False}
(root/'ABLATIONS_STATUS.json').write_text(json.dumps(doc,indent=2)+'\n')
if doc['complete']:
    (root/'ABLATIONS_COMPLETE.json').write_text(json.dumps(doc,indent=2)+'\n')
else:
    raise SystemExit(30)
PY
