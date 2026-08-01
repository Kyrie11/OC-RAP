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

OUTPUTDIR="${OUTPUTDIR:-runs/ocrap_v48_28_provenance_margin_dedicated_4828}"
OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
SOURCE_RUN="${SOURCE_RUN:-runs/ocrap_v48_13_terra_proxy_4801}"
PROTOCOL_ROOT="${PROTOCOL_ROOT:-$OCRAP_ROOT/calibration_v48_14_prism_4814}"
CAL_SAFE="${CAL_SAFE:-$OCRAP_ROOT/calibration_safe}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
ALLOW_PARTIAL_VARIANTS="${ALLOW_PARTIAL_VARIANTS:-0}"
PROPOSAL_TOP_K="${PROPOSAL_TOP_K:-3}"

ALPHA="${ALPHA:-0.2}"
BETA="${BETA:-0.2}"
TOP_M="${TOP_M:-8}"
POSITIVE_GAIN="${POSITIVE_GAIN:-0.015}"
DEPLOYABLE_MACRO_IDS="${DEPLOYABLE_MACRO_IDS:-2,3,5,6,7}"
COMPONENT_HARM_DRS_TOLERANCE="${COMPONENT_HARM_DRS_TOLERANCE:-0.05}"
COMPONENT_HARM_DEP_TOLERANCE="${COMPONENT_HARM_DEP_TOLERANCE:-0.05}"
COMPONENT_HARM_GAP_TOLERANCE="${COMPONENT_HARM_GAP_TOLERANCE:-0.05}"
COMPONENT_HARM_HARD_TOLERANCE="${COMPONENT_HARM_HARD_TOLERANCE:-0.05}"
COMPONENT_HARM_PROXY_TOLERANCE="${COMPONENT_HARM_PROXY_TOLERANCE:-0.05}"

mkdir -p "$OUTPUTDIR/logs"
TRAIN_NEAR="$PROTOCOL_ROOT/evidence_adapt_train_near_contact"
TRAIN_CONTACT="$PROTOCOL_ROOT/evidence_adapt_train_contact"
DEV_NEAR="$PROTOCOL_ROOT/evidence_adapt_dev_near_contact"
DEV_CONTACT="$PROTOCOL_ROOT/evidence_adapt_dev_contact"
CERT_NEAR="$PROTOCOL_ROOT/certificate_pool_near_contact"
CERT_CONTACT="$PROTOCOL_ROOT/certificate_pool_contact"
GROUP_INDEX="$OUTPUTDIR/evidence_adapt_teacher_pcd_index.jsonl"
GROUP_SUMMARY="$OUTPUTDIR/evidence_adapt_teacher_pcd_index_summary.json"
VAL_GROUP_INDEX="$OUTPUTDIR/evidence_adapt_dev_teacher_pcd_index.jsonl"
VAL_GROUP_SUMMARY="$OUTPUTDIR/evidence_adapt_dev_teacher_pcd_index_summary.json"

write_pipeline_failure() {
  local stage="$1" raw_rc="$2" detail="${3:-}" balanced_rc="${4:-}" precision_rc="${5:-}"
  python - "$OUTPUTDIR" "$PROTOCOL_ROOT" "$SOURCE_RUN" "$stage" "$raw_rc" "$detail" "$balanced_rc" "$precision_rc" <<'PY'
import hashlib,json,pathlib,sys,time
root=pathlib.Path(sys.argv[1]); protocol=pathlib.Path(sys.argv[2]); source=pathlib.Path(sys.argv[3])
stage=sys.argv[4]; raw_rc=int(sys.argv[5]); detail=sys.argv[6]
def maybe_int(x):
    try: return int(x)
    except Exception: return None
balanced_rc=maybe_int(sys.argv[7]); precision_rc=maybe_int(sys.argv[8])
variants={}
for name in ('balanced','precision'):
    p=root/'candidates'/name/'model_v48_trac_sr'/'best.pt'
    if p.is_file():
        variants[name]={'checkpoint':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
failed={'event':'v48_28_pipeline_failed','created_unix':time.time(),'stage':stage,
        'raw_exit_code':raw_rc,'normalized_exit_code':30,'detail':detail,
        'adaptation_exit_codes':{'balanced':balanced_rc,'precision':precision_rc},
        'pipeline_valid':False,'test_roots_read':False}
(root/'PIPELINE_FAILED.json').write_text(json.dumps(failed,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
doc={'event':'v48_28_provenance_margin_bridge_controller_complete','created_unix':time.time(),
     'source_run':str(source),'protocol_root':str(protocol),'variants':variants,
     'raw_certificate_exit_code':raw_rc if stage=='certificate' else None,
     'certificate_exit_code':30,'gate_evaluated':False,'gate_passed':False,
     'pipeline_valid':False,'failure_stage':stage,'test_roots_read':False}
(root/'V48_27_COMPLETE.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
PY
}

set +e
python tools/audit_dedicated_protocol_v48_16.py \
  --protocol-root "$PROTOCOL_ROOT" \
  --output "$OUTPUTDIR/dedicated_protocol_audit.json" \
  >"$OUTPUTDIR/logs/dedicated_protocol_audit.log" 2>&1
protocol_rc=$?
set -e
if [[ "$protocol_rc" != 0 ]]; then
  write_pipeline_failure protocol_audit "$protocol_rc" "$OUTPUTDIR/logs/dedicated_protocol_audit.log"
  exit 30
fi

index_contract_args=(
  --summary "$GROUP_SUMMARY"
  --expected-dataset "$TRAIN_NEAR,$TRAIN_CONTACT"
  --alpha "$ALPHA" --beta "$BETA" --top-m "$TOP_M"
  --positive-gain "$POSITIVE_GAIN"
  --deployable-macro-ids "$DEPLOYABLE_MACRO_IDS"
  --component-harm-drs-tolerance "$COMPONENT_HARM_DRS_TOLERANCE"
  --component-harm-dep-tolerance "$COMPONENT_HARM_DEP_TOLERANCE"
  --component-harm-gap-tolerance "$COMPONENT_HARM_GAP_TOLERANCE"
  --component-harm-hard-tolerance "$COMPONENT_HARM_HARD_TOLERANCE"
  --component-harm-proxy-tolerance "$COMPONENT_HARM_PROXY_TOLERANCE"
)

rebuild_index="${REBUILD_ADAPT_INDEX:-0}"
if [[ "$rebuild_index" != 1 && -f "$GROUP_INDEX" && -f "$GROUP_SUMMARY" ]]; then
  set +e
  python tools/check_v48_19_target_support.py "${index_contract_args[@]}" \
    --mode contract --output "$OUTPUTDIR/SUPPORT_INDEX_CONTRACT.json" \
    >"$OUTPUTDIR/logs/check_teacher_index_contract.log" 2>&1
  contract_rc=$?
  set -e
  if [[ "$contract_rc" != 0 ]]; then
    echo "teacher-index contract changed; rebuilding the index" | tee -a "$OUTPUTDIR/logs/check_teacher_index_contract.log"
    rebuild_index=1
  fi
else
  rebuild_index=1
fi

if [[ "$rebuild_index" == 1 ]]; then
  rm -f "$GROUP_INDEX" "$GROUP_SUMMARY"
  set +e
  python tools/build_teacher_pcd_index_v48.py \
    --dataset "$TRAIN_NEAR,$TRAIN_CONTACT" \
    --output "$GROUP_INDEX" --summary-output "$GROUP_SUMMARY" \
    --alpha "$ALPHA" --beta "$BETA" --top-m "$TOP_M" \
    --positive-gain "$POSITIVE_GAIN" --deployable-macro-ids "$DEPLOYABLE_MACRO_IDS" \
    --quality-mode warn \
    --component-harm-drs-tolerance "$COMPONENT_HARM_DRS_TOLERANCE" \
    --component-harm-dep-tolerance "$COMPONENT_HARM_DEP_TOLERANCE" \
    --component-harm-gap-tolerance "$COMPONENT_HARM_GAP_TOLERANCE" \
    --component-harm-hard-tolerance "$COMPONENT_HARM_HARD_TOLERANCE" \
    --component-harm-proxy-tolerance "$COMPONENT_HARM_PROXY_TOLERANCE" \
    >"$OUTPUTDIR/logs/build_adapt_teacher_index.log" 2>&1
  index_rc=$?
  set -e
  if [[ "$index_rc" != 0 ]]; then
    write_pipeline_failure teacher_index_build "$index_rc" "$OUTPUTDIR/logs/build_adapt_teacher_index.log"
    exit 30
  fi
fi

# Re-audit after a rebuild so SUPPORT_INDEX_CONTRACT.json always describes the
# index that will actually be used, never the stale index that triggered it.
set +e
python tools/check_v48_19_target_support.py "${index_contract_args[@]}" \
  --mode contract --output "$OUTPUTDIR/SUPPORT_INDEX_CONTRACT.json" \
  >"$OUTPUTDIR/logs/check_teacher_index_contract.log" 2>&1
contract_rc=$?
set -e
if [[ "$contract_rc" != 0 ]]; then
  write_pipeline_failure teacher_index_contract "$contract_rc" "$OUTPUTDIR/SUPPORT_INDEX_CONTRACT.json"
  exit 30
fi

set +e
python tools/check_v48_19_target_support.py "${index_contract_args[@]}" \
  --mode all --output "$OUTPUTDIR/SUPPORT_TARGET_SUPPORT.json" \
  >"$OUTPUTDIR/logs/check_support_target_support.log" 2>&1
target_rc=$?
set -e
if [[ "$target_rc" != 0 ]]; then
  write_pipeline_failure target_support "$target_rc" "$OUTPUTDIR/SUPPORT_TARGET_SUPPORT.json"
  exit 30
fi

# Validation batching must use labels computed from adaptation-dev itself.
# v48.24 accidentally reused the training index for validation, so dev strata
# were reported as dead/mixed and checkpoint selection was not trustworthy.
rm -f "$VAL_GROUP_INDEX" "$VAL_GROUP_SUMMARY"
set +e
python tools/build_teacher_pcd_index_v48.py \
  --dataset "$DEV_NEAR,$DEV_CONTACT" \
  --output "$VAL_GROUP_INDEX" --summary-output "$VAL_GROUP_SUMMARY" \
  --alpha "$ALPHA" --beta "$BETA" --top-m "$TOP_M" \
  --positive-gain "$POSITIVE_GAIN" --deployable-macro-ids "$DEPLOYABLE_MACRO_IDS" \
  --quality-mode warn \
  --component-harm-drs-tolerance "$COMPONENT_HARM_DRS_TOLERANCE" \
  --component-harm-dep-tolerance "$COMPONENT_HARM_DEP_TOLERANCE" \
  --component-harm-gap-tolerance "$COMPONENT_HARM_GAP_TOLERANCE" \
  --component-harm-hard-tolerance "$COMPONENT_HARM_HARD_TOLERANCE" \
  --component-harm-proxy-tolerance "$COMPONENT_HARM_PROXY_TOLERANCE" \
  >"$OUTPUTDIR/logs/build_adapt_dev_teacher_index.log" 2>&1
val_index_rc=$?
set -e
if [[ "$val_index_rc" != 0 ]]; then
  write_pipeline_failure validation_teacher_index_build "$val_index_rc" "$OUTPUTDIR/logs/build_adapt_dev_teacher_index.log"
  exit 30
fi

run_variant() {
  local variant="$1" gpu="$2"
  local source="$SOURCE_RUN/candidates/$variant/model_v48_trac_sr/best.pt"
  local run="$OUTPUTDIR/candidates/$variant"
  [[ -f "$source" ]] || { echo "missing source checkpoint $source" >&2; return 30; }
  set +e
  RUN="$run" INIT_CKPT="$source" VARIANT="$variant" TRAIN_GPU="$gpu" \
  TRAIN_MIX="$TRAIN_NEAR,$TRAIN_CONTACT" VAL_MIX="$DEV_NEAR,$DEV_CONTACT" GROUP_INDEX="$GROUP_INDEX" VAL_GROUP_INDEX="$VAL_GROUP_INDEX" \
  TRAIN_OCRAP_ROOT="$OCRAP_ROOT" EVAL_OCRAP_ROOT="$OCRAP_ROOT" \
  NUM_WORKERS="${NUM_WORKERS:-3}" PREFETCH_FACTOR="${PREFETCH_FACTOR:-3}" BATCH_SIZE="${BATCH_SIZE:-96}" \
  EVIDENCE_ADAPT_EPOCHS="${EVIDENCE_ADAPT_EPOCHS:-28}" EVIDENCE_ADAPT_PATIENCE="${EVIDENCE_ADAPT_PATIENCE:-8}" \
  EVIDENCE_ADAPT_LR="${EVIDENCE_ADAPT_LR:-0.00015}" \
  PROPOSAL_TOP_K="$PROPOSAL_TOP_K" \
    EVIDENCE_COMPONENT_SCALE="${EVIDENCE_COMPONENT_SCALE:-6.0}" bash scripts/adapt_ocrap_v48_28_provenance_margin_variant.sh >"$OUTPUTDIR/logs/adapt_${variant}.log" 2>&1
  local rc=$?
  set -e
  if [[ "$rc" != 0 ]]; then
    python - "$OUTPUTDIR" "$variant" "$rc" "$OUTPUTDIR/logs/adapt_${variant}.log" <<'PY'
import json,pathlib,sys,time
root=pathlib.Path(sys.argv[1]); variant=sys.argv[2]; rc=int(sys.argv[3]); log=pathlib.Path(sys.argv[4])
tail='\n'.join(log.read_text(errors='replace').splitlines()[-100:]) if log.exists() else ''
(root/f'ADAPTATION_FAILED_{variant}.json').write_text(json.dumps({
    'event':'adaptation_failed','variant':variant,'exit_code':rc,'log':str(log),
    'log_tail':tail,'created_unix':time.time()},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
PY
  fi
  return "$rc"
}

run_variant balanced "$GPU0" & p0=$!
run_variant precision "$GPU1" & p1=$!
set +e
wait "$p0"; s0=$?
wait "$p1"; s1=$?
set -e
printf 'balanced=%s precision=%s\n' "$s0" "$s1" | tee "$OUTPUTDIR/logs/adaptation_status.log"

if [[ "$s0" != 0 && "$s1" != 0 ]]; then
  write_pipeline_failure adaptation 30 "both variants failed" "$s0" "$s1"
  python tools/check_v48_16_learning_gates.py --run "$OUTPUTDIR" --output "$OUTPUTDIR/learning_gates_v48_28.json" --version v48.28-PROVENANCE-MARGIN-BRIDGE || true
  exit 30
fi
if [[ "$ALLOW_PARTIAL_VARIANTS" != 1 && ( "$s0" != 0 || "$s1" != 0 ) ]]; then
  write_pipeline_failure adaptation 30 "one variant failed; set ALLOW_PARTIAL_VARIANTS=1 only for explicit debugging" "$s0" "$s1"
  python tools/check_v48_16_learning_gates.py --run "$OUTPUTDIR" --output "$OUTPUTDIR/learning_gates_v48_28.json" --version v48.28-PROVENANCE-MARGIN-BRIDGE || true
  exit 30
fi

# v48.28 fail-closed preflight: training and downstream inference must
# construct the same frontier/prior/admission model.
for variant in balanced precision; do
  [[ "$variant" == balanced && "$s0" != 0 ]] && continue
  [[ "$variant" == precision && "$s1" != 0 ]] && continue
  ckpt="$OUTPUTDIR/candidates/$variant/model_v48_trac_sr/best.pt"
  set +e
  python tools/check_v48_28_model_contract.py \
    --checkpoint "$ckpt" \
    --output "$OUTPUTDIR/candidates/$variant/MODEL_INFERENCE_CONTRACT.json" \
    --expect-frontier true --expect-admission-bounded true \
    --expect-component-prior-logit -2.0 --expect-component-count 5 --expect-component-scale "${EVIDENCE_COMPONENT_SCALE:-6.0}" \
    >"$OUTPUTDIR/logs/model_contract_${variant}.log" 2>&1
  contract_model_rc=$?
  set -e
  if [[ "$contract_model_rc" != 0 ]]; then
    write_pipeline_failure model_inference_contract "$contract_model_rc" "$OUTPUTDIR/candidates/$variant/MODEL_INFERENCE_CONTRACT.json" "$s0" "$s1"
    exit 30
  fi
done

variants=""
[[ "$s0" == 0 ]] && variants="balanced"
[[ "$s1" == 0 ]] && variants="${variants:+$variants,}precision"
set +e
OUTPUTDIR="$OUTPUTDIR" CAL_SAFE="$CAL_SAFE" CERT_NEAR="$CERT_NEAR" CERT_CONTACT="$CERT_CONTACT" DEV_NEAR="$DEV_NEAR" DEV_CONTACT="$DEV_CONTACT" \
GPU0="$GPU0" GPU1="$GPU1" VARIANTS="$variants" \
  OPPORTUNITY_LABEL_MODE=raw_benefit GATE_POSITIVE_MODE=safe_benefit bash scripts/calibrate_v48_28_certificate_pool.sh >"$OUTPUTDIR/logs/certificate_controller.log" 2>&1
raw_cert_rc=$?
set -e
case "$raw_cert_rc" in
  0|20) cert_rc="$raw_cert_rc" ;;
  *) cert_rc=30 ;;
esac

python tools/check_v48_16_learning_gates.py --run "$OUTPUTDIR" --output "$OUTPUTDIR/learning_gates_v48_28.json" --version v48.28-PROVENANCE-MARGIN-BRIDGE || true
python tools/summarize_v48_28_gate_failure.py --run "$OUTPUTDIR" --output "$OUTPUTDIR/GATE_FAILURE_DECOMPOSITION.json" || true
if [[ "$cert_rc" == 30 ]]; then
  write_pipeline_failure certificate "$raw_cert_rc" "$OUTPUTDIR/logs/certificate_controller.log" "$s0" "$s1"
  exit 30
fi

python - "$OUTPUTDIR" "$PROTOCOL_ROOT" "$SOURCE_RUN" "$raw_cert_rc" "$cert_rc" <<'PY'
import hashlib,json,pathlib,sys,time
root=pathlib.Path(sys.argv[1]); protocol=pathlib.Path(sys.argv[2]); source=pathlib.Path(sys.argv[3])
raw_rc=int(sys.argv[4]); rc=int(sys.argv[5]); variants={}
for name in ('balanced','precision'):
    p=root/'candidates'/name/'model_v48_trac_sr'/'best.pt'
    if p.is_file(): variants[name]={'checkpoint':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
doc={'event':'v48_28_provenance_margin_bridge_controller_complete','created_unix':time.time(),
     'source_run':str(source),'protocol_root':str(protocol),'variants':variants,
     'raw_certificate_exit_code':raw_rc,'certificate_exit_code':rc,
     'gate_evaluated':True,'gate_passed':(root/'NEXT_COMMANDS.txt').is_file(),
     'pipeline_valid':True,'test_roots_read':False}
(root/'V48_27_COMPLETE.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
PY
exit "$cert_rc"
