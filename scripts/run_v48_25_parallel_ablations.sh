#!/usr/bin/env bash
set -euo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
# Run one task per GPU in four waves. Each A30 receives exactly four tasks,
# avoiding the prior four-process-per-card VRAM/host contention.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"

ROOT="${ABLATION_ROOT:-runs/ocrap_v48_25_integrity_ablations_4825}"
SOURCE_RUN="${SOURCE_RUN:-runs/ocrap_v48_13_terra_proxy_4801}"
OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
PROTOCOL_ROOT="${PROTOCOL_ROOT:-$OCRAP_ROOT/calibration_v48_14_prism_4814}"
CAL_SAFE="${CAL_SAFE:-$OCRAP_ROOT/calibration_safe}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
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
root=pathlib.Path(sys.argv[1]); stage=sys.argv[2]; raw=int(sys.argv[3]); detail=sys.argv[4]
doc={'complete':False,'version':'v48.25-INTEGRITY-BRIDGE','stage':stage,
     'raw_exit_code':raw,'normalized_exit_code':30,'detail':detail,
     'created_unix':time.time(),'test_roots_read':False}
(root/'ABLATIONS_FAILED.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
PY
}

set +e
python tools/audit_dedicated_protocol_v48_16.py --protocol-root "$PROTOCOL_ROOT" \
  --output "$ROOT/dedicated_protocol_audit.json" >"$ROOT/logs/dedicated_protocol_audit.log" 2>&1
protocol_rc=$?
set -e
if [[ "$protocol_rc" != 0 ]]; then fail_controller protocol_audit "$protocol_rc" "$ROOT/logs/dedicated_protocol_audit.log"; exit 30; fi

index_contract_args=(
  --summary "$GROUP_SUMMARY" --expected-dataset "$TRAIN_NEAR,$TRAIN_CONTACT"
  --alpha "$ALPHA" --beta "$BETA" --top-m "$TOP_M" --positive-gain "$POSITIVE_GAIN"
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
  python tools/check_v48_19_target_support.py "${index_contract_args[@]}" --mode contract \
    --output "$ROOT/SUPPORT_INDEX_CONTRACT.json" >"$ROOT/logs/check_teacher_index_contract.log" 2>&1
  contract_rc=$?
  set -e
  [[ "$contract_rc" == 0 ]] || rebuild_index=1
else
  rebuild_index=1
fi
if [[ "$rebuild_index" == 1 ]]; then
  rm -f "$GROUP_INDEX" "$GROUP_SUMMARY"
  set +e
  python tools/build_teacher_pcd_index_v48.py --dataset "$TRAIN_NEAR,$TRAIN_CONTACT" --output "$GROUP_INDEX" \
    --summary-output "$GROUP_SUMMARY" --alpha "$ALPHA" --beta "$BETA" --top-m "$TOP_M" \
    --positive-gain "$POSITIVE_GAIN" --deployable-macro-ids "$DEPLOYABLE_MACRO_IDS" --quality-mode warn \
    --component-harm-drs-tolerance "$COMPONENT_HARM_DRS_TOLERANCE" \
    --component-harm-dep-tolerance "$COMPONENT_HARM_DEP_TOLERANCE" \
    --component-harm-gap-tolerance "$COMPONENT_HARM_GAP_TOLERANCE" \
    --component-harm-hard-tolerance "$COMPONENT_HARM_HARD_TOLERANCE" \
    --component-harm-proxy-tolerance "$COMPONENT_HARM_PROXY_TOLERANCE" \
    >"$ROOT/logs/build_teacher_index.log" 2>&1
  index_rc=$?
  set -e
  if [[ "$index_rc" != 0 ]]; then fail_controller teacher_index_build "$index_rc" "$ROOT/logs/build_teacher_index.log"; exit 30; fi
fi
set +e
python tools/check_v48_19_target_support.py "${index_contract_args[@]}" --mode contract \
  --output "$ROOT/SUPPORT_INDEX_CONTRACT.json" >"$ROOT/logs/check_teacher_index_contract.log" 2>&1
contract_rc=$?
set -e
if [[ "$contract_rc" != 0 ]]; then fail_controller teacher_index_contract "$contract_rc" "$ROOT/SUPPORT_INDEX_CONTRACT.json"; exit 30; fi
set +e
python tools/check_v48_19_target_support.py "${index_contract_args[@]}" --mode all \
  --output "$ROOT/SUPPORT_TARGET_SUPPORT.json" >"$ROOT/logs/check_support_target_support.log" 2>&1
target_rc=$?
set -e
if [[ "$target_rc" != 0 ]]; then fail_controller target_support "$target_rc" "$ROOT/SUPPORT_TARGET_SUPPORT.json"; exit 30; fi

rm -f "$VAL_GROUP_INDEX" "$VAL_GROUP_SUMMARY"
set +e
python tools/build_teacher_pcd_index_v48.py --dataset "$DEV_NEAR,$DEV_CONTACT" --output "$VAL_GROUP_INDEX" \
  --summary-output "$VAL_GROUP_SUMMARY" --alpha "$ALPHA" --beta "$BETA" --top-m "$TOP_M" \
  --positive-gain "$POSITIVE_GAIN" --deployable-macro-ids "$DEPLOYABLE_MACRO_IDS" --quality-mode warn \
  --component-harm-drs-tolerance "$COMPONENT_HARM_DRS_TOLERANCE" \
  --component-harm-dep-tolerance "$COMPONENT_HARM_DEP_TOLERANCE" \
  --component-harm-gap-tolerance "$COMPONENT_HARM_GAP_TOLERANCE" \
  --component-harm-hard-tolerance "$COMPONENT_HARM_HARD_TOLERANCE" \
  --component-harm-proxy-tolerance "$COMPONENT_HARM_PROXY_TOLERANCE" \
  >"$ROOT/logs/build_dev_teacher_index.log" 2>&1
val_index_rc=$?
set -e
if [[ "$val_index_rc" != 0 ]]; then fail_controller validation_teacher_index_build "$val_index_rc" "$ROOT/logs/build_dev_teacher_index.log"; exit 30; fi

run_task() {
  local group="$1" variant="$2" gpu="$3"
  local out="$ROOT/tasks/${group}_${variant}"
  local source="$SOURCE_RUN/candidates/$variant/model_v48_trac_sr/best.pt"
  local proposal_k=3 safe_reg=1.0 safe_list=0.75 frontier_pairwise=0.25 benefit_listwise=0.0
  local admission_bounded=true best_metric=direct_frontier_selection_risk
  case "$group" in
    A_wiring_fix_bounded)
      ;;
    B_add_integrity_checkpoint)
      best_metric=direct_integrity_selection_risk
      ;;
    C_add_unbounded_admission)
      best_metric=direct_integrity_selection_risk; admission_bounded=false
      ;;
    D_full_integrity_bridge)
      best_metric=direct_integrity_selection_risk; admission_bounded=false
      benefit_listwise=0.75; frontier_pairwise=0.75
      ;;
    *) echo "unknown group $group" >&2; return 30 ;;
  esac
  [[ -f "$source" ]] || { echo "missing source $source" >&2; return 30; }
  rm -rf "$out"; mkdir -p "$out/logs"
  set +e
  RUN="$out/candidates/$variant" INIT_CKPT="$source" VARIANT="$variant" TRAIN_GPU="$gpu" \
  TRAIN_MIX="$TRAIN_NEAR,$TRAIN_CONTACT" VAL_MIX="$DEV_NEAR,$DEV_CONTACT" GROUP_INDEX="$GROUP_INDEX" VAL_GROUP_INDEX="$VAL_GROUP_INDEX" \
  TRAIN_OCRAP_ROOT="$OCRAP_ROOT" EVAL_OCRAP_ROOT="$OCRAP_ROOT" \
  NUM_WORKERS="${NUM_WORKERS:-2}" PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}" BATCH_SIZE="${BATCH_SIZE:-72}" \
  EVIDENCE_ADAPT_EPOCHS="${EVIDENCE_ADAPT_EPOCHS:-28}" EVIDENCE_ADAPT_PATIENCE="${EVIDENCE_ADAPT_PATIENCE:-8}" \
  EVIDENCE_ADAPT_LR="${EVIDENCE_ADAPT_LR:-0.00015}" PROPOSAL_TOP_K="$proposal_k" \
  ORDINAL_EVIDENCE_GROUP_OPPORTUNITY_WEIGHT=0.0 \
  ORDINAL_EVIDENCE_SAFE_BENEFIT_TARGET=true GROUP_BATCH_SAFE_POSITIVE_TARGET=true \
  ORDINAL_EVIDENCE_BENEFIT_LISTWISE_WEIGHT="$benefit_listwise" \
  ORDINAL_EVIDENCE_SAFE_UTILITY_REGRESSION_WEIGHT="$safe_reg" \
  ORDINAL_EVIDENCE_SAFE_UTILITY_LISTWISE_WEIGHT="$safe_list" \
  ORDINAL_EVIDENCE_FRONTIER_PAIRWISE_WEIGHT="$frontier_pairwise" \
  EVIDENCE_ADMISSION_BOUNDED="$admission_bounded" EVALUATE_INITIAL_CHECKPOINT=true BEST_METRIC="$best_metric" \
    bash scripts/adapt_ocrap_v48_25_integrity_variant.sh >"$out/logs/adapt.log" 2>&1
  local adapt_rc=$?
  set -e
  if [[ "$adapt_rc" != 0 ]]; then
    printf '{"complete":false,"stage":"adaptation","raw_exit_code":%s,"normalized_exit_code":30}\n' "$adapt_rc" > "$out/TASK_FAILED.json"
    return 30
  fi
  set +e
  OUTPUTDIR="$out" CAL_SAFE="$CAL_SAFE" CERT_NEAR="$CERT_NEAR" CERT_CONTACT="$CERT_CONTACT" DEV_NEAR="$DEV_NEAR" DEV_CONTACT="$DEV_CONTACT" \
  GPU0="$gpu" GPU1="$gpu" VARIANTS="$variant" CERTIFICATE_GRID_SIZE="${CERTIFICATE_GRID_SIZE:-31}" \
    bash scripts/calibrate_v48_25_certificate_pool.sh >"$out/logs/calibrate.log" 2>&1
  local cert_rc=$?
  set -e
  if [[ "$cert_rc" != 0 && "$cert_rc" != 20 ]]; then
    printf '{"complete":false,"stage":"certificate","raw_exit_code":%s,"normalized_exit_code":30}\n' "$cert_rc" > "$out/TASK_FAILED.json"
    return 30
  fi
  python - "$out" "$group" "$variant" "$cert_rc" "$proposal_k" "$safe_reg" "$safe_list" "$frontier_pairwise" "$benefit_listwise" "$admission_bounded" "$best_metric" <<'PY_TASK'
import hashlib,json,pathlib,sys,time
out=pathlib.Path(sys.argv[1]); group=sys.argv[2]; variant=sys.argv[3]; rc=int(sys.argv[4])
proposal_k=int(sys.argv[5]); safe_reg=float(sys.argv[6]); safe_list=float(sys.argv[7]); frontier=float(sys.argv[8]); benefit_listwise=float(sys.argv[9]); admission_bounded=sys.argv[10].lower()=='true'; best_metric=sys.argv[11]
base=out/'candidates'/variant; ckpt=base/'model_v48_trac_sr'/'best.pt'; cal=base/'calibration'
required=[ckpt,cal/'CERTIFICATE_CALIBRATION_COMPLETE.json',cal/'gamma_rec_by_bucket_v48.json',cal/'direct_value_risk_near_v48.json',cal/'direct_value_risk_contact_v48.json']
missing=[str(p) for p in required if not p.is_file()]
if missing: raise SystemExit('incomplete: '+','.join(missing))
regimes={}
for regime in ('near','contact'):
    d=json.load(open(cal/f'direct_value_risk_{regime}_v48.json'))
    if int(d.get('num_groups',0) or 0)<=0 or int(d.get('num_scenes',0) or 0)<=0: raise SystemExit(f'empty {regime}')
    regimes[regime]={'oracle':d.get('proposal_constrained_oracle_gate'),'support_curve':d.get('proposal_support_curve'),'valid':d.get('valid_for_deployment')}
doc={'complete':True,'version':'v48.25-INTEGRITY-BRIDGE','group':group,'variant':variant,
     'proposal_top_k':proposal_k,'safe_utility_regression_weight':safe_reg,
     'safe_utility_listwise_weight':safe_list,'frontier_pairwise_weight':frontier,
     'benefit_listwise_weight':benefit_listwise,'admission_bounded':admission_bounded,'best_metric':best_metric,
     'certificate_exit':rc,'gate_passed':rc==0,'regimes':regimes,'created_unix':time.time(),
     'checkpoint_sha256':hashlib.sha256(ckpt.read_bytes()).hexdigest(),'test_roots_read':False}
(out/'TASK_COMPLETE.json').write_text(json.dumps(doc,indent=2)+'\n')
PY_TASK
}

groups=(A_wiring_fix_bounded B_add_integrity_checkpoint C_add_unbounded_admission D_full_integrity_bridge)
: > "$ROOT/TASK_GPU_ASSIGNMENT.txt"
failures=0
set +e
for group in "${groups[@]}"; do
  echo "${group}_balanced:gpu${GPU0}" >> "$ROOT/TASK_GPU_ASSIGNMENT.txt"
  echo "${group}_precision:gpu${GPU1}" >> "$ROOT/TASK_GPU_ASSIGNMENT.txt"
  run_task "$group" balanced "$GPU0" & p0=$!
  run_task "$group" precision "$GPU1" & p1=$!
  wait "$p0"; r0=$?
  wait "$p1"; r1=$?
  if [[ "$r0" != 0 ]]; then echo "[failed] ${group}_balanced rc=$r0" >&2; failures=$((failures+1)); fi
  if [[ "$r1" != 0 ]]; then echo "[failed] ${group}_precision rc=$r1" >&2; failures=$((failures+1)); fi
done
set -e

python tools/summarize_v48_14_ablations.py --root "$ROOT" --output "$ROOT/ablation_summary_v48_25.json" --version v48.25-INTEGRITY-BRIDGE || true
set +e
python - "$ROOT" "$failures" <<'PY_STATUS'
import json,pathlib,sys,time
root=pathlib.Path(sys.argv[1]); failures=int(sys.argv[2])
groups=['A_wiring_fix_bounded','B_add_integrity_checkpoint','C_add_unbounded_admission','D_full_integrity_bridge']
expected=[f'{g}_{v}' for g in groups for v in ('balanced','precision')]
missing=[x for x in expected if not (root/'tasks'/x/'TASK_COMPLETE.json').is_file()]
doc={'complete':not missing and failures==0,'version':'v48.25-INTEGRITY-BRIDGE','max_concurrent_tasks':2,
     'execution':'four waves; one task per A30; four tasks assigned to each GPU','workers_per_task':2,
     'expected_tasks':expected,'missing_tasks':missing,'failed_waits':failures,
     'created_unix':time.time(),'test_roots_read':False}
(root/'ABLATIONS_STATUS.json').write_text(json.dumps(doc,indent=2)+'\n')
if doc['complete']:
    (root/'ABLATIONS_COMPLETE.json').write_text(json.dumps(doc,indent=2)+'\n')
else:
    raise SystemExit('incomplete: '+','.join(missing))
PY_STATUS
final_rc=$?
set -e
if [[ "$final_rc" != 0 ]]; then fail_controller task_completion "$final_rc" "$ROOT/ABLATIONS_STATUS.json"; exit 30; fi
exit 0
