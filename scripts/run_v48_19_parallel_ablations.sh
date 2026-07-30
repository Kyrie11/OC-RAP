#!/usr/bin/env bash
set -euo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
# Four concurrent jobs share two A30 cards. Cap host math threads so data loading
# and small calibration MLPs do not oversubscribe CPU cores.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"

ROOT="${ABLATION_ROOT:-runs/ocrap_v48_19_facet_ablations}"
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

fail_controller() {
  local stage="$1" raw_rc="$2" detail="${3:-}"
  python - "$ROOT" "$stage" "$raw_rc" "$detail" <<'PY'
import json,pathlib,sys,time
root=pathlib.Path(sys.argv[1]); stage=sys.argv[2]; raw=int(sys.argv[3]); detail=sys.argv[4]
doc={'complete':False,'version':'v48.19-FACET-BRIDGE','stage':stage,
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
    --output "$ROOT/FACET_INDEX_CONTRACT.json" >"$ROOT/logs/check_teacher_index_contract.log" 2>&1
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
  --output "$ROOT/FACET_INDEX_CONTRACT.json" >"$ROOT/logs/check_teacher_index_contract.log" 2>&1
contract_rc=$?
set -e
if [[ "$contract_rc" != 0 ]]; then fail_controller teacher_index_contract "$contract_rc" "$ROOT/FACET_INDEX_CONTRACT.json"; exit 30; fi
set +e
python tools/check_v48_19_target_support.py "${index_contract_args[@]}" --mode all \
  --output "$ROOT/FACET_TARGET_SUPPORT.json" >"$ROOT/logs/check_facet_target_support.log" 2>&1
target_rc=$?
set -e
if [[ "$target_rc" != 0 ]]; then fail_controller target_support "$target_rc" "$ROOT/FACET_TARGET_SUPPORT.json"; exit 30; fi

run_task() {
  local group="$1" variant="$2" gpu="$3"
  local out="$ROOT/tasks/${group}_${variant}"
  local source="$SOURCE_RUN/candidates/$variant/model_v48_trac_sr/best.pt"
  local shared=true regime_scale=0.25 metric=direct_certificate_risk_fold_robust
  case "$group" in
    A_component_veto_separate) shared=false ;;
    B_shared_component_veto) ;;
    C_shared_only_no_regime_residual) regime_scale=0.0 ;;
    D_full_facet) metric=direct_facet_selection_risk ;;
    *) echo "unknown group $group" >&2; return 30 ;;
  esac
  [[ -f "$source" ]] || { echo "missing source $source" >&2; return 30; }
  rm -rf "$out"; mkdir -p "$out/logs"
  set +e
  RUN="$out/candidates/$variant" INIT_CKPT="$source" VARIANT="$variant" TRAIN_GPU="$gpu" \
  TRAIN_MIX="$TRAIN_NEAR,$TRAIN_CONTACT" VAL_MIX="$DEV_NEAR,$DEV_CONTACT" GROUP_INDEX="$GROUP_INDEX" \
  TRAIN_OCRAP_ROOT="$OCRAP_ROOT" EVAL_OCRAP_ROOT="$OCRAP_ROOT" \
  NUM_WORKERS="${NUM_WORKERS:-1}" PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}" BATCH_SIZE="${BATCH_SIZE:-64}" \
  EVIDENCE_ADAPT_EPOCHS="${EVIDENCE_ADAPT_EPOCHS:-18}" EVIDENCE_ADAPT_PATIENCE="${EVIDENCE_ADAPT_PATIENCE:-5}" \
  EVIDENCE_ADAPT_LR="${EVIDENCE_ADAPT_LR:-0.00025}" EVIDENCE_CALIBRATOR_MODE=dual_tail_context \
  EVIDENCE_CALIBRATOR_CONTEXT=true EVIDENCE_CALIBRATOR_CONTEXT_SOURCE=tournament \
  EVIDENCE_CALIBRATOR_SHARED="$shared" EVIDENCE_CALIBRATOR_REGIME_SCALE="$regime_scale" \
  GROUP_BATCH_STRATIFIED=true ORDINAL_EVIDENCE_BATCH_BALANCED=true \
  ORDINAL_EVIDENCE_BALANCED_REPLACES_ERM=true ORDINAL_EVIDENCE_INDEPENDENT_TAILS=true \
  ORDINAL_EVIDENCE_FACTORIZED_HARM=true POSITIVE_GROUP_BOOST="${POSITIVE_GROUP_BOOST:-1.0}" BEST_METRIC="$metric" \
    bash scripts/adapt_ocrap_v48_19_facet_variant.sh >"$out/logs/adapt.log" 2>&1
  local adapt_rc=$?
  set -e
  if [[ "$adapt_rc" != 0 ]]; then
    printf '{"complete":false,"stage":"adaptation","raw_exit_code":%s,"normalized_exit_code":30}\n' "$adapt_rc" > "$out/TASK_FAILED.json"
    return 30
  fi
  set +e
  OUTPUTDIR="$out" CAL_SAFE="$CAL_SAFE" CERT_NEAR="$CERT_NEAR" CERT_CONTACT="$CERT_CONTACT" \
  GPU0="$gpu" GPU1="$gpu" VARIANTS="$variant" \
    bash scripts/calibrate_v48_19_certificate_pool.sh >"$out/logs/calibrate.log" 2>&1
  local cert_rc=$?
  set -e
  if [[ "$cert_rc" != 0 && "$cert_rc" != 20 ]]; then
    printf '{"complete":false,"stage":"certificate","raw_exit_code":%s,"normalized_exit_code":30}\n' "$cert_rc" > "$out/TASK_FAILED.json"
    return 30
  fi
  set +e
  python - "$out" "$group" "$variant" "$cert_rc" "$shared" "$regime_scale" "$metric" <<'PY'
import hashlib,json,pathlib,sys,time
out=pathlib.Path(sys.argv[1]); group,variant=sys.argv[2:4]; rc=int(sys.argv[4]); shared=sys.argv[5].lower()=='true'; scale=float(sys.argv[6]); metric=sys.argv[7]
base=out/'candidates'/variant; ckpt=base/'model_v48_trac_sr'/'best.pt'; cal=base/'calibration'
required=[ckpt,cal/'CERTIFICATE_CALIBRATION_COMPLETE.json',cal/'gamma_rec_by_bucket_v48.json',cal/'direct_value_risk_near_v48.json',cal/'direct_value_risk_contact_v48.json']
missing=[str(p) for p in required if not p.is_file()]
if missing: raise SystemExit('incomplete: '+','.join(missing))
for regime in ('near','contact'):
    d=json.load(open(cal/f'direct_value_risk_{regime}_v48.json'))
    if int(d.get('num_groups',0) or 0)<=0 or int(d.get('num_scenes',0) or 0)<=0: raise SystemExit(f'empty {regime}')
    if not bool((d.get('certificate_support_feasibility') or {}).get('overall',False)): raise SystemExit(f'infeasible certificate {regime}')
doc={'complete':True,'version':'v48.19-FACET-BRIDGE','group':group,'variant':variant,
     'shared_calibrator':shared,'regime_residual_scale':scale,'checkpoint_metric':metric,
     'certificate_exit':rc,'gate_passed':rc==0,'created_unix':time.time(),
     'checkpoint_sha256':hashlib.sha256(ckpt.read_bytes()).hexdigest(),'test_roots_read':False}
(out/'TASK_COMPLETE.json').write_text(json.dumps(doc,indent=2)+'\n')
PY
  local artifact_rc=$?
  set -e
  if [[ "$artifact_rc" != 0 ]]; then
    printf '{"complete":false,"stage":"artifact_validation","raw_exit_code":%s,"normalized_exit_code":30}\n' "$artifact_rc" > "$out/TASK_FAILED.json"
    return 30
  fi
  return 0
}

groups=(A_component_veto_separate B_shared_component_veto C_shared_only_no_regime_residual D_full_facet)
run_wave() {
  local variant="$1"
  local pids=() labels=() idx=0 failures=0
  for group in "${groups[@]}"; do
    local gpu="$GPU0"
    (( idx % 2 == 1 )) && gpu="$GPU1"
    run_task "$group" "$variant" "$gpu" & pids+=("$!")
    labels+=("${group}_${variant}")
    idx=$((idx+1))
  done
  set +e
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}"; rc=$?
    if [[ "$rc" != 0 ]]; then echo "[failed] ${labels[$i]} rc=$rc" >&2; failures=$((failures+1)); fi
  done
  set -e
  return "$failures"
}

failures=0
set +e
run_wave balanced; failures=$((failures+$?))
run_wave precision; failures=$((failures+$?))
set -e
python tools/summarize_v48_14_ablations.py --root "$ROOT" --output "$ROOT/ablation_summary_v48_19.json" --version v48.19-FACET-BRIDGE || true
set +e
python - "$ROOT" "$failures" <<'PY'
import json,pathlib,sys,time
root=pathlib.Path(sys.argv[1]); failures=int(sys.argv[2]); groups=['A_component_veto_separate','B_shared_component_veto','C_shared_only_no_regime_residual','D_full_facet']
expected=[f'{g}_{v}' for v in ('balanced','precision') for g in groups]
missing=[x for x in expected if not (root/'tasks'/x/'TASK_COMPLETE.json').is_file()]
doc={'complete':not missing and failures==0,'version':'v48.19-FACET-BRIDGE','max_concurrent_tasks':4,
     'execution':'two waves; four tasks per wave; two tasks per A30','workers_per_task':1,
     'expected_tasks':expected,'missing_tasks':missing,'failed_waits':failures,
     'created_unix':time.time(),'test_roots_read':False}
(root/'ABLATIONS_STATUS.json').write_text(json.dumps(doc,indent=2)+'\n')
if doc['complete']:
    (root/'ABLATIONS_COMPLETE.json').write_text(json.dumps(doc,indent=2)+'\n')
else:
    raise SystemExit('incomplete: '+','.join(missing))
PY
final_rc=$?
set -e
if [[ "$final_rc" != 0 ]]; then fail_controller task_completion "$final_rc" "$ROOT/ABLATIONS_STATUS.json"; exit 30; fi
exit 0
