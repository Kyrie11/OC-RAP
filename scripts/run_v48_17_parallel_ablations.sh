#!/usr/bin/env bash
set -euo pipefail

# BRIDGE component ablations. For each variant, A/C share GPU0 and B uses GPU1.
# The uploaded v48.16 D_full_anchor result is the scalar center/width baseline;
# this suite only runs the three new components needed to attribute v48.17.
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
ROOT="${ABLATION_ROOT:-runs/ocrap_v48_17_bridge_ablations}"
SOURCE_RUN="${SOURCE_RUN:-runs/ocrap_v48_13_terra_proxy_4801}"
OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
PROTOCOL_ROOT="${PROTOCOL_ROOT:-$OCRAP_ROOT/calibration_v48_14_prism_4814}"
CAL_SAFE="${CAL_SAFE:-$OCRAP_ROOT/calibration_safe}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
mkdir -p "$ROOT/tasks" "$ROOT/logs"

python tools/audit_dedicated_protocol_v48_16.py \
  --protocol-root "$PROTOCOL_ROOT" \
  --output "$ROOT/dedicated_protocol_audit.json"
TRAIN_NEAR="$PROTOCOL_ROOT/evidence_adapt_train_near_contact"
TRAIN_CONTACT="$PROTOCOL_ROOT/evidence_adapt_train_contact"
DEV_NEAR="$PROTOCOL_ROOT/evidence_adapt_dev_near_contact"
DEV_CONTACT="$PROTOCOL_ROOT/evidence_adapt_dev_contact"
CERT_NEAR="$PROTOCOL_ROOT/certificate_pool_near_contact"
CERT_CONTACT="$PROTOCOL_ROOT/certificate_pool_contact"
GROUP_INDEX="$ROOT/evidence_adapt_teacher_pcd_index.jsonl"
GROUP_SUMMARY="$ROOT/evidence_adapt_teacher_pcd_index_summary.json"
if [[ ! -f "$GROUP_INDEX" || "${REBUILD_ADAPT_INDEX:-0}" == 1 ]]; then
  python tools/build_teacher_pcd_index_v48.py \
    --dataset "$TRAIN_NEAR,$TRAIN_CONTACT" \
    --output "$GROUP_INDEX" \
    --summary-output "$GROUP_SUMMARY" \
    --positive-gain="${POSITIVE_GAIN:-0.015}" \
    --quality-mode=warn \
    >"$ROOT/logs/build_teacher_index.log" 2>&1
fi

run_task() {
  local group="$1" variant="$2" gpu="$3"
  local out="$ROOT/tasks/${group}_${variant}"
  local source="$SOURCE_RUN/candidates/$variant/model_v48_trac_sr/best.pt"
  local context=false stratified=false batch_balanced=false min_recall=0 recall_weight=0 positive_boost=5.0
  case "$group" in
    A_simplex_scalar)
      context=false
      ;;
    B_context_simplex)
      context=true
      ;;
    C_full_bridge)
      context=true
      stratified=true
      batch_balanced=true
      min_recall="${POLICY_METRIC_MIN_POSITIVE_RECALL:-0.25}"
      recall_weight="${POLICY_METRIC_RECALL_SHORTFALL_WEIGHT:-4.0}"
      positive_boost=1.0
      ;;
    *) echo "unknown ablation group: $group" >&2; return 30 ;;
  esac
  [[ -f "$source" ]] || { echo "missing source checkpoint: $source" >&2; return 30; }
  rm -rf "$out"
  mkdir -p "$out/logs"
  set +e
  RUN="$out/candidates/$variant" INIT_CKPT="$source" VARIANT="$variant" TRAIN_GPU="$gpu" \
  TRAIN_MIX="$TRAIN_NEAR,$TRAIN_CONTACT" VAL_MIX="$DEV_NEAR,$DEV_CONTACT" GROUP_INDEX="$GROUP_INDEX" \
  TRAIN_OCRAP_ROOT="$OCRAP_ROOT" EVAL_OCRAP_ROOT="$OCRAP_ROOT" \
  NUM_WORKERS="${NUM_WORKERS:-3}" PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}" BATCH_SIZE="${BATCH_SIZE:-72}" \
  EVIDENCE_ADAPT_EPOCHS="${EVIDENCE_ADAPT_EPOCHS:-20}" EVIDENCE_ADAPT_PATIENCE="${EVIDENCE_ADAPT_PATIENCE:-5}" \
  EVIDENCE_ADAPT_LR="${EVIDENCE_ADAPT_LR:-0.00030}" \
  EVIDENCE_CALIBRATOR_MODE=simplex_context EVIDENCE_CALIBRATOR_CONTEXT="$context" \
  GROUP_BATCH_STRATIFIED="$stratified" ORDINAL_EVIDENCE_BATCH_BALANCED="$batch_balanced" \
  POSITIVE_GROUP_BOOST="$positive_boost" \
  POLICY_METRIC_MIN_POSITIVE_RECALL="$min_recall" POLICY_METRIC_RECALL_SHORTFALL_WEIGHT="$recall_weight" \
    bash scripts/adapt_ocrap_v48_17_bridge_variant.sh >"$out/logs/adapt.log" 2>&1
  local adapt_rc=$?
  set -e
  if [[ "$adapt_rc" != 0 ]]; then
    printf '{"complete":false,"stage":"adaptation","exit_code":%s}\n' "$adapt_rc" >"$out/TASK_FAILED.json"
    return 30
  fi
  set +e
  OUTPUTDIR="$out" CAL_SAFE="$CAL_SAFE" CERT_NEAR="$CERT_NEAR" CERT_CONTACT="$CERT_CONTACT" \
  GPU0="$gpu" GPU1="$gpu" VARIANTS="$variant" \
    bash scripts/calibrate_v48_16_certificate_pool.sh >"$out/logs/calibrate.log" 2>&1
  local cert_rc=$?
  set -e
  if [[ "$cert_rc" != 0 && "$cert_rc" != 20 ]]; then
    printf '{"complete":false,"stage":"certificate","exit_code":%s}\n' "$cert_rc" >"$out/TASK_FAILED.json"
    return 30
  fi
  python - "$out" "$group" "$variant" "$cert_rc" <<'PY'
import hashlib, json, pathlib, sys, time
out=pathlib.Path(sys.argv[1]); group=sys.argv[2]; variant=sys.argv[3]; rc=int(sys.argv[4])
base=out/'candidates'/variant; ckpt=base/'model_v48_trac_sr'/'best.pt'; cal=base/'calibration'
required=[ckpt, cal/'CERTIFICATE_CALIBRATION_COMPLETE.json', cal/'gamma_rec_by_bucket_v48.json', cal/'direct_value_risk_near_v48.json', cal/'direct_value_risk_contact_v48.json']
missing=[str(path) for path in required if not path.is_file()]
if missing: raise SystemExit('incomplete task: '+','.join(missing))
for bucket in ('near','contact'):
    doc=json.load(open(cal/f'direct_value_risk_{bucket}_v48.json'))
    if int(doc.get('num_groups',0) or 0)<=0 or int(doc.get('num_scenes',0) or 0)<=0:
        raise SystemExit(f'empty {bucket} certificate')
summary={'complete':True,'version':'v48.17-BRIDGE','group':group,'variant':variant,'certificate_exit':rc,'gate_passed':rc==0,'created_unix':time.time(),'checkpoint_sha256':hashlib.sha256(ckpt.read_bytes()).hexdigest()}
(out/'TASK_COMPLETE.json').write_text(json.dumps(summary,indent=2)+'\n')
PY
}

SPECS=("A_simplex_scalar 0" "B_context_simplex 1" "C_full_bridge 0")
failures=0
for variant in balanced precision; do
  pids=(); labels=()
  for spec in "${SPECS[@]}"; do
    read -r group slot <<<"$spec"
    gpu="$GPU0"; [[ "$slot" == 1 ]] && gpu="$GPU1"
    run_task "$group" "$variant" "$gpu" &
    pids+=("$!"); labels+=("${group}_${variant}")
  done
  set +e
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}"; rc=$?
    if [[ "$rc" != 0 ]]; then
      echo "[failed] ${labels[$i]} rc=$rc" >&2
      failures=$((failures+1))
    fi
  done
  set -e
done
python tools/summarize_v48_14_ablations.py \
  --root "$ROOT" --output "$ROOT/ablation_summary_v48_17.json" --version v48.17-BRIDGE || true
python - "$ROOT" "$failures" <<'PY'
import json, pathlib, sys, time
root=pathlib.Path(sys.argv[1]); failures=int(sys.argv[2])
groups=['A_simplex_scalar','B_context_simplex','C_full_bridge']
expected=[f'{group}_{variant}' for variant in ('balanced','precision') for group in groups]
missing=[name for name in expected if not (root/'tasks'/name/'TASK_COMPLETE.json').is_file()]
doc={'complete':not missing and failures==0,'version':'v48.17-BRIDGE','max_concurrent_tasks':3,'gpu_assignment':'GPU0:A+C, GPU1:B','expected_tasks':expected,'missing_tasks':missing,'failed_waits':failures,'created_unix':time.time()}
(root/'ABLATIONS_STATUS.json').write_text(json.dumps(doc,indent=2)+'\n')
if doc['complete']:
    (root/'ABLATIONS_COMPLETE.json').write_text(json.dumps(doc,indent=2)+'\n')
else:
    raise SystemExit('ablation suite incomplete: '+','.join(missing))
PY
