#!/usr/bin/env bash
set -euo pipefail

REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

OUTPUTDIR="${OUTPUTDIR:-runs/ocrap_v48_14_prism_dedicated_4814}"
OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
SOURCE_RUN="${SOURCE_RUN:-runs/ocrap_v48_13_terra_proxy_4801}"
PROTOCOL_ROOT="${PROTOCOL_ROOT:-$OCRAP_ROOT/calibration_v48_14_prism_4814}"
CAL_SAFE="${CAL_SAFE:-$OCRAP_ROOT/calibration_safe}"
CAL_NEAR="${CAL_NEAR:-$OCRAP_ROOT/calibration_near_contact}"
CAL_CONTACT="${CAL_CONTACT:-$OCRAP_ROOT/calibration_contact}"
GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
mkdir -p "$OUTPUTDIR/logs"

# Prepare a strict three-role scene split once. Re-running without overwrite
# reuses the immutable protocol manifest.
if [[ ! -f "$PROTOCOL_ROOT/CALIBRATION_PROTOCOL_COMPLETE.json" ]]; then
  python tools/partition_dedicated_calibration_v48_14.py \
    --near "$CAL_NEAR" --contact "$CAL_CONTACT" --output-root "$PROTOCOL_ROOT" \
    --adapt-train-fraction="${ADAPT_TRAIN_FRACTION:-0.45}" \
    --adapt-dev-fraction="${ADAPT_DEV_FRACTION:-0.15}" \
    --seed="${PROTOCOL_SEED:-4814}" --link-mode="${LINK_MODE:-hardlink}" --overwrite \
    2>&1 | tee "$OUTPUTDIR/logs/partition_dedicated_protocol.log"
fi

ADAPT_TRAIN_NEAR="$PROTOCOL_ROOT/evidence_adapt_train_near_contact"
ADAPT_TRAIN_CONTACT="$PROTOCOL_ROOT/evidence_adapt_train_contact"
ADAPT_DEV_NEAR="$PROTOCOL_ROOT/evidence_adapt_dev_near_contact"
ADAPT_DEV_CONTACT="$PROTOCOL_ROOT/evidence_adapt_dev_contact"
CERT_NEAR="$PROTOCOL_ROOT/certificate_pool_near_contact"
CERT_CONTACT="$PROTOCOL_ROOT/certificate_pool_contact"
GROUP_INDEX="$OUTPUTDIR/evidence_adapt_teacher_pcd_index.jsonl"
GROUP_SUMMARY="$OUTPUTDIR/evidence_adapt_teacher_pcd_index_summary.json"

if [[ ! -f "$GROUP_INDEX" || "${REBUILD_ADAPT_INDEX:-0}" == 1 ]]; then
  python tools/build_teacher_pcd_index_v48.py \
    --dataset "$ADAPT_TRAIN_NEAR,$ADAPT_TRAIN_CONTACT" \
    --output "$GROUP_INDEX" --summary-output "$GROUP_SUMMARY" \
    --positive-gain="${POSITIVE_GAIN:-0.015}" --quality-mode=warn \
    2>&1 | tee "$OUTPUTDIR/logs/build_adapt_teacher_index.log"
fi

run_variant() {
  local variant="$1" gpu="$2"
  local source="$SOURCE_RUN/candidates/$variant/model_v48_trac_sr/best.pt"
  local run="$OUTPUTDIR/candidates/$variant"
  [[ -f "$source" ]] || { echo "missing source checkpoint $source" >&2; return 2; }
  mkdir -p "$run"
  RUN="$run" INIT_CKPT="$source" VARIANT="$variant" TRAIN_GPU="$gpu" \
  TRAIN_MIX="$ADAPT_TRAIN_NEAR,$ADAPT_TRAIN_CONTACT" \
  VAL_MIX="$ADAPT_DEV_NEAR,$ADAPT_DEV_CONTACT" GROUP_INDEX="$GROUP_INDEX" \
  TRAIN_OCRAP_ROOT="$OCRAP_ROOT" EVAL_OCRAP_ROOT="$OCRAP_ROOT" \
  NUM_WORKERS="${NUM_WORKERS:-6}" PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}" BATCH_SIZE="${BATCH_SIZE:-72}" \
  EVIDENCE_ADAPT_EPOCHS="${EVIDENCE_ADAPT_EPOCHS:-8}" EVIDENCE_ADAPT_PATIENCE="${EVIDENCE_ADAPT_PATIENCE:-3}" \
  ORDINAL_EVIDENCE_HARD_HARM_WEIGHT="${ORDINAL_EVIDENCE_HARD_HARM_WEIGHT:-2.50}" \
  ORDINAL_EVIDENCE_HARD_BENEFIT_WEIGHT="${ORDINAL_EVIDENCE_HARD_BENEFIT_WEIGHT:-0.75}" \
    bash scripts/adapt_ocrap_v48_14_prism_variant.sh >"$OUTPUTDIR/logs/adapt_${variant}.log" 2>&1
}

run_variant balanced "$GPU0" & P0=$!
run_variant precision "$GPU1" & P1=$!
set +e; wait "$P0"; S0=$?; wait "$P1"; S1=$?; set -e
printf 'balanced=%s precision=%s\n' "$S0" "$S1" | tee "$OUTPUTDIR/logs/adaptation_status.log"
[[ "$S0" == 0 || "$S1" == 0 ]] || exit 3

set +e
OUTPUTDIR="$OUTPUTDIR" CAL_SAFE="$CAL_SAFE" CERT_NEAR="$CERT_NEAR" CERT_CONTACT="$CERT_CONTACT" \
GPU0="$GPU0" GPU1="$GPU1" \
  bash scripts/calibrate_v48_14_certificate_pool.sh \
  >"$OUTPUTDIR/logs/certificate_controller.log" 2>&1
rc=$?
set -e
if [[ "$rc" != 0 && "$rc" != 20 ]]; then exit "$rc"; fi

python - "$OUTPUTDIR" "$PROTOCOL_ROOT" "$SOURCE_RUN" "$rc" <<'PY'
import hashlib,json,pathlib,sys,time
root,protocol,source=map(pathlib.Path,sys.argv[1:4]); rc=int(sys.argv[4])
required=[protocol/'CALIBRATION_PROTOCOL_COMPLETE.json',root/'evidence_adapt_teacher_pcd_index.jsonl']
missing=[str(p) for p in required if not p.is_file()]
variants={}
for name in ('balanced','precision'):
    p=root/'candidates'/name/'model_v48_trac_sr'/'best.pt'
    if p.is_file(): variants[name]={'checkpoint':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
doc={'event':'v48_14_prism_controller_complete','created_unix':time.time(),'source_run':str(source),
     'protocol_root':str(protocol),'variants':variants,'certificate_exit_code':rc,'gate_passed':(root/'NEXT_COMMANDS.txt').is_file(),
     'test_roots_read':False,'missing':missing}
(root/'V48_14_COMPLETE.json').write_text(json.dumps(doc,indent=2)+'\n')
if missing or not variants: raise SystemExit('incomplete v48.14 run')
PY

exit "$rc"
