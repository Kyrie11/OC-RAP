#!/usr/bin/env bash
set -euo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"; export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"; export PYTHONUNBUFFERED=1
OUTPUTDIR="${OUTPUTDIR:-runs/ocrap_v48_15_prism_cc_dedicated_4815}"
OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
SOURCE_RUN="${SOURCE_RUN:-runs/ocrap_v48_13_terra_proxy_4801}"
PROTOCOL_ROOT="${PROTOCOL_ROOT:-$OCRAP_ROOT/calibration_v48_14_prism_4814}"
CAL_SAFE="${CAL_SAFE:-$OCRAP_ROOT/calibration_safe}"
GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
mkdir -p "$OUTPUTDIR/logs"
[[ -f "$PROTOCOL_ROOT/CALIBRATION_PROTOCOL_COMPLETE.json" ]] || { echo "missing dedicated protocol $PROTOCOL_ROOT" >&2; exit 2; }
TRAIN_NEAR="$PROTOCOL_ROOT/evidence_adapt_train_near_contact"; TRAIN_CONTACT="$PROTOCOL_ROOT/evidence_adapt_train_contact"
DEV_NEAR="$PROTOCOL_ROOT/evidence_adapt_dev_near_contact"; DEV_CONTACT="$PROTOCOL_ROOT/evidence_adapt_dev_contact"
CERT_NEAR="$PROTOCOL_ROOT/certificate_pool_near_contact"; CERT_CONTACT="$PROTOCOL_ROOT/certificate_pool_contact"
GROUP_INDEX="$OUTPUTDIR/evidence_adapt_teacher_pcd_index.jsonl"; GROUP_SUMMARY="$OUTPUTDIR/evidence_adapt_teacher_pcd_index_summary.json"
if [[ ! -f "$GROUP_INDEX" || "${REBUILD_ADAPT_INDEX:-0}" == 1 ]]; then
  python tools/build_teacher_pcd_index_v48.py --dataset "$TRAIN_NEAR,$TRAIN_CONTACT" \
    --output "$GROUP_INDEX" --summary-output "$GROUP_SUMMARY" --positive-gain="${POSITIVE_GAIN:-0.015}" --quality-mode=warn \
    >"$OUTPUTDIR/logs/build_adapt_teacher_index.log" 2>&1
fi
run_variant(){
  local variant="$1" gpu="$2" source="$SOURCE_RUN/candidates/$variant/model_v48_trac_sr/best.pt" run="$OUTPUTDIR/candidates/$variant"
  [[ -f "$source" ]] || { echo "missing source checkpoint $source" >&2; return 2; }
  RUN="$run" INIT_CKPT="$source" VARIANT="$variant" TRAIN_GPU="$gpu" \
  TRAIN_MIX="$TRAIN_NEAR,$TRAIN_CONTACT" VAL_MIX="$DEV_NEAR,$DEV_CONTACT" GROUP_INDEX="$GROUP_INDEX" \
  TRAIN_OCRAP_ROOT="$OCRAP_ROOT" EVAL_OCRAP_ROOT="$OCRAP_ROOT" \
  NUM_WORKERS="${NUM_WORKERS:-4}" PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}" BATCH_SIZE="${BATCH_SIZE:-72}" \
  EVIDENCE_ADAPT_EPOCHS="${EVIDENCE_ADAPT_EPOCHS:-10}" EVIDENCE_ADAPT_PATIENCE="${EVIDENCE_ADAPT_PATIENCE:-4}" \
  EVIDENCE_ADAPT_LR="${EVIDENCE_ADAPT_LR:-0.00020}" EVIDENCE_CALIBRATOR_HIDDEN="${EVIDENCE_CALIBRATOR_HIDDEN:-8}" \
  EVIDENCE_CALIBRATOR_SCALE="${EVIDENCE_CALIBRATOR_SCALE:-0.30}" \
  ORDINAL_EVIDENCE_HARD_HARM_WEIGHT="${ORDINAL_EVIDENCE_HARD_HARM_WEIGHT:-1.00}" \
  ORDINAL_EVIDENCE_HARD_BENEFIT_WEIGHT="${ORDINAL_EVIDENCE_HARD_BENEFIT_WEIGHT:-1.00}" \
    bash scripts/adapt_ocrap_v48_15_prism_cc_variant.sh >"$OUTPUTDIR/logs/adapt_${variant}.log" 2>&1
}
run_variant balanced "$GPU0" & p0=$!; run_variant precision "$GPU1" & p1=$!
set +e; wait "$p0"; s0=$?; wait "$p1"; s1=$?; set -e
printf 'balanced=%s precision=%s\n' "$s0" "$s1" | tee "$OUTPUTDIR/logs/adaptation_status.log"
[[ "$s0" == 0 || "$s1" == 0 ]] || exit 3
set +e
OUTPUTDIR="$OUTPUTDIR" CAL_SAFE="$CAL_SAFE" CERT_NEAR="$CERT_NEAR" CERT_CONTACT="$CERT_CONTACT" GPU0="$GPU0" GPU1="$GPU1" \
  bash scripts/calibrate_v48_14_certificate_pool.sh >"$OUTPUTDIR/logs/certificate_controller.log" 2>&1
rc=$?
set -e
python - "$OUTPUTDIR" "$PROTOCOL_ROOT" "$SOURCE_RUN" "$rc" <<'PY'
import hashlib,json,pathlib,sys,time
root,protocol,source=map(pathlib.Path,sys.argv[1:4]); rc=int(sys.argv[4]); variants={}
for name in ('balanced','precision'):
 p=root/'candidates'/name/'model_v48_trac_sr'/'best.pt'
 if p.is_file(): variants[name]={'checkpoint':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
doc={'event':'v48_15_prism_cc_controller_complete','created_unix':time.time(),'source_run':str(source),
 'protocol_root':str(protocol),'variants':variants,'certificate_exit_code':rc,
 'gate_evaluated':rc in (0,20),'gate_passed':(root/'NEXT_COMMANDS.txt').is_file(),'test_roots_read':False}
(root/'V48_15_COMPLETE.json').write_text(json.dumps(doc,indent=2)+'\n')
PY
if [[ "$rc" != 0 && "$rc" != 20 ]]; then exit "$rc"; fi
exit "$rc"
