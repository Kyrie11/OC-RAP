#!/usr/bin/env bash
set -euo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
SOURCE_OUTPUTDIR="${SOURCE_OUTPUTDIR:?set SOURCE_OUTPUTDIR to the server-side v48.25 run directory}"
OUTPUTDIR="${OUTPUTDIR:-runs/ocrap_v48_25_certificate_repair_with_v48_26}"
OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
PROTOCOL_ROOT="${PROTOCOL_ROOT:-$OCRAP_ROOT/calibration_v48_14_prism_4814}"
CAL_SAFE="${CAL_SAFE:-$OCRAP_ROOT/calibration_safe}"
GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
CERT_NEAR="$PROTOCOL_ROOT/certificate_pool_near_contact"
CERT_CONTACT="$PROTOCOL_ROOT/certificate_pool_contact"
DEV_NEAR="$PROTOCOL_ROOT/evidence_adapt_dev_near_contact"
DEV_CONTACT="$PROTOCOL_ROOT/evidence_adapt_dev_contact"
mkdir -p "$OUTPUTDIR/logs" "$OUTPUTDIR/candidates"
for variant in balanced precision; do
  src="$SOURCE_OUTPUTDIR/candidates/$variant"
  [[ -f "$src/model_v48_trac_sr/best.pt" ]] || { echo "missing $src/model_v48_trac_sr/best.pt" >&2; exit 30; }
  mkdir -p "$OUTPUTDIR/candidates/$variant"
  ln -sfn "$(realpath "$src/model_v48_trac_sr")" "$OUTPUTDIR/candidates/$variant/model_v48_trac_sr"
  if [[ -f "$src/POLICY_CONTRACT.env" ]]; then
    ln -sfn "$(realpath "$src/POLICY_CONTRACT.env")" "$OUTPUTDIR/candidates/$variant/POLICY_CONTRACT.env"
  else
    echo "missing $src/POLICY_CONTRACT.env" >&2; exit 30
  fi
  python tools/check_v48_26_model_contract.py \
    --checkpoint "$src/model_v48_trac_sr/best.pt" \
    --output "$OUTPUTDIR/candidates/$variant/MODEL_INFERENCE_CONTRACT.json" \
    --expect-frontier true --expect-admission-bounded false --expect-component-prior-logit -2.0 \
    >"$OUTPUTDIR/logs/model_contract_${variant}.log" 2>&1
done
set +e
OUTPUTDIR="$OUTPUTDIR" CAL_SAFE="$CAL_SAFE" CERT_NEAR="$CERT_NEAR" CERT_CONTACT="$CERT_CONTACT" \
DEV_NEAR="$DEV_NEAR" DEV_CONTACT="$DEV_CONTACT" GPU0="$GPU0" GPU1="$GPU1" \
VARIANTS=balanced,precision bash scripts/calibrate_v48_26_certificate_pool.sh \
  >"$OUTPUTDIR/logs/certificate_controller.log" 2>&1
rc=$?
set -e
python - "$OUTPUTDIR" "$SOURCE_OUTPUTDIR" "$rc" <<'PY'
import json,pathlib,sys,time
root=pathlib.Path(sys.argv[1]); src=sys.argv[2]; rc=int(sys.argv[3])
doc={'event':'v48_25_certificate_repair_with_v48_26_complete','created_unix':time.time(),
     'source_outputdir':src,'raw_exit_code':rc,'normalized_exit_code':0 if rc==0 else (20 if rc==20 else 30),
     'retrained':False,'test_roots_read':False,
     'interpretation':'diagnostic re-evaluation of existing v48.25 checkpoints with corrected inference and JSON serialization; it does not validate v48.26 training changes'}
(root/'REPAIR_COMPLETE.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n')
PY
case "$rc" in 0) exit 0;; 20) exit 20;; *) exit 30;; esac
