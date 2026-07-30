#!/usr/bin/env bash
set -euo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"; export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"; export PYTHONUNBUFFERED=1
OUTPUTDIR="${OUTPUTDIR:-runs/ocrap_v48_17_bridge_dedicated_4817}"
OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
PROTOCOL_ROOT="${PROTOCOL_ROOT:-$OCRAP_ROOT/calibration_v48_14_prism_4814}"
CAL_SAFE="${CAL_SAFE:-$OCRAP_ROOT/calibration_safe}"
GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
MAX_EVIDENCE_CALIBRATOR_PARAMS="${MAX_EVIDENCE_CALIBRATOR_PARAMS:-100000}"
mkdir -p "$OUTPUTDIR/logs"
variants=""
for variant in balanced precision; do
  run="$OUTPUTDIR/candidates/$variant"; ckpt="$run/model_v48_trac_sr/best.pt"
  [[ -f "$ckpt" && -f "$run/POLICY_CONTRACT.env" ]] || continue
  python - "$run" "$ckpt" "$MAX_EVIDENCE_CALIBRATOR_PARAMS" <<'PY'
import hashlib,json,pathlib,sys,time,torch
run=pathlib.Path(sys.argv[1]); ckpt=pathlib.Path(sys.argv[2]); cap=int(sys.argv[3])
try: doc=torch.load(ckpt,map_location='cpu',weights_only=False)
except TypeError: doc=torch.load(ckpt,map_location='cpu')
state=doc.get('model_state',{}); n=sum(v.numel() for k,v in state.items() if k.startswith('direct_evidence_calibrators.'))
if n<=0 or n>cap: raise SystemExit(f'calibrator params {n} outside (0,{cap}]')
out={'event':'v48_17_bridge_evidence_correction_recovered','created_unix':time.time(),'checkpoint':str(ckpt),'checkpoint_sha256':hashlib.sha256(ckpt.read_bytes()).hexdigest(),'trainable_prefixes':['direct_evidence_calibrators'],'trainable_state_params':n,'recovered_from_postcheck_guard':True,'test_roots_read':False}
(run/'EVIDENCE_CORRECTION_COMPLETE.json').write_text(json.dumps(out,indent=2)+'\n')
PY
  variants="${variants:+$variants,}$variant"
done
if [[ -z "$variants" ]]; then
  printf '{"event":"v48_17_recovery_failed","reason":"no_valid_trained_checkpoint"}\n' > "$OUTPUTDIR/PIPELINE_FAILED.json"
  python tools/check_v48_16_learning_gates.py --run "$OUTPUTDIR" --output "$OUTPUTDIR/learning_gates_v48_17.json" --version v48.17-BRIDGE || true
  exit 30
fi
CERT_NEAR="$PROTOCOL_ROOT/certificate_pool_near_contact"; CERT_CONTACT="$PROTOCOL_ROOT/certificate_pool_contact"
set +e
OUTPUTDIR="$OUTPUTDIR" CAL_SAFE="$CAL_SAFE" CERT_NEAR="$CERT_NEAR" CERT_CONTACT="$CERT_CONTACT" GPU0="$GPU0" GPU1="$GPU1" VARIANTS="$variants" \
  bash scripts/calibrate_v48_16_certificate_pool.sh >"$OUTPUTDIR/logs/certificate_recovery_controller.log" 2>&1
rc=$?
set -e
python tools/check_v48_16_learning_gates.py --run "$OUTPUTDIR" --output "$OUTPUTDIR/learning_gates_v48_17.json" --version v48.17-BRIDGE || true
python - "$OUTPUTDIR" "$rc" "$variants" <<'PY'
import json,pathlib,sys,time
root=pathlib.Path(sys.argv[1]); rc=int(sys.argv[2]); variants=sys.argv[3].split(',')
doc={'event':'v48_17_postcheck_recovery_complete','created_unix':time.time(),'variants':variants,'certificate_exit_code':rc,'gate_evaluated':rc in (0,20),'gate_passed':(root/'NEXT_COMMANDS.txt').is_file(),'pipeline_valid':rc in (0,20),'test_roots_read':False}
(root/'V48_17_RECOVERY_COMPLETE.json').write_text(json.dumps(doc,indent=2)+'\n')
PY
exit "$rc"
