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

OUT="${OUT:?OUT is required; point it to the v48.24 dedicated run}"
OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
PROTOCOL_ROOT="${PROTOCOL_ROOT:-$OCRAP_ROOT/calibration_v48_14_prism_4814}"
GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
DEV_NEAR="${DEV_NEAR:-$PROTOCOL_ROOT/evidence_adapt_dev_near_contact}"
DEV_CONTACT="${DEV_CONTACT:-$PROTOCOL_ROOT/evidence_adapt_dev_contact}"
ROOT="$OUT/dev_shadow_closed_loop"
mkdir -p "$ROOT"
python - "$ROOT" "$DEV_NEAR" "$DEV_CONTACT" <<'PY'
import json,pathlib,sys,time
root=pathlib.Path(sys.argv[1]); root.mkdir(parents=True,exist_ok=True)
(root/'DIAGNOSTIC_ONLY_NO_PAPER.json').write_text(json.dumps({
 'event':'v48_24_dev_shadow_closed_loop','created_unix':time.time(),
 'near_dataset':sys.argv[2],'contact_dataset':sys.argv[3],
 'uses_adaptation_dev_only':True,'uses_certificate_pool':False,
 'uses_test_or_stress':False,'paper_result':False,
 'purpose':'diagnose offline-label/closed-loop transfer after Natural-gate rejection'
},indent=2)+'\n')
PY

run_variant() {
  local variant="$1" gpu="$2"
  local base="$OUT/candidates/$variant"
  [[ -f "$base/model_v48_trac_sr/best.pt" ]] || { echo "missing $variant checkpoint" >&2; return 30; }
  [[ -d "$base/calibration" ]] || { echo "missing $variant calibration" >&2; return 30; }
  local run="$ROOT/$variant"
  mkdir -p "$run"
  # Run Near then Contact sequentially on one A30. Balanced and Precision run
  # concurrently on separate cards, so the diagnostic never oversubscribes VRAM.
  DEV_SHADOW_DIAGNOSTIC=1 BASE_RUN="$base" RUN="$run" OCRAP_ROOT="$OCRAP_ROOT" \
    NEAR_TEST="$DEV_NEAR" CONTACT_TEST="$DEV_CONTACT" BUCKET_SPLIT=evidence_adapt_dev \
    RUN_OFFLINE_EVAL=0 RUN_AUDITS=1 RUN_SAFE_CLOSED_LOOP=0 RUN_SCALAR_BASELINES=1 \
    RUN_NEAR_AUDIT=1 RUN_CONTACT_AUDIT=0 GPU_NEAR="$gpu" GPU_CONTACT="$gpu" \
    AUDIT_TARGETS="${AUDIT_TARGETS:-16}" AUDIT_LABELS="${AUDIT_LABELS:-192}" \
    AUDIT_MAX_ROLLOUTS="${AUDIT_MAX_ROLLOUTS:-8}" AUDIT_MAX_STEPS="${AUDIT_MAX_STEPS:-24}" \
    CL_RESUME="${CL_RESUME:-0}" bash scripts/run_ocrap_v48_trac_sr.sh \
    >"$run/near_controller.log" 2>&1
  DEV_SHADOW_DIAGNOSTIC=1 BASE_RUN="$base" RUN="$run" OCRAP_ROOT="$OCRAP_ROOT" \
    NEAR_TEST="$DEV_NEAR" CONTACT_TEST="$DEV_CONTACT" BUCKET_SPLIT=evidence_adapt_dev \
    RUN_OFFLINE_EVAL=0 RUN_AUDITS=1 RUN_SAFE_CLOSED_LOOP=0 RUN_SCALAR_BASELINES=1 \
    RUN_NEAR_AUDIT=0 RUN_CONTACT_AUDIT=1 GPU_NEAR="$gpu" GPU_CONTACT="$gpu" \
    AUDIT_TARGETS="${AUDIT_TARGETS:-16}" AUDIT_LABELS="${AUDIT_LABELS:-192}" \
    AUDIT_MAX_ROLLOUTS="${AUDIT_MAX_ROLLOUTS:-8}" AUDIT_MAX_STEPS="${AUDIT_MAX_STEPS:-24}" \
    CL_RESUME="${CL_RESUME:-0}" bash scripts/run_ocrap_v48_trac_sr.sh \
    >"$run/contact_controller.log" 2>&1
  python tools/compare_paired_closed_loop.py \
    "$run/audit_near_contact_selected_topk_v48_scalar.json" \
    "$run/audit_near_contact_selected_topk_v48_v48.json" \
    --output "$run/near_dev_shadow_paired.json" --bootstrap "${SHADOW_BOOTSTRAP:-2000}"
  python tools/compare_paired_closed_loop.py \
    "$run/audit_contact_selected_topk_v48_scalar.json" \
    "$run/audit_contact_selected_topk_v48_v48.json" \
    --output "$run/contact_dev_shadow_paired.json" --bootstrap "${SHADOW_BOOTSTRAP:-2000}"
  set +e
  python tools/check_v48_24_regime_targets.py "$run" \
    --near-paired "$run/near_dev_shadow_paired.json" \
    --contact-paired "$run/contact_dev_shadow_paired.json" \
    >"$run/dev_shadow_target_check.log" 2>&1
  local target_rc=$?
  set -e
  printf '%s\n' "$target_rc" > "$run/dev_shadow_target_check.rc"
}

run_variant balanced "$GPU0" & p0=$!
run_variant precision "$GPU1" & p1=$!
set +e
wait "$p0"; r0=$?
wait "$p1"; r1=$?
set -e
python - "$ROOT" "$r0" "$r1" <<'PY'
import json,pathlib,sys,time
root=pathlib.Path(sys.argv[1]); r0=int(sys.argv[2]); r1=int(sys.argv[3])
doc={'event':'v48_24_dev_shadow_complete','created_unix':time.time(),
     'balanced_exit':r0,'precision_exit':r1,'complete':r0==0 and r1==0,
     'uses_test_or_stress':False,'paper_result':False}
(root/'DEV_SHADOW_COMPLETE.json').write_text(json.dumps(doc,indent=2)+'\n')
if not doc['complete']: raise SystemExit(30)
PY
