#!/usr/bin/env bash
set -euo pipefail

# Four ablation groups are submitted together.  Per variant, A/C share GPU0 and
# B/D share GPU1; the observed ~1 GB/task footprint keeps this within two A30s.
# Balanced and Precision run in separate waves to limit CPU/IO contention.
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"; export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
ROOT="${ABLATION_ROOT:-runs/ocrap_v48_14_ablations}"
SOURCE_RUN="${SOURCE_RUN:-runs/ocrap_v48_13_terra_proxy_4801}"
OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
PROTOCOL_ROOT="${PROTOCOL_ROOT:-$OCRAP_ROOT/calibration_v48_14_prism_4814}"
CAL_SAFE="${CAL_SAFE:-$OCRAP_ROOT/calibration_safe}"
GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
mkdir -p "$ROOT/tasks" "$ROOT/logs"
[[ -f "$PROTOCOL_ROOT/CALIBRATION_PROTOCOL_COMPLETE.json" ]] || { echo "prepare protocol first" >&2; exit 2; }

TRAIN_NEAR="$PROTOCOL_ROOT/evidence_adapt_train_near_contact"
TRAIN_CONTACT="$PROTOCOL_ROOT/evidence_adapt_train_contact"
DEV_NEAR="$PROTOCOL_ROOT/evidence_adapt_dev_near_contact"
DEV_CONTACT="$PROTOCOL_ROOT/evidence_adapt_dev_contact"
CERT_NEAR="$PROTOCOL_ROOT/certificate_pool_near_contact"
CERT_CONTACT="$PROTOCOL_ROOT/certificate_pool_contact"
GROUP_INDEX="$ROOT/evidence_adapt_teacher_pcd_index.jsonl"
GROUP_SUMMARY="$ROOT/evidence_adapt_teacher_pcd_index_summary.json"
if [[ ! -f "$GROUP_INDEX" ]]; then
  python tools/build_teacher_pcd_index_v48.py --dataset "$TRAIN_NEAR,$TRAIN_CONTACT" \
    --output "$GROUP_INDEX" --summary-output "$GROUP_SUMMARY" --quality-mode=warn \
    >"$ROOT/logs/build_teacher_index.log" 2>&1
fi

run_task(){
  local group="$1" variant="$2" gpu="$3" adapt="$4" hard_harm="$5" hard_benefit="$6" intra_b="$7" intra_h="$8"
  local out="$ROOT/tasks/${group}_${variant}" source="$SOURCE_RUN/candidates/$variant"
  rm -rf "$out"; mkdir -p "$out/candidates/$variant" "$out/logs"
  if [[ "$adapt" == 0 ]]; then
    local dst="$out/candidates/$variant"
    ln -s "$(realpath --relative-to="$dst" "$source/model_v48_trac_sr")" "$dst/model_v48_trac_sr"
    ln -s "$(realpath --relative-to="$dst" "$source/POLICY_CONTRACT.env")" "$dst/POLICY_CONTRACT.env"
  else
    RUN="$out/candidates/$variant" INIT_CKPT="$source/model_v48_trac_sr/best.pt" VARIANT="$variant" TRAIN_GPU="$gpu" \
      TRAIN_MIX="$TRAIN_NEAR,$TRAIN_CONTACT" VAL_MIX="$DEV_NEAR,$DEV_CONTACT" GROUP_INDEX="$GROUP_INDEX" \
      TRAIN_OCRAP_ROOT="$OCRAP_ROOT" EVAL_OCRAP_ROOT="$OCRAP_ROOT" \
      BATCH_SIZE="${BATCH_SIZE:-72}" NUM_WORKERS="${NUM_WORKERS:-4}" PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}" \
      EVIDENCE_ADAPT_EPOCHS="${EVIDENCE_ADAPT_EPOCHS:-8}" EVIDENCE_ADAPT_PATIENCE="${EVIDENCE_ADAPT_PATIENCE:-3}" \
      ORDINAL_EVIDENCE_HARD_HARM_WEIGHT="$hard_harm" ORDINAL_EVIDENCE_HARD_BENEFIT_WEIGHT="$hard_benefit" \
      ORDINAL_EVIDENCE_INTRAGROUP_BENEFIT_WEIGHT="$intra_b" ORDINAL_EVIDENCE_INTRAGROUP_HARM_WEIGHT="$intra_h" \
        bash scripts/adapt_ocrap_v48_14_prism_variant.sh >"$out/logs/adapt.log" 2>&1
  fi
  set +e
  OUTPUTDIR="$out" CAL_SAFE="$CAL_SAFE" CERT_NEAR="$CERT_NEAR" CERT_CONTACT="$CERT_CONTACT" GPU0="$gpu" GPU1="$gpu" \
    bash scripts/calibrate_v48_14_certificate_pool.sh >"$out/logs/calibrate.log" 2>&1
  local rc=$?; set -e
  if [[ "$rc" != 0 && "$rc" != 20 ]]; then
    printf '{"complete":false,"controller_exit":%s}\n' "$rc" > "$out/TASK_FAILED.json"; return "$rc"
  fi
  python - "$out" "$group" "$variant" "$rc" <<'PY'
import hashlib,json,pathlib,sys,time
out=pathlib.Path(sys.argv[1]); group=sys.argv[2]; variant=sys.argv[3]; rc=int(sys.argv[4])
ck=out/'candidates'/variant/'model_v48_trac_sr'/'best.pt'
required=[ck,out/'candidates'/variant/'calibration'/'gamma_rec_by_bucket_v48.json',
 out/'candidates'/variant/'calibration'/'direct_value_risk_near_v48.json',
 out/'candidates'/variant/'calibration'/'direct_value_risk_contact_v48.json']
missing=[str(p) for p in required if not p.is_file()]
if missing: raise SystemExit('incomplete task: '+','.join(missing))
doc={'complete':True,'group':group,'variant':variant,'certificate_exit':rc,'created_unix':time.time(),
 'checkpoint_sha256':hashlib.sha256(ck.read_bytes()).hexdigest()}
(out/'TASK_COMPLETE.json').write_text(json.dumps(doc,indent=2)+'\n')
PY
}

# group, gpu-slot, adapt, hard-harm, hard-benefit, intra-benefit, intra-harm
ABLATION_SPECS=(
  "A_dedicated_recalibration 0 0 0.0 0.0 0.0 0.0"
  "B_target_adaptation 1 1 0.0 0.0 0.0 0.0"
  "C_hard_harm_adaptation 0 1 2.5 0.75 0.0 0.0"
  "D_full_prism 1 1 2.5 0.75 0.50 1.20"
)
failures=0
for variant in balanced precision; do
  pids=(); labels=()
  for spec in "${ABLATION_SPECS[@]}"; do
    read -r group slot adapt hh hb ib ih <<<"$spec"
    gpu="$GPU0"; [[ "$slot" == 1 ]] && gpu="$GPU1"
    run_task "$group" "$variant" "$gpu" "$adapt" "$hh" "$hb" "$ib" "$ih" &
    pids+=("$!"); labels+=("${group}_${variant}")
  done
  set +e
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}"; rc=$?
    if [[ "$rc" != 0 ]]; then echo "[failed] ${labels[$i]} rc=$rc" >&2; failures=$((failures+1)); fi
  done
  set -e
done
python tools/summarize_v48_14_ablations.py --root "$ROOT" --output "$ROOT/ablation_summary_v48_14.json"
python - "$ROOT" "$failures" <<'PY'
import json,pathlib,sys,time
root=pathlib.Path(sys.argv[1]); failures=int(sys.argv[2])
groups=['A_dedicated_recalibration','B_target_adaptation','C_hard_harm_adaptation','D_full_prism']
expected=[f'{g}_{v}' for v in ('balanced','precision') for g in groups]
missing=[x for x in expected if not (root/'tasks'/x/'TASK_COMPLETE.json').is_file()]
doc={'complete':not missing and failures==0,'version':'v48.14-PRISM','max_concurrent_tasks':4,
 'tasks_per_gpu':2,'expected_tasks':expected,'missing_tasks':missing,'failed_waits':failures,'created_unix':time.time()}
(root/'ABLATIONS_STATUS.json').write_text(json.dumps(doc,indent=2)+'\n')
if doc['complete']:(root/'ABLATIONS_COMPLETE.json').write_text(json.dumps(doc,indent=2)+'\n')
else: raise SystemExit('ablation suite incomplete: '+','.join(missing))
PY
