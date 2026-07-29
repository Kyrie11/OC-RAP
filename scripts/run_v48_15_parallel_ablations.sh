#!/usr/bin/env bash
set -euo pipefail

# Four ablations run concurrently in each variant wave.  A/C share GPU0 and
# B/D share GPU1; the measured task footprint is ~1 GB, so this uses the two
# 24-GB A30s without launching all eight CPU/IO-heavy jobs at once.
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"; export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
ROOT="${ABLATION_ROOT:-runs/ocrap_v48_15_ablations}"
SOURCE_RUN="${SOURCE_RUN:-runs/ocrap_v48_13_terra_proxy_4801}"
OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
PROTOCOL_ROOT="${PROTOCOL_ROOT:-$OCRAP_ROOT/calibration_v48_14_prism_4814}"
CAL_SAFE="${CAL_SAFE:-$OCRAP_ROOT/calibration_safe}"
GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
mkdir -p "$ROOT/tasks" "$ROOT/logs"
[[ -f "$PROTOCOL_ROOT/CALIBRATION_PROTOCOL_COMPLETE.json" ]] || { echo "prepare dedicated protocol first" >&2; exit 2; }
TRAIN_NEAR="$PROTOCOL_ROOT/evidence_adapt_train_near_contact"; TRAIN_CONTACT="$PROTOCOL_ROOT/evidence_adapt_train_contact"
DEV_NEAR="$PROTOCOL_ROOT/evidence_adapt_dev_near_contact"; DEV_CONTACT="$PROTOCOL_ROOT/evidence_adapt_dev_contact"
CERT_NEAR="$PROTOCOL_ROOT/certificate_pool_near_contact"; CERT_CONTACT="$PROTOCOL_ROOT/certificate_pool_contact"
GROUP_INDEX="$ROOT/evidence_adapt_teacher_pcd_index.jsonl"; GROUP_SUMMARY="$ROOT/evidence_adapt_teacher_pcd_index_summary.json"
if [[ ! -f "$GROUP_INDEX" ]]; then
  python tools/build_teacher_pcd_index_v48.py --dataset "$TRAIN_NEAR,$TRAIN_CONTACT" \
    --output "$GROUP_INDEX" --summary-output "$GROUP_SUMMARY" --quality-mode=warn \
    >"$ROOT/logs/build_teacher_index.log" 2>&1
fi

run_task(){
  local group="$1" variant="$2" gpu="$3" mode="$4"
  local out="$ROOT/tasks/${group}_${variant}" source="$SOURCE_RUN/candidates/$variant"
  rm -rf "$out"; mkdir -p "$out/candidates/$variant" "$out/logs"
  case "$mode" in
    source)
      local dst="$out/candidates/$variant"
      ln -s "$(realpath --relative-to="$dst" "$source/model_v48_trac_sr")" "$dst/model_v48_trac_sr"
      ln -s "$(realpath --relative-to="$dst" "$source/POLICY_CONTRACT.env")" "$dst/POLICY_CONTRACT.env"
      ;;
    full_adapter)
      RUN="$out/candidates/$variant" INIT_CKPT="$source/model_v48_trac_sr/best.pt" VARIANT="$variant" TRAIN_GPU="$gpu" \
      TRAIN_MIX="$TRAIN_NEAR,$TRAIN_CONTACT" VAL_MIX="$DEV_NEAR,$DEV_CONTACT" GROUP_INDEX="$GROUP_INDEX" \
      TRAIN_OCRAP_ROOT="$OCRAP_ROOT" EVAL_OCRAP_ROOT="$OCRAP_ROOT" BATCH_SIZE="${BATCH_SIZE:-72}" \
      NUM_WORKERS="${NUM_WORKERS:-4}" PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}" \
      EVIDENCE_ADAPT_EPOCHS="${EVIDENCE_ADAPT_EPOCHS:-8}" EVIDENCE_ADAPT_PATIENCE="${EVIDENCE_ADAPT_PATIENCE:-3}" \
      ORDINAL_EVIDENCE_HARD_HARM_WEIGHT=2.50 ORDINAL_EVIDENCE_HARD_BENEFIT_WEIGHT=0.75 \
      ORDINAL_EVIDENCE_INTRAGROUP_BENEFIT_WEIGHT=0.50 ORDINAL_EVIDENCE_INTRAGROUP_HARM_WEIGHT=1.20 \
        bash scripts/adapt_ocrap_v48_14_prism_variant.sh >"$out/logs/adapt.log" 2>&1
      ;;
    tiny_plain|tiny_balanced)
      hh=0.0; hb=0.0
      [[ "$mode" == tiny_balanced ]] && { hh=1.0; hb=1.0; }
      RUN="$out/candidates/$variant" INIT_CKPT="$source/model_v48_trac_sr/best.pt" VARIANT="$variant" TRAIN_GPU="$gpu" \
      TRAIN_MIX="$TRAIN_NEAR,$TRAIN_CONTACT" VAL_MIX="$DEV_NEAR,$DEV_CONTACT" GROUP_INDEX="$GROUP_INDEX" \
      TRAIN_OCRAP_ROOT="$OCRAP_ROOT" EVAL_OCRAP_ROOT="$OCRAP_ROOT" BATCH_SIZE="${BATCH_SIZE:-72}" \
      NUM_WORKERS="${NUM_WORKERS:-4}" PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}" \
      EVIDENCE_ADAPT_EPOCHS="${EVIDENCE_ADAPT_EPOCHS:-10}" EVIDENCE_ADAPT_PATIENCE="${EVIDENCE_ADAPT_PATIENCE:-4}" \
      ORDINAL_EVIDENCE_HARD_HARM_WEIGHT="$hh" ORDINAL_EVIDENCE_HARD_BENEFIT_WEIGHT="$hb" \
        bash scripts/adapt_ocrap_v48_15_prism_cc_variant.sh >"$out/logs/adapt.log" 2>&1
      ;;
    *) echo "unknown mode $mode" >&2; return 2;;
  esac
  set +e
  OUTPUTDIR="$out" CAL_SAFE="$CAL_SAFE" CERT_NEAR="$CERT_NEAR" CERT_CONTACT="$CERT_CONTACT" \
  GPU0="$gpu" GPU1="$gpu" VARIANTS="$variant" \
    bash scripts/calibrate_v48_14_certificate_pool.sh >"$out/logs/calibrate.log" 2>&1
  local rc=$?
  set -e
  if [[ "$rc" != 0 && "$rc" != 20 ]]; then
    printf '{"complete":false,"controller_exit":%s}\n' "$rc" > "$out/TASK_FAILED.json"
    return "$rc"
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

SPECS=(
  "A_source_dedicated source 0"
  "B_full_adapter_prism full_adapter 1"
  "C_tiny_calibrator tiny_plain 0"
  "D_full_prism_cc tiny_balanced 1"
)
failures=0
for variant in balanced precision; do
  pids=(); labels=()
  for spec in "${SPECS[@]}"; do
    read -r group mode slot <<<"$spec"; gpu="$GPU0"; [[ "$slot" == 1 ]] && gpu="$GPU1"
    run_task "$group" "$variant" "$gpu" "$mode" & pids+=("$!"); labels+=("${group}_${variant}")
  done
  set +e
  for i in "${!pids[@]}"; do wait "${pids[$i]}"; rc=$?; if [[ "$rc" != 0 ]]; then echo "[failed] ${labels[$i]} rc=$rc" >&2; failures=$((failures+1)); fi; done
  set -e
done
python tools/summarize_v48_14_ablations.py --root "$ROOT" --output "$ROOT/ablation_summary_v48_15.json"
python - "$ROOT" "$failures" <<'PY'
import json,pathlib,sys,time
root=pathlib.Path(sys.argv[1]); failures=int(sys.argv[2])
groups=['A_source_dedicated','B_full_adapter_prism','C_tiny_calibrator','D_full_prism_cc']
expected=[f'{g}_{v}' for v in ('balanced','precision') for g in groups]
missing=[x for x in expected if not (root/'tasks'/x/'TASK_COMPLETE.json').is_file()]
doc={'complete':not missing and failures==0,'version':'v48.15-PRISM-CC','max_concurrent_tasks':4,'tasks_per_gpu':2,
 'expected_tasks':expected,'missing_tasks':missing,'failed_waits':failures,'created_unix':time.time()}
(root/'ABLATIONS_STATUS.json').write_text(json.dumps(doc,indent=2)+'\n')
if doc['complete']:(root/'ABLATIONS_COMPLETE.json').write_text(json.dumps(doc,indent=2)+'\n')
else: raise SystemExit('ablation suite incomplete: '+','.join(missing))
PY
