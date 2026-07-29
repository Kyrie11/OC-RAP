#!/usr/bin/env bash
set -euo pipefail

# v48.13 TERRA ablations. Four ablations run concurrently per variant wave.
# With two A30 cards, A/C share GPU0 and B/D share GPU1; each task remains a
# single-GPU process. Precision starts only after the balanced wave completes,
# limiting CPU and dataset I/O contention while using the available VRAM.

REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"; export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
ROOT="${ABLATION_ROOT:-runs/ocrap_v48_13_ablations}"
ASSET_ROOT="${ASSET_ROOT:-runs/ocrap_v48_8_shared_assets_4801}"
INIT_CKPT="${INIT_CKPT:?Set INIT_CKPT}"
GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"
mkdir -p "$ROOT/tasks" "$ROOT/logs"

COMMON=(
  "TRAIN_OCRAP_ROOT=${TRAIN_OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
  "EVAL_OCRAP_ROOT=${EVAL_OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
  "INIT_CKPT=$INIT_CKPT"
  "INIT_CKPT_BALANCED=${INIT_CKPT_BALANCED:-$INIT_CKPT}"
  "INIT_CKPT_PRECISION=${INIT_CKPT_PRECISION:-$INIT_CKPT}"
  "CALIBRATION_MODE=proxy_val_split" "CALIBRATION_FRACTION=${CALIBRATION_FRACTION:-0.50}"
  "CALIBRATION_SEED=${CALIBRATION_SEED:-4801}"
  "BUILD_TRAIN=0" "BUILD_CALIBRATION=0" "STRICT_TRAIN_DATA_GATE=0"
  "REUSE_TEACHER_INDEX=1" "AUTO_ENSURE_MANIFESTS=0"
  "PREBUILT_SPLIT_ROOT=$ASSET_ROOT/dataset_splits" "REUSE_PREBUILT_SPLITS=1"
  "SHARED_GROUP_INDEX=$ASSET_ROOT/teacher_pcd_train_index.jsonl"
  "SHARED_GROUP_SUMMARY=$ASSET_ROOT/teacher_pcd_train_index_summary.json"
  "BATCH_SIZE=${BATCH_SIZE:-72}" "NUM_WORKERS=${ABLATION_NUM_WORKERS:-3}"
  "PREFETCH_FACTOR=${PREFETCH_FACTOR:-2}"
  "PREFERENCE_EPOCHS=${PREFERENCE_EPOCHS:-14}" "PREFERENCE_PATIENCE=${PREFERENCE_PATIENCE:-4}"
  "EVIDENCE_EPOCHS=${EVIDENCE_EPOCHS:-12}" "EVIDENCE_PATIENCE=${EVIDENCE_PATIENCE:-4}"
  "FOREGROUND=1" "EXACT_TEACHER_PCD=true" "GROUP_DRO_WEIGHT=0"
  "TRAIN_SCRIPT=scripts/train_ocrap_v48_13_terra.sh"
  "RISK_SOURCE=ordinal_evidence" "CONDITIONAL_RECOVERY_RANKING=true"
  "PROPOSAL_TOP_K=${PROPOSAL_TOP_K:-3}"
  "MACRO_CONSTRAINT_MODE=opportunity_normalized"
  "MAX_MACRO_EXCESS_SHARE=${MAX_MACRO_EXCESS_SHARE:-0.15}"
)

run_task() {
  local group="$1" variant="$2" gpu="$3" proposal_w="$4" proposal_evidence_w="$5" intra_b="$6" intra_h="$7" rerank="$8"
  local out="$ROOT/tasks/${group}_${variant}"
  if [[ -f "$out/TASK_COMPLETE.json" ]]; then echo "[skip] $group $variant"; return 0; fi
  mkdir -p "$out"
  set +e
  env "${COMMON[@]}" OUTPUTDIR="$out" VARIANTS="$variant" GPU0="$gpu" GPU1="$gpu" \
    PREFERENCE_PROPOSAL_TOPK_WEIGHT="$proposal_w" \
    ORDERED_EVIDENCE_PROPOSAL_TOPK_WEIGHT="$proposal_evidence_w" \
    ORDERED_EVIDENCE_INTRAGROUP_BENEFIT_WEIGHT="$intra_b" \
    ORDERED_EVIDENCE_INTRAGROUP_HARM_WEIGHT="$intra_h" \
    EVIDENCE_RERANK_TOP_K="$rerank" \
    bash run_v48_two_gpu_fast_commands.txt >"$ROOT/logs/${group}_${variant}.log" 2>&1
  rc=$?; set -e
  echo "$rc" >"$out/controller.exit_code"
  if [[ "$rc" != 0 && "$rc" != 20 ]]; then
    printf '{"complete":false,"group":"%s","variant":"%s","controller_exit":%s}\n' "$group" "$variant" "$rc" >"$out/TASK_FAILED.json"
    return "$rc"
  fi
  python - "$out" "$group" "$variant" "$rc" <<'PY'
import hashlib,json,pathlib,sys,time
out=pathlib.Path(sys.argv[1]); group=sys.argv[2]; variant=sys.argv[3]; rc=int(sys.argv[4])
base=out/'candidates'/variant
required=[base/'model_v48_trac_sr'/'best.pt',base/'calibration'/'direct_value_risk_near_v48.json',base/'calibration'/'direct_value_risk_contact_v48.json',base/'POLICY_CONTRACT.env']
missing=[str(p) for p in required if not p.is_file()]
if missing: raise SystemExit('incomplete task: '+','.join(missing))
ck=required[0]
(out/'TASK_COMPLETE.json').write_text(json.dumps({'complete':True,'group':group,'variant':variant,'controller_exit':rc,'created_unix':time.time(),'checkpoint_sha256':hashlib.sha256(ck.read_bytes()).hexdigest()},indent=2)+'\n')
PY
}

# A: contract-fixed top1 baseline.
# B: proposal training only, still deploy top1.
# C: proposal-distribution evidence and reranking on the old tournament.
# D: full TERRA.
GROUPS=(
  "A_top1_contract 0 0.0 0.0 0.0 0.0 false"
  "B_proposal_only 1 1.25 0.0 0.0 0.0 false"
  "C_evidence_rerank 0 0.0 3.0 0.80 1.80 true"
  "D_full_terra 1 1.25 3.0 0.80 1.80 true"
)

failures=0
for variant in balanced precision; do
  pids=(); labels=()
  for spec in "${GROUPS[@]}"; do
    read -r group slot proposal_w proposal_evidence_w intra_b intra_h rerank <<<"$spec"
    gpu="$GPU0"; [[ "$slot" == 1 ]] && gpu="$GPU1"
    run_task "$group" "$variant" "$gpu" "$proposal_w" "$proposal_evidence_w" "$intra_b" "$intra_h" "$rerank" &
    pids+=("$!"); labels+=("${group}_${variant}")
  done
  set +e
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}"; rc=$?
    if [[ "$rc" != 0 ]]; then echo "[failed] ${labels[$i]} rc=$rc" >&2; failures=$((failures+1)); fi
  done
  set -e
done

python tools/summarize_v48_13_ablations.py --root "$ROOT" --output "$ROOT/ablation_summary_v48_13.json"
python - "$ROOT" "$failures" <<'PY'
import json,pathlib,sys,time
root=pathlib.Path(sys.argv[1]); failures=int(sys.argv[2])
groups=['A_top1_contract','B_proposal_only','C_evidence_rerank','D_full_terra']
expected=[f'{g}_{v}' for v in ('balanced','precision') for g in groups]
missing=[x for x in expected if not (root/'tasks'/x/'TASK_COMPLETE.json').is_file()]
doc={'complete':not missing and failures==0,'version':'v48.13','max_concurrent_tasks':4,'tasks_per_gpu':2,'expected_tasks':expected,'missing_tasks':missing,'failed_waits':failures,'created_unix':time.time()}
(root/'ABLATIONS_STATUS.json').write_text(json.dumps(doc,indent=2)+'\n')
if doc['complete']:(root/'ABLATIONS_COMPLETE.json').write_text(json.dumps(doc,indent=2)+'\n')
else: raise SystemExit('ablation suite incomplete: '+','.join(missing))
PY
