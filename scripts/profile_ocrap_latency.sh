#!/usr/bin/env bash
# Publication latency for the frozen OC-RAP Main: one process, one GPU, exact final target locks.
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"; cd "$REPO"
BASE_OUT="${BASE_OUT:-/home/senzeyu2/code/OC-RAP/runs}"
MODEL_RUN="${MODEL_RUN:-$BASE_OUT/ocrap_v48_80_dcp_drfc_bcde_rifa_pistc_main}"
TARGET_ROOT="${OCRAP_FINAL_CHARACTERIZATION_OUT:-$BASE_OUT/ocrap_v48_124_final_characterization}"
OUT="${OCRAP_LATENCY_OUT:-$BASE_OUT/ocrap_v48_124_latency_isolated}"
GPU="${LATENCY_GPU:-0}"
REGIME="${1:-all}"
WOMD_ROLE="${WOMD_ROLE:-validation}"
MAX_STEPS="${MAX_STEPS:-40}"
LATENCY_WARMUP_DECISIONS="${LATENCY_WARMUP_DECISIONS:-3}"
for r in safe near contact; do [[ -s "$TARGET_ROOT/target_keys/$r.json" ]] || { echo "missing target lock: $TARGET_ROOT/target_keys/$r.json" >&2; exit 30; }; done
case "$REGIME" in safe|near|contact|all) ;; *) echo "usage: $0 [safe|near|contact|all]" >&2; exit 2;; esac
for variant in balanced precision; do
  rs=0; rn=0; rc=0
  case "$REGIME" in safe) rs=1;; near) rn=1;; contact) rc=1;; all) rs=1; rn=1; rc=1;; esac
  env WOMD_ROLE="$WOMD_ROLE" MODEL_RUN="$MODEL_RUN" MODEL_VARIANT="$variant" \
    OUT="$OUT/ocrap/$variant" CUDA_DEVICES="$GPU" MAX_SCENARIOS=0 MAX_STEPS="$MAX_STEPS" \
    RUN_SAFE="$rs" RUN_NEAR="$rn" RUN_CONTACT="$rc" \
    SAFE_TARGET_KEYS_FILE="$TARGET_ROOT/target_keys/safe.json" \
    NEAR_TARGET_KEYS_FILE="$TARGET_ROOT/target_keys/near.json" \
    CONTACT_TARGET_KEYS_FILE="$TARGET_ROOT/target_keys/contact.json" \
    PROFILE_TIMING=true LATENCY_EXECUTION_CONTRACT=isolated_single_process_single_gpu SKIP_COMPLETE_REGIMES=false \
    LATENCY_WARMUP_DECISIONS="$LATENCY_WARMUP_DECISIONS" \
    INCLUDE_SCENES_IN_RESULT=false RESULT_SCENE_DETAIL=metrics SCENE_JOURNAL_DETAIL=metrics \
    RENDER_SAFE=false RENDER_NEAR=false RENDER_CONTACT=false \
    bash scripts/run_ocrap_three_regime_evaluation.sh
done
python - "$OUT" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]); rows={}
for variant in ('balanced','precision'):
    rows[variant]={}
    for regime in ('safe','near','contact'):
        p=root/'ocrap'/variant/regime/'closed_loop_ocrap.json'
        if p.is_file():
            d=json.loads(p.read_text()); rows[variant][regime]={
                'num_scenes':d.get('num_scenes'),
                'execution_contract':(d.get('timing') or {}).get('execution_contract'),
                'steady_state_deployed_planner_s':(d.get('timing') or {}).get('steady_state_deployed_planner_s'),
            }
out={'schema':'ocrap-final-isolated-latency-v1','status':'COMPLETE','jobs_per_gpu':1,'processes':1,'rows':rows}
(root/'LATENCY_INDEX.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print(json.dumps({'event':'ocrap_isolated_latency_complete','output':str(root/'LATENCY_INDEX.json')}))
PY
