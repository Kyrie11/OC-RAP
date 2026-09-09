#!/usr/bin/env bash
# Frozen-checkpoint functional ablations for the V48.111 submission stack.
# No retraining, no recalibration, no audit-only V48.109-111 mechanisms.
set -Eeuo pipefail
REPO="${OCRAP_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

BASE_OUT="${BASE_OUT:-/home/senzeyu2/code/OC-RAP/runs}"
MODEL_RUN="${MODEL_RUN:-$BASE_OUT/ocrap_v48_80_dcp_drfc_bcde_rifa_pistc_main}"
OUT_ROOT="${OUT_ROOT:-$BASE_OUT/ocrap_v48_111_submission_ablations}"
VARIANTS="${VARIANTS:-balanced,precision}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
OCRAP_ROOT="${OCRAP_ROOT:-/data0/senzeyu2/dataset/OCRAP}"
MAX_SCENARIOS="${MAX_SCENARIOS:-0}"
MAX_STEPS="${MAX_STEPS:-40}"
NUM_CANDIDATES="${NUM_CANDIDATES:-24}"
NUM_RECOVERY_OPTIONS="${NUM_RECOVERY_OPTIONS:-12}"
# main = five paper-facing ablations; supplementary adds active-set/route knockouts.
ABLATION_SET="${ABLATION_SET:-main}"
ABLATIONS="${ABLATIONS:-}"

mkdir -p "$OUT_ROOT"
python tools/check_v48_111_runtime_code_contract.py \
  --repo "$REPO" \
  --output "$OUT_ROOT/OC-RAP-v48.111-runtime-code-contract.ablations.json"

# name|config|safe|near|contact|tier|description
ARMS=(
  "no_obs_consistency|configs/ablations/without_observation_kernel.yaml|0|1|1|main|Hidden-root/branch-wise option selection instead of observation-consistent OC-MERO"
  "mean_tail|configs/ablations/without_lower_tail.yaml|0|1|1|main|Weighted mean instead of lower-tail aggregation"
  "no_actuator_projection|configs/ablations/submission_no_actuator_projection.yaml|0|1|1|main|Execute desired recovery without actuator-envelope projection"
  "no_persistent_reentry|configs/ablations/submission_no_persistent_reentry.yaml|0|0|1|main|Remove persistent post-contact re-entry semantics"
  "no_rifa_absolute_admission|configs/ablations/submission_no_rifa_absolute_admission.yaml|1|1|1|main|Remove RIFA absolute-admission set gate while preserving frozen relative/scoring path"
  "no_active_set_alignment|configs/ablations/submission_no_active_set_alignment.yaml|0|1|1|supplementary|Remove active-set alignment repair"
  "no_route_alignment|configs/ablations/submission_no_route_alignment.yaml|0|1|1|supplementary|Remove executable route-alignment semantics"
)

selected() {
  local name="$1" tier="$2"
  if [[ -n "$ABLATIONS" ]]; then
    [[ ",${ABLATIONS}," == *",${name},"* ]]
    return
  fi
  if [[ "$ABLATION_SET" == "all" ]]; then return 0; fi
  if [[ "$ABLATION_SET" == "main" ]]; then [[ "$tier" == "main" ]]; return; fi
  echo "Unsupported ABLATION_SET=$ABLATION_SET (expected main|all or set ABLATIONS=name1,name2)" >&2
  exit 2
}

MANIFEST="$OUT_ROOT/ablation_run_manifest.tsv"
printf 'arm\tconfig\tsafe\tnear\tcontact\ttier\tdescription\n' > "$MANIFEST"

IFS=',' read -r -a variants <<< "$VARIANTS"
for variant_raw in "${variants[@]}"; do
  variant="$(echo "$variant_raw" | xargs)"
  [[ -n "$variant" ]] || continue
  # Validate that every functional knockout starts from the exact deployed owner.
  python tools/check_v48_111_deployable_stack.py \
    --model-run "$MODEL_RUN" --variant "$variant" \
    --output "$OUT_ROOT/V48.111-DEPLOYABLE-STACK-${variant}.json"

  for spec in "${ARMS[@]}"; do
    IFS='|' read -r arm config run_safe run_near run_contact tier description <<< "$spec"
    selected "$arm" "$tier" || continue
    [[ -f "$config" ]] || { echo "Missing ablation config: $config" >&2; exit 30; }
    if ! grep -q "^${arm}[[:space:]]" "$MANIFEST" 2>/dev/null; then
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$arm" "$config" "$run_safe" "$run_near" "$run_contact" "$tier" "$description" >> "$MANIFEST"
    fi
    out="$OUT_ROOT/$arm/$variant"
    echo "[V48.111 ablation] arm=$arm variant=$variant regimes=S${run_safe}/N${run_near}/C${run_contact}"
    CONFIG="$config" \
    MODEL_RUN="$MODEL_RUN" MODEL_VARIANT="$variant" \
    CUDA_DEVICES="$CUDA_DEVICES" OCRAP_ROOT="$OCRAP_ROOT" OUT="$out" \
    MAX_SCENARIOS="$MAX_SCENARIOS" MAX_STEPS="$MAX_STEPS" \
    NUM_CANDIDATES="$NUM_CANDIDATES" NUM_RECOVERY_OPTIONS="$NUM_RECOVERY_OPTIONS" \
    ALLOW_DIAGNOSTIC_RC20=1 \
    RUN_SAFE="$run_safe" RUN_NEAR="$run_near" RUN_CONTACT="$run_contact" \
    SKIP_COMPLETE_REGIMES=true FINALIZE_COMPLETE_JOURNALS=true \
    bash scripts/run_ocrap_three_regime_closed_loop.sh
  done
done

printf '\nAblations complete.\nRoot: %s\nManifest: %s\n' "$OUT_ROOT" "$MANIFEST"
