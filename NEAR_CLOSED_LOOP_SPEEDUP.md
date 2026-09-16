# Near-Contact closed-loop speed fix

## Root cause

`run_external_baselines_near.sh` previously defaulted to `CL_LABEL_MODE=selected` and `CL_AUDIT_EVERY_N_STEPS=0`.
In `closed_loop_runner.py`, the cadence was parsed as `max(1, int(value or 1))`, so `0` became `1`.
As a result, every replan step performed a selected-candidate teacher audit after policy selection.
That audit calls `build_labeled_samples_for_candidate_indices(..., compact_audit=True)`, which still must generate counterfactual futures and compute recovery teacher margins/Waymax recovery rollouts over the recovery-option library. It is diagnostic-only and is not on the deployed action-selection path.

## Safe changes

1. Near-Contact publication closed loop now defaults to `CL_LABEL_MODE=fast`, matching Safe/Contact and the external-main-table contract.
2. `closed_loop.audit_every_n_steps <= 0` is now a real audit-disable switch instead of silently becoming every-step audit.
3. Near-Contact enables `closed_loop.fast_waymax_history=true`. This direct state-to-history path already has an exact-equivalence regression test against the legacy RawScenario reconstruction path.
4. Teacher diagnostics remain available explicitly with `CL_LABEL_MODE=selected CL_AUDIT_EVERY_N_STEPS=1` (or another positive cadence).

These changes do not alter the frozen target set, WOMD source, candidate count, recovery-option count, external policy selector, Waymax stepping, physical metrics, target-lock contract, or main Near comparison-table schema. They remove optional post-selection teacher work from the publication accuracy path and reduce state-history conversion overhead.

## Recommended rerun

Use a fresh final output root because changing closed-loop configuration changes the run fingerprint and should not be mixed with partial artifacts from the old selected-audit run:

```bash
FINAL_EXTERNAL_BASELINE_OUT="$BASE_OUT/external_baselines_v48_124_final_v2_fast" \
PRETRAINED_BASELINE_ROOT="$BASE_OUT/external_baselines_v48_111" \
bash scripts/run_final_external_baselines.sh near
```

Optional diagnostic rerun (separate output root):

```bash
FINAL_EXTERNAL_BASELINE_OUT="$BASE_OUT/external_baselines_v48_124_near_teacher_audit" \
PRETRAINED_BASELINE_ROOT="$BASE_OUT/external_baselines_v48_111" \
CL_LABEL_MODE=selected CL_AUDIT_EVERY_N_STEPS=1 \
bash scripts/run_final_external_baselines.sh near
```

## Validation

The patched tree passes the complete repository test suite: `299 passed`.
