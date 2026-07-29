# OC-RAP v48.13 TERRA Change Manifest

## Algorithm

- `src/ocrap/models/losses.py`
  - Added differentiable teacher-acceptable top-k proposal inclusion loss.
  - Added proposal-distribution ordered three-state NLL with rank-decayed weights.
  - Added same-group beneficial/harmful counterfactual evidence comparisons.
  - Retained exact teacher-PCD and disabled v48.12 all-pairs/cross-group objectives in the main TERRA script.
- `src/ocrap/cli/train.py`
  - Added proposal/evidence loss configuration plumbing.
  - Made certificate validation and early stopping follow the same top-k evidence-reranking policy used by calibration/runtime.
  - Added certificate-candidate regret, top-1 and evidence-margin statistics.
- `configs/default.yaml`
  - Added TERRA proposal/evidence controls and deployment-aligned policy-metric defaults.
- `scripts/train_ocrap_v48_trac_sr.sh`
  - Added environment-to-config mappings for all TERRA loss and policy-metric settings.

## Calibration and deployment contract

- `tools/calibrate_policy_risk_v48.py`
  - Added `--proposal-top-k` and `--evidence-rerank-top-k`.
  - Added frozen proposal then evidence rerank selection.
  - Added positive-group proposal hit metrics and proposal-evidence top-1 diagnostics.
  - Stored proposal/rerank selector overrides in calibration JSON.
- `src/ocrap/planning/selector.py`
  - Added top-k proposal and evidence-rerank runtime path with abstention.
- `src/ocrap/evaluation/baselines.py`
  - Passed proposal/rerank selector configuration through offline and closed-loop evaluation.
- `src/ocrap/config/defaults.py`
  - Added selector defaults for proposal size and evidence reranking.
- `run_v48_two_gpu_fast_commands.txt`
  - Sources immutable per-variant `POLICY_CONTRACT.env` before calibration.
  - Passes proposal/rerank settings to both Near and Contact calibration.

## New scripts and tools

- `scripts/train_ocrap_v48_13_terra.sh`
  - Two-stage proposal/evidence training and immutable staged checkpoint contract.
- `scripts/recalibrate_v48_13_multiseed.sh`
  - Fixed-checkpoint 4801/4802/4803 calibration with contract parity.
- `scripts/recalibrate_v48_13_on_dedicated_set.sh`
  - Dedicated Safe/Near/Contact recalibration without retraining.
- `scripts/run_v48_13_parallel_ablations.sh`
  - Four concurrent ablations per variant wave; two processes per A30.
- `tools/check_v48_13_learning_gates.py`
  - Layered proposal, proposal-evidence and Natural-gate diagnostics.
- `tools/summarize_v48_13_ablations.py`
  - v48.13 ablation aggregation.

## Engineering corrections

- Restored missing `run_v47_two_gpu_fast_commands.txt` required by historical regression tests.
- Unified main, multi-seed and dedicated conditional-harm/macro/policy contracts.
- Added Stage-E checkpoint selection that evaluates the same candidate distribution as deployment.

## Tests

- `tests/test_v48_13_terra.py`
  - Proposal inclusion behavior.
  - Same-group ordered harm separation.
  - Evidence rerank restricted to frozen proposal.
  - Controller contract propagation.
  - Stage-E checkpoint/deployment contract parity.
