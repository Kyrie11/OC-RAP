# OC-RAP: Observation-Consistent Recoverability Planning

This repository is a cleaned OC-RAP implementation migrated from the earlier ReCAP prototype.  The old package name `recap` is retained for import compatibility, but the main algorithmic path is now:

```text
BEV history + ego state + route command
  -> action prefix proposals
  -> executable recovery affordance tokens
  -> ReCoT root-shared counterfactual evidence
  -> OC-MERO observation-consistent recoverability profiles
  -> CRISP calibrated recovery-preserving selector
```

The migration removes the old CARE/MERO core semantics from the deployable path.  Backward-compatible aliases remain (`CARE = ReCoT`, legacy `MERO` helpers), but the main outputs and labels are signed recovery margins, observation-consistent witnesses, and four-offset CRISP calibration.

## What changed for OC-RAP

Key paper-alignment fixes implemented in this version:

- Recovery specification margin is `max(G_no, G_mr, G_post)`, not `min(...)` across mutually exclusive regimes.
- Dataset labels include `spec_margin_star`, `spec_id_star`, `margin_option`, `g_star`, `y_star`, `obs_class`, `obs_equiv`, `beta_star`, `witness_oc`, `Y_oc`, and `R_star` computed from `Y_oc`.
- OC-MERO uses a μ-weighted existential operator:
  `sigmoid(tau_R * logsumexp(log_mu + v/tau_R) - c_R)`, with default `c_R=0.0`.
- ReCoT rejects oracle-only keys in forward: labels, root-mode seed params, observation classes, witnesses, teacher success, and future data are loss/eval only.
- CRISP consumes all four offsets: `q_R`, `q_H`, `q_delta`, and `q_C`.
- Recovery option generation distinguishes relaxed validity from teacher success and preserves one semantic token per required tag when `L` allows.
- Offline metrics separate deployable OC-RAP recovery from oracle option-max diagnostics.


## Alignment review and fixes in this zip

This optimized zip includes `OC_RAP_ALIGNMENT_REVIEW.md`, which records the paper/code audit, remaining simplifications, dataset-generation caveats, and deployability concerns.  In this pass, the teacher-label builder was changed so observation-consistent labels use one observable post-prefix signature per `(action, root mode)` instead of collapsing all modes to the same ego-only signature.  Real MetaDrive rollouts now store compact visible-actor observations in `RolloutTrace`, and CRISP rule labels are reduced through the observation-consistent witness rather than being polluted by invalid/padded recovery tokens.

For paper-style datasets, prefer:

```bash
python scripts/build_teacher_labels.py \
  --config configs/dataset_metadrive.yaml \
  --split train \
  --root-dir data/recap/roots_raw \
  --bev-dir data/recap/bev/train.zarr \
  --rollout-backend metadrive \
  --scenario-dir /path/to/scenarionet_database \
  --output data/recap/train.zarr
```

Use `--disable-root-time-replay`, `--disable-root-alignment-check`, or `--allow-temporal-root-rollout` only for diagnostics.  They weaken assumptions needed for paper-final teacher labels.  Synthetic roots remain useful for smoke tests only.

## Installation

```bash
pip install -r requirements.txt
pip install -e .
pytest -q
```

The test suite includes synthetic checks for observation-consistent witness selection, no oracle existential leakage, μ-weighted duplicate invariance, model forward leakage rejection, selector behavior, and recovery-token validity semantics.

## Dataset and training pipeline

### 1. Collect roots

Synthetic smoke-test roots:

```bash
python scripts/collect_roots.py \
  --config configs/dataset_metadrive_diagnostic.yaml \
  --output data/debug/roots
```

Real MetaDrive/ScenarioNet roots use the existing collectors:

```bash
python scripts/collect_metadrive_roots.py \
  --config configs/dataset_metadrive.yaml \
  --output data/recap/roots_raw
```

For paper-final real teacher labels, avoid temporal root leakage unless the runner restores the exact root tick.  The label builder fails fast by default when real temporal roots are unsafe.

### 2. Rasterize BEV

```bash
python scripts/rasterize_bev.py \
  --root-dir data/recap/roots_raw \
  --split train \
  --bev-config configs/bev_256.yaml \
  --channels compact \
  --history-steps 10 \
  --num-workers 8 \
  --output data/recap/bev/train.zarr
```

### 3. Build OC-RAP teacher labels

```bash
python scripts/build_teacher_labels.py \
  --config configs/dataset_metadrive.yaml \
  --split train \
  --root-dir data/recap/roots_raw \
  --bev-dir data/recap/bev/train.zarr \
  --output data/recap/train.zarr

python scripts/build_teacher_labels.py \
  --config configs/dataset_metadrive.yaml \
  --split calib \
  --root-dir data/recap/roots_raw \
  --bev-dir data/recap/bev/calib.zarr \
  --output data/recap/calib.zarr

python scripts/build_teacher_labels.py \
  --config configs/dataset_metadrive.yaml \
  --split test \
  --root-dir data/recap/roots_raw \
  --bev-dir data/recap/bev/test.zarr \
  --output data/recap/test.zarr
```

The builder writes both deployable OC labels and explicit oracle diagnostics.  Use `R_star`/`Y_oc`/`witness_oc` for OC-RAP metrics; use `Y_action`/`witness_raw_oracle` only for ablation diagnostics.

### 4. Train action proposal and ReCoT

The legacy script name remains available, but it now imports the ReCoT/OC-RAP model path through compatibility wrappers:

```bash
python scripts/train_action_proposal.py \
  --config configs/train_action_proposal.yaml \
  --dataset data/recap/train.zarr \
  --output checkpoints/action_proposal

python scripts/train_care.py \
  --config configs/train_care.yaml \
  --dataset data/recap/train.zarr \
  --proposal-checkpoint checkpoints/action_proposal/best.pt \
  --output checkpoints/recot
```

When writing new experiments, prefer module names `recap.models.recot`, `recap.models.oc_mero`, and `recap.models.selector`.

### 5. Calibrate CRISP

```bash
python scripts/calibrate.py \
  --config configs/train_care.yaml \
  --dataset data/recap/calib.zarr \
  --checkpoint checkpoints/recot/best.pt \
  --split calib \
  --output outputs/calibration
```

`q_values.json` now contains `q_R`, `q_H`, `q_delta`, `q_C`, thresholds, UCBs, and split metadata.  Calibration is selected-action based rather than merely checking whether any admitted set member is bad.

### 6. Offline evaluation

```bash
python scripts/offline_eval.py \
  --config configs/train_care.yaml \
  --dataset data/recap/test.zarr \
  --checkpoint checkpoints/recot/best.pt \
  --calibration outputs/calibration/q_values.json \
  --method ours \
  --output outputs/offline/ocrap
```

The evaluation code reports deployable OC-RAP recovery metrics separately from oracle option-max diagnostics.

### 7. Closed-loop evaluation

```bash
python scripts/eval_closed_loop.py \
  --config configs/eval_closed_loop.yaml \
  --dataset data/recap/test.zarr \
  --checkpoint checkpoints/recot/best.pt \
  --calibration outputs/calibration/q_values.json \
  --method ours \
  --split test \
  --output outputs/eval/ocrap
```

If the simulator backend is unavailable, closed-loop evaluation falls back to offline selected-action evaluation.

## OC-RAP dataset schema

Each training sample should expose these deployable inputs:

```text
bev, ego_info, route_command
actions_states, actions_controls, actions_params, action_mask
token_states_ref, token_controls_ref, token_params, token_anchor, token_hard_shell, option_mask
mode_probs
```

Loss/eval-only teacher fields:

```text
g_star, y_star, h_star, k_star, u_star, c_rule_star
spec_margin_star, spec_id_star, margin_option
obs_class, obs_equiv, beta_star, witness_oc, Y_oc, R_star
```

Do not pass loss/eval-only fields to model forward.  `mode_seed_params` is stored for teacher debugging only and is also forbidden in forward.

## Tests

```bash
pytest -q
```

Expected result after this migration: all tests pass.

## Paper-aligned experiment support added in this optimized version

This codebase now supports the deployable OC-RAP path described in `post-collision.tex` without silently replacing OC-RAP by oracle teacher selection.

### Important alignment guarantees

- `method=ours`, `method=ocrap`, and `method=crisp` use CRISP selection over OC-MERO profiles. They no longer call the oracle selector.
- Closed-loop fallback keeps the OC-RAP selector. If a simulator backend is unavailable, the fallback is explicitly reported as `offline_same_candidate_fallback` and `oracle_selector_used_for_ours=false`.
- CRISP controlled relaxation now uses all calibrated offsets from the paper equation: `q_R`, `q_H`, `q_delta`, and `q_C`, including absolute harm and relative harm-gap violations.
- ReCoT posterior logits are action-conditioned through the prefix/root fused representation, instead of being copied across all candidate prefixes.
- `train_care.py` now passes named tensors to `ReCoT.forward`; the previous positional call mis-routed `options_states_ref` into the `actions_controls` slot and could train/evaluate the wrong computational graph.
- Calibration can use learned checkpoint predictions when `--checkpoint` is supplied; otherwise it falls back to teacher profiles and marks this in output metadata.
- Offline and closed-loop metrics distinguish deployable observation-consistent success from non-deployable oracle option-max diagnostics.

### Smoke-test pipeline

```bash
# 0. Install
pip install -r requirements.txt
pip install -e .

# 1. Build a tiny synthetic diagnostic dataset
python scripts/collect_roots.py \
  --config configs/dataset_metadrive_diagnostic.yaml \
  --output data/smoke/roots \
  --max-roots 8

python scripts/build_teacher_labels.py \
  --config configs/dataset_metadrive_diagnostic.yaml \
  --split all \
  --root-dir data/smoke/roots \
  --output data/smoke/dataset \
  --max-roots 8 \
  --shard-size 4

# 2. Train ReCoT on the diagnostic data
python scripts/train_care.py \
  --dataset data/smoke/dataset \
  --output checkpoints/smoke_recot \
  --epochs 1 \
  --batch-size 1

# 3. Calibrate CRISP on a held-out calibration split in real runs.
# For this tiny smoke test, the same dataset is reused only to verify plumbing.
python scripts/calibrate.py \
  --dataset data/smoke/dataset \
  --checkpoint checkpoints/smoke_recot/best.pt \
  --output outputs/smoke_calibration \
  --batch-size 1

# 4. Evaluate deployable OC-RAP/CRISP, not oracle selection
python scripts/offline_eval.py \
  --dataset data/smoke/dataset \
  --checkpoint checkpoints/smoke_recot/best.pt \
  --calibration outputs/smoke_calibration \
  --method ours \
  --output outputs/smoke_eval \
  --batch-size 1

# 5. Closed-loop entrypoint; falls back to same-candidate offline replay if no simulator backend is connected
python scripts/eval_closed_loop.py \
  --dataset data/smoke/dataset \
  --checkpoint checkpoints/smoke_recot/best.pt \
  --calibration outputs/smoke_calibration \
  --method ours \
  --output outputs/smoke_closed_loop \
  --batch-size 1
```

### Full paper-style experiment commands

The paper's external baseline comparison can be omitted, but all OC-RAP metrics and ablations are supported through script flags.

```bash
# OC-RAP only, with all internal ablations
python scripts/run_all_experiments.py \
  --dataset data/recap/test.zarr \
  --checkpoint checkpoints/recot/best.pt \
  --calibration outputs/calibration/q_values.json \
  --output outputs/experiments/ocrap_suite \
  --skip-baselines \
  --batch-size 4

# Include built-in simple baselines: nominal, risk_aware, backup_filter, oracle diagnostic
python scripts/run_all_experiments.py \
  --dataset data/recap/test.zarr \
  --checkpoint checkpoints/recot/best.pt \
  --calibration outputs/calibration/q_values.json \
  --output outputs/experiments/all_builtin \
  --batch-size 4

# Single ablation switch
python scripts/run_ablation.py \
  --ablation no_harm_constraint \
  --dataset data/recap/test.zarr \
  --checkpoint checkpoints/recot/best.pt \
  --calibration outputs/calibration/q_values.json \
  --output outputs/ablations/no_harm_constraint \
  --batch-size 4
```

Supported ablations:

```text
no_harm_constraint
no_rule_constraint
no_controlled_relaxation
no_recovery_constraint
penalize_uncertainty
oracle_witness       # diagnostic flag; use OLG/ORS to quantify leakage
```

### Metrics written by evaluation scripts

`metrics.json` and `all_metrics.json` include the experiment metrics needed by the paper tables:

```text
OCS   observation-consistent selected recovery success
ORS_oracle_option_success   non-deployable option-max diagnostic
FAR   selected-action false admission rate relative to eta_R
SLR   selected lower-tail recoverability
OLG   oracle leakage gap, oracle option-max recoverability minus OC recoverability
SRA   same-root pairwise ranking accuracy, when predicted profiles are available
SRR   same-root recoverability regret
R_MAE OC-MERO recoverability profile MAE
WAcc  witness accuracy on non-ambiguous witness-gap samples
OCV_JS observation-consistency JS violation, when predicted mu is available
HNIV  harm non-inferiority violation rate
MIR   minimal-intervention regret when nominal is admissible
utility_mean selected nominal utility
```

### Dataset requirements for paper-final claims

The synthetic diagnostic generator is suitable for smoke tests and regression tests only. Paper-final results require a real MetaDrive/ScenarioNet dataset with:

- split-by-root and split-by-original-scenario leakage control;
- `normal`, `low_headroom`, `near_contact`, and `contact_post_contact` strata;
- root-shared modes reused across all candidate prefixes;
- `Y_oc`, `witness_oc`, `obs_equiv`, `beta_star`, and `R_star` labels;
- `H_action_star`, `c_rule_star`, and post-contact `K`/secondary-collision fields for CRISP constraints and post-contact metrics;
- crash termination disabled in contact/post-contact closed-loop evaluation.

If `--checkpoint` is omitted, evaluation uses teacher profiles for OC-RAP and sets `uses_teacher_profiles_for_ours=true`; this is acceptable only for debugging the selector and metric code, not for learned-inference paper tables.
