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
