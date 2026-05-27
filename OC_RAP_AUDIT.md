# OC-RAP migration audit

## Alignment findings

The original codebase was a good engineering skeleton, but its deployable core still matched the earlier ReCAP/CARE/MERO prototype more than the current OC-RAP method:

1. `CARE` predicted legacy scalar evidence (`P/G/C/U/H/K`) rather than signed recovery-margin vectors.
2. `MERO` performed an option existential aggregation without the paper's observation-consistent witness policy and μ-weighted existential operator.
3. `R_star` was built from `Y.max(axis=1)`, which is oracle-existential: each hidden mode could use a different option even when post-prefix observations could not distinguish those modes.
4. The selector only supported `q_R` and `q_H`, with `q_H` effectively used as a relative harm-gap offset; it lacked absolute harm and rule offsets.
5. `build_teacher_labels.py` did not store the OC-RAP schema (`obs_class`, `obs_equiv`, `beta_star`, `witness_oc`, `Y_oc`, signed margin vectors, or spec IDs).
6. `recovery_options.py` sometimes failed to preserve one required recovery semantic token per prefix when `L` allowed it.

## Main changes made

- Added `RecoveryAffordanceToken` schema while keeping `RecoveryOption` compatibility.
- Added `teacher/recovery_specs.py` implementing `max(G_no, G_mr, G_post)` and signed `g_vector` labels.
- Added `teacher/observation_classes.py` implementing observable signatures, equivalence classes, beta posterior, and class-consistent witnesses.
- Updated evidence labels to emit `g_star`, `y_star`, `spec_margin_star`, `spec_id_star`, `margin_option`, `k_star`, and `c_rule_star`.
- Updated teacher-label builder to compute `R_star` from `Y_oc`, not from oracle `max_j Y_option`; oracle labels remain only as diagnostics.
- Added `models/recot.py`; `CARE` remains an alias for compatibility.
- Added `models/oc_mero.py` with μ-weighted existential aggregation and default `c_R=0.0`.
- Rewrote `selector.py` to accept `q_R`, `q_H`, `q_delta`, and `q_C` and apply recovery, absolute harm, relative harm, and rule constraints.
- Rewrote calibration to simulate selected-action CRISP decisions over candidate offsets.
- Updated dataset loading to surface OC-RAP keys and map old option names to token names.
- Updated metrics to prefer deployable OC recovery (`Y_oc`/`witness_oc`) and keep option-max as `oracle_recovery_success`.
- Added unit tests for spec max, observation-consistent witness, no-oracle R label, μ-weighted duplicate invariance, and forward leakage rejection.
- Updated README with the OC-RAP pipeline and required commands.

## Remaining limitations

This is a semantic and API-level migration of the current codebase. Some research-grade parts remain simplified relative to a full production/paper-final implementation:

- The observation signature is conservative and currently uses observable post-prefix ego/actor summaries; richer occupancy, traffic-control, and occlusion summaries should be plugged in when real simulator observations are available.
- `ReCoT` uses a compact transformer-style factorized architecture, not a fully optimized large model.
- `build_teacher_labels.py` preserves synthetic rollout compatibility; real MetaDrive label quality still depends on the simulator runner and root-time restoration.
- Some legacy wrappers and script names remain to avoid breaking existing commands.

## Validation

Ran:

```bash
pytest -q
```

Result:

```text
52 passed
```
