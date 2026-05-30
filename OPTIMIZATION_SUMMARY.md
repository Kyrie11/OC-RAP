# OC-RAP optimization summary

This zip contains a paper-alignment pass driven by `post-collision.tex` and the diagnostic report.  The main goal was to turn the repository from a paper-aligned smoke test into a more faithful, fail-fast OC-RAP implementation.

## Implemented code-level fixes

- **BEV history**: changed the default history to `H_h=5` and rebuilt `BEVBuilder.build_from_state()` so the temporal dimension contains real historical ego/actor frames instead of repeated current-frame copies.
- **Action prefix projection**: replaced the clipping-only projection with sequential kinematic-bicycle rollout, acceleration/curvature clipping, jerk and curvature-rate limiting, route heading/lateral checks, and swept oriented-box checks against static obstacle polygons.
- **Teacher candidate coverage**: `build_teacher_labels.py` now passes `K_raw` into the lattice generator instead of silently ignoring the config value.
- **Recovery tokens**: recovery option tensors now expose richer 12-D hard-shell / potential proxy features and preserve one option per semantic tag before filling the remaining `L` slots.
- **Observation-consistent runtime OC-MERO**: `compute_ocmero_profiles()` now uses the predicted `beta_logits` to tie witness distributions across post-prefix observation-equivalence posteriors.  The `no_observation_consistency` ablation disables this runtime tying.
- **Harm and rule profiles**: `H_action_star` uses upper-tail CVaR over modes rather than a hard max; `c_rule_star` is reduced through the observation-consistent witness and aggregated by upper-tail CVaR during calibration/evaluation.
- **Metrics**: invalid/padded actions are masked in harm non-inferiority; selected action `-1` is handled explicitly; pairwise ranking accuracy no longer double-counts correct pairs.
- **Training**: `train_care.py` now reads config values for hidden dimension, optimizer, learning rate, epochs, batch size, loss weights, and OC-MERO parameters.  The default hidden dimension is restored to 128.
- **Calibration**: CRISP calibration now searches `q_R`, `q_H`, `q_delta`, and `q_C`; it keeps separate `delta_R`, `delta_H`, `delta_delta`, and `delta_C` targets.
- **Ablations**: `run_ablation.py` now changes inference/evaluation behavior for `no_observation_consistency` and `oracle_witness` instead of only recording flags.
- **README/runtime commands**: fixed the real MetaDrive root collection command mismatch and updated all BEV examples to 5 history frames.
- **Hybrid stress roots**: stress actor histories are now kinematically consistent or intentionally hidden for occluded-release roots.  JSON-level stress roots are marked not paper-final and real MetaDrive teacher labels fail fast unless the stress actor is injected into ScenarioEnv or explicitly allowed for diagnostics.
- **Closed-loop guardrail**: the existing closed-loop entrypoint is now explicitly labeled `offline_same_candidate_fallback` when no live simulator backend is connected, and `--require-simulator` can be used to fail fast.

## Remaining simulator-dependent work

The following items cannot be honestly completed inside a portable source-only zip without the live MetaDrive/WOMD/ScenarioNet environment and scenario-writing hooks:

- true online closed-loop replanning with simulator execution every 0.2--0.6 s;
- writing hybrid stress actors back into ScenarioNet scenario files, or spawning and controlling them inside MetaDrive;
- full all-actor / traffic-light / policy-state / RNG root-time restore beyond the existing runner checks;
- neural action proposal training, if the paper-final system requires learned proposals instead of the improved lattice/projection fallback;
- full QP projection solver with route-graph and continuous swept-geometry constraints.

For those cases the code now avoids silently producing paper-final-looking metrics from diagnostic fallbacks.

## Validation performed

```bash
python -m compileall -q ocrap scripts tests
pytest -q
# 52 passed

# Additional tiny synthetic smoke test exercised:
# collect_roots -> build_teacher_labels -> train_care -> calibrate -> offline_eval
```
