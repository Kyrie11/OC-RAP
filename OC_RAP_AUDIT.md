# OC-RAP paper/code alignment audit

This audit was produced after reading `post-collision.tex` sections Abstract, Introduction, Method, Experiments, and Implementation/Reproducibility Details, then inspecting the code and running smoke tests.

## Critical issues found and fixed

1. **`method=ours` was not deployable.**
   - Problem: offline evaluation treated unknown methods, including `ours`, as oracle teacher selection. Closed-loop fallback explicitly replaced `ours` with `oracle`.
   - Fix: `recap/evaluation/offline_eval.py` now implements OC-RAP/CRISP selection. `recap/evaluation/closed_loop_eval.py` preserves `method=ours` and reports fallback status.

2. **CRISP controlled relaxation did not match the paper.**
   - Problem: relaxation ignored `q_R`, `q_H`, `q_delta`, `q_C` and omitted the absolute harm violation term.
   - Fix: `recap/models/selector.py` now uses the calibrated relaxation objective from the paper equation.

3. **ReCoT training call passed tensors positionally into the wrong arguments.**
   - Problem: `train_care.py` passed `options_states_ref` where `actions_controls` was expected, then masks where option tensors were expected.
   - Fix: named arguments are now used for `actions_states`, `token_states_ref`, controls, token metadata, and masks.

4. **Predicted posterior was not action-conditioned.**
   - Problem: `beta_logits` were produced from root-mode features and copied over actions, while the paper defines a post-prefix posterior.
   - Fix: ReCoT now predicts `beta_logits` from the action/root fused representation.

5. **Calibration and evaluation ignored learned checkpoints.**
   - Problem: scripts used teacher labels even when `--checkpoint` was supplied.
   - Fix: added `recap/evaluation/inference.py`, and updated `calibrate.py`, `offline_eval.py`, `eval_closed_loop.py`, `run_ablation.py`, and `run_all_experiments.py` to run ReCoT + OC-MERO predictions when a checkpoint is provided.

6. **Oracle option-max metric was too optimistic.**
   - Problem: `oracle_recovery_success` returned success if any option succeeded in any mode.
   - Fix: it now computes mode-wise option-max success averaged over modes; `OLG` reports oracle leakage gap separately.

7. **Experiment scripts were placeholders.**
   - Problem: `run_all_experiments.py` and `run_ablation.py` only wrote status/flag files.
   - Fix: they now run built-in OC-RAP methods, internal ablations, and write JSON/CSV metrics.

8. **Heavy SciPy import delayed calibration CLI.**
   - Problem: importing `scipy.stats` at module import time could stall script startup.
   - Fix: calibration uses a conservative Hoeffding upper confidence bound and avoids the heavy import.

## Remaining limitations and how to interpret them

- The synthetic teacher remains a diagnostic approximation. It is enough for smoke tests and selector/metric correctness, but not enough for paper-final claims.
- External baselines from the paper are intentionally not fully implemented, per the user request. Built-in `nominal`, `risk_aware`, `backup_filter`, and `oracle` are available as lightweight diagnostics.
- A real paper run still requires full MetaDrive/ScenarioNet roots, root-shared mode replay, real teacher rollouts, balanced splits, and disabled crash termination for contact/post-contact evaluation.
- If `--checkpoint` is omitted, evaluation uses teacher profiles and marks `uses_teacher_profiles_for_ours=true`; do not use that mode for learned-inference tables.

## Smoke checks run

- Built diagnostic roots and teacher labels.
- Trained one ReCoT smoke checkpoint.
- Ran calibration and OC-RAP offline evaluation with a checkpoint.
- Ran focused tests covering selector, OC-MERO semantics, no-oracle-leakage, label semantics, and recovery options.
- Ran `python -m compileall -q recap scripts`.
