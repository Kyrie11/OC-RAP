# OC-RAP alignment review after paper/code audit

This review is based on `post-collision.tex` Abstract, Introduction, Method, and Implementation/Reproducibility sections plus the code in this repository.

## Paper algorithm that the code should implement

OC-RAP evaluates short executable ego prefixes by the recovery headroom they preserve.  The paper path is:

1. Build deployable belief inputs: BEV/vector map, route, traffic controls, actor history, occupancy, occlusion, uncertainty.
2. Generate prefix proposals and project them to executable controls under ego dynamics, input limits, static swept-geometry constraints, and route-compatible headings.
3. Generate executable recovery affordance tokens per prefix: lane/stop/yield/gap/escape/rejoin/stabilize anchors, hard-shell constraints, fallback controller, conservative recovery potential.
4. Run root-shared counterfactual teacher/evidence: the same latent root modes are reused across candidate prefixes.
5. Compute option margin as `max(G_no, G_mr, G_post)`, not `min` across mutually exclusive specs.
6. Compute observation-consistent witness labels: modes with equivalent post-prefix observations must share the same witness distribution.
7. Aggregate with the mu-weighted OC-MERO existential operator and lower-tail LCVaR.
8. Select with CRISP using calibrated recoverability, absolute harm, relative harm, and rule/hard-shell offsets.

## Alignment status

### Aligned / now fixed

- Option-level recovery semantics use `max(G_no, G_mr, G_post)` in `recap/teacher/recovery_specs.py`.
- ReCoT blocks oracle-only teacher fields from forward input.
- OC-MERO uses the paper's mu-weighted log-sum-exp existential operator with default `c_R=0`.
- CRISP uses all four calibration offsets in both admissibility and controlled relaxation.
- Dataset schema stores `Y_oc`, `witness_oc`, `obs_equiv`, `beta_star`, and `R_star` separately from oracle diagnostics.
- Offline/closed-loop evaluation uses OC-RAP/CRISP for `ours` and reports oracle option-max only as a diagnostic.

### Bugs fixed in this pass

1. **Observation classes were degenerate.**
   `scripts/build_teacher_labels.py` previously built each action's observation signatures from the same ego post-prefix state for every latent mode.  This merged all modes into one observation class even when MetaDrive replay exposed different visible actor states.  The code now stores one observable post-prefix signature per `(action, mode)` using `post_prefix_observation_signature(trace, ...)`, including visible actor summaries when the rollout backend provides them.

2. **CRISP rule labels were polluted by invalid options.**
   Invalid/padded recovery tokens previously set `c_rule_star[a,m] = 1.0`; since most actions have some invalid/padded options, this could make otherwise admissible actions look rule-violating.  The code now records `c_rule_option[a,j,m]` and reduces it through the observation-consistent witness; invalid/padded tokens do not contaminate the action-level CRISP label.

3. **MetaDrive rollouts did not expose actor states to observation consistency.**
   `recap/teacher/metadrive_rollout.py` collected actors during rollout but did not store them in `RolloutTrace`.  The trace now stores a compact visible-actor observation array so observation equivalence can include mode-dependent visible traffic.

## Simplified implementations that remain

These are acceptable for smoke tests and code-path validation, but they are not paper-final fidelity.

- **Prefix projection** is clipping plus static obstacle bbox checks, not the paper's QP with full kinematic bicycle constraints, swept geometry, route-compatible heading constraints, and initial-state continuity.
- **Recovery affordance tokens** are polynomial reference options with semantic tags.  They do not yet implement full anchor generation from conflict-zone boundaries, legal stop regions, free-space pockets, rejoin lanes, and post-contact stabilization basins.
- **Hard shell / recovery potential** are proxy margins from rollout traces.  The learned residual potential, calibrated residual `q_V,j`, one-step decrease condition, friction circle, yaw/sideslip, protected-space, and full secondary-collision hard-shell terms are approximated.
- **Teacher root modes** are fixed semantic perturbation slots, not a learned or calibrated posterior over latent futures.
- **Observation signature** now includes visible actor summary when available, but still does not fully implement traffic-control disagreement, occupancy-IoU distance, occlusion-mask disagreement, and ego-state weighting as in the paper.
- **Synthetic diagnostic backend** has no real actor observations, so observation classes can still collapse in synthetic runs.  Use it only to test plumbing.
- **Closed-loop fallback** is still an offline same-candidate fallback if no simulator backend is available; it should not be reported as true closed-loop evidence.

## Dataset generation assessment

The proposed MetaDrive + WOMD/ScenarioNet route is reasonable for OC-RAP research because MetaDrive/ScenarioNet support real-data scenario reconstruction and interaction, while WOMD provides object tracks and map data.  However, the current codebase should distinguish:

- **Synthetic diagnostic datasets:** useful for unit tests and catching semantic bugs, not paper claims.
- **Real MetaDrive/ScenarioNet WOMD datasets:** potentially suitable if root-time restore, coordinate centralization, split leakage control, actor replay/reactivity, and disabled crash termination are verified.
- **Hybrid stress datasets:** useful and likely necessary for low-headroom, near-contact, and post-contact balance, but should be reported separately because they are perturbed/generated rather than natural WOMD distribution.

## Inputs that are theoretically risky or not directly deployable

- `mode_seed_params` are hidden teacher parameters. They must stay loss/debug only and never enter ReCoT forward.
- `obs_class`, `obs_equiv`, `beta_star`, `witness_oc`, `Y_oc`, `R_star`, `spec_*`, and future teacher margins are labels/evaluation fields only.
- Conflict regions, protected zones, occlusion masks, and uncertainty channels are deployable only if derived from online perception/map prediction. WOMD future trajectories or hidden scenario fields must not be used to build online inputs.
- In real deployment, “conflict region” should be built from current map topology, traffic rules, predicted occupancy, and reachable sets, not from future logged interactions.
- Post-contact stabilization labels require a simulator/dynamics model that keeps rolling after contact. If MetaDrive terminates on crash, those labels are invalid unless crash termination is disabled or a post-contact dynamics model is used.

## Validation run

- `pytest -q`: 52 passed.
- `python -m compileall -q recap scripts`: passed.
