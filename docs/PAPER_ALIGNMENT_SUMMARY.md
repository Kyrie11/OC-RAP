# Paper alignment summary

The implementation follows the paper story:

- The planner is recoverability-centered, not a collision-risk or emergency-braking-only planner.
- CARE predicts decision-sufficient evidence `(P,G,C,U,H,Kdef)` over action-option-mode tuples instead of a direct recoverability scalar.
- MERO converts evidence to profiles through signed monotone scoring, existential option aggregation, lower-tail LCVaR over modes, and upper-tail profiles for bottleneck, uncertainty, harm, and post-contact deficit.
- The selector constructs calibrated recovery and harm sets, accepts nominal actions only when both constraints hold, otherwise chooses constrained or controlled-relaxation actions.
- Experiments and metrics cover RS, FR, SLR, PRA, SRR, AIEE, MPE, WA, BF1, CR, OR, Prog, PCSS, HNIV, MIR, and runtime.
- Appendix details are implemented as code invariants: neural proposal interface plus projection, recovery option vocabulary, root-shared modes, teacher margins, calibrated affordances, CARE losses, MERO LCVaR, and q calibration.

Important implementation corrections from the guide are encoded as tests:

- Prefix/recovery stages are horizon boundaries, not pre-event/post-event labels.
- Root-shared modes fix latent traffic context, not open-loop non-ego futures.
- H is prefix-level first-contact harm exposure and is excluded from R.
- MERO does not average options and normalizes the existential aggregate by valid option count.
- First-contact collision margin does not automatically kill post-contact recoverability.
- Selector does not double-penalize U in the main method.
- Splits are by root scene, not `(root, action, option, mode)` tuples.
