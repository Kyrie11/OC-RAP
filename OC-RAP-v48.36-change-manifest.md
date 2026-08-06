# OC-RAP v48.36 OCAF Change Manifest

## Algorithm

- Added `ObservationConditionedActionFrontierBridge` in `src/ocrap/models/ocrap.py`.
- Replaced action-only evidence context in the main configuration with candidate action × nominal-observation interaction (`physical_interaction`).
- Preserved one network, five continuous signed safety components, one non-compensatory frontier, and one shared deployment rule; regime labels remain audit-only.
- Added configurable OCAF hidden width/dropout and source-consensus prior scale; propagated them through training, checkpoint serialization and inference reconstruction.

## Training and calibration

- Added v48.36 single-stage/three-stage adaptation scripts and dedicated controller.
- OCAF bridge is trainable in factor and identity stages; final calibration is disabled by default.
- Factor-cache identity now includes context source, interaction width/dropout, consensus scale and admission prior mode.
- Added v48.36 shared-rule fitter and metric-calibration identity checker.

## Scientific controls

- Added independent 2×2 ablations: action-only/OCAF × compensatory slack/non-compensatory frontier.
- Contract expectations are arm-aware so controls cannot be converted to RC=30 merely for lacking the main component.
- Near/Contact remain worst-stratum audits, not policy cases.

## Engineering integrity

- Added canonical dataset-root and legacy-alias rejection.
- Added attempt-scoped authoritative resolver, exact RC/NEXT_COMMANDS checks, and OCAF-specific resume authorization.
- Added v48.36-only Safe/stress wrappers and result packager.
- Added dependency-closure, bridge, model, training, calibration and dataset contracts.

## Paper

- Added `paper/post-collision-v48.36-OCAF.tex` with continuous-headroom/OCAF formulation and removed regime-conditioned admission behavior.
- Fixed the malformed `hyperref` declaration. The uploaded materials did not include `post-collision.bib`; citations therefore remain unresolved until the bibliography is restored.

## Tests

- Added `tests/test_v48_36_ocaf.py`: zero-action, magnitude/gradient, nominal observation anchoring, scene modulation, non-compensatory cap, canonical root rejection, RC resolver, independent/arm-aware 2×2, and OCAF resume semantics.
