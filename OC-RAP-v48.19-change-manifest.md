# OC-RAP v48.19 Change Manifest

| Type | File | Purpose |
|---|---|---|
| ADD | `src/ocrap/algorithms/evidence_targets.py` | Shared factorized benefit/component-veto harm target implementation. |
| ADD | `src/ocrap/evaluation/certificate_stats.py` | Explicit one-/two-sided Wilson bounds and optimistic gate feasibility. |
| MOD | `src/ocrap/models/ocrap.py` | Shared calibrator plus bounded regime residuals and nominal pinning. |
| MOD | `src/ocrap/models/losses.py` | Independent factorized-tail supervision. |
| MOD | `src/ocrap/cli/train.py` | FACET metrics, factorized labels, checkpoint/config persistence. |
| MOD | `src/ocrap/models/inference.py` | Load shared calibrator and regime-residual configuration. |
| MOD | `tools/build_teacher_pcd_index_v48.py` | Component-veto labels, support counts, manifest/parameter index contract. |
| MOD | `tools/calibrate_policy_risk_v48.py` | Shared target semantics, declared Wilson bounds, support preflight. |
| ADD | `tools/check_v48_19_target_support.py` | Index-contract and binary-tail support validation. |
| ADD | `scripts/run_v48_19_facet_dedicated.sh` | Two-A30 controller with strict RC attribution and both-variant requirement. |
| ADD | `scripts/run_v48_19_parallel_ablations.sh` | Four-task concurrent non-repetition ablation controller. |
| ADD | `scripts/calibrate_v48_19_certificate_pool.sh` | Manifest-bound preregistered scene-disjoint certificate. |
| ADD | `scripts/adapt_ocrap_v48_19_facet_variant.sh` | FACET calibrator-only adaptation. |
| ADD | `scripts/run_v48_19_stress_if_authorized.sh` | NEXT_COMMANDS-gated stress execution. |
| ADD | `tests/test_v48_19_facet_bridge.py` | Target overlap, boundary, shared bridge, gate feasibility and contract tests. |
| MOD | `ALGORITHM_CHANGELOG.md` | v48.19 algorithm, result attribution and non-repetition log. |

## Compatibility and invariants

- v48.13 source proposal/evidence checkpoints remain the initialization source.
- Safe nominal lock and test sealing are unchanged.
- v48.19 uses a new output directory/protocol; it does not mutate v48.18 results.
- Existing legacy LCB90/UCB90 JSON field names are retained for downstream readers, while new metadata records the actual confidence level and bound type.
