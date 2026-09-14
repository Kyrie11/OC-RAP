# OC-RAP external-baseline audit and optimization

Date: 2026-09-14

## Scope

This audit covers the uploaded OC-RAP code/data bundle and the uploaded source implementations for **Diffusion Planner** and **Flow Planner**. It also audits the shared WOMD/Waymax adapter, the one-regime launch path in `CLEAN_COMMANDS.md`, the shared closed-loop metric implementation, and publication-latency accounting. The optimized tree is an **interface-adapted reproduction** for the OC-RAP executable-candidate protocol, not a claim of checkpoint compatibility with either authors' nuPlan implementation.

The supplied raw WOMD path `/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/` is not mounted in the review environment, so I could not execute end-to-end Waymax replay here. Source-level, numerical-unit, launcher-contract, and regression tests were run; the final one-regime commands should still be run on the machine where WOMD is present.

## Executive verdict

| Component | Supplied OC-RAP v60 | Audit verdict | Optimized v61 |
|---|---|---|---|
| Diffusion Planner | Generic pooled scene encoder, candidate-wise VP corruption, deterministic pseudo-noise + one denoising pass at test, learned candidate prior | **Not source-faithful.** It was diffusion-inspired rather than a reproduction of the uploaded planner. | VP/x-start objective, fixed current state, source-style 10-step order-2 multistep DPM-Solver++ with logSNR spacing, scene-conditioned denoiser, generate-once then candidate projection. **Still partial** because current serialized OC-RAP groups do not contain neighbor future GT. |
| Flow Planner | Overlapping segments and a flow-like loss, but whole-scene CFG dropout, one endpoint-like test update, generic pooled fusion | **Partial only.** Several paper-defining operators were missing. | CondOT x-start flow matching, overlapping tokenization + consistency/averaging, distance-scaled scene/trajectory attention, modality-specific processing, nearest-neighbor CFG, packed conditional/unconditional pass, four-step explicit midpoint ODE. **Paper-core retained, WOMD-interface adapted.** |
| WOMD bridge | Observation-only ego/agent/map/route tensors; nearest observed agents; ego-centric frame | **Conceptually correct and leakage-safe**, but not native nuPlan preprocessing. | Retained; fidelity limitations are now explicitly recorded in provenance. |
| Closed-loop dispatch | `diffusion_planner`, `flow_planner`, and `plan_r1` were missing from the external learned-method sets | **Critical integration bug.** Direct closed-loop execution could take the wrong model-loading path. | Fixed and regression-tested. |
| Metrics | Most Safe/Near/Contact metrics were geometrically and temporally consistent | One semantic bug: `secondary_overlap_event` was effectively overwritten by re-contact, so it did not mean collision with a second object. | Fixed using exact oriented-box overlapping object IDs; unknown partner identity yields NaN rather than a false claim. |
| Latency | Correct broad inclusion/exclusion boundary, but normal commands run 3 jobs/GPU and therefore produce contended timing; no steady-state warm-up summary | **Not suitable for publication latency as run.** | Explicit CUDA synchronization, per-decision samples, warm-up exclusion, mean/p50/p95 steady-state summaries, and a serial one-process/one-GPU profiler. |

## 1. What the uploaded source implementations actually require

### Diffusion Planner

The uploaded source uses a VP diffusion process and x-start style prediction. In `diffusion_planner/model/module/decoder.py` and `diffusion_planner/loss.py`, the model jointly represents ego planning and neighboring-agent futures; the current state is prepended/fixed and future states are diffused. Its sampler in `diffusion_planner/model/diffusion_utils/sampling.py` uses `diffusion_steps=10`, DPM-Solver++ (`algorithm_type="dpmsolver++"`), order 2, `skip_type="logSNR"`, `method="multistep"`, and `denoise_to_zero=True`. The README states approximately 20 Hz inference for the native system.

The original OC-RAP v60 port did not retain this inference operator. It evaluated a fixed intermediate time with deterministic pseudo-noise and one network pass, did not perform DPM-Solver++ integration, did not jointly generate neighbor futures, and added a learned candidate prior that is not part of the source generative objective. It should therefore not be described as an operator-faithful Diffusion Planner reproduction.

The v61 port restores the source diffusion operator that can be implemented with the currently serialized OC-RAP samples: VP corruption, x-start training, fixed current state, source-style DPM-Solver++ sampling, and scene-conditioned trajectory denoising. At inference it generates one native ego trajectory and projects that trajectory onto the common 24-candidate executable lattice. This avoids changing the native generative loss into a candidate-classification loss.

A fully source-faithful Diffusion Planner training objective is **not possible from the current `.npz` groups alone**, because the source jointly supervises ego and neighboring-agent future trajectories while the OC-RAP external-baseline samples intentionally expose observed history but not neighboring-agent future GT. Exact reproduction would require joining each sample back to raw WOMD by scenario/time and building an additional neighbor-future target tensor. The v61 provenance therefore says `partial source reproduction / ego-only interface adaptation`, not “official” or “exact”. Optional classifier guidance, the source static normalizer, and the source current-state perturbation/quintic augmentation are also not enabled in this port.

### Flow Planner

The uploaded paper/source has three central algorithmic pieces: (i) overlapping fine-grained trajectory tokens, (ii) interaction-enhanced spatiotemporal fusion including spatial-distance scaling and modality-specific processing, and (iii) flow matching with classifier-free guidance. The released config uses 32 neighbors, CFG on the nearest 10 neighbors, `cfg_prob=0.3`, `cfg_weight=1.8`, 80 future points, action length 20, overlap 10, and four midpoint ODE steps. The source packs conditioned/unconditioned behavior into the guidance velocity and averages overlaps when assembling the final trajectory.

The v60 OC-RAP port preserved overlap segmentation and a flow target, but its CFG removed the whole pooled scene rather than only the selected neighboring-agent condition, its test path was not the released four-step midpoint ODE, and its generic pooled scene encoder omitted the paper's scale-adaptive joint spatial fusion and modality-specific normalization/FFN structure. Those are core algorithmic differences, not cosmetic implementation details.

The v61 port restores the paper-core operator: conditional-OT x-start training, overlapping tokens, overlap consistency, distance-scaled attention over scene/trajectory tokens, modality-specific normalization/FFNs, nearest-neighbor CFG dropout, batched conditional/unconditional guidance `(1-w)u_uncond + w u_cond`, four explicit-midpoint steps, and overlap averaging. The common OC-RAP horizon is shorter, so segment length/overlap are rounded proportional adaptations rather than native 80-step settings. Static training-set normalization and the source perturbation/quintic augmentation remain omitted and are documented as gaps.

## 2. WOMD/Waymax dataset integration

`src/ocrap/external_baselines/data.py::_source_scene_arrays` is observation-side only. It builds the ego plus nearest observed actors, sorts agents by current observed distance, uses an ego-centric coordinate system, provides current ego state/history, builds nearest map polylines, and exposes route/traffic/speed information. I did not find a path that feeds OC-RAP counterfactual teacher futures into Diffusion/Flow action selection. `allow_teacher_supervision` is false in both optimized configs, and training targets remain logged-nominal imitation targets.

This interface is correct for a fair common-protocol comparison, but it is not native nuPlan preprocessing. Important adaptations are: 11 WOMD history states rather than the native 21-history setting used by the uploaded planners; 32 neighbors and 70 map polylines in v61 to better match source capacity; a common 20-step executable horizon rather than Flow Planner's 80-step native horizon; and a candidate-lattice projection after native generation. These adaptations must be disclosed in the paper/table caption or provenance appendix.

The dataset-property bundle is internally consistent with the cleaned command contract for validation/test data. Audited test sizes are Safe 3,216 samples / 175 scenes / 402 groups, Near-Contact 4,723 / 250 / 595, and Contact 6,687 / 209 / 747. The visible validation and test provenance points to standard WOMD `validation/validation_tfexample.tfrecord@150`. `CLEAN_COMMANDS.md` also explicitly states that publication validation/test/calibration use standard `validation`, not `validation_interactive`.

This exposes a paper-text error in the uploaded TeX: the original Appendix data paragraph said Near-Contact and Contact held-out tests used `validation-interactive`. The corrected TeX changes all publication validation/calibration/test roles to deterministic scene-disjoint slices of standard WOMD validation and explicitly says `validation_interactive` is not a publication source. The exported official WOMD testing split is still not claimed.

## 3. One-regime training/testing logic

The user-selected entry point, Section 4 of `CLEAN_COMMANDS.md`, is the right logic for a regime-wise comparison:

- Safe adds Diffusion Planner to the Safe external suite.
- Near-Contact adds Flow Planner (plus Plan-R1 and expanded BeTopNet in the existing suite).
- Contact retains the post-contact baselines; Diffusion/Flow are not silently reused as Contact-specific methods.
- `--womd-role validation` is the publication role.
- `--test-only` correctly fails closed when required learned checkpoints or Near calibration artifacts are absent/incompatible.

A critical runner bug was fixed: Diffusion Planner, Flow Planner, and Plan-R1 were not originally registered in `EXTERNAL_CLOSED_LOOP_METHODS` / `EXTERNAL_LEARNED_METHODS`. The optimized runner registers their aliases, so a one-regime learned baseline now goes through the external checkpoint/model path rather than falling into an OC-RAP bundle path.

## 4. Metric audit

### Safe

Collision and off-road are scene-level endpoints. Nominal-utility preservation is bounded, and intervention is separated from safety endpoints. The metrics are appropriate for the paper's Safe regime provided the same scene cohort and execution horizon are used for every method.

### Near-Contact

Clearance is computed with oriented-box signed distance, including penetration when boxes overlap. TTC uses swept oriented boxes under a **constant-velocity, frozen-heading/shape** motion model; it is a TTC proxy, not a simulator-oracle future TTC. Critical-TTC exposure uses left-endpoint intervals so duration is `count * dt` over intervals rather than accidentally treating `N+1` sampled states as `N+1` durations. Low-tail aggregation is performed from per-scene minima, which is the correct hierarchy for a scene-level lower-tail metric.

The corrected TeX now calls this quantity a “constant-velocity swept oriented-box TTC proxy” in the prose so the implementation is not overstated.

### Contact

Post-contact metrics are entered only after simulator-observed overlap and require at least one state after first overlap; otherwise post-contact outcomes are NaN rather than fake failures/successes. Terminal clearance, normalized free-space AUC, clearance gain, escape, overlap duration, stable-stop quality, off-road, and re-contact are computed after the contact anchor as intended.

The original secondary-overlap implementation was wrong semantically: a later overlap episode was ultimately used as both re-contact and secondary overlap. v61 now tracks the object-slot IDs whose oriented boxes overlap the ego. `recontact_event` remains “a later overlap episode”; `secondary_overlap_event` is only true when a later overlap involves an object outside the first-contact partner set. If partner identity is unavailable, the secondary-overlap metric is NaN. A separate generic multi-overlap-episode diagnostic remains available.

## 5. Latency audit

The existing high-level timing boundary was sensible: deployed-planner latency includes state-history construction, candidate-feature construction, and policy selection/model inference, while teacher-label computation, audit-only labels, and Waymax simulator stepping are excluded. That is the right distinction for planner latency.

However, the normal Section-4 commands use six concurrent workers (`3 jobs/GPU × 2 GPUs`). Those commands are excellent for throughput but **cannot produce uncontended publication latency**. In addition, asynchronous CUDA launches should be synchronized at timing boundaries, and warm-up decisions should not dominate small-scene averages.

The optimized code therefore:

- explicitly synchronizes CUDA around external learned-policy timing;
- stores per-decision deployed latency samples;
- excludes the first 3 planner decisions per scene by default for steady-state latency;
- reports steady-state mean, p50, and p95;
- makes `tools/build_regime_comparison_tables.py` prefer the steady-state mean when available;
- adds `scripts/profile_external_baselines_latency.sh`, which runs exactly one process on one GPU in `--test-only` mode and reuses trained checkpoints; for Near it also reuses the existing conformal calibration artifact.

Do **not** compare the paper's Flow Planner ~12 Hz/A6000 statement or Diffusion Planner ~20 Hz native statement directly to the OC-RAP number unless GPU model, precision, horizon, scene tensor sizes, candidate projection, warm-up, and timing boundary are reported together.

## 6. Safe acceleration: what can and cannot be sped up without changing the algorithm

The optimized changes target redundant work rather than reducing source solver quality:

1. **Target-only native generative training.** The old port applied generative work across all 24 executable candidates. The source objectives learn the expert/logged trajectory distribution, so v61 performs the native denoising/flow regression only on the logged target. Candidate CE is set to zero. This can remove up to 24-way duplicated decoder work, although end-to-end speedup will be smaller because scene encoding, data loading, and optimizer work remain.
2. **Generate once, project once.** At test time Diffusion/Flow generate one native trajectory per scene and then score/projection-match the 24 executable candidates. They do not run a diffusion/flow solver separately for every candidate.
3. **Packed Flow CFG.** Conditioned and unconditioned branches are concatenated in one model batch per ODE evaluation. This preserves the exact CFG equation while reducing Python/kernel-launch overhead and making better use of the GPU.
4. **Keep source solver depth.** Publication mode keeps 10 DPM-Solver++ steps for Diffusion and 4 midpoint steps for Flow. Reducing these would be a separate accuracy/latency ablation, not a “free” optimization.
5. **Keep throughput and latency runs separate.** Six worker slots remain appropriate for completing the large metric sweep quickly. Latency is measured later in an isolated pass reusing checkpoints, so you do not sacrifice experiment throughput just to obtain clean timing.
6. **AMP / fused optimizer / pinned persistent data loading remain enabled** in the learned configs where supported. These are implementation accelerations and do not change the model objective.

The optimized default is 100 epochs for these two ports, compared with only 24 in the prior OC-RAP configs. This is intentionally a more credible training budget, not a claim of source-training parity. Because the target-only decoder removes the largest accidental 24-candidate duplication, the increased epoch budget is still much more defensible than obtaining speed by replacing the published sampler. For final publication results, inspect validation curves and repeat training if 100 epochs has not converged; do not claim author-level reproduction merely from completing the configured budget.

## 7. Commands to run on the WOMD machine

Accuracy/closed-loop metric sweep remains exactly the Section-4 regime-wise workflow:

```bash
bash scripts/run_external_baselines.sh \
  --regime safe \
  --out "$BASE_OUT/external_baselines_v48_111" \
  --gpus 0,1 --jobs-per-gpu 3 --max-parallel 6 --max-scenarios 0 --womd-role validation

bash scripts/run_external_baselines.sh \
  --regime near \
  --out "$BASE_OUT/external_baselines_v48_111" \
  --gpus 0,1 --jobs-per-gpu 3 --max-parallel 6 --max-scenarios 0 --womd-role validation

bash scripts/run_external_baselines.sh \
  --regime contact \
  --out "$BASE_OUT/external_baselines_v48_111" \
  --gpus 0,1 --jobs-per-gpu 3 --max-parallel 6 --max-scenarios 0 --womd-role validation
```

Then obtain publication latency in a separate uncontended pass, for example:

```bash
bash scripts/profile_external_baselines_latency.sh \
  --regime safe \
  --source-run "$BASE_OUT/external_baselines_v48_111" \
  --gpu 0 --max-scenarios 0 --womd-role validation

bash scripts/profile_external_baselines_latency.sh \
  --regime near \
  --source-run "$BASE_OUT/external_baselines_v48_111" \
  --gpu 0 --max-scenarios 0 --womd-role validation
```

Use the same GPU type, power mode, precision mode, and software environment for every baseline latency row. Run Contact similarly if a Contact baseline has learned GPU inference that needs an uncontended latency number.

## 8. Files changed in the optimized tree

Core changes are in:

- `configs/external_baselines/diffusion_planner.yaml`
- `configs/external_baselines/flow_planner.yaml`
- `src/ocrap/external_baselines/generative_ports.py`
- `src/ocrap/external_baselines/models.py`
- `src/ocrap/external_baselines/train.py`
- `src/ocrap/external_baselines/evaluate.py`
- `src/ocrap/external_baselines/provenance.py`
- `src/ocrap/external_baselines/third_party/dpm_solver_pytorch.py`
- `src/ocrap/simulation/closed_loop_runner.py`
- `tools/build_regime_comparison_tables.py`
- `scripts/run_external_baselines_safe.sh`
- `scripts/run_external_baselines_near.sh`
- `scripts/profile_external_baselines_latency.sh`
- `CLEAN_COMMANDS.md`
- regression tests under `tests/`.

The vendored DPM-Solver code includes a notice file and is used to preserve the uploaded Diffusion Planner sampler semantics rather than replacing them with a homemade one-step approximation.

## 9. Verification performed

After the final changes, the full OC-RAP unit/regression suite was executed in the optimized tree:

```text
216 passed in 8.54s
```

The new tests exercise native Diffusion/Flow sampling paths, deterministic seeded evaluation, external learned-method registration, numerical losses, launcher contracts, and steady-state timing aggregation. They do not substitute for end-to-end Waymax replay on the user's WOMD mount.

## 10. Remaining limitations before calling the results “reproduced baselines”

For a strong paper claim, use wording such as **“source-derived WOMD interface port”** rather than “official implementation”. Diffusion Planner remains ego-only because neighbor-future supervision is absent from serialized OC-RAP samples; Flow Planner omits source static normalization and current-state perturbation/quintic augmentation; both use a shorter common executable horizon and candidate-lattice projection; neither is checkpoint-compatible with author weights. These are transparent interface adaptations, not hidden equivalences.

If exact Diffusion Planner fidelity is required, the next engineering step is to add a raw-WOMD future-join cache keyed by scenario/time and train the joint ego+neighbor diffusion target. If exact Flow Planner preprocessing fidelity is required, compute train-split static normalization statistics and implement the source perturbation/quintic augmentation before training. Those additions should be versioned as a new fidelity tier rather than silently folded into v61.
