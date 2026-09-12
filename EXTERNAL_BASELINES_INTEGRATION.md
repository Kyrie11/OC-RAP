# Supplementary external-baseline integration

This integration adds source-derived OC-RAP ports for Diffusion Planner, Flow Planner,
Plan-R1, and an expanded BeTopNet adapter. The ports deliberately reuse OC-RAP's
existing regime datasets, candidate-lattice contract, offline evaluator, Waymax
closed-loop runner, and timing instrumentation so that metric definitions remain
comparable with the already-integrated external baselines.

## Regime assignment

| Method | Primary regime | Rationale |
|---|---|---|
| Diffusion Planner | Safe | Nominal multimodal planning with conditional diffusion and guidance is best matched to collision-free / ordinary planning. |
| Flow Planner | Near-Contact | Its scene-level interaction modeling, overlapping trajectory tokens, flow matching, and CFG are most informative when ego/agent futures are tightly coupled before contact. |
| Plan-R1 | Near-Contact | Its safety-aligned reward design (collision/TTC/rules/comfort) acts primarily before impact; VD-GRPO is useful for rare safety-critical groups, but the method is not a post-impact recovery controller. |
| BeTopNet | Near-Contact | Behavioral-topology / interaction reasoning is naturally a pre-contact interaction planner. The prior lite implementation is retained; `betopnet` now selects the expanded adapter. |

No new method is assigned to Contact. Treating any of the four as a post-impact
controller would change its published problem definition because none provides an
explicit post-impact state-recovery / secondary-collision-control mechanism.

## Fidelity and adaptation boundary

These are source-derived benchmark ports, not binary/checkpoint-compatible copies of
the authors' repositories. The uploaded repositories use different data and planner
interfaces (for example, native trajectory generation / native simulator plumbing).
OC-RAP needs every external method to consume the same regime examples and execute
through the same Waymax candidate/action path. The port therefore preserves the
method-specific mechanism while projecting it onto OC-RAP's common candidate lattice.

* `diffusion_planner`: VP-style corruption, timestep-conditioned denoising, scene
  conditioning, and x0 reconstruction training. At evaluation time candidate energies
  are scored in one vectorized denoising pass rather than running a full multi-step
  DPM solver separately for every OC-RAP candidate.
* `flow_planner`: conditional-OT flow matching, overlapping fine-grained trajectory
  tokens, classifier-free condition dropout, and velocity-field supervision. Candidate
  scoring uses a vectorized midpoint flow-energy projection rather than a separate ODE
  solve per lattice candidate.
* `plan_r1`: uses the uploaded author's 1024-entry Vehicle motion-token codebook,
  autoregressive next-token modeling, reference/planning policies, and VD-GRPO-style
  group centering with a fixed scaling factor (no per-group standard-deviation
  normalization). OC-RAP's existing candidate labels provide the safety/utility reward
  proxy so the method remains on the common dataset contract.
* `betopnet`: keeps the prior topology encoder/fuser and top-K topology attention, and
  extends it with source-scene context plus prefix-trajectory confidence correction.
  The old implementation remains available as `betopnet_lite`.

This design makes the comparison executable and metric-consistent, but results should
be described as OC-RAP ports/adaptations rather than exact reproduction of the
original authors' reported numbers.

## Files added / changed

New implementation/config assets:

* `src/ocrap/external_baselines/generative_ports.py`
* `configs/external_baselines/diffusion_planner.yaml`
* `configs/external_baselines/flow_planner.yaml`
* `configs/external_baselines/plan_r1.yaml`
* `configs/external_baselines/betopnet.yaml`
* `assets/external_baselines/planr1_tokens_1024.pt`
* `tests/test_new_external_baseline_ports.py`

The common external-baseline bridge was extended in:

* `src/ocrap/external_baselines/{models,data,train,evaluate,policies,provenance}.py`
* `src/ocrap/simulation/closed_loop_runner.py`

Launchers were updated in:

* `scripts/run_external_baselines.sh`
* `scripts/run_external_baselines_safe.sh`
* `scripts/run_external_baselines_near.sh`
* `scripts/run_external_baselines_contact.sh`
* `CLEAN_COMMANDS.md`

## Dataset / WOMD behavior

Training and offline testing continue to use the existing OC-RAP regime dataset roots
(e.g. `$OCRAP_ROOT/train_safe`, `test_near`, etc.). Closed-loop replay uses the raw WOMD
root already expected by the project:

`/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/`

`--womd-role validation` is kept as the default publication replay role. The launcher
still supports `--womd-role validation_interactive` explicitly when that source split
is desired; the integration does not silently substitute one role for the other.

All four additions pass through the existing closed-loop runner with
`closed_loop.profile_timing=true`, so planner latency is measured with the same timing
contract as the existing external baselines.

## One-regime commands and GPU scheduling

The commands in `CLEAN_COMMANDS.md` now use:

```bash
--gpus 0,1 --jobs-per-gpu 3 --max-parallel 6
```

The slot list is `GPU0,GPU1,GPU0,GPU1,GPU0,GPU1`, hence at most three active jobs are
assigned to each GPU and six jobs run concurrently. If a regime has more than six
enabled methods, the dynamic scheduler launches the next method on the GPU whose slot
becomes free.

Safe enables the historical six methods plus Diffusion Planner. Near-Contact enables
the historical six controls plus Flow Planner, Plan-R1, and expanded BeTopNet. Contact
keeps the six historical post-impact baselines. Historical table-only behavior can be
restored with `RUN_SUPPLEMENTARY_SAFE=false` and/or `RUN_SUPPLEMENTARY_NEAR=false`.

Example:

```bash
bash scripts/run_external_baselines.sh \
  --regime near \
  --out "$BASE_OUT/external_baselines_v48_111" \
  --gpus 0,1 --jobs-per-gpu 3 --max-parallel 6 \
  --max-scenarios 0 --womd-role validation
```

Non-learning controls are registered rather than fitted; learned baselines are trained
or reuse a validated checkpoint. Offline and closed-loop evaluation are scheduled
through the same six-slot GPU queue.

## Validation performed in this package

The following checks passed in the integration workspace:

* 18 external-baseline tests (`pytest`) including launcher contracts, runtime
  regressions, regime/provenance checks, and synthetic forward/loss checks for all four
  newly integrated methods.
* `py_compile` for every modified Python runtime module.
* `bash -n` for the wrapper and Safe / Near-Contact / Contact launchers.
* config loading for all four new YAML files.

A real full WOMD training/Waymax benchmark run was not possible in the packaging
sandbox because the user's `/data0/...` datasets and target GPUs are not mounted here.
The scripts are therefore statically/unit validated but still require the intended
machine for full numerical reproduction and GPU-memory validation.
