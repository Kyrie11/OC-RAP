# BeTop + Plan-R1 external-baseline audit and v62 optimization

## Scope

This audit compares the uploaded `Plan-R1` and `BeTop` papers/source trees against the current OC-RAP external-baseline adapters, then checks the WOMD/Waymax bridge, the one-regime-at-a-time launcher contract, metric attribution, and timing boundaries. The target publication interface remains OC-RAP's common executable candidate set; teacher/counterfactual labels are not exposed to deployment-time baseline inputs.

The optimized implementation is tagged **v62**. It preserves the existing `CLEAN_COMMANDS.md` Section 4 commands; no command-line change is required.

## Executive verdict

| Baseline | Uploaded source availability | v60/current input state | v62 status | Publication label that is supportable |
|---|---|---|---|---|
| Plan-R1 | Full nuPlan training/planning source + paper | Approximate tokenization; predictor/planner trained together; moving KL reference; candidate-level surrogate GRPO; non-source residual candidate prior | Source tokenizer geometry/reconstruction, 1024 codebook, 6x128/8-head token LM settings, two-stage 32+5 schedule, frozen predictor, token-level VD-GRPO/KL operator restored. Common WOMD scene encoder and candidate reward interface remain adaptations. | **paper/source-core partial reproduction / candidate-rollout adapted** |
| BeTopNet planning | Paper gives planning architecture; uploaded repo releases full WOMD **prediction** implementation, not the nuPlan planning implementation | Source-structured topology adapter but planning dimensions/K/topology weight differed from paper; observation-only proxy topology; common candidate head | Planning D=128, 4 layers, K=32, 2 s history, 256 map polygons, λ1=50, 25 epochs, t_b=3 and λ_m=0.5 restored where the paper specifies them. Exact 6-mode cascaded/branched planning head cannot be checkpoint-reproduced from supplied public source and remains a common-lattice adapter. | **paper-core topology partial reproduction / planning-head adapted** |

The key distinction is important: **Plan-R1 can preserve substantially more of the released source algorithm than BeTop planning**, because Plan-R1 ships its planner source. The uploaded BeTop repository is a WOMD prediction release; claiming an exact official BeTop planning reproduction would overstate what the supplied artifacts support.

## 1. Plan-R1: source-to-port audit

### 1.1 What the uploaded source actually does

Relevant source locations:

- `Plan-R1-main/transforms/token_builder.py`
- `Plan-R1-main/utils/process_data.py`
- `Plan-R1-main/model/PlanR1.py`
- `Plan-R1-main/config/train/pred.yaml`
- `Plan-R1-main/config/train/plan.yaml`
- `Plan-R1-main/rewards/*.py`

Core source behavior:

1. Motion is discretized with a **1024-token vehicle codebook**. `TokenBuilder` downsamples with `interval=5` (0.5 s at 10 Hz), measures each next displacement relative to the **previous reconstructed token pose**, and chooses the codebook entry with minimum **average corresponding-corner distance**. With the default `[1,1,1,1]` shape, the source corner helper divides each extent by two.
2. Predictor pretraining is next-token classification with label smoothing 0.1. The source configuration uses hidden size 128, 8 heads, 6 attention layers, dropout 0.1, AdamW 3e-4 / weight decay 1e-4, 32 epochs.
3. Planner training is a **separate second stage**, initialized from the predictor. The predictor/reference remains fixed. Planner fine-tuning uses 5 epochs, learning rate 4e-6, `num_samples=4`, `beta=0.1`, and fixed advantage scaling `c=0.1`.
4. The source process advantage explicitly group-centers rewards and divides by the fixed scaling constant; the standard-deviation normalization code is commented out. It then computes a reverse cumulative token advantage over valid rollout steps.
5. The policy loss uses the zero-value/non-zero-gradient ratio `exp(logp - stopgrad(logp))`; the reverse-KL estimator is `exp(ref_logp-plan_logp) - (ref_logp-plan_logp) - 1`.
6. The native rollout is **multi-agent**: the trainable ego planner is coupled to a frozen predictor/world model for other agents. Ego actions are sampled during rollout (source `rollout_top_k=50`); deployment uses top-1 token selection.
7. Rule rewards combine hard on-road / obstacle-collision / agent-collision factors with comfort, TTC, speed-limit and progress terms. Source weights are comfort 2, TTC 5, speed 2, progress 1.

### 1.2 Bugs / fidelity gaps found in the old OC-RAP Plan-R1 adapter

The prior port was not source-faithful in several material ways:

- Tokenization used direct observed-pose deltas plus an ad-hoc XY/yaw quadratic distance instead of source iterative reconstruction + average-corner distance.
- Predictor and planner objectives were optimized in the same training pass, so the supposed KL reference was not a fixed source predictor.
- The planner contained an extra candidate prior that is not part of Plan-R1.
- The token head was a single linear layer rather than the source two-layer MLP head.
- Stage-2 GRPO operated on a softmax across OC-RAP candidate sequence scores instead of token log-probabilities.
- Hyperparameters were 4 layers / 28 total epochs instead of the released 6-layer and 32+5 stage schedule.

### 1.3 v62 Plan-R1 fixes

Modified files:

- `src/ocrap/external_baselines/generative_ports.py`
- `src/ocrap/external_baselines/models.py`
- `src/ocrap/external_baselines/train.py`
- `configs/external_baselines/plan_r1.yaml`

Changes:

- Uses the copied author codebook at `assets/external_baselines/planr1_tokens_1024.pt`.
- Precomputes source-equivalent token corners once and performs iterative reconstruction tokenization. This is both a correctness fix and a safe speedup.
- Restores two-layer token heads.
- Separates predictor and planner scene/token/transformer/head parameters.
- Runs 32 predictor epochs then clones predictor weights into the planner and freezes the predictor for 5 planner epochs.
- Overrides `train()` so the frozen predictor stays in eval mode; otherwise recursive `model.train()` would re-enable dropout and make the KL reference stochastic.
- Wraps frozen predictor evaluation in `torch.no_grad()` during stage 2, reducing activation memory/runtime without changing the objective.
- Stage 1 is pure source-style token CE. Stage 2 is token-level ratio/KL, with fixed group centering and no reward-variance normalization.
- Removes the non-source candidate-prior residual.
- Uses the source learning rates (3e-4 pretrain, 4e-6 fine-tune), β=0.1 and c=0.1.

### 1.4 Unavoidable Plan-R1 adaptations

These are not hidden as "exact reproduction":

- The source is nuPlan-native and has a factorized agent/map/temporal graph backbone. OC-RAP uses its observation-only WOMD scene bridge. The v62 port is therefore not author-checkpoint compatible.
- Serialized OC-RAP regime groups intentionally do **not** contain surrounding-agent future GT. Consequently, the exact source all-agent token pretraining and frozen reactive world-model rollout cannot be reproduced without a raw-WOMD future join.
- Stage-2 uses the common executable candidate group and common candidate safety/utility reward proxy. The source instead samples ego token rollouts against the frozen multi-agent predictor and computes rule rewards over rollout steps.
- Because the common candidate reward is one scalar per candidate, v62 broadcasts the centered candidate advantage over valid candidate token decisions. It keeps the source token-level policy/KL operator, but it is not identical to source per-step process-reward supervision.
- OC-RAP's executable prefix is shorter than Plan-R1's native 8 s rollout.

These limitations are recorded in `src/ocrap/external_baselines/provenance.py` so result packaging does not accidentally relabel the adapter as an exact source reproduction.

## 2. BeTopNet: source-to-port audit

### 2.1 What is actually released

The uploaded `BeTop-main` tree contains a full WOMD **prediction** implementation under:

- `womd/betopnet/models/*`
- `womd/betopnet/utils/topo_utils.py`
- `womd/tools/cfg/BeTopNet_e2e_6.yaml`

The paper, however, describes a separate nuPlan planning variant. The supplied repository does **not** contain a drop-in source implementation of that planning system. Therefore exact planning-weight/checkpoint reproduction cannot be verified from the uploaded source alone.

### 2.2 Paper planning components that can be checked

Appendix C specifies:

- planning history: 2 s at 10 Hz; ego keeps current state to reduce closed-loop/open-loop gap;
- 256 map segments × 20 points;
- hidden dimension D=128;
- 4 scene encoder layers and 4 planning decoder layers;
- 6 learnable planning modes;
- topology-guided local attention selecting K=32 agents;
- cascaded planning: short plan first, branching time `t_b=3`, then `M_J=6` branches per mode;
- contingency recombination uses top `K_M=4` interactive agents;
- imitation loss `L_V + λ1 L_E`, `λ1=50` for topology BCE;
- contingency weight `λ2=5`;
- training 25 epochs for planning;
- inference combines confidence and short-term term with `λ_m=0.5`.

The released topology utilities also allow exact checking of two operators:

- actor behavior braid: trajectory-pair segment crossing after the source coordinate transform;
- map braid: at each future time, only the **nearest valid polyline** is marked when distance <3 m, then labels are accumulated over time.

### 2.3 Old adapter issues and v62 fixes

The old expanded BeTop adapter had useful source structure but several paper-planning mismatches: D=192, 16 topology agents, `num_topo=24`, 1.1 s history, 64 source map polygons, topology loss weight 10, and 30 training epochs.

v62 changes `configs/external_baselines/betopnet.yaml` to:

- D=128;
- 4 layers;
- K=32 topology agents / `num_topo=32`;
- 21 history samples (2 s past plus current at 10 Hz in the OC-RAP tensor convention);
- 32 source agents;
- 256 source map polygons × 20 points;
- topology weight λ1=50;
- 25 training epochs;
- `t_b=3`, `λ_m=0.5` retained for contingency selection.

The existing data bridge already had the important source-derived map-label correction: exactly one nearest polyline can be activated per future time before temporal accumulation. Candidate-independent actor/map sorting is performed once per group and reused across all candidate prefixes.

### 2.4 Unavoidable BeTop planning adaptations

- There is no supplied official planning implementation to reproduce checkpoint-for-checkpoint.
- The paper's 6-mode cascaded + 6-branch head and top-4-agent joint recombination are represented through the common executable-candidate interface rather than an official planning decoder.
- OC-RAP does not serialize neighbor futures. Actor behavior-braid supervision therefore uses **observation-only constant-velocity extrapolation**. It is a proxy and must not be called source-equivalent GT topology.
- The paper defines a repulsive potential as a cost while also writing inference as a highest-score `p + λ_m C_M`. The supplied planning code needed to disambiguate the learned cost sign is absent. OC-RAP therefore keeps an explicit, documented safe convention: positive observed repulsive potential is a penalty to confidence. This avoids silently rewarding smaller separation but is an interface interpretation, not a verified author implementation.
- Native 8 s trajectories are compared through OC-RAP's shorter common executable prefix.

## 3. WOMD / regime integration

### 3.1 No deployment-time teacher leakage

Both adapters consume the common observation-side tensors: ego/agent history, current state, map polylines/route, and executable candidate prefixes. `allow_teacher_supervision: false` remains set. OC-RAP counterfactual teacher labels are not model inputs at deployment.

### 3.2 Raw WOMD test path

The existing one-regime launcher remains the correct entry point and can resolve raw validation TFExamples using the existing WOMD role/path plumbing. The expected publication source remains standard WOMD validation, e.g. the user's machine path:

`/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation/`

This environment does not mount that `/data0` dataset, so I could validate code contracts and unit/integration tests but could not execute a full raw-WOMD Waymax closed-loop replay here.

### 3.3 Regime placement

The provenance registry keeps:

- **Plan-R1** as a Near-Contact supplementary external baseline;
- **expanded BeTopNet** as a Near-Contact supplementary external baseline;
- the historical `betopnet_lite` architecture/topology control remains in the Safe supplementary set and should not be presented as the official full BeTop planning reproduction.

The Section 4 one-regime commands remain unchanged. `scripts/run_external_baselines_near.sh` version tags were updated to v62 but its CLI contract was not changed.

## 4. Metric audit

### Safe regime

For any Safe control that includes `betopnet_lite`, evaluate it with the same Safe closed-loop metrics used for all Safe baselines. Do not import BeTop paper WOMD minADE/minFDE or nuPlan composite score as OC-RAP Safe metrics.

### Near-Contact regime: Plan-R1 and BeTopNet

Plan-R1 and expanded BeTopNet are Near-Contact baselines and therefore must be compared with the OC-RAP Near metric suite. In particular, the current closed-loop runner reports low-tail clearance and TTC/exposure quantities consistently across all Near baselines, including `min_clearance_m_p05`, `ttc_s_p05`, `critical_ttc_exposure_duration_s`, and near-zero-clearance exposure. Their paper-specific training rewards/scores are **not** substituted for these benchmark metrics.

TTC remains correctly described as an observation-side swept oriented-box / constant-velocity and frozen-heading TTC proxy. It must not be described as a learned prediction of true future time-to-collision.

### Contact regime

Neither Plan-R1 nor expanded BeTopNet is registered as a Contact supplementary baseline. They should not receive post-contact metrics in the main comparison. The common Contact runner still maintains the fixed object-identity distinction between re-contact with the same object and secondary overlap with a different object.

### Latency

Publication latency must use `scripts/profile_external_baselines_latency.sh`, not the normal 3-jobs-per-GPU × 2-GPU throughput launcher. The timing path synchronizes CUDA and excludes warm-up from steady-state aggregation. Report at least steady-state mean / p50 / p95 per planning call on one isolated process/GPU. Do not compare throughput-run wall-clock samples as planner latency.

## 5. Safe acceleration changes

The criterion used here is: remove redundant work **without changing the baseline operator or evaluation semantics**.

### Plan-R1

Safe optimizations now enabled:

1. Precompute codebook corner geometry once instead of rebuilding it for every token comparison.
2. Vectorize codebook-to-target corner distance over all 1024 tokens.
3. Do not execute the uninitialized planner branch during stage-1 validation.
4. During stage 2, run the frozen predictor under `torch.no_grad()` and force it to eval mode. This saves activation memory and avoids stochastic dropout while leaving KL logits unchanged.
5. Inactive OC-RAP auxiliary losses remain skipped rather than computed and multiplied by zero.
6. Existing AMP / fused AdamW / pinned and persistent DataLoader workers remain enabled where supported.

Unsafe/non-equivalent shortcuts intentionally **not** taken:

- reducing the 1024-token vocabulary;
- reducing 6 attention layers for publication runs;
- merging pretrain and planner fine-tuning;
- making the KL reference trainable;
- replacing the fixed-scale VD-GRPO centering with standard-deviation normalization.

### BeTopNet

Safe optimizations already preserved:

1. Candidate-independent scene actor/map sorting and truncation is cached once per group.
2. Only K=32 topology-selected agents participate in the local interaction aggregation, matching the paper planning K rather than attending every actor.
3. Inactive losses are skipped.
4. AMP/fused optimizer/DataLoader optimizations remain enabled.

Not treated as "safe acceleration": reducing K below 32, D below 128, fewer than four planning layers, reducing the 2 s history, or lowering topology loss solely for speed. v62 in fact restores these paper settings even where they cost more than the old approximate adapter.

## 6. Verification

Executed in the packaged tree:

```text
python -m compileall -q src
pytest -q
219 passed
```

Added `tests/test_planr1_betop_v62_fidelity.py` to guard:

- Plan-R1 source corner geometry;
- predictor→planner clone/freeze transition;
- frozen-reference eval behavior after `model.train()`;
- Plan-R1 source-stage hyperparameters;
- BeTop paper planning hyperparameters.

A full Waymax raw-WOMD closed-loop run must still be executed on the machine where the user's validation TFRecords exist.

## 7. Recommended result wording

Use these labels in tables/captions:

- **Plan-R1 (source-core WOMD token/VD-GRPO adapter)** — partial source reproduction; common candidate rollout/reward interface adapted.
- **BeTopNet (paper-core topology WOMD planning adapter)** — paper-core topology reproduction; planning head/interface adapted because the supplied public repo does not release the paper's nuPlan planning implementation.

Avoid claims such as "official exact reproduction" or "author-checkpoint equivalent" for either adapter, and especially for BeTop planning.
