## v48.46 — OC-SWIC / OBSERVATION-CLASS SHARED-WITNESS IDENTIFIABILITY CALIBRATION (2026-08-12)

**类别：有效算法迭代 + execution-equivalent 性能优化。没有新增 Safe/Near/Contact identifier、router、regime-specific policy/threshold/loss/budget。**

### v48.45.6 authoritative 2x2 attribution

本轮上传的 v48.45.6 A/B/C/D(Main) 四个 arm 均为 authoritative `RC=20`，且
`pipeline_valid=true`、certificate 已执行、Natural gate 已评估、`test_roots_read=false`。
四个 arm 的 dedicated-calibration protocol seal、Balanced/Precision source checkpoint
SHA256、gate protocol 与所有 development/certificate manifest SHA256 完全一致；因此这是
v48.45 SOWR 第一轮真正可进行 B-A、C-A、D-B-C+A 因果归因的有效负结果，而非工程失败。

Precision selector 的主要结果：

- A/Near certificate recall `0.1111`，harmful-selected UCB90 `0.2361`；deployability
  harmful-vs-safe-positive AUC `0.4510`，safe-positive false veto `14/16`。
- A/Contact certificate recall `0`，UCB90 `0.2921`；deployability AUC `0.5252`，false veto
  `30/31`；development safe-positive `pred_adv>=0` 仅 `1/37`。
- B(root+margin) 将 Near/Contact candidate safe-positive AUC 分别提高约 `+0.026/+0.027`，
  Contact certificate 首次得到 `1/20` positive recall `0.05`，false veto `30/31 -> 29/31`；
  但 Near/Contact certificate harmful UCB90 分别恶化到 `0.2980/0.3151`，不满足风险约束。
- C(obs-only) 的主要正信号是 certificate harmful UCB90：Near `0.2361 -> 0.1532`、Contact
  `0.2921 -> 0.2368`；Contact development `pred_adv>=0` 从 `1/37 -> 2/37`。但 Near
  certificate positive capture 从 `1` 降为 `0`，无法单独作为主算法。
- D/Main 基本退化为 B；它没有同时保留 C 的 risk suppression 与 B 的 capture signal。
  Near/Contact certificate UCB90 为 `0.2980/0.3189`，Contact development sign 又回到 `1/37`。
  因而 v48.45 joint SOWR 不吸收为成功模块。

Balanced selector 给出相同方向的弱证据：B/C/D 可把 Contact development recall 从
`0.1176` 提到 `0.1765`，但 certificate 仍只有 `0.05`，Near 仍极弱，且 risk/generalization
不稳定。结果支持“上游 witness/option semantics 有问题”，不支持继续堆 downstream capacity。

### SOWR 组件归因与保留/拒绝

v48.45.6 witness validation 显示 B/D 的 margin、deployability、oracle、option-q/admission/best-option
loss 都下降，但 `root_loss` 反而略升；C 的 observation loss 明显下降，同时有 certificate risk
suppression。由此：

1. **拒绝 root-logit recalibration。** 它没有独立正证据，且与 margin 更新绑定时 root objective
   反向漂移。v48.46 所有新 witness stage 都 byte-level 冻结 `root_logit_head`。
2. **保留 observation recalibration 作为风险侧候选。** 它只能被视作局部正信号，不能单独宣称
   解决 positive capture。
3. **保留 margin recalibration 作为 capture 侧候选。** B 的弱 AUC/Contact-positive 改善与
   margin/option loss 改善一致，但必须摆脱 root update 并验证不再用 harmful risk 换 recall。
4. **拒绝 simultaneous joint SOWR。** D 没有正 interaction，因此 v48.46 改为
   `observation -> freeze -> margin -> freeze` 的顺序识别，不再联合更新三个 witness heads。

### 根本算法语义审计：历史 global-one-option 与论文 OC-MERO 不一致

本轮对论文、`src/ocrap/algorithms/ocmero.py` 与历史 v48.5+ DRS 辅助路径进行逐项审计后发现：
核心 OC-MERO 一直正确实现为对每个 post-prefix observation-conditioned row `q[i,l]` 先
`max_l`，再在 roots 上做 outer LCVaR。也就是说，不可区分的 roots 通过 compatibility-weighted
lower-tail `q[i,l]` 被迫共享 compatible option，但**不同的可区分 post-prefix observation class
可以选择不同 recovery option**。

历史 v48.5 的“exact policy contract”却把它解释成“整个候选所有 roots 只能选择一个 globally
shared option”，并把这一更强约束扩散到 best-option/DRS auxiliary loss、teacher-PCD、calibration
和部分 execution diagnostics。这个约束不是论文 Eq. OC-MERO 所要求的 observation consistency。
当 Safe 的 best-option diversity≈1 时影响极小；Near/Contact 的 best-option diversity≈1.4–1.6，
它会系统性制造 false veto。当前 Precision certificate 的 `14/16` Near 和 `29–30/31` Contact
safe-positive deployability false veto 与这一语义错配高度一致。

v48.46 不通过降低 harm budget 或放松 observation consistency 来修复，而是新增显式、可审计的
两种训练监督语义：`global`（历史对照）与 `observation_class`（论文一致）。新的
`best_observation_consistent_option_indices()` / `predicted_observation_consistent_option_success()`
直接从 OC-MERO `q[i,l]` 做 row-wise option selection；class-specific success/best-option losses
只移除对**可区分 classes**的全局绑死，compatible roots 仍被 q 的 observation-kernel lower-tail
共同约束。

### v48.46 scientific 2x2：固定同一评价尺子

为了避免“改变算法同时改变 certificate ground-truth 定义”的混杂，四个 arm 的最终
calibration/certificate/closed-loop **全部固定为论文一致的 `observation_class` execution
semantics**。2x2 的因素只改变训练监督和 staged witness：

- **A:** legacy `global` option-witness/teacher supervision；无 staged witness。
- **B:** `observation_class`-aligned option-witness/teacher supervision；无 staged witness。
- **C:** legacy `global` supervision + sequential `obs -> margin` witness。
- **D/Main:** `observation_class`-aligned supervision + sequential `obs -> margin` witness。

因此 B-A 是“训练语义与 OC-MERO 对齐”的主效应；C-A 是“顺序 witness identifiability”的主效应；
D-B-C+A 是二者 interaction。A/B/C/D 共享同一个 paper-consistent gate protocol、dataset labels、
risk budgets、source checkpoint、dual-ROCT、top-k=5 和 shared continuous rule；comparator 对任何
identity mismatch fail-closed。

顺序 witness 的 obs stage 只允许 `obs_embed_head` 更新，margin stage 只允许 `margin_head` 更新；
`root_logit_head`、shared encoder/root decoder/direct policy/OCAF/ROCT heads 全部冻结。每一 stage
复用 v48.45.6 source-architecture/isolation contract，禁止 hidden partial checkpoint load。

### Pre-registered v48.46 go/no-go

本轮不以“loss 下降”作为成功条件：

- **Near:** 优先要求 certificate safe-positive deployability false veto 从 `14/16` 明显下降，
  screening target `<=10/16`；deployability harmful-vs-safe-positive AUC 向 `>=0.56` 移动；
  verify recall 至少接近 `0.20`，并保持 harmful-selected UCB90 `<=0.25`。
- **Contact:** development safe-positive `pred_adv>=0` 必须从 `1–2/37` 显著增加，screening target
  约 `>=10/37`；certificate recall 向 `0.15–0.20` 移动，harmful-selected UCB90 `<=0.25`，
  false veto 必须显著低于 `29/31`。
- **D/Main:** 必须同时保留 B 的 capture/frontier 改善和 C 的 risk suppression；若 interaction
  再次为负，不吸收 sequential witness package。
- **Safe:** 只有 RC=0 后才进入 paper-level paired closed-loop non-inferiority；不因为 Safe 数据简单
  而添加 Safe-specific strategy。

内部 CCF-A readiness 仍不是 venue 官方阈值：Near verify recall 目标约 `0.25–0.33`、harmful UCB
`<=0.25`、precision LCB `>=0.40`；Contact recall `0.20–0.30`、UCB `<=0.25`，并需要 secondary
collision、post-contact TTC、stable-stop/rejoin 等 closed-loop 改善；Safe 需要严格 nominal utility /
progress / intervention / FRA non-inferiority。当前 v48.45.6 与这些目标仍有明显距离。

### 不重复的历史失败方向

遵守 v48.45 stop rule，本轮明确**不**增加 ROCT width/scale，不放松 harmful budgets，不 densify
threshold grid，不扩大 top-k，不做 positive oversampling，不再堆 generic pairwise/listwise rank
loss，不恢复 generic harm residual/unbounded factor，不重做 v48.42 partial pooling/rank skip，
不恢复 v48.43 POET free alias transport，不 broad fine-tune encoder，也不加入 regime routing。

若 v48.46 仍失败，下一步才进入 recovery-option taxonomy/continuous-parameter coverage、margin teacher
calibration 和 observation-class identifiability 的数据/teacher 审计，而不是再加 downstream head。

### Execution-equivalent speed path

上传 v48.45.6 四个 arm 实际串行 wall time 约：A `59.2 min`、B `97.3 min`、C `71.6 min`、D
`67.6 min`。主要时间仍在 factor/witness training；同一 train/val NPZ 在不同 variant/stage 中重复
materialize 约 90–250 s/次，是剩余可去除的 I/O 开销。

v48.46 的加速不改变样本、epoch、batch、loss、seed、sampler、top-k 或 gate：

1. `OCRAPSampleDataset` 增加跨进程/跨 stage 的 **persistent decoded-tensor cache**。key 只绑定
   dataset manifest/path、tensor geometry 和 feature-construction settings；model head/optimizer/ROCT/
   option-semantic设置不改变 tensor，因此不再错误地制造不同 cache。`fcntl` lock + atomic replace
   保证 A/B 与 Balanced/Precision 并发 warm-up 时不会产生损坏 cache。
2. A/B **同时运行**，分别占 GPU0/GPU1；完成后 C/D 同时运行。每个 ablation 内 Balanced 与
   Precision 默认在该 arm 的同一 GPU 上并发，因为用户已确认单 arm 显存有 headroom。
3. 若单卡同时跑 Balanced+Precision 出现 OOM/吞吐下降，只设置 `V4846_VARIANT_MODE=serial`；
   两个 ablation 仍分别在两张 GPU 上并发，科学协议不变。
4. v48.45.6 已将单 A 从约 2.13 h 降到约 59 min；v48.46 进一步消除跨 stage 重复解压。按上传
   v48.45.6 wall time 粗略上界，A/B + C/D pairing 在无 contention 时可把四臂串行约 296 min
   降到约 169 min，再叠加 persistent cache 收益。真实值必须由 A30 实测，不在代码审计环境中宣称。
5. persistent cache schema v3 优先以 `weights_only + mmap` 读取，多个 arm/variant 可共享 OS page cache，
   避免每个 trainer 再复制一整份 decoded tensors 到 host RAM；旧 PyTorch 自动 fallback。cache 若损坏/截断
   会在 flock 内视为 miss 并原子重建，不再把缓存介质问题升级成实验 RC=30。
6. certificate config 同时显式写入 `training/calibration/evaluation.option_execution_semantics`，避免未来 config
   优先级或默认值改变时把 paper-consistent evaluation 静默退回 legacy global 语义。

### Legacy architecture debt kept out of this 2x2

共享 v48.45 rebuilt source 仍含历史 `DELTA_REGIME_EXPERTS=true`/bucket-conditioned internal geometry。
v48.46 **不新增也不扩大**这一 legacy 路径，并在 factor contract 中声明
`strategy_regime_conditioning=false`。为了保持 2x2 单因素可归因，本轮不同时重建 source 去除该
历史结构。若 v48.46 得到明确正结果，论文最终版前应单独做一次 controlled source rebuild，移除
legacy bucket-conditioned policy internals，并把 Appendix 的 “Regime-conditioned recovery admission”
改写为连续 deployable headroom + observation compatibility + harm envelope；该结构清理必须作为
独立实验，不能和 v48.46 混在一起。

### Engineering / attribution contracts and local validation

- 新增 `test_v48_46_ocswic.py`，验证 global-vs-observation-class 最小反例、class loss gradients、
  persistent cache key/mmap/损坏重建、explicit evaluation semantics、telemetry summary 与 2x2 launcher/factor contract。
- `compare_v48_46_ocswic_2x2.py` 要求四臂 source checkpoint SHA、protocol seal、**完整同一 gate
  protocol**、development/certificate manifest 完全一致；RC 必须为 authoritative `0/20`。
- 四臂最终 evaluation semantics 固定 `observation_class`；training semantics 才是可变因素，避免
  change-of-metric confounding。
- 当前本地完成回归：v48.42–v48.46 + v48.45 engineering/protocol 组合测试 `70/70`；
  v48.37–41 `29/29`；v48.36 OCAF `14 passed / 1 skipped`；v48.36 transfer/terminal/idempotence
  `18/18`。去重后合计 **131 passed / 1 skipped**。
- `python -m compileall -q src tools` PASS；全部 **103/103** shell scripts `bash -n` PASS。
- 本地环境没有用户 A30/WOMD 数据，因此未声称真实 v48.46 GPU/certificate 结果。


## v48.45.6 — SOWR STAGE-ISOLATION + SOURCE-ARCHITECTURE + EXACT-I/O HOTFIX (2026-08-12)

**类别：工程/执行性能修复；SOWR、dual-ROCT、shared rule、risk/gate 算法语义不变。**

### Uploaded v48.45.5 evidence

- Arm A is **not** a pipeline failure. It completed certificate and gate with `pipeline_valid=true`, authoritative `RC=20`; it is a valid negative algorithm result.
- Arms B/C/D(Main) all terminate before SOWR training with normalized `RC=30`, both Balanced and Precision failing in `shared_option_witness_recalibration` with the same constructor exception: `ValueError: ROCT requires component-head physical evidence`.
- The dedicated-calibration protocol introduced in v48.45.5 is valid in all four arms. The shared protocol seal is identical and no test root was read.
- Source checkpoint hashes are valid and identical across arms: Balanced `070218c66e506d66f25a12bf53b4127581992d75481bbb635d7ea658f4cfd352`, Precision `8f7528b76ce4b2424c5f153fe3109844de27a08a976a067b1292baee393f768d`.

### Root cause 1 — downstream ROCT environment leaked into witness-only SOWR

`run_v48_45_sowr_ablation_arm.sh` exports downstream v48.44-D `EVIDENCE_ROCT_BENEFIT=true` and `EVIDENCE_ROCT_DEPLOYABILITY=true`. B/C/D execute the optional SOWR stage before downstream factor adaptation. v48.45.5 called the generic trainer without stage-local overrides, while the generic trainer defaults `EVIDENCE_COMPONENT_HEADS=false`. The witness model therefore received `ROCT=true && component_heads=false` and correctly failed its constructor guard. A has no SOWR stage, so it does not hit this leak.

The repair does **not** remove the ROCT guard and does **not** fabricate reliability for the observed constant `hard_rule`/`harm_proxy` components. During SOWR only, all downstream OCAF/ROCT/component-evidence flags are explicitly disabled. After SOWR, downstream factor adaptation still runs the original dual-ROCT configuration with component heads enabled.

### Root cause 2 — hidden source-architecture drift after fixing root cause 1

The rebuilt v48.45 source uses a fixed architecture (`preference_head=false`, set tournament enabled 48/4/0.05 with replacement, `delta_mode=ordinal_evidence`, delta hidden 48/dropout .02, regime experts/policy features enabled). Generic trainer defaults differ materially. v48.45.5 SOWR did not pin the source architecture, so a simple ROCT-only fix could have produced partial `strict=False` checkpoint loads and silently replaced frozen direct-policy layers.

v48.45.6 therefore pins the exact rebuilt-source architecture in the SOWR subprocess and adds fail-closed pre/post contracts:

- `check_v48_45_sowr_source_architecture.py` validates immutable source metadata and required state prefixes before training.
- `check_v48_45_sowr_stage_isolation.py` verifies architecture metadata preservation, key-set preservation outside the allowed witness prefixes, exact trainable-prefix declaration, no downstream evidence flags in the SOWR checkpoint, and no parameter change outside the intended B/C/D witness heads. Epoch-zero/identity is allowed because “no beneficial update” is a legitimate algorithm outcome.

### Execution-equivalent I/O acceleration

Uploaded valid Arm A took about **7683.7 s = 2.13 h** end-to-end. The initial contracts/index phase was about 3.5 min, factor adaptation about 85 min, and certificate/calibration about 34–38 min. Both A30 jobs show a synchronized epoch-14 stall (`~1188 s` Balanced, `~1203 s` Precision) while ordinary epochs are roughly 140–267 s, consistent with shared compressed-NPZ/storage/CPU data-pipeline contention rather than a single-GPU compute failure.

v48.45.6 adds only execution-equivalent fast paths:

1. NPZ member-selective loading: model/index/calibration readers materialize only fields actually consumed by the computation.
2. Optional `training.cache_samples_in_memory`: each training process decodes the final flat feature + recovery-label CPU tensors once and reuses them for all epochs; raw BEV/map/debug arrays are not cached.
3. v48.45 launcher defaults to `ABLATION_CACHE_SAMPLES_IN_MEMORY=true`, `ABLATION_NUM_WORKERS=3`, `ABLATION_PREFETCH_FACTOR=3`; batch size, epochs, sampler, loss, LR, seed, top-k, ROCT parameters and gate are unchanged.
4. `dataset_materialization_done` logs cache time and byte count so the next real GPU run can quantify the improvement.
5. Certificate/support-index paths also use selective NPZ loading. Setwise candidate scoring semantics are unchanged; no cross-scene batching was introduced.
6. Keep `MAX_PARALLEL_ARMS=1`: every arm already runs Balanced on GPU0 and Precision on GPU1. Arm-level concurrency >1 puts multiple training processes on each GPU and reintroduces I/O/GPU contention.

### Resume and attribution safety

- `prepare_v48_45_6_resume.py` can preserve an existing authoritative RC=0/20 arm only when its protocol seal and current source checkpoint SHA256 match. Pipeline-invalid RC=30 arms are deleted for a clean retry.
- Fast engineering recovery can therefore keep the uploaded valid A and rerun only B/C/D. For paper-quality final 2x2 attribution, rerun all four from the same v48.45.6 checkout (`FORCE_RERUN_VALID_ARMS=1`).
- No Safe/Near/Contact identifier, router, policy, threshold or risk budget was added. `test_roots_read=false` remains a hard contract.

### Local validation

- v48.45.6 new tests: **7/7 passed**.
- v48.44 + v48.45 focused regression: **43/43 passed**.
- v48.40–v48.45 regression: **74/74 passed** (warnings are existing PyTorch nested-tensor warnings).
- v48.36 stage/terminal/OCAF regression: **32 passed / 1 skipped**.
- v48.37–v48.39 regression: **18/18 passed**.
- `python -m compileall -q src tools tests`: PASS.
- **100/100** shell scripts pass `bash -n`; operator command passes `bash -n`.
- repository-wide same-`local` nounset RHS audit: **0 findings**.
- The delivery environment has no user WOMD files or A30 GPUs, so no new real training/certificate result is claimed.

## v48.45.5 — DEDICATED CALIBRATION PROTOCOL BOOTSTRAP HOTFIX (2026-08-12)

**Category: engineering-only. The v48.45 SOWR algorithm, A/B/C/D factors, shared source,
ROCT/top-k/shared-rule settings, harm budgets, certificate semantics and Natural gate are unchanged.**

### Uploaded failure diagnosis

The uploaded v48.45.4 shared source is now valid: `S1_SOURCE_POLICY_STATUS.json` records
`balanced_exit_code=0`, `precision_exit_code=0`, both rebuilt checkpoints pass the
SHA256 source contract, and the source-quality contract is valid.  The uploaded A/B/C/D
SOWR runs all fail before adaptation with the same authoritative engineering state:
`stage=dataset_root_contract`, `raw_exit_code=4`, `pipeline_exit_code=30`,
`certificate_executed=false`, `gate_evaluated=false`.

The dataset contract shows the exact missing roots.  Every arm resolved the legacy
dedicated protocol root
`$OCRAP_ROOT/calibration_v48_14_prism_4814` and expected six derived directories:
`evidence_adapt_{train,dev}_{near_contact,contact}` plus
`certificate_pool_{near_contact,contact}`.  All six canonical-path checks were true but
all six `*_exists` checks were false.  `calibration_safe` existed.

The operator command was the defect: it validated `train_*`/`val_*` for source rebuild
but never invoked the repository's deterministic
`partition_dedicated_calibration_v48_14.py` on the actual
`calibration_near_contact` and `calibration_contact` roots.  The v48.36 controller was
therefore correct to fail closed; the missing protocol bootstrap made all four arms
guaranteed RC=30.  This upload contains no new SOWR algorithm evidence.

### Engineering fixes

1. Added `scripts/prepare_v48_45_protocol.sh`.  Before any arm starts it deterministically
   derives the dedicated protocol from `calibration_near_contact/contact` with the
   preregistered seed `4814`, adaptation-train fraction `0.45`, adaptation-dev fraction
   `0.15`, and certificate fraction `0.40`.  `calibration_safe` remains the safe
   calibration root.  No `test_*` root is read.
2. Added `tools/check_v48_45_protocol_seal.py`, an independent fail-closed seal.  It
   verifies source split IDs are exactly `calibration`, all six derived role manifests
   are non-empty, role labels are exact, every referenced sample exists, Near and
   Contact train/dev/certificate scene sets are mutually disjoint, their union exactly
   recovers each source calibration population, and every scene assignment exactly
   matches the seed/fraction hash rule.  Source and role manifest SHA256 hashes are
   recorded for attribution.
3. Protocol preparation is resumable.  A valid existing protocol is independently
   revalidated and reused without repartitioning.  An invalid/partial protocol is moved
   aside, rebuilt, audited and sealed; failure rolls back to the previous directory.
4. `run_v48_45_sowr_2x2_parallel.sh` now prepares/verifies this one shared protocol
   before launching A/B/C/D.  Direct single-arm invocation also prepares/verifies it,
   so the previous four-way repeated dataset-root failure cannot recur silently.
5. The new operator command explicitly validates all train/val/calibration manifests,
   prepares the shared protocol before deleting failed arm outputs, reuses the already
   successful v48.45 source, and captures launcher RC so any future engineering failure
   prints each arm's `PIPELINE_FAILED.json` instead of being mistaken for algorithm
   evidence.
6. Every arm recomputes a fast independent protocol-seal check at its boundary and records
   `protocol_seal_sha256` in `ATTEMPT_STARTED.json`.  The v48.45 comparator now fails closed
   unless all four arms have the same protocol seal, source checkpoint SHA256 pair, gate
   protocol SHA256 and certificate/development manifest SHA256 identities.  This prevents
   B-A/C-A/D-B-C+A from being reported when shared inputs drift.
7. Arm implementation-version strings are advanced to v48.45.5 for provenance only;
   `OCRAP_ALGORITHM_VERSION`, SOWR factor definitions and all algorithm hyperparameters
   remain unchanged.

### Attribution boundary

Safe/Near/Contact remain dataset/evaluation strata only.  No regime identifier, router,
per-regime policy, per-regime threshold or relaxed risk budget is introduced.  The
strict 2x2 remains A=no SOWR, B=root/margin witness only, C=observation-kernel only,
D=both, with one shared source and one shared scene-disjoint calibration protocol.
Only authoritative arm RC 0/20 results are eligible for B-A, C-A and interaction
`D-B-C+A` attribution.

### Validation

- New protocol/bootstrap focused tests: **4/4 passed**; complete v48.45 focused set:
  **29/29 passed**.
- v48.42-v48.45 regression: **56/56 passed**.
- v48.36 controller/terminal suites: **32 passed / 1 skipped**.
- v48.37-v48.41 regression: **29/29 passed**.
- Combined completed v48.36-v48.45 validation: **117 passed / 1 skipped**.
- `python -m compileall -q src tools tests`: PASS.
- **100/100** shell scripts pass `bash -n`; the v48.45.5 operator command also passes.
- Repository-wide nounset same-`local` self-dependency scan: **0 findings**.
- Synthetic end-to-end arm smoke passes `source_checkpoint_contract`,
  `dataset_root_contract`, `dedicated_protocol_audit`, multigroup loss and OCAF bridge,
  then stops only at the expected CUDA preflight because the validation container has
  no training GPU.  Thus the exact uploaded `dataset_root_contract` failure is removed
  from the real controller path before any algorithm stage.

## v48.45.4 — S1 NOUNSET SOURCE-REBUILD ENGINEERING HOTFIX (2026-08-12)

**Category: engineering-only.  The v48.45 SOWR algorithm, 2x2 factors, source data mix,
ROCT/top-k/shared-rule settings, harm budgets, certificate and gate are unchanged.**

### Uploaded failure diagnosis

The uploaded `ocrap_v48_45_source_rebuild_s7` is not an algorithm result.  S0 completed
successfully (`50114` train samples, `12250` validation samples, `13` epochs, best epoch
`7`, best validation loss `5.2515695061`, and an existing `best.pt` on the run machine),
but the source rebuild terminated at `S1_source_policy_heads`.  The retained status log
is exactly `balanced=1 precision=1`; neither S1 candidate directory nor its
`train_summary.json` was created.  The outer source driver then normalized the failed
S1 stage to RC=30.  No v48.45 SOWR adaptation/certificate/gate result can therefore be
attributed from this upload.

The direct cause is Bash nounset evaluation in `rebuild_v48_45_shared_source.sh`:

```bash
train_source_variant() {
  local variant="$1" gpu="$2" run="$SOURCE_OUT/candidates/$variant"
  ...
}
```

The script uses `set -u`.  Bash expands all RHS expressions before the `local` builtin
establishes those local variables, so `$variant` in the third assignment is unbound.
Both background S1 functions abort before `mkdir -p "$run/logs"`, producing the observed
`balanced=1 precision=1` signature.  A minimal shell reproduction returns the same two
exit codes.

### Engineering fixes

1. `train_source_variant` now declares `variant gpu run` first and assigns them on
   separate commands; S1 additionally writes `S1_SOURCE_POLICY_STATUS.json` with both
   raw exits before the outer RC=30 mapping.
2. Resuming an unsealed source removes only stale failure/status markers.  A complete S0
   (`best.pt` + `TRAINING_COMPLETE.json`) is preserved and reused; it is not retrained.
3. The new operator commands no longer delete the entire incomplete `SOURCE_RUN`, so the
   already-completed S0 from the uploaded attempt can be reused on the original machine.
4. `run_v48_45_sowr_2x2_parallel.sh` now defaults to `MAX_PARALLEL_ARMS=1`.  Each arm
   already runs Balanced on GPU0 and Precision on GPU1 concurrently, so this uses both
   GPUs while avoiding 8 concurrent train processes on two GPUs.
5. The same nounset-local dependency class was removed from the latent v48.36 ablation
   task launcher and both v48.34 video launchers.  A repository-wide shell test now
   scans every script for this exact unsafe `local a=... b=...$a` pattern.
6. The new operator command runs source hash/quality contracts before A/B/C/D and builds
   `ocrap_v48_45_sowr_2x2_comparison.json` only after all four arms are authoritative
   RC=0/20 results.

### Attribution boundary

No new Safe/Near/Contact identifier, router, regime-specific threshold or regime-specific
policy is introduced.  Because the uploaded run stopped before S1 training, no further
algorithm change is justified by this result.  The next algorithmic decision remains the
pre-registered SOWR 2x2: A=no witness recalibration, B=root/margin witness only,
C=observation kernel only, D=both.  Only after these four arms complete with RC 0/20 may
we decide whether SOWR is effective.

### Validation

- v48.42-v48.45 focused regression: **52 passed**.
- v48.45 focused regression: **25 passed**.
- `python -m compileall -q src tools tests`: PASS.
- **99/99** scripts pass `bash -n`; v48.45.4 operator command also passes `bash -n`.
- Repository-wide nounset same-`local` self-dependency scan: **0 findings**.
- A larger v48.36-v48.41 regression batch was started and reached its environment time
  limit without reporting a test failure; it is not counted as a completed validation.

## v48.45.3 — SOURCE-REBUILD EMPTY-OVERRIDE ENGINEERING HOTFIX (2026-08-11)

**Category: engineering-only; v48.45 SOWR algorithm and v48.45.2 source-rebuild attribution protocol are unchanged.**

### Uploaded v48.45.2 failure diagnosis

The uploaded `ocrap_v48_45_source_rebuild_s7` did not complete S0.  Its shared-backbone
log scans the pooled Safe/Near/Contact train and validation roots successfully, then
crashes before epoch 1 while constructing `OCRAPModel`:

```text
ValueError: could not convert string to float: 'None'
... direct_recovery_evidence_component_reliability
```

The shell trainer intentionally passes optional string settings as explicit empty CLI
overrides, e.g. `--set model.direct_recovery_evidence_component_reliability=` and
`--set training.init_checkpoint=` in scratch mode.  The generic override parser used
`yaml.safe_load("")`, which returns Python `None`.  `train.py` then converted the
reliability value with `str(None)`, producing the literal `"None"`; the model attempted
`float("None")` and S0 terminated.  Since S0 produced neither `best.pt` nor
`TRAINING_COMPLETE.json`, S1 never produced Balanced/Precision source checkpoints.
The uploaded A/B/C runs therefore all fail later at `source_checkpoint_contract` with
RC=30.  Adaptation, certificate and gate were never executed, so those runs are not
algorithm-attribution evidence.

### Engineering fixes

1. CLI `--set key=` now preserves the explicit empty string.  YAML null remains
   available explicitly via `key=null` or `key=~`.
2. Train-time model and ordinal-evidence reliability plumbing normalizes optional
   `None` to the original unspecified/default semantics instead of the string
   `"None"`.
3. Model, loss and inference/checkpoint paths defensively accept `None`, empty string,
   and legacy textual null spellings as an unspecified reliability vector, retaining
   the pre-existing default of all-one component reliability.  Nonempty numeric CSVs
   are unchanged.
4. `rebuild_v48_45_shared_source.sh` now writes `SOURCE_REBUILD_FAILED.json` with the
   exact stage (`preflight`, teacher-index build, S0, S1, sealing, or final contracts)
   on any nonzero source-rebuild exit.  This prevents a primary S0 error from being
   visible only later as missing source checkpoints.
5. The v48.45.3 operator commands explicitly delete only an **incomplete** source
   (never a sealed `SOURCE_REBUILD_COMPLETE.json`), run an empty-override semantic
   precheck, fail immediately if source rebuild is nonzero or the manifest is absent,
   and only then launch A/B/C/D.

### Attribution boundary

No loss weight, S0/S1 dataset mix, architecture, SOWR switch, ROCT parameter, top-k,
shared Natural rule, risk budget, certificate, gate, or regime-specific policy is
changed.  After rebuilding one fresh sealed source, A/B/C/D remain attributable within
that source identity.  The failed uploaded v48.45.2 A/B/C runs must be discarded.

### Validation

- Focused v48.13/v48.36-v48.45 regression: **117 passed / 3 skipped**.
- v48.45.3 exact S0 shell-argument capture confirms scratch `init_checkpoint`,
  freeze/trainable prefixes, model reliability and ordinal reliability are all parsed
  as `""`, not `None`.
- `python -m compileall -q src tools tests`: PASS.
- **99/99** shell scripts pass `bash -n`; backslash-continuation/comment hazard scan: 0.
- Dynamic source-failure simulation writes `SOURCE_REBUILD_FAILED.json` with the
  correct `preflight` stage and preserves the original nonzero exit.

## v48.45.1 — SOWR ENGINEERING HOTFIX / SOURCE-RUN RESOLUTION (2026-08-10)

This revision changes **no algorithmic factor, loss target, shared rule, gate, top-k,
ROCT scale, regime handling, or risk budget**. It fixes the execution layer only.

The failed v48.45 A/B/C runs are authoritative engineering failures (`RC=30`,
`pipeline_valid=false`, `failure_stage=adaptation`, certificate not executed). Their
result packages contain no candidate directories or `adapt_balanced/precision.log`,
which means both variants returned before entering adaptation. The run metadata records
`source_run=runs/ocrap_v48_13_terra_proxy_4801` while the v48.45 command changes cwd to
the versioned checkout. In the successful v48.44 results, the actual source checkpoint
was `/home/senzeyu2/code/OC-RAP/runs/ocrap_v48_13_terra_proxy_4801/.../best.pt`. The
relative default therefore resolved against the wrong repository in v48.45. This is the
root cause; there is no OOM evidence.

Engineering fixes:

- v48.45 arm launcher resolves the immutable v48.13 source run from persistent
  `BASE_OUT` first, while preserving an explicit `SOURCE_RUN` override; the resolved
  path is made absolute before entering the dedicated controller.
- the dedicated controller now performs a source-checkpoint contract **before** dataset
  index construction or GPU adaptation, checking both Balanced and Precision source
  checkpoints and recording paths, size and SHA256 in `SOURCE_CHECKPOINT_CONTRACT.json`;
  a missing source now fails as `source_checkpoint_contract`, not as the ambiguous
  `both variants failed`.
- the per-variant fallback path now always writes an in-run adaptation log,
  `VARIANT_STAGE_FAILED.json`, `FAILURE_SIGNATURE_*.json`, and
  `ADAPTATION_FAILED_*.json` if a source/factor-cache prerequisite disappears after the
  global preflight.
- SOWR controls are explicitly passed across the dedicated-controller -> variant-process
  boundary instead of relying only on inherited export state.
- the internal SOWR witness stage now sets `SKIP_POST_TRAIN_CALIBRATION=1`; generic
  bucket calibration is not run inside witness recalibration, while the downstream authoritative
  v48.36 calibration/certificate path remains unchanged.
- the v48.45 2x2 launcher now treats RC=20 as a valid Natural-gate result, not an
  engineering crash; only non-{0,20} arm exits make the launcher fail. Launcher logs and
  exit codes are stored inside each run directory so result ZIPs retain the primary
  failure evidence.

Validation after the hotfix: compileall PASS; 98/98 shell scripts `bash -n` PASS; focused
v48.36-v48.45 regression 102 passed / 1 skipped; dedicated missing-source simulation
correctly terminates before dataset/adaptation; synthetic four-arm RC=20 launch returns
success and classifies all four arms as valid Natural-gate failures; synthetic RC=30
launch is correctly classified as engineering failure.

The existing v48.45 SOWR run commands remain valid when the repaired code is extracted
under the same `/home/senzeyu2/code/OC-RAP-v48.45-SOWR` path. Do not delete the immutable
source run `/home/senzeyu2/code/OC-RAP/runs/ocrap_v48_13_terra_proxy_4801` when cleaning
old v48.45 outputs.

## v48.45 — SOWR / SHARED-OPTION WITNESS RECALIBRATION (2026-08-10)

### v48.44 ROCT final gate attribution

All uploaded v48.44 A/B/C/D runs are authoritative Natural-gate algorithm results:
`authoritative_exit_code=20`, `pipeline_valid=true`, certificate executed, gate evaluated,
`gate_passed_false=true`, and no pipeline/calibration-failure marker. They are therefore
valid for algorithm attribution and must not be treated as engineering failures.

**B (deployability-side ROCT) is rejected as a Near frontier rotation.** Relative to A,
Near certificate deployability safe-positive false veto changes only 16/18 -> 15/18,
harmful-vs-safe-positive AUC 0.416 -> 0.418, while harmful false-safe rises
0.162 -> 0.301. Certificate recall remains 0.111. This misses the pre-registered
strong signal (roughly <=12/18 false veto and AUC >=0.56 without contamination).
Do not absorb B as evidence that the Near deployability bottleneck is solved.

**C (benefit-side ROCT) is rejected.** Contact development still has exactly **0/37**
safe-positive proposal rows with `pred_adv >= 0`; certificate recall falls from 0.10 to
0.05; candidate safe-positive AUC 0.544 -> 0.523 and proposal safe-positive AUC
0.555 -> 0.543. It therefore does not break the physical-sign collapse and must not
be absorbed.

**D/Main (dual ROCT) is not accepted as a successful shared-rule solution.** Precision
development gets one Near safe-positive but zero Contact safe-positive; Contact remains
0/37 on `pred_adv >= 0`. Balanced development does select one Near and two Contact
safe-positives under one shared rule, but harmful-selected UCB90 is 0.282 (Near) and
0.387 (Contact), both above the 0.22 fit budget, while precision LCBs are far below 0.50.
The two-sided positive-capture condition is therefore not achieved without harmful
contamination.

D nevertheless provides one useful *diagnostic* signal. It reduces certificate
deployability harmful false-safe to 0.076 (Near) / 0.068 (Contact) and improves Contact
deployability AUC to 0.651, while its Near development deployability AUC reaches 0.764.
But Near certificate AUC collapses to 0.487 and false-veto returns from 9/17 development
to 16/18 certificate. The useful structure is not stable across splits.

### Bottleneck re-diagnosis

The top-5 proposal oracle remains feasible in both Near and Contact, so candidate support
is not the first bottleneck. More importantly, the current direct-evidence adaptation
freezes the paper-matched `root_logit_head`, `margin_head`, and `obs_embed_head`; only
OCAF/ROCT-side evidence corrections are adapted. Yet source training already contains
teacher supervision for root assignment, signed recovery margins, observation
compatibility, deployability/oracle scores, option-resolved shared-recovery `q`, shared
option admission/success, and best shared option.

The D split gap shows why adding another residual or increasing ROCT width/scale is the
wrong next move: development can fit a useful deployability ordering, but the frozen
recovery witness does not carry that ordering stably to certificate. This is consistent
with stale/miscalibrated recovery-option margins, root probabilities, or observation
compatibility rather than a lack of downstream ranking capacity.

### v48.45 algorithm: SOWR

**SOWR = Shared-Option Witness Recalibration.** Before the existing factor/OCAF/ROCT
adaptation, insert a short, low-learning-rate, regime-agnostic witness stage that uses
only the existing training/validation teacher contract. It does **not** add a
Safe/Near/Contact identifier, regime router, regime-specific threshold, or regime-specific
policy.

The shared encoder and root decoder stay frozen. SOWR only permits the following
paper-matched semantic heads to move:

- margin-witness factor: `root_logit_head,margin_head`;
- observation factor: `obs_embed_head`.

The stage uses full OC-MERO forward semantics plus explicit root-assignment and
observation-equivalence losses. The explicit losses are required because lower-tail /
top-m OC-MERO aggregation alone does not guarantee dense gradients to root/observation
heads. Default SOWR settings are 8 epochs, patience 3, LR 5e-5, batch 72. It uses
train/validation only; held-out test roots remain unread. After SOWR, all witness heads
are frozen again and the **same v48.44-D dual ROCT, top-k=5, shared continuous rule and
risk budgets** are run unchanged.

### v48.45 strict 2x2

- **A:** v48.44-D dual-ROCT reference, no witness recalibration.
- **B:** A + root-probability/recovery-margin witness recalibration.
- **C:** A + observation-kernel recalibration.
- **D/Main:** A + both witness factors.

This 2x2 answers whether the frozen recovery witness is the current bottleneck without
confounding the experiment with another selector, larger ROCT scale, new rank loss, or
regime routing.

### Pre-registered go/no-go interpretation

**B / Near:** require a material certificate deployability frontier rotation, not only
lower training loss. Strong signal: safe-positive false veto <=12/18, deployability
AUC >=0.56, and no increase in harmful false-safe / harmful-selected UCB. The witness
stage should also reduce validation margin/option-q/best-option loss from its initial
checkpoint.

**C / Contact:** first requirement remains breaking the physical-sign collapse:
Contact development safe-positive `pred_adv >= 0` must become non-zero (strong signal
about >=25%), with candidate/proposal safe-positive AUC improving by roughly 0.05 and
recall moving toward 0.20 without a risk rebound. Lower observation BCE alone is not a
go decision.

**D/Main:** the same shared rule must have `positive_selected > 0` in both Near and
Contact development, retain B/C structural signals, and satisfy the existing harmful
budgets rather than relaxing them. A key generalization signal is that the large
Near development->certificate deployability AUC/FV gap should shrink materially.

If SOWR fails, **do not** increase ROCT scale/width, loosen harmful budgets, densify the
threshold grid, expand top-k, oversample positives, add generic pairwise/listwise rank
stacks, re-add generic harm residuals, or introduce regime-specific routing. The next
algorithmic audit should move upstream to recovery-option coverage/taxonomy, continuous
option parameter coverage, margin-teacher calibration and option-resolved witness
identifiability.

### Historical modifications explicitly not repeated

v48.45 intentionally avoids previously unsuccessful families: threshold densification,
top-k expansion, positive oversampling, generic rank stacking, generic harm residuals,
unbounded residuals/factors, full factorization, v48.42 partial-pooling/rank-skip,
v48.43 POET free alias transport, ROCT width/scale increases, broad encoder fine-tuning,
and regime-conditioned policy/threshold routing. The retained downstream stack remains
candidate-relative physical OCAF + dual-task bridge + bounded component veto/support
reliability + deterministic joint reserve + shared rule.

### Engineering changes and validation

- `scripts/train_ocrap_v48_trac_sr.sh`: source loss weights and direct-only fast path are
  now environment-overridable while legacy defaults remain exactly unchanged.
- New `scripts/adapt_ocrap_v48_45_sowr_stage.sh` implements the head-only witness stage.
- `scripts/adapt_ocrap_v48_36_ocaf_variant.sh` optionally inserts SOWR and binds the
  resulting checkpoint hash into the factor-cache contract.
- New A/B/C/D runners and `tools/compare_v48_45_sowr_2x2.py` report SOWR initial/best
  validation diagnostics plus development/certificate metrics without test-root reads.
- Focused regression audit after implementation: **95 passed / 1 skipped** across the
  retained v48.36-v48.45 contracts; v48.44+v48.45 targeted matrix **14 passed**.
- `compileall` PASS; **98/98** shell scripts pass `bash -n`; new v48.45 runtime scripts
  pass a static no-regime-routing and no-test-boundary check.

## v48.44 — ROCT / RECOVERY-OPTION COMPATIBILITY TRANSPORT (2026-08-10)

### v48.43 POET final 2x2 attribution

The uploaded v48.43 A/B/C/D runs are all authoritative Natural-gate algorithm results
(`authoritative_exit_code=20`, `pipeline_valid=true`).  They therefore support causal
algorithm attribution rather than engineering-failure diagnosis.

**B (harm-side POET) is rejected.**  Near deployability safe-positive false veto is
unchanged at 16/18 relative to A, while harmful-vs-safe-positive deployability AUC
falls from 0.416 to 0.409.  Near certificate harmful-selected UCB90 worsens from
0.546 to 0.640.  B therefore does not rotate the Near deployability frontier; it
mostly changes score level/cleanliness trade-offs and is not retained.

**C (benefit-side POET) is rejected.**  Contact candidate safe-positive AUC changes
only from 0.544 to 0.556, proposal safe-positive AUC falls from 0.555 to 0.526, and
certificate positive recall remains 0.10.  The development Contact safe-positive
rows still have 0/37 with `pred_adv >= 0`.  C does not convert post-prefix alias
geometry into physical benefit capture and is not retained.

**D/Main (dual POET) is rejected as an implementation, but supplies a structural
clue.**  It improves Contact candidate safe-positive AUC to 0.578 and deployability
safe-positive false veto from 26/31 to 17/31; DRS harmful-vs-safe-positive AUC rises
from 0.554 to 0.706.  However Near deployability false veto worsens to 17/18,
Contact deployability harmful false-safe rises from 0.148 to 0.255, Contact
certificate recall falls to zero, and the fitted shared development rule selects
zero candidates in both Near and Contact.  This is evidence that candidate-specific
post-prefix structure carries useful signal, but free dual context transport creates
incompatible scales/negative transfer across evidence coordinates.

A decisive development diagnostic is common to all A/B/C/D arms: Contact has 37
safe-positive proposal rows and **none has `pred_adv >= 0`**.  Since the shared rule's
semantic score domain starts at zero, no threshold-grid densification can recover
those positives.  The current bottleneck is therefore sign/structural identifiability,
not grid resolution.

### Bottleneck re-diagnosis

POET measures whether latent roots remain observation-aliased, but the paper's
recoverability definition is stricter: observation-equivalent roots must admit a
**shared compatible recovery option**.  Alias mass alone cannot distinguish benign
ambiguity (roots share a recovery maneuver) from harmful ambiguity (roots require
incompatible maneuvers).  Moreover v48.43 injected the same structural vector into
whole benefit/harm OCAF contexts, permitting one useful Contact coordinate to rotate
unrelated DRS/gap coordinates and damage the shared rule.

v48.44 therefore changes the structural statistic and the injection locality rather
than adding generic capacity, another ranking loss, more proposal candidates, or a
regime-specific policy.

### Algorithm: Recovery-Option Compatibility Transport (ROCT)

For every candidate prefix, ROCT reuses the frozen latent-root decoder, predicted
post-prefix observation kernel, and frozen recovery-option margin decoder.  It computes
the same OC-MERO geometry already used by the paper and forms a bounded four-coordinate
signature:

1. normalized deployable recoverability `0.5*(tanh(R_dep)+1)`;
2. bounded oracle-to-deployable gap `tanh(relu(R_orc-R_dep))`;
3. observation-weighted shared-option conflict pressure: aliased root pairs receive
   high pressure only when no recovery option has high common support for both roots;
4. root-probability mass that has a feasible observation-consistent recovery option.

Only candidate-minus-nominal evidence is learned.  The structural teacher is detached,
and both ROCT projections are bias-free and zero-initialized, preserving the exact A
forward pass at initialization and exact zero correction on nominal candidates.
Corrections are bounded with `tanh`; the v48.44 experiment uses a shared logit bound of
3.0 (at `tau_b=0.05`, at most 0.15 physical benefit-margin correction), avoiding the
historically failed unbounded-residual path.

ROCT is deliberately **semantically local**:

- benefit-side ROCT corrects only the unified benefit logit;
- safety-side ROCT corrects only component index 1 (deployability), which is the
  paper-matched observation-consistent recoverability coordinate;
- it does not rotate DRS, gap, hard-rule, harm-proxy, the shared OCAF bridge, proposal
  generator, or deployment thresholds.

No Safe/Near/Contact identifier, routing branch, per-regime threshold, per-regime loss,
or state machine is introduced.  Regimes remain evaluation slices only.

### v48.44 2x2 ablation

- **A:** v48.43-A retained shared dual-OCAF reference; no POET and no ROCT.
- **B:** A + deployability-side ROCT only.  Primary causal readout: Near deployability
  false-veto/AUC/false-safe, with Contact reported as a cross-regime consistency check.
- **C:** A + benefit-side ROCT only.  Primary readout: Contact safe-positive benefit
  AUC plus the development `pred_adv >= 0` sign-capture rate.
- **D/Main:** A + both semantically local ROCT corrections.  Tests whether one shared
  development rule can select safe-positive examples in both Near and Contact while
  preserving harmful-selection budgets.

Every arm force-disables v48.43 POET, v48.42 partial pooling/rank skip, unbounded
benefit/harm factors, full component factorization, and regime routing.  Proposal
top-k remains 5 and the shared development rule is unchanged.

### Pre-registered v48.44 go/no-go

- **B vs A:** Near deployability false veto must materially decrease (strong signal:
  <=12/18), harmful-vs-safe-positive AUC must increase (strong signal: >=0.56), and
  harmful false-safe / selected-UCB must not materially worsen.
- **C vs A:** Contact development safe-positive `pred_adv >= 0` must move from 0/37 to
  a non-trivial fraction (strong signal: >=25%), Contact candidate/proposal
  safe-positive or positive AUC should improve by about >=0.05, and certificate recall
  should move toward >=0.20 without worse harmful UCB.
- **D:** Near and Contact development must both have `positive_selected > 0` under the
  same shared rule.  Contact sign capture and B's Near deployability gain must both be
  retained.  A lower constraint deficit with zero positives is explicitly not a go.
- If B/C/D fail these structural criteria, do **not** increase ROCT width/scale or
  loosen shared-rule harm budgets.  The next diagnosis must target calibration of the
  frozen recovery-option/observation teacher or the definition/coverage of the
  recovery-option set.

### Engineering changes

- Added ROCT flags/hyperparameters to training, checkpoint, inference materialization,
  factor-cache identity, architecture records, model/training contracts, and
  stage-transfer approved prefixes.
- Direct-only and full forward paths both consume root/option validity masks when
  constructing ROCT evidence.
- Added v48.44 tests for exact zero-init identity, bounded/candidate-relative signature,
  nominal-zero correction, component-local deployability injection, benefit/harm
  separation, detached structural teacher, script cleanliness, and contract binding.
- Added `run_v48_44_roct_ablation_arm.sh`, `run_v48_44_roct_2x2_parallel.sh`,
  `run_v48_44_roct_dedicated.sh`, and a development/certificate-only v48.44 comparator
  with explicit Contact/Near sign-geometry diagnostics.

### Non-repetition list extended by v48.43

Do not repeat or simply amplify: POET alias-only free dual-context transport, POET
width/scale increases, threshold-grid densification, top-k expansion, positive
oversampling, generic rank-loss stacking, generic harm residuals, full component
factorization, unbounded factors, or regime-conditioned routing/threshold/policy.

Retain: candidate-relative physical OCAF, dual task bridge, bounded HAF/component veto,
support reliability, deterministic joint reserve, proposal top-k=5, and one shared
continuous development/deployment rule.

## v48.43 — POET / POST-PREFIX OBSERVATION-EQUIVALENCE TRANSPORT (2026-08-09)

### v48.42 final valid 2x2 attribution (supersedes the earlier RC30-only interpretation)

The newly uploaded v48.42/48.42.1 A/B/C/D results are all pipeline-valid algorithm
results: `authoritative_exit_code=20`, `pipeline_valid=true`,
`certificate_executed=true`, and `gate_evaluated=true`.  Consequently the earlier
v48.41-era statement that the rank-skip arms could not be evaluated no longer applies
to this experiment set.

All four Precision arms remain blocked at `development_rule_fit`, while the proposal
oracle remains feasible (Near top-5: 9 safe-positive groups; Contact: 20).  The failure
is therefore downstream of candidate generation.

**B (hierarchical partial-pooling harm residual) is rejected as a main-algorithm
component.**  In Near certificate, deployability safe-positive false veto changes only
from 16/18 to 15/18 while harmful-vs-safe-positive deployability AUC degrades from
about 0.482 to 0.420 and harmful false-safe fraction rises from about 0.106 to 0.184.
In Contact, B reduces deployability false veto from 27/31 to 21/31, but harmful
false-safe fraction rises and certificate harmful-selected UCB90 worsens from about
0.484 to 0.563.  This is primarily a score/calibration shift, not improved conditional
frontier discrimination.

**C (bounded rank-benefit skip) is rejected.**  Contact candidate safe-positive AUC
changes from about 0.582 to 0.580; proposal positive/safe-positive AUC does not improve;
certificate positive recall remains 0.10.  The strong frozen rank signal previously
observed is not converted into deployable physical benefit evidence by this skip.
Further ranking-objective/skip stacking is therefore stopped.

**D/Main (partial pooling + rank skip) is also rejected.**  It reaches Contact
certificate recall 0.15 versus A's 0.10, but harmful-selected UCB90 worsens to about
0.503 and Contact harmful false-safe rises sharply.  The nearest shared development
rule has fewer formal failures (4 versus 6) and lower aggregate deficit, yet selects
zero safe-positive examples in both Near and Contact development strata.  It has not
entered a statistically meaningful feasible window.

These results activate the pre-registered stop rules from v48.42: do not add more
generic harm residual capacity after B fails the Near deployability frontier, and do
not stack more ranking objectives after a valid C run fails to improve Contact benefit
capture.

### Bottleneck re-diagnosis

The retained v48.40/v48.42-A architecture already sends candidate-relative executable
physical action features to dual benefit/harm OCAF branches, but its observation
context is a broadcast of the nominal row's current observation features.  During the
`direct_only` evidence-adaptation path it does not explicitly expose the candidate's
predicted **post-prefix observation-equivalence geometry**, despite observation
consistency after the candidate prefix being the central recoverability object in the
paper.

v48.43 therefore treats the next problem as **structural identifiability**, not
additional classifier capacity.  The same learned latent-root/observation model is
used for every sample and every regime; no Safe/Near/Contact identifier, branch,
threshold, router, or loss is introduced.

### Algorithm: Post-prefix Observation-Equivalence Transport (POET)

For each candidate prefix `a`, the frozen latent-root decoder and post-prefix
observation embedding predict a compact four-coordinate structural signature:

1. normalized latent-root entropy;
2. probability-weighted off-diagonal observation-alias mass;
3. probability-weighted peak alias pressure;
4. maximum latent-root probability.

Let this bounded signature be `psi(a)` and let `a0` be the nominal candidate in the
same proposal group.  POET uses only the candidate-relative structural evidence

`delta_psi(a) = psi(a) - psi(a0)`.

Two optional bias-free linear projections inject this signal into the already
validated dual-OCAF task contexts:

`z_b' = z_b + s W_b delta_psi`,
`z_h' = z_h + s W_h delta_psi`.

`W_b` and `W_h` are zero-initialized.  Therefore enabling POET is exactly identical to
the shared dual-OCAF reference at initialization; the nominal candidate has exactly
zero transport; and the frozen root/observation teacher cannot be rotated by sparse
evidence gradients.  Only the tiny transport projection is learned.  The mechanism is
continuous, observation/action based, and regime agnostic.

### v48.43 2x2 ablation

- **A:** retained shared dual-OCAF reference; no POET.
- **B:** A + harm-side POET only.  Tests whether candidate-specific post-prefix
  observability improves Near deployability discrimination rather than merely shifting
  its calibration.
- **C:** A + benefit-side POET only.  Tests whether the missing Contact benefit signal
  is recoverable from candidate-specific post-prefix observability without another
  rank objective.
- **D/Main:** A + benefit and harm POET.  Tests complementarity and the shared-rule
  feasibility window.

Every v48.43 arm force-disables v48.42 partial pooling and rank-benefit skip, as well as
historical unbounded factors, factorized harm interaction, one-sided tail losses, and
regime-conditioned routing.  Proposal top-k remains 5 and the same shared Natural rule
is fitted.

### Go/no-go interpretation

The v48.43 mechanism is considered supported only if its causal readout improves
conditional discrimination, not merely score level:

- B should materially reduce Near deployability safe-positive false veto **and** raise
  harmful-vs-safe-positive deployability AUC without a comparable rise in harmful
  false-safe/selected UCB.
- C should materially improve Contact safe-positive/positive AUC and certificate
  positive capture without worsening harm control.
- D should retain B's Near harm cleanliness and C's Contact benefit gain, and the
  shared development rule must begin selecting safe-positive examples in both Near and
  Contact.  A lower numerical constraint deficit with zero positives is not sufficient.
- If POET fails these tests, do not increase its width/scale as the next reaction;
  inspect recovery-option conflict/teacher observability calibration instead.

### Engineering contract changes

- POET flags/scale are checkpoint-, inference-, training-, architecture-, factor-cache-
  and contract-bound.
- Stage-transfer approved-prefix logic now recognizes both POET projections, including
  non-reserve future paths.
- Runtime preflight checks signature shape/finiteness, nominal exact-zero transport,
  and full forward/backward finiteness.
- Added a v48.43 2x2 comparator that reads development/certificate artifacts only and
  explicitly sets `test_roots_read=false`.
- Two-GPU launcher supports four concurrent arms; `MAX_PARALLEL_ARMS=2` provides two
  waves on memory-constrained GPUs without changing experiment semantics.

### Non-repetition list extended by the valid v48.42 results

Do not repeat: threshold-grid densification, top-k expansion, aggressive positive
oversampling, hardest-negative population distortion, generic pairwise/listwise loss
stacking, full identity-stage factor updates, learned admission residual, v48.38
one-sided tail loss, v48.39 unbounded benefit/harm, v48.40 `frontier_tanh`, v48.41 full
component factorization, **v48.42 partial-pooling harm residual**, **v48.42 bounded
rank-benefit skip**, or any regime-conditioned routing/threshold/policy.

Retain: OCAF, factor preservation, bounded HAF semantics, dual task OCAF, component
veto, support reliability, aligned deterministic joint reserve, one shared continuous
deployment rule, and proposal top-k=5.

## v48.42.1 — HPFR METRIC-PROVENANCE HOTFIX (engineering-only, 2026-08-08)

### Scope

This is an **engineering-only** repair of the uploaded v48.42 HPFR main run. The algorithm remains `v48.42-HPFR`: no model tensor computation, loss, optimizer setting, dataset, candidate set, factor-cache semantic setting, Natural gate, threshold, Safe/Near/Contact treatment, or shared continuous deployment rule is changed. `src/` and `configs/` are byte-identical to v48.42.

### Uploaded RC=30 root cause

- Balanced and Precision adaptation both completed with RC=0 and valid training/model/stage-transfer/cache contracts.
- The controller entered certificate execution, but `check_v48_36_metric_calibration_contract.py` rejected the reserve-only final stage before policy-certificate verification.
- `materialize_v48_38_reserve_stage.py` had recorded factor provenance as repo-relative strings such as `runs/<run>/candidates/<variant>/factor_stage/model_v48_trac_sr/train_summary.json`.
- The checker incorrectly treated every relative provenance path as relative to the final model directory, producing a duplicated path of the form `.../model_v48_trac_sr/runs/<run>/.../factor_stage/...`.
- Both certificate workers therefore exited RC=1 with `metric source train summary missing`; the certificate controller normalized the artifact failure to RC=30. The Natural gate was never evaluated.

### Engineering changes

1. Metric-source provenance resolution is SHA256-pinned and anchor-aware. It accepts absolute paths, controller-CWD/repo-relative historical paths, genuine stage-parent-relative paths, and a relocation-safe `factor_stage/...` suffix. A candidate is accepted only when its bytes match the already-recorded SHA256; mismatches still fail closed.
2. New zero-update materialization writes canonical absolute source paths and explicit candidate-relative provenance fields (`provenance_path_schema_version=2`) so future runs do not depend on an implicit relative-path anchor.
3. Certificate ordering now executes adaptation-dev proposal extraction, pooled shared-rule fitting, and the metric-calibration provenance contract **before reading certificate samples**. This changes no score or gate semantics but prevents an engineering provenance error from consuming certificate I/O or leaving misleading partial verification artifacts.
4. Added `run_v48_42_1_hpfr_from_exact_factor_cache.sh`. It requires a new OUTPUTDIR and reuses old Balanced/Precision factor stages only through the existing exact factor-cache SHA/semantic contract; it never silently reuses a checkpoint. This avoids repeating the 20-epoch factor training after the engineering-only failure.
5. The implementation identifier is `v48.42.1-HPFR-METRIC-PROVENANCE-HOTFIX`; the algorithm identifier remains exactly `v48.42-HPFR`.

### Interpretation

The uploaded v48.42 RC=30 is **not an algorithm result**. Development diagnostics produced before the failed provenance contract must not be used as a Natural-gate outcome. The next valid scientific result is the first pipeline-valid RC=0 or RC=20 obtained from the byte-identical v48.42 HPFR factor checkpoint through this repaired certificate path.

# v48 OC-TRAC-SR Algorithm Changelog

## v48.42 — HPFR / HIERARCHICAL PARTIAL-POOLING FRONTIER RESERVE (2026-08-08)

- v48.41 A/B are valid RC=20 algorithm results; C/D are RC=30 engineering failures caused by a module-prefix-only verifier rejecting the exact scalar key `direct_evidence_rank_benefit_log_gain`. v48.42 fixes exact-or-dotted parameter matching in adaptation and stage-transfer contracts.
- v48.41 full component-factorized harm is not retained: it improves DRS locally but worsens aggregate rare-frontier discrimination because deployability/gap lose useful shared regularization. Near safe-positive false veto remains deployability-dominated; Contact still has benefit + harmful-false-safe bottlenecks.
- Retain v48.40 dual task OCAF and shared harm representation. Add zero-initialized, bounded, component-specific residual heads on a detached copy of the shared harm evidence (partial pooling). This permits physical-factor correction without rotating the shared OCAF bridge.
- Re-test the bounded monotone rank-benefit skip only after the checker repair; v48.41 never produced a valid gate result for that mechanism.
- New 2x2: A shared-harm reference; B + partial-pool harm residual; C + repaired rank skip; D/main = B+C. No regime labels, regime-specific thresholds, or discrete Safe/Near/Contact policies are introduced.
- Calibration diagnostics additionally emit the ordered teacher component-veto terms so future false-veto/false-safe analysis can be attributed directly to DRS/deployability/gap rather than aggregate harm alone.

## v48.22 — OC-TRAC-COVENANT-BRIDGE (2026-07-30)

### v48.21 result attribution

- The v48.21 dedicated controller is valid and records `pipeline_valid=true`,
  `test_roots_read=false`, and `RC=20`; Near/Contact fit and verify support are
  non-empty and feasible. This is a genuine Natural-gate rejection, not a data,
  parameter-count, unsupported-gate, or controller failure.
- v48.21 converted part of the Near signal into learned selector evidence, but only
  in one objective branch. Balanced Near reaches candidate benefit AUC 0.841 and
  learned frozen-top-k benefit AUC 0.805, with positive top-1 regret 0.005. Precision
  Near collapses to 0.201/0.339. The Near signal is therefore real but not stable
  under the current training objective.
- Contact shows the first partial evidence of transferable safe-benefit selection in
  one branch: Precision candidate/learned benefit AUC is 0.586/0.613 and learned
  top-1 correlation is 0.166. Balanced Contact remains 0.540/0.432 with correlation
  -0.165. Contact capability is not jointly present with the strongest Near model.
- Component-risk learning is retained: learned harm AUC is approximately 0.60--0.65
  in Near and 0.64--0.66 in Contact. However, harmful top-1 switch rates remain
  0.45--0.70. Global harm AUC mainly separates harmful candidates from abundant dead
  candidates and does not adequately separate harmful candidates from high-benefit
  safe candidates at the deployment frontier.
- Every main and ablation certificate still has zero verify coverage. The nearest
  Balanced-Near fit rule selects 12 groups with only 2 positives, precision LCB
  0.071 and harmful UCB 0.673. The nearest Precision-Contact rule selects 19 groups
  with 2 positives, precision LCB 0.045 and harmful UCB 0.292. This is not a small
  threshold miss.
- The frozen proposal remains non-limiting: positive top-k oracle hit is about
  0.97--1.00. Proposal retraining remains prohibited in this round.

### Root defects found in v48.21

1. **Three deployment semantics were compressed into two heads.** Raw PCD benefit,
   componentwise harm, and final safe admission are different hypotheses. v48.21
   trained its benefit head on safe benefit while the preregistered gate continued
   to measure raw benefit plus an independent harm veto. This made one logit serve
   incompatible training, reporting, and certificate meanings.
2. **The safe-opportunity MIL probability was not safe.** The noisy-OR group loss used
   only `sigmoid(opportunity)` and ignored harm, even though its target was “at least
   one raw-beneficial and non-harmful candidate”. It therefore rewarded high
   opportunity on positive-but-harmful candidates—the exact false-safe failure mode
   exposed by Contact.
3. **Benefit and harm were combined differently in loss, checkpoint selection,
   calibration, and runtime.** Group MIL, safe-set loss, soft dev metrics and
   deployment did not all consume one explicit admission score. Candidate AUC could
   improve without improving the action actually certified or executed.
4. **Sparse admission gradients still competed semantically with raw benefit.** The
   same benefit logit was expected to preserve raw opportunities and also encode the
   conjunction `benefit AND non-harm`. Balanced learned Near while Precision learned
   part of Contact, producing complementary specialization rather than one model
   that works in both regimes.
5. **The zero-initialized source identity was never evaluated as a checkpoint.**
   Training started checkpoint comparison after epoch 1. A useful source/consensus
   initialization could not win if the first update damaged one regime.
6. **Sampler semantics needed an explicit task choice.** Raw benefit and safe
   admission now have separate heads. The positive group sampler must deliberately
   oversample safe-admission groups without changing the raw-benefit target.
7. **Certificate diagnostics were incomplete at the safety frontier.** Global
   benefit/harm AUC concealed failure among high-opportunity candidates. Safe-positive
   AUC and conditional high-opportunity harm AUC are required diagnostics.
8. **New admission branch had an uncovered runtime error during development.** The
   loss accepted `pred_admission_logit` but initially failed to initialize the local
   `admission_logits` tensor. A gradient-level unit test exposed and fixed this before
   delivery; static compilation alone would not have caught it.

### v48.22 algorithm: COVENANT-BRIDGE

**COVENANT = Cross-regime Opportunity, Veto Evidence, and Non-regime-specific
Admission with Nominal-preserving Transfer.** It remains one unified model over Safe,
Near and Contact. No regime ID, bucket-selected calibrator, or regime-specific
residual is available at inference.

1. **Three factorized hypotheses.** The unified model now predicts:
   - raw recovery benefit, trained on total PCD improvement and used by the primary
     opportunity gate;
   - DRS/deployability/gap component harm, aggregated by exact non-compensatory
     maximum and used by the independent harm veto;
   - final safe admission, trained on `raw benefit AND no component veto` and used
     for top-k reranking and group admission.
2. **Detached conservative admission prior.** The admission logit starts from
   `detach(raw_benefit_logit) - softplus(detach(harm_logit))` plus a zero-initialized,
   bounded admission residual. Sparse admission gradients cannot overwrite benefit
   or risk heads, while the prior remains conservative and context-correctable.
3. **Correct safe-opportunity MIL.** Frozen-top-k noisy-OR now uses the explicit
   admission probability. In the two-head ablation it uses
   `P(raw benefit) * (1-P(harm))`; it never uses opportunity alone. Candidates
   outside frozen top-k receive no group-opportunity gradient.
4. **One deployment score everywhere.** Training safe-set loss, soft checkpoint
   metrics, calibration, evaluator, selector and closed-loop runtime all use the
   explicit candidate-vs-nominal admission score. Legacy checkpoints fall back to
   their historical opportunity-minus-harm score.
5. **Raw-benefit/safe-admission sampler decoupling.** Raw benefit remains the benefit
   head target, while `group_batch_safe_positive_target=true` stratifies minibatches
   using safe-positive groups. Harmful-benefit overlap groups still supervise raw
   benefit and harm tails, but are not presented as positive admission groups.
6. **Epoch-zero checkpoint evaluation.** The source/consensus identity is validated
   and saved before any optimizer step and may win early stopping.
7. **COVENANT checkpoint risk.** The threshold-free CONCORD risk is retained and adds
   worst-regime harmful policy mass and false-admission penalties. This is dev-only,
   uses no test root, and does not relax the Natural gate.
8. **Frontier diagnostics.** Certificate reports now include candidate safe-positive
   AUC, learned top-k safe-positive AUC, and harm AUC conditioned on high opportunity,
   separating broad risk discrimination from the safety-critical decision frontier.

### Non-repeated v48.22 ablations

1. `A_two_head_safe_probability`: raw benefit + component harm, corrected
   `P(benefit)*(1-P(harm))` group probability, no third admission head. This isolates
   the v48.21 MIL/score engineering correction.
2. `B_triad_candidate_only`: three heads and candidate admission BCE, but no group MIL
   or setwise objective. This isolates the third hypothesis.
3. `C_triad_group_mil_aggregate`: admission head plus group objectives with one
   aggregate harm tail. This isolates component risk heads.
4. `D_full_covenant`: raw benefit, three component veto heads, admission head,
   deployment-exact safe-set loss, safe-opportunity MIL and COVENANT checkpoint risk.

All eight tasks (four groups times Balanced/Precision) are launched together. Round-
robin assignment places four tasks on GPU0 and four tasks on GPU1. Each task defaults
to batch size 48, one DataLoader worker, and bounded host math threads to reduce A30,
CPU and disk contention.

### Decision and non-repetition rules

- `RC=0`: run only the authorization-checked stress command generated by the
  independent certificate, then multi-seed confirmation and final Safe paired
  non-interference evaluation using the same selected checkpoint.
- `RC=20`: do not read test. Use A/B/C/D to identify whether the missing factor is the
  admission hypothesis, group supervision, or component veto. Do not tune the gate
  or proposal.
- `RC=30`: no algorithm conclusion is allowed. Repair the named protocol, index,
  training, checkpoint or certificate stage first.
- Do not repeat safe-benefit-overloaded opportunity heads, opportunity-only MIL,
  shared two-head admission semantics, fixed-threshold early stopping, exact-min
  benefit transfer, regime-specific calibrators, raw high-dimensional context,
  proposal retraining, threshold-grid-only tuning, or dataset regeneration in this
  round.

### Local validation

- `pytest`: 209 passed, 5 warnings.
- `python -m compileall -q src tools tests`: passed.
- `bash -n` for all shell scripts: passed.
- New tests cover three-head bucket invariance/capacity, detached-gradient isolation,
  harmful-benefit MIL rejection, independent safe-positive sampling, explicit
  admission-score plumbing, epoch-zero selection, and eight-task/two-GPU assignment.
- The delivery environment has no real WOMD/Waymax data or A30 GPUs. No v48.22 gate or
  closed-loop result is claimed.


## Motivation

v47 improved candidate-level positive-recovery AUC but failed policy-level top-1 selection and scene-disjoint risk verification, especially in Contact.

## Algorithm

- Added tri-state candidate-vs-nominal supervision: positive, dead-zone, harmful.
- Added nominal as an explicit setwise abstention class.
- Added independent harmful-switch head.
- Added conservative two-expert aggregation: lower confidence for gain/opportunity, upper confidence for harm.
- Added asymmetric expert specialization without hidden regime routing.
- Unfroze shared observation encoder with layer-wise learning rate.
- Added direct-only fast path.
- Aligned sampler, loss and calibration to exact teacher PCD.
- Added scene-balanced exact positive-group sampling.
- Added policy-level joint opportunity/harm/gain/macro calibration.
- Disabled historical handwritten rescue certificates in the v48 main-policy evaluation path.

## Data and protocol

- Added clean WOMD training-based Near/Contact builder.
- Added dedicated standard-validation calibration builder.
- Added filtering to exclude all existing val/test scenes from dedicated calibration roots.
- Added scene overlap audit.
- Added scene-disjoint low-cost split from existing val roots.
- Added positive-group and positive-scene minimum quality gates.
- Preserved existing user val/test roots.

## Engineering

- Added BF16/TF32, pinned/persistent data loading and prefetch.
- Added partial checkpoint loading for shape-changed heads.
- Fixed sequential two-GPU data-worker synchronization.
- Added background controller and centralized logs.
- Added harm prediction propagation through selector, offline evaluator and closed-loop runner.

## Validation

- Python compileall passed.
- Shell syntax checks passed.
- 97 tests passed, 2 non-failing warnings.
- No v48 WOMD/GPU experiment has yet been executed in this environment.

## v48.45.2 — LOST-SOURCE RECONSTRUCTION SUPPORT

- A/B/C uploaded failures are RC=30 source preflight failures: historical Balanced/Precision source checkpoints are gone; adaptation/certificate/gate never ran.
- Exact v48.13 reproduction is unavailable because the archived `train_ocrap_v48_13_terra.sh` recipe is missing.
- New source identity `ocrap_v48_45_source_rebuild_s7`: pooled Safe/Near/Contact S0 recovery witness from scratch once, then frozen S0 → Balanced/Precision S1 source heads.
- A manifest seals S0/Balanced/Precision SHA256; all v48.45 A/B/C/D arms must consume those exact bytes.
- SOWR 2x2 algorithm, dual ROCT, top-k=5, one shared rule and risk budgets are unchanged.
- Rebuilt-round A/B/C/D comparisons are valid; direct absolute comparison to historical v48.44 source runs is not.
- Recommended `MAX_PARALLEL_ARMS=1` because each arm already uses both GPUs internally.
