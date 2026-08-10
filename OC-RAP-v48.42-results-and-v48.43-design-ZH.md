# OC-RAP v48.42 有效 2×2 结果归因与 v48.43 POET 算法设计

日期：2026-08-09  
目标：在不修改数据集、不放松 Natural gate、不引入 Safe/Near/Contact 分支策略的前提下，定位 v48.42 的真实瓶颈并给出下一轮最小、可归因、与论文核心机制一致的算法修改。

---

## 1. 结论先行

本轮上传的 A/B/C/D(Main) **四组都是有效 RC=20 算法结果**。四组均满足：

- `pipeline_valid=true`
- `certificate_executed=true`
- `gate_evaluated=true`
- `dominant_failure_layer=development_rule_fit`

因此，上一轮基于 v48.41 的“C/D 因 scalar checker bug 不能评价”的判断已经被当前 v48.42/48.42.1 有效结果覆盖。本轮 C、D 可以且必须用于算法归因。

最终判断：

1. **v48.42-B partial-pooling harm residual 不值得吸收进主算法。** Near deployability safe-positive false-veto 仅 16/18→15/18，同时 conditional AUC 0.482→0.420，说明它主要在移动分数而不是改善安全临界面的可分性。Contact 虽把 deployability false-veto 27/31→21/31，但 harmful false-safe 和 harmful-selected UCB 明显恶化。
2. **v48.42-C rank-benefit skip 不值得吸收。** Contact candidate safe-positive AUC 0.582→0.580，certificate recall 保持 0.10，frozen rank 中已有的信息没有被该 skip 转成可部署的物理 benefit frontier。
3. **v48.42-D/Main 也不值得保留为主算法。** Contact recall 0.10→0.15 是局部改善，但 harmful UCB 0.484→0.503；development 虽少了两个 constraint failures，却在 Near 和 Contact 都选不到 safe-positive，属于“数值 deficit 变小但策略更没有有效正例”的假改善。
4. **shared development rule 仍未进入 statistically feasible window。** proposal oracle 在 Near/Contact 都可行，所以主要问题不是 candidate generation，而是 candidate 已存在后，benefit/harm evidence 无法形成同一套可部署 shared rule。
5. **下一步不应再加 harm capacity 或 rank loss。** v48.42 自己预注册的 stop rule 已被触发。更合理的问题是：模型是否拥有与论文“candidate prefix 执行后的 observation equivalence”直接对应的候选相关结构信息。
6. 因此下一版实现 **v48.43 POET — Post-prefix Observation-Equivalence Transport**：将 frozen latent-root/post-prefix observation kernel 的低维结构签名，按 candidate-minus-nominal 的方式注入 dual OCAF benefit/harm context。它不使用 regime ID，不做 regime routing，不设置 regime-specific threshold。

---

## 2. v48.42 四组实验的权威性

| Arm | Partial-pool harm | Rank-benefit skip | authoritative RC | 可做算法归因 |
|---|---:|---:|---:|---:|
| A | × | × | 20 | 是 |
| B | ✓ | × | 20 | 是 |
| C | × | ✓ | 20 | 是 |
| D/Main | ✓ | ✓ | 20 | 是 |

四组的 proposal oracle 都仍然可行：

- Near-contact top-5：约 9 个 safe-positive proposal groups；
- Contact top-5：约 20 个 safe-positive proposal groups。

因此不要再优先改 proposal top-k、候选生成或 threshold grid。历史 changelog 已经多轮表明这些方向不是当前主瓶颈。

---

## 3. 这一轮最值得观察的四个量：逐项结论

### 3.1 B 是否显著降低 Near safe-positive deployability false-veto？

**否。**

Precision certificate：

| 指标 | A | B | 变化 |
|---|---:|---:|---:|
| Near deployability safe-positive false-veto | 16/18 = 0.889 | 15/18 = 0.833 | 只救回 1 个 |
| deployability harmful-vs-safe-positive AUC | 0.482 | 0.420 | **下降** |
| safe-positive deployability harm median | 0.806 | 0.782 | 小幅平移 |
| harmful false-safe fraction | 0.106 | 0.184 | **上升** |
| certificate recall | 0.000 | 0.111 | +1 个正例 |
| harmful-selected UCB90 | 0.503 | 0.433 | 表面改善，但 conditional discrimination 变差 |

这组结果不符合“partial pooling 学会了 rare-frontier deployability”的解释。更合理的解释是：residual 改变了 calibration / score level，使少量 candidate 穿过阈值，但没有把 safe-positive 与 harmful 在 deployability 维度分开。

因此 **partial pooling 的参数化设计不进入主算法**。

### 3.2 C 是否把 Contact frozen rank AUC≈0.733 转成更高 opportunity / positive capture？

**否。**

Precision certificate：

| 指标 | A | C |
|---|---:|---:|
| Contact candidate safe-positive AUC | 0.5821 | 0.5797 |
| candidate positive AUC | 0.5631 | 0.5612 |
| proposal safe-positive AUC | 0.5609 | 0.5609 |
| proposal positive AUC | 0.4470 | 0.4440 |
| certificate positive selected | 2 | 2 |
| certificate recall | 0.10 | 0.10 |
| harmful UCB90 | 0.484 | 0.498 |

这说明“把 frozen rank_adv 以 bounded positive skip 加到 benefit raw score”不是缺失的映射。rank knowledge 可能描述的是一般 preference/top-k correctness，而不是与 teacher PCD benefit frontier 同构的因果特征。

因此 **停止继续 stacking rank objectives / rank skip**。

### 3.3 D 是否同时保持 Near 低 harmful contamination 和 Contact benefit improvement？

**否。**

D 的 Contact candidate safe-positive AUC 提升到 0.594，recall 提升到 0.15，但：

- harmful-selected UCB90：A 0.484 → D 0.503；
- deployability harmful false-safe：A 0.096 → D 0.195；
- D Contact selected 53、harmful 22；
- Near harmful-selected UCB90：A 0.503 → D 0.535；
- Near development positive-selected 从 A 的 1 变成 0；
- Contact development positive-selected 始终为 0。

所以 D 的行为不是“两个机制互补”，而是 **partial-pool 的更低 veto + rank skip 的 score shift 一起扩大 admission 面积**，带来一些 positive capture，同时也明显引入 harmful candidate。

### 3.4 shared development rule 是否进入 statistically feasible window？

**仍然没有。**

Precision development nearest shared rule：

| Arm | Near sel/pos/harm | Contact sel/pos/harm | failures | deficit |
|---|---:|---:|---:|---:|
| A | 5 / 1 / 0 | 9 / 0 / 3 | 6 | 141.14 |
| B | 6 / 1 / 0 | 9 / 0 / 4 | 5 | 148.71 |
| C | 5 / 1 / 0 | 7 / 0 / 3 | 6 | 154.37 |
| D | 6 / 0 / 0 | 16 / 0 / 6 | 4 | 135.48 |

D 的 failures/deficit 最低不能被解释成接近 RC=0，因为两个关键 stratum 都没有 safe-positive selected。真正的可行窗口应至少先出现：

- Near 和 Contact development **同时有非零 safe-positive capture**；
- harmful selected 不随 coverage 同比例上涨；
- precision LCB / harmful UCB / min-selected 同时趋近门槛。

当前还没有发生。

---

## 4. 哪些 v48.42 设计值得吸收？

### 不吸收的具体模块

- `partial_pool_harm_residual`：不吸收；
- `rank_benefit_skip`：不吸收；
- 两者的 D 组合：不吸收。

### 值得吸收的是“实验结论”，而不是参数模块

1. B 在 Contact 上可以大幅降低 safe-positive veto，证明当前 harm score 的 operating point 确实过于保守；但 harmful false-safe 同步增加，说明 **缺的是 conditional information，不是更低阈值/更大 residual**。
2. C 的失败证明 generic rank signal 与 deployable benefit 并不等价；下一步 benefit 改进应找 **与论文物理定义一致的候选后验结构量**。
3. v48.40 的 dual OCAF 仍应保留：benefit/harm task-level decoupling 有历史正证据，v48.42 并未否定它。
4. shared harm representation、component veto、bounded HAF、joint reserve、support reliability、top-k=5 和 one shared rule 继续保留。

---

## 5. Near-contact / Contact 面向 CCF-A 的投稿目标

CCF-A 本身没有规定这些具体数值阈值。下列数值应理解为本项目已有 `OC-RAP-CCF-A-targets.csv` 中的 **内部 submission-readiness bar**，用于避免“RC=0 就等于论文已够强”的误判。

### 5.1 Near-contact

算法/证书层：

- fit precision LCB ≥ 0.50；verify precision LCB ≥ 0.40；
- recall 至少 0.25–0.33；
- harmful-selected UCB ≤ 0.22–0.25；
- shared rule 必须不是靠近乎全 abstain 达成；
- deployability conditional AUC 应明显离开 0.5，建议内部强目标至少 ~0.65，最好 ≥0.70；
- Safe NUP loss ≤2–3%。

closed-loop 层：

- min-TTC p05/LCB 改善 ≥0.2s 和/或 clearance ≥0.1m，并有 CI；
- collision / hard recovery / hard brake 明显下降；
- route progress 不显著退化；
- 不能靠 freeze/stop 获得 safety。

**当前差距**：certificate recall 最好只有 0.111，距 0.25–0.33 约需要 2–3 倍正例 capture；harmful UCB 大致仍在 0.43–0.65 区间；deployability rare frontier 接近随机。

### 5.2 Contact

算法/证书层：

- fit/verify precision LCB ≥0.50/0.40；
- recall ≥0.20–0.30；
- harmful-selected UCB ≤0.22–0.25；
- shared development rule Contact positive-selected 必须从当前的 0 变成稳定非零；
- benefit/safe-positive candidate discrimination 至少进入有明显结构信号的区间，内部建议 candidate safe-positive AUC ≥~0.65，再通过 policy-level top-k/selector 证明可部署 capture。

closed-loop 层：

- secondary collision / re-contact 绝对下降 ≥2pp 或获得同等级统计明确改善；
- post-contact TTC +≥0.2s；
- overlap duration / impact delta-v 降低；
- stable stop / route rejoin 改善；
- yaw、progress 不劣。

**当前差距**：recall 只有 0.10–0.15，harmful UCB 约 0.48–0.56；大致需要把 positive capture 翻倍，并把 harmful selection 风险近乎减半，才接近内部投稿线。

---

## 6. 当前主要瓶颈

### 6.1 Near：deployability frontier 的可识别性，而不是 capacity

v48.42-A 中 18 个真正 safe-positive candidate 有 16 个被 deployability harm>0.5 veto。B 只救回 1 个，而且 AUC 更差。这说明问题不是“head 太小”，而是输入证据不足以回答：

> 执行这个候选 prefix 以后，哪些 latent futures 仍会在观测上混淆？这些混淆是否使 recovery option 不可共同执行？

当前 OCAF 的 action 侧已经是 candidate-minus-nominal executable physics；但 observation 侧主要是 nominal row 的 current/pre-prefix observation 广播。它并没有显式把 candidate-dependent **post-prefix observation-equivalence geometry** 交给 evidence adapter。

### 6.2 Contact：benefit identifiability + harm frontier 双瓶颈

Contact 的 candidate safe-positive AUC 约 0.58，说明 benefit identification 较弱；与此同时 deployability safe-positive 仍大量被 veto。rank skip 失败后，不能再把 frozen preference rank 当作 benefit proxy。

更合理的统一假设是：Contact 与 Near 都需要知道“动作执行后信息结构如何改变”。Near 主要利用它判断 false recoverability / unsafe ambiguity；Contact 则可能利用它识别哪些 recovery prefix 真正降低后续不可辨识冲突、保留稳定 recovery affordance。

这正好是同一个物理/信息结构机制，而不是两个 regime 策略。

---

## 7. v48.43 POET：为什么是当前最该做的修改

论文的核心定义明确依赖 **post-prefix observation equivalence**：候选 action 被执行后，部署系统不能区分的 latent roots 必须共享兼容 recovery option。当前 direct evidence adaptation 没有把这个候选相关结构显式送入 OCAF，属于论文定义和学习路径之间的缺口。

### 7.1 结构签名

对 candidate `a`，使用已有、冻结的 latent-root decoder / observation embedding 得到：

- root distribution `p(z|a)`；
- post-prefix observation compatibility `C_a(i,j)`。

构造 4D bounded signature：

1. `H_norm(p)`：root entropy；
2. `sum_ij p_i p_j C_ij, i!=j`：weighted alias mass；
3. `sum_i p_i max_{j!=i} C_ij`：peak alias pressure；
4. `max_i p_i`：root concentration。

然后只用相对 nominal 的结构变化：

`Δψ(a)=ψ(a)-ψ(a0)`。

### 7.2 注入方式

已有 dual OCAF context：`z_b`, `z_h`。

POET：

- `z_b' = z_b + s W_b Δψ(a)`
- `z_h' = z_h + s W_h Δψ(a)`

其中：

- `W_b/W_h` 是 bias-free linear；
- 零初始化；
- `ψ` detach / frozen teacher；
- nominal candidate `Δψ=0`，严格零 transport；
- 不改变 shared root/observation model；
- 不含 regime ID；
- 无 Near/Contact 专属参数。

这与 v48.42 partial-pool 最大区别是：它增加的不是“每个 harm component 更多自由度”，而是 **候选动作执行后观测结构的缺失变量**。

---

## 8. v48.43 2×2 为什么足够合理

| Arm | harm-side POET | benefit-side POET | 回答的问题 |
|---|---:|---:|---|
| A | × | × | v48.42-A / dual-OCAF reference |
| B | ✓ | × | post-prefix observability 是否修 Near harm identifiability |
| C | × | ✓ | post-prefix observability 是否修 Contact benefit identifiability |
| D/Main | ✓ | ✓ | 两者是否互补并推动 shared rule 可行 |

它比继续做“更大 residual / 更多 loss”更容易做 CCF-A 级别的机制归因：修改对应论文的 central object，且 B/C 分别隔离两个任务侧，D 检验组合。

### 预注册 go/no-go（算法判定，不是最终论文 gate）

B 强信号：

- Near deployability false-veto 从 16/18 实质下降，建议 <=12/18；
- deployability conditional AUC 至少提升到 ~0.56 或更高；
- harmful false-safe / harmful UCB 不显著恶化。

C 强信号：

- Contact candidate/proposal safe-positive 或 positive AUC 相对 A 至少 +~0.05；
- certificate recall 至少进入 ~0.20；
- harmful UCB 不恶化。

D 强信号：

- Near harmful contamination 保持或优于 A/B；
- Contact recall ≥0.20，同时 harmful UCB 至少不高于 A；
- shared development Near/Contact **两边都出现 safe-positive selected**；
- 最终目标是 shared rule valid / RC=0，而不是只降低 deficit。

如果 POET 失败：不要放大 transport width/scale；下一阶段应转向 **recovery-option conflict signature / observation-teacher calibration**，仍然保持 regime-agnostic。

---

## 9. 已明确不要再重复的算法方向

在现有 changelog 基础上，本轮新增两项负结果：

- v48.42 partial-pooling harm residual；
- v48.42 bounded rank-benefit skip。

完整重点 non-repeat：

- threshold grid densification；
- top-k expansion；
- aggressive positive oversampling；
- hardest-negative population distortion；
- generic pairwise/listwise loss stacking；
- full identity-stage factor update；
- learned admission residual；
- v48.38 one-sided tail loss；
- v48.39 unbounded benefit/harm；
- v48.40 frontier_tanh；
- v48.41 full component factorization；
- v48.42 partial pooling；
- v48.42 rank-benefit skip；
- regime-conditioned routing / threshold / policy。

继续保留：OCAF、factor preservation、bounded HAF、dual task OCAF、component veto、support reliability、aligned deterministic joint reserve、one shared continuous rule、top-k=5。

---

## 10. 数据集对算法判断的含义（不建议本轮改数据）

reports 中 Near/Contact 的规模足够支持继续做算法归因：

- Near train: 13,324 samples / 1,800 groups / 600 scenes；calibration: 6,039 / 765 / 316；test: 4,723 / 595 / 250。
- Contact train: 16,790 / 2,000 / 500；calibration: 16,843 / 1,896 / 543；test: 6,687 / 747 / 209。
- Near train artifact fraction ~18.9%，calibration ~24.0%；
- Contact train ~16.6%，calibration ~21.2%；
- Near/Contact calibration 都具有较强 oracle-to-deployable gap / negative deployability 样本。

所以当前 RC=20 不能简单归因成“完全没有正例/没有 oracle artifact”。proposal oracle 也已经证明 candidate support 存在。

但投稿前另有一个**非本轮算法优化问题**：reports 的 `paper_support.supports_womd_primary_claim=false`。这不影响现在继续做算法优化，但最终 CCF-A 稿件里的数据来源/生成 provenance 表述必须与真实 report contract 对齐，不能写出比 artifact provenance 更强的主张。

---

## 11. 论文文本投稿前需要同步清理的一处冲突

当前 tex 主线非常明确：OC-RAP 依赖统一的 observation-consistent recovery principle，并声称连接 nominal、low-headroom、near-contact、post-contact under a single planning principle。

但 Appendix 中仍有：

`Regime-conditioned recovery admission`，并明确写了“second channel ... only in low-headroom contact regimes”。

这与本轮用户要求以及当前统一算法主线冲突，也容易让审稿人认为 novelty 依赖 hand-crafted regime switch。建议在最终论文版本中删除或改写为 **连续、可观测物理条件下的统一 protective admissibility**，不要出现 Contact-specific channel。这个论文修改应在算法结果稳定后同步完成。

---

## 12. 工程落地与复核

已实现：

- model POET flags / scale；
- 4D post-prefix observation-equivalence signature；
- candidate-minus-nominal transport；
- benefit/harm 两个独立 zero-init adapters；
- direct-only 和 full-forward 一致支持；
- checkpoint / inference / factor-cache / model contract / training contract / stage architecture 全部绑定；
- stage-transfer 白名单支持 POET exact prefix；
- v48.43 A/B/C/D wrappers；
- 双 GPU 2×2 parallel launcher；
- component/frontier summarizer；
- 2×2 comparator（明确 `test_roots_read=false`）；
- changelog 已更新。

专项验证：

- v48.43 POET tests：8 passed；
- v48.36/40/41/42/43 focused matrix：此前 49 passed / 1 CUDA-only skipped；新增 POET tests 后需以交付 validation 文件中的最终复跑为准；
- CPU full direct-evidence forward/backward preflight：PASS；
- `compileall`：PASS；
- 91 个 shell scripts `bash -n`：PASS。

完整历史 pytest suite 的第一次全量尝试没有形成 clean pass：首个确定 failure 来自历史 `tests/test_v48_12_trident.py` 读取一个当前代码包本来就不存在的 `scripts/train_ocrap_v48_12_trident.sh`，属于历史快照/交付缺文件，不是 POET runtime 回归。全量调用随后因环境时间限制未完整归类，因此最终验证状态不会把“全量 suite”伪写成 PASS；以 v48.36+ 当前链路 focused matrix、POET runtime preflight、compile/shell checks 作为本版本工程证据。

---

## 13. 下一步实验原则

1. 一次完成 A/B/C/D，避免先看一个 arm 后改变标准。
2. 两张 GPU 上允许四个 arm 同时启动；显存不够时 `MAX_PARALLEL_ARMS=2` 两波运行，不能为了 OOM 临时修改算法/损失/gate。
3. 只认每个 run 的 `AUTHORITATIVE_RUN_STATUS.json`。
4. RC=30 不做算法归因；RC=20 做完整 2×2；RC=0 仍跑完消融。
5. D/Main 只有 RC=0 才执行 controller 生成的 `NEXT_COMMANDS.txt`，进入 Safe paired non-inferiority 与 Near/Contact held-out closed-loop。
6. 不因本轮结果临时放宽 Natural gate、top-k、threshold grid 或读取 test root。

