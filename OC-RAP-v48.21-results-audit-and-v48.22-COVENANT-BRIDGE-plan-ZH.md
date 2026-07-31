# OC-RAP v48.21 结果审计与 v48.22 COVENANT-BRIDGE 优化方案

## 一、结论摘要

v48.21 的 `RC=20` 是一次真实的 Natural-gate 拒绝，而不是历史版本中出现过的参数 guard、不可满足 gate、空 certificate、数据泄漏或单分支失败。主控制器记录 `pipeline_valid=true`、`gate_evaluated=true`、`test_roots_read=false`，Balanced 与 Precision 都完成了独立、scene-disjoint 的 adaptation/dev/certificate 流程。

v48.21 确实学到了部分可迁移收益能力，但没有形成一个能够同时覆盖 Near-contact 与 Contact 的统一安全准入器：

- Balanced 在 Near 上将收益信号转换为 learned top-k benefit AUC 0.805，positive top-1 regret 仅 0.005；
- Precision 在 Contact 上首次获得 learned benefit AUC 0.613、相关性 0.166；
- 但这两种能力分别存在于两个训练分支，不能在同一分支共同出现；
- 所有主实验和消融的 verify coverage 仍为 0；
- harmful top-1 switch 仍高达约 0.445–0.698。

因此当前失败不是 proposal 找不到候选，也不是单纯阈值偏保守，而是 **raw benefit、component harm 与 final safe admission 三种不同统计语义被压缩进两个头，并由不正确的 group MIL 连接**。v48.22 将其改造成三个显式、互不污染的统一假设。

---

## 二、工程层面审计

### 2.1 Natural gate 是否有效执行

v48.21 主实验满足：

- Balanced、Precision 均有合法 checkpoint；
- adaptation、dev、certificate 按 scene disjoint；
- Near/Contact fit 和 verify pool 非空；
- target-support contract 与数据 manifest 哈希一致；
- certificate support feasibility 为 true；
- 未读取 test root；
- controller 原始及归一化返回码均为 20。

所以本轮不能把失败归因于 gate 数学不可行，也不能手工生成 `NEXT_COMMANDS.txt` 绕过证书。

### 2.2 v48.21 中会污染算法归因的工程/实现问题

#### 问题 A：安全机会 MIL 实际没有使用安全性

v48.21 的 group target 表示“top-k 中至少存在一个 raw-beneficial 且 non-harmful 的候选”，但 noisy-OR 只使用：

```text
P(any opportunity) = 1 - ∏(1 - P(opportunity_i))
```

它没有乘入 non-harm，也没有使用独立 admission probability。因此当候选同时具有总收益和 component harm 时，MIL 会奖励它的 opportunity logit 上升，这与 Contact 中的 false-safe failure 完全同向。

#### 问题 B：三个任务被塞入两个头

实际决策需要区分：

1. raw PCD benefit：恢复是否带来总收益；
2. component harm：是否造成不可补偿的 DRS、deployability 或 gap 退化；
3. safe admission：在最终部署中是否应当离开 nominal 并选择该候选。

v48.21 只有 benefit/harm 两个头，benefit 又被 safe-benefit 标签和 group admission 共同训练，而 primary gate 仍报告 raw benefit。训练语义、报告语义与证书语义不一致。

#### 问题 C：最终执行 score 没有形成唯一契约

Group MIL、safe-set loss、dev soft metric、certificate calibration、selector 和 closed-loop 没有全部消费同一个显式 admission score。于是候选级 AUC 改善并不能保证最终动作选择改善。

#### 问题 D：epoch 0 不参与 checkpoint 竞争

模型采用零初始化 residual，理论上初始状态应保持 source/consensus identity，但 v48.21 从 epoch 1 后才评估 checkpoint。如果第一个更新破坏了某个 regime，原始 identity 无法被选为 best checkpoint。

#### 问题 E：缺少安全前沿诊断

全局 harm AUC 主要衡量 harmful 与大量 dead candidate 的可分性。真正影响 gate 的是高 opportunity 候选中 safe/harmful 的区别。v48.21 没有单独报告 safe-positive AUC 和 high-opportunity conditional harm AUC，容易错误地把 0.65 左右的全局 harm AUC解释为准入能力已经足够。

### 2.3 v48.22 已加入的工程保护

- raw-benefit、component-harm、safe-admission 三个头分别保存、加载和审计；
- admission score 贯通训练、checkpoint、calibration、evaluator、selector 和 closed-loop；
- teacher index 继续绑定数据 manifest、PCD 参数和 component tolerances；
- Balanced/Precision 任一分支异常时主实验统一返回 `RC=30`；
- certificate 只允许 `0/20`，其他底层返回码归一化为 `30`；
- epoch 0 在任何 optimizer step 前执行 dev 验证并可成为 best checkpoint；
- 新增真实梯度级测试，验证高收益但 harmful 的候选会被 safe-opportunity MIL 惩罚；
- 开发过程中该测试发现 admission 分支局部变量未初始化的真实 `NameError`，已在交付前修复；
- 故障注入验证缺失 protocol 时输出 `PIPELINE_FAILED.json`，`gate_evaluated=false`、`test_roots_read=false`、最终 `RC=30`。

---

## 三、Near-contact：收益信号是否真正转化

### 3.1 主实验结果

| Variant | Candidate benefit AUC | Learned top-k benefit AUC | Learned corr. | Harm AUC | Harmful top-1 switch | Verify coverage |
|---|---:|---:|---:|---:|---:|---:|
| Balanced | 0.841 | **0.805** | -0.016 | 0.653 | 0.698 | 0 |
| Precision | 0.201 | 0.339 | 0.081 | 0.600 | 0.577 | 0 |

Balanced 的 learned AUC 0.805 和 positive top-1 regret 0.005 表明：**Near 的收益信号不是数据噪声，v48.21 的确能在一个分支中把它转化为 top-k 内候选排序能力。**

但这种转化没有成为最终结果，原因是：

- 同一模型分支的 harmful top-1 switch 仍为 0.698；
- learned score 与真实收益相关性仍接近 0；
- 最近的 fit rule 选 12 组只命中 2 个机会，precision LCB 0.071，harmful UCB 0.673；
- verify 最终仍全 abstain。

所以 v48.21 改善的是局部 candidate ranking，不是可认证 admission。

### 3.2 消融归因

- `A_safe_target_legacy_trunk Balanced`：Near learned AUC 0.808；
- `C_concord_group_mil_aggregate Balanced`：0.789；
- `D_full_concord Balanced`：0.804；
- `B_concord_candidate_only Precision`：0.720。

这说明 Near benefit 不依赖单一模块，source/context 中确有稳定信息；但 safe-target、group MIL、component heads 的当前组合无法让这种信息在 Balanced/Precision 两种 objective 下同时稳定。

### 3.3 Near 当前投稿状态

Near 尚未达到初步 CCF-A 主结果要求。它已经具备高召回 proposal 和可学习 benefit，但仍缺少：

- 非零且可迁移的 verify coverage；
- 同时满足 precision LCB 与 harmful UCB 的 admission rule；
- gate-authorized closed-loop OC-RAP 结果；
- 在同一 checkpoint 下同时保持低干预、低二次碰撞和高 NUP。

---

## 四、Contact：是否学到真实且可迁移的安全收益

### 4.1 主实验结果

| Variant | Candidate benefit AUC | Learned top-k benefit AUC | Learned corr. | Harm AUC | Harmful top-1 switch | Verify coverage |
|---|---:|---:|---:|---:|---:|---:|
| Balanced | 0.540 | 0.432 | -0.165 | 0.659 | 0.492 | 0 |
| Precision | 0.586 | **0.613** | **0.166** | 0.648 | 0.445 | 0 |

Precision-Contact 的 0.613 AUC 和 0.166 correlation 是迄今第一批表明模型开始学到部分 Contact action benefit 的证据。因此不能说 Contact 完全没有改善。

但是，这种能力仍远不足以称为“真实且可迁移的安全收益选择器”：

- Balanced-Contact 仍低于随机方向或接近随机；
- positive top-1 regret 仍约 0.141–0.178；
- Precision 最近 fit rule 选 19 组只命中 2 个机会，precision LCB 0.045，harmful UCB 0.292；
- verify coverage 仍为 0；
- Contact 的能力没有与最强 Near 能力同时出现在一个分支中。

### 4.2 为什么全局 harm AUC 改善却 gate 仍失败

Contact learned harm AUC 已约 0.648–0.659，说明 component risk 方向有效。但 harmful top-1 switch 仍约 0.445–0.492，说明风险头更擅长区分 harmful 与 dead，而不是区分：

```text
高收益且安全的恢复候选
vs.
高收益但仍存在残余风险的恢复候选
```

Natural gate 正是由第二种前沿判别决定。因此下一步不能删除 component risk，而要把它与 raw benefit 分开，并让最终 admission 对二者的 conjunction 建模。

### 4.3 Contact 当前投稿状态

Contact 仍明显未达到投稿主结果要求。投稿前至少需要：

- Contact benefit 在相同模型/相同 objective 下稳定高于随机，并跨 fit→verify 保持；
- high-opportunity conditional harm discrimination 显著提高；
- Contact verify 出现非零覆盖；
- 在非零覆盖下同时满足 precision LCB 与 harmful UCB；
- 获得授权 closed-loop，并在 DRS/deployability 接近强恢复 baseline 的同时显著降低 intervention、维持 NUP、控制二次碰撞。

---

## 五、v48.21 主实验与消融的总归因

### 5.1 已证明有效、应继续保留

1. **冻结 observation-consistent proposal/top-k**：positive top-k oracle hit 约 0.97–1.00，proposal 不是瓶颈。
2. **Safe nominal lock**：Safe 继续作为统一模型的 nominal invariant，而不是独立策略路由。
3. **Scene-disjoint adaptation/dev/certificate**：能够真实暴露 fit→verify 迁移失败。
4. **独立 component harm**：维持约 0.60–0.66 的 learned harm AUC，比早期随机 harm 明显更强。
5. **统一、无 regime ID 的 context transfer**：Near 与 Contact 的局部能力都可从同一特征体系中出现，没必要采用先分类 regime 再调用策略。
6. **零初始化有界 residual**：只在语义相同的 raw-benefit transfer 中保留；新的 admission residual 也零初始化，但不反向污染 benefit/harm。

### 5.2 被证明无效或不能按原样重复

- safe-benefit 直接覆盖 raw-benefit head；
- opportunity-only noisy-OR MIL；
- 用两个头承担三个决策语义；
- 仅依赖 candidate BCE；
- 仅提高全局 harm AUC而不检查 safety frontier；
- exact-min expert benefit；
- 固定阈值 early stopping；
- regime-specific calibrator/residual；
- proposal retraining；
- threshold-grid-only tuning；
- 在当前数据阶段重新构建 dataset。

---

## 六、v48.22 COVENANT-BRIDGE

全称：

> **Cross-regime Opportunity, Veto Evidence, and Non-regime-specific Admission with Nominal-preserving Transfer**

它仍然是一个统一模型：推理时不输入 Near/Contact/Safe regime ID，不选择不同 calibrator，也没有 regime-specific residual。

### 6.1 三个显式假设

#### Raw benefit head

- 保留 source expert consensus 与 context；
- 目标是原始 total-PCD improvement；
- 供 primary opportunity gate 和论文收益分析使用。

#### Component harm heads

- 分别预测 DRS、deployability、gap degradation；
- 使用精确 max 做 non-compensatory veto；
- 不允许一个低风险分量稀释另一个高风险分量。

#### Safe admission head

- 目标是 `raw benefit AND no component veto`；
- 用于 top-k reranking、group admission、calibration 和 runtime；
- 不再强迫 raw-benefit logit 同时表达最终执行决策。

### 6.2 Detached conservative admission prior

```text
admission_logit = detach(raw_benefit_logit)
                  - softplus(detach(harm_logit))
                  + bounded_zero_init_residual
```

Admission 梯度不会修改 benefit/harm heads，从结构上消除 Balanced 学 Near、Precision 学 Contact 的语义竞争。Admission residual 只学习 source benefit 与 component veto 尚不能表达的 context-dependent correction。

### 6.3 修正后的 group MIL

完整模型使用显式 admission probability：

```text
P(any safe action in frozen top-k)
= 1 - ∏(1 - P(admission_i))
```

两头消融没有 admission head 时，使用：

```text
P_safe_i = P(raw benefit_i) × (1 - P(harm_i))
```

不再出现 opportunity-only MIL。冻结 top-k 外候选没有 group-opportunity 梯度。

### 6.4 单一部署 score

下列阶段现在都使用同一个 candidate-vs-nominal admission score：

- safe-set/setwise loss；
- threshold-free dev checkpoint metric；
- certificate calibration；
- evaluator；
- planning selector；
- closed-loop runner。

### 6.5 Sampler 与标签解耦

- raw benefit head 始终学习 raw benefit；
- group sampler 使用 safe-positive group 做分层；
- beneficial-but-harmful overlap candidate 同时监督 benefit=1、harm=1、admission=0；
- 它们不会再被作为正 admission group 过采样。

### 6.6 Epoch-zero 与新诊断

- source/consensus identity 在训练前验证并可成为 best checkpoint；
- 新增 candidate safe-positive AUC；
- 新增 learned top-k safe-positive AUC；
- 新增 high-opportunity conditional harm AUC；
- checkpoint metric 增加最坏 regime harmful policy mass 和 false-admission mass，但不更改 primary Natural gate。

---

## 七、v48.22 非重复消融

| Group | 设计 | 回答的问题 |
|---|---|---|
| A_two_head_safe_probability | raw benefit + component harm；修正为 `P_b(1-P_h)`；无 admission head | v48.21 的主要失败是否只是 MIL/score 工程错误 |
| B_triad_candidate_only | 三头 + admission BCE；无 group MIL/setwise | 第三个 admission hypothesis 本身是否有效 |
| C_triad_group_mil_aggregate | admission + group objectives；单 aggregate harm | component heads 是否有独立增益 |
| D_full_covenant | 三头、component veto、safe-set、MIL、统一 score | 完整方法 |

判读规则：

- A 显著改善：说明 v48.21 的主要问题是 unsafe MIL 与 score 不一致；
- B > A：说明显式 admission hypothesis 有独立价值；
- C > B：说明 group-level supervision 有独立价值；
- D > C：说明 component risk heads 对安全前沿有独立价值；
- 全部仍为 0 coverage：停止调 gate，应检查 Contact context/teacher 是否对高机会安全前沿具有信息充分性，并考虑不改变 dataset 的 representation-level counterfactual consistency 学习。

所有 8 个任务一次性同时启动，按任务序号轮询分配：GPU0 四个、GPU1 四个。每任务默认 `NUM_WORKERS=1`、`BATCH_SIZE=48`，限制 OMP/MKL/OpenBLAS 线程以减少 CPU 与磁盘争用。

---

## 八、下一轮成功标准

Natural gate 仍是唯一通往 stress/test 的授权条件。除了 `RC=0`，还需要重点检查：

1. Near 与 Contact learned safe-positive AUC 是否在同一 variant 同时提高；
2. high-opportunity conditional harm AUC 是否明显高于全局随机水平；
3. harmful top-1 switch 是否下降，而不是只提高 global harm AUC；
4. fit 中是否出现非零、较高 precision 的 rule，并可迁移到 verify；
5. Near 与 Contact verify coverage 是否同时非零；
6. Balanced/Precision 是否不再呈现互补 specialization；
7. epoch 0 是否优于所有训练 epoch——若是，说明当前 adaptation 仍在破坏 source representation。

若主实验 `RC=0`，只运行 certificate 自动授权的 stress command，再做多 seed、同 checkpoint 的 Safe paired non-interference 和 Near/Contact closed-loop。若 `RC=20`，运行本轮 A/B/C/D，但不得读取 test。若 `RC=30`，只修复日志指出的工程阶段，不能做算法结论。

---

## 九、本地验证

- `pytest`：209 passed，5 warnings；
- `python -m compileall -q src tools tests`：通过；
- 全部 `scripts/*.sh`：`bash -n` 通过；
- COVENANT 小型单测模型可训练参数：benefit 389、harm 415、admission 389，总计 1193，小于 8000；
- 缺失 protocol 故障注入：正确返回 RC=30；
- `PIPELINE_FAILED.json` stage 为 `protocol_audit`；
- `gate_evaluated=false`、`pipeline_valid=false`、`test_roots_read=false`。

当前环境没有真实 WOMD/Waymax 数据和两张 A30，因此没有声称 v48.22 已通过 gate 或获得闭环提升。
