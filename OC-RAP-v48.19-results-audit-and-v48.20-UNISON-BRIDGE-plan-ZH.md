# OC-RAP v48.19 结果审计与 v48.20 UNISON-BRIDGE 优化报告

## 1. 结论摘要

本轮 `RC=20` 与 v48.18 的旧协议不可行问题不同。v48.19 的 Near/Contact fit 与 verify 支持可行性检查全部通过，adaptation、certificate、scene-disjoint、manifest 绑定和 test seal 均有效，controller 记录 `pipeline_valid=true`、`test_roots_read=false`。因此这次 `RC=20` 是一个真实的 Natural-gate 拒绝。

但它不能被简单解释成“论文 idea 无效”。代码审计发现 v48.19 的训练目标、harm 语义和部署动作之间仍有多处系统性错位，足以让候选级 AUC 与最终 gate 完全脱节：

1. `balanced_replaces_group_erm=true` 使候选级平衡 BCE 整体替换了组级动作准入目标；
2. v48.19 的部署是在冻结 top-k 内按 `sigmoid(benefit)-sigmoid(harm)` 重排，训练却用全候选上的“冻结 PCD delta + log-sigmoid tails”作为 set decision；
3. FACET 改变了 harm 的含义，但仍把旧的 signed-PCD source harm logit 作为新 component harm 的加性基线；
4. independent-tail 的部分 class weight、hard mask、intragroup mask 仍来自 signed total PCD 三分类，而不是 component-veto 标签；
5. shared + regime residual 仍通过 bucket ID 选择专属模块，与“同一模型连续适用于三个 regime”的目标不一致；
6. 单一 aggregate harm tail 没有显式输出 DRS、deployability、gap 三类主要风险分量。

因此，当前失败的根本链路是：

> **候选生成已经成功，但 admission evidence 的监督语义、组级优化对象和部署选择规则没有闭合。**

v48.20 不放宽 gate，不重建数据，不读取 test，而是把模型改为一个不接收 regime ID 的统一证据模型，并把训练目标与 certificate/closed-loop 的真实 top-k 动作选择严格对齐。

---

## 2. 三个 regime 是否达到初步 CCF-A 投稿目标

### 2.1 Safe：可以形成初步非干扰主张，但不能单独支撑投稿

Safe 外部结果中，nominal replay、log replay 和 Wayformer-BC 均达到：

- DRS = 1.000；
- bounded NUP = 1.000；
- intervention = 0；
- secondary collision = 0；
- yaw-rate violation = 0。

BeTopNet-lite 和 GameFormer-lite 虽然 DRS 同为 1.000，但 intervention 分别为 0.176 和 0.392，GameFormer-lite 的 NUP 下降到 0.947。

因此 Safe 的正确论文主张不是“恢复算法在 Safe 中提高收益”，而是：

> **同一个统一恢复机制在无恢复必要时锁定 nominal，不污染正常驾驶。**

已有 Safe nominal lock/non-inferiority 结果足以作为初步安全边界证据。但 Safe 本身没有恢复挑战，不能替代 Near/Contact 的核心有效性结果。

**状态：初步可用。**

### 2.2 Near-contact：尚未达到初步投稿目标

v48.19 Near 最好的候选级 benefit AUC 仍有方向性：

- main Balanced：0.759；
- main Precision：0.708；
- ablation C shared-only Balanced：0.800。

冻结 proposal 也非常强：main 中 oracle-best hit 为 0.986–0.990，消融中多数为 1.000。这证明恢复集合中几乎总能找到正确候选，候选生成不是当前瓶颈。

但 gate 需要的是经过选择后的可部署结果，而不是 candidate AUC：

- 所有 main/ablation verify 都选择 0 个 group；
- non-positive group false switch 仍为 0.859–0.906；
- harmful ranked switch 为 0.376–0.397；
- proposal evidence harm AUC 仅约 0.480；
- evidence/teacher correlation 很弱，top-1 correlation 仍为负。

Near 外部 offline baseline 中，predictive safety filter 的当前参照为：

- DRS 0.973；
- deployability 0.547；
- bounded NUP 0.988；
- intervention 0.446；
- secondary collision 0.098；
- FRA 0.127；
- ODG 0.174。

OC-RAP 目前没有 gate-authorized closed-loop Near 结果，无法证明优于该 baseline。Near 的若干 closed-loop baseline 压缩包仍处于 30–34/50 scenes 的 `running_scene`，其中 `closed_loop_marc_lite.json` 声称 50 scenes，但 progress 为 34、journal 只有 31 行，不能进入论文表格。

**状态：未达到投稿目标。**

### 2.3 Contact：明显未达到初步投稿目标

Contact 是目前最薄弱的 regime：

- main Balanced candidate benefit AUC 0.580、harm AUC 0.477；
- main Precision benefit AUC 0.479、harm AUC 0.526；
- Precision proposal benefit AUC 只有 0.357；
- evidence/teacher correlation 为负或接近零；
- positive top-1 accuracy 仅 0.438–0.594；
- positive regret 约 0.119–0.152；
- 所有证书规则最终均为 0 coverage。

最接近 gate 的 Precision-Contact fit 规则选择 26 个 group，却只有 3 个正机会、5 个 harmful：

- precision = 0.115；
- one-sided 90% precision LCB = 0.057；
- conditional harmful UCB = 0.308；
- mean teacher advantage = -0.014。

同一规则到 verify 后选择 37 个，只命中 1 个正机会，并包含 16 个 harmful：

- precision = 0.027；
- precision LCB = 0.008；
- harmful rate = 0.432；
- harmful UCB = 0.537；
- selected macro 高度集中。

这不是阈值差一点，而是 ranking/admission 在独立 scene 上失真。

Contact 完整 50-scene 外部 closed-loop baseline 的参照为：

| 方法 | DRS | Deployability | NUP | Intervention | Collision scenes |
|---|---:|---:|---:|---:|---:|
| Post-impact MPC | 0.527 | 0.348 | 0.454 | 0.976 | 0.18 |
| Restoration | 0.494 | 0.324 | 0.807 | 0.833 | 0.08 |
| Severity minimization | 0.492 | 0.320 | 0.455 | 0.945 | 0.10 |
| Post-crash braking | 0.417 | 0.297 | 0.462 | 1.000 | 0.28 |

OC-RAP 的潜在论文优势应是：接近 Restoration/MPC 的 DRS 与 deployability，同时显著降低 intervention、维持更高 NUP，并严格控制 collision。但当前没有有效 gate，更没有授权 closed-loop 结果，尚不能提出该优越性主张。

**状态：未达到投稿目标。**

### 2.4 总体判断

当前整体尚不具备 CCF-A 投稿所需的主结果。Safe 主张可用，Near 有重要的 proposal/benefit 前兆，但 Near/Contact 的 selective recovery 仍未被独立证书和 closed-loop 证明。

---

## 3. v48.19 主实验说明了什么

### 3.1 Proposal 已经不是瓶颈

四个 main variant/regime 的 proposal oracle-best hit 为 0.982–0.991；positive groups 上的 any-positive hit 也约为 0.969–1.000。说明：

- recovery set 覆盖率足够高；
- 不应继续频繁修改 proposal、macro 生成或 top-k 基础排序；
- 当前资源应集中到 evidence 与 admission。

冻结 proposal 还保留了因果归因：新版本的变化只影响“是否安全执行”和“top-k 内选择谁”。

### 3.2 Near benefit 可学，但没有转化为安全动作

Near candidate benefit AUC 0.708–0.759，说明 source features/context 包含恢复收益信号。但从 candidate 到 policy/top-k evidence 后 AUC 降低，harm AUC 接近随机，且 non-positive false switch 极高。

这意味着 benefit classifier 的局部排序能力不能自动构成 selective policy；必须用组级 nominal-vs-recovery objective 训练实际动作。

### 3.3 Contact benefit 与 harm 都没有稳定可辨识性

Contact benefit/harm 在不同 variant、不同表示和不同 checkpoint 间围绕 0.5 摆动，且 fit→verify 明显崩塌。这说明不是“模型参数少了”或“grid 太粗”，而是监督和 representation 对 Contact 的关键风险机制不充分。

### 3.4 数据支持有限，但不是当前 gate 失败的工程借口

FACET deployable candidates 中：

- Near beneficial 45/1425，component-harmful 769/1425；
- Contact beneficial 138/4086，component-harmful 1855/4086。

进一步按“beneficial 且不 component-harmful”的 safe positive group 统计：

- Near 只有 11 groups、7 scenes；
- Contact 41 groups、17 scenes。

Near 的 group-level safe-admission 监督非常稀疏，泛化难度确实高。但 v48.19 的协议支持检查已经证明 gate 不是数学不可满足；本轮首先修复训练语义和部署对齐，而不是重建数据或放宽证书。

---

## 4. v48.19 消融的因果结论

| 消融 | Balanced Near B/H AUC | Balanced Contact B/H AUC | Precision Near B/H AUC | Precision Contact B/H AUC | Gate |
|---|---:|---:|---:|---:|---:|
| A component-veto separate | 0.764 / 0.570 | 0.573 / 0.496 | 0.742 / 0.513 | 0.485 / 0.499 | 全失败 |
| B shared + regime residual | 0.772 / 0.544 | 0.580 / 0.472 | 0.712 / 0.526 | 0.480 / 0.527 | 全失败 |
| C shared-only | 0.800 / 0.504 | 0.584 / 0.441 | 0.717 / 0.527 | 0.479 / 0.525 | 全失败 |
| D full FACET | 0.718 / 0.586 | 0.585 / 0.497 | 0.712 / 0.526 | 0.480 / 0.527 | 全失败 |

### A：component-veto 标签方向合理，但 aggregate tail 不够

A 在 Balanced Near 获得本轮相对较高的 harm AUC 0.570，说明 component-veto 比 signed-total-delta harm 更接近真实问题。但 Contact 仍约为随机，说明把多个物理风险压成一个 scalar tail 会丢失结构。

### B：shared + regime residual 没有产生稳定增益

B 对 Balanced Contact benefit 有很小改善，但 harm 反而下降；Precision 与 D 几乎完全相同。没有证据表明 bucket-selected residual 是必要模块。

### C：去掉 regime residual 后 Near benefit 最好，但 harm 恶化

C Balanced Near benefit AUC 0.800，是所有消融最高；这支持“共享跨 regime 表示”而不是 regime-first routing。但其 Contact harm AUC 0.441，说明 shared-only 仍需显式 component supervision 和组级 admission，而不是回到一个共享 scalar tail。

### D：当前 checkpoint metric 未被证明有效

D Precision 与 B 的 best epoch、Near/Contact AUC 和 certificate 结果相同；Balanced 虽选择不同 epoch，但没有产生 coverage。它只改变 early stopping 分数，不修复错误的训练目标，因此不能成为有效模块的证据。

---

## 5. 到目前为止足够有效、应继续保留的设计

1. **Observation-consistent frozen recovery proposal / top-k tournament。** Oracle hit 约 98–100%，是目前最确定的算法资产。
2. **Scene-disjoint adaptation/dev/certificate。** 能区分训练失败、阈值过拟合和真正证书拒绝，必须保留。
3. **Protocol preflight、manifest SHA256、teacher-index contract、test seal。** 这些工程约束已经消除了 v48.17/v48.18 的归因污染。
4. **Safe nominal lock。** Safe 的正确目标是无污染，且外部 baseline 也证明主动干预没有必要。
5. **Zero-init、bounded correction 的思想。** 当 source 与 target 语义相同时，它能保护 warm start；但不能用于把旧 signed-PCD harm 强行迁移到新 component harm。
6. **Context 条件化思想。** 候选集合关系有信息，但当前 tournament embedding/aggregate tail 只获得不稳定小幅提升，需改变利用方式。
7. **独立 benefit/harm 假设。** 一个候选可以同时有恢复收益与残余风险，该建模原则应继续保留。

---

## 6. 已被证明无效或当前没有证据支持的设计

1. **先按 regime/bucket 选 calibrator 或 residual。** 没有稳定增益，且削弱统一算法的 novelty。
2. **候选级 balanced BCE 替换组级 ERM。** 这是 v48.19 的关键实现错误；它优化标签分类，不优化最终动作。
3. **改变 harm 语义后仍锚定旧 source harm。** 旧 source harm 是 signed PCD 的互补 tail，证书已显示其近随机；不能作为 component risk 的先验。
4. **单一 aggregate harm scalar。** DRS、deployability 和 gap 是当前主要触发源，hard/harm_proxy 几乎无支持；必须至少显式监督前三个分量。
5. **Sampler/balance 本身。** 只能改变样本频率，不能解决 train-deploy mismatch 或标签语义错误。
6. **当前 cross-regime checkpoint metric。** 在错误 objective 上选 epoch，不能创造可辨识性。
7. **继续增加 raw context 或 calibrator 参数。** 数据的 safe-positive scene 支持太少，大模型更容易记忆 scene。
8. **继续调 threshold/grid。** Precision-Contact 的 near miss 与 gate 相距很远；增加 grid 仅用于避免搜索离散误差，不能根治。
9. **把 hard violation 和 harm_proxy 作为主要可学习 tail。** 当前 harm_proxy 没有正增量，hard increase 极少；应暂时保留为 certificate deterministic veto，而不是让网络拟合不存在的信号。

---

## 7. 发现并修复的工程错误

### 7.1 候选 BCE 替换组级动作目标

v48.19 实际配置：

```text
ORDINAL_EVIDENCE_BALANCED_REPLACES_ERM=true
SETWISE_W=0
pairwise/intragroup/setwise admission ≈ 0
```

这导致主要梯度来自候选级 benefit/harm BCE，而部署执行 nominal 与 top-k recovery 的组级选择。

v48.20 改为：

```text
BALANCED_REPLACES_ERM=false
setwise safe-set objective = primary
candidate balance = auxiliary
```

### 7.2 训练 score 与部署 score 不一致

旧 setwise 路径使用：

```text
frozen PCD delta + log sigmoid(benefit) + log sigmoid(non-harm)
```

并在所有 recovery candidate 上训练。

certificate/closed loop 实际使用：

```text
frozen rank top-k
→ score = sigmoid(benefit) - sigmoid(harm)
→ top-k 内 evidence rerank
```

v48.20 的 safe-set、selective harmful mass 和 coverage 现在全部只在冻结 top-k 上计算，并使用完全相同的 deployed score。新增梯度测试保证 top-k 外候选不会污染该组级 loss。

### 7.3 Harm semantic transfer 错误

FACET 把 harm 从 signed total PCD 改成 component veto，却仍把旧 source harm logit 加到 residual 上。v48.20 采用非对称 transfer：

- benefit：保留 source transfer；
- harm：semantic reset，component heads 从绝对零 logit 初始化，不再加旧 harm base。

### 7.4 Factorized labels 仍被三分类 mask 污染

旧代码中 independent tails 的部分 class weight、hard mining、margin、intragroup ranking 仍使用 `teacher_delta <= -negative_gain` 的 signed class。v48.20 全部改成 component-specific/binary masks。

### 7.5 Component 边界判定不一致

训练/证书统一使用严格 `margin > 0`。`margin == 0` 不再被视作 harmful 的软边界。

### 7.6 不安全的 normalized smooth envelope

归一化 soft-min/soft-max 位于专家/分量取值范围内部，会高估最差专家 benefit，并可能让低风险分量稀释一个高风险 veto。v48.20 改为：

- source benefit = 两个专家的精确 `min`；
- component harm = 三个分量 logit 的精确 `max`。

### 7.7 Gradient norm 数值稳定性

补充 double-precision global norm 计算，避免有限 float32 梯度在 norm 聚合时溢出并被误判为训练失败。

### 7.8 外部 baseline 工件一致性

新增 baseline audit：closed-loop summary 仅在 progress=`complete` 且 progress/summary/scene-journal 数量一致时可用于论文。Near 当前 7 个 closed-loop 工件均不满足；Safe offline 与 Contact 50-scene closed-loop 可用。

---

## 8. v48.20 UNISON-BRIDGE

**UNISON = Unified Non-regime-specific Intervention Selection with Observation-consistent Non-compensatory evidence。**

### 8.1 一个模型，而不是三个 regime 策略

- 模型不接收 regime/bucket ID；
- 不选择 Near/Contact calibrator；
- 每个候选同时评估两个冻结 source experts；
- 统一 evidence calibrator 读取两专家输出、均值、分歧、冻结 policy margin 和 tournament context；
- Safe 不是另一套 policy，而是统一动作集合中的 nominal invariant boundary。

Near/Contact 仍用于统计报告与 worst-regime checkpoint evaluation，但不用于模型路由。

### 8.2 Conservative benefit transfer

Benefit 使用两个冻结 source experts 的精确下包络：

```text
base_benefit = min(benefit_expert_1, benefit_expert_2)
benefit = base_benefit + bounded shared residual
```

这保留已有 Near benefit，同时把专家分歧作为置信下降，而不是先判断 regime。

### 8.3 Componentwise harm semantic reset

统一 calibrator 显式输出：

```text
harm_DRS
harm_deployability
harm_gap
```

每个 head 为零初始化、有界、candidate-vs-nominal 的绝对 logit；aggregate harm 为精确最大值。任一风险分量高即 veto，其他分量不能补偿。

hard violation 与 harm_proxy 因监督支持不足，当前保留为 certificate deterministic veto，不伪造可学习增益。

### 8.4 Deployment-exact safe-set objective

每个 group：

1. 冻结 tournament 生成 top-k；
2. teacher safe set = `beneficial AND not component-harmful` 且位于 top-k 的候选；
3. 若 safe set 为空，nominal 是唯一组级目标；
4. 若非空，在 safe set 内按 teacher advantage 给软分布；
5. 模型 score 与部署完全一致：`sigmoid(benefit)-sigmoid(harm)`；
6. candidate tail balance、component BCE、intragroup ranking 只作为辅助监督，不替代组级 objective。

### 8.5 Novelty 的论文表达

建议将方法链写成：

```text
Observation-consistent recovery set
→ frozen high-recall proposal
→ unified dual-source conservative benefit transfer
→ componentwise non-compensatory harm evidence
→ deployment-exact safe-set admission
→ scene-disjoint statistical certificate
→ Safe nominal invariant
```

它不是“先分类 regime，再调用三套策略”，而是在一个统一候选—证据—准入模型中自然覆盖无碰撞、临界接触和碰撞后状态。

---

## 9. 下一步非重复消融

四组均与 v48.19 A/B/C/D 不重复：

1. **A_candidate_tail_only**：统一模型 + component heads，只训练 candidate tails，不启用 safe-set。验证组级 objective 是否为关键增益源。
2. **B_safe_set_aggregate_harm**：统一模型 + deployment-exact safe-set，但不使用 component heads。验证显式分量风险是否必要。
3. **C_component_safe_set_no_balance**：统一模型 + component heads + safe-set，关闭全局 auxiliary balance。验证 balance 是否只是辅助。
4. **D_full_unison**：component heads + deployment-exact safe-set + global auxiliary balance + robust checkpoint metric。

每个 variant 分一波，四任务并发：GPU0 两个、GPU1 两个；Balanced 波完成后再运行 Precision 波，避免 8 个任务争抢 CPU/磁盘。

---

## 10. 实验决策规则

### 主实验 `RC=0`

- 只在自动生成 `NEXT_COMMANDS.txt` 后运行 stress/closed-loop；
- 先报告 Near/Contact certificate coverage、LCB/UCB、macro support；
- 再与可用 baseline 比较，不引用未完成 Near closed-loop baseline。

### 主实验 `RC=20`

- 不调 gate、不读 test；
- 运行四组 v48.20 消融；
- 重点判断 D 相对 A/B/C 是否同时改善：Near/Contact positive recall、harmful UCB、false intervention、fit→verify 稳定性；
- 如果 B 明显优于 A，证明 safe-set 是关键；如果 D 优于 B/C，证明 component + balance 具有互补作用。

### 主实验 `RC=30`

只排查协议、index contract、训练、checkpoint、certificate 工件。不得把它解释为算法失败，也不得在同一输出目录修改协议后续跑。

---

## 11. 本地验证边界

交付环境已完成 196 项 Python 单测、compileall、Shell 语法和静态工件检查，但没有真实 WOMD/Waymax 数据和两张 A30，因此没有声称 v48.20 已通过 Natural gate，也没有伪造 closed-loop 指标。新版本的价值是把下一次 `RC=20` 尽可能收敛为真正的算法归因，而不是训练—部署或标签语义错误。
