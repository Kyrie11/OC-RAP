# OC-RAP v48.16 结果审计与 v48.17 BRIDGE 优化报告

日期：2026-07-30

## 0. 结论先行

1. **目前 Natural gate 没有通过，而且这一次是有效的算法性拒绝，不是空 certificate 或 controller 误报。** 上传的 v48.16 消融包包含 8 个完整任务，Near 与 Contact 的 certificate 均为非空、scene-disjoint、可统计的独立验证；所有任务最终在 verify 上都选择 0 个 group，因此 gate 返回 20 是真实结论。
2. **根本瓶颈不是 proposal 找不到恢复动作，而是 Evidence 无法识别“哪个 proposal 值得安全执行”。** source balanced 在正机会组上的 top-3 命中率为 100%；Near/Contact 的正机会 top-1 准确率约为 0.643/0.594。相反，Evidence 的 harmful AUC 接近或低于随机，非正组无约束 false-switch 率超过 0.90。
3. **v48.16 ANCHOR 基本没有改变目标域 decision boundary。** B/C/D 与 A_source 的 candidate AUC 差异仅约 1e-5 到 2.6e-4，所有 verify 仍为 0 coverage。这不是“约束太严格”这么简单，而是训练目标、采样和校准器条件表达能力共同失效。
4. **Safe 在当前上传结果的可用指标上通过非劣，但还不能形成论文级 Safe claim。** 120 个 paired scenes 上 collision、offroad、NUP、intervention、jerk、yaw-rate 完全相同；然而 route progression 未输出，旧分析器也没有对 jerk/yaw-rate设置非劣 margin。Safe 结果主要证明 nominal lock 的工程正确性，不证明恢复策略有效。
5. 已完成代码升级为 **v48.17 BRIDGE**：保留 top-k proposal、scene-disjoint certificate、source identity 与 Safe nominal lock；加入三概率单纯形残差、候选上下文、batch/regime 级类别平衡、Evidence 分层 batch，以及最低正机会召回约束的 checkpoint 选择。

---

## 1. 审计范围与判定原则

本轮同时阅读和交叉核对了：

- 论文 `post-collision(1).tex`：用于还原研究问题、OC-RAP/OC-MERO/CRISP、R_dep 与 R_orc、FRA/ODG/DRS/NUP 等理论目标；
- `大模型建议(1).md`：用于确认 v48.16 的实验逻辑、gate 合同、Safe paired 修复及禁止重复项；
- `OC-RAP.zip`：作为当前真实算法与工程逻辑的唯一实现依据；
- `ALGORITHM_CHANGELOG.md`：用于排除已经失败或已经尝试过的方向；
- `ocrap_v48_16_ablations(1).zip`：用于判定 v48.16 各组件的真实效果；
- `ocrap_v48_16_safe_paired_calibration(1).zip`：用于判定 Safe 非劣；
- `reports(1).zip`：用于分析三个 regime 的固定数据性质。

遵循你的要求：**论文与代码不一致时，以代码为准；本轮不重建数据集；test/stress 不用于继续调参。**

---

## 2. 论文 idea 与当前代码的主线关系

论文的核心问题不是普通碰撞避免，而是：在观测混叠、隐藏 future roots 和动作后果不完全可辨的条件下，判断一个恢复动作是否对所有观测一致的潜在根都足够安全。当前代码已经把这一 idea 工程化为四层：

1. **Recovery proposal**：从恢复动作集合中提出 top-k 候选；
2. **Evidence**：估计候选相对 nominal 的 harmful/dead-zone/beneficial 证据；
3. **Selective certificate**：在 scene-disjoint certificate 上学习阈值，并用 precision LCB、harmful UCB、positive recall、teacher advantage 等约束 Natural gate；
4. **Closed loop**：只有 gate 授权后才允许 Near/Contact test/stress，Safe 始终 nominal lock。

这个实现方向与论文“可恢复性不是单一 oracle future 上的最优性，而是观测一致条件下可部署的共享动作证据”是相符的。现在真正需要投稿强化的，不是再增加一个普通策略网络，而是把 **proposal recall 与 selective evidence certification 的分解**做成清晰、可验证、可消融的算法贡献。

---

## 3. Natural gate 的有效判定

### 3.1 不是工程空数据

本次 8 个任务均具备：

- `TASK_COMPLETE.json`；
- 独立 `certificate_pool` 的 Near/Contact risk JSON；
- 非零 group、scene、fit、verify；
- `GATE_FAILED.json`，对应有效 certificate 的返回码 20。

共同 verify 支持量：

- Near：163 groups，6 个正机会；
- Contact：380 groups，14 个正机会。

所有任务在两个 regime 上均 `num_selected=0`、`positive_recall=0`。因此当前结论是：

> **v48.16 Natural gate 已被有效评估，并被算法性拒绝。**

### 3.2 八组结果

| 组件 | Variant | Near benefit AUC | Near harm AUC | Contact benefit AUC | Contact harm AUC | Near verify selected | Contact verify selected | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| A_source | balanced | 0.818 | 0.515 | 0.579 | 0.404 | 0 | 0 | 未通过 |
| A_source | precision | 0.755 | 0.525 | 0.484 | 0.474 | 0 | 0 | 未通过 |
| B_old_tiny | balanced | 0.818 | 0.515 | 0.579 | 0.404 | 0 | 0 | 未通过 |
| B_old_tiny | precision | 0.756 | 0.525 | 0.484 | 0.474 | 0 | 0 | 未通过 |
| C_balanced_margin | balanced | 0.818 | 0.515 | 0.579 | 0.404 | 0 | 0 | 未通过 |
| C_balanced_margin | precision | 0.756 | 0.525 | 0.484 | 0.474 | 0 | 0 | 未通过 |
| D_full_anchor | balanced | 0.818 | 0.515 | 0.579 | 0.404 | 0 | 0 | 未通过 |
| D_full_anchor | precision | 0.756 | 0.525 | 0.484 | 0.474 | 0 | 0 | 未通过 |

跨 A/B/C/D 的变化范围极小：

- Balanced Near benefit AUC range = 0；harm AUC range = 8.5e-5；
- Balanced Contact benefit AUC range = 6.9e-5；harm AUC range = 2.59e-4；
- Precision 的最大变化也仅 2.35e-4。

这证明 v48.16 的训练虽然执行了，但没有产生足以改变 certificate decision boundary 的学习效果。

---

## 4. 根本原因：proposal 有机会，Evidence 不会选

### 4.1 Proposal 被证明有效

A_source balanced 的独立 certificate 显示：

| 指标 | Near | Contact |
|---|---:|---:|
| proposal top-k | 3 | 3 |
| 正机会组 oracle-best top-k 命中 | 1.000 | 1.000 |
| 正机会组 any-positive top-k 命中 | 1.000 | 1.000 |
| 正机会组 top-1 accuracy | 0.643 | 0.594 |
| 严格 top-1 accuracy | 0.500 | 0.563 |
| rank-margin correctness AUC | 0.911 | 0.567 |

所以当前不应该再次修改 proposal。尤其 Near 的 rank margin 已经有很强的排序诊断价值。重新训练 proposal 会破坏当前最重要的可归因性：无法区分改进来自候选召回还是 Evidence 选择。

### 4.2 Evidence 的 false-safe 与 false-switch 很严重

A_source balanced：

| 指标 | Near | Contact |
|---|---:|---:|
| proposal Evidence benefit AUC | 0.771 | 0.536 |
| proposal Evidence harm AUC | 0.440 | 0.432 |
| 非正组 false-switch rate | 0.906 | 0.932 |
| harmful-ranked switch rate | 0.345 | 0.357 |

Near 的 benefit 排序尚有信息，但 harm tail 不可靠；Contact 的 benefit/harm 两端都接近随机。Natural gate 选择 0 并不是 gate 本身“过于保守”，而是 gate 在正确阻止一个无法证明安全的 Evidence 模型进入闭环。

### 4.3 fit→verify 反转说明跨场景证据不稳定

Near 最接近约束的 fit 规则：12 个 selected、0 positive、4 harmful、teacher advantage -0.195。映射到 verify 后变成 10 selected、3 positive、3 harmful、recall 0.5，但 precision LCB90 仅 0.127、harmful UCB90 为 0.558、平均 teacher advantage 仍为 -0.049。

Contact 更严重：fit 为 20 selected、1 positive、2 harmful；同一规则到 verify 后为 24 selected、0 positive、14 harmful，harmful rate 0.583、UCB90 0.732、teacher advantage -0.338。

这说明阈值不是主要问题。即使放松阈值，也只是把 coverage 从 0 变成大量 harmful selection。Contact 的 Evidence 在 scene-disjoint fold 上发生了明显的语义反转。

---

## 5. v48.16 为什么没有起作用

### 5.1 校准器条件表达不足

旧 tiny/ANCHOR 校准器只看到四个标量：source Evidence center/width 与两个 rank margin。两个候选只要这四个统计量相似，即使几何关系、动作类型、相对 nominal 表示和 regime 风险上下文不同，校准器也只能给出近似相同的修正。

这会直接导致：

- Near 中无法区分“小收益可恢复”与“轻微副作用”；
- Contact 中无法区分“稳定停车/脱离二次接触”与“同样看似保守但会 recontact 的动作”；
- source 分数有偏时，低容量标量 residual 只能整体推高或压低，最终趋向 always-abstain。

### 5.2 class-balanced loss 的作用域错误

v48.16 宣称 class-balanced，但代码在单个 scene-time proposal group 内计算 harmful/dead/beneficial 类均值。大多数 group 只有一个 teacher 类，因此“类平衡”退化成该 group 的普通 NLL；随后全 batch 聚合时，大量 dead-zone group 仍然占据主导。

这解释了为什么 C/D 与 B 几乎完全相同：新增 loss 在真实 batch 结构中经常没有提供额外梯度。

### 5.3 采样没有保证 evidence strata 共存

旧 weighted replacement 只提高稀有正组被抽中的概率，但没有保证每个 minibatch 同时出现：

- beneficial group；
- harmful-only group；
- dead/mixed group；
- Near 与 Contact 两个 regime。

因此 bipolar margins、class balance 和 hard-mining 经常在某个 batch 中缺少对应类别，无法真正学习两个尾部。

### 5.4 checkpoint 仍可选择 always-abstain

v48.16 的 best epoch 经常为 epoch 1。虽然有 missed-opportunity penalty，但没有最低 positive recall 约束。当模型通过降低 admission 迅速减少 harmful/false intervention 时，综合风险仍可能偏好低 coverage 模型。

---

## 6. 已证明有效、应保留深化的算法

### 6.1 Scene-disjoint adaptation/dev/certificate 合同

这是投稿可信度的核心。它已经成功暴露了 Contact 的 fit→verify 反转，必须完整保留。Natural gate 的 0/20/30 返回码、只有 0 才授权 stress，也是正确的工程和实验伦理设计。

### 6.2 Top-k recovery proposal

Top-3 正机会命中率接近 100%，说明“恢复动作候选存在且能被提出”。论文可以将其定位为 high-recall proposal stage，并将主要 novelty 放在 observation-consistent selective Evidence 上。

### 6.3 Source identity 与 bounded residual

零初始化使适配开始时等于 source，低样本目标域不能无证据重写已有能力。这个原则正确，但残差参数化需要从 center/width 升级为三类概率空间的条件残差。

### 6.4 Safe nominal lock 与 paired protocol

Safe 的 candidate 与 scalar baseline 在 120 个 paired scenes 上完全一致，说明 nominal lock、scene matching、并行 rollout 与结果配对已基本正确。它应作为“不牺牲正常驾驶”的独立 safety contract，而不是用来证明 recovery gain。

---

## 7. 已证明无效、不要重复的方向

1. **v48.14 全量 Evidence adapter**：约 39 万参数对几十个正机会 group，明显过参数化并破坏 source benefit；不再重训。
2. **v48.15/v48.16 center-width tiny calibrator**：主要形成 abstain，且本轮 A/B/C/D 指标几乎不动；不再仅调 hidden/scale 重复运行。
3. **当前形式的 hard-harm/hard-benefit**：此前消融未改变 best epoch 与 gate；本轮关闭。
4. **same-group pairwise**：proposal group 未稳定同时包含正、dead、harm，pair 支持过少；本轮关闭。
5. **通过放松 Natural gate 换 coverage**：Contact near-miss 已证明会引入 58.3% harmful selection；不可接受。
6. **同时改 proposal 与 Evidence**：当前 proposal 已有效，同时修改会破坏因果归因；本轮冻结。
7. **重建三个 regime 数据集**：你已明确暂不重建，本轮通过采样/损失/选择逻辑消化稀疏性。

---

## 8. 三个 regime 的独立分析

### 8.1 Safe

**已有证据**：120 paired scenes，collision scene rate 0.00833 vs 0.00833，offroad 0 vs 0，NUP 1 vs 1，intervention 0 vs 0，jerk/yaw-rate 完全相同。

**含义**：nominal lock 有效，当前 recovery 模块没有污染 Safe。

**缺陷**：route progression 未输出；旧 analyzer 对 jerk/yaw-rate 没有 margin；candidate 与 baseline 完全同 fingerprint，因此这只是非退化检查。

**修复**：代码已加入固定路线上的有符号 arc-length progression；有 Waymax SDC route 时直接使用，否则将现有 logged-future route proxy 一次性变换到全局坐标，并明确输出来源。jerk/yaw-rate 默认使用 5% relative non-inferiority margin。Safe paper-ready flag 要求所有指标都存在并通过。

### 8.2 Near-contact

**数据性质**：calibration Near 的 artifact fraction 约 0.240、negative deployable 约 0.448、oracle recoverable 约 0.792、R_dep 均值约 -0.509、oracle gap 均值约 0.406、alias incompatibility 约 0.196。它有明显可恢复机会，也有较强观测不兼容。

**算法缺陷**：正机会 support 很少；benefit 有一定可分性，但 harm tail 弱。模型容易把 coverage 压到 0，或者在放宽时混入 harmful/dead。

**目标**：先在独立 verify 上建立非零 selected、正 teacher advantage、可接受 precision LCB/harm UCB，再进入 closed loop 检验 clearance、TTC、DRS、PCD、FRA 与 ODG。不要直接以 recall 0.5 的 near-miss 为成功，因为其 harmful UCB 仍为 0.558。

### 8.3 Contact

**数据性质**：calibration Contact 的 artifact fraction 约 0.212、negative deployable 约 0.417、oracle recoverable 约 0.795、R_dep 均值约 -0.351、oracle gap 均值约 0.288、alias incompatibility 约 0.142。

**算法缺陷**：proposal 正机会召回好，但 Evidence benefit/harm 都弱，且 fit→verify 反转严重。Contact 中动作后果高度依赖相对几何、碰撞后速度/朝向、宏动作类型和 route rejoin 条件；四标量校准器不具备这些条件表达。

**目标**：优先降低 false-safe harmful，并保证 positive recall 不归零。通过 gate 后再验证 secondary overlap、recontact、stable stop、post-contact clearance、route rejoin，而不是只看离线分类 AUC。

---

## 9. 统一算法优化：v48.17 BRIDGE

**BRIDGE = Batch-balanced Regime-conditioned Identity-preserving Discriminative Group Evidence**

### 9.1 三概率单纯形残差

旧方式只修正 center/width。新方式在 source 的 harmful/dead/beneficial log probability 上加入有界三维 residual，再 softmax：

- residual=0 时严格等于 source；
- 三类概率始终归一化且非负；
- beneficial 和 harmful 两个 tail 可以独立修正；
- 不需要改 proposal/source encoder。

### 9.2 候选相对 nominal 的冻结上下文

校准器输入扩展为：source Evidence 统计、proposal margins、冻结的 candidate-vs-nominal relative representation。上下文默认 detach，只有 tiny calibrator 更新。这样 Contact 的几何/动作条件差异能够进入校准，但不会把小样本梯度反传污染 source。

### 9.3 Batch/regime 级类别平衡

将候选收集到整个 minibatch，先在 Near/Contact 内分别计算 harmful/dead/beneficial 均值，再平均当前存在的类别和 regime。这样 dead-zone 数量再多，也不能通过“全部 abstain”支配损失。

### 9.4 Evidence 分层 scene-time batch

基于 exact teacher PCD，把 group 分为 beneficial、harmful-only、dead/mixed，默认比例 0.35/0.35/0.30，从各 strata 内 replacement sampling 并交错组装。scene-time group 完整性不变，避免 option 泄漏。

### 9.5 Recall-constrained checkpoint

adaptation dev 的 checkpoint metric 加入最低 positive recall 0.25 与 shortfall penalty 4.0。它不是放宽部署 gate，而是防止训练阶段把 always-abstain 选为最佳模型。最终 certificate 约束完全不变。

### 9.6 为什么这一设计更贴合 CCF-A 投稿目标

潜在 novelty 不在“又一个 MLP 校准器”，而在以下组合：

- high-recall recovery proposal 与 selective Evidence certificate 的明确分解；
- observation-consistent source Evidence 上的 identity-preserving conditional simplex adaptation；
- 面向稀疏 counterfactual recovery opportunity 的 batch/regime 支持控制；
- 独立 scene certificate 与 closed-loop authorization 合同。

只有在 v48.17 通过 gate、闭环改善、Safe 非劣和组件消融共同成立后，才可将其包装成 CCF-A 级贡献。目前只能说设计具有清晰 novelty 路径，不能提前承诺录用或结果一定改善。

---

## 10. 已完成的工程修复

1. 新增 calibrator mode/context/detach 的 config、checkpoint、inference 完整兼容；旧 checkpoint 仍可加载。
2. 新增 tri-simplex identity、概率边界、参数量测试。
3. class-balanced Evidence 改到 batch/regime 级；旧逻辑默认保持兼容。
4. sampler 支持 exact teacher strata、比例配置、stratum 统计与缺失索引 hard fail。
5. checkpoint metric 新增 minimum recall/shortfall penalty。
6. 修复 gate checker 的字段名：`precision_wilson_lcb90`、`teacher_advantage_mean`。
7. 重写消融 summarizer，修复 dedicated path、proposal 字段和版本元数据。
8. Safe scene 输出新增 route progression 与 route source；analyzer 新增 jerk/yaw-rate margin，paper-ready 不再忽略缺失指标。
9. 新增 v48.17 主控制器、组件消融控制器、stress 授权 wrapper，严格区分返回码 0/20/30。
10. 两张 A30 的主实验固定为 Balanced→GPU0、Precision→GPU1；组件消融为 GPU0 跑 A/C、GPU1 跑 B，减少重复加载与空闲时间。
11. 清理测试入口要求，统一使用 `PYTHONPATH=src`。

---

## 11. 数据集固定条件下的统一策略

固定数据集并不意味着三个 regime 完全共享同一决策边界。BRIDGE 采用“共享原则、条件化校正”：

- Safe：不训练 recovery，nominal lock；
- Near/Contact：共享 source proposal 和 Evidence 表示；
- 校准损失按 regime 分开平衡，避免 Contact 数量淹没 Near；
- context 允许同一 residual 网络根据 regime/相对状态学习不同修正；
- certificate 仍分别给 Near/Contact gate，不允许一个 regime 的收益掩盖另一个 regime 的 harm。

这比训练三个完全独立模型更节省样本，也比强制一个统一阈值更符合两个危险 regime 的不同因果结构。

---

## 12. 下一轮判定标准

### 主实验返回码

- `RC=0`：两类 certificate 有效且通过；才运行 Near/Contact stress closed loop。
- `RC=20`：工程有效但算法仍失败；禁止 test/stress，只跑组件消融和 val-only diagnostic。
- `RC=30`：工程/数据/checkpoint/controller 失败；不能做算法归因。

### v48.17 离线优先观察

1. 与 v48.16 D_full_anchor 相比，candidate benefit/harm AUC 是否出现实质变化，而非 1e-4 噪声；
2. adaptation dev 的 Near/Contact positive recall 是否不再集中在 0；
3. verify 是否出现非零 selected；
4. teacher advantage 是否为正；
5. precision LCB90 与 harmful UCB90 是否同时改善；
6. Contact fit→verify harmful 反转是否显著收敛；
7. improvement 是否来自 `B_context_simplex` 或只有 `C_full_bridge` 才出现，以确认 novelty 组件。

### 通过 gate 后的论文实验

- Safe paired ≥100 scenes，包含 route progression、jerk、yaw-rate margins；
- Near/Contact closed-loop 的 recovery、安全、舒适度、路由恢复指标；
- 至少 3 个训练 seed 或 bootstrap 稳定性；
- v48.16 baseline + A/B/C 组件消融；
- 失败案例按观测混叠、宏动作、recontact、route-rejoin 分类；
- test 只在算法冻结后运行一次主协议。

---

## 13. 固定数据报告摘要

| 数据集 | samples | groups | scenes | artifact | negative deployable | oracle recoverable | mean R_dep | mean oracle gap | alias incompatibility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train_safe | 20,000 | 2,500 | 1,171 | 0.000 | 0.099 | 0.901 | 0.937 | 0.000 | 0.000 |
| val_safe | 2,328 | 291 | 132 | 0.000 | 0.073 | 0.927 | 0.974 | 0.000 | 0.000 |
| calibration_safe | 2,544 | 318 | 135 | 0.000 | 0.053 | 0.947 | 1.110 | 0.000 | 0.000 |
| test_safe | 3,216 | 402 | 175 | 0.000 | 0.069 | 0.931 | 0.988 | 0.000 | 0.000 |
| train_near_contact | 13,324 | 1,800 | 600 | 0.189 | 0.553 | 0.636 | -1.794 | 0.326 | 0.161 |
| val_near_contact | 3,445 | 433 | 176 | 0.246 | 0.504 | 0.742 | -0.801 | 0.414 | 0.204 |
| calibration_near_contact | 6,039 | 765 | 316 | 0.240 | 0.448 | 0.792 | -0.509 | 0.406 | 0.196 |
| test_near_contact | 4,723 | 595 | 250 | 0.244 | 0.488 | 0.756 | -0.690 | 0.422 | 0.209 |
| train_contact | 16,790 | 2,000 | 500 | 0.166 | 0.543 | 0.623 | -1.792 | 0.228 | 0.095 |
| val_contact | 6,477 | 723 | 211 | 0.219 | 0.461 | 0.757 | -0.561 | 0.302 | 0.135 |
| calibration_contact | 16,843 | 1,896 | 543 | 0.212 | 0.417 | 0.795 | -0.351 | 0.288 | 0.142 |
| test_contact | 6,687 | 747 | 209 | 0.218 | 0.444 | 0.774 | -0.572 | 0.298 | 0.141 |

这些统计支持两点：

1. Near/Contact 从 train 到 calibration/val/test 存在明显 shift，目标域小容量校正方向是合理的；
2. Safe 没有 oracle artifact，而 Near/Contact 同时存在较高 oracle recoverability 与负 deployability，正适合检验论文提出的 oracle recovery 与 deployable recovery gap。

---

## 14. 诚实边界

本轮已完成静态代码审计、结果包统计审计、代码修改和本地单元测试，但当前环境没有真实 WOMD/Waymax 数据路径和两张 A30，因而没有执行 v48.17 训练、certificate 或 closed loop。新版本理论上直接针对已定位的梯度失效与条件表达问题，但最终是否通过 Natural gate 必须以服务器返回码和独立 certificate 为准。
