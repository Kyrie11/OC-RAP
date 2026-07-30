# OC-RAP v48.18 结果审计与 v48.19 FACET-BRIDGE 优化报告

日期：2026-07-30

## 1. 结论

`RC=20` 不能笼统解释成“工程错误”或“模型失败”。本次审计后的归因如下：

- **v48.17**：早期缺少 gate 文件/返回 `RC=30` 是 78,630 参数被 20,000 上限误杀的工程问题；在修复 guard、独立 certificate 非空且 `pipeline_valid=true` 后，用户报告的 `RC=20` 才是旧算法在旧协议下的真实拒绝。
- **v48.18**：是**协议层与算法层混合失败**。流水线确实完成、test 未读取、certificate 非空，但 Near-fit gate 在当前支持度下数学不可满足；同时 Contact 的 benefit/harm 排序接近随机，证明即使修正协议，原 DUET 也没有足够证据通过。
- **根本修复不是继续扫阈值**。需要同时修复：统计门槛的支持度与置信界定义、harm 标签的语义、跨 regime 的稀疏样本共享、返回码与数据/索引契约。

交付代码实现 v48.19 **FACET-BRIDGE**。它不重建用户的数据集，不改 frozen top-k proposal，不读取 test；把优化集中在可归因的 Evidence admission 与独立证书上。

## 2. 对论文的理解

论文主张将 recoverability 从碰撞后的补救动作提升为规划阶段的一等优化目标。核心问题不是“每个隐藏未来是否各自存在某个恢复动作”，而是：候选前缀执行后，针对观测上不可区分的潜在未来，车辆是否仍能从当时可见观测中选择一个共同可执行的恢复动作。

论文的方法链条为：

1. recovery-sufficient latent root generator：按恢复可供性而不是纯轨迹距离压缩未来；
2. post-prefix observation equivalence kernel：建模执行候选前缀后哪些 roots 对车辆不可区分；
3. affordance-conditioned recovery margin head：预测 stop、brake、escape、rejoin、post-contact stabilize、avoid secondary 等恢复模式的约束裕量；
4. OC-MERO：在观测兼容集合内先做共享恢复动作的 lower-tail 聚合，再对 roots 做 lower-tail 聚合；
5. CRISP：在 nominal recoverability 充足时锁定 nominal，否则只接纳经校准的恢复候选。

论文指标包括 FRA、ODG、DRS、NUP，以及 Contact 下的二次碰撞、稳定停车、横摆稳定、路线重入、干预率等。代码当前的 Safe/Near/Contact 三 regime 是该思想的工程化验证切片：Safe 验证 nominal preservation；Near 验证预碰撞恢复机会；Contact 验证碰撞后稳定与二次风险。

## 3. 数据集审计结论

用户明确暂不重建数据集，因此 v48.19 只增加标签/索引契约和支持度审计，不改变样本。

| Regime | 已有数据规模 |
|---|---|
| Safe | train 1171 scenes/20000 samples; val 132 scenes/2328 samples; test 175 scenes/3216 samples; calibration 135 scenes/2544 samples |
| Near | train 600 scenes/13324 samples; val 176 scenes/3445 samples; test 250 scenes/4723 samples; calibration 316 scenes/6039 samples |
| Contact | train 500 scenes/16790 samples; val 211 scenes/6477 samples; test 209 scenes/6687 samples; calibration 543 scenes/16843 samples |

关键分布事实：Near/Contact 的 `harm_proxy` 在 train 中非零，但在 val/test/calibration 中最大值为 0。继续把 harm 主要绑定到该 proxy 会造成明显 train-to-target contract shift。FACET 的 harm 因此使用 DRS、deployability gate、gap quality、hard violation 和 harm proxy 的分量式 nominal-relative veto，而不是依赖单一 proxy。

## 4. v48.18 主实验到底表明什么

主控制器记录 `certificate_exit_code=20`、`gate_evaluated=true`、`pipeline_valid=true`、`test_roots_read=false`。这排除了 OOM、训练未完成、certificate 为空和 test 泄漏，但不排除统计协议本身不可满足。

| Variant | Regime | Groups / Scenes | Fit / Verify opportunities | Candidate benefit AUC | Candidate harm AUC | Policy benefit AUC | Policy harm AUC | Verify selected |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Balanced | Near | 290 / 123 | 8 / 6 | 0.814 | 0.516 | 0.788 | 0.436 | 0 |
| Balanced | Contact | 764 / 215 | 18 / 14 | 0.580 | 0.404 | 0.509 | 0.422 | 0 |
| Precision | Near | 290 / 123 | 8 / 6 | 0.697 | 0.550 | 0.621 | 0.501 | 0 |
| Precision | Contact | 764 / 215 | 18 / 14 | 0.484 | 0.494 | 0.426 | 0.535 | 0 |

现象：

- Balanced 的 Near benefit AUC 仍强（0.814），说明 top-k proposal 内确实有可恢复候选，主要瓶颈仍是 admission evidence；
- Near harm AUC 仅 0.516，Contact Balanced harm AUC 甚至 0.404；
- Precision 虽把 Near harm AUC 推到 0.550，但 Near benefit 降到 0.697，Contact benefit/harm 仍约 0.5；
- 所有分支 verify selected 都为 0，即策略仍处于全 abstain；
- 这不是单纯“阈值太严格”：放松到能覆盖机会时，原排序会同时引入大量 harmful/false intervention 候选。

## 5. Natural gate 的协议缺陷

历史代码把字段命名为 `LCB90/UCB90`，却使用 `z=1.6448536`。该 z 对应 central two-sided 90% 区间，或 one-sided 95% 界；而 gate 实际声明的是方向性的 precision LCB 与 harm UCB。更关键的是 v48.18 Near fit 只有 8 个正机会，却要求至少选 12 个、precision LCB ≥ 0.50：

- 历史实现：oracle `8/12` 的 LCB = **0.43149**，不可满足；
- 即使改成 one-sided 90%，`8/12` LCB = **0.48181**，仍不可满足；
- v48.19 新协议：Near fit 采用 10，oracle `8/10` LCB = **0.60160**；Near verify 保留 8，`6/8` LCB = **0.52371**，零 harm UCB = **0.17033**。

因此，v48.18 的 `RC=20` 不能被当作纯算法失败。v48.19 会在读取候选分数前冻结 `GATE_SPEC.json`，并先计算 oracle optimistic feasibility；若门槛仍不可满足，返回 `RC=30`（协议/数据支持错误），而不是 `RC=20`。

**科研口径注意**：10/8 是审计后建立的全新、预注册协议，不能反向把 v48.18 改写成通过，也不能与旧 12/8 结果直接做同协议比较。

## 6. v48.18 消融实验表明的因果现象

表中 AUC 为 candidate benefit / harm；最后一列为 Near / Contact verify selected。

| Task | Best epoch | Near AUC | Contact AUC | Verify selected |
|---|---:|---:|---:|---:|
| A_dual_scalar_balanced | 1 | 0.817 / 0.515 | 0.579 / 0.404 | 0 / 0 |
| B_dual_tournament_balanced | 6 | 0.770 / 0.551 | 0.585 / 0.409 | 0 / 0 |
| C_dual_tournament_balanced_balanced | 1 | 0.816 / 0.517 | 0.580 / 0.404 | 0 / 0 |
| D_full_duet_balanced | 1 | 0.816 / 0.517 | 0.580 / 0.404 | 0 / 0 |
| A_dual_scalar_precision | 2 | 0.756 / 0.525 | 0.485 / 0.474 | 0 / 0 |
| B_dual_tournament_precision | 6 | 0.724 / 0.546 | 0.481 / 0.479 | 0 / 0 |
| C_dual_tournament_balanced_precision | 8 | 0.697 / 0.553 | 0.485 / 0.493 | 0 / 0 |
| D_full_duet_precision | 8 | 0.697 / 0.553 | 0.485 / 0.493 | 0 / 0 |

可保留的结论：

- **冻结 top-k proposal：有效且必须保留。** Near benefit 排序在不同结构下反复出现，说明 proposal 不是当前第一瓶颈。
- **context 条件化思想：方向正确，但 v48.18 的 tournament context 实现没有形成净增益。** B 对 harm AUC 有小幅方向性改善，却稳定损伤 benefit；Contact 仍不可分。
- **零初始化、有界 residual：有效的安全结构。** 它保证适配起点等于 source evidence，且不能无限重写源策略。
- **scene-disjoint adaptation/dev/certificate：有效且必须保留。** 它使工程失败、checkpoint 过拟合和真实 gate 拒绝可区分。
- **Safe nominal lock：有效。** Safe 的论文主张应是“不污染正常驾驶”，而不是强行在 Safe 中制造 recovery gain。

无效或当前证据不支持的设计：

- A 的 scalar-only dual tail：只保住 Near benefit，不能学习 harm；
- B 的当前 tournament context：harm 小幅上升不足以抵消 benefit 损失；
- C 的 strict balance：Balanced 基本退回 A，Precision 以更高误干预/风险换极小 harm AUC；
- D 的 `direct_duet_selection_risk`：与 C 的 best epoch、训练长度和证书指标逐项一致，未改变 checkpoint 选择；
- 继续 threshold sweep：无效，因为 Contact 的证据接近随机；
- “输出双尾”本身：不够，因为标签仍来自同一个 signed total delta，监督上仍互斥。

## 7. v48.19 FACET-BRIDGE

**FACET-BRIDGE = Factorized Advantage and Componentwise Evidence Transfer with shared cross-regime bridge。**

### 7.1 真正独立的 benefit/harm 语义

- Benefit：候选相对 nominal 的总 PCD advantage；
- Harm：任何安全相关分量相对 nominal 的退化超过容差，即 `max(ΔDRS, Δdeployability-gate, Δgap-quality, Δhard-violation, Δharm-proxy)`；
- 分量改进不能补偿另一安全分量的显著退化；
- 同一候选可以同时 benefit=1、harm=1，最终由 harm veto 拒绝。

这直接针对 Contact 中“总体恢复价值提高但仍有二次碰撞/失稳风险”的样本语义，是 v48.18 没有真正实现的部分。

### 7.2 三 regime 的共同结构

- Safe、Near、Contact 都以 nominal-relative component margins 作为共同安全语义；
- Safe 是该 admission 机制的 nominal boundary condition：无充分恢复机会时 residual 固定为 0，并由 nominal lock 保持原策略；
- Near/Contact 使用一个共享 calibrator 学共同规律，再用 `0.25` 的 bounded regime residual 表达预碰撞与碰撞后的差异；
- 共享校准器与两个 residual 总参数 2,298，适合当前稀疏 adaptation 数据，避免 78,630 参数 raw-context 的场景记忆。

### 7.3 checkpoint metric

新的 `direct_facet_selection_risk` 优先最小化 `min(recall_near, recall_contact)` 的缺口，并只在 harmful-switch/false-intervention 超预算时施加强惩罚。它仍只读取 adaptation dev，不读取 certificate/test。由于 v48.18 的 D 没有效果，v48.19 保留专门的 C-vs-D 消融来验证该 metric 是否真实改变 epoch，而不是直接宣称有效。

## 8. 工程修复清单

- 训练与 certificate 共用 `evidence_targets.py`，消除 harm 定义漂移；
- component margin 必须严格大于容差才是 harmful，边界相等不再得到 0.5 模糊标签；
- teacher index 绑定 train roots、manifest SHA256、alpha/beta/top-M、positive gain、macro ids 和全部 harm tolerances；契约变化自动重建；
- `GATE_SPEC.json` 绑定 Safe/Near/Contact manifest SHA256，禁止同一输出目录静默换数据或阈值；
- 训练前检查 Near/Contact 的 benefit/harm 正负样本支持；不支持时 `RC=30`；
- 主实验默认要求 Balanced 与 Precision 都训练成功，单分支失败不会继续伪装成可评估 run；
- 返回码统一：0=有效通过，20=协议支持且工件有效的算法拒绝，30=工程/数据/协议/工件错误；
- 四消融并发按两波运行，每波 4 个任务，两张 A30 各 2 个任务，并限制每任务 worker 与 CPU BLAS 线程；
- stress/test 只有 `NEXT_COMMANDS.txt` 存在时才允许执行。

## 9. 下一轮应如何解释结果

- `RC=0`：FACET 在新预注册协议下通过 Natural gate；随后运行 Safe paired 与 stress closed loop，并开始论文主表的 closed-loop 归因。
- `RC=20`：协议支持度已通过、数据/索引/工件有效，但算法仍失败。此时运行 v48.19 四组非重复消融，不要改 gate 阈值，不要读 test。
- `RC=30`：不是算法结论。按 `PIPELINE_FAILED.json` 的 stage 排查 protocol audit、teacher index、target support、adaptation 或 certificate artifact。

v48.19 消融：

1. A_component_veto_separate：只验证 factorized harm；
2. B_shared_component_veto：验证 cross-regime partial pooling；
3. C_shared_only_no_regime_residual：验证 regime residual 是否必要；
4. D_full_facet：验证 checkpoint metric 是否真的改变 checkpoint 与证书结果。

## 10. 面向 CCF-A 的论文建议

论文最强 novelty 仍应是 observation-consistent deployable recoverability，而不是把 FACET 单独包装成一个普通校准 MLP。更有说服力的叙事是：

- OC-MERO/CRISP 定义“什么是可部署恢复”；
- FACET-BRIDGE 解决稀疏 Near/Contact 数据下“如何可靠地把该概念转成选择性 admission evidence”；
- Safe nominal lock 证明恢复机制不会污染正常驾驶；
- Near/Contact 联合证书证明同一共享机制跨阶段有效；
- oracle artifact、benefit-harm overlap、全 abstain 与 false-safe 的案例分析解释为什么传统 scalar risk/simplex/threshold filter 失败。

当前 TeX 还存在投稿前必须处理的问题：`\usepackage[hidelinks]` 缺少包名（应为 `\usepackage[hidelinks]{hyperref}`）、实验主表与消融表仍为占位符、oracle artifact 图注仍写 placeholder。它们不影响本轮算法实验，但在形成最终结果后必须清理。

## 11. 本地验证与限制

- pytest：185 passed，5 warnings；
- Python compileall：通过；
- 所有 `scripts/*.sh` 的 `bash -n`：通过；
- 当前交付环境没有用户服务器上的真实 WOMD、Waymax 和两张 A30，因此没有运行或伪造 v48.19 Natural gate/closed-loop 结果。
