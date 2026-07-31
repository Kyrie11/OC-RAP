# OC-RAP v48.22 结果审计与 v48.23 FRONTIER-BRIDGE 优化方案

## 一、结论摘要

v48.22 的 `RC=20` 仍然是一次有效的 Natural-gate 拒绝。主控制器记录：

- `pipeline_valid=true`；
- `gate_evaluated=true`；
- `test_roots_read=false`；
- Balanced、Precision 两个分支均完成训练；
- Near/Contact 的 fit、verify certificate 均非空、scene-disjoint 且支持可行；
- 没有发生参数上限、空证书池、旧 teacher index、单分支失败或返回码归类错误。

但本轮失败也不能归因成“gate 太严，模型其实已经足够好”。v48.22 的 closest-fit 规则在 Near 和 Contact 中只有约 5%–10% 的机会精度，与 gate 要求存在数量级差距；所有主实验和消融的 verify coverage 仍为 0。当前主要问题仍然是模型不能把 proposal 中已有的恢复机会稳定转换成高收益、非 harmful、可准入的 action。

Natural gate 的确非常严格：在最低选择量下，Near-fit 需要至少 8/10 个机会且 0 个 harmful；Near-verify 需要至少 5/8 且 0 harmful；Contact-fit 需要至少 11/16 且最多 1 harmful；Contact-verify 需要至少 6/10 且 0 harmful。它接近 oracle-quality selective policy，但数学上可满足。当前模型距离这些要求较远，因此不应在同一协议中事后降低 gate。

正确的下一步不是越过 gate 运行 held-out test/stress，而是增加两类不读取 test 的诊断：

1. **Proposal-constrained oracle gate audit**：判断冻结 top-k 内是否本来就存在足够多的安全机会，可以在理论上通过当前 gate；
2. **Adaptation-dev shadow closed loop**：只在 adaptation-dev 上验证离线 Evidence 是否转化为最小间距、TTC、二次接触、撞后自由空间等物理改善。该结果只能定位问题，不能作为论文主结果或调 held-out test 的依据。

## 二、v48.22 主实验结果

### 2.1 Natural-gate 与数据支持

训练目标极度稀疏：

| Regime | Deployable candidates | Raw-beneficial | Component-harmful | Safe-beneficial | Safe groups | Safe scenes |
|---|---:|---:|---:|---:|---:|---:|
| Near | 1,425 | 45（3.16%） | 769（53.96%） | 25（1.75%） | 11 | 7 |
| Contact | 4,086 | 138（3.38%） | 1,855（45.40%） | 106（2.59%） | 41 | 17 |

这意味着模型面对的是“约 2% 的安全收益正样本 + 约一半的风险样本”。单独报告全局 candidate AUC 很容易被大量 dead 或显然 harmful 候选主导，无法代表高收益安全前沿。

### 2.2 Balanced

Balanced 选择了 **epoch 0**，训练了 8 个 epoch 后仍回退到训练前 identity：

| 指标 | Near | Contact |
|---|---:|---:|
| Candidate benefit AUC | 0.847 | 0.551 |
| Candidate safe-benefit AUC | 0.846 | 0.566 |
| Candidate harm AUC | 0.500 | 0.500 |
| Learned top-k benefit AUC | 0.805 | 0.444 |
| Learned top-k safe-benefit AUC | 0.815 | 0.499 |
| Learned top-k harm AUC | 0.500 | 0.500 |
| High-opportunity conditional harm AUC | 0.500 | 0.500 |
| Learned top-k correlation | -0.014 | -0.153 |
| Learned harmful top-1 switch | 0.000 | 0.429 |
| Positive top-1 regret | 0.005 | 0.141 |
| Verify coverage | 0 | 0 |

Near 的 source/consensus benefit signal 很强，且 top-k 内的收益排序 regret 很低。但 harm 恒为 0.5，说明三头模型在零初始化时并没有表示“deadband 内低风险”，而是表示“50% harmful”。Admission 又额外减去 `softplus(0)=log(2)`，因此 identity 在结构上被推向 abstain。Balanced 选择 epoch 0，不是证明全 abstain 最优，而是暴露了风险和 admission 初始语义错误。

Contact 的收益排序仍不可用：top-k benefit AUC 0.444、相关性 -0.153、regret 0.141。冻结 proposal 有候选，但 selector 没有学会识别真正有用的撞后动作。

### 2.3 Precision

Precision 选择 epoch 14：

| 指标 | Near | Contact |
|---|---:|---:|
| Candidate benefit AUC | 0.692 | 0.548 |
| Candidate safe-benefit AUC | 0.678 | 0.570 |
| Candidate harm AUC | 0.657 | 0.638 |
| Learned top-k benefit AUC | 0.664 | 0.422 |
| Learned top-k safe-benefit AUC | 0.648 | 0.497 |
| Learned top-k harm AUC | 0.607 | 0.649 |
| High-opportunity conditional harm AUC | 0.489 | 0.558 |
| Learned top-k correlation | 0.037 | -0.003 |
| Non-positive false switch | 0.188 | 0.246 |
| Learned harmful top-1 switch | 0.527 | 0.441 |
| Positive top-1 regret | 0.097 | 0.106 |
| Verify coverage | 0 | 0 |

Precision 证明 component risk 可以学习到一部分全局风险结构，但安全前沿仍没有建立：Near 的 conditional harm AUC 甚至低于随机，Contact 只有 0.558。风险头主要学会了“明显 harmful vs. 大量 dead”，没有学会“高收益安全 action vs. 高收益 harmful action”。

最接近 gate 的规则仍很远：

- Near-fit：选择 10 个，仅 1 个正机会，precision 0.10、LCB 0.030、1 个 harmful、harmful UCB 0.282；
- Contact-fit：选择约 20 个，仅 1 个正机会，precision约 0.05，且存在 harmful selection。

所以这不是增加 threshold grid 就能解决的问题。

## 三、v48.22 消融实验的因果结论

### A — `two_head_safe_probability`

- 保留 Near benefit：Balanced/Precision top-k safe-benefit AUC 约 0.815/0.835；
- harm 恒为 0.5；
- Contact top-k benefit 仅约 0.37–0.44；
- coverage 全为 0。

A 没有公平隔离第三 admission head，因为其两头 score 为 `P_b(1-P_h)-0.5`。当 `P_h=0.5` 时，只有 `P_b` 接近 1 才能非负，因此它天然过度 abstain。

### B — `triad_candidate_only`

Precision 中：

- Near top-k benefit/safe AUC 约 0.716/0.712；
- Contact top-k benefit/safe AUC 约 0.420/0.495；
- Near/Contact harm AUC 约 0.593/0.636；
- coverage 仍为 0。

第三 admission hypothesis 和 candidate BCE 本身不足以学会组级准入。

### C — `triad_group_mil_aggregate`

Precision 中：

- Near benefit进一步降到约 0.593；
- Contact benefit约 0.417；
- harm 保持约 0.60–0.62；
- Near top-1 regret恶化到约 0.141；
- coverage仍为0。

Noisy-OR group MIL 没有带来机会准入，反而让收益排序退化。

### D — `full_covenant`

Precision 中：

- Near top-k benefit/safe AUC约0.634/0.625；
- Contact约0.430/0.516；
- harm约0.592/0.647；
- coverage仍为0。

Component heads 保留了部分风险识别，但三头 + noisy-OR + 当前 checkpoint objective 没有把风险信号转成安全动作选择。所有 Balanced 的 B/C/D 又都选择 epoch 0，进一步证明初始 harm/admission 语义是系统性问题。

## 四、gate 太严还是模型没学到

答案是：**两者同时存在，但现阶段主要问题仍是模型/目标定义，而不是 gate 单独过严。**

### 4.1 gate 确实严格

在当前有限机会数下，LCB/UCB 要求接近 oracle：

| 子集 | 最小选择 | 最少正机会 | 最多 harmful |
|---|---:|---:|---:|
| Near fit | 10 | 8 | 0 |
| Near verify | 8 | 5 | 0 |
| Contact fit | 16 | 11 | 1 |
| Contact verify | 10 | 6 | 0 |

这会导致只要有一两个 false-safe，certificate 就失败。

### 4.2 但当前模型离 gate 太远

Closest rule 的机会精度仅约 5%–10%，而不是 50%–80%；Contact benefit/correlation 接近随机；Near 的强 benefit 也没有稳定 harm veto。因此即使把 gate适度放宽，当前策略仍难以形成可信的 closed-loop policy。

### 4.3 是否需要先跑完整闭环

- **不能**在 gate failed 后直接运行 held-out test/stress 完整闭环。这样会把 test 变成开发集，破坏 scene-disjoint 和预注册主张。
- **应该**运行 adaptation-dev shadow closed loop。它不读取 certificate/test/stress，只用于判断：离线 AUC、准入分数是否改善真实物理轨迹。
- 最终 held-out closed loop 仍必须由 `RC=0` 后生成的 `NEXT_COMMANDS.txt` 授权。

### 4.4 如何判断 gate 本身是否成为主要瓶颈

v48.23 新增 proposal-constrained oracle audit：

- **Oracle audit 失败**：冻结 top-k、标签定义和 gate 样本支持的组合本身不可通过。继续训练 calibrator 没有意义，需要在新版本中预注册新的 proposal/标签/证书协议；
- **Oracle audit 通过、模型失败**：明确是表示、收益排序、风险前沿或 admission 学习问题；
- **Oracle audit 通过，dev shadow 物理指标明显改善，但主 gate 仍失败**：模型有实际收益，当前 finite-sample gate 可能过于保守。可以基于已记录证据设计全新的预注册协议或扩大 certificate 支持，但不能事后修改 v48.22/v48.23 gate；
- **Oracle audit 通过，dev shadow 也不改善**：当前 offline teacher/feature 与 closed-loop 物理目标不一致，需要继续改算法和监督，而不是改 gate。

## 五、Near-contact 应达到的目标

Near-contact 没有发生实际碰撞，因此目标不是“撞后恢复”，而是用最少干预提前扩大安全裕度，同时不破坏正常驾驶和任务进度。

### 5.1 论文/代码已有核心指标

开发目标：

- PCD ≥ 0.54；
- FRA ≤ 0.12；
- DRS ≥ 0.88；
- bounded NUP ≥ 0.995；
- intervention rate ≤ 0.02；
- intervention episode rate ≤ 0.012；
- maximum intervention run ≤ 1；
- paper-PCD selector miss ≤ 0.034（开发），≤ 0.025（投稿目标）；
- route progression、jerk、yaw-rate 不劣于 nominal；
- collision/overlap 保持 0 或 paired non-inferior。

### 5.2 新增物理指标

- **Minimum clearance**：全轨迹最小接触距离，paired improvement 目标至少 +0.10 m；
- **Minimum TTC**：最小 TTC，paired improvement 目标至少 +0.20 s；
- **Near-contact exposure duration**：低于安全距离阈值的累计时长；
- **Critical-TTC exposure duration**：TTC 低于危险阈值的累计时长；
- **Clearance deficit AUC**：相对 2 m 安全裕度的亏损积分；
- **TTC deficit AUC**：相对 3 s 安全裕度的亏损积分；
- 低干预频率、短干预 episode、无宏动作抖动；
- route progression、offroad、NUP、舒适性保持非劣。

Near 的成功应表现为：**不发生碰撞、不频繁接管，但危险暴露时间显著缩短，最小距离/TTC 提升，且行驶任务和舒适性不变差。**

## 六、Contact 应达到的目标

Contact 中的关键不是简单制动，而是降低二次碰撞、快速脱离障碍物接触区域并建立可用逃逸空间。

### 6.1 论文/代码已有核心指标

开发目标：

- PCD ≥ 0.52；
- FRA ≤ 0.16；
- DRS ≥ 0.84；
- bounded NUP ≥ 0.985；
- intervention rate ≤ 0.04；
- intervention episode rate ≤ 0.025；
- maximum intervention run ≤ 2；
- paired secondary-overlap scene-rate delta ≤ -0.02；
- new stable-stop event delta ≥ +0.02；
- route、offroad、jerk、yaw-rate 满足非劣或约束。

### 6.2 新增撞后物理指标

- overlap duration 和 longest overlap run；
- secondary/re-contact scene rate；
- first-contact 后的 minimum/mean/max clearance；
- **post-contact free-space AUC**：撞后净空随时间积分；
- **sustained escape event**：连续多步达到安全间距；
- **time to sustained escape**；
- stable stop rate；
- residual overlap/impact severity；
- 同时控制越界、航向角速度、jerk 和任务进度。

Contact 的成功应表现为：**减少二次接触和接触持续时间，尽快扩大撞后自由空间，形成持续脱离或稳定停车，并且不通过高频激烈动作换取表面 DRS。**

## 七、v48.22 工程问题及修复

### 7.1 Harm identity 错误

旧代码：component residual 为 0 时 harm logit=0，即 harm probability=0.5。

修复：v48.23 使用默认 `component_prior_logit=-2.0`，在 deadband 内表示低风险先验；风险头学习相对该先验的有界 residual。

### 7.2 Admission identity 错误

旧代码：

```text
admission = benefit - softplus(harm)
```

零初始化固定减去约 0.693。

修复：

```text
admission = benefit
            - [softplus(harm) - softplus(harm_prior)]
            + bounded admission residual
```

零 residual 时 admission 精确等于 transferred benefit，之后风险才能进行语义明确的 veto。

### 7.3 Noisy-OR 与执行策略不一致

旧模型将 top-k 中多个 action 当成独立 Bernoulli 机会，但 runtime 只能选择一个动作或 nominal。

修复：使用 nominal + frozen top-k 的 categorical softmax group policy。组级 loss、checkpoint 和部署都围绕同一个 one-action event。

### 7.4 Benefit 只有二分类，缺少连续排序

修复：新增 top-k continuous PCD listwise/KL loss，让 action 按真实收益幅度排序；top-k 外不产生梯度，避免无关候选污染。

### 7.5 高收益 safety frontier 没有直接监督

修复：在同组内显式要求 safe-beneficial admission 高于 beneficial-but-harmful action，直接优化 Natural gate 最关心的边界。

### 7.6 Checkpoint metric 目标错误

修复：`direct_frontier_selection_risk` 重点惩罚 high-opportunity harmful policy mass、false admission 和 worst-regime regret；global harm 只作为小权重 tie-break。

### 7.7 Contact 汇总错误

旧代码将 `secondary_overlap_event` 跨场景取最大值。现在改为 scene rate，并新增：

- `secondary_overlap_scene_rate`；
- `new_stable_stop_scene_rate`；
- `post_contact_escape_scene_rate`。

### 7.8 工件/返回码与兼容性

- Teacher index 继续绑定 manifest 和标签参数；
- 任何非 0/20 底层错误统一 `RC=30`；
- 两个主分支必须都完整；
- 旧 v48.18–v48.22 checkpoint 路径保留；
- test/stress 仍由授权文件保护。

## 八、v48.23 FRONTIER-BRIDGE

FRONTIER 的完整思想是：

> 在统一、无 regime 路由的模型中，冻结高召回 recovery proposal；用连续收益排序学习“哪个 action 更有价值”，用不可补偿分量风险学习“哪个 action 不能执行”，再用与实际 one-action 选择一致的 categorical admission 决定是否离开 nominal。

推理不输入 Safe/Near/Contact 标识。Safe 通过 nominal lock 和低风险下的 identity 保证不污染；Near/Contact 只作为 dev/certificate 的 worst-stratum 分析，不是策略路由条件。

核心组件：

1. Semantic low-risk component prior；
2. Identity-preserving centered admission；
3. Categorical nominal-vs-top-k group objective；
4. Continuous PCD listwise benefit ranking；
5. Safe-benefit vs harmful-benefit frontier contrast；
6. Exact component `max` veto；
7. Frontier-aware checkpoint metric；
8. Proposal oracle audit；
9. Dev-only physical shadow closed loop。

## 九、非重复消融设计

| 组 | 内容 | 主要回答的问题 |
|---|---|---|
| A_semantic_prior_categorical | 语义 risk prior + centered admission + categorical policy | v48.22 是否主要被工程初始语义/Noisy-OR 阻断 |
| B_add_benefit_listwise | A + continuous benefit listwise | 连续收益排序能否改善 correlation、Contact top-1 regret |
| C_add_frontier_contrast | A + safety-frontier contrast | 能否降低 high-benefit harmful switch |
| D_full_frontier | B + C + component veto | 两种能力能否在同一统一模型中共同存在并产生 coverage |

全部 8 个任务一次性启动，GPU0/GPU1 各 4 个任务。

## 十、运行顺序与结果解释

1. 先运行 v48.23 主实验；
2. `RC=0`：只运行授权 stress；
3. `RC=20`：先查看 proposal-constrained oracle audit；随后运行 dev shadow closed loop，再运行 8 个并行消融；
4. `RC=30`：只修复工件中标明的工程阶段，不能做算法结论。

消融判读：

- A 显著优于 v48.22：初始语义和 categorical objective 是关键；
- B > A：连续收益幅度监督有效；
- C > A：安全前沿监督有效；
- D 同时优于 B/C：收益排序与安全前沿可兼容；
- Oracle pass、D 离线改善但 dev shadow 不改善：teacher/feature 与物理目标错位；
- Oracle pass、dev shadow 改善但 gate 仍失败：有限样本 certificate 过于保守，需要未来新预注册协议或增加证书支持；
- Oracle fail：当前 proposal/标签/gate 合同不可通过，不再重复训练 calibrator。

## 十一、性能优化

- 新损失全部在已有 forward 输出上向量化计算，不新增 encoder 或重复 proposal；
- 主实验 Balanced/Precision 分别独占 GPU0/GPU1；默认 batch size 96、3 workers、prefetch 3、pinned persistent workers、bfloat16 AMP；
- 8 个消融同时开始，GPU0/GPU1 各 4 个；每任务 batch size 56、1 worker；
- OMP/MKL/OpenBLAS 线程限制为 2，减少 CPU 争抢；
- teacher index 复用且有合同校验，避免重复扫描和错误复用；
- dev shadow 中每个 variant 的 Near/Contact 在同一 GPU 顺序执行，Balanced/Precision 两卡并发，避免 VRAM 超订；
- closed-loop inference 使用轻量 sample view 和共享场景特征，减少历史/map/BEV 重复复制。

## 十二、本地验证

- `pytest`: 216 passed，5 warnings；
- Python `compileall`: passed；
- 全部 shell `bash -n`: passed；
- 缺失 protocol 故障注入：process、`PIPELINE_FAILED.json`、`V48_23_COMPLETE.json` 均归一化为 `RC=30`，`test_roots_read=false`；
- 新增测试覆盖 risk prior、admission identity、continuous listwise 梯度、frontier contrast 梯度、categorical policy、checkpoint risk、oracle/dev-shadow plumbing、8任务双卡分配。

当前环境没有真实 WOMD/Waymax 和 A30，因此本报告不声称 v48.23 已通过 Natural gate，也不伪造闭环结果。
