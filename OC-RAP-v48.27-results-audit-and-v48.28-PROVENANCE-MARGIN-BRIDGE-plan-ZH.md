# OC-RAP v48.27 结果审计与 v48.28 PROVENANCE-MARGIN-BRIDGE 方案

## 1. 最终结论

v48.27 的主实验 `RC=20` 是有效 Natural-gate 拒绝：训练、checkpoint、certificate 数据读取和 gate 评估均完成，未读取 held-out test/stress。它不是工程 pipeline failure。

但本轮存在两条会妨碍算法归因的独立问题：

1. dev-shadow 使用了错误的 WOMD 数据源，导致 16 个 adaptation-dev target 在完整扫描 43,479 个 raw scenario 后仍然匹配 0 个；
2. factor stage 的 checkpoint 指标与新训练的 component-harm heads 无关，Balanced 和 Precision 都选择了 epoch 0，五个风险 head 实际没有完成学习。

此外，风险 head 的参数范围存在表示上限：`prior=-2`、`scale=2` 时，candidate component logit 最大只能达到 0，即单因子 `p(harm)` 最大为 0.5，不能拟合强 harmful teacher target。

因此，v48.27 的 RC=20 主要表现为 development-rule fitting failure，但其根因同时包含 checkpoint 工程合同和风险表示能力缺陷。

---

## 2. dev-shadow 失败的进一步根因

### 2.1 日志事实

Balanced 和 Precision 都出现相同结果：

- 成功加载 16 个 bucket targets；
- `raw_max_scenarios=None`，说明已经完整扫描，不再是旧版 900-scene 上限；
- 完整扫描 43,479 个 raw scenarios；
- 匹配 target 数为 0；
- fail-fast 正常触发。

所以本轮已经排除：扫描上限过小、未启用 fail-fast、空指标被误写为正常 0。

### 2.2 真正的数据 split 错配

数据集构建脚本 `build_v48_calibration_regimes.sh` 明确使用：

```text
.../validation/validation_tfexample.tfrecord@150
```

而 v48.27 的 `run_ocrap_v48_trac_sr.sh` 在 dev-shadow 中默认使用：

```text
.../validation_interactive/validation_interactive_tfexample.tfrecord@150
```

这两个路径对应不同 TFRecord population。adaptation-dev target 来自 standard validation，却在 validation_interactive 中搜索，因此即使完整扫描也无法匹配。

### 2.3 target ID 本身也不稳定

旧 Waymax loader 没有保留 WOMD 的官方 `scenario/id`。本地代码退化为：

```text
SHA1(object_ids + first timestamps) + __wx########
```

该 ID 依赖：

- 使用的数据 split；
- Waymax 的 `max_num_objects` 和对象截断；
- 对象排序及 dataloader 配置；
- shard 枚举顺序；
- 本地 source index。

所以它只能作为旧数据迁移键，不能作为长期稳定 scene identity。

### 2.4 v48.28 修复

- dev-shadow 默认改为 standard validation；
- 自定义 Waymax loader 显式保留官方 `scenario/id`；
- NPZ/manifest 保存 official ID、legacy hash、source index、source role、source pattern、`max_num_objects`；
- official ID 为主键；
- 老数据仅在相同 source role 下允许 source-index fallback；
- shadow 开始前生成 `SHADOW_PROVENANCE_AUDIT.json`；
- split 不一致直接 fail-closed；
- 提供 repair-only 脚本重跑 v48.27 checkpoint。

现有 v48.27 shadow 没有产生任何有效 closed-loop scene，因此其中所有物理指标均不可用于判断算法是否改善。

---

## 3. RC=20 与 gate failed 的真实层次

### 3.1 proposal/certificate 数学支持不是主瓶颈

完整 certificate 的 top-3 oracle：

| Regime | Certificate groups | Safe-positive groups | Oracle precision LCB | Feasible |
|---|---:|---:|---:|---|
| Near | 290 | 9 | 0.846 | 是 |
| Contact | 764 | 20 | 0.924 | 是 |

adaptation-dev 中也存在足够 oracle safe opportunities。因此可以排除：

- top-3 proposal 根本不包含安全机会；
- certificate 定义数学上无法满足；
- 必须删除 certificate 才能继续。

### 3.2 真实失败是 development-rule fitting

Balanced/Precision、Near/Contact 全部为：

```text
rejection_kind = development_rule_fit_rejection
```

没有任何 opportunity/harm/score/rank-margin 联合阈值能同时满足：

- 最小选择数；
- safe-positive precision LCB；
- harmful group/selected UCB；
- positive recall；
- macro concentration；
- selected teacher advantage。

因此 certificate 没有“导致”失败；selector 在 adaptation-dev 阶段就没有形成可冻结的安全规则。

### 3.3 gate 要求是否合理

当前 gate 较严格，但 oracle 已证明它在 Near 和 Contact 上可行，所以不能把失败归因于 gate 本身不合理。它同时约束安全性、统计置信度、覆盖和动作多样性，这符合 selective safety policy 的目标。

需要改进的是诊断，而不是事后降低阈值：

- 明确区分 proposal infeasibility；
- development-rule fitting failure；
- certificate generalization failure；
- macro constraint failure；
- finite-sample LCB/UCB failure。

v48.28 新增 `GATE_FAILURE_DECOMPOSITION.json`，把这些层次显式分离。

---

## 4. v48.27 中最关键的学习工程错误

### 4.1 factor checkpoint 选择了 epoch 0

Balanced 和 Precision 的 factor-stage `best_epoch` 都是 0。

训练轨迹显示 `loss_direct_recovery_value` 持续下降：

- Balanced：约 5.15 → 3.99；
- Precision：约 10.73 → 8.52。

但旧 `direct_factor_selection_risk` 在所有 epoch 完全不变。它使用 rank head 的 teacher harmful-top1 与 preference risk，并不依赖新初始化的 component-harm heads。

结果是：

1. epoch 0 被选中；
2. stage 2 冻结 epoch-0 factor heads；
3. certificate 中全部 Evidence harm AUC 恰好为 0.5；
4. 所谓“两阶段训练”没有真正完成第一阶段。

v48.28 使用包含实际监督 factor loss 的 `direct_factor_supervised_risk`，并禁止 factor/admission 两阶段选择 initial checkpoint。

### 4.2 component harmful 表示范围不足

v48.27 的 component logit：

```text
component_logit = -2 + 2 × tanh(raw)
```

范围是：

```text
[-4, 0]
```

所以最大 harmful probability 为 0.5。teacher 在分量退化超过 veto tolerance 后需要大于 0.5 的 harmful target，模型无法表达。

v48.28 改为：

```text
component_logit = -2 + 6 × tanh(raw)
```

范围约为：

```text
[-8, 4]
```

既保留低风险先验，又能表达强 harmful evidence。

---

## 5. Near-contact 的正向信号与缺陷

### 5.1 主模型结果

Balanced Near：

- candidate positive AUC：0.847；
- Evidence safe-positive AUC：0.815；
- Evidence harm AUC：0.500；
- certificate selected：5；
- safe-positive selected：3；
- empirical precision：0.60；
- precision LCB：0.330；
- harmful selected：1；
- recall：0.333；
- selected teacher advantage mean：+0.251。

这说明 Near 中存在真实正向区域：raw-benefit 能转成一部分正收益选择。但它还不能叫 safe admission，因为统计 LCB 未达到 gate，仍选择了 harmful action，支持规模也太小。

Precision Near：

- candidate positive AUC：0.817；
- Evidence safe-positive AUC：0.815；
- Evidence harm AUC：0.500；
- selected：7；
- safe-positive selected：0；
- harmful selected：2；
- mean advantage：-0.171。

Precision 保留了候选收益可分性，但 admission/risk head 没有把它转成安全 action。

### 5.2 应保留的设计

- frozen top-3 proposal；
- raw-benefit 与 final safe-admission 语义分离；
- factor→admission 两阶段结构；
- bounded、identity-preserving admission；
- nominal + top-k categorical one-action objective；
- legacy Noisy-OR 关闭；
- regression-only safe utility。

C 组 two-stage regression 是唯一在 Balanced Near 中得到明显正向 certificate 区域的设计。

### 5.3 损害模型或未经结果支持的设计

- 五因子 joint training：Balanced Near harmful selection 明显增加，mean advantage 为负；
- 当前 listwise/frontier：D 与 C 的 Balanced 结果完全相同，Precision 仅增加选择却没有增加安全收益；
- epoch-0 factor checkpoint；
- component scale=2；
- top-8 扩张；
- unbounded admission；
- 同一阶段同时施加稀疏 admission 和 dense factor 梯度。

---

## 6. Contact 的主要缺陷

Balanced Contact：

- Evidence safe-positive AUC：0.499；
- correlation：-0.159；
- selected：15；
- safe-positive：0；
- harmful：7；
- mean advantage：-0.249。

Precision Contact：

- Evidence safe-positive AUC：0.477；
- correlation：-0.118；
- selected：19；
- safe-positive：1；
- harmful：7；
- recall：0.05；
- mean advantage：-0.139。

Contact 尚未证明 safe admission。其主要问题不是阈值太保守，而是 action-level benefit/risk ordering 本身接近随机或负相关。

当前必须先验证以下修复是否使五个风险 factor 真正学会：

- post-contact hard-rule；
- harm proxy；
- deployability；
- DRS；
- oracle-to-deployable gap。

之后再根据有效 shadow 判断离线 teacher 是否缺少时间物理过程。若修复后离线 ranking 提升、但 shadow 中 secondary contact、overlap、free-space、escape 和 stable stop 仍不改善，下一步才应预注册 candidate-level temporal physical teacher/auxiliary head，而不是继续改变 certificate 阈值。

---

## 7. v48.27 消融速度分析

八个任务中：

- median training：约 965 秒；
- median post-training certificate：约 743 秒；
- training 和 certificate 都是主要耗时；
- 原脚本最大并发只有 2，每个 wave 等两任务完成后再启动下一 wave。

单任务显存约 1 GB，而每张 A30 有 24 GB，因此 GPU 显存不是限制。v48.28 改为：

- 8 个任务一次性启动；
- GPU0 同时运行 4 个 Balanced 任务；
- GPU1 同时运行 4 个 Precision 任务；
- 最大并发 8；
- 每任务 `NUM_WORKERS=1`；
- OMP/MKL/OpenBLAS 每任务 1 线程。

这能显著缩短总 wall time，但速度上限可能转移到 CPU、TFRecord I/O 和 certificate 扫描。若磁盘争用严重，可将 `TASKS_PER_GPU` 降到 2；默认按用户机器能力使用 4。

---

## 8. v48.28 PROVENANCE-MARGIN-BRIDGE

### 8.1 主算法

- top-3 frozen proposal；
- five-factor wide-range harm representation；
- stage 1：raw-benefit + five harm factors；
- stage 2：冻结 factors，只训练 bounded admission；
- deployment-exact safe-utility regression；
- categorical one-action；
- listwise/frontier 默认关闭，只在 D 消融中验证。

### 8.2 四组消融

1. `A_three_factor_wide_range`：验证 3 因子是否足够；
2. `B_five_factor_old_range`：验证旧 scale=2 上限；
3. `C_five_factor_wide_range_regression`：v48.28 主设计；
4. `D_add_listwise_frontier`：验证 listwise/frontier 是否在 factor 修复后才有效。

### 8.3 关键 fail-closed 工件

- `SHADOW_PROVENANCE_AUDIT.json`；
- `FACTOR_TRANSFER_INTEGRITY.json`；
- `MODEL_INFERENCE_CONTRACT.json`；
- `GATE_FAILURE_DECOMPOSITION.json`；
- `DEV_SHADOW_COMPLETE.json`；
- `ABLATIONS_STATUS.json`。

---

## 9. 下一轮判读顺序

1. 先 repair-only 重跑 v48.27 shadow，确认 standard validation + legacy source-index 能匹配 target；
2. 查看 factor-stage `best_epoch`，必须大于 0；
3. 查看 `FACTOR_TRANSFER_INTEGRITY.json`；
4. 检查 Evidence harm AUC 是否脱离 0.5；
5. 检查 Near 的 5-selected/3-positive 正向区域是否扩大且 harmful 下降；
6. 检查 Contact correlation、safe-positive AUC、regret 和 harmful switch；
7. 再看 development-rule 是否满足；
8. 只有 dev rule 有效后，才分析 certificate generalization；
9. RC=20 时运行有效 dev-shadow，观察物理指标；
10. 只有 RC=0 自动授权后才运行 held-out stress/test。

## 10. 不能提前声称的内容

当前环境没有真实 WOMD/Waymax 与两张 A30，不能预先声称 v48.28 会 RC=0、三个 regime 全部通过、或达到 CCF-A 投稿目标。v48.28 的价值是修复实验可解释性和风险表示上限，使下一轮能够得到有效算法结论。
