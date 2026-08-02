# OC-RAP v48.31 结果审计与 v48.32 IDENTITY-UTILITY-BRIDGE 优化方案

## 0. 审计范围与结论边界

本次审计基于以下上传材料：

- v48.31 代码包 `OC-RAP(1).zip`；
- 主实验与 adaptation-dev physical shadow 结果包 `ocrap_v48_31_contract_slack_rank_dedicated_4831.zip`；
- 八个消融任务结果包 `ocrap_v48_31_contract_slack_rank_ablations_4831.zip`。

审计严格区分三类结果：

1. **工程失败（RC=30）**：certificate/gate 未被有效执行，不能据此判断算法好坏；
2. **有效 gate 拒绝（RC=20）**：pipeline 与 certificate 合同有效，但冻结规则没有通过 Natural gate；
3. **有效通过（RC=0）**：只有此路径才允许生成并执行 `NEXT_COMMANDS.txt`，继续读取授权的 held-out stress/test。

本轮主实验属于第 1 类。消融中只有两个 Balanced 任务属于第 2 类，其余六个任务没有形成有效 certificate 结论。

---

## 1. v48.31 主实验的真实 RC

### 1.1 真实返回值是 RC=30，不是 RC=0

顶层 `V48_31_COMPLETE.json` 与 `PIPELINE_FAILED.json` 的一致证据为：

| 字段 | 值 |
|---|---:|
| Balanced adaptation exit | 0 |
| Precision adaptation exit | 31 |
| pipeline failure stage | adaptation |
| raw certificate exit | null |
| gate evaluated | false |
| pipeline valid | false |
| controller normalized RC | **30** |
| test/stress roots read | false |

`V48_31_COMPLETE.json` 中出现的 `certificate_exit_code=30` 容易令人误认为 certificate 已经运行并失败；但同一文件明确记录 `raw_certificate_exit_code=null`、`failure_stage=adaptation`。因此该字段实际是顶层 pipeline 错误码被写进了 certificate 字段，并不是 certificate 的返回值。

### 1.2 本轮主实验不是 `development_rule_fit_rejection`

`development_rule_fit_rejection` 的前提是：

- adaptation 成功；
- certificate controller 被执行；
- adaptation-dev 冻结规则拟合完成但不满足联合约束；
- controller 返回 RC=20。

v48.31 主实验在 Precision adaptation 的 stage-transfer 检查处提前终止，certificate 未运行，gate 未评估。因此：

> 主实验的正确分类是 **adaptation-stage engineering failure / RC=30**，不能称为 `development_rule_fit_rejection`，也不能据此判断 v48.31 主算法通过或未通过 Natural gate。

---

## 2. 为什么没有生成 `NEXT_COMMANDS.txt`

### 2.1 直接原因

v48.31 只在 certificate controller 的有效 RC=0 分支生成：

```text
$OUTPUTDIR/NEXT_COMMANDS.txt
```

主 controller 因 Precision adaptation 返回 31，在 certificate 之前以 RC=30 退出，所以没有进入文件生成分支。对于该实际执行路径，文件缺失是符合旧代码逻辑的。

### 2.2 导致提前退出的真实工程错误

Precision 最终训练选择了 epoch 0，即 Stage-3 的初始 Stage-2 checkpoint。其状态是：

- `stage3_allowed_changed_parameter_count=0`；
- `stage3_disallowed_changed_parameter_count=0`；
- 没有参数丢失；
- checkpoint selection 合法地判断后续 epoch 没有优于初始 checkpoint。

v48.31 的 stage-transfer checker 却把“允许参数一个都没变”直接判为 corruption，并返回 31：

```text
stage3 did not update any allowed evidence calibrator parameter
```

这是错误判定。启用了 `EVALUATE_INITIAL_CHECKPOINT=true` 时，epoch-0 fallback 是预期的 fail-safe 行为，表示优化没有找到更安全的更新，并不表示训练链路损坏。

### 2.3 v48.32 的修复

v48.32 对 `NEXT_COMMANDS` 与 RC 建立显式互斥状态：

- RC=0：生成 `NEXT_COMMANDS.txt` 与 `NEXT_COMMANDS_STATUS.json`；
- RC=20：不生成命令，生成 `NEXT_COMMANDS_BLOCKED.json`，原因 `natural_gate_failed`；
- RC=30：不生成命令，生成 `NEXT_COMMANDS_BLOCKED.json`，记录具体工程阶段；
- 顶层完成文件单独记录 `pipeline_exit_code`；certificate 没有运行时，`certificate_exit_code=null`；
- controller 最后强制核对 RC 与命令文件状态，不一致则转为 RC=30。

Stage-2 和 Stage-3 合法选择 epoch 0 均被接受；只要没有冻结参数变化、参数丢失或架构不一致，就不再产生假 RC=30。

---

## 3. 八个消融实验的真实完成状态

`ABLATIONS_STATUS.json` 明确记录 `complete=false`，六个任务缺失。实际状态如下：

| 消融任务 | 状态 | 可否用于算法结论 |
|---|---|---|
| A Balanced | post-adaptation contract 失败，未记录 TASK_FAILED | 否 |
| A Precision | post-adaptation contract 失败，未记录 TASK_FAILED | 否 |
| B Balanced | post-adaptation contract 失败，未记录 TASK_FAILED | 否 |
| B Precision | post-adaptation contract 失败，未记录 TASK_FAILED | 否 |
| C Balanced | certificate RC=20 | 是 |
| C Precision | adaptation RC=31 | 否 |
| D Balanced | certificate RC=20 | 是 |
| D Precision | adaptation RC=31 | 否 |

A/B 的训练 checkpoint 实际存在，但关闭 Stage-3 后，外层代码只复制了 checkpoint、架构和 policy contract，没有复制：

- `TRAINING_COMPLETE.json`；
- `EVIDENCE_CORRECTION_COMPLETE.json`。

随后训练合同检查抛出 `FileNotFoundError`。这是元数据复制错误，不是算法失败。

C/D Precision 与主实验 Precision 一样，被 epoch-0 no-op 误判阻断。只有 C/D Balanced 完成了完整 certificate，它们都得到有效 RC=20，拒绝类型均为：

```text
development_rule_fit_rejection
```

因此，v48.31 的算法分析只能使用 C/D Balanced 的结果以及有效的 adaptation-dev 诊断；不能把八个任务视为完整消融，也不能把 Precision 或 A/B 的缺失结果当作性能退化。

---

## 4. physical shadow 是否形成了闭环证据

没有。

`DEV_SHADOW_COMPLETE.json` 记录：

```text
balanced_exit=2
precision_exit=2
complete=false
paper_result=false
```

主 pipeline 在 certificate 前失败，因而没有生成 calibration gamma。旧 shadow controller 仍然启动，随后因 gamma 缺失而退出，未产生有效配对 Waymax 指标。

因此，本轮不能对 TTC、clearance、free-space、re-contact、stable stop 或 intervention burden 作任何 v48.31 闭环优劣结论。

v48.32 在调用 Waymax 前强制检查：

- `V48_32_COMPLETE.json` 存在；
- `pipeline_valid=true`；
- `gate_evaluated=true`；
- certificate RC 只能是 0 或 20；
- checkpoint、gamma、目标列表均非空；
- provenance、runtime、物理指标支持和配对场景合同有效。

不满足时直接写 `SHADOW_BLOCKED.json`，不再运行空仿真或产生可误读的零指标。

---

## 5. 已确认的 v48.31 工程错误与 v48.32 修复

| ID | v48.31 问题 | 影响 | v48.32 修复 |
|---|---|---|---|
| E1 | Stage-3 epoch-0 fallback 被当作损坏 | 主流程和两个 Precision 消融假 RC=30 | Stage-2/3 no-op selection 均合法；仍严格禁止冻结参数变化 |
| E2 | no-joint 路径漏复制完成元数据 | A/B 四个任务无法完成合同检查 | 复制完整 checkpoint、训练与 evidence 元数据 |
| E3 | 成功重跑后旧 `ADAPTATION_FAILED_*.json` 残留 | learning gate 与人工审计被旧状态污染 | 每次 controller 启动清理 run-local 状态；成功分支删除失败标记 |
| E4 | 无效主 pipeline 仍启动 shadow | 空仿真或缺 gamma，被误认为物理结果 | Waymax 前 fail-closed preflight |
| E5 | certificate 未运行却写 `certificate_exit_code=30` | 用户容易误判 RC 与 gate 状态 | 分离 `pipeline_exit_code` 和 nullable `certificate_exit_code` |
| E6 | `set -e` 后处理失败未统一写 TASK_FAILED | 六个任务缺失但原因不透明 | 每个 task 和根级失败均写结构化状态与日志尾部 |
| E7 | `NEXT_COMMANDS` 缺失没有原因文件 | 只能依靠猜测 RC | 始终生成 `NEXT_COMMANDS_STATUS.json`，失败时生成 BLOCKED 文件 |
| E8 | 缓存仅检查文件存在 | 不同数据/参数可能静默复用 | 源 checkpoint、索引、support contract、variant 与超参数全量 SHA 合同 |
| E9 | adaptation-dev 索引每次无条件重建或消融中无合同复用 | 浪费时间或可能读到陈旧索引 | 索引只有通过 exact contract audit 才复用，否则自动重建 |
| E10 | Stage-1 缓存在消融间没有身份合同 | 提速可能破坏消融公平性 | `FACTOR_CACHE_CONTRACT.json` 精确验证后才复制 |

这些修复主要提高**结论可信度**，不应被当作模型性能提升本身。

---

## 6. v48.31 中哪些修改已经显示有效

### 6.1 可以确定保留，且没有观察到算法副作用的部分

这里的“无副作用”仅指不会改变模型决策语义或不会放松 gate，不表示已经带来性能增益。

1. **三阶段均采用自然总体、无放回采样。** 该设计修复训练先验与部署总体不一致的问题，没有引入 regime routing，应保留。
2. **训练验证使用精确 executable eligibility。** checkpoint metric 与 certificate 对 feasible、hard-rule、nominal deviation 的总体定义一致，应保留。
3. **valid-safe admission all-abstain 屏障。** harmful/invalid selection 不再被算作解除 abstention，应保留。
4. **Natural certificate、test/stress 封闭和三值 RC 合同。** 这些是可信投稿证据链的一部分，应保留，不能为追求 RC=0 而降低标准。
5. **top-3 frozen proposal。** certificate oracle 仍含 Near 9 个、Contact 20 个 safe-positive group，proposal 支持不是当前瓶颈，无需扩到 top-8。
6. **统一 candidate-vs-nominal 连续物理语义与 independent measured hard veto。** 没有输入 regime ID，也没有三套策略，符合论文 novelty 方向。
7. **candidate AUC 与 group top-1 指标分离。** 这使得“场景/候选可分类”不再被误写成“动作排序可部署”。

### 6.2 有正向信号，但存在副作用，不能直接作为最终设计结论

**support reliability** 在 Near 上产生了保守化信号：

- C Balanced certificate Near：选择 3，safe-positive 0，harmful 2；
- D Balanced certificate Near：选择 0，harmful 0。

它减少了 Near 有害选择，但主要通过完全 abstain 实现。Contact 上则出现反向变化：

- C Balanced Contact：选择 17，harmful 3，平均 teacher advantage `-0.0684`；
- D Balanced Contact：选择 17，harmful 5，平均 advantage `-0.1412`，macro excess 从 `0.0882` 增至 `0.2059`。

因此不能宣称 support reliability “提升且无副作用”。更准确的结论是：

> 对无数据支持坐标进行全局收缩是必要的工程与统计保护，但当前静态 reliability 与最终动作排序的耦合仍需优化；它在 Near 降低 harmful exposure，却没有解决 safe admission，并在 Contact 上可能改变排序边界。

**Stage-3 joint refinement** 也没有完整消融证据。只有 Balanced C/D 有效，Precision 被工程错误阻断；C/D 的候选级指标非常接近，不能证明 joint refinement 的独立净贡献。

---

## 7. Near-contact 投稿成熟度与核心问题

### 7.1 有效结果

D Balanced adaptation-dev Near：

- 110 groups / 49 scenes；
- candidate safe-positive AUC `0.9023`；
- proposal safe top-1 AUC `0.9175`；
- proposal evidence top-1 correlation `+0.0607`；
- 没有满足 gate 的可执行规则；
- closest near-miss：选择 4、safe-positive 1、harmful 0、recall `0.125`、precision LCB90 `0.0781`、平均 advantage `+0.1556`；
- macro share 过度集中。

D Balanced certificate Near：

- 290 groups / 123 scenes；
- candidate safe-positive AUC `0.8253`；
- group top-1 correlation `-0.0218`；
- 9 个 proposal-contained safe-positive opportunities；
- 最终选择 0，safe recall 0。

C Balanced certificate Near 虽选择 3 个动作，但 0 个 safe-positive、2 个 harmful、平均 advantage `-0.3844`。

### 7.2 核心问题

Near 的主要问题已不是“模型完全看不到安全恢复机会”。候选级和 safe top-1 AUC 已经较高。真正瓶颈是：

1. **同一 proposal 内的动作身份排序不稳定。** candidate discrimination 没有转化成相对 nominal/high-benefit-harmful action 的可靠 top-1 顺序；
2. **准入覆盖与精度无法同时满足。** 保守规则完全 abstain，放松规则又迅速产生 harmful/negative-advantage selection；
3. **开发集信号向 certificate 转移不足。** correlation 从微弱正值变成接近零/负值；
4. **macro concentration。** 少量命中集中在少数场景或宏动作，无法形成稳定 population-level 证据；
5. **没有本轮闭环物理证据。** 离线 teacher 指标尚未被有效 Waymax shadow 验证。

### 7.3 投稿成熟度

Near 当前属于“有明确算法信号、可作为方法迭代依据”的阶段，但还不能作为 CCF-A 主结果：

- 有较好的候选识别信号；
- 没有 certificate safe-positive selection；
- 没有通过 precision/recall/selection-count/macro 联合 gate；
- 没有有效 paired physical shadow。

离投稿主表要求仍至少缺少：有效 Natural gate、独立/预注册证书上的稳定 safe admission、非负且有实际幅度的物理改善、Safe 非劣性。

---

## 8. Contact 投稿成熟度与核心问题

### 8.1 有效结果

D Balanced adaptation-dev Contact：

- 279 groups / 82 scenes；
- candidate safe-positive AUC `0.6260`；
- proposal evidence top-1 correlation `-0.1360`；
- best near-miss：选择 11、safe-positive 1、harmful 2；
- precision LCB90 `0.0276`；
- recall `0.0588`；
- mean advantage `-0.0418`。

D Balanced certificate Contact：

- 764 groups / 215 scenes；
- candidate safe-positive AUC `0.5795`；
- proposal safe top-1 AUC `0.4502`，低于随机排序参考；
- group top-1 correlation `-0.1701`；
- 20 个 proposal-contained safe-positive opportunities；
- 选择 17、safe-positive 0、harmful 5；
- mean advantage `-0.1412`；
- macro excess `0.2059`，超过注册上限。

C Balanced 也未命中 safe-positive：选择 17、harmful 3、mean advantage `-0.0684`。

### 8.2 核心问题

Contact 比 Near 更不成熟，其问题是结构性的：

1. **候选表示本身区分度不足。** safe-positive AUC 仅约 0.58–0.63；
2. **组内排序方向错误。** top-1 correlation 显著为负，safe top-1 AUC 在 certificate 低于 0.5；
3. **高表面收益 harmful action 被优先。** 选择数量不低，却没有 safe-positive 命中；
4. **开发集与 certificate 都没有稳定正 advantage。** 不是单纯阈值校准问题；
5. **macro concentration 与分布异质性更严重。** 单一全局阈值难以修复错误动作身份；
6. **离线 recovery benefit 尚未充分表达撞后时间过程。** 当前标签更接近静态 recoverability，不能直接保证 secondary-contact avoidance、escape quality 或 stable stop。

### 8.3 投稿成熟度

Contact 尚未建立 safe admission 的基本证据，距离 CCF-A 主结果明显更远。当前结果只能支持：

- top-3 proposal 中存在理论可行安全动作；
- 当前 selector 没有学会识别和排序这些动作；
- 需要先修复 representation/ranking coupling，再讨论阈值或 certificate generalization。

---

## 9. v48.31 算法失败的根本原因

### 9.1 candidate classification 与 deployment top-1 脱节

v48.31 的 Stage-2 只训练 admission calibrator，并冻结 benefit/harm factor heads。模型中的 safety-slack admission prior 又对这些量显式 `.detach()`：

```text
admission prior = detached benefit - penalty * relu(max(detached component margins))
```

因此 safe-utility listwise、hard-negative、setwise 等部署相关损失主要只能调整一个小的 admission residual，不能反向修正：

- 哪个 candidate 的 benefit 被高估；
- 哪个 component safety margin 的符号/幅度错误；
- 为什么同组 harmful action 排在 safe action 前面。

这正好解释了本轮组合现象：

- candidate AUC 看似较高；
- group top-1 correlation 接近零或为负；
- 所有可用最终 checkpoint 的 `direct_contract_valid_safe_admission_total=0`；
- Stage-2 后从有限正信号坍缩为 abstention 或 harmful admission。

### 9.2 Contact 还存在表示不足

Contact 的 candidate AUC 本身较低，仅靠调 admission 阈值不可能解决。它需要让 benefit 与连续 safety slack 在组内共同学习，并由真实 safe-utility gap 决定排序强度，而不是继续叠加二分类 loss 或扩大 proposal。

---

## 10. v48.32 IDENTITY-UTILITY-BRIDGE

### 10.1 统一语义，不做 regime routing

v48.32 不输入 Safe/Near/Contact ID，也不建立三套 planner。三个 regime 仍只是报告分层。

对 candidate 相对 nominal 定义全局连续安全余量：

```text
m = [m_DRS, m_deployability, m_gap, m_hard-rule, m_harm-proxy]
r = global support reliability
s(a) = max_k r_k * m_k(a)
U(a) = B(a) - lambda * relu(s(a))
```

其中 measured hard veto 继续独立、不可由 benefit 补偿。该表示在 Safe、Near、Contact 中连续变化，不需要 case 1/2/3 策略。

### 10.2 Stage-1：自然总体物理因子学习

保留 v48.31 中有效的：

- natural without replacement；
- raw benefit；
- signed component margin regression；
- support-weighted监督；
- exact executable validation population。

### 10.3 Stage-2：proposal-local action identity

默认同时训练三个 compact calibrator：

```text
benefit calibrator
component-harm calibrator
admission calibrator
```

关键修改是取消 Stage-2 safety-utility prior 的梯度阻断：

```text
safe-utility/listwise/hard-negative gradient
    -> admission
    -> benefit
    -> supported component margins
```

模型输出与推理公式不变，只改变训练梯度路径；没有增加 regime-specific head。

为了限制 joint update 破坏 Stage-1 物理语义，保留 calibrator anchor，并使用低学习率。

### 10.4 自适应 teacher-gap hardest negative

固定 margin 无法区分“safe action 仅略好于 nominal”与“safe action 明显优于 harmful competitor”。v48.32 使用连续 teacher gap：

```text
required_margin = base_margin
                + scale * clamp(U_teacher(safe) - U_teacher(hard_negative), 0, 0.25)
```

没有 safe action 的 group 则根据 teacher no-op depth 强化 nominal 优先。该损失仍对所有 regime 使用同一物理量与同一公式。

### 10.5 Stage-3：admission-only 低率校准

Stage-3 只校准 admission residual，用于在不重新扭曲 benefit/component identity 的情况下调整覆盖。epoch 0 是合法 fallback；如果没有更安全的更新，直接保留 Stage-2 checkpoint。

### 10.6 新消融

| 组 | Stage-2 trainable | prior gradient | hard-negative margin |
|---|---|---|---|
| A | admission only | detached | fixed |
| B | benefit + harm + admission | detached | fixed |
| C | benefit + harm + admission | coupled | fixed |
| D | benefit + harm + admission | coupled | adaptive teacher gap |

所有组均保留 support reliability，避免再次把已知的 unsupported-coordinate 工程保护与 identity learning 混在同一对比中。

判读：

- B>A：联合更新 compact factors 是否有价值；
- C>B：部署 safe utility 的梯度耦合是否是关键；
- D>C：连续 teacher-gap margin 是否改善安全 top-1 和 Contact harmful ranking；
- 任何 dev 提升都必须在 certificate 和 paired physical shadow 上复核。

---

## 11. 应保留、深化和替换的设计

### 11.1 保留

- top-3 frozen proposal；
- raw benefit 与 safe admission 语义分离；
- natural without-replacement population；
- five-coordinate nominal-relative continuous representation；
- independent measured hard veto；
- bounded admission residual；
- nominal + top-k categorical one-action policy；
- exact executable checkpoint metric；
- Natural certificate、test/stress sealing、三值 RC；
- global support audit；
- no regime ID / no regime-specific policy。

### 11.2 深化

- group-local safe top-1 supervision；
- signed physical margin regression；
- safe-utility hardest negative；
- macro concentration penalty；
- calibration/certificate population consistency；
- paired adaptation-dev physical shadow。

### 11.3 修改或替换

- **替换 Stage-2 admission-only frozen factors**：改为 compact factor + admission 联合 identity learning；
- **替换 detached deployment prior**：Stage-2 使用 coupled gradient，Stage-3 再冻结 factors；
- **修改固定 hard-negative margin**：使用连续 teacher-gap margin；
- **不要把 support reliability 当作性能模块单独宣称**：它是数据支持保护，效果要结合 identity learning；
- **不要继续只叠加全局二分类 loss**：Contact 的主要问题是组内动作身份与物理时序表示；
- **不要扩大 proposal 或降低 gate**：oracle 已可行，扩大候选会增加 harmful exposure 和校准负担；
- **不要把三个 regime 拆成独立状态策略**：保持统一连续物理表示。

如果 v48.32 的离线 safe top-1 显著改善、但 physical shadow 仍无改善，下一步才应预注册 candidate-level Waymax temporal rollout auxiliary targets，而不是从现有静态标签中虚构二次接触或稳定停车监督。

---

## 12. 训练和评估速度审计

### 12.1 主要瓶颈

v48.31 主实验中，Stage-1 factor training 占总训练时间约 `58.4%`。八个消融的有效训练时间中，Stage-1 占约 `65.3%`：

- 八次 factor training 合计约 `11,302 s`，即 `3.14 h`；
- 每个消融的 Stage-1 数据、标签和 factor 超参数实际上相同，仅 variant 不同；
- 重复训练八次没有为 A/B/C/D 对比提供额外信息。

### 12.2 v48.32 优化

1. 每个 variant 最多训练一次 Stage-1，B/C/D 复用 A 的 factor stage；
2. 推荐在主实验之后运行消融，并把主实验 Balanced/Precision factor stage 作为 A 的缓存，这样消融可额外训练 **0 次** factor stage；
3. 缓存必须通过 exact SHA contract：source checkpoint、train/dev index、support contract、variant、模型和训练超参数任一变化都会拒绝复用；
4. train 与 adaptation-dev teacher index 只有通过 label/data contract audit 才复用，否则自动重建；
5. 两个 variant 继续分配到两张 A30 并行；四个消融 wave 维持每次两个并发任务，避免单卡过载；
6. certificate 和 Waymax 物理仿真不削减样本、不缩短 horizon，也不以缓存结果代替新模型评估。

估计节省：

- 独立运行消融：从 8 次 factor training 降为 2 次，约节省 `2.38 h`；
- 按推荐顺序复用主实验 factor：从 8 次降为 0 次，约节省 `3.14 h`。

### 12.3 不建议的“提速”

- 不降低 certificate population；
- 不缩短 Waymax rollout 或 target list；
- 不跳过 Precision；
- 不关闭 exact eligibility；
- 不用已多次查看的 certificate 调参；
- 不把 adaptation-dev shadow 当 paper test。

---

## 13. 下一轮判定标准

### RC=30

- 先查看 `PIPELINE_FAILED.json`、`NEXT_COMMANDS_BLOCKED.json` 与对应合同；
- 不做算法优劣结论；
- 不运行 Waymax shadow、test 或 stress；
- 修复工程错误后重跑同一预注册配置。

### RC=20

- 这是有效 Natural gate 拒绝；
- 运行 v48.32 四组消融和 adaptation-dev physical shadow；
- 优先检查：safe top-1 hits、group correlation、harmful selection、mean advantage、macro excess；
- 不读取 held-out stress/test。

### RC=0

- `NEXT_COMMANDS.txt` 必须存在，且 `NEXT_COMMANDS_STATUS.json` 记录 generated=true；
- 只执行自动生成的命令；
- 完成 Safe paired non-inferiority 和授权的 held-out stress；
- 对最终论文仍应使用新封闭或预注册 certificate，因为当前 certificate 已被多轮查看。

---

## 14. 本地验证边界

v48.32 本地完成：

- pytest：285 passed，5 warnings；
- `compileall`：通过；
- 全部 shell `bash -n`：通过；
- v48.32 shell-to-tool dependency audit：无缺失；
- stage-transfer no-op、cache mismatch、NEXT state、shadow preflight 等专项回归：通过。

当前环境没有服务器上的原始 WOMD/Waymax 数据与两张 A30，不能预先声称 v48.32 会得到 RC=0。v48.32 的目标是先消除 v48.31 的假 RC=30，并直接验证“动作身份—safe utility 梯度耦合”是否能同时改善 Near safe admission 与 Contact harmful ranking。
