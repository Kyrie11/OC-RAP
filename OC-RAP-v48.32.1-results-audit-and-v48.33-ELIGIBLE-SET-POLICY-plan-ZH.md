# OC-RAP v48.32.1 结果审计与 v48.33 ELIGIBLE-SET-POLICY 方案

## 1. 执行结论

v48.32.1 主流程的观测结果为：

- `certificate_exit_code=20`
- `pipeline_exit_code=20`
- `certificate_executed=true`
- `gate_evaluated=true`
- `gate_passed=false`
- `NEXT_COMMANDS.txt` 未生成
- `test_roots_read=false`

因此，这一轮不是 RC=30 工程崩溃。控制器给出的直接拒绝类型是：

```text
development_rule_fit_rejection
```

它说明 v48.32.1 的冻结 selector 没有找到一条能够从 adaptation-dev 冻结并在 scene-disjoint certificate 上满足 Natural gate 的联合规则。

但是，本轮同时存在一个重要的统计协议实现错误：dev rule fitting 命令把较宽松的 verify 阈值传给了 `--min-fit-* / --max-fit-*`。因此：

1. “算法没有通过”这一结论仍然成立；
2. 本轮并没有发生错误放行；
3. 旧输出中的 constraint deficit、fit oracle feasibility 和无条件的 `pipeline_valid=true` 不能作为严格预注册协议下的正式论据；
4. v48.33 必须先修复该合同，再用新结果判断算法优劣。

## 2. 预注册合同与实际执行合同的偏差

### 2.1 GATE_SPEC 中声明的 fit 条件

Near：

- 最少选择数：10
- precision one-sided 90% Wilson LCB：至少 0.50
- harmful group exposure UCB：不超过 0.12
- selected harmful UCB：不超过 0.22

Contact：

- 最少选择数：16
- precision LCB：至少 0.50
- harmful group exposure UCB：不超过 0.14
- selected harmful UCB：不超过 0.22

### 2.2 v48.32.1 实际用于 dev fitting 的条件

Near 实际使用：`8 / 0.40 / 0.14 / 0.25`。

Contact 实际使用：`10 / 0.40 / 0.16 / 0.25`。

这些正好是 verify 条件，而不是 fit 条件。该错误不会把当前失败伪造成成功，因为在更宽松条件下仍然没有通过；但它会低估算法与正式目标之间的距离，并使 proposal support 的“可行”判断过于乐观。

## 3. RC=20 的算法根因

### 3.1 模型学到的是场景机会，而不是 proposal 内动作身份

Near 的候选级区分能力已经不弱：

| Variant | Near candidate positive AUC | Near candidate safe-positive AUC |
|---|---:|---:|
| Balanced | 0.825 | 0.831 |
| Precision | 0.819 | 0.796 |

但真正决定部署动作的 proposal evidence top-1 correlation 为：

- Balanced Near：约 -0.014
- Precision Near：约 -0.011

也就是说，模型可以判断“某个场景附近可能有恢复收益”，却没有稳定判断“同一个 scene-time proposal 内哪个 candidate 才是安全恢复动作”。

Contact 更严重：

- candidate safe-positive AUC 只有约 0.581 / 0.553；
- proposal evidence correlation 约 -0.136 / -0.167；
- candidate 表征本身接近随机，proposal 内排序又是反相关。

### 3.2 训练 checkpoint 与实际部署的动作选择顺序不一致

v48.32.1 的训练验证统计执行：

```text
rank top-k -> evidence top-1 -> 检查 opportunity/harm -> 可能 abstain
```

而 calibration/runtime 执行：

```text
rank top-k -> opportunity/harm 过滤 -> 在 eligible candidates 中 evidence rerank
```

这两个策略不等价。若最高 evidence candidate 不合格、第二名是合格的安全动作：

- 训练验证会把整个 group 当作 abstain；
- runtime 会选择第二名；
- checkpoint selection 无法奖励真正会部署的安全 runner-up。

### 3.3 软 early-stopping risk 仍然忽略 eligibility

即使硬统计修正为“先过滤再重排”，原软 checkpoint risk 仍然只对 evidence 做 softmax，没有让 opportunity/harm eligibility 进入动作概率。于是，一个 runtime 一定会拒绝的高 evidence harmful candidate，仍可能降低训练风险并推动 checkpoint 被选中。

这会直接造成：

- candidate AUC 看起来良好；
- soft validation risk 改善；
- 真实可执行 top-1 仍然没有安全命中。

### 3.4 top-3 在严格 Near fit 合同下结构性不可行

adaptation-dev 的 proposal support 为：

| Variant | Near top-1 | Near top-3 | Near top-5 | Near top-8 |
|---|---:|---:|---:|---:|
| Balanced | 5 | 7 | 8 | 8 |
| Precision | 3 | 7 | 8 | 8 |

严格 Near fit 至少选择 10 个动作，并要求 precision LCB≥0.50。

即使 top-3 中的 7 个安全机会全部选对，再补 3 个非正动作，7/10 的 one-sided 90% Wilson LCB 约为 0.4974，仍低于 0.50。因此 top-3 在严格 fit 下没有数学可行性。

top-5 包含 8 个安全机会。理想的 8/10 LCB 约为 0.6016，严格 fit 至少在支持度上变得可行。

这一扩容仍是统一策略：所有 regime 都使用相同 top-5，没有 Safe/Near/Contact routing。

### 3.5 Stage 3 没有修复表示问题

v48.32.1 的最终 admission calibration：

- Balanced 最终阶段 best epoch 为 0；
- Precision 最终阶段 best epoch 为 0；
- v48.32 消融中 Precision A/B/C/D 基本完全相同；
- adaptive teacher-gap C/D 也基本相同。

因此，额外 admission-only Stage 3 主要增加训练时间，并没有证据表明它能够修复 benefit/harm/action identity。

## 4. Near-contact 投稿成熟度

### 4.1 当前最好的正向信号

Precision Near adaptation-dev 最近规则：

- selected：10
- safe positive：3
- harmful：1
- precision：0.30
- precision LCB90：0.154
- safe recall：0.375
- mean teacher advantage：+0.191
- max selected macro share：0.60

这说明 Near 已经存在真实的方法信号：模型在开发集上能挑出若干有正收益的安全动作，而且平均收益为正。

Balanced Near 也显示：

- candidate safe-positive AUC≈0.831；
- harmful 可被压到 0；
- 选择的平均 teacher advantage≈+0.156。

### 4.2 主要缺陷

冻结到 certificate 后：

- Balanced：选择 0，安全命中 0；
- Precision：选择 8，安全命中 0，harmful 4，平均 advantage -0.298，macro share 0.875。

因此 Near 当前仍不具备论文主结果所需的：

1. scene-disjoint safe-positive 命中；
2. 稳定的 dev→certificate 迁移；
3. 低 harmful selection；
4. 跨 macro 的非集中选择；
5. 可配合 closed-loop 物理指标的有效干预。

Near 的成熟度应描述为：

> 已经有较强候选级与开发集方法信号，适合继续作为核心研究方向；但部署级动作身份和泛化尚未建立，不能作为 CCF-A 稿件的主要成功结论。

## 5. Contact 投稿成熟度

### 5.1 当前结果

Balanced Contact adaptation-dev 最近规则：

- selected：11
- safe positive：1
- harmful：1
- recall：0.059
- mean advantage：+0.007

Precision Contact：

- selected：19
- safe positive：2
- harmful：3
- recall：0.118
- mean advantage：-0.018

certificate：

| Variant | selected | safe positive | harmful | mean teacher advantage |
|---|---:|---:|---:|---:|
| Balanced | 18 | 0 | 5 | -0.134 |
| Precision | 30 | 0 | 15 | -0.230 |

### 5.2 核心问题

Contact 不只是 threshold 或 recall 不足，而是三层能力同时不足：

1. **候选表示弱。** safe-positive AUC 接近随机；
2. **动作排序反向。** proposal evidence correlation 明显为负；
3. **高表面收益 harmful action 过度占优。** 模型仍通过损害连续安全余量换取 raw benefit；
4. **开发集选择已经包含负平均收益。** Precision Contact 在 dev 就没有建立正向 utility；
5. **certificate 宏动作集中。** Precision 30 次选择中 macro 5 占 25 次。

Contact 当前成熟度应描述为：

> 仍处于初步算法验证阶段，尚未建立 safe admission 或安全动作排序。它目前不能承担论文中“撞后恢复有效”的主要实证结论。

## 6. 哪些设计有效、值得保留

### 6.1 工程与统计上明确有效

以下修改提高了结论可信度，并没有性能意义上的副作用：

- RC=0/20/30 三值合同；
- certificate/test root 隔离；
- scene-disjoint adaptation-dev 与 certificate；
- exact executable eligibility；
- 每组恰好一个 nominal 的 fail-closed 检查；
- strict tensor shape contract；
- 自然总体、无放回训练；
- checkpoint、缓存、索引、support contract 的 SHA/参数身份检查；
- independent measured hard veto；
- `NEXT_COMMANDS` 显式授权状态。

### 6.2 算法上有正向证据、应继续深化

- raw benefit 与 safe admission 分离；
- candidate-vs-nominal 连续物理余量；
- 不向模型输入 regime ID；
- proposal-constrained one-action policy；
- support reliability 原则；
- bounded admission residual；
- coupled benefit/component/admission gradient。

v48.32 消融中，Balanced Contact 从 detached joint 的 8 个 harmful、advantage -0.187，改善到 coupled 的 5 个 harmful、advantage -0.134。说明 coupling 至少能抑制部分 harmful degradation。

但是它会导致 Near 更保守甚至全 abstain，因此不能称为“无副作用性能提升”。它值得保留，但必须与 exact eligible-set objective 结合，而不是单独宣称成功。

## 7. 哪些设计无效或需要替换

### 7.1 adaptive teacher-gap margin

C 与 D 基本相同，说明当前 adaptive scale 没有产生可测贡献。默认关闭，保留 fixed hardest-negative 即可。

### 7.2 admission-only Stage 3

反复选择 epoch 0，没有改善 certificate。默认删除以减少时间和避免无效参数漂移。

### 7.3 all-head detached joint training

Balanced Contact harmful 从 admission-only 的 6 增加到 8，mean advantage 更负。说明更新 benefit/harm heads但阻断部署 utility 梯度，会扩大表示漂移而不改善最终动作。

### 7.4 top-3 proposal

在严格 Near fit 下结构性不可行，需要替换为统一 top-5。

### 7.5 candidate AUC 作为主要 checkpoint 或投稿证据

Near 已经证明 candidate AUC 可以很高但 certificate safe hit 为 0。候选分类只能作为辅助诊断，核心指标必须是 exact proposal eligible-set top-1、safe admission、harmful mass、teacher utility 与 closed-loop physical metrics。

## 8. v48.33 ELIGIBLE-SET-POLICY

### 8.1 统一语义

仍然不识别 Safe、Near、Contact 三种状态，也不使用三套策略。

对所有场景统一执行：

```text
frozen rank top-5 proposal
-> opportunity/harm eligibility
-> evidence rerank within eligible set
-> choose one recovery action or nominal
```

### 8.2 differentiable eligible-set policy

对 proposal 中每个 candidate，学生 policy logit 由两部分组成：

- bounded admission evidence；
- opportunity/harm 的连续 soft eligibility。

nominal 是显式 abstention 类。teacher distribution 使用连续 safe utility。

这使训练能够：

- 奖励合格的安全 runner-up；
- 抑制 runtime 一定会过滤的高 evidence candidate；
- 同时向 benefit、supported harm margins 和 admission head 传播梯度；
- 直接优化一次动作选择，而不是独立优化多个候选二分类。

### 8.3 checkpoint、calibration、runtime 完全同序

v48.33 同时修复：

- 硬 checkpoint top-1；
- 软 early-stopping distribution；
- dev rule fitting；
- certificate verification；
- POLICY_CONTRACT/GATE_SPEC 元数据。

所有环节都采用 `rank_topk_then_filter_then_evidence_rerank`。

### 8.4 严格 fit 协议

v48.33 只把 preregistered fit 阈值传给 fit，不再混用 verify 阈值。

在读取 certificate 前，新 checker 必须确认：

- train metric 与 dev calibration 的 eligible group 数相同；
- proposal safe-opportunity 数相同；
- fit 阈值与 GATE_SPEC 完全一致；
- proposal top-k 全链一致；
- selection semantics 一致；
- strict proposal oracle fit 可行。

否则返回 RC=30，不允许读取 certificate。

### 8.5 两阶段训练

Stage 1：

- natural population；
- raw benefit；
- signed continuous physical margins；
- support reliability；
- no admission head。

Stage 2：

- natural population；
- top-5 proposal identity；
- coupled compact heads；
- safe utility regression/listwise；
- eligible-set policy KL；
- fixed hardest-negative；
- bounded admission。

Stage 3 默认关闭。

## 9. 运行时间优化

v48.33 不通过裁剪 certificate 或减少 Waymax 场景来提速。

主要优化来自：

1. 删除无证据收益的 Stage 3；v48.32.1 每个 variant 额外训练约 5 个 epoch 且最终选择 epoch 0；
2. v48.33 消融全部复用主实验的 top-5 Stage-1 factor；
3. 每波只训练两个 identity task，一张卡一个；
4. teacher index 在数据与标签合同不变时复用；
5. shadow 默认 fast physical label mode，保留真实 policy/Waymax physics，减少在线 teacher relabel 开销；
6. 不复用旧 top-3 factor，因为这会违反 top-k 监督和缓存合同。

## 10. 下一轮判据

### RC=30

只分析工程或协议。禁止消融、shadow、test、stress。

### RC=20

说明严格 pipeline 与 Natural gate 有效。重点比较：

- Near certificate 是否首次出现 safe-positive hit；
- Near candidate AUC 是否转化为 positive proposal correlation；
- Contact harmful selection 和负 advantage 是否显著下降；
- top-5 是否提高 safe recall而没有使 harmful mass失控；
- eligible-set loss对 admission-only 和 joint-coupled 的净贡献；
- adaptation-dev shadow是否出现一致的 TTC/clearance/free-space 改善。

### RC=0

只执行自动生成的 NEXT_COMMANDS，并继续 Safe paired、test/stress 的正式投稿证据链。

## 11. 结论

v48.32.1 的 RC=20 是实际失败，但旧 fit 实现没有严格遵守 preregistration。当前最核心的算法问题不是 proposal 完全没有机会，也不是单纯阈值太严格，而是：

> 模型没有把候选级机会识别转化为与真实部署顺序一致的 proposal 内安全动作选择。

v48.33 不拆分 regime，而是统一修复 proposal 支持、eligible-set 动作语义、梯度路径、checkpoint 与统计合同。它是目前最有证据支撑的下一步；真实是否改善必须由新的严格 certificate 结果决定。
