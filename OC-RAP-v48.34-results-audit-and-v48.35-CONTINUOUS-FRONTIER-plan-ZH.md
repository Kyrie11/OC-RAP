# OC-RAP v48.34 结果、工程与算法审计，以及 v48.35 CONTINUOUS-FRONTIER 方案

## 1. 审计范围与结论

本次审计基于以下上传材料：

- 论文：`post-collision(3).tex`；
- 上一轮分析：`大模型建议(2).md`；
- 代码：`OC-RAP(5).zip`；
- 数据集分析：`reports(3).zip`；
- v48.34 主实验：`ocrap_v48_34_barrier_crossfit_dedicated_4834(3).zip`；
- v48.34 消融：`ocrap_v48_34_barrier_crossfit_ablations_4834(3).zip`。

核心结论如下。

1. **当前 RC=20 是有效算法拒绝，不是本轮主实验的工程假失败。** 主控制器记录 `pipeline_valid=true`、`certificate_executed=true`、`gate_evaluated=true`、`certificate_exit_code=20`、`test_roots_read=false`。
2. **v48.34 的主要失败不是 top-5 候选中没有安全恢复机会，而是统一 selector 无法稳定识别并准入这些机会。** proposal oracle 显示 Near/Contact 都存在可恢复候选。
3. **Near 已出现可继续发展的表示信号，但距离 CCF-A 主结果仍明显不足。** Balanced 在独立 certificate 上只命中 1 个 safe-positive，LCB 和 recall 极低；Precision 还伴随较多 harmful selection。
4. **Contact 仍未形成正确的动作排序方向。** 两个 variant 的 certificate safe-positive 命中均为 0，选择动作平均 teacher advantage 为负，candidate safe-positive AUC 约 0.55，接近弱随机区分。
5. **上一轮对 v48.34 的算法方向判断大体正确，但代码与指标中存在会误导消融归因的工程/语义错误。** 最重要的是：Near/Contact 实际各自拟合了不同 rule；“exact eligible”指标并未使用真实冻结 rule；训练边界与部署 rule 不一致；软 barrier 仍允许补偿；后置命令还存在旧脚本依赖。
6. **本轮给出的 v48.35 不把三个 regime 拆成三种策略。** 它使用候选相对 nominal 的连续可执行前缀物理量、统一五分量安全前沿和单一共享 rule。Near/Contact 只作为最坏分层审计，Safe 通过 nominal lock 和 paired non-inferiority 检验。
7. **本地没有 WOMD/Waymax 与两张 A30，无法诚实声称 v48.35 已经 RC=0。** 本次交付完成的是算法与工程实现、契约测试、运行链和实验设计。

## 2. 论文理解与当前实现的对应关系

论文的核心主张是把 recoverability 提升为规划的一等目标，用 observation-consistent roots、signed recovery margins、OC-MERO 和 calibrated selector，将正常驾驶、低余量交互、near-contact 与 post-contact stabilization 纳入同一规划原则。论文明确写到：同一 recoverability criterion 应在 normal-to-critical continuum 上工作，并在附录中强调统一 margin 在 normal 和 post-contact 下都保持定义。

这个主张与用户要求一致：不能把 Safe、Near、Contact 变成三套离散 case policy。

但当前稿件和代码仍有三个需要在投稿前统一的地方。

### 2.1 论文主方法与实际训练器的抽象层级不一致

论文主线强调 OC-MERO、root compatibility、lower-tail recoverability 和 CRISP；当前 v48.x 主实验更接近“在 top-k proposal 上训练 compact teacher-label selector”。这并不一定错误，但投稿时必须明确：

- compact selector 是 OC-MERO 输出的可部署近似，还是一个独立替代模块；
- 五个 signed component margins 与论文 recovery constraints 的一一对应；
- shared rule 如何对应 calibrated planning primitive，而不是后处理 heuristic。

若不澄清，审稿人容易认为论文方法与实验代码不是同一个算法。

### 2.2 论文中的 regime-conditioned protective certificate 应改写

稿件附录中存在“仅在 low-headroom contact regime 使用的 regime-conditioned protective certificate”表述。这与统一算法主张冲突，也与用户明确要求冲突。建议改为：

> 保护性准入由连续低余量 signed frontier 触发；regime 标签仅用于评价分层，不进入模型或策略路由。

保护 macro 的语义约束可以保留，但触发变量必须是连续物理 margin，而不是 `if contact then ...`。

### 2.3 结果表仍为空

论文主表与消融表仍有大量 `--`。当前结果尚不能填入 CCF-A 结论表；尤其 Contact 不能用现有 certificate 结果包装成成功。建议在 v48.35/v48.36 通过统一 gate、Safe non-inferiority、paired closed-loop 和多 seed 后再填主表。

## 3. 数据集审计中需要承认、但本轮不重构的问题

本轮遵循用户要求，不建议重建数据集。现有数据足以继续做算法诊断，但论文和代码必须正确处理其限制。

上传报告显示大致规模如下：

| Split | Safe | Near-contact | Contact |
|---|---:|---:|---:|
| Train samples | 20,000 | 13,324 | 16,790 |
| Validation samples | 2,328 | 3,445 | 6,477 |
| Calibration samples | 2,544 | 6,039 | 16,843 |
| Test samples | 3,216 | 4,723 | 6,687 |

优点：

- train/dev/certificate scene-disjoint；
- Near/Contact top-k 中存在相当数量 oracle-recoverable opportunity；
- 数据覆盖了从正常到低余量再到接触后的连续物理变化。

需要在论文限制中承认：

- 论文写五类 regime，当前实证重点只有三类；
- Contact 某些事件型指标存在 floor/ceiling saturation，连续 clearance、deployability、gap、stability margin 更有信息量；
- 有过时重复文件名，例如 `traincontact.json` 与 canonical `train_contact.json`，代码必须只接受 canonical manifest；
- 当前不重构数据，因此算法必须提高 scene-disjoint generalization，不能靠 regime-specific threshold 吸收数据偏差。

## 4. RC=20 是否被工程错误误导

### 4.1 主实验 RC=20 的真实性

主实验不是 RC=30。完整状态说明：

- pipeline 已执行到 certificate；
- gate 已被评价；
- test root 未读取；
- 失败类型是 development/shared-rule fit 与 certificate performance 不满足注册条件。

因此，不能把 v48.34 失败归因于缓存、脚本或 checker 后直接认为算法可能已经通过。

### 4.2 会误导算法归因的工程问题

| 问题 | 对结论的影响 | v48.35 处理 |
|---|---|---|
| 上传源码混有 v48.34、v48.34.1、v50.x 痕迹 | 容易把未执行代码当成生成结果的代码 | 独立 v48.35 版本化 controller、contract 与 changelog |
| Near/Contact 分别拟合 frozen rule | 实际部署策略被分叉，削弱统一算法 novelty，也可能掩盖表示问题 | pooled adaptation-dev 只拟合一个共享 rule |
| `proposal_exact_eligible_*` 使用固定 0.65/0.30 诊断阈值 | 名称“exact”不成立，可能与真实部署动作不同 | 新增 `proposal_deployed_rule_*`；旧字段仅保留为 deprecated alias |
| hard-boundary loss 用的边界与最终 fitted rule 不一致 | C 消融近乎不变不能说明 boundary 无效 | 训练使用注册语义边界；共享 rule 被限制在同一安全语义域 |
| barrier 是软乘法/软惩罚 | 高 benefit 或大 residual 仍可补偿 component deterioration | non-compensatory smooth cap |
| compact context 缺少候选可执行前缀物理差异 | Contact action identity 欠识别；模型利用 scene shortcut | candidate-minus-nominal physical-relative context |
| dev fitter RC=3 被 shell 当成失败 | 有效算法拒绝可被转换为 RC=30 | dev RC 0/3 均保留结果；只有缺失/损坏为 RC=30 |
| 生成命令引用旧/缺失脚本 | gate 通过后仍可能工程失败或绕过新授权 | 新 Safe/stress wrapper，验证 v48.35 completion 和 shared-rule SHA |
| 历史测试引用上传 ZIP 中不存在的旧脚本 | 全量 pytest 失败会被误读为新代码回归 | 明确区分缺失历史资产与 supported release test matrix |

### 4.3 v48.34 消融为什么不能按表面结果解释

v48.34 的 2×2 消融大致为：reference、barrier、hard boundary、full。

- barrier 组通常提高了选择量，但 harmful selection 同时增加；这说明它改变了 coverage，却没有修正动作安全排序。
- hard-boundary 组与 reference 多处几乎一致；由于训练边界与部署 rule 不同，这不能证明 boundary continuation 没用，只能说明该实现没有作用到真实部署边界。
- full 组在多项 Near/Contact 指标上更差；这说明 soft barrier + mismatched boundary 的组合不能解决表示欠识别。

因此，v48.34 消融真正支持的是：**只改变准入几何，而不补充动作级物理表示，无法解决 Contact；补偿式 barrier 的正向信号不足。**

## 5. Near-contact 的核心问题与投稿成熟度

### 5.1 当前核心问题

Near 的 top-5 proposal 不是主要瓶颈。Balanced certificate 中：

- 290 个 group；
- 5 个 selected；
- 1 个 safe-positive；
- 0 个 harmful；
- precision=0.20，但 90% Wilson LCB 只有约 0.062；
- positive recall 约 0.111；
- selected teacher advantage mean 约 +0.124；
- candidate safe-positive AUC 约 0.868。

Precision certificate 中：

- 9 个 selected；
- 1 个 safe-positive；
- 4 个 harmful；
- precision LCB 约 0.034；
- recall 约 0.111；
- teacher advantage mean 约 -0.191；
- macro share 约 0.778。

这说明 Near 已有“候选区分信号”，但没有形成可靠部署策略。核心缺陷是：

1. candidate-level AUC 尚可，但 group-level eligible top-1 与最终 admission 不稳定；
2. 安全机会稀疏，checkpoint 容易被少数 scene shortcut 支配；
3. coverage、precision 与 macro concentration 之间不稳定；
4. 单个 clean hit 的样本量不足以形成论文级证据；
5. 目前 closed-loop 只有非常小量级的 TTC/clearance 改变，尚未展示稳定控制收益。

### 5.2 CCF-A 投稿成熟度

Near 可作为“方法有潜力、值得继续”的内部结果，但不能作为 CCF-A 主结论。它至少还缺：

- 同一共享 rule 的 adaptation-dev 与 certificate gate 通过；
- 多 seed 置信区间；
- paired closed-loop 在同 scene/target 上稳定改善；
- Safe non-inferiority；
- failure examples 与 intervention cost 的完整报告。

## 6. Contact 的核心问题与投稿成熟度

### 6.1 当前核心问题

Balanced Contact certificate：

- 764 groups；
- 41 selected；
- 0 safe-positive；
- 16 harmful；
- teacher advantage mean 约 -0.164；
- candidate safe-positive AUC 约 0.562。

Precision Contact certificate：

- 34 selected；
- 0 safe-positive；
- 17 harmful；
- teacher advantage mean 约 -0.221；
- candidate safe-positive AUC 约 0.545。

Contact 的错误主要来自 deployability、gap 和 DRS component，而不是 hard violation 或 harm proxy 单一头。审计计数显示，在 harmful candidate 中，deployability deterioration 几乎覆盖绝大多数样本，gap deterioration 也很常见。这意味着模型没有识别“动作如何改变可执行恢复走廊”，而不是简单地没有把碰撞概率阈值设严。

Contact 不能通过以下手段修复：

- 降低/提高 admission threshold；
- 仅调 gamma、temperature；
- 增大 barrier penalty；
- Contact 单独设更严格 rule；
- 只增加 abstention。

这些方法最多改变选择量，不能把 AUC≈0.55 的错误排序变成正确排序。

### 6.2 CCF-A 投稿成熟度

Contact 目前远未达到主实验投稿程度。主要缺陷是：

- certificate safe-positive 命中为 0；
- harmful selection 数量高；
- 平均收益为负；
- action representation 方向欠识别；
- 现有 closed-loop 指标存在饱和，连续物理收益也很弱。

论文若现在投稿，Contact 很可能成为审稿人否定“跨正常—接触连续统一方法”主张的关键证据。

## 7. RC=20 的根本因果链

本轮把根因归纳为以下链条：

1. **候选支持存在。** top-5 proposal 有安全恢复候选，因此 candidate generator 不是第一瓶颈。
2. **compact evidence representation 信息不足。** 旧 bridge 主要看到少量 expert/scalar context，没有直接看到候选相对 nominal 的 prefix geometry、state evolution 和 control sequence。
3. **模型因此依赖场景相关性而非动作因果差异。** Near 在 dev 可学到局部排序但 transfer 很弱；Contact AUC 接近随机。
4. **soft safety geometry 可被补偿。** 高 raw benefit 或 admission residual 可以覆盖 component deterioration，导致 barrier 提高 coverage 的同时放大 harmful actions。
5. **checkpoint 与部署边界不完全一致。** 训练优化的是语义/软边界，最终却按各自 fitted rule 部署；hard-boundary 消融难以产生真实作用。
6. **分别拟合 rule 进一步掩盖表示失败。** 每个 stratum 的阈值可以吸收分布差异，却不能证明同一连续规划 primitive 成立。
7. **certificate 最终拒绝。** Near support 不足且 LCB 极低；Contact 无 safe-positive 命中。

根因不是单个超参数，而是“动作表示 + 非补偿安全几何 + 训练/部署契约 + 共享校准”四者没有同时闭合。

## 8. v48.35 CONTINUOUS-FRONTIER

### 8.1 物理相对动作表示

新增 `physical_relative` context。对同一 scene-time group 内候选，提取：

- prefix continuous parameters；
- macro one-hot/identity；
- prefix states；
- controls。

然后执行 `candidate - nominal`。

明确排除：

- absolute ego state；
- utility、hard violation、harm proxy、feasible、is_nominal、time 等 scalar/audit block；
- agents、map、BEV 等 scene-shared suffix；
- regime ID。

这样做的目的不是增加更多场景信息，而是恢复“这个动作相对 nominal 做了什么”的可执行物理差异。

### 8.2 非补偿连续安全前沿

五个 component logits 表示候选相对 nominal 的连续非退化 margin。令最坏 component 为安全 cap：

```text
c(a) = - max_k component_logit_k(a)
```

free admission 由 benefit 与 learned residual 给出：

```text
f(a) = benefit(a) + residual(a)
```

最终：

```text
admission(a) = smooth_min(f(a), c(a))
```

该 smooth-min 始终不高于任一输入。因此，只要任一 component 预测为越过零边界，最终 admission 就不能被大 benefit/residual 拉回安全侧。

### 8.3 部署 rule 的语义域约束

仅在模型内 cap 还不够。如果最终 score threshold 被拟合成负数，负安全 cap 仍可能通过。因此 v48.35 强制共享 rule 满足：

- opportunity threshold ≥ 0.5；
- harm threshold ≤ 0.5；
- score threshold ≥ 0；
- rank margin threshold ≥ 0。

这是本轮代码审计后新增的关键闭环，保证“非补偿”在部署端也成立。

### 8.4 单一共享 rule

Near/Contact adaptation-dev proposal rows 被合并，拟合一份四阈值 rule。每个 stratum 的 min selected、precision LCB、harmful group UCB、harmful selected UCB、macro share 都必须由同一 rule 同时满足。

输出中：

- `shared_rule_count=1`；
- `strategy_regime_conditioning=false`；
- `audit_strata_only=[near, contact]`；
- 两个 certificate worker 校验同一 JSON SHA256。

### 8.5 RC 语义

- shared dev rule 不满足约束：fitter RC=3，控制器继续保存诊断并最终形成算法 RC=20；
- certificate 不通过：RC=20；
- 文件缺失、缓存身份不符、共享 SHA 不一致、空 rows、checkpoint 不一致：RC=30；
- 只有 RC=0 生成 held-out test 命令。

## 9. 哪些设计保留、修改、降级

### 9.1 保留

- scene-disjoint train/dev/certificate protocol；
- test sealing 与 RC=0 才授权 test；
- top-5 proposal；
- 五个连续 component margins；
- candidate-vs-nominal 统一物理语义；
- natural-population training；
- hard-first checkpoint 思路；
- Safe nominal lock + paired non-inferiority；
- exact checkpoint/cache SHA 审计；
- proposal oracle feasibility 分析。

### 9.2 修改

- compact context → executable physical-relative context；
- soft compensatory barrier → non-compensatory frontier cap；
- separate Near/Contact rule → one shared rule；
- fixed diagnostic “exact” metric → deployed-rule metric；
- train/deploy boundary mismatch → semantic domain closed loop；
- “scene-crossfit”措辞 → 更准确地表述为 scene-fold worst-stratum checkpoint robustness，除非后续真的做 out-of-fold refit。

### 9.3 降级或删除主方法地位

- `barrier_gated_slack` 只保留为消融；
- regime-conditioned protective certificate 不应保留为主算法；
- 独立 Near/Contact threshold 不应进入主实验；
- legacy evidence-only 指标只能做历史诊断；
- 单纯 threshold/gamma/temperature tuning 不应作为下一轮主改动。

## 10. 值得继续升级的正向信号

1. Near candidate safe-positive AUC 达 0.81–0.87，说明已有机会识别基础。
2. Balanced Near certificate 出现一个 clean positive 且无 harmful，说明不是完全错误方向。
3. top-5 proposal oracle 支持充足，允许集中优化 selector，而不必先重写 candidate generator。
4. deployability、DRS、gap 三个 component 提供了 Contact 的明确诊断坐标。
5. natural-population、scene-disjoint、test sealing 和 cache contract 已形成较成熟实验基础设施。
6. v48.34 barrier 提高 coverage 的现象说明 admission 几何确实能控制激活量，只是需要非补偿安全约束和更好的动作表示。

## 11. 下一步实验决策

### 11.1 首先执行 v48.35 主实验

不要先跑 test，也不要先调 gate。主实验同时训练 Balanced/Precision，拟合一个共享 rule，并在独立 certificate 上验证。

### 11.2 若 RC=20，按 failure signature 决定 v48.36

- **Near AUC 上升、Contact AUC 仍≈0.55：** 增强 temporal executable physics，例如 prefix jerk/yaw-rate envelope、contact-frame relative velocity、future free-space gradient；仍然使用 candidate-minus-nominal，不加入 regime ID。
- **两者 AUC 都改善但 shared rule fit 失败：** 优化 group-level listwise/ranking 与 uncertainty calibration，不改单独阈值。
- **dev shared rule 通过但 certificate 失败：** 做真正的 scene-level invariant/OOF calibration、ensemble uncertainty 或 worst-scene reweighting；不读取 test。
- **certificate 通过但 closed-loop 无改善：** teacher target 与 rollout physical effect 不一致，需要候选级 temporal teacher，而不是继续调 selector。
- **Safe non-inferiority 失败：** 检查 nominal lock、checkpoint compatibility 与 candidate generation side effect；不能用 Near/Contact 收益抵消 Safe 退化。

### 11.3 2×2 消融

统一 shared rule 下比较：

| 组 | 表示 | Admission geometry |
|---|---|---|
| A | legacy relative | safety_slack |
| B | physical_relative | safety_slack |
| C | legacy relative | frontier_capped_slack |
| D | physical_relative | frontier_capped_slack |

该设计可分别回答：动作物理表示是否有效、非补偿 cap 是否有效、二者是否协同。不能再把 Near/Contact 分别调成两套策略。

## 12. CCF-A 投稿最低证据链

三个 regime 都达到投稿程度，至少需要：

1. one network + one shared rule；
2. Near/Contact scene-disjoint certificate 同时通过；
3. Safe paired non-inferiority 通过；
4. paired closed-loop 使用同 scene ID、同 target、同 horizon；
5. 多 seed 或 bootstrap CI；
6. 与强 baseline 的同源公平对比；
7. positive 与 failure videos 同时报告；
8. 论文中的 OC-MERO/CRISP 与代码 selector 明确对应；
9. 消融能隔离 observation consistency、physical-relative action identity、frontier cap、calibration；
10. 不用 test 选择 checkpoint、threshold 或算法版本。

## 13. 本地验证

本地完成：

- v48.35 新增测试：9 passed；
- 上传包实际支持的 core、v48.33、v48.34、v48.35、v50 compatibility 矩阵：158 passed，6 warnings；
- 55 个 shell 脚本 `bash -n`（最终打包前重新统计）；
- `compileall`；
- 关键 CLI `--help`；
- continuous frontier finite-gradient/cap/context contract；
- 生成命令依赖闭包；
- shared rule synthetic fit；
- Safe/stress authorization fail-closed contract。

全量历史 pytest 不能作为有效发布指标，因为上传 ZIP 缺少测试所引用的多份 v48.12–v48.32 历史脚本。交付版没有伪造这些历史脚本，而是把该问题记录为上传包完整性限制。

## 14. 最终判断

- **Near：** 有明确正向信号，但目前只是 promising，不是 CCF-A-ready。
- **Contact：** 当前结果远未达到投稿要求，核心是动作相对 nominal 的物理表示与安全排序方向欠识别。
- **RC=20：** 真实且有信息量，说明 v48.34 算法未通过，不应通过单独调阈值包装。
- **下一步最合理路线：** 保留统一连续 margin、top-5、scene-disjoint 和 protocol sealing；用 physical-relative action identity、non-compensatory frontier 和 one shared rule 修复主链；先执行 v48.35，再根据 RC signature 决定是否加强 temporal physical teacher 或 scene-invariant calibration。
