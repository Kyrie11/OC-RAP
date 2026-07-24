# OC-RAP v46 失败分析与 v47 OC-TRAC 优化说明

## 1. 结论

v46 的 `no checkpoint passed opportunity+score calibration` 不是单一阈值过严，而是 **工程链路错误、训练目标与部署决策错位、以及算法可辨识性不足共同导致**。继续降低 calibration threshold 会放大误接纳，不能解决问题。

本轮实现 v47 **OC-TRAC（Observation-Consistent Tri-state Risk-Calibrated Recovery Admission Certificate）**。它不替换论文的 OC-MERO/CRISP 核心，而是把“候选恢复是否值得从 nominal 切换”改造成一个与论文核心指标 PCD、FRA 和部署风险一致的三状态选择问题：

1. 正恢复增益：候选相对 nominal 的 teacher PCD 增益大于正阈值；
2. dead-zone/tie：增益接近零，不强迫其成为负 margin；
3. 有害切换：候选相对 nominal 的 teacher PCD 显著下降。

部署时只允许通过 **opportunity + benefit + harm-veto + macro support + physical actionability** 的 setwise top-1 候选，否则 abstain 到 nominal。

## 2. v46 实验事实

### 2.1 Near-contact

| 变体 | 全候选相关性 | 候选正例 AUC | 不加门限的 group top-1 相关性 | top-1 teacher advantage 均值 | top-1 有害率 | 结果 |
|---|---:|---:|---:|---:|---:|---|
| balanced | 0.1242 | 0.6725 | -0.0792 | -0.1618 | 31.2% | 无可行 fit rule，verify 0 selections |
| precision | 0.1343 | 0.6652 | -0.0701 | -0.1559 | 30.4% | 无可行 fit rule，verify 0 selections |

候选级 AUC 高于随机说明网络并非完全没有信号；但部署规则是在每个 scene-time 候选集合中取 top-1，集合内排序却是弱负相关。因此失败不是“门限略高”，而是 **pointwise 可分性没有转化为 policy-level setwise 排序能力**。

Near 的 teacher advantage 中位数和 75% 分位均为 0，大量候选属于 tie。v46 虽修复了 v45 的 tie-as-negative，但稀有正组没有被采样器增强：日志明确显示 `positive_advantage_groups=0`、`positive_advantage_boost=0`、`replacement=false`。训练仍被 tie 和负例主导。

### 2.2 Contact

| 变体 | fit selections / precision | verify selections / precision | verify 有害率 | verify teacher advantage 均值 | 结果 |
|---|---:|---:|---:|---:|---|
| balanced | 20 / 55.0% | 25 / 48.0% | 44.0% | -0.1333 | fit 到 verify 明显崩溃 |
| precision | 6 / 100% | 12 / 66.7% | 33.3% | -0.0465 | 低样本规则过拟合；全局相关性未过门 |

Contact 的可行规则主要依赖 macro 5（merge）。fit fold 看起来很好，但 verify 中误接纳明显增加，说明规则学到了有限样本的 macro/score 偏差，而非稳定的恢复增益。Contact validation 只有 70 个原始 scene，不能支撑论文 final contract 所需的独立、稳定、足量 calibration。

## 3. 工程层面的确定问题

### 3.1 v46 仍初始化于旧 v39 checkpoint

两个结果包的 `train_summary.json` 都记录：

```text
runs/ocrap_v39_ocrac_balanced/model_v39_ocrac/best.pt
```

而不是新建的 clean-base run。大量 encoder、root decoder、option embedding 和 certificate heads 被冻结。结果是 v46 新 head 在旧表示上拟合稀有的相对 PCD 事件，上限被旧 backbone 限制。

v47 主脚本在训练结束后强制比较 `init_checkpoint` 与本轮 clean-base 的绝对路径，不一致立即退出。

### 3.2 稀有正组采样逻辑实际未工作

v46 日志：

```text
replacement=False
positive_advantage_boost=0.0
positive_advantage_groups=0
```

旧采样器即使启用也使用 `r_dep` proxy，而 calibration 评价的是 teacher PCD。v47 改为逐样本计算与 OC-MERO/DRS/ODG 相同的 teacher PCD，按 `PCD(candidate)-PCD(nominal)` 识别正组，并启用 replacement 权重采样。日志会输出 `positive_advantage_target=pcd` 和计算失败数。

### 3.3 Safe/Near 闭环目标被 split 过滤为空

开发阶段传入的是 `val_safe`、`val_near_contact`、`val_contact`，但原执行函数硬编码：

```text
closed_loop.bucket_split=test
```

`_load_closed_loop_targets` 会严格比较每个样本的 `split_id`，因此 val roots 中所有样本都被过滤。这是 Safe/Near 没有闭环输出的直接工程原因之一。

v47 新增 `SAFE_BUCKET_SPLIT`、`NEAR_BUCKET_SPLIT`、`CONTACT_BUCKET_SPLIT`，开发阶段显式设为 `val`，held-out 阶段显式设为 `test`；并新增回归测试验证 val target 能被加载。

### 3.4 Natural gate 失败导致整个闭环链提前退出

旧流程把“learned checkpoint 是否可部署”和“数据集是否能计算闭环指标”绑在一起。校准失败后直接退出，所以 Safe nominal reference、Near nominal reference 也不会运行。

v47 将两类输出分开：

- **learned-policy closed loop**：必须通过证书才运行；
- **certificate-independent nominal reference closed loop**：即使 learned gate 失败也可运行，覆盖 Safe/Near/Contact，作为物理基准，但不得冒充 OC-TRAC 的学习策略结果。

### 3.5 数据重建脚本没有等待 Safe/Near workers

`rebuild_ocrap_val_test_regimes.sh` 的说明要求每块 GPU 同时只运行一个 Waymax/JAX worker，但 Safe 与 Near 两对 worker 后的 `wait_pair` 被注释，PID 又被下一对任务覆盖。结果可能是六个构建并发竞争显存/编译缓存，且主脚本只等待 Contact workers；这与用户观察到 Near/Contact val/test 尚未完整构建一致。v47 恢复 Safe -> Near -> Contact 的逐对等待，并保留失败日志 tail。

### 3.6 恢复候选前沿在质量裁剪前缺少有效多样性

原 prefix generator 的重复 `merge` 每两个 variant 就出现参数重复，`stabilize` 的所有 variant 完全相同；而 stress 数据最终每个 group 只保留 8/9 个候选，很多预算先分给不进入 direct certificate 的 keep/lane-shift/perturb。v46 的正机会高度集中在 merge，说明 candidate frontier 本身也是上限因素。

v47 新增 `prefix_macro_schedule`：按 macro 独立计数 variant，并在 Near/Contact 重建脚本中优先生成多组 `merge/brake/stabilize/yield`。同时让 merge、stabilize、yield 的 variant 参数真正不同。该变化需要重建 stress 数据后才生效，不能通过只重训旧 NPZ 获得。

### 3.7 Stress bucket 与闭环 raw WOMD 来源不一致

同步的数据重建脚本明确从 standard validation 构建 Near/Contact，以避免 interaction-mined source 引入分布偏差；但旧闭环脚本固定扫描 `validation_interactive`。即使 `split_id` 正确，也可能无法匹配任何 scene id。v47 新增 `WOMD_STRESS`，默认等于 `WOMD_VAL`，与同步重建契约一致。只有旧 bucket 确实由 interactive 数据构建时，才显式设置 `WOMD_STRESS=/.../validation_interactive_tfexample.tfrecord`。

### 3.8 预测 harm 与数据 harm proxy 键冲突

校准草案中同名 `harm` 字段曾先存网络预测、再被 `harm_proxy` 覆盖。v47 已分离为：

- `predicted_harm`：学习的误接纳概率；
- `harm_proxy`：数据/物理约束代理。

并添加 harm-veto 单元测试。

## 4. 算法层面的缺陷

### 4.1 隐藏 regime router 不可辨识，也不符合论文核心论点

v46 router accuracy 约 0.52–0.56，接近随机。更重要的是，论文强调 observation-indistinguishable futures 不能依赖隐藏分支身份选择恢复动作；若算法卖点变成“从观察猜 Near/Contact，再选专家”，会削弱核心论证。

v47 移除默认隐藏 regime routing。两个专家都看全部 stress 数据，分别采用互补的风险权重：

- recovery-seeking expert：强调正增益召回和 setwise opportunity；
- harm-averse expert：强调 false-admission/harm 抑制。

部署采用候选无关的均匀权重，并以专家分歧构造保守证书：benefit/opportunity 使用均值减分歧，harm 使用均值加分歧。专家是“风险态度假设”，不是 oracle regime classifier。

### 4.2 Pointwise loss 与 setwise policy 不一致

v46 候选 AUC 尚可，但每组 top-1 失败，说明需要直接训练“选候选或 abstain”的集合决策。v47 增加：

- nominal-relative pair loss；
- tri-state opportunity/harm losses；
- deployable macro 集合内 listwise admission CE；
- setwise abstention：没有正恢复候选时 nominal 是正确类别；
- checkpoint selection 使用 Near/Contact worst-regime validation objective。

### 4.3 没有独立的有害切换预测

仅预测 gain/opportunity，无法区分“低置信收益”和“明确伤害”。v47 新增 harm head，并在校准和 selector 中使用显式 harm upper gate。这样风险控制分母是“实际会执行的动作”，不会被低 coverage 的全组分母稀释。

### 4.4 反复调 rescue 手工门限作用有限

现有多层 `brake_tail`、`challenge rescue`、macro-specific rescue 对少量样本可提高 coverage，但不能修复 score 与 teacher 的集合内排序关系。v47 仍保留旧机制用于兼容和消融，但新主路径不以放宽 rescue 阈值制造 gate 通过。

### 4.5 不再默认恢复 raw flattened action adapter

v42 已显示 raw flattened states/controls 没有准确率证据且可能增加噪声。v47 默认仍使用结构化 token 的 `candidate_concat`，raw adapter 仅保留为显式消融，不作为主配置。

## 5. v47 OC-TRAC 设计与论文核心的关系

论文核心不是一般的 MoE、普通 contingency planning 或单纯碰撞风险最小化，而是：

> 一个 branch-wise 可恢复的候选，只有当恢复决策能由执行候选后的 observation 选择，并且在 observation-equivalent roots 内保持兼容时，才是 deployably recoverable。

OC-MERO 继续计算 observation-consistent deployable recoverability；OC-TRAC 只解决随后一个必要问题：当 nominal 和 recovery candidates 都已被 OC-MERO 评价时，如何在稀有正机会、tie 和有害切换共存的情况下，给出可校准的“切换或 abstain”证书。

因此论文中建议把 v47 描述为：

- **核心规划原语**：OC-MERO + oracle-to-deployable gap；
- **部署选择器**：CRISP；
- **选择性恢复接纳扩展**：OC-TRAC tri-state risk certificate。

不要把“两个专家”本身写成主 novelty；MoE 规划已有相关工作。真正可防守的新增点是：**针对 observation-consistent recovery affordance 的 policy-level、setwise、tri-state、risk-calibrated selective admission**。

## 6. 闭环指标支持结论

上传的 reports 显示 Safe/Near/Contact 的 val/test roots：

- `schema.missing_field_samples=0`；
- `future_generation.waymax_runtime_fraction=1.0`；
- 均含 `scene_id` 与 `time_index` 对应关系；
- `supports_waymax_runtime_claim=true`。

因此数据的离线样本 schema 支持回查 WOMD 场景并运行 Waymax。真正运行前仍需确认本机 WOMD TFRecord 中能够匹配这些 `scene_id`；v47 的 preflight 会检查路径与 metadata，runner 会记录实际 matched/missing targets。

### 6.1 三个 regime 统一输出的物理闭环指标

v47 新增或显式汇总：

- collision scene/step rate；
- offroad scene/step rate；
- minimum clearance；
- minimum TTC；
- path length、net displacement、progress efficiency；
- longitudinal acceleration mean/max；
- hard-brake rate；
- jerk mean/max；
- yaw-rate mean/max；
- intervention rate 与 bounded NUP。

Near/Contact 另外继续输出 FRA、DRS、ODG 等 recovery-specific 指标。Safe 本身没有 oracle-artifact stress 构造，因此报告中 `supports_fra=false`、`supports_odg=false` 是语义上合理的；Safe 用物理安全、舒适、进度和 nominal preservation 做对照。

Route-following/rejoin 指标依赖 WOMD 1.3.1 的 `sdc_paths/path_samples` 及 `closed_loop.use_sdc_paths=true`；没有该字段时，route-free 指标仍可完整计算。

## 7. 三 regime 的理论改进路径

### Safe

- 保持 nominal lock，不让稀有恢复 head 改写正常驾驶；
- 以 collision/offroad、comfort、progress、NUP 证明不退化；
- v47 的 Safe 闭环不再被 stress gate 阻断。

### Near-contact

- 精确 PCD 正组增采样提高 rare opportunity 学习概率；
- setwise abstention 解决候选 AUC 与 top-1 失配；
- harm head 抑制 30% 左右的 top-1 有害候选；
- 专家分歧惩罚对观察证据不足的候选更保守。

### Contact

- 不再依赖 macro 5 的单一 score 阈值；
- risk-attitude experts 与 harm upper gate 降低 fit→verify 误接纳；
- final contract 仍需更多独立 Contact scenes，否则统计 UCB 无法稳定收紧。

这些改动提高结果的理论合理性，但不能在未运行新实验前保证 SOTA。若 v47 仍无可行规则，应首先检查 candidate frontier 中是否存在足够多的正 PCD recovery，而不是继续降低 gate。

## 8. 必须做的消融

1. v46 原版；
2. + exact PCD sampler；
3. + tri-state tie/harm；
4. + setwise abstention；
5. + harm head/veto；
6. + all-stress asymmetric experts；
7. + disagreement certificate；
8. full OC-TRAC。

此外报告 hard regime router、uniform robust aggregation、single head 的对比，证明提升不是来自泄露 regime 标签。

## 9. 验证状态

- 全量单元测试：110 passed；
- 新增 split regression、harm veto、tri-state calibration、PCD sampler、route-free metrics 测试；
- Python compileall 与 shell syntax 在打包时重新执行；
- 未修改论文 tex、已有实验事实、图表或引用。

## 10. 仍然存在的投稿阻塞项

1. reports 中 calibration split 仍为空，final contract 需要独立 calibration roots；
2. Contact val/test scene 数量分别约 70/74，难以满足严格风险 UCB 与多随机种子统计；
3. 需要完整、同数据、同候选预算、同闭环 horizon 的外部 baseline；
4. 需要至少 3 seeds，报告均值、置信区间和 paired scene bootstrap；
5. nominal reference 闭环不能代替 learned OC-TRAC 闭环结果。
