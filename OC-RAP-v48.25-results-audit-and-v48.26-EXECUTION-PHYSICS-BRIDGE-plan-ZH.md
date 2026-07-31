# OC-RAP v48.25 结果审计与 v48.26 EXECUTION-PHYSICS-BRIDGE 优化方案

## 结论摘要

v48.25 的 `RC=30` 不是算法退化，也不是 Natural gate 拒绝。Balanced 与 Precision 均完成训练并生成 checkpoint；四个 Near/Contact certificate worker 也都完成了数据加载和模型打分。流水线在写最终 JSON 时才因 `pathlib.PosixPath` 无法被 `json.dumps` 序列化而崩溃。因此：

- `gate_evaluated=false`；
- 没有任何 v48.25 Natural-gate 结果；
- 不能根据本轮 RC=30 判断 certificate 是否可行；
- 不能把缺失的 certificate JSON 当成模型没有机会或 gate failed。

进一步代码审计发现，单修 JSON 仍不足以得到可信结论。v48.25 训练入口与 checkpoint 推理入口构造了不同的模型语义；checkpoint 指标也没有使用 Natural gate 的 safe-positive 定义。v48.26 首先修复这些会污染算法归因的工程问题，然后才保留能够被实验验证的算法修改。

---

## 1. RC=30 的首个根本原因

### 1.1 实际执行进度

`PIPELINE_FAILED.json` 表明：

- failure stage：`certificate`；
- Balanced adaptation：`RC=0`；
- Precision adaptation：`RC=0`；
- gate 未评估；
- test/stress 未读取。

四个 worker 在异常前均已经完成：

- Near：读取 2,412 个样本，形成约 305 个 scene-time group；
- Contact：读取 6,929 个样本，形成约 780 个 scene-time group；
- adaptation-dev rule 搜索也已经执行。

最终异常为：

```text
TypeError: Object of type PosixPath is not JSON serializable
```

触发位置是 `tools/calibrate_policy_risk_v48.py` 将 `vars(args)` 写入结果 JSON。`args.frozen_rule_json` 是 `Path`，导致所有 policy certificate 文件在最后一步写盘失败，随后 controller 又将“缺少最终文件”报告为 pipeline failure。

### 1.2 正确归类

这是确定的工程错误。它不支持以下任何结论：

- v48.25 算法变差；
- certificate 拒绝了模型；
- proposal support 不足；
- learned gate failed；
- Near/Contact 无法通过 gate。

v48.26 对 `Path`、NumPy scalar/array、tuple/set 和嵌套字典执行递归 JSON-safe 转换，并增加单元测试。结构支持不足或 learned rule 拒绝返回 worker `RC=3`，最终 controller `RC=20`；只有空 population、损坏 checkpoint、协议/索引异常才返回 `RC=30`。

---

## 2. 会污染算法归因的其他工程问题

### 2.1 训练模型与 certificate 推理模型不一致

v48.25 训练构造器已经传入：

```text
direct_recovery_evidence_frontier
direct_recovery_evidence_component_prior_logit
direct_recovery_evidence_admission_bounded
```

但 `src/ocrap/models/inference.py` 漏传这三个字段。结果是：

- 训练时可使用 semantic low-risk prior；
- certificate 推理却可能回到 legacy frontier 语义；
- 训练时 admission 可以 unbounded；
- certificate 推理仍使用默认 bounded admission；
- 同一 checkpoint 在训练验证和 certificate 中不是同一个算法。

因此，即使 JSON 写盘不报错，v48.25 certificate 结果仍不能用于算法归因。

v48.26：

1. checkpoint 显式保存上述字段；
2. inference 按 checkpoint 优先恢复；
3. 推理完成后比较 expected/actual contract；
4. controller 在 certificate 前运行 `check_v48_26_model_contract.py`；
5. 任一不一致立即 fail-closed，返回 RC=30。

### 2.2 checkpoint 的“正机会命中”定义与 gate 不一致

v48.25 的 checkpoint 指标存在两个问题：

1. 分母使用完整 group 中的 raw-benefit opportunity，而 gate 使用 proposal-contained safe-positive；
2. 只要动作有收益，即使该动作 harmful，也可能被记为 positive admission hit。

这会让 checkpoint 指标夸大模型的 safe admission，并选择错误 epoch。

v48.26 将机会定义统一为：

```text
冻结 proposal 内存在 teacher_advantage >= positive_gain
并且 component_harmful = false 的 recovery action
```

新增 checkpoint 指标：

- `direct_safe_positive_admission_recall_*`；
- `direct_safe_admission_precision_*`；
- `direct_invalid_admission_rate_*`；
- `direct_evidence_safe_top1_accuracy_*`；
- `direct_evidence_safe_top1_regret_*`。

完整性 checkpoint metric 直接惩罚 safe recall shortfall、低 precision、invalid admission 和 safe top-1 regret。

### 2.3 safe-utility 的训练尺度与运行时尺度不一致

v48.25 safe-utility regression 使用：

```text
tanh(admission_delta / 2)
```

运行时 selector 使用：

```text
sigmoid(admission_delta) - 0.5
```

数学上前者是后者的两倍。虽然 threshold calibration 能部分吸收尺度差，但 listwise/regression 学到的并不是部署时真正使用的分数。

v48.26 训练直接使用 `sigmoid(x)-0.5`，teacher target 也裁剪到同一 `[-0.5, 0.5]` 执行范围。

### 2.4 calibration split alias 边界不够严格

旧通用角色：

```text
calibration -> {calibration, certificate_pool}
```

本轮实际目录没有产生 Safe/certificate 串读，但该设计允许未来在混合目录中误读 certificate population。v48.26 增加 `calibration.exact_split_ids=true`：

- Safe calibration 只读 `split_id=calibration`；
- Near/Contact certificate 只读 `split_id=certificate_pool`；
- adaptation-dev rule 只读 `split_id=evidence_adapt_dev`。

### 2.5 Contact 物理指标的 bucket 判断存在潜在误分类

初版 v48.26 扩展物理指标时若使用字符串包含关系，`near_contact` 也会被识别为 contact。最终交付版已改为精确 alias 集并增加测试：

- `near_contact`、`test_near_contact` 永远不是 post-contact；
- `contact`、`post_contact`、`post_collision` 等才使用 post-contact anchor。

---

## 3. v48.25 是否改善了 safe admission

### 3.1 能够使用的证据范围

由于 gate 未评估、certificate 推理合同错误，当前只能查看训练过程内部的 adaptation-dev 指标。它们可以用于诊断，但不能等价于独立 certificate 结论。

### 3.2 与 v48.24 相比的局部变化

v48.24 的两个 best checkpoint 都是绝对全 abstain：Near/Contact raw admission 和 positive admission recall 都为 0。

v48.25：

| Variant | Near raw admission | Contact raw admission | Near raw-positive recall | Contact raw-positive recall |
|---|---:|---:|---:|---:|
| Balanced | 0.0084 | 0.0345 | 0 | 0.0357 |
| Precision | 0.0084 | 0.0483 | 0 | 0.0357 |

因此 v48.25 的 admission 修改有一个非常有限的正向作用：**打破了绝对全 abstain**。但这不是 safe admission：

- Near 正机会准入仍为 0；
- Contact 只命中约 3.6% 的 raw-positive groups；
- 当时的指标没有要求选中动作 non-harmful；
- false intervention 和 harmful switch 相比绝对 abstain 自然上升。

### 3.3 被 checkpoint 丢弃的 Contact 正向信号

Balanced Contact 在训练后期出现局部正向信号：

- evidence 正机会 top-1 accuracy：约 0.321 → 0.500；
- positive top-1 regret：约 0.265 → 0.204；
- harmful switch 和 false intervention 在后期下降。

但 Balanced best checkpoint 被选为 epoch 0。说明 v48.25 的 checkpoint metric 没有把“安全正机会排序改善”作为主目标，反而丢弃了可能有价值的后期模型。

Precision 的改善更弱且不稳定：Near/Contact admission 后期重新趋向 0，Contact 正机会排序没有形成持续增益。

### 3.4 Near 结论

Near 的强 raw-benefit 信号仍未转成安全准入：

- best checkpoint 的 Near raw-positive recall 为 0；
- proposal-contained safe-positive recall 当时没有被正确计算；
- Near dev safe-positive 只有 9 groups、5 scenes，其中 8/9 的最佳机会集中在 macro 2；
- checkpoint 选择不能区分“正确 abstain”与“错过全部安全机会”。

所以不能说 Near 问题已经改善。

### 3.5 Contact 结论

Contact 有局部排序学习信号，但没有建立可证书化策略：

- top-1 accuracy 的后期改善只出现在 Balanced；
- best checkpoint 没有保留该改善；
- raw-positive recall 极低；
- 没有独立 certificate precision/harm UCB；
- 没有有效 shadow closed-loop 物理结果。

结论是：**部分算法分量可能有效，但 v48.25 不能证明 Contact safe admission 已建立。**

---

## 4. certificate 设计是否合理

### 4.1 应保留 certificate

certificate 的职责不是让模型通过，而是回答：

> 已在 adaptation-dev 冻结的 selector，在 scene-disjoint population 上是否以足够支持、足够 precision、足够低 harmful UCB 选择动作？

删除 certificate 只会允许模型在没有独立风险证据时读取 test/stress，无法解决 Near/Contact 的物理问题。

### 4.2 推荐协议

v48.26 保留 v48.25 的基本方向，并明确为：

1. 在 `evidence_adapt_dev` 拟合 opportunity/harm/score/rank thresholds；
2. 冻结完整 rule 和 SHA256；
3. 完整 `certificate_pool` 仅执行 verification；
4. certificate 标签绝不反向修改 rule；
5. structure support 或 learned gate 拒绝返回 RC=20；
6. 工件/协议异常返回 RC=30。

这比把稀少 certificate 再拆为 fit/verify 更合理。

### 4.3 diagnostic rule 的边界

若 adaptation-dev 没有任何 rule 满足开发约束，工具可以保存 `diagnostic_fit_rule` 供 shadow 定位，但必须记录：

```text
source_rule_satisfied_dev_constraints=false
```

独立 certificate 即使验证该 deterministic rule，也不能把它描述成“开发阶段已经满足约束”。v48.26 已将该 provenance 写入结果。

### 4.4 Safe regime 的现实状态

目前没有注册独立、scene-disjoint 的 Safe policy certificate population。因此 Safe 不能声称通过与 Near/Contact 相同的 Natural gate。当前合理合同是：

- Safe `gamma_rec` 标准校准；
- nominal-first selector；
- held-out Safe paired non-inferiority；
- intervention、route progression、jerk、yaw-rate、collision/offroad 不劣化。

v48.26 明确写出 `SAFE_REGIME_STATUS.json`，避免把标准阈值校准误称为 Safe policy gate。

### 4.5 投稿用 certificate

当前 certificate 已被多轮查看，只能继续作为 development certificate。最终 CCF-A 论文结果应使用新封存、预注册、scene-disjoint 的 certificate population，或将当前 population 明确标为 development 并另设 final confirmation split。

---

## 5. Near/Contact 当前算法层面的关键缺陷

### 5.1 稀少且场景集中的 safe-positive 监督

训练集支持：

| Regime | Safe-beneficial candidates | Groups | Scenes |
|---|---:|---:|---:|
| Near | 25 | 11 | 7 |
| Contact | 106 | 41 | 17 |

adaptation-dev：

| Regime | Safe-beneficial candidates | Groups | Scenes |
|---|---:|---:|---:|
| Near | 25 | 9 | 5 |
| Contact | 63 | 23 | 8 |

Near 特别稀疏并集中于少数 scene/macro。当前不重建数据集的前提下，只能通过 exact group stratification、scene balancing、safe-positive checkpoint 和多 seed 判断稳定性；不能把单 seed 小样本改善包装成强泛化结论。

### 5.2 admission 与 ranking 是两个不同问题

v48.25 已能产生少量非 nominal admission，但：

- Near 没有命中正机会；
- Contact 命中率极低；
- 后期排序改善没有被 checkpoint 选择。

这说明“让 admission logit 能跨过 0”并不足够。必须同时满足：

1. proposal 内选对 safe-positive action；
2. harmful beneficial action 被 veto；
3. nominal-vs-recovery admission 有正确尺度；
4. checkpoint 优先保留 safe-positive recall/precision；
5. certificate 用同一执行分数验证。

v48.26 的 safe-positive metric 和 execution-exact safe utility 正是针对这条链路。

### 5.3 Contact 的离线 teacher 目标没有充分表达时间过程

当前离线 PCD 主要为：

```text
DRS × sigmoid(R_dep) × exp(-gap)
```

它能表达 deployable recovery headroom，但没有直接区分：

- 一次短 overlap 与持续 overlap；
- 初次脱离后重新接触；
- 迅速建立自由空间与缓慢漂移；
- 持续 escape 与瞬时 clearance spike；
- 稳定停车与低速但高 yaw/offroad 的不稳定状态。

这很可能是 Contact 反复出现“风险 AUC 尚可，但收益相关性、top-1 regret 和闭环恢复不稳定”的原因。

由于现有 NPZ/teacher index 不包含这些 candidate-level closed-loop physical labels，本轮没有伪造辅助训练标签。v48.26 先把它们完整输出到 dev-shadow/held-out closed loop。若工程干净后出现“离线 safe ranking 改善、物理闭环不改善”，下一步应在 adaptation-dev 上预注册 candidate-level physical teacher rollout，构造：

```text
contact_physical_gain =
  w1 * (- secondary/recontact)
+ w2 * (- overlap duration)
+ w3 * free-space AUC
+ w4 * sustained escape
+ w5 * (- time-to-escape)
+ w6 * stable-stop quality
```

再作为独立 auxiliary head，而不是事后修改 certificate 标签。

### 5.4 Near 的有效性不能只看 minimum clearance/TTC

仅看 minimum 值容易被单帧噪声或短时动作影响。Near 还需要：

- exposure episode count；
- longest continuous exposure run；
- time-to-min clearance/TTC；
- terminal clearance/TTC；
- 从最危险点到 rollout 结束的 recovery gain；
- clearance/TTC deficit AUC；
- collision/overlap 保持 0；
- route progression、jerk、yaw-rate 和 intervention burst 不劣化。

v48.26 已补齐这些输出。

---

## 6. v48.26 EXECUTION-PHYSICS-BRIDGE 的代码修改

### 工程完整性

- 修复 certificate JSON serialization；
- 修复训练/推理模型构造不一致；
- checkpoint 保存完整 evidence contract；
- certificate 前 fail-closed model-contract preflight；
- 精确 split ID 防止校准/certificate 串读；
- 正确区分 RC=20 与 RC=30；
- 提供无需重训的 v48.25 certificate repair-only 诊断路径。

### 算法合同

- checkpoint 使用 proposal-contained safe-positive；
- harmful action 不能计入 positive admission；
- 增加 safe precision、invalid admission 和 safe top-1 regret；
- safe-utility 使用运行时精确分数；
- 保留 nominal + top-k categorical one-action objective；
- legacy Noisy-OR 保持关闭；
- proposal 默认 top-3；
- Near/Contact exact safe-positive stratified group sampling 保持开启。

### Contact 物理输出

新增/修正：

- causal contact anchor；
- re-contact episode/event/scene rate；
- overlap episode、duration、longest run；
- normalized free-space AUC；
- clearance deficit AUC；
- terminal clearance、clearance gain、time-to-peak；
- sustained escape、time-to-escape；
- stable-stop quality：同时约束 speed、overlap、offroad、yaw-rate；
- time-to-stable-stop quality。

### Near 物理输出

新增：

- near/critical exposure episode count；
- longest exposure run；
- time-to-min clearance/TTC；
- terminal clearance/TTC；
- clearance/TTC recovery gain；
- acceleration/deceleration/jerk/yaw-rate extrema。

### 消融

四组 × Balanced/Precision，共 8 个任务，四个 wave：

1. `A_engineering_contract_only`：只验证工程合同；
2. `B_add_safe_checkpoint_contract`：增加 gate-exact checkpoint；
3. `C_add_execution_exact_safe_utility`：增加执行尺度 safe utility；
4. `D_full_execution_physics_bridge`：完整模型。

每个 wave：Balanced 独占 GPU0，Precision 独占 GPU1。最大并发 2，每张 A30 总计四个任务。

---

## 7. 下一轮判读顺序

### 第一步：repair-only，不重训

使用服务器上保留的 v48.25 checkpoint，以修复后的 inference/JSON 重新跑 certificate。目的只是回答：

- 旧 checkpoint 在一致模型语义下是否能产生完整 certificate；
- RC 应是 0、20 还是 30；
- 不能验证新的 checkpoint/safe-utility 修改。

### 第二步：完整 v48.26 主实验

先检查：

- `MODEL_INFERENCE_CONTRACT.json` 全部通过；
- best checkpoint 的 Near/Contact safe-positive recall；
- safe admission precision；
- invalid admission rate；
- evidence safe top-1 regret；
- dev rule 的 source provenance。

再查看 full certificate：

- population/support 是否有效；
- verify selected count；
- precision LCB；
- harmful group/selected UCB；
- macro concentration。

### 第三步：RC=20 的定位

- safe recall 仍为 0：admission/ranking 或 checkpoint 仍失败；
- dev 好、certificate 差：泛化/样本支持问题；
- offline 好、shadow 物理不改善：PCD teacher 与物理恢复目标错位；
- Contact 物理改善而 gate failed：证书支持/有限样本保守性问题，需要新预注册协议，不能事后降低当前 gate。

### 第四步：RC=0

仅运行自动授权的 Safe paired 和 stress/held-out closed loop，然后检查论文目标。不得手工创建 `NEXT_COMMANDS.txt`。

---

## 8. 本地验证范围

当前环境没有 WOMD/Waymax 数据和 A30，因此没有声称 v48.26 已通过 gate或达到闭环目标。完成的本地验证：

- `pytest`：233 passed，5 warnings；
- `compileall`：通过；
- 所有 Shell `bash -n`：通过；
- certificate JSON-safe、model parity、safe-positive metric、execution-exact score、exact split IDs、Near/Contact bucket 区分和物理字段均有测试覆盖。
