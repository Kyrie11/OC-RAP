# OC-RAP v48.17 结果审计与 v48.18 DUET-BRIDGE 优化方案

## 1. 审计结论

本轮需要把三个状态严格区分：

1. **工程失败（RC=30）**：没有完成独立 certificate，因此不能判断 Natural gate。
2. **有效 certificate 但 gate 拒绝（RC=20）**：可以进行算法归因，但不能运行 test/stress closed loop。
3. **有效 certificate 且 Near、Contact 联合通过（RC=0）**：生成 `NEXT_COMMANDS.txt`，才允许运行 stress closed loop。

本次上传包未包含 `runs/ocrap_v48_17_bridge_dedicated_4817` 主实验目录，因此无法直接读取主实验 controller 日志；但当前主脚本调用的正是上传消融 B/C 所使用的同一适配脚本，而 B/C 四份日志均可重复观察到训练完成后被 78,630/20,000 参数 guard 拒绝。结合主控制器的确定性早退分支，可以判定用户描述的主实验属于第一类：它不是因为 Natural gate 被拒绝而缺失文件，而是 post-check 把有效 checkpoint 判成失败，controller 在 certificate 之前退出。

能够被正式判定的只有 `A_simplex_scalar` 两个消融任务：它们完成了非空、scene-disjoint certificate，并被 Natural gate 真实拒绝。`B_context_simplex` 与 `C_full_bridge` 已训练完成，但没有进入 certificate，因此不能宣称通过或失败。

---

## 2. `learning_gates_v48_17.json` 为什么没有生成

### 2.1 直接原因

`adapt_ocrap_v48_17_bridge_variant.sh` 在训练完成、`best.pt` 和 `train_summary.json` 已经存在后，执行了如下后验检查：

```python
trainable = sum(
    v.numel()
    for k, v in state.items()
    if k.startswith("direct_evidence_calibrators.")
)
if trainable <= 0 or trainable > 20000:
    raise SystemExit(...)
```

v48.17 默认启用 raw relative context。实际校准器参数量为：

```text
78,630
```

因此 Balanced 和 Precision 的日志都以如下错误结束：

```text
unexpected BRIDGE calibrator parameter count: 78630
```

这不是 CUDA OOM、训练崩溃或 checkpoint 损坏。训练已经完成，失败发生在训练后的 marker 写入阶段。

### 2.2 控制流原因

`run_v48_17_bridge_dedicated.sh` 原逻辑是：

```text
Balanced adaptation ┐
                    ├─ 两者都非0 → 写 PIPELINE_FAILED.json → exit 30
Precision adaptation┘

只有至少一个 adaptation 返回0时：
    → calibrate_v48_16_certificate_pool.sh
    → check_v48_16_learning_gates.py
    → learning_gates_v48_17.json
```

两个 variant 都因为参数上限返回非零，导致脚本在 gate checker 之前退出。因此 `learning_gates_v48_17.json` 没有生成的根因是：

> **gate 汇总代码位于 adaptation 双失败的早退分支之后。**

它不代表 gate 失败，也不代表 certificate 是空的；certificate 根本没有执行。

### 2.3 已完成的工程修复

- v48.17 的参数检查改为可配置：

```bash
MAX_EVIDENCE_CALIBRATOR_PARAMS=100000
```

- 主控制器即使 RC=30，也会生成：

```text
learning_gates_v48_17.json
V48_17_COMPLETE.json
PIPELINE_FAILED.json
```

- `learning_gates` 报告新增：

```text
pipeline_failed
adaptation_failures
certificate_data_valid
natural_gate_passed
```

以后不会再通过“文件是否存在”猜测失败阶段。

- 新增 `recover_v48_17_after_param_guard.sh`，可直接利用现有 `best.pt` 补写完成标记并运行 certificate，无需重新训练 v48.17。

---

## 3. `NEXT_COMMANDS.txt` 为什么没有生成

`NEXT_COMMANDS.txt` 由 dedicated certificate controller 在以下条件全部满足时生成：

1. checkpoint、policy contract、标准 calibration 和 risk JSON 均存在；
2. Near、Contact certificate 均非空；
3. fit/verify scene-disjoint；
4. 至少一个 variant 的 Near 和 Contact 均满足 `valid_for_deployment=true`。

原主实验没有进入 certificate，因此不可能生成该文件。

需要强调：修复工程逻辑不能直接“补造” `NEXT_COMMANDS.txt`。正确解决路径是：

```text
先恢复 v48.17 certificate
        ↓
RC=0：真实通过，自动生成 NEXT_COMMANDS.txt，运行 stress
RC=20：真实未通过，不允许绕过，进入 v48.18
RC=30：仍有工程问题，先修工程
```

本次代码保证：只要 gate 真正通过，文件一定生成；如果未通过，则不会通过伪造授权让 test/stress 泄漏。

---

## 4. 当前 Natural gate 是否通过

### 4.1 v48.17 主实验

**尚未被有效评估。**

因为 Balanced/Precision 均在 post-check 阶段被误判失败，没有 dedicated risk JSON，不能给出 Natural gate 的算法结论。

### 4.2 A_simplex_scalar 消融

A 组完成了有效 certificate，两个 variant 均真实失败。

| Variant | Regime | Groups / Scenes | Fit / Verify groups | Benefit AUC | Harm AUC | Verify selected | Verify positive recall | Gate |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Balanced | Near | 290 / 123 | 127 / 163 | 0.819 | 0.515 | 0 | 0 | Fail |
| Balanced | Contact | 764 / 215 | 384 / 380 | 0.580 | 0.404 | 0 | 0 | Fail |
| Precision | Near | 290 / 123 | 127 / 163 | 0.768 | 0.525 | 0 | 0 | Fail |
| Precision | Contact | 764 / 215 | 384 / 380 | 0.494 | 0.469 | 0 | 0 | Fail |

共同 warning 为：

- 没有 joint opportunity–harm–score 规则满足 fit 约束；
- verify selection 数不足；
- precision Wilson LCB 不足；
- harmful-selected UCB 超过预算。

这说明 scalar Evidence 的 Near benefit 排序具有信号，但 harm tail 很弱；Contact 的 benefit 和 harm 基本都接近随机。单纯调阈值只会在“全 abstain”和“高 harmful selection”之间移动，不能解决问题。

### 4.3 B/C 组

B/C 只能分析 adaptation-dev 趋势，不能作为 Natural-gate 结论：

| Task | Best epoch | Near recall | Contact recall | Near harm | Contact harm | Robust risk |
|---|---:|---:|---:|---:|---:|---:|
| A Balanced | 1 | 0.000 | 0.036 | 0.000 | 0.014 | 1.693 |
| A Precision | 5 | 0.111 | 0.071 | 0.008 | 0.024 | 1.575 |
| B Context Balanced | 3 | 0.222 | 0.036 | 0.025 | 0.003 | 1.619 |
| B Context Precision | 2 | 0.111 | 0.107 | 0.025 | 0.059 | 1.604 |
| C Full Balanced | 1 | 0.111 | 0.036 | 0.000 | 0.017 | 1.842 |
| C Full Precision | 10 | 0.333 | 0.071 | 0.017 | 0.028 | 1.825 |

可支持的结论是：

- Context 对 Near recall 有真实的方向性增益；
- Precision 中 Context 对 Contact recall 有一定增益；
- C 的“full”配置没有改善 robust risk，反而明显变差；
- recall 提升伴随 false intervention/harm 增长，尚未形成可部署边界。

---

## 5. v48.17 中哪些设计有效

### 5.1 冻结 top-k proposal：继续保留

此前结果和本轮 certificate 均表明主要矛盾不是候选完全缺失，而是 Evidence 无法判断哪些候选可以执行。当前阶段继续修改 proposal 会使候选生成与 Evidence 校准同时变化，破坏因果归因。

### 5.2 Source identity、零初始化 bounded residual：继续保留

校准器最后一层零初始化，初始时严格恢复 source Evidence；残差有界，不会在小样本适配第一步重写源策略。这对 CCF-A 论文中的“target adaptation with nominal preservation”是合理的设计。

### 5.3 Scene-disjoint adaptation/dev/certificate：必须保留

这套协议能够明确区分：

- 学习失败；
- 阈值过拟合；
- 工程失败；
- 独立统计 gate 失败。

该协议是论文可信度的重要组成部分。

### 5.4 Context 条件化方向：有效但实现需要更换

B/C 的 dev 结果说明，四个 scalar 不足，候选上下文中存在可学习信息。应保留“Evidence correction must be candidate-conditional”这一思想，但不能继续使用 4,890 维 raw context。

### 5.5 Safe nominal lock：已得到强支持

120 个 paired Safe scenes 中：

- collision scene rate：0.00833 vs 0.00833；
- offroad：0 vs 0；
- bounded NUP：1.0 vs 1.0；
- intervention episode：0 vs 0；
- route progression、jerk p95、yaw-rate p95 均无差异；
- 所有可用 non-inferiority 指标通过；
- `paper_safe_claim_ready=true`。

Safe 的论文结论可以是：恢复模块在 Safe regime 被 nominal lock 隔离，不污染正常驾驶。它不是恢复性能证明，但它满足三-regime 统一框架中的 safety invariance 目标。

---

## 6. v48.17 中哪些设计无效或未起效

### 6.1 78,630 参数 raw-context calibrator：不再重复

适配数据只有：

| Regime | Deployable positive groups | Positive scenes |
|---|---:|---:|
| Near | 16 | 10 |
| Contact | 44 | 17 |
| Total | 60 | — |

用 78,630 参数学习目标域校准，样本/参数关系严重不足。即使移除错误的 20k 检查，它仍有较高方差和场景记忆风险。

### 6.2 Simplex tri-class correction：不适合当前 Contact 歧义

v48.17 将 harm、dead、benefit 归一化为总和 1。Contact 中可能出现：

```text
有恢复收益信号
同时仍有未消除的二次碰撞/失稳风险
```

Simplex 会迫使 benefit 上升时 harm 下降，或反之，容易制造 false-safe。安全策略更合理的是分别估计：

```text
P(benefit)
P(harm)
```

即使两者同时高，也应由 harm veto 拒绝，而不是强迫模型给出单一类别。

### 6.3 “Batch balanced” 实际只是附加损失

v48.17 代码仍保留：

- per-group top-1 NLL；
- per-group top-k NLL；

随后再加 class-balanced minibatch loss。dead/mixed group 数量远多于 beneficial group，因此主梯度仍由原始 ERM 主导。实现与算法说明中的“由 balanced objective 解决 all-abstain”不一致。

### 6.4 Stratified sampler 单独使用不足

Sampler 成功构造了 harmful/dead/beneficial 比例，但如果 loss 中原始 ERM 仍存在，它只改变采样频率，不能消除同一 batch 内 dead 候选的数量优势。

### 6.5 Checkpoint metric 没有强制双 regime 可行

Fold-robust risk 允许 Near 改善、Contact 仍接近 0 recall，或者相反。论文目标要求同一机制在 Near 与 Contact 都成立，因此 checkpoint selection 必须对：

```text
min(recall_near, recall_contact)
max(harm_near, harm_contact)
max(false_near, false_contact)
```

显式惩罚。

---

## 7. 三个 regime 的独立问题

### 7.1 Safe

**当前状态：工程和算法均基本正确。**

缺陷不是性能不足，而是论文表达需要准确：Safe 是 nominal invariance，不是恢复增益。外部 baseline 进一步支持 lock 的必要性：

| Safe offline method | Intervention | bounded NUP | max yaw violation |
|---|---:|---:|---:|
| Nominal/log replay | 0 | 1.000 | 0 |
| Wayformer BC | 0 | 1.000 | 0 |
| BeTopNet-lite | 0.176 | 1.000 | 0.176 |
| GameFormer-lite | 0.392 | 0.947 | 0.392 |

恢复/交互策略在 Safe 中主动切换反而可能降低 NUP、增加动态偏差。因此统一算法不应在 Safe 学习“更积极恢复”，而应学习 regime-conditioned abstention/nominal lock。

### 7.2 Near-contact

特点：

- 正机会极少；
- candidate benefit AUC 可以较高；
- 收益幅度较小；
- harmful/dead 边界不清；
- verify 支持和 Wilson 下界很难建立。

强外部 offline baseline 是 predictive safety filter：

```text
DRS                  0.973
FRA_exec             0.127
ODG                  0.174
bounded NUP          0.988
intervention         0.446
secondary collision  0.098
deployability        0.547
```

Near 的投稿目标不是最大化 intervention，而是以更低 intervention 获得接近或超过该 DRS/deployability，并保持 NUP 和 secondary collision。当前外部 Near closed-loop 仅完成 30–34/50 scenes，状态仍为 `running_scene`，不能作为最终论文数字。

### 7.3 Contact

特点：

- 恢复机会比 Near 多；
- benefit/harm Evidence 的跨 scene 泛化最差；
- 恢复动作经常既有短期收益又有 secondary/recontact 风险；
- Simplex 更容易错误消除 harm tail。

Contact 外部 closed loop（50 scenes）表明现有方法的主要代价是高 intervention：

| Method | DRS | FRA | Deployability | NUP | Intervention | Collision scenes |
|---|---:|---:|---:|---:|---:|---:|
| Restoration | 0.494 | 0.506 | 0.324 | 0.807 | 0.833 | 0.08 |
| Post-impact MPC | 0.527 | 0.473 | 0.348 | 0.454 | 0.976 | 0.18 |
| Severity minimization | 0.492 | 0.510 | 0.320 | 0.455 | 0.945 | 0.10 |
| Post-crash braking | 0.417 | 0.584 | 0.297 | 0.462 | 1.000 | 0.28 |

OC-RAP 的差异化目标应是：利用 recovery set 取得 comparable DRS/deployability，同时通过 selective Evidence 大幅降低 intervention、提高 NUP，并约束二次碰撞。

---

## 8. 是否存在三个 regime 的统一优化方法

存在，但统一的不是“所有 regime 使用同一执行强度”，而是同一选择原则：

```text
高召回 proposal
      ↓
regime-conditioned benefit tail
regime-conditioned harm tail
      ↓
harm veto + abstention
      ↓
Safe: nominal lock
Near: 稀疏高精度 intervention
Contact: 更强恢复，但要求独立 harm 证据通过
```

该统一结构与论文 idea 更匹配：恢复不是一个无条件 policy，而是带有机会、风险和不确定性的 conditional recovery decision。

---

## 9. v48.18 DUET-BRIDGE

**DUET-BRIDGE：Dual-tail Uncoupled Evidence Transfer with frozen tournament context and balanced target adaptation。**

### 9.1 Frozen tournament context

不再输入约 4,890 维 raw relative representation，而是复用冻结 Recovery Set Tournament 的 48 维 contextual embedding。

优点：

- 包含候选与同组其他候选的关系；
- 与已验证 proposal 语义一致；
- 不更新 proposal；
- 默认两-regime 校准器参数降到 1,532；
- 更适合 60 个正机会 group 的小样本适配。

### 9.2 Independent dual tails

校准器输出两个 bounded residual：

```text
benefit_logit' = benefit_logit_source + Δbenefit
harm_logit'    = harm_logit_source    + Δharm
```

Nominal 行在校准后仍被显式钉扎为零 logit，避免训练后的 residual bias 污染 Safe/nominal 语义。

不进行三类 softmax。部署继续使用：

```text
benefit 足够高
AND harm 足够低
AND rank/support/macro constraints 满足
```

### 9.3 Independent-tail supervision

| Teacher state | Benefit target | Harm target |
|---|---:|---:|
| Beneficial | 1 | 0 |
| Dead-zone | 0 | 0 |
| Harmful | 0 | 1 |

使用两个 BCE tail loss。这样 harmful 与 dead 的区别由 harm tail 学习，beneficial 与 dead 的区别由 benefit tail 学习。

### 9.4 Strict balanced replacement

在 calibrator-only adaptation 中：

```text
旧：group ERM + class-balanced auxiliary
新：per-regime/per-class balanced evidence objective replaces group Evidence ERM
```

仍保留 residual anchor 和显式启用的 cross-group 项，但不再让 dead-zone-dominated top-k NLL成为主目标。

### 9.5 Cross-regime checkpoint selection

新增：

```text
direct_duet_selection_risk
```

它基于 dev certificate risk，附加：

- Near/Contact 最小 recall shortfall；
- 最坏 regime harmful-switch；
- 最坏 regime false intervention。

这只用于 adaptation dev early stopping，不改变最终 Natural-gate 阈值，不读取 test。

---

## 10. 四组消融

| Group | Dual tails | Tournament context | Strict balanced replacement | Cross-regime checkpoint |
|---|---:|---:|---:|---:|
| A_dual_scalar | ✓ | — | — | — |
| B_dual_tournament | ✓ | ✓ | — | — |
| C_dual_tournament_balanced | ✓ | ✓ | ✓ | — |
| D_full_duet | ✓ | ✓ | ✓ | ✓ |

Balanced/Precision 共 8 个任务一次性启动：

```text
GPU0：4 tasks
GPU1：4 tasks
NUM_WORKERS=1 per task
```

消融结果必须以独立 certificate 为准，不再只比较 dev best epoch。

---

## 11. 工程完整性修复

已完成：

- 修复 v48.17 参数量误杀；
- 提供 v48.17 checkpoint 恢复入口；
- RC=30 也写 learning-gate report；
- checkpoint/inference 同步 context source；
- RecoverySetTournament 暴露冻结 contextual embedding；
- 新增 dual-tail inference；
- 新增 independent-tail loss；
- balanced loss 可严格替换 Evidence ERM；
- 新增 cross-regime selection metric；
- v48.18 main controller 始终写完成状态；
- stress wrapper 仍强制检查 `NEXT_COMMANDS.txt`；
- 8 个消融全部并发；
- 更新 `ALGORITHM_CHANGELOG.md`；
- Python compileall 通过；
- 全部 Shell `bash -n` 通过；
- pytest：176 passed，5 warnings。

当前环境没有真实 WOMD/Waymax 和 A30，因此没有伪造训练或 gate 结果。v48.18 是否通过必须由服务器上的独立 certificate 决定。

---

## 12. 下一步判定规则

### 首先恢复现有 v48.17

- RC=0：v48.17 真正通过；立即运行 gate-authorized stress，并保留为 v48.18 的主要对照。
- RC=20：v48.17 真正失败；不读 test，运行 v48.18 主实验。
- RC=30：查看 `learning_gates_v48_17.json`、`V48_17_RECOVERY_COMPLETE.json` 和 certificate log，先修工程。

### v48.18 主实验

- RC=0：自动生成 `NEXT_COMMANDS.txt`，运行 stress。
- RC=20：运行四组消融，确定失败来自 dual tails、context、balanced replacement 或 checkpoint metric。
- RC=30：不能做算法归因。

### 投稿判据

Natural gate 只是进入 closed loop 的前置条件。最终 CCF-A 主张需要同时满足：

1. Safe paired non-inferiority；
2. Near 对 predictive-safety / risk-filter baselines 的选择效率优势；
3. Contact 对 restoration/MPC/severity baselines 的 intervention–NUP–DRS Pareto 优势；
4. 多 seed、scene-disjoint certificate 和 test/stress 一致；
5. 四组消融证明 dual tails、tournament context、strict balancing、cross-regime selection 各自贡献。

