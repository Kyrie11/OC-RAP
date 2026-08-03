# OC-RAP v48.33：论文、数据、代码与现有结果联合审阅

## 1. 结论先行

本次审阅覆盖：

- 论文 `post-collision.tex`；
- 三类数据集的 `reports.zip`；
- 当前实现 `OC-RAP.zip`；
- 主实验结果 `ocrap_v48_33_eligible_set_policy_dedicated_4833.zip`；
- 消融结果 `ocrap_v48_33_eligible_set_policy_ablations_4833.zip`。

核心判断如下。

1. **论文的核心 idea 是清晰且有辨识度的**：规划器不只判断每个潜在未来是否“各自存在恢复动作”，而是要求在执行候选前缀后，针对观测上无法区分的潜在未来，仍存在可由同一后验观测选择的兼容恢复策略。这是 oracle recoverability 与 deployable recoverability 的区别。
2. **当前代码仍保留了论文的核心计算原语**：`src/ocrap/algorithms/ocmero.py` 中的 OC-MERO 对恢复 margin 矩阵、latent-root 概率和 observation-equivalence kernel 做两层 lower-tail 聚合，显式输出 `r_dep`、`r_orc` 和 gap。
3. **v48.33 的真实运行策略已经明显复杂于论文中的简洁 CRISP 描述**。当前 selector 增加了 structured transformer、direct recovery-value/evidence heads、top-k proposal、evidence rerank、regime-specific certificate、宏动作白/黑名单、干预预算和多个 rescue channel。这不是错误，但论文实验部分应把它称为“当前工程实现变体”，不能让读者误以为只用了论文正文中最简洁的阈值式 admission。
4. **数据管线本身是健康的**：12份 canonical report 均 `failures=0`，Waymax runtime fraction 均为 1.0。near-contact/contact 数据具有较高比例的 oracle artifact、negative deployable candidate 和 incompatible alias pair，能够支持论文主问题；safe 数据不支持 FRA/ODG/observation consistency，应只用于 nominal non-inferiority、舒适性、进度和不引入额外风险。
5. **当前失败是 selector/admission 的可靠性失败，不是 Waymax 闭环故障**。Natural gate 已评估但未通过；候选级 AUC 尚可，然而 scene-time group 内 top-1 correlation 近零或为负，选中动作 precision 很低，且被选动作的 teacher advantage 均值为负。
6. **现有 dev shadow 只有每个 stress regime 8个场景，干预集中在一个场景，不能用于总体性能结论**。它适合验证代码路径和可视化，但不足以证明近接触或碰撞后性能提升。
7. 为满足“先跑完整闭环、展示模型作用”的目标，已实现一个**独立且显式 opt-in 的 ungated exploratory 流程**。它不会修改正式 `V48_33_COMPLETE.json`，只有设置 `ALLOW_UNGATED_TEST=1` 才会读取 test roots，并在输出中写入不可用于部署证书的标记。

---

## 2. 论文方法理解

### 2.1 论文要解决的真正问题

论文不是普通的“降低碰撞概率”方法，而是在问：

> 当前动作执行以后，车辆是否仍保留一种能够根据届时可获得的观测来选择和执行的恢复方式？

典型 oracle artifact 是：

- 隐藏未来 A 中应继续；
- 隐藏未来 B 中应制动；
- 执行候选动作后，A 与 B 对车辆仍不可区分；
- oracle 因知道未来身份，可以分别选择不同恢复动作，因此判定每个分支都可恢复；
- 部署车辆无法知道自己处于 A 还是 B，因而没有可部署的统一决策依据。

所以论文区分：

- local safety：短时风险或约束是否满足；
- oracle recoverability：知道潜在未来身份时，每个分支是否存在某个恢复动作；
- deployable recoverability：只使用动作执行后的真实观测，恢复动作是否仍可选择并执行。

### 2.2 论文算法链路

论文中的主要链路是：

1. 生成短时 candidate prefix；
2. 构造 recovery-sufficient latent roots；
3. 预测候选前缀执行后的 observation-equivalence kernel；
4. 对每个 latent root × recovery option 计算 affordance-conditioned signed margin；
5. 用 OC-MERO 聚合 deployable recovery margin；
6. 通过校准 admission 规则决定保留 nominal 还是切换到恢复型候选；
7. 在 receding-horizon Waymax 闭环中重复重规划。

恢复选项包括受控制动、车道内制动、侧向脱离、让行并回归、靠边、接触缓解、碰撞后稳定和避免二次碰撞等语义宏动作。

### 2.3 OC-MERO 的代码实现

`src/ocrap/algorithms/ocmero.py` 中：

- `M[K,L]`：K个 latent root 与 L个 recovery option 的 signed margin；
- `p[K]`：root 概率；
- `C[K,K]`：后缀观测兼容/等价权重；
- oracle 先对每个 root 独立选择最佳 option，再做外层 lower-tail 聚合；
- deployable 先在与当前 root 观测兼容的 root 集合上，对同一 option 做内层 lower-tail 聚合，再选 option，最后做外层 lower-tail 聚合；
- 输出 `r_orc`、`r_dep` 和 `gap = r_orc - r_dep`。

这一实现与论文的 oracle-to-deployable gap 思想一致。

### 2.4 当前 v48.33 与论文正文的差异

当前模型和策略不是“仅预测论文中的几个量后做一次阈值判断”，而是一个经过多轮实验演化的系统：

- structured transformer encoder；
- 8个 latent roots；
- 24个 recovery options；
- direct recovery-value、rank、uncertainty、opportunity、harm 和 evidence admission 等头；
- 统一 top-5 proposal；
- top-k 内 evidence rerank；
- 只允许物理可行动且符合宏动作合同的候选参与挑战 nominal；
- regime-specific calibration/certificate；
- running intervention budget、cooldown、最大连续干预；
- relative recovery、protective macro、brake rescue、PCD rescue 等通道。

建议论文中明确区分：

- **核心方法层**：observation-consistent deployable recoverability + OC-MERO；
- **v48.33工程策略层**：用于学习 proposal、可靠 admission 和 nominal-preserving deployment 的具体实现。

---

## 3. 三类数据集性质

### 3.1 数据规模

| Split | Safe：样本/场景 | Near-contact：样本/场景 | Contact：样本/场景 |
|---|---:|---:|---:|
| train | 20,000 / 1,171 | 13,324 / 600 | 16,790 / 500 |
| val | 2,328 / 132 | 3,445 / 176 | 6,477 / 211 |
| calibration | 2,544 / 135 | 6,039 / 316 | 16,843 / 543 |
| test | 3,216 / 175 | 4,723 / 250 | 6,687 / 209 |

全部12份报告：

- `failures=0`；
- `waymax_runtime_fraction=1.0`。

### 3.2 calibration 数据的关键性质

Near-contact：

- 平均 7.894 candidates/group；
- feasible fraction 0.862；
- 平均 11个 futures；
- oracle artifact fraction 0.240；
- negative deployable fraction 0.448；
- 平均有效 roots 6.475/8；
- incompatible alias pair fraction 0.196。

Contact：

- 平均 8.883 candidates/group；
- feasible fraction 0.878；
- 平均 13个 futures；
- oracle artifact fraction 0.212；
- negative deployable fraction 0.417；
- 平均有效 roots 7.979/8；
- incompatible alias pair fraction 0.142。

这些统计说明 near/contact 中确实存在大量“oracle看似可恢复，但部署不可恢复”或观测别名不兼容的样本，能够支撑论文主张。

### 3.3 Safe regime 的正确用途

Safe 数据：

- 只有3个 futures；
- 平均有效 roots 2/8；
- artifact fraction 0；
- incompatible alias pair fraction 0；
- report 明确标记不支持 FRA、ODG 和 observation consistency。

因此 safe 只应回答：

- OC-RAP 是否不必要地频繁干预；
- 是否引入碰撞、off-road或舒适性恶化；
- route progression / nominal utility 是否非劣；
- 计算开销是否可接受。

不应使用 safe 数据宣称 observation-consistency 或 oracle-artifact 消除能力。

---

## 4. v48.33 当前结果诊断

### 4.1 主流程状态

`V48_33_COMPLETE.json` 表明：

- `pipeline_valid=true`；
- `certificate_executed=true`；
- `gate_evaluated=true`；
- `gate_passed=false`；
- `test_roots_read=false`。

也就是说，训练、协议和 certificate 管线已执行，但严格 Natural gate 阻止了 test。

### 4.2 Gate 失败的具体原因

Balanced：

| Regime | Candidate positive AUC | Group top-1 correlation | Selected | Precision | Harmful selected rate | Selected teacher advantage mean |
|---|---:|---:|---:|---:|---:|---:|
| near | 0.8464 | -0.0775 | 5 | 0.0000 | 0.2000 | -0.1153 |
| contact | 0.5651 | -0.0318 | 43 | 0.0233 | 0.3953 | -0.2042 |

Precision：

| Regime | Candidate positive AUC | Group top-1 correlation | Selected | Precision | Harmful selected rate | Selected teacher advantage mean |
|---|---:|---:|---:|---:|---:|---:|
| near | 0.7831 | -0.0915 | 14 | 0.0000 | 0.5000 | -0.2626 |
| contact | 0.5417 | +0.0269 | 45 | 0.0222 | 0.4222 | -0.1593 |

这组数字的含义是：

- 模型在“把候选分成可能正/负”层面有一定信号，尤其 balanced-near；
- 但真正部署需要在同一 scene-time group 内选出最佳候选；该排序相关性近零或为负；
- 被 admission 选出的动作几乎没有正 precision，且平均 teacher advantage 为负；
- 因此 gate 失败是合理的，不能通过放宽阈值来伪装成通过。

### 4.3 现有 dev shadow 能说明什么

每个 variant × regime 仅有8个 paired scenes。以 precision 为例：

Near-contact：

- 平均干预率从 0 增至 0.0417；
- 只有1/8场景产生实际行为差异；
- min clearance 的均值和最小值没有变化；
- min TTC 平均增加约0.015 s；
- clearance deficit AUC 平均降低约0.0118 m·s；
- TTC deficit AUC 平均降低约0.0119 s²。

Contact：

- 平均干预率约0.0156；
- 只有1/8场景产生行为差异；
- post-contact free-space AUC 有极小正变化；
- terminal recovery gain、jerk等部分量存在退化；
- 当前唯一实际干预场景在新筛选器中被归为 regression，而不是被包装为改善案例。

所以 dev shadow 只证明闭环和 selector 能执行，不能证明总体有效。

### 4.4 消融结果不能用于算法结论

`ABLATIONS_STATUS.json` 中8个任务全部未完成。它们在 adaptation 阶段以 RC=30 退出，统一原因是：

`factor cache inputs or hyperparameters changed`

因此当前消融包说明的是 cache-contract 工程问题，不是“某个模块有效或无效”。

---

## 5. 推荐的完整 closed-loop 评估协议

### 5.1 主对比

每个 variant 分别运行：

- scalar/nominal-preserving control；
- OC-RAP v48.33 method；
- 二者使用相同 target_key、同一 start time、同一 Waymax source 和相同随机种子；
- 每个 test scene 使用一个 canonical target，避免同一 scene 的多个相邻时间窗被当作独立样本；
- safe 40步，near-contact 50步，contact 60步；
- replan interval=1；
- 输出 scene-level traces。

Test 场景数按当前 report 为：

- safe 175 scenes；
- near-contact 250 scenes；
- contact 209 scenes。

脚本不会硬编码这些数字，而是从实际 dataset root 重新计数。

### 5.2 为什么全量物理闭环使用 `label_mode=fast`

这里的 `fast` 只代表不在每个闭环 step 上重新计算昂贵的 oracle teacher/PCD labels；它仍然会：

- 读取当前 Waymax 状态；
- 构造候选；
- 执行当前模型的 selector；
- 将控制输入真正送入 Waymax dynamics；
- 更新场景并重新规划；
- 计算真实物理轨迹指标。

因此它是物理 closed-loop，不是 open-loop replay。

对于筛出的少量 critical scenes，可再用 `selected_topk` 做稀疏 teacher/PCD audit。这样避免对所有test场景做极昂贵的在线标签，同时能解释关键场景中模型是否错过更好的 recoverable candidate。

### 5.3 三类 regime 应报告的指标

Safe：

- overlap/offroad；
- bounded NUP；
- route progression；
- intervention rate / episodes；
- acceleration p95、deceleration、jerk、yaw rate；
- nominal non-inferiority。

Near-contact：

- `min_clearance_m_min`、`min_clearance_m_p05`；
- `ttc_s_min`、`ttc_s_p05`；
- near-contact / critical-TTC exposure rate、duration、episodes；
- clearance deficit AUC、TTC deficit AUC；
- terminal clearance/TTC 和 recovery gain；
- 新 overlap、secondary overlap、offroad；
- intervention rate。

Contact：

- overlap duration、longest overlap run；
- recontact/secondary collision；
- post-contact mean/max/terminal clearance；
- post-contact free-space AUC及归一化值；
- post-contact clearance deficit AUC；
- escape event / time；
- stable-stop quality / time；
- speed、yaw-rate、acceleration、jerk；
- offroad 和 route progression。

### 5.4 统计规则

- 必须按 target_key 做 paired comparison；
- 报告 control mean、method mean、paired delta和scene-level bootstrap 95% CI；
- 不只报告总体均值，还要报告实际干预子集；
- 任何改善不能以新 overlap、recontact、offroad或严重舒适性退化为代价；
- 在 gate 未通过时，所有 test 输出都必须写作“ungated exploratory diagnostic”，不能称为 deployment-valid result。

---

## 6. Critical 场景筛选与可视化

### 6.1 筛选原则

新增筛选器不是“只找最好看的正例”，而是同时输出：

1. largest improvements；
2. largest regressions；
3. most critical control scenes；
4. largest intervention scenes。

Near-contact 改善分数重点考虑：

- 最低/p05 clearance；
- 最低/p05 TTC；
- exposure rate/duration；
- clearance/TTC deficit AUC；
- recovery gain和terminal margin；
- overlap、secondary overlap和offroad硬惩罚。

Contact 改善分数重点考虑：

- overlap/recontact；
- post-contact free-space AUC和clearance；
- escape/stable stop；
- yaw rate、jerk；
- offroad硬惩罚。

默认只有模型确实发生非nominal干预的场景才可进入“largest improvements”；但 regression 和 control-critical 场景不会受此限制。

### 6.2 每个场景的可视化

Near-contact：

- clearance时间序列，带2 m参考线；
- TTC时间序列，带3 s参考线；
- ego speed；
- acceleration proxy；
- selected macro和干预时刻；
- 若新运行保存trace，则输出control与OC-RAP的ego XY trajectory。

Contact：

- post-contact clearance，带0.5 m参考线；
- overlap/recontact flag；
- speed，带0.5 m/s稳定停止参考线；
- absolute yaw rate，带0.25 rad/s参考线；
- absolute acceleration proxy；
- selected macro、contact anchor和干预时刻；
- ego XY trajectory。

总体图：

- paired scatter：control vs OC-RAP；
- critical scene signed score：正数为改善，负数为退化；
- HTML index汇总所有图和场景说明。

---

## 7. 已实现的代码

### 7.1 闭环 runner

`src/ocrap/simulation/closed_loop_runner.py`

- 新增 `closed_loop.save_traces`；
- 保存 `trace_dt_s`、`state_xy_trace`、`metric_trace`；
- 新增 `closed_loop.target_keys_file`，可精确复跑筛出的critical targets；
- 保留旧 `save_trace_npz` 兼容入口。

### 7.2 配置与主运行脚本

`src/ocrap/config/defaults.py`

- 增加 `target_keys_file` 和 `save_traces` 默认配置。

`scripts/run_ocrap_v48_trac_sr.sh`

- near/contact可分别设置max rollouts、max steps和target-key文件；
- 支持保存trace；
- near/contact target数量可独立设置。

### 7.3 新增工具

- `tools/count_closed_loop_targets.py`：按实际dataset root计数闭环targets；
- `tools/compare_paired_closed_loop.py`：扩充碰撞、offroad、舒适性和进度指标；
- `tools/select_critical_closed_loop_scenes.py`：双向critical筛选；
- `tools/visualize_closed_loop_critical.py`：scene-level与aggregate可视化；
- `tools/summarize_ungated_closed_loop.py`：三regime合并摘要；
- `scripts/run_v48_33_ungated_full_closed_loop.sh`：完整ungated闭环控制器。

### 7.4 安全/协议保护

新控制器：

- 检查正式主流程状态；
- pipeline无效时拒绝运行；
- gate失败时必须显式 `ALLOW_UNGATED_TEST=1`；
- 不修改正式完成状态；
- 写入 `UNGATED_EXPLORATORY_ONLY.json`；
- 输出summary中明确 `valid_for_deployment=false`。

---

## 8. 预期输出结构

```text
$OUT/ungated_full_closed_loop/
├── UNGATED_EXPLORATORY_ONLY.json
├── FULL_TARGET_COUNTS.json
├── balanced/
│   ├── closed_loop_safe_fast_v48_scalar.json
│   ├── closed_loop_safe_fast_v48.json
│   ├── audit_near_contact_selected_topk_v48_scalar.json
│   ├── audit_near_contact_selected_topk_v48_v48.json
│   ├── audit_contact_selected_topk_v48_scalar.json
│   ├── audit_contact_selected_topk_v48_v48.json
│   ├── safe_test_paired.json/.md
│   ├── near_test_paired.json/.md
│   ├── contact_test_paired.json/.md
│   ├── critical_near_contact.json/.csv
│   ├── critical_contact.json/.csv
│   ├── visualizations/near_contact/index.html
│   ├── visualizations/contact/index.html
│   └── UNGATED_FULL_CLOSED_LOOP_SUMMARY.json/.md
└── precision/
    └── ...同上
```

设置 `RERUN_CRITICAL_WITH_LABELS=1` 后，还会生成 `critical_label_audit/`。

---

## 9. 运行后如何判断“模型有效”

Near-contact 可以支持正面结论，至少应同时满足：

- paired min-clearance或TTC有实际提升，或deficit AUC/exposure显著下降；
- 改善不是只来自一个场景；
- 没有新增overlap、secondary overlap或offroad；
- safe中的进度和舒适性没有不可接受退化；
- worst regression cases可以解释且数量受控。

Contact 可以支持正面结论，至少应同时满足：

- recontact/overlap不增加；
- post-contact free-space、terminal clearance或escape/stable-stop有一致改善；
- yaw-rate/jerk没有系统性恶化；
- 改善发生在实际干预场景中；
- critical可视化显示宏动作与物理结果之间有可解释因果链。

若完整结果仍表现为低干预、效果只集中在1–2个场景，最合理的结论不是“模型显著有效”，而是：

- OC-RAP核心计算可运行；
- 当前v48.33 learned selector/admission仍是瓶颈；
- 需要改进group-wise ranking与selective admission，而非继续微调gate阈值。

---

## 10. 当前无法在本环境直接完成的部分

上传的结果压缩包未包含 `model_v48_trac_sr/best.pt`，并且本环境没有用户机器上的WOMD 150 shards与OCRAP test roots。因此这里无法真实执行175/250/209个Waymax闭环。

已完成的验证包括：

- 所有新增/修改Python文件通过编译；
- 两个bash脚本通过 `bash -n`；
- critical selector与可视化测试共5项通过；
- 使用现有precision dev-shadow JSON成功生成near/contact双向critical报告、时间序列图和HTML index；
- 现有contact唯一干预场景被正确识别为regression，证明工具不会只挑正例。
