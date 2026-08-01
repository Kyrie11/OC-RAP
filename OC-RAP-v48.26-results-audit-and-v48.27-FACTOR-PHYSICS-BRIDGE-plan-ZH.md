# OC-RAP v48.26 结果联合审计与 v48.27 FACTOR-PHYSICS-BRIDGE 优化方案

## 1. 审计范围与结论边界

本次联合审计覆盖：

- v48.26 完整代码；
- 主实验 `ocrap_v48_26_execution_physics_dedicated_4826`；
- adaptation-dev shadow closed-loop 运行日志与中间 JSON；
- 8 个双卡消融实验；
- Balanced/Precision 的训练轨迹、checkpoint、adaptation-dev 冻结规则、完整 certificate 验证结果；
- Near/Contact proposal-constrained oracle、candidate/Evidence 排序、safe admission 与宏动作分布；
- v48.26 新增 Near/Contact 物理指标的实现和实际运行状态。

最重要的结论有三点：

1. v48.26 的 `RC=20` 是一个真实、有效的 Natural-gate 拒绝，不是 pipeline failure。`pipeline_valid=true`、`gate_evaluated=true`、`test_roots_read=false`，因此可以做算法归因。
2. proposal-constrained oracle 在 adaptation-dev 和完整 certificate 上对 Near、Contact 都可行。当前 gate 失败不是 top-3 proposal 支持不足，也不是 certificate 定义数学上不可满足，而是 learned selector 没有形成满足联合约束的开发规则。
3. dev-shadow 没有运行出任何有效物理场景。四组审计均为 0 scene，所有新增物理指标未被实际计算，因此不能用现有 shadow JSON 判断 v48.26 是否改善了 clearance、TTC、二次接触、escape 或 stable stop。

## 2. dev-shadow closed loop 的根本工程错误

### 2.1 实际故障链

四个 Near/Contact 审计任务均成功读取了 16 个离线 target，但只扫描：

```text
validation_interactive 的前 900 个 raw scenarios
```

离线 target 的 scene id 含有类似：

```text
<stable_scenario_id>__wx00011519
```

其中 `__wx########` 是数据加载顺序后缀，不是 WOMD 场景本体标识。目标位于较后位置，而脚本固定：

```text
closed_loop.raw_max_scenarios=900
```

所以四个任务均得到：

```text
bucket_target_count = 16
bucket_matched_rollouts = 0
raw_scenarios_seen = 900
num_scenes = 0
```

原脚本还没有设置：

```text
closed_loop.require_bucket_targets=true
```

因此 runner 将 0 scene 结果写成合法 JSON，直到 `compare_paired_closed_loop.py` 找不到 paired scenes 才失败。

### 2.2 为什么现有物理指标不可解释

当 `num_scenes=0` 时，v48.26 聚合器的一部分字段仍输出 0。这个 0 不是“没有碰撞”“没有二次接触”或“干预率为零”，而是没有任何 rollout。故现有 shadow 结果中的：

- collision/overlap；
- minimum clearance、minimum TTC；
- near/critical exposure；
- secondary contact；
- overlap duration；
- free-space AUC；
- sustained escape；
- stable stop；

均没有实验意义。

### 2.3 v48.27 修复

v48.27 做了以下工程修复：

1. `DEV_SHADOW_RAW_MAX_SCENARIOS=0` 默认扫描完整 raw source，不再固定 900。
2. 允许显式设置 `DEV_SHADOW_WOMD_SOURCE`，确保使用与离线 target 对应的完整 `validation_interactive` shard 集。
3. target 优先使用 `original_scenario_id`；若只有保存时 scene id，则去掉 `__wx########` 操作后缀后匹配。
4. 设置 `closed_loop.require_bucket_targets=true`；目标列表为空或完整扫描后匹配数为 0 时立即失败。
5. 空聚合输出 `metrics_valid=false`、`empty_reason=no_closed_loop_scenes`，关键指标为 `null`，不再伪装成 0。
6. paired comparator 和 shadow controller 都要求非空、有效、成对场景。
7. 提供无需重训的 repair-only 脚本，使用 v48.27 runner 重跑现有 v48.26 checkpoint 的 adaptation-dev shadow。

## 3. RC=20 的根本原因

### 3.1 不是 proposal support 或 certificate 不可行

完整 certificate 的 top-3 oracle：

| Regime | Groups | proposal 内 safe-positive groups | Oracle precision LCB | Feasible |
|---|---:|---:|---:|---|
| Near | 290 | 9 | 0.8457 | 是 |
| Contact | 764 | 20 | 0.9241 | 是 |

adaptation-dev oracle 也可行：

| Variant/Regime | Dev groups | Safe opportunities | Oracle feasible |
|---|---:|---:|---|
| Balanced Near | 110 | 8 | 是 |
| Balanced Contact | 279 | 17 | 是 |
| Precision Near | 110 | 8 | 是 |
| Precision Contact | 279 | 17 | 是 |

因此，冻结 proposal 中存在可满足 gate 的安全机会。certificate 不是在拒绝一个结构上不可能的任务。

### 3.2 失败发生在 adaptation-dev rule fitting

四个 adaptation-dev 阶段均出现：

```text
no joint opportunity-harm-score rule satisfied fit constraints
```

最终冻结的是 `diagnostic_fit_rule`，而不是满足约束的正式 rule：

```text
source_rule_satisfied_dev_constraints = false
source_valid_for_deployment = false
```

v48.26 最终 certificate 的 `rejection_kind` 被笼统写成 `learned_gate_rejection`，容易误以为开发规则有效、只是在 certificate 泛化失败。实际更准确的归因是：

```text
development_rule_fit_rejection
```

即 learned selector 在 adaptation-dev 上就没有形成可授权的联合规则。v48.27 已区分：

- `development_rule_fit_rejection`；
- `certificate_verification_rejection`；
- `structural_support_infeasible`；
- 工程/协议失败。

## 4. Near-contact 是否把 raw-benefit 转成了 safe admission

### 4.1 Balanced Near

主实验：

- candidate positive AUC：0.448；
- candidate safe-positive AUC：0.473；
- Evidence top-1 correlation：0.183；
- Evidence safe-positive AUC：0.452；
- Evidence harm AUC：0.426；
- false switch：0.737；
- harmful switch：0.423。

certificate：

- 选择 12 个；
- safe-positive 0 个；
- harmful 9 个，占 75%；
- safe-positive recall 0；
- selected teacher advantage mean = -0.244。

Balanced Near 的原有强收益信号在 v48.26 主模型中反而明显崩塌，未形成安全准入。

### 4.2 Precision Near

Precision 保留了一个明确的局部正向信号：

- candidate positive AUC：0.823；
- candidate safe-positive AUC：0.803；
- Evidence safe-positive AUC：0.828；
- high-opportunity conditional harm AUC：0.842；
- nonpositive false switch：0.085；
- positive top-1 regret：0.069。

但最终安全准入仍失败：

- certificate 选择 11 个；
- safe-positive 0 个；
- harmful 3 个；
- recall 0；
- selected teacher advantage mean = -0.161。

因此 Precision Near 已具备“识别部分收益与高机会风险”的能力，但 nominal-vs-recovery admission 与正确 action 选择仍没有闭合。

### 4.3 消融中的可利用正向区域

A/B 消融的 Balanced Near：

- 选择 5 个；
- safe-positive 3 个；
- empirical precision 0.60；
- harmful 1 个；
- recall 0.333；
- selected teacher advantage mean +0.251。

这说明当前数据和 frozen proposal 中并非完全没有可学习信号。问题是 v48.26 的 exact safe-utility 和 full objective 加入后破坏了这一局部区域：C/D 的 precision 降到约 0.154/0.125，harmful selection 上升，平均收益重新为负。

## 5. Contact 是否证明了 safe admission

没有。

### 5.1 Balanced Contact

- Evidence correlation：0.056；
- safe-positive AUC：0.541；
- harm AUC：0.436；
- false switch：0.782；
- harmful switch：0.464；
- top-1 regret：0.309。

certificate：

- 选择 50 个；
- positive 1 个；
- precision 0.02；
- harmful 22 个，占 44%；
- recall 0.05；
- mean advantage -0.247；
- macro concentration 0.58。

### 5.2 Precision Contact

- Evidence correlation：-0.102；
- safe-positive AUC：0.461；
- harm AUC：0.387；
- conditional harm AUC：0.422；
- harmful switch：0.420；
- top-1 regret：0.145。

certificate：

- 选择 46 个；
- positive 1 个；
- precision 0.0217；
- harmful 21 个，占 45.7%；
- recall 0.05；
- mean advantage -0.225；
- macro concentration 0.739。

Contact 仍然没有证明哪一个撞后 action 能带来真实、非 harmful、可迁移的恢复收益。

## 6. v48.26 中仍存在的学习实现缺陷

### 6.1 harm 标签是五因子，模型只预测三因子

Natural-gate 的 `component_harmful` 由五个不可补偿风险分量定义：

1. DRS 退化；
2. deployability 退化；
3. oracle-to-deployable gap 恶化；
4. hard-rule violation；
5. harm proxy 恶化。

但 v48.26 的 component harm head 只有前三个输出，loss 也只监督前三个。于是：

- hard-rule harmful；
- harm-proxy harmful；

在模型表示层没有独立可学习通道。最终 gate 用五因子判 harmful，而模型最多显式预测三因子，造成训练目标与 gate 合同不一致。

### 6.2 safe-utility regression 与 listwise/frontier 尺度不一致

v48.26 regression 使用部署分数：

```text
sigmoid(admission_logit) - 0.5
```

但 safe-utility listwise 和 frontier contrast 仍使用原始 admission logit，与 `[-0.5, 0.5]` teacher utility 比较。原始 logit 无界，listwise 梯度可压过 regression，正好对应 C/D 消融明显退化。

v48.27 将 regression、listwise、frontier 全部统一为真实部署 safe-utility 尺度。

### 6.3 opportunity 语义冲突

同一个 opportunity head 同时承担：

- raw-benefit 连续排序；
- safe-benefit BCE/准入语义。

有收益但 harmful 的 candidate 对 raw-benefit 是正样本，对 safe-benefit 是负样本，梯度冲突。v48.27 明确分解：

- opportunity head：只学习 raw benefit；
- 五个 component heads：分别学习不可补偿风险；
- admission head：学习 raw benefit 与五因子 veto 后的最终 safe utility；
- Natural gate ground truth 仍使用 safe-benefit。

### 6.4 稀疏 admission 梯度污染 dense factor learning

v48.26 联合训练 benefit、harm、admission。Near/Contact safe-positive 很稀疏，admission 的 setwise、frontier、coverage 和 harmful penalty 会反向扭曲本来可用的 raw-benefit 排序和风险因子。

v48.27 使用两阶段训练：

**Stage 1**

- 只训练 raw-benefit head 与五个 harm factor heads；
- 关闭 admission、setwise admission、selective risk/coverage；
- 使用 dense candidate 与 group 内排序监督。

**Stage 2**

- 冻结 benefit 与五个 harm heads；
- 只训练有界 admission residual；
- 使用 deployment-exact safe utility、categorical nominal+top-k、轻量 frontier；
- 防止 sparse admission 梯度破坏 factor ranking。

### 6.5 sampler 日志语义容易误导

原日志中的 `num_safe_positive` 是旧 root/sample 层统计，不等同于 Natural-gate 使用的 scene-time safe-positive group 数。它可能显示 0，而 `positive_advantage_groups` 非零。v48.27 保留旧字段以兼容，同时新增：

- `legacy_root_safe_sample_count`；
- `safe_positive_group_count`。

## 7. 物理指标是否正确、是否能帮助归因

### 7.1 本轮不能做数值验证

因为 0 scene matched，现有结果没有真正执行任何物理指标计算。不能声称 v48.26 的指标值正确，也不能依据它们判断算法有效性。

### 7.2 静态定义仍然合理

v48.26/27 的指标定义可覆盖投稿所需主要性质：

**Near**

- minimum clearance、minimum TTC；
- near/critical exposure duration；
- episode count 与 longest run；
- clearance/TTC deficit AUC；
- terminal recovery gain；
- time to worst point；
- collision/offroad；
- intervention、route progression、jerk、yaw-rate。

**Contact**

- secondary/re-contact event/episode/scene rate；
- overlap duration、episode count、longest run；
- normalized post-contact free-space AUC；
- clearance deficit AUC、terminal clearance、time-to-peak；
- sustained escape、time-to-escape；
- stable-stop quality 与 time-to-quality-stable-stop；
- offroad、route progression、jerk、yaw-rate、intervention burden。

v48.27 修复了空结果、scene-id、post-contact target 匹配和 paired validness，并增加 focused tests。但指标的数值正确性仍需 repair-only shadow 或完整 v48.27 shadow 在真实 Waymax 上验证。

### 7.3 何时应加入物理训练监督

当前 teacher index 没有 candidate-level 的 overlap duration、recontact、free-space AUC、escape 和 stable-stop 标签。v48.27 不会把闭环汇总指标伪装成训练 target。

下一步判据：

- 若离线 safe ranking/gate 改善，shadow 物理指标也改善：保留当前 PCD teacher；
- 若离线改善但 shadow 不改善：预注册 candidate-level physical teacher rollout，新增独立 temporal recovery auxiliary head；
- 不允许根据看过的 certificate/shadow 结果事后改 gate 标签。

## 8. v48.27 FACTOR-PHYSICS-BRIDGE

核心设计：

```text
Raw benefit factor
+ five non-compensatory harm factors
+ frozen-factor safe admission
+ exact execution score
+ valid physical shadow
```

主要修改：

1. 五个 component harm heads，与 gate harmful 定义一一对应。
2. raw benefit 与 safe gate positive 明确分离。
3. 两阶段 factor→admission 训练，关闭第一阶段残留的稀疏 admission 梯度。
4. regression/listwise/frontier 统一使用 `sigmoid(logit)-0.5`。
5. admission 恢复有界 residual，避免 v48.25/26 83%–94% invalid admission。
6. certificate rejection taxonomy 修正。
7. dev-shadow 完整扫描、canonical ID、fail-fast、空指标无效化。
8. sampler 日志区分 sample-level 与 group-level safe positive。
9. 保留 top-3 proposal；oracle 结果不支持扩大 proposal。
10. Safe 不伪装成与 Near/Contact 相同的 Natural gate：继续用 nominal-first 与 paired non-inferiority；最终论文需独立封存 Safe/certificate population。

## 9. v48.27 消融设计

四组 × Balanced/Precision，四个 wave，GPU0/GPU1 各四个任务：

1. `A_three_factor_joint`：旧三因子联合训练；
2. `B_five_factor_joint`：只验证补齐 hard/proxy 两个风险因子；
3. `C_five_factor_two_stage_regression`：五因子 + 两阶段 + safe-utility regression；
4. `D_full_factor_physics_bridge`：C + execution-exact listwise/frontier。

判读：

- B>A：五因子表示缺失是关键瓶颈；
- C>B：联合梯度冲突是关键瓶颈；
- D>C：deployment-exact ranking/frontier 有额外价值；
- offline 改善但 shadow 不改善：PCD teacher 与真实物理过程错位；
- dev rule 有效但 certificate 失败：泛化/有限样本问题；
- dev rule 仍无效：继续优化 factor/admission，不应提前读取 test/stress。

## 10. 预期与限制

v48.27 修复的是已被结果直接支持的根本问题，但不能保证一次实验得到 RC=0。达到 RC=0 至少需要：

- adaptation-dev 产生满足联合约束的正式 rule；
- Near/Contact certificate precision、harm UCB、recall 与宏动作集中度同时通过；
- dev-shadow 输出非空 paired 物理结果；
- Near 满足 clearance/TTC/暴露与低干预目标；
- Contact 满足 secondary contact、overlap、free-space、escape、stable-stop 与动力学约束。

当前环境无法运行真实 WOMD/Waymax 和两张 A30，因此本报告没有虚构 v48.27 gate 或闭环结果。
