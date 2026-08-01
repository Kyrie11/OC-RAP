# OC-RAP v48.28 结果审计与 v48.29 VETO-RANK-PHYSICS-BRIDGE 方案

## 1. 结论摘要

v48.28 的 `RC=20` 是有效的 Natural-gate 拒绝，不是训练崩溃或 certificate 工件错误。Balanced、Precision 均完成两阶段训练，模型合同与 factor transfer 检查通过，certificate 实际执行，且没有读取 test/stress。

完整 certificate 上 Near/Contact 的 proposal-constrained oracle 均可满足 gate，说明 top-3 proposal、safe-positive 标签和 certificate 统计要求在数学上兼容。四个分支统一失败在 `development_rule_fit`：adaptation-dev 上不存在一条 learned opportunity/harm/score/rank 联合规则，可以同时满足最低选择数、安全 precision 下界、harm 上界、safe-positive recall、正 teacher advantage 和 macro concentration。

本轮不应删除 certificate 或降低 gate。需要修复的核心是：

1. dev-shadow 虽已成功匹配 8 个 paired scenes，但 provenance 前缀没有被运行时 selector 解析，导致 Near/Contact 的 calibrated `gamma_rec` 均退回 0；
2. `evidence_adapt_dev_contact` 没被识别为 post-contact bucket，Contact 的 contact anchor、re-contact、post-contact free-space、escape、stable-stop 指标未按正确语义计算；
3. v48.28 的五因子风险范围扩展是有效的，但同一个最大风险又进入 admission 软惩罚、又作为独立 hard veto，形成双重风险惩罚，压低 safe-positive action；
4. safe-positive 很稀疏，平均式 regression 无法直接解决 top-1 错选，必须训练“teacher-best safe action 高于 nominal 和组内最难 non-safe action”；
5. closed-loop 极慢的主要原因不是 Waymax 或模型推理，而是在线 `selected_topk` OC-MERO teacher relabeling，约占每次 rollout wall time 的 98.48%–98.57%。

v48.29 因此采用：五因子独立不可补偿 veto、benefit-only admission prior、组内 hardest-negative safe ranking、正确的 regime alias/runtime contract，以及 fast physical shadow + 可选稀疏 teacher audit。

---

## 2. v48.28 RC=20 的分层归因

### 2.1 工程链有效

控制器记录：

- `raw_certificate_exit_code=20`；
- `certificate_exit_code=20`；
- `gate_evaluated=true`；
- `gate_passed=false`；
- `pipeline_valid=true`；
- `test_roots_read=false`。

因此不能将本轮解释为 pipeline failure。

### 2.2 proposal/certificate 合同可行

完整 certificate 的 top-3 oracle：

| Regime | groups | safe-positive groups | oracle precision LCB | 结论 |
|---|---:|---:|---:|---|
| Near | 290 | 9 | 0.8457 | 可行 |
| Contact | 764 | 20 | 约 0.924 | 可行 |

所以当前不是“top-3 中没有足够机会”，也不是 certificate 定义本身不可通过。

### 2.3 learned rule 无法满足联合约束

主实验四个分支均为 `development_rule_fit_rejection`。

| Variant / Regime | Dev selected | Dev positive | Dev precision LCB | Dev harmful / UCB | Dev recall | Dev mean advantage |
|---|---:|---:|---:|---:|---:|---:|
| Balanced Near | 10 | 2 | 0.0862 | 1 / 0.2824 | 0.250 | +0.0668 |
| Balanced Contact | 14 | 4 | 0.1601 | 5 / 0.5281 | 0.235 | -0.0329 |
| Precision Near | 11 | 2 | 0.0781 | 1 / 0.2605 | 0.250 | +0.1173 |
| Precision Contact | 13 | 1 | 0.0233 | 2 / 0.3196 | 0.0588 | -0.0356 |

最接近规则仍远离 gate，尤其是：

- Near 的正向 advantage 已出现，但样本选择 precision/recall 不够，harm UCB 仍过高；
- Contact 的 selected teacher advantage 仍为负，说明 top-1 action 排序尚未建立；
- certificate 冻结验证后，安全 precision/recall 进一步下降，harmful selection 明显增加。

这是算法表示与排序问题，而不是仅仅调一个 admission threshold 可以解决的问题。

---

## 3. Near-contact：哪些核心任务初步实现，哪些尚未实现

### 3.1 已实现的正向信号

v48.28 已经修复 v48.27 的 factor-stage epoch-0 问题。五因子 checkpoint 训练到了后期 epoch，Evidence harm AUC 脱离 0.5。Near 离线信号明显存在：

- Balanced Near candidate positive AUC 约 0.878，Evidence positive/safe-positive AUC 约 0.896；
- Precision Near candidate positive AUC 约 0.898，Evidence safe-positive AUC 约 0.878，harm AUC 约 0.750；
- 最近开发规则的 selected mean advantage 分别为 +0.0668 和 +0.1173。

因此，v48.28 初步实现了“识别某些可能有收益的 Near recovery action”和“学习一部分风险因子”。

### 3.2 尚未实现 safe conversion

仍未完成：

- safe-positive top-1 的稳定命中；
- harmful candidate 的可靠 veto；
- 满足统计下界的 safe precision；
- 足够的 safe-positive recall；
- 从离线收益排序转成有效的物理闭环 improvement。

certificate 中 Balanced/Precision Near 都只有 1 个 safe-positive 命中，harmful selection 分别为 11/15 和 2/7，不能称为 safe admission。

### 3.3 v48.28 shadow 不能验证 Near 投稿目标

shadow 这次匹配成功，但场景 bucket 为 `evidence_adapt_dev_near_contact`，旧运行时只识别精确 `near_contact`。结果中：

- `gamma_rec_by_bucket.near_contact≈0.1975` 已存在；
- 实际每个 scene 的 `gamma_rec=0.0`；
- bucket-specific intervention/evidence 规则也可能未加载。

因此当前看到的 TTC 约 +0.008～+0.013 s、clearance 基本不变、仅 1/8 scene 改善等数值不是按正式 calibrated selector 得到的结果，不能用来判断是否达到 +0.20 s TTC、+0.10 m clearance 等目标。

---

## 4. Contact：目前仍未证明 safe admission

Contact 的主要问题仍是 action ranking：

- Balanced Contact Evidence correlation 约 0.015；
- Precision Contact correlation 为负，约 -0.071；
- 最近开发规则 selected mean advantage 分别为 -0.0329 和 -0.0356；
- certificate 中安全 precision 约 0.056/0，harmful selection 4/18 和 12/28。

这说明模型可以学到一部分静态 recoverability 或广义风险，却仍不能稳定判断哪个撞后 action 会产生真实、持续、可迁移的恢复过程。

### 当前 Contact shadow 的物理指标无效

`evidence_adapt_dev_contact` 未被旧代码识别为 post-contact target，因此：

- `contact_anchor_step=None`；
- `post_contact_free_space_auc_*` 为 NaN；
- `time_to_post_contact_escape_*` 为 NaN；
- re-contact/secondary-contact 可能被错误表示为 0；
- stable-stop 指标没有以正确的 contact anchor 解释。

所以不能根据 v48.28 输出声称 secondary contact、overlap duration、time-to-escape、free-space、sustained escape 或 stable stop 已改善。

---

## 5. 消融归因

v48.28 的消融支持以下判断。

### 有效、应保留

1. **五个不可补偿风险因子。** hard-rule 与 harm-proxy 不能从 gate 标签中删除。
2. **扩大风险 logit 动态范围。** scale=6 相比旧 scale=2 通常显著改善 harm AUC。
3. **两阶段 factor→admission。** 风险因子先学习、随后冻结，避免稀疏 admission 梯度破坏 factor learning。
4. **top-3 frozen proposal。** oracle 已证明支持可行，没有证据支持扩到 top-8。
5. **bounded one-action admission。** 部署只能选一个 recovery 或 nominal，categorical 目标与执行一致。

### 当前实现有害或无稳定增益

1. **risk-centered admission prior。** 五因子最大风险已经作为 hard veto，又通过 `softplus(max_harm)` 扣减 admission，造成双重否决。
2. **默认 listwise/frontier。** D 组相对 C 组没有稳定提高 precision、recall 或降低 harmful selection，继续默认叠加会增加梯度冲突。
3. **平均式 safe-utility regression 单独使用。** 它改善整体拟合，却不直接处理稀疏组内 top-1 hardest negative。
4. **三因子模型。** 它能多命中一些收益 action，但 harmful selection 很高，不能作为主模型。

---

## 6. v48.29 算法修改

### 6.1 独立风险 veto 与收益排序解耦

新增：

```text
model.direct_recovery_evidence_admission_prior_mode=benefit_only
```

此模式下 admission prior 只继承 raw-benefit evidence；五个 harm factors 不再通过同一个 max-harm soft penalty 重复扣分，而是在 selector 中作为独立、不可补偿的 harm veto。

这不意味着忽略安全。harmful candidate 在 safe-utility teacher 中仍为负，并且部署时仍必须通过 harm threshold。

### 6.2 组内 hardest-negative safe ranking

对存在 safe-positive 的 scene-time group：

```text
score(teacher-best safe action)
  > max(score(nominal), score(hardest non-safe proposal)) + margin
```

对不存在 safe-positive 的 group：

```text
max recovery score < nominal - margin
```

这直接针对当前 gate 失败的 top-1 错选，而不是让大量简单负样本稀释梯度。

### 6.3 主模型不默认启用 listwise/frontier

v48.29 主模型使用：

- five-factor wide-range stage-1；
- benefit-only bounded admission；
- deployment-exact safe-utility regression；
- categorical nominal+top-k；
- hardest-negative safe margin。

listwise/frontier 仅在 D 消融中恢复，用来验证它是否在新解耦结构上产生额外增益。

---

## 7. dev-shadow 工程修复与加速

### 7.1 统一 canonical regime alias

新增统一解析：

- `evidence_adapt_dev_near_contact → near_contact`；
- `evidence_adapt_dev_contact → contact`；
- `certificate_pool_contact → contact`；
- Near 必须在 Contact 前匹配，避免被误判为 post-contact。

该解析同时用于：

- `gamma_rec_by_bucket`；
- selector 的所有 `*_by_bucket`/`*_by_regime` 配置；
- Contact anchor 与 post-contact 物理指标；
- runtime contract 检查。

新 shadow 在结果写出后强制检查：

- Near/Contact canonical regime 正确；
- 每个 scene 的 `gamma_rec>0`；
- Contact scene 的 `post_contact_target=true`；
- Contact `contact_anchor_step` 有限；
- `metrics_valid=true`。

### 7.2 闭环耗时根因

8 个 scalar/model shadow 运行中：

- 每个任务 192 decisions；
- scene wall time 约 4,371～8,090 s；
- online `audit_labels` 占 98.48%～98.57%；
- policy selection 总计仅约 7～16 s；
- Waymax step metrics 约 46～87 s。

慢的根本原因是 `label_mode=selected_topk`：每四步对 selected + top-k candidates 重算 OC-MERO teacher rollout。物理闭环指标本身不依赖这一在线 teacher 审计。

v48.29 默认：

```text
SHADOW_LABEL_MODE=fast
SHADOW_AUDIT_LABELS=0
```

它保留真实策略执行、Waymax 状态更新和全部物理指标，跳过在线 teacher relabel。需要离线/物理错位诊断时，可另起目录运行稀疏审计：top-k=3、每 8 步、少量 labels，避免覆盖主 physical shadow。

---

## 8. v48.29 消融

八个任务一次性并发，GPU0 四个 Balanced、GPU1 四个 Precision：

1. `A_risk_centered_reference`：保留旧 risk-centered prior；
2. `B_veto_decoupled`：改为 benefit-only，验证双重风险惩罚；
3. `C_add_safe_hard_negative`：B + hardest-negative safe ranking，v48.29 主模型；
4. `D_add_frontier_to_hard_negative`：C + 轻量 frontier，检查是否有额外收益。

每个任务 `NUM_WORKERS=1`，OMP/MKL/OpenBLAS 各 1 线程，避免八任务争抢 CPU/磁盘。

---

## 9. 判读标准

### 工程完整性

- model/inference contract 一致；
- five factors、scale=6、bounded admission、benefit-only prior 均写入 checkpoint；
- factor transfer valid；
- shadow runtime contract valid；
- RC=30 只允许表示工程/工件/协议失败。

### Near

至少应看到：

- safe-positive top-1 regret 降低；
- harmful switch 明显下降；
- dev precision LCB、recall 同时上升；
- certificate selected teacher advantage 为正；
- 修复后的 shadow 中 clearance/TTC、exposure、deficit AUC 有一致改善，且干预负担受控。

### Contact

至少应看到：

- Evidence correlation 转正；
- selected mean advantage 转正；
- harmful selection 下降；
- 修复后的 post-contact free-space、escape、re-contact、overlap 和 stable-stop 指标可用并改善。

### 停止规则

- Oracle 不可行：重新审查 proposal/label/gate 合同；
- Oracle 可行但 development rule 失败：继续改表示/排序，不改 certificate；
- Dev rule 通过但 certificate 失败：处理泛化/支持问题，不用 certificate 反调阈值；
- Offline 改善但有效 shadow 不改善：构建预注册的 candidate-level temporal physical teacher，而不是事后修改 gate。

