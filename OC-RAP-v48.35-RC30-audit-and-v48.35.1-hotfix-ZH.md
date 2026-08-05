# OC-RAP v48.35 RC=30 工程审计、根因分析与 v48.35.1 修复

## 1. 结论

本次上传的 v48.35 结果**不是一次有效的算法优劣性实验**。两条 adaptation（balanced、precision）均已正常结束并返回 0，但 pipeline 在 certificate 之前的 `training_contract` 静态审计中停止：

- `failure_stage=training_contract`
- `raw_exit_code=4`
- `normalized_exit_code=30`
- `balanced_adaptation_rc=0`
- `precision_adaptation_rc=0`
- `certificate_executed=false`
- `gate_evaluated=false`
- `test_roots_read=false`

唯一失败项是 `exact_eligibility_all_stages=false`。根因是**元数据键名和审计器不一致**：训练脚本确实启用了 exact deployment eligibility，但 stage metadata 只写入旧的 `semantic_frontier_eligibility_metric=true`；原审计器却只读取从未写入的 `exact_deployment_eligibility_metric`。因此，一个完成训练的 run 被错误归类为 RC=30，certificate 根本没有执行。

v48.35.1 是纯工程热修复：不修改模型、损失、候选集合、阈值搜索、共享 rule、certificate、gate 或数据集。正确的下一步不是继续改算法，而是在原实验机器上校验并复用字节完全一致的六个 stage checkpoint，重新执行协议/索引/模型/训练契约和原注册 certificate。

## 2. 对论文的理解

论文主线是把 recoverability 从事后应急行为提升为规划时的一等目标，并专门解决“oracle 可恢复、部署不可恢复”的假准入问题。完整方法链条为：

1. 用 recovery-sufficient latent roots 压缩会改变恢复可行性的隐变量未来；
2. 在候选 prefix 执行后的 observation 上构造 observation-equivalence kernel；
3. 对每个 latent root 与恢复选项预测 signed recovery margin；
4. 先在观测不可区分的 root 内要求共享恢复动作，再用 OC-MERO 做下尾风险聚合；
5. 用 CRISP 对候选动作做校准准入，并在恢复余量充分时保持 nominal；
6. 用 false recoverability admission、oracle–deployability gap、deployable recovery success、nominal utility preservation 和二次碰撞等指标验证。

这一理论叙事天然适合“一个连续机制覆盖 normal/safe、low-headroom、near-contact、post-contact/contact”的论文贡献，而不适合按 regime 输入策略 ID 或拟合三套规则。

### 2.1 必须在投稿前修正的论文/代码矛盾

`post-collision.tex` 第 684–708 行的 **Regime-conditioned recovery admission** 明确写了第二条“只在 low-headroom contact regime 使用”的准入通道，并以 normal/contact regime 区分策略。这与用户要求、论文 novelty 方向以及 v48.35 代码的“一网络、一套连续物理语义、一份共享部署规则”不一致。

建议在获得有效 v48.35 证书结果后，将该段改写为：

- 不使用 regime ID、离散状态机或 per-regime threshold；
- 仅使用候选相对 nominal 的连续 signed physical margins；
- 用统一的 non-compensatory frontier 和共享校准 rule 控制准入；
- Near/Contact/Safe 只作为最坏分层审计和报告维度，而不是策略输入。

附录第 658–666 行的 regime 定义可以保留为数据分析/评测标签，但不能成为 deployment policy 的分支条件。

## 3. 对上一轮大模型分析的复核

上一轮对 v48.34 的关键归因总体正确：RC=20 发生在证书已执行、gate 已评估之后，因此是有效算法拒绝，不是普通缓存/运行错误。Near 出现正向信号但不足以支撑 CCF-A 主结果；Contact 的候选排序和有害动作抑制仍明显不足。

上一轮指出的工程风险也真实存在，并且是 v48.35 设计值得保留的动机：

- Near/Contact 分别拟合规则会形成隐性策略分叉；
- 名称为 exact eligibility 的诊断量曾与真实部署规则语义不一致；
- hard-boundary loss 与最终 rule 边界不一致会让消融结论失真；
- 可补偿 barrier 会允许 benefit/residual 越过不安全 component；
- compact evidence 缺少候选相对 nominal 的可执行前缀物理信息；
- RC=3/20/30 的传播必须严格区分算法拒绝与工程错误；
- 混合版本脚本和旧命令会破坏结果归属。

v48.35 的 `physical_relative`、五个 signed component、smooth minimum non-compensatory cap、one shared adaptation-dev rule、共享 rule SHA256 验证等方向是合理的，当前没有证据要求撤销这些设计。

## 4. RC=30 的证据链

### 4.1 Pipeline 状态

上传结果中的 `PIPELINE_FAILED.json` 和 `V48_35_COMPLETE.json` 一致表明：

- balanced 与 precision adaptation 都为 0；
- balanced 最终 checkpoint SHA256 为 `89fd99d1e70d826022cb6e3e489bab7058a3915d3fd729bd79ff5d60f863989a`；
- precision 最终 checkpoint SHA256 为 `883fd1b0f04a7d2d9a1f1797ef7856a988c5f797347fa684806e1bad4bd90f89`；
- certificate 未执行，gate 未评估，test roots 未读取；
- 失败阶段为 training contract，raw RC=4 被 pipeline 正确规范化为 RC=30。

因此，“RC=30 是工程错误类别”这一分类是正确的；错误之处在于触发 RC=30 的契约检查本身使用了过期键名。

### 4.2 代码级根因

训练 wrapper `scripts/train_ocrap_v48_trac_sr.sh` 接收：

```text
POLICY_METRIC_EXACT_ELIGIBILITY=true
```

并映射为：

```text
training.direct_policy_metric_exact_eligibility=true
```

训练 checkpoint 由 `src/ocrap/cli/train.py` 保存完整 `cfg`，因此原实验机器上的 factor、identity、final checkpoint 可以独立证明该配置位。

但 `scripts/adapt_ocrap_v48_35_continuous_frontier_single_stage.sh` 原先只写：

```json
"semantic_frontier_eligibility_metric": true
```

而原 `tools/check_v48_35_training_contract.py` 只检查：

```python
architecture["exact_deployment_eligibility_metric"] is True
```

这两个字段没有生产者/消费者闭环，造成唯一失败项。

## 5. v48.35.1 修复设计

### 5.1 新运行的正确元数据

stage metadata 同时写入：

```json
"semantic_frontier_eligibility_metric": true,
"exact_deployment_eligibility_metric": true,
"exact_deployment_eligibility_provenance": "checkpoint_cfg.training.direct_policy_metric_exact_eligibility"
```

### 5.2 旧 v48.35 产物的安全兼容

训练契约审计现在读取 factor、identity、final checkpoint 的 `cfg.training.direct_policy_metric_exact_eligibility`。旧 metadata 仅在以下条件同时满足时兼容：

- 新 exact key **不存在**，而不是显式为 false；
- 旧 semantic key 为 true；
- 三个 checkpoint 的 exact config 位全部为 true。

显式 `exact_deployment_eligibility_metric=false` 会被视为矛盾并拒绝，不能被旧字段掩盖。

### 5.3 无重训续跑授权

新增 `tools/check_v48_35_resume_contract.py`，只接受本次已知签名：

- 原失败 event/stage/raw RC/normalized RC 完全一致；
- 两个 adaptation 均为 0；
- certificate、gate、test 均未访问；
- source run 与 protocol root 未改变；
- 不存在 calibration/GATE/NEXT_COMMANDS 等证书痕迹；
- factor/identity/final checkpoint SHA256 同时匹配 stage completion、three-stage completion 和 controller completion；
- factor support SHA256、stage transfer、统一 physical semantics 均有效；
- 每个 checkpoint config 都证明 exact eligibility；
- `retraining_authorized=false`。

任何其他 RC=30、RC=20、checkpoint 缺失或变更、索引契约变化、已有 certificate 产物都会拒绝续跑。

### 5.4 Controller 行为

`RESUME_AFTER_ADAPTATION=1` 只有在上述授权成功后才生效。之后：

- 不重建 adaptation train/dev index；
- 不启动 balanced/precision training；
- 重跑 protocol、index、frontier、model inference、training contracts；
- 调用原始 `calibrate_v48_35_shared_certificate_pool.sh`；
- 原样保留 RC 语义：0=有效通过，20=有效算法拒绝，30=工程/协议错误；
- completion 中记录 `adaptation_reused_without_retraining=true`。

## 6. 目前能否判断 v48.35 算法有效

不能。v48.35 没有共享 rule、Near/Contact certificate 或 gate 输出，因此不能回答 continuous frontier 是否改善了 v48.34 的 Near/Contact 问题。

训练期有一些正向但不足以投稿归因的调试信号：

| variant | best epoch（0-based） | safe top-1 recall min | evidence safe top-1 accuracy | fixed diagnostic valid admission | all abstain |
|---|---:|---:|---:|---:|---:|
| balanced | 2 | 0.625 | 0.72 | 0 | 1 |
| precision | 9 | 0.500 | 0.64 | 0 | 1 |

这些值只能说明候选级 evidence ranking 并非完全失效，同时固定训练诊断阈值下仍全部 abstain。因为 metadata 明确写了 `train_metric_uses_final_fitted_thresholds=false`，最终共享 rule 尚未拟合，所以不能据此认定 gate 一定失败，也不能据此继续调 threshold。

另一个需要在论文中如实表述的点是 factor support reliability 为 `[1,1,1,0,0]`：当前五个 component 中，前三个有数据支持，`hard_rule` 与 `harm_proxy` 在本次 adaptation population 中缺少可学习变化。代码通过 reliability shrinkage 和独立 measured hard veto 防止它们制造伪学习信号；论文不应将五个坐标都描述成同等数据支持的 learned factor。

## 7. Near/Contact 当前投稿程度

以下判断只能沿用最后一次**有效**的 v48.34 certificate，而不是本次 v48.35：

### Near-contact

Near 已经表现出候选级机会识别信号，但有效准入样本、safe recall、precision 下界和跨场景稳定性仍不足。当前更接近“promising but not submission-ready”，可以作为方法潜力或诊断结果，尚不能成为 CCF-A 主结论。

### Contact

Contact 仍明显不足：候选 safe-positive 区分接近弱随机，准入动作中 harmful 比例高，平均 teacher advantage 为负。问题核心不是简单阈值松紧，而是候选相对 nominal 的 deployability/recovery-gap/DRS 变化排序尚不可靠。

### v48.35 的意义

v48.35 正是针对上述结构性问题设计的，但 certificate 尚未运行。必须先执行 v48.35.1 修复，才能判断这些修改产生了正向、无效还是负向结果。

## 8. 获得有效证书后的算法优化顺序

仅在修复后得到 `pipeline_valid=true` 且 RC=0 或 RC=20 时继续算法修改。

### 8.1 应保留

- 一个网络，不输入 regime ID；
- candidate-relative `physical_relative` 表示；
- signed component margins；
- non-compensatory smooth-min frontier；
- independent measured hard veto；
- pooled adaptation-dev 上的一份 shared rule；
- Near/Contact 作为 worst-stratum audit，而不是策略分支；
- certificate-only verification，不允许 certificate label 回流拟合。

### 8.2 若 RC=20，按证据选择修改点

1. **候选 AUC/Top-1 好，但 shared rule 无法选出足够动作**：优先改进 group/listwise safe-utility 对齐、component margin 尺度和共享单调校准；不要拟合 per-regime threshold。
2. **Contact 候选排序仍弱**：在所有样本上统一增加连续的 prefix-relative temporal physics，例如 stopping-reserve change、lateral escape reserve、yaw/stability change、recontact margin、free-space contraction/expansion；这些量必须在 Safe→Near→Contact 上连续定义，不使用 regime 标签。
3. **排序和 calibration 均好但 harmful 仍多**：检查 component target 与 runtime veto 的同语义性，强化不可补偿的 component-wise supervision，而不是提高一个总 barrier penalty。
4. **全局 all-abstain**：先检查共享 rule 的 margin scale、temperature、置信界与样本支持，不直接降低安全阈值；任何放宽必须通过 pooled dev 和 worst-stratum constraints。
5. **某些 factor 无支持**：保留 measured veto/neutral shrinkage；只有现有数据字段能提供连续非退化监督时才升级为 learned target，不要求重构三套数据集。

## 9. 数据集审计的使用边界

按用户要求，本轮不重构 Safe/Near/Contact 数据集。reports 中存在 regime 数量不平衡、部分 split/类型为空、Contact oracle artifact 和 negative deployable 比例较高等现象，这些会影响统计功效和论文措辞，但不是本次 RC=30 根因。

当前策略是：

- 保持数据集不变；
- 在论文中透明报告每个 split 的 scene/group/sample 支持；
- 使用一套共享策略并报告 worst-stratum 置信界；
- 不把 regime imbalance 转换成策略分支；
- 不在 certificate/test 上做任何阈值拟合。

## 10. 工程风险清单

在后续算法落地前必须继续检查：

- producer/consumer 元数据字段是否完全对应；
- checkpoint cfg、stage JSON、completion JSON、controller JSON 的 SHA/语义是否闭环；
- train-time metric 与 deployment rule 是否同语义；
- adaptation-dev 与 certificate/test 是否严格隔离；
- shared rule 是否字节一致地供 Near/Contact 读取；
- RC=3、20、30 是否在每层脚本保持原含义；
- resume 是否可能重训、重建 index 或读取旧 certificate；
- `physical_relative` 是否意外混入 regime、absolute ego、utility/harm oracle scalar、scene suffix；
- non-compensatory cap 是否在数值上始终不高于任一安全 component；
- 消融是否真正作用于部署路径，而不是只改未使用的诊断分支；
- 命令引用的脚本、版本、输出目录是否一致；
- 上传/归档是否包含复现实验所需 checkpoint，且 SHA256 可验证。

## 11. 本地验证

已完成：

- 17 个 v48.35/v48.35.1 focused tests：全部通过；
- 174 个当前上传包支持的 release-matrix tests：全部通过，6 个非致命 PyTorch warning；
- 57 个 shell 脚本：全部通过 `bash -n`；
- `python -m compileall -q src tools tests`：通过；
- continuous-frontier contract：有限梯度、candidate-relative 输入隔离、non-compensatory upper bound、无 regime ID、一份 shared policy contract 全部通过。

上传的结果 ZIP 为节省体积未包含 `.pt` checkpoint，因此无法在本地对实际 run 完成 checkpoint-config/SHA 续跑授权。该检查不会被绕过：必须在保存原 checkpoint 的实验机器上执行。若原机器也缺失任一 checkpoint，则不能安全续跑，必须新建输出目录重跑 adaptation。

## 12. 最终归因规则

- 修复后 `RC=0`：v48.35 主 certificate 有效通过，才执行生成的 Safe paired non-inferiority、held-out stress 和注册消融。
- 修复后 `RC=20`：pipeline 有效但算法未满足 gate；可以据 certificate 进行下一轮统一连续机制优化。
- 修复后 `RC=30`：仍是工程/协议/产物问题，不得分析算法优劣。

