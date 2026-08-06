# OC-RAP v48.36.1 RC=30 审计与 v48.36.2 修复方案

## 1. 权威结论

本轮 `RC=30` 是确定的工程契约错误，不是 OCAF 算法被 gate 拒绝，也不是上轮 CUDA 修复失效。

上传结果的权威状态为：

- `authoritative_exit_code=30`；
- `pipeline_valid=false`；
- `certificate_executed=false`；
- `gate_evaluated=false`；
- `test_roots_read=false`；
- balanced adaptation RC=31；
- precision adaptation RC=31。

与 v48.36 的首 batch CUDA 崩溃不同，本轮两个 A30 preflight 均通过，factor 和 identity 训练均已完成。两个 variant 都在训练后的 `stage_transfer_integrity` 阶段失败。

因此：

1. 上一轮 CUDA `index_put_`/广播修复有效；
2. 现有 checkpoint 具备继续校准和 certificate 的条件；
3. 本轮结果不能用于判断 Near/Contact 的最终算法表现；
4. 推荐修复契约并无重训恢复，而不是再次完整训练。

## 2. RC=30 的具体原因

### 2.1 实际训练契约

v48.36.1 的 identity stage 明确将以下四个前缀设为 trainable：

```text
direct_evidence_concord_benefit_calibrator
direct_evidence_concord_harm_calibrator
direct_evidence_concord_admission_calibrator
direct_evidence_interaction_bridge
```

该事实同时记录在：

- `identity_stage/STAGE_ARCHITECTURE.json`；
- `identity_stage/EVIDENCE_CORRECTION_COMPLETE.json`；
- variant runner 的 `identity_prefixes` 构造逻辑。

OCAF interaction bridge 必须在 identity stage 更新，否则 observation-conditioned action effect 无法继续针对安全准入目标进行校正。

### 2.2 实际完整性检查

训练结束后，v48.36 runner 仍调用：

```text
tools/check_v48_32_stage_transfer.py
```

该旧工具的 identity 白名单只有：

```text
direct_evidence_concord_benefit_calibrator
direct_evidence_concord_harm_calibrator
direct_evidence_concord_admission_calibrator
```

它不知道 v48.36 新增的 `direct_evidence_interaction_bridge`，因此将正常训练产生的 bridge 参数变化误判为 frozen drift。

### 2.3 两个 variant 的失败完全一致

Balanced：

- identity allowed changed parameter count：18；
- identity disallowed changed parameter count：10；
- 10 个所谓 disallowed 参数全部属于 `direct_evidence_interaction_bridge.*`；
- final stage disabled，identity→final 没有任何参数变化。

Precision：

- identity allowed changed parameter count：18；
- identity disallowed changed parameter count：10；
- 10 个所谓 disallowed 参数同样全部属于 `direct_evidence_interaction_bridge.*`；
- final stage disabled，identity→final 没有任何参数变化。

日志没有报告 encoder、source experts、proposal generator、frozen policy 或其他非 OCAF 参数发生变化。因此这是白名单版本错配，而非真正的跨阶段污染。

## 3. 本轮可观察到的算法信号及其限制

上传包保留了 adaptation-dev 训练摘要，但没有：

- shared development rule；
- calibration 结果；
- Near/Contact certificate；
- gate 输出；
- Safe/stress 结果。

所以不能据此修改算法或宣称 OCAF 有效/无效。

仅作为调试信号：

### Balanced identity best checkpoint

- best epoch：4；
- `direct_contract_safe_rank_risk=13.798461`；
- `direct_contract_safe_top1_recall_min=0.50`；
- `direct_contract_valid_safe_admission_total=0`；
- `direct_integrity_all_abstain=1`。

Balanced 在 adaptation-dev 的选择语义下仍是全 abstain，但该状态尚未经过共享规则拟合，不能直接归因于算法失败。

### Precision identity best checkpoint

- best epoch：6；
- `direct_contract_safe_rank_risk=7.176799`；
- `direct_contract_safe_top1_recall_min=0.625`；
- `direct_contract_valid_safe_admission_total=1`；
- `direct_integrity_all_abstain=0`。

Precision 出现至少一个有效安全准入，是值得继续完成 pipeline 的信号，但样本支持过小，不能作为投稿结果或算法修改依据。

**本版本不修改 OCAF 表征、损失或超参数。** 下一次只有得到有效 `RC=0` 或 `RC=20` 后，才能进行算法归因。

## 4. 发现的工程问题

### E1：v48.36 runner 调用 v48.32 静态 stage-transfer checker

这是本次 RC=30 的直接原因。

修复：新增 `check_v48_36_stage_transfer.py`，由 controller 显式传入 trainable prefix，并与 stage architecture 做 exact-set 校验。

### E2：缺少 OCAF stage-transfer 回归测试

上轮测试验证了 CUDA bridge forward/backward，但没有构造“factor→identity 中 bridge 合法变化”的 checkpoint 对比，因此旧 checker 复用未被发现。

修复：新增测试复现旧 checker 的 RC=31，同时验证新 checker 接受 bridge、拒绝 encoder drift 和 architecture mismatch。

### E3：完成元数据由内联 Python 非原子生成

旧 variant script 在 checker 后通过大段内联 Python 写 `THREE_STAGE_TRAINING_COMPLETE.json`，缺少统一的输入 SHA、transfer version 和精确 trainable-prefix 记录。

修复：新增 `finalize_v48_36_adaptation_variant.py`，在 checkpoint/`TRAINING_COMPLETE` SHA256 一致后原子写入完成文件。

### E4：现有 resume contract 不识别本次精确失败签名

旧 resume contract 只允许较早的 `training_contract raw RC=4` 情况；当前顶层是 adaptation RC=30，variant RC=31，直接设置 `RESUME_AFTER_ADAPTATION=1` 会被拒绝。

修复：增加专用 repair contract。只有在以下条件全部满足时才允许无重训恢复：

- 两个 variant 都在 stage transfer 返回 31；
- 训练 checkpoint、completion SHA 和 controller SHA 一致；
- 旧 checker 唯一错误是 interaction bridge 被误分类；
- 没有 calibration/certificate/gate/test 访问；
- source run 和 protocol root 不变；
- 新 checker 对两个 variant 均通过。

### E5：reference identity 分支存在潜在契约矛盾

当 `IDENTITY_TRAIN_ALL=0` 且 context 为 `physical_interaction` 时，variant runner 仍会训练 admission head 与 interaction bridge；旧 training contract 却只接受 admission head。

修复：期望集合改为 admission + interaction bridge。主实验默认 `IDENTITY_TRAIN_ALL=1`，所以该问题没有触发本轮失败，但会误杀相关消融。

### E6：failure signature 使用旧 v48.34 event

本轮 `FAILURE_SIGNATURE_*.json` 的 event 仍是 `v48_34_failure_signature`，容易造成版本归属混乱。

修复：新增 `extract_v48_36_failure_signature.py`，记录 base algorithm version 和 implementation version。

### E7：缺少无重训失败后的结构化差异产物

若新 checker 发现真实 frozen drift，不应只输出 shell return code。

修复：新 checker记录 allowed/disallowed 参数名、最大差异、added/missing/shape/dtype 情况；repair 失败时保留 `STAGE_TRANSFER_REPAIR_REJECTED.v48.36.2.json`。

## 5. v48.36.2 的代码修改

本版本是工程热修复，算法保持 v48.36 OCAF 不变。

新增：

- `tools/check_v48_36_stage_transfer.py`；
- `tools/finalize_v48_36_adaptation_variant.py`；
- `tools/repair_v48_36_1_stage_transfer_failure.py`；
- `tools/extract_v48_36_failure_signature.py`；
- `scripts/repair_v48_36_1_stage_transfer_with_v48_36_2.sh`；
- `tests/test_v48_36_2_stage_transfer_hotfix.py`。

修改：

- `scripts/adapt_ocrap_v48_36_ocaf_variant.sh`；
- `scripts/run_v48_36_ocaf_dedicated.sh`；
- `tools/check_v48_36_resume_contract.py`；
- `tools/check_v48_36_ocaf_training_contract.py`；
- `ALGORITHM_CHANGELOG.md`。

## 6. Checkpoint 缺失对本次分析的影响

上传的结果 ZIP 不包含 `.pt` 文件。该缺失不影响本次根因定位，因为：

- stage architecture 明确注册 bridge 为 trainable；
- evidence completion 明确记录四个 trainable prefixes；
- old transfer JSON 明确列出全部误判参数；
- balanced/precision 在同一阶段出现同构失败；
- training completion 和 checkpoint SHA 均已记录。

但**无重训恢复必须在原实验机器上读取 checkpoint 字节**，以重新确认：

- factor/identity/final SHA；
- source checkpoint SHA；
- 只有允许前缀发生变化；
- checkpoint config 中 exact eligibility 仍为 true。

无需默认上传大 checkpoint。若修复工具在原机器上拒绝，请上传以下小文件：

```text
V48_36_2_STAGE_TRANSFER_REPAIR.json
V48_36_RESUME_CONTRACT.json
candidates/*/STAGE_TRANSFER_REPAIR_REJECTED.v48.36.2.json
candidates/*/TRAINING_CONTRACT.json
logs/resume_contract.log
logs/training_contract_*.log
```

只有当这些文件显示无法解释的 state-key/shape/hash 矛盾时，才需要进一步提供 checkpoint 的参数清单或局部 state-diff；通常不需要上传完整 checkpoint。

## 7. 下一步判定

### 推荐路径：无重训恢复

在原 v48.36.1 输出目录和 checkpoint 仍存在时，先执行 repair wrapper。它将：

1. 验证精确失败签名；
2. 用新 checker 重新审计两个 variant；
3. 归档旧失败 transfer 文件；
4. 原子生成完成元数据；
5. 以 `RESUME_AFTER_ADAPTATION=1` 继续 model/training contract、共享校准、certificate 和 gate。

### 退化路径：重新训练

只有在以下情况使用新输出目录完整重训：

- factor/identity/final checkpoint 已删除；
- source checkpoint 已改变或缺失；
- teacher index/summary 已改变或缺失且不能通过 contract；
- repair checker 发现 interaction bridge 之外的 frozen drift；
- checkpoint SHA 与 completion/controller 记录不一致。

### 结果解释

- `RC=0`：pipeline 有效且 gate 通过，才执行 Safe/stress；
- `RC=20`：pipeline 有效、自然 gate 拒绝，随后进行算法层分析；
- `RC=30`：仍是工程/契约失败，读取新的结构化 failure 和 rejected transfer 文件。
