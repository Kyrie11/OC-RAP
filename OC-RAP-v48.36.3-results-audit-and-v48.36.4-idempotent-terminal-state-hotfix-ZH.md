# OC-RAP v48.36.3 结果联合审计与 v48.36.4 幂等终态修复报告

## 1. 本轮范围与论文约束

本轮联合阅读和审计了论文 `post-collision.tex`、上一轮分析 `大模型建议.md`、完整代码与 `ALGORITHM_CHANGELOG.md`、运行指令、上传的 v48.36.1 结果包以及三类数据报告。修复严格限定为 pipeline/controller、attempt provenance、terminal-state contract 与恢复工具：没有改动算法、数据集、模型结构、损失、阈值、gate、checkpoint 或分数。

论文的核心不是把 Safe、Near-contact、Contact 做成三个离散 case 后分别路由，而是使用 observation-consistent recoverability 与在三类场景中连续、非退化的物理余量来表达可恢复性、机会和伤害。本补丁没有加入 regime classifier、regime ID 输入或分支策略；`src/` 与 `configs/` 均逐字节不变。后续算法优化应继续保持这一统一连续表征。

## 2. RC=30 的完整故障链

### 2.1 第一层：stage-transfer checker 误报（历史故障）

旧 checker 不认识合法训练的 `direct_evidence_interaction_bridge.*`，把它误报成冻结参数漂移并归一为 RC=30。v48.36.2 的 stage-transfer repair 已经处理这一层。上传结果中已经存在 balanced/precision calibration、certificate 与 gate 证据，说明当前包不再停在这一层。

### 2.2 第二层：attempt ID namespace 不一致（历史故障）

v48.36 controller 使用 `V4836_ATTEMPT_ID`，旧 calibration 路径读取 `V4835_ATTEMPT_ID`，导致 certificate/gate 状态成为 `legacy-untracked`。权威审计拒绝跨 attempt 状态并把 raw RC=4 归一为 RC=30。v48.36.3 已统一 attempt ID，并加入 certificate-status contract 与 terminal-state repair。

### 2.3 第三层：当前上传结果的活动 RC=30 是 resume 覆盖

结果包的 `status_history` 证明 pipeline 后来至少两次完成为有效 RC=20：

- 2026-08-06 14:22:36 UTC：attempt `v4836-1786007588707518274-896b640dc8d2`，RC=20，`pipeline_valid=true`。
- 2026-08-06 15:25:44 UTC：attempt `v4836-1786026418223617262-fa2fee0dce4c`，RC=20，`pipeline_valid=true`；这是与活动 certificate/gate 元数据一致的最新权威结果。

随后多次执行 `RESUME_AFTER_ADAPTATION=1`。旧 controller 的顺序是：先创建新 attempt，resume contract 拒绝后仍归档已有终态，再调用 `write_pipeline_failure` 发布 `stage=resume_authorization`、raw RC=4、pipeline RC=30。15:54--15:57 UTC 的重复调用连续制造了新的 RC=30，最后把活动状态覆盖为 attempt `v4836-1786031819422010486-2c77396f9ca8`。

因此，当前 RC=30 不是训练、calibration 或 certificate 再次失败，也不是模型突然退化；它是“已经完成的 RC=20 被一次不被授权的 resume 命令覆盖”的 controller 幂等性缺陷。

## 3. 正确科学终态

从最新匹配的归档权威 bundle 恢复后：

```text
attempt_id             = v4836-1786026418223617262-fa2fee0dce4c
authoritative_exit_code = 20
pipeline_valid          = true
natural_gate_failed     = true
gate_passed             = false
```

balanced 与 precision 的 certificate controller 均返回 20，`valid_candidates=[]`。这表示 pipeline 成功完成，但当前模型没有通过 Natural gate。不能把 RC=20 写成 RC=0，也不能执行只对 RC=0 授权的 Safe/stress closed-loop。

## 4. v48.36.4 代码修复

### 4.1 Pre-attempt re-entry contract

`tools/check_v48_36_reentry_contract.py` 在 `ATTEMPT_STARTED.json`、状态清理或 GPU 工作之前执行：

- 活动终态为有效 RC=0/20：直接返回已有权威退出码，不创建 attempt、不改 terminal markers。
- 活动状态是精确的 `resume_authorization` RC=30，且历史中存在与当前 selection、GATE_SPEC、certificate-status、candidate certificate/Safe status、test-root seal 全部一致的 RC=20：授权恢复。
- 无匹配归档或状态矛盾：fail closed，保留当前状态。
- 真正未完成的运行：允许正常 pipeline 继续。

### 4.2 非破坏式 resume 拒绝

`run_v48_36_ocaf_dedicated.sh` 现在在 attempt 创建前运行 resume contract。拒绝时只原子写入 `V48_36_RESUME_REFUSED.json`，不再移动 `GATE_FAILED.json`，不再删除 `NEXT_COMMANDS.txt`，不再调用 `write_pipeline_failure`，也不发布新的 RC=30 terminal bundle。

### 4.3 精确 RC=20 恢复与回滚

`tools/restore_v48_36_terminal_state_after_refused_resume.py`：

1. 只接受精确 resume-clobber 签名；
2. 只恢复已有的权威 RC=20 bundle；
3. 要求 archived `GATE_FAILED` 与活动 `dedicated_recalibration_status.json` 字节语义完全一致；
4. 要求 selection、GATE_SPEC、certificate-status、各 candidate 的 certificate 和 Safe status 都属于同一 attempt；
5. 备份每个将被触碰的文件；
6. 恢复后重新执行 certificate-status contract 和 authoritative resolver；
7. 任一检查失败时 byte-for-byte rollback。

它不读取 checkpoint，不改分数、阈值、certificate 统计或 gate 决策。上传结果包没有 `.pt` 也能够完成此恢复，因为被恢复的是已经产生并归档的权威终态，而不是重新构造算法结果。

### 4.4 RC=0 的 fail-closed 边界

历史 controller 在归档终态时没有保存 `NEXT_COMMANDS.txt`。因此，已有活动 RC=0 可以幂等返回，但被覆盖后不能从旧 archive 完整重建 RC=0。本补丁明确拒绝自动恢复 archived RC=0，避免伪造 success artifact。当前问题是 RC=20，因此不影响本次恢复。

### 4.5 统一恢复入口

`scripts/recover_v48_36_pipeline_with_v48_36_4.sh` 按精确签名顺序处理：

1. 当前活动 RC=0/20 或 resume-clobber；
2. v48.36.3 terminal attempt mismatch；
3. v48.36.2 stage-transfer failure；
4. 必要时才执行被授权的 no-retraining resume。

这样避免用户在多个互斥 repair 命令间反复试错。

## 5. 工程误差隔离

发布前检查结果：

- `src/`：109/109 文件逐字节相同；
- `configs/`：28/28 文件逐字节相同；
- `src/ocrap/models/ocrap.py` SHA256：`e542ba4938832f8abbaaed398784765162e1a2ec16d593fc6e5c19c3342d9570`，修复前后相同；
- 代码包内 `.pt`：0；
- 没有引入 regime routing；
- repair 标记 `test_roots_read=false`；
- 原子写、fsync、完整备份、后置 contract 和 rollback 均有测试覆盖；
- 故意覆盖已完成目录必须显式设置 `ALLOW_COMPLETED_RUN_OVERWRITE=1`。

`V48_36_COMPLETE.json` 保留原始 scientific run 的 implementation version；v48.36.4 作为 terminal-state management version 记录在新 contract/restore provenance 中，避免把元数据恢复伪装成重新执行算法。

## 6. 验证

定向兼容矩阵：

```text
64 passed
1 CUDA-only skipped
```

新增 6 个测试覆盖活动 RC=20 幂等、精确恢复、controller 自动恢复、未知 resume 非破坏拒绝、后置失败回滚和源码执行顺序。Python compileall 通过；70 个 shell 脚本 `bash -n` 通过。

在上传结果副本上的真实 replay：

```text
V48_36_4_REENTRY_RESTORE.valid            = true
restored                                   = true
authoritative_exit_code                    = 20
pipeline_valid                             = true
natural_gate_failed                        = true
V48_36_CERTIFICATE_STATUS_CONTRACT checks  = 35/35 true
AUTHORITATIVE_RUN_STATUS checks             = 16/16 true
algorithm_changed                          = false
retraining_performed                       = false
recalibration_performed                    = false
certificate_scores_changed                 = false
gate_decision_changed                      = false
test_roots_read                            = false
```

再次运行统一恢复入口仍返回 RC=20，终态不变。

## 7. 下一步边界

本轮不据此评价算法优劣。恢复后的 RC=20 说明现有实验足以进入下一轮“Natural gate 未通过原因”的算法诊断，但不授权 RC=0-only downstream。下一轮应基于现有 calibration、Near/Contact certificate、`GATE_FAILURE_DECOMPOSITION.json` 和 `learning_gates_v48_36.json`，从统一连续机制的 admission、ranking、opportunity 与 harm 控制分析，而不是把三种 regime 变成三个离散 case。
