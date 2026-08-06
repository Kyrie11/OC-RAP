# OC-RAP v48.36.2 结果审计与 v48.36.3 终态契约修复

## 1. 本轮边界

本轮只修复 pipeline 工程链路，不修改算法。以下内容保持不变：

- v48.36 OCAF 的 observation-conditioned action interaction bridge；
- 候选生成、模型结构、损失、训练和 checkpoint selection；
- Safe / Near-contact / Contact 的数据集及其划分；
- shared development rule、calibration/certificate 统计量与 gate 阈值；
- 一个统一的连续物理表征，不引入 regime ID、regime-specific head、case routing 或三套策略；
- Safe/stress closed-loop 协议。

论文的核心问题是 oracle recoverability 与可部署 recoverability 之间的差距；当前代码继续用在三种 regime 中连续、非退化的 observation/action/clearance/contact/relative-motion 等物理余量建立统一 admission/ranking 机制。本补丁没有削弱这一约束。

## 2. 审计材料

审计了：

- `post-collision.tex`；
- `大模型建议.md`；
- `OC-RAP.zip` 全部代码、日志、变更记录和运行指令；
- `ocrap_v48_36_1_ocaf_cuda_hotfix_48361.zip` 的活动状态、历史状态、calibration/certificate/gate 产物；
- `reports.zip` 的 Safe/Near/Contact 数据集报告。

上传 ZIP 均无路径穿越条目和符号链接攻击条目。结果 ZIP 不含 `.pt` checkpoint，因此本地可以完成根因定位和只读签名审计，但不能替代原实验机上的 checkpoint 字节校验。

## 3. 两层 RC=30：不能只停留在上一轮结论

### 3.1 第一层：v48.36.1 stage-transfer 假失败

上一轮定位正确：balanced 和 precision 都完成了 factor/identity 训练，但 runner 调用了旧的 `check_v48_32_stage_transfer.py`。旧 checker 不认识 v48.36 新增且明确注册为 trainable 的：

```text
direct_evidence_interaction_bridge.*
```

它把 10 个合法变化的 bridge tensor 误报为冻结参数漂移，两个 adaptation RC=31 被 controller 归一为 pipeline RC=30。原代码包已经包含 v48.36.2 专用 checker 和无重训 repair。

### 3.2 上传结果已经越过第一层失败

当前结果不是“仍停在 stage transfer”。以下证据均已存在且有效：

- `V48_36_2_STAGE_TRANSFER_REPAIR.json`: `valid=true`，`retraining_performed=false`；
- `V48_36_RESUME_CONTRACT.json`: `valid=true`，`failure_mode=repaired_stage_transfer`；
- balanced/precision adaptation exit code 均为 0；
- calibration 和 Near/Contact certificate 已执行；
- gate 已执行；
- balanced 和 precision 的 certificate controller exit code 均为 20；
- 两个 candidate 的 Near/Contact `valid_for_deployment=false`，`valid_candidates=[]`。

因此当前活动 RC=30 是第二个工程错误。

## 4. 当前 RC=30 的真实根因

### 4.1 attempt namespace 版本迁移残留

v48.36 controller：

```bash
ATTEMPT_ID="${V4836_ATTEMPT_ID:-...}"
export V4836_ATTEMPT_ID="$ATTEMPT_ID"
```

但上传代码中的 v48.36 calibration launcher 是从 v48.35 复制后未完整迁移，仍然读取：

```bash
V4835_ATTEMPT_ID
```

controller 没有设置该变量，所以 calibration/gate 产物被写为：

```text
attempt_id = legacy-untracked
```

受影响的活动/历史状态包括：

- `GATE_SPEC.json`；
- `dedicated_recalibration_status.json`；
- balanced/precision 的 `CERTIFICATE_CALIBRATION_COMPLETE.json`；
- balanced/precision 的 `SAFE_REGIME_STATUS.json`；
- calibration 生成后被 controller 归档的 `GATE_FAILED.json`。

当前 controller attempt 是：

```text
v4836-1786007588707518274-896b640dc8d2
```

### 4.2 RC 转换链

1. 两个 candidate 都是自然 gate 拒绝：certificate controller RC=20。
2. calibration launcher 正确生成 `GATE_FAILED` 和 `NEXT_COMMANDS_BLOCKED(reason=natural_gate_failed, exit_code=20)`，但 attempt ID 是 `legacy-untracked`。
3. controller 先写出 pipeline-valid RC=20 completion。
4. `resolve_v48_36_authoritative_result.py` 检查 attempt 一致性，发现：

```text
gate_failed belongs to an older attempt but is required for RC=20
```

5. authoritative resolver 返回 RC=4。
6. controller 把 terminal-state contract 的 RC=4 归一为工程 RC=30，并覆盖活动终态。

所以：

- **当前 RC=30 的直接原因是 terminal-state attempt contract 失败**；
- **其底层算法结果是自然 gate rejection RC=20**；
- 不能把它解释成 pipeline 尚未执行 calibration，也不能把它解释成 gate 通过；
- 修复后 pipeline 会成为有效 RC=20，而不是 RC=0。

## 5. v48.36.3 修复内容

### 5.1 统一 attempt namespace，并 fail closed

修改 `scripts/calibrate_v48_36_shared_certificate_pool.sh`：

- 全部 `V4835_ATTEMPT_ID` 改为 `V4836_ATTEMPT_ID`；
- 缺失、空值或 `legacy-untracked` 时直接 RC=30；
- gate spec、candidate selection、blocked/generated status 记录同一 implementation version；
- selection 明确记录 `requested_variants`。

修改 `scripts/run_v48_36_ocaf_dedicated.sh`：

- 明确向 calibration 子进程传递 `V4836_ATTEMPT_ID="$ATTEMPT_ID"`；
- controller 本身拒绝 legacy attempt；
- 默认 implementation version 为 `v48.36.3-TERMINAL-STATE-HOTFIX`。

### 5.2 新增 certificate-status contract

新增：

```text
tools/check_v48_36_certificate_status_contract.py
```

在写 terminal completion 前检查：

- `GATE_SPEC.json`、candidate selection、NEXT status 属于同一 attempt；
- requested variants 的 certificate completion 和 Safe status 属于同一 attempt；
- certificate/gate 确实执行；
- RC=0 时存在 NEXT_COMMANDS 且不存在 blocked/gate-failed；
- RC=20 时不存在 NEXT_COMMANDS，且 blocked/gate-failed 都存在；
- test roots 保持 sealed；
- 任一状态缺失、不可读、legacy 或跨 attempt 都 fail closed 为工程错误。

这样 attempt 错误会在终态发布前被明确定位，而不会等到 resolver 才模糊地变成 RC=30。

### 5.3 当前结果的精确无重训/无重校准 repair

新增：

```text
tools/repair_v48_36_2_terminal_state_failure.py
scripts/repair_v48_36_2_terminal_state_with_v48_36_3.sh
```

repair 只接受上传结果对应的精确签名：

- 活动失败必须是 `terminal_state_contract`、raw RC=4、normalized RC=30；
- adaptation 必须为 0/0；
- v48.36.2 stage-transfer repair 和 resume contract 必须有效；
- active calibration/gate status 必须全部是 `legacy-untracked`；
- balanced/precision 都必须是自然 gate RC=20，`valid_candidates=[]`；
- Near/Contact certificate 数据必须有效但 `valid_for_deployment=false`；
- archived `GATE_FAILED` 必须唯一且与 active selection 完全相同；
- checkpoint 路径、controller hash、stage-repair hash 和 checkpoint 实际字节必须一致；
- source run、protocol root、test-root seal 必须一致；
- authoritative RC20 失败必须只有已观察到的 attempt contradiction。

repair 行为：

- 对所有将修改的文件做原样备份；
- 仅修正 attempt/provenance/terminal status 元数据；
- 不修改 checkpoint、分数、阈值、证书统计或 gate 决策；
- 不训练、不 calibration、不读 test roots；
- 重建 pipeline-valid RC=20 状态；
- 重新执行 certificate-status contract 和 authoritative resolver；
- 任一后置检查失败时 byte-for-byte rollback。

### 5.4 失败证据保留和版本一致性

- post-certificate diagnostics、certificate-status、completion、terminal-state 失败前，先归档 gate spec、candidate selection、diagnostics 和 terminal markers；
- adaptation stage-transfer/finalization 和 resume metadata 传播 v48.36.3 implementation version；
- v48.36.2 stage-transfer repair artifact 仍被兼容，不需要重新训练。

## 6. 工程错误审计

| 风险 | 原状态 | v48.36.3 处理 | 是否改变算法 |
|---|---|---|---|
| v48.35/v48.36 attempt 环境变量混用 | 已触发 RC=30 | 统一为 V4836，显式传递 | 否 |
| calibration 可静默使用 legacy attempt | 可发生 | 入口 fail closed | 否 |
| terminal completion 前无证书状态一致性检查 | 缺失 | 新增 certificate-status contract | 否 |
| terminal audit 覆盖前置 gate 证据 | 部分证据仅在 history/log | 扩大原子归档范围 | 否 |
| repair 通过改 JSON 绕过 gate | 潜在风险 | 精确签名、checkpoint 字节、certificate、archived marker 和双重后审计 | 否 |
| repair 中途失败留下半修状态 | 潜在风险 | 全文件备份与 byte rollback | 否 |
| implementation version 混乱 | root/child 元数据不一致 | 统一传播 v48.36.3 | 否 |
| partial variants 被未请求的默认 RC 干扰 | selection 含歧义 | 记录 requested_variants 并按其审计 | 否 |
| 历史测试失败误归因于本补丁 | 全仓测试含缺失旧脚本 | 单独记录 inherited failures，定向矩阵全通过 | 否 |

## 7. 验证结果

### 7.1 定向兼容矩阵

逐文件执行：

- `test_v48_16_anchor.py`: 4 passed；
- `test_v48_35_continuous_frontier.py`: 9 passed；
- `test_v48_35_1_rc30_training_contract_hotfix.py`: 8 passed；
- `test_v48_35_2_engineering_integrity.py`: 11 passed；
- `test_v48_36_ocaf.py`: 14 passed，1 CUDA-only skipped；
- `test_v48_36_2_stage_transfer_hotfix.py`: 6 passed；
- `test_v48_36_3_terminal_state_hotfix.py`: 6 passed。

合计：**58 passed，1 skipped**。

新增测试覆盖：

- legacy attempt 被 certificate-status contract 拒绝；
- 精确 terminal-state RC=30 修复为 authoritative RC=20；
- 任一 candidate gate 结果被改写时 repair 拒绝；
- calibration 缺失 V4836 attempt 时 RC=30；
- post-repair resolver 失败时所有状态 byte rollback；
- controller/calibration 只使用 V4836 namespace。

### 7.2 静态验证

- Python `compileall`: PASS；
- `scripts/` 和 `tools/` 下 69 个 Shell 脚本 `bash -n`: PASS；
- 新工具 import / `--help`: PASS；
- `src/ocrap/models/ocrap.py` 与上传代码 byte-identical；
- v48.36 calibration 活动脚本无 `V4835_ATTEMPT_ID`；
- ZIP 中不包含 checkpoint。

### 7.3 全历史测试说明

全仓结果：352 passed、1 skipped、38 failed。失败不是 v48.36.3 回归：

- 37 个测试引用上传包中本来就不存在的 v48.12–v48.32 历史脚本；
- 1 个 v48.30 源码字符串断言在原上传代码中同样失败；
- 上述失败均可在未修改的 `OC-RAP.zip` 复现。

为避免扩大本轮范围，没有伪造或恢复历史发布脚本，也没有为了让旧文本测试通过而改当前算法源码。

## 8. 当前结果应如何解释

在原实验机完成 metadata repair 后，预期权威状态是：

```text
pipeline_valid=true
authoritative_exit_code=20
natural_gate_failed=true
algorithm_changed=false
retraining_performed=false
recalibration_performed=false
```

这表示 pipeline 工程链路成功执行并正确终止，但当前模型没有通过 Natural gate。由于 RC=20，不得执行只对 RC=0 授权的 Safe/stress closed-loop 指令。

本轮不据此修改算法。下一轮应基于已完成的 calibration/certificate/gate 产物，先分析 balanced/precision 在 Near/Contact 中的共同连续余量、admission/ranking/harm failure decomposition，再决定算法修改；仍不得把三个 regime 拆成独立状态或三套策略。

## 9. 上传结果的局限

上传结果 ZIP 未包含：

```text
candidates/balanced/model_v48_trac_sr/best.pt
candidates/precision/model_v48_trac_sr/best.pt
```

只读 repair 审计中，除四个 checkpoint presence/byte checks 外，其余精确签名全部通过。因此：

- 根因定位是确定的；
- 本地不能声称已经修改上传结果的权威状态；
- 在保存真实 checkpoint 的原实验机上运行 repair 才能完成最终授权；
- checkpoint 已删除时，不能通过编辑 JSON 绕过，只能恢复相同字节的 checkpoint 或重新训练。
