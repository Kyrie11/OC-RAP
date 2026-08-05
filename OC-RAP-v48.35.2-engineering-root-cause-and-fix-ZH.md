# OC-RAP v48.35.2 工程根因审计与修复

## 1. 审计边界

本轮只审计工程完整性，不评价模型、损失、候选动作、连续安全前沿或 shared rule 的算法优劣。代码修改不改变训练目标、模型参数化、证书阈值、Natural gate、数据集或三种 regime 的统一连续语义。

## 2. 上传结果的真实终态

上传的结果 ZIP 同时包含两个来自不同时间点的终态文件：

| 文件 | UTC 时间 | 含义 |
|---|---|---|
| `PIPELINE_FAILED.json` | 2026-08-04 23:47:34.926 | 旧的 training-contract 失败，raw RC=4，归一化 RC=30，certificate/gate 未执行 |
| `GATE_FAILED.json` | 2026-08-05 06:59:06.855 | 后续续跑已经执行 certificate 和 Natural gate，结果为自然 gate 未通过 |
| `NEXT_COMMANDS_STATUS.json` | 2026-08-05 06:59:06.860 | `reason=natural_gate_failed`，未生成后续命令 |
| `V48_35_COMPLETE.json` | 2026-08-05 06:59:07.148 | 后续续跑完整结束，raw/certificate/pipeline RC=20，`pipeline_valid=true` |

因此，**本次上传结果的权威终态不是 RC=30，而是有效 pipeline 的 RC=20**。本报告不解释 RC=20 的算法含义，只确认 pipeline 已完成。

结果中的 `learning_gates_v48_35.json` 也把 `pipeline_failed` 记录为 false，说明续跑结束时活动运行目录已不再把旧 `PIPELINE_FAILED.json` 作为当前状态。上传 ZIP 却仍保留旧 entry，表明发布包没有从空白 archive 按当前活动目录重新构建。无论具体操作是复用已有 ZIP 做增量更新，还是等价的旧包复用流程，根本缺陷都是：**归档过程没有删除已经从源目录消失的旧终态 entry，也没有在归档前验证唯一权威状态。**

## 3. 根本性工程错误

### E01：用“文件是否存在”代替终态状态机

旧代码和人工检查容易把任意 `PIPELINE_FAILED.json` 的存在解释为当前 RC=30。该逻辑没有比较：

- `V48_35_COMPLETE.json` 的 pipeline RC；
- 文件时间；
- 同一 attempt ID；
- `NEXT_COMMANDS`、`GATE_FAILED` 和 blocked 状态是否相互一致。

因此，一个来自旧 attempt 的 marker 可以覆盖更新、更完整的终态。

### E02：结果 ZIP 不是事务性全量重建

旧流程没有版本专用 packager，也没有强制：

- 删除旧目标 ZIP；
- 以 write mode 新建；
- 排除 stale terminal marker；
- 检查 duplicate entry；
- 写入权威状态和 manifest；
- ZIP 往返哈希验证。

这使已被源目录删除的旧 `PIPELINE_FAILED.json` 继续留在上传包中。

### E03：共享规则与旧诊断脚本不兼容

v48.35 正确使用 `dev_frozen_shared_rule_v48.json`。但后续诊断脚本仍寻找：

- `dev_frozen_rule_near_v48.json`；
- `dev_frozen_rule_contact_v48.json`。

因此 `controller.resume-v48.35.1.log` 出现 `FileNotFoundError`。这是纯工程兼容错误，不是算法失败。

### E04：诊断异常被 `|| true` 静默吞掉

旧 controller 对 learning-gate 和 gate-decomposition 使用 best-effort 调用。脚本崩溃后主 RC 不变，但诊断文件缺失，后续人员无法区分“诊断未执行”和“诊断结论为空”。

### E05：同一运行目录缺少 attempt 隔离

旧状态文件没有统一 attempt ID。resume、失败、certificate 和 completion 写入同一顶层命名空间，历史状态通常直接删除而不是归档。出现中断、复制、打包复用或人工恢复时，文件来源无法可靠对应某次执行。

### E06：终态文件非原子写入

旧脚本多处直接 `write_text`。进程中断、磁盘异常或并发读取时可能暴露部分 JSON，进一步把解析错误误归因为 pipeline 失败。

### E07：resume 被拒绝时没有发布完整终态

旧 resume contract 返回非零时直接 `exit 30`，可能只留下 `ATTEMPT_STARTED` 或上一次状态，而没有本次 attempt 的 `PIPELINE_FAILED`、completion 和 blocked-next-command 契约。

### E08：failure-signature 提取错误被隐藏

adaptation 失败路径使用 `extract... || true`。若签名工具本身失败，最终 `ADAPTATION_FAILED` 无法区分“没有签名”与“签名提取器坏了”。

### E09：下游授权依赖脆弱 marker

Safe non-inferiority 和 stress wrapper 原先没有统一调用权威终态审计。陈旧 `NEXT_COMMANDS`、旧 completion 或 marker 混合可能造成错误授权或错误阻断。

### E10：生成命令包含机器特定数据路径

certificate 通过后生成的命令曾直接嵌入固定 `SAFE_WOMD_SOURCE`。在另一台机器上即使主 pipeline 正确，也可能在后续步骤产生与算法无关的路径失败。

### E11：打包器生成文件重复

故障注入发现：当运行目录已有 `AUTHORITATIVE_RUN_STATUS.json` 或 `PACKAGING_MANIFEST.json` 时，初版新 packager 会先复制旧文件再写入新文件，形成 duplicate ZIP entry。现已修复为始终排除源目录中的生成型元数据，并在 ZIP 内只写一次。

### E12：历史测试资产不完整且版本混杂

仓库保留 17 个历史测试文件，但其 v48.12–v48.32 launcher/tool 未随代码包提供；同时还混入与本发布无关的 v50 测试。直接运行全历史 `pytest` 会把缺失历史资产或跨版本运行时污染误报为 v48.35.2 回归。修复版提供明确的 v48.35.2 release matrix，不补造历史脚本，也不把 v50 结果计入 v48.35.2 发布判据。

## 4. 修复方案

### 4.1 权威终态解析器

新增 `tools/audit_v48_35_run_state.py`：

- 以 `V48_35_COMPLETE.json` 为终态入口；
- 校验 RC 只能是 0、20、30；
- 使用 attempt ID 和时间判断 marker 是 same-attempt、stale 还是矛盾；
- 校验 RC 与 `pipeline_valid`、certificate、gate、`NEXT_COMMANDS`、blocked、GATE/PIPELINE marker 的完整契约；
- 同一 attempt 的矛盾 fail-closed；
- 可将旧 marker 移入 `status_history/`；
- 原子写入 `AUTHORITATIVE_RUN_STATUS.json`。

终态契约：

- RC=0：pipeline valid、gate passed、`NEXT_COMMANDS.txt` 存在、无活动失败 marker；
- RC=20：pipeline valid、gate 已评估但未通过、活动 `GATE_FAILED` 和 blocked 状态存在、无活动 pipeline failure；
- RC=30：pipeline invalid、活动同 attempt `PIPELINE_FAILED` 和 blocked 状态存在、无后续授权。

### 4.2 attempt 隔离与状态历史

controller 每次启动生成 `ATTEMPT_ID` 并写入 `ATTEMPT_STARTED.json`。新 attempt 开始前，旧活动状态移至 `status_history/pre-<attempt>-<timestamp>/`，不再无痕删除，也不会与当前状态混合。

### 4.3 原子状态发布

v48.35 相关 terminal、gate、calibration、candidate-selection、completion、learning-gate 文件均改为临时文件写入、flush、`fsync`、`os.replace`。

### 4.4 fail-closed 诊断

- gate decomposition 同时支持 shared rule 和 legacy rule；
- 缺失或坏 artifact 返回非零；
- post-certificate learning-gate/decomposition 任一失败，pipeline 以 `post_certificate_diagnostics` RC=30 结束；
- 不再用 `|| true` 隐藏关键诊断失败；
- adaptation failure-signature 独立记录工具 RC。

### 4.5 安全 resume

resume contract 拒绝时，本次 attempt 会先归档旧终态，再发布完整 RC=30 终态。不会出现“命令返回 30，但目录里仍只有旧 completion”的不一致。

### 4.6 新结果打包器

新增 `tools/package_v48_35_results.py` 与 shell wrapper：

- 打包前要求权威状态 valid；
- 删除旧目标 ZIP 和 SHA 文件；
- `ZipFile(..., "w")` 全量重建；
- 默认排除 checkpoint 和 `status_history`；
- 排除 stale marker；
- 源目录已有的 generated metadata 不直接复制；
- 重新写入唯一 `AUTHORITATIVE_RUN_STATUS.json` 和 `PACKAGING_MANIFEST.json`；
- 检查 duplicate names；
- 对每个 entry 做 round-trip SHA256；
- 输出 ZIP SHA256。

### 4.7 下游授权

Safe 和 stress wrapper 在执行前必须通过权威状态审计并确认 RC=0。单个 marker 或旧 `NEXT_COMMANDS` 不再具有授权效力。

## 5. 对上传结果的无重训修复

对上传结果副本执行：

1. 权威状态审计，确认新 completion 为 RC=20；
2. 将旧 `PIPELINE_FAILED.json` 移入 `status_history`；
3. 使用 shared-rule 兼容诊断重新生成 learning-gate 与 gate decomposition；
4. 使用新 packager 生成 clean ZIP。

修复后：

- `authoritative_exit_code=20`；
- `pipeline_valid=true`；
- `pipeline_failed=false`；
- gate decomposition `artifact_valid=true`；
- `development_rule_modes=["shared"]`；
- errors 为空；
- clean ZIP 无 duplicate entry；
- 顶层不包含 stale `PIPELINE_FAILED.json`；
- `AUTHORITATIVE_RUN_STATUS.json` 和 `PACKAGING_MANIFEST.json` 各只有一份。

这只是工程状态修复，不重新计算 certificate，不改变 RC=20，也不进行算法评价。

## 6. 验证

- 192 个 v48.35.2 支持矩阵测试通过；
- 其中 28 个为 v48.35/v48.35.1/v48.35.2 focused tests；
- 57 个 shell 脚本全部通过 `bash -n`；
- `compileall` 通过；
- RC=0、RC=20、RC=30 三种终态一致性均有测试；
- 同 attempt 矛盾 marker fail-closed；
- stale RC=30 marker 可被正确识别和归档；
- shared-rule decomposition 通过；
- fresh ZIP、重复 entry、packager 自包含、旧 generated metadata、hash round-trip 均有回归测试。

## 7. 非声明

本地没有重新运行 WOMD/Waymax、GPU adaptation 或 certificate。本修复只保证：后续实验产生的状态、诊断、授权和归档可以被可靠解释，从而使下一轮结果能够用于算法优劣评估。
