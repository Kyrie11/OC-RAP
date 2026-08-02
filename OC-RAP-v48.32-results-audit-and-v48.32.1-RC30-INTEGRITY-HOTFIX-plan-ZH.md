# OC-RAP v48.32 RC=30 审计与 v48.32.1 工程热修方案

## 1. 结论边界

上传的 v48.32 主实验是有效的工程失败，不是算法 gate 结果：

- `pipeline_exit_code=30`
- Balanced adaptation 原始退出码 `1`
- Precision adaptation 原始退出码 `1`
- `failure_stage=adaptation`
- `certificate_executed=false`
- `certificate_exit_code=null`
- `gate_evaluated=false`
- `pipeline_valid=false`
- `test_roots_read=false`

因此本轮不能判断 Identity-Utility Bridge 的优劣，也不能将结果解释为 `development_rule_fit_rejection`。不运行消融、shadow、test 或 stress 是正确的。

## 2. RC=30 的确定性根因

两个 variant 都完成了 Stage-1 factor 训练，然后在 Stage-2 identity training 的 epoch-0 validation 以同一异常退出：

```text
IndexError: too many indices for tensor of dimension 0
src/ocrap/models/losses.py
candidate_gap=teacher_gap[recs]
```

v48.32 在进入 scene-time group 循环前定义了候选级向量：

```python
teacher_gap = torch.clamp(tro - trd, min=0.0)
```

但 adaptive hardest-negative 分支在循环内部复用了同一个名字：

```python
teacher_gap = (
    safe_utility_target[best_safe] - negative_teacher
).clamp(min=0.0, max=0.25)
```

第一组存在 safe-positive 的 group 执行后，`teacher_gap` 从 `[N]` 向量变为零维标量；后续 group 的 factorized component-veto 再执行 `teacher_gap[recs]` 就必然崩溃。Balanced 和 Precision 的日志完全一致，因此这是共享代码路径上的确定性工程错误，而不是数据随机性、显存、checkpoint 或 gate 失败。

## 3. v48.32.1 直接修复

v48.32.1 不改变算法语义、损失权重、gate 阈值、proposal、数据或训练阶段，仅修复工程完整性：

1. 将候选级物理 gap 固定命名为 `teacher_gap_vector`。
2. 将组内 adaptive margin 标量命名为 `adaptive_teacher_gap`。
3. 新增两组 scene-time proposal 的 exact-path loss preflight，在构建索引和启动 GPU 训练前执行 factorized-harm、adaptive-margin、forward 和 backward。
4. preflight 同时执行 AST 变量遮蔽检查，禁止组循环覆盖外层张量。
5. 为 direct recovery loss 增加 strict shape contract。v48.32 原来用 `n=min(sizes)` 静默截断不一致张量；热修主实验中改为 fail-closed。
6. 每个 group 在 strict 模式下必须恰好有一个 nominal，避免错误 nominal 对齐静默进入训练。
7. 设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`；LCVaR 在 deterministic CUDA 模式下使用严格下三角矩阵乘法计算 exclusive prefix，避免已观察到的 CUDA `cumsum` 非确定性警告路径。
8. 增加 stage-aware `VARIANT_STAGE_FAILED.json` 与异常签名提取，RC=30 会直接给出 stage、异常类型、文件、行号和日志尾部。
9. 补齐 certificate 控制器引用的 v48.32.1 metric-calibration population identity 检查工具，并对新增脚本执行依赖闭包扫描，防止训练完成后因漏打包工具再次 RC=30。

## 4. Stage-1 复用修复与运行提速

这轮两个 Stage-1 已成功完成：

| Variant | Stage-1 elapsed | Best epoch | Train/val samples |
|---|---:|---:|---:|
| Balanced | 1175.16 s | 20 | 10015 / 3526 |
| Precision | 1147.04 s | 18 | 10015 / 3526 |

若服务器原运行目录仍保留 `factor_stage/model_v48_trac_sr/best.pt`，下一轮不应重复训练 Stage-1。

v48.32.1 的缓存复用满足以下条件才允许继续：

- source checkpoint SHA 相同；
- train teacher index SHA 相同；
- adaptation-dev teacher index SHA 相同；
- support contract 语义 SHA 相同；
- variant 相同；
- 全部 Stage-1 关键超参数相同；
- 缓存 checkpoint SHA 与 `TRAINING_COMPLETE.json`、`EVIDENCE_CORRECTION_COMPLETE.json` 一致；
- 复制后 checkpoint SHA 再次一致。

复制后的完成元数据会重写到新运行目录，避免 v48.32 直接复制元数据后仍指向旧 checkpoint。建议同时复制上一轮的四个 teacher index/summary 文件到新 OUTPUTDIR，经合同审计后复用，从而保证 factor cache 的 index SHA 精确一致。

## 5. 其他会误导后续分析的工程风险

### 5.1 静默 shape 截断

旧 loss 将所有输入裁剪到最短长度。这会把数据装载、mask 或模型输出数量不一致伪装成可训练 batch，并可能导致 candidate、nominal 和 teacher 标签错位。热修主实验默认严格拒绝。

### 5.2 RC 与 gate 状态混淆

旧 controller 在 certificate stage 的任意 RC=30 上可能直接写 `gate_evaluated=true`。热修后：

- adaptation/contract failure：`certificate_executed=false, gate_evaluated=false`；
- certificate 被调用但 artifact/protocol failure：`certificate_executed=true, gate_evaluated=false`；
- Natural gate 被有效评估且拒绝：`pipeline_exit_code=20, gate_evaluated=true`；
- gate 通过：`pipeline_exit_code=0, gate_evaluated=true`。

### 5.3 缓存污染

旧缓存复制不验证实际输出 checkpoint SHA，也不重写旧路径。热修改为原子复制、输入/输出双验签、目标路径重写，并禁止缓存源目录与目标目录相同。

### 5.4 失败定位不足

旧 `ADAPTATION_FAILED_*.json` 只有大段日志尾部。热修增加具体 stage 和异常签名，避免再次把训练代码崩溃误认为算法无效。

## 6. 下一轮判定规则

下一轮只运行 v48.32.1 主实验：

- `RC=30`：仍是工程/协议失败；不运行消融、shadow、test、stress。检查 `MULTIGROUP_LOSS_CONTRACT.json`、`PIPELINE_FAILED.json`、`FAILURE_SIGNATURE_*.json` 和 `VARIANT_STAGE_FAILED.json`。
- `RC=20`：pipeline 与 certificate 有效，Natural gate 被评估但拒绝；这时才允许对 v48.32 算法作负面结论并设计消融/physical shadow。
- `RC=0`：必须同时存在 `NEXT_COMMANDS.txt` 与 generated status；仅执行该文件授权的后续命令。

不要降低 gate、改变数据集、跳过 preflight、手工创建 `NEXT_COMMANDS.txt`，也不要在 RC=30 时从 Stage-1 训练指标推断算法优劣。

## 7. 本地验证边界

本地完成：

- exact multi-group loss preflight：PASS；
- forward/backward：有限 loss，非零 admission gradient；
- 全量单元测试：294 passed，5 warnings；
- Python compileall：PASS；
- 全部 Shell `bash -n`：PASS；
- 新脚本依赖闭包：PASS；
- ZIP 完整性与 SHA256：发布时验证。

当前环境没有真实 WOMD/Waymax、服务器运行目录中的 `.pt` checkpoint 和两张 A30，因此不预先宣称 v48.32.1 会得到 RC=0 或 RC=20。它只保证已知的 v48.32 确定性 RC=30 崩溃被修复，并用 preflight 防止同类错误再次消耗完整训练。
