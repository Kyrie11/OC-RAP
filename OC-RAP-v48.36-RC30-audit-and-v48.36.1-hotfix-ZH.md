# OC-RAP v48.36 RC=30 审计与 v48.36.1 工程热修复

## 1. 权威结论

本次结果不能用于判断 OCAF 算法优劣。权威完成状态为：

- `pipeline_exit_code=30`
- `pipeline_valid=false`
- failure stage 为 `adaptation`
- balanced/precision 的 adaptation 子进程均返回 1
- `certificate_executed=false`
- `gate_evaluated=false`
- `test_roots_read=false`

因此不存在 Near/contact certificate、共享规则、gate 或 Safe/stress 指标。本轮不能据此修改阈值、损失权重或 OCAF 算法结构。

## 2. 第一现场

两个 variant 都在 factor stage 的 epoch 1、第一个 batch 进入 OCAF 上下文构造时失败，异常完全相同：

```text
RuntimeError: linearIndex.numel()*sliceSize*nElemBefore == expandedValue.numel()
INTERNAL ASSERT FAILED at ATen/native/cuda/Indexing.cu:490
number of flattened indices did not match number of elements in the value tensor:
4232 vs 279841
```

堆栈指向：

```text
src/ocrap/models/ocrap.py
OCRAPModel._direct_nominal_observation_features
out[idx] = raw[noms[0]:noms[0] + 1]
```

真实几何为：

- 当前 group 有 8 行；
- nominal observation 维度为 529；
- 目标元素数应为 `8 × 529 = 4232`；
- CUDA advanced-index assignment 却把 RHS 扩展成了 `529 × 529 = 279841`。

这不是数据不完整、候选不足、显存不足或 gate 拒绝，而是 GPU 高级索引赋值路径中的形状广播实现错误。CPU 单元测试之所以没有发现，是因为 CPU 上同一写法能够运行；原有 OCAF preflight 也只测试了独立 bridge，没有覆盖“复合 group index + nominal 行广播 + A30 backward”的完整路径。

## 3. 已排除的原因

在崩溃之前，以下契约均已通过：

- canonical dataset root contract；
- dedicated protocol audit；
- multigroup loss finite-gradient contract；
- CPU OCAF bridge contract；
- adaptation train/dev teacher index contract；
- 训练集 10015 行、1167 groups；dev 3526 行、409 groups；
- source checkpoint 成功加载；
- trainable prefix 与 74768 个可训练参数符合 factor-stage 设计。

训练尚未完成一次 optimizer step，因此本轮没有算法方向信号。

## 4. v48.36.1 修复

### 4.1 CUDA-safe group row operations

将以下两类操作从 tensor-valued scalar slice 和 `tensor[idx] = value` 改为显式 row gather/scatter：

```python
group_rows = raw.index_select(0, idx)
nominal_row = raw.index_select(0, noms[:1])
out.index_copy_(0, idx, group_rows - nominal_row)
```

以及：

```python
nominal_row = raw.index_select(0, noms[:1])
broadcast_rows = nominal_row.expand(idx.numel(), -1).contiguous()
out.index_copy_(0, idx, broadcast_rows)
```

该方式明确给出 `[N,D]` source geometry，不再依赖 CUDA `index_put_` 内部广播。

同类 group-wise 写入也在 recovery tournament、set-context adapter 和 group-relative feature 中统一替换为 `index_select/index_copy_`，避免下一个阶段再次触发同类 GPU 路径。

### 4.2 零动作 RMS 的有限梯度

原实现：

```python
sqrt(mean(action_relative ** 2))
```

nominal 行的 action difference 严格为零，零点导数可能在 non-detached 调试、消融或未来联合训练中产生 NaN。修复后使用 float32 的稳定 RMS：

```python
scale_sq = action_relative.float().square().mean(...)
scale = scale_sq.clamp_min(1e-12).sqrt().to(action_relative.dtype)
```

所有 action projection 均无 bias，因此零动作仍然严格输出零，没有改变 OCAF 的物理语义。

### 4.3 两张 GPU 的真实几何 preflight

新增：

```text
tools/check_v48_36_cuda_group_broadcast_contract.py
```

主 runner 在构建/复用 index 和启动 adaptation 之前，分别在 `GPU0`、`GPU1` 上执行：

- batch 96；
- group size 8；
- composite group index；
- action dimension 141；
- nominal observation dimension 529；
- exact-zero nominal context；
- forward finite；
- input 和 bridge parameter backward finite。

任何一张卡失败都会被记录成 attempt-scoped RC=30，且不会消耗完整训练时间。

## 5. 对算法设计的处理

本热修复不改变：

- 一个网络、一套连续物理语义、一份共享 rule；
- 不输入 regime ID；
- OCAF action × nominal-observation interaction；
- source consensus prior scale；
- 五个 signed safety components；
- non-compensatory frontier cap；
- factor/identity loss 权重；
- candidate proposal、certificate、gate 和数据集。

只有在修复后得到 RC=20 或 RC=0，才能继续分析 Near/contact 的排序、precision/recall、harmful admission、macro collapse 和 shared-rule 稳定性。

## 6. 重新执行原则

- 必须设置 `RESUME_AFTER_ADAPTATION=0`。本次两个 variant 均未生成 factor checkpoint，不满足 adaptation resume 条件。
- 建议使用新的输出目录 `runs/ocrap_v48_36_1_ocaf_cuda_hotfix_48361`。
- 可以把旧目录中的四个 teacher-index 文件复制到新目录，并把两个 summary 的 `output` provenance 更新为新路径；runner 会重新做 manifest SHA、dataset roots、label 参数和 tolerance contract 校验。校验不通过会自动拒绝或重建。
- 不要手工复制旧 factor/identity stage、完成标记或 failure marker。

## 7. 本地验证边界

已完成：

- focused regression：42 passed，1 个 CUDA-only test 因本容器无 GPU 被 skip；
- batch=96、group=8、141-D/529-D 完整 factor-stage geometry forward/backward smoke；
- compileall；
- 67 个 shell 脚本 `bash -n`；
- version-scoped tools import/`--help`；
- dependency closure；
- ZIP hash round-trip。

本容器无 A30、WOMD/Waymax 和 source checkpoint，因此没有声称真实 CUDA 主实验已经通过，也没有声称 RC=0。
