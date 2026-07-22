# OC-RAP WOMD 数据集重建报错修复说明

## 1. `builder.py is missing the single-application scenario_start_index fix`

这是脚本预检误报，不是 `builder.py` 缺失修复。

旧脚本使用：

```python
inspect.getsource(builder.build_dataset)
```

但 `scenario_start_index` 的实际实现位于内部函数 `_build_dataset_unlocked()`，`build_dataset()` 只是输出锁包装器，因此固定字符串检查必然失败。

本次修复取消源码字符串匹配，改为功能测试：

- 检查中央扫描函数是否存在；
- 使用整数序列验证 `start_index=3, stride=6, worker=4` 得到 `[7, 13, 19]`；
- 验证传给 WOMD/Waymax loader 的 start/stride/worker 已归零；
- 验证 loader 的原始扫描上限足以覆盖目标全局索引。

同时，start/stride/worker 现在统一由 builder 执行一次。Waymax loader 收到中性扫描参数，避免 loader 和 builder 重复应用。

## 2. `train_safe workers failed: worker0=1, worker1=1`

已复现到确切异常：

```text
worker: unbound variable
```

旧脚本写法：

```bash
local worker="$1" gpu="$2" out="${SHARD_ROOT}/worker${worker}"
```

脚本启用了 `set -u`。Bash 在同一条 `local` 命令中会先展开右侧表达式，此时 `worker` 尚未完成赋值，所以两个 worker 都立即退出为 1。

修复为：

```bash
local worker="$1"
local gpu="$2"
local out="${SHARD_ROOT}/worker${worker}"
```

## 3. 额外加固

- train 脚本强制检查 1000 个 WOMD training shards；
- val/test 脚本强制检查 150 个 validation shards；
- 检查 Python 确实导入当前仓库代码，而非旧全局安装；
- 检查 JAX GPU、TensorFlow 和 Waymax 导入；
- GPU0/GPU1 不允许设置成同一设备；
- worker 失败时自动打印两个日志的最后 160 行，不再只显示退出码；
- `RESET_OUTPUT=1` 可安全删除 `train_safe` 和隐藏的 `.train_safe_shards`；
- `RESET_DATASETS=val_safe,test_safe` 可只重建指定 val/test 目录；
- `PREFLIGHT_ONLY=1` 不会删除或修改数据集；
- 新增全局索引分区无重叠测试。

## 4. 推荐执行命令

### 先执行预检

```bash
cd /path/to/OC-RAP_WOMD_fixed

export WOMD_ROOT=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example
export OCRAP_ROOT=/data0/senzeyu2/dataset/OCRAP

PREFLIGHT_ONLY=1 \
GPU0=0 GPU1=1 \
bash scripts/rebuild_ocrap_train_safe_two_gpu.sh

PREFLIGHT_ONLY=1 \
GPU0=0 GPU1=1 \
bash scripts/rebuild_ocrap_val_test_regimes.sh
```

### 完全替换 `train_safe`

删除最终目录还不够，旧脚本的隐藏 worker 目录可能仍然存在。建议直接使用：

```bash
RESET_OUTPUT=1 \
RESUME=1 \
GPU0=0 GPU1=1 \
RAW_PER_WORKER=6000 \
MIN_SAMPLES=15000 \
MAX_SAMPLES=20000 \
bash scripts/rebuild_ocrap_train_safe_two_gpu.sh
```

这会同时删除并重建：

```text
$OCRAP_ROOT/train_safe
$OCRAP_ROOT/.train_safe_shards
```

中断后续跑时不要再设置 `RESET_OUTPUT=1`：

```bash
RESET_OUTPUT=0 \
RESUME=1 \
GPU0=0 GPU1=1 \
RAW_PER_WORKER=6000 \
MIN_SAMPLES=15000 \
MAX_SAMPLES=20000 \
bash scripts/rebuild_ocrap_train_safe_two_gpu.sh
```

### 重建 safe val/test，同时保留 near/contact partial 数据

```bash
RESET_DATASETS=val_safe,test_safe \
RESUME=1 \
ADOPT_LEGACY_RESUME=0 \
GPU0=0 GPU1=1 \
bash scripts/rebuild_ocrap_val_test_regimes.sh
```

若 near/contact 目录是旧版代码生成且没有 `resume_contract.json`，第一次继续它们时使用：

```bash
RESET_DATASETS=val_safe,test_safe \
RESUME=1 \
ADOPT_LEGACY_RESUME=1 \
GPU0=0 GPU1=1 \
bash scripts/rebuild_ocrap_val_test_regimes.sh
```

成功写入 contract 后，后续改回 `ADOPT_LEGACY_RESUME=0`。

## 5. 验证结果

- Python `compileall`：通过；
- Shell `bash -n`：通过；
- Pytest：95 passed；
- start/stride/worker 分区功能测试：通过；
- 1000/150 shard 预检模拟：通过；
- worker 失败日志自动回显测试：通过。

## 6. 旧 partial 数据兼容性

本次修改改变的是 `scenario_start_index` 与 `scenario_stride` 同时非默认时的全局索引语义。
当前 train/val/test 重建脚本都设置 `scenario_start_index=0`，因此已有的 start=0 near/contact partial 分区与新实现一致，可以继续断点续增。

若某个旧 partial 数据集曾使用非零 `scenario_start_index`，不要在原目录上继续追加；应删除该输出或使用脚本的 reset 参数重新构建，以免旧索引语义和新索引语义混合。
